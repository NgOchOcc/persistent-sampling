import logging
import torch
import gc

from typing import List, Optional
from vllm import LLM, SamplingParams
from src.persistent_smc import Particle

logger = logging.getLogger(__name__)

class VLLMGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct",
                 tensor_parallel_size: int = 1, max_model_len: int = None, **vllm_kwargs):
        logger.info(f"Loading {model_name}...")

        defaults = {
            'gpu_memory_utilization': 0.9,
            'enable_prefix_caching': True,
            'trust_remote_code': True
        }
        if max_model_len:
            defaults['max_model_len'] = max_model_len
        defaults.update(vllm_kwargs)

        self.llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, **defaults)
        self.tokenizer = self.llm.get_tokenizer()
        self.vocab_size = len(self.tokenizer)
        self.model_name = model_name
        self.max_model_len = max_model_len or getattr(self.llm.llm_engine.model_config, 'max_model_len', 32768)
        logger.info(f"Vocab size: {self.vocab_size}, Max model length: {self.max_model_len}")

    def generate_batch(self, prompts: List[str], max_tokens: int = 100,
                      temperature: float = 0.8, top_p: float = 0.95,
                      stop: Optional[List[str]] = None) -> List[Particle]:

        stop = stop or ['\n\n', '####']

        # Check prompt lengths and truncate if needed
        safe_prompts = []
        max_prompt_len = self.max_model_len - max_tokens - 100  # Reserve space for generation

        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) > max_prompt_len:
                logger.warning(f"Prompt too long ({len(tokens)} tokens), truncating to {max_prompt_len}")
                # Truncate from the beginning, keep the end (more recent context)
                truncated_tokens = tokens[-max_prompt_len:]
                prompt = self.tokenizer.decode(truncated_tokens)
            safe_prompts.append(prompt)

        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            logprobs=1,
            prompt_logprobs=0
        )

        try:
            outputs = self.llm.generate(safe_prompts, params, use_tqdm=False)
            return [
                Particle(
                    text=out.prompt + out.outputs[0].text,
                    logprobs=out.outputs[0].logprobs or [],
                    finished=self._is_finished(out.outputs[0])
                )
                for out in outputs
            ]
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            # Return empty particles on failure
            return [
                Particle(
                    text=prompt,
                    logprobs=[],
                    finished=True
                )
                for prompt in safe_prompts
            ]

    def format_math_prompt(self, problem: str, system_prompt: Optional[str] = None) -> str:
        system_prompt = system_prompt or (
            "You are a mathematical problem solver. "
            "Solve step by step and end with #### followed by the answer."
        )

        if hasattr(self.tokenizer, 'apply_chat_template'):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem}
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        return f"{system_prompt}\n\nProblem: {problem}\n\nSolution: Let's solve step by step.\n"

    def _is_finished(self, output) -> bool:
        return (
            output.finish_reason == 'stop' or
            '####' in output.text or
            '<|endoftext|>' in output.text or
            '<|im_end|>' in output.text
        )

    def clear_cache(self, aggressive: bool = True):
        """
        Clear vLLM cache to free memory after processing a sample

        Args:
            aggressive: If True, clear both vLLM cache and CUDA cache
                       If False, only clear CUDA cache
        """
        if aggressive:
            try:
                # Clear vLLM KV cache
                if hasattr(self.llm, 'llm_engine'):
                    # For vLLM v1 (newer)
                    if hasattr(self.llm.llm_engine, 'scheduler'):
                        for scheduler in self.llm.llm_engine.scheduler:
                            if hasattr(scheduler, 'free_finished_seq_groups'):
                                scheduler.free_finished_seq_groups()

                    # Clear cache manager
                    if hasattr(self.llm.llm_engine, 'cache_config'):
                        logger.debug("Clearing vLLM cache manager")

                logger.debug("Cleared vLLM internal cache")
            except Exception as e:
                logger.warning(f"Failed to clear vLLM cache: {e}")

        # Always clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.debug("Cleared CUDA cache")

    def get_memory_stats(self) -> dict:
        """Get current GPU memory usage statistics"""
        if not torch.cuda.is_available():
            return {}

        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
        }
