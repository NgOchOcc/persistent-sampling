import logging
import torch
import gc

from typing import List, Optional
from vllm import LLM, SamplingParams
from src.persistent_smc import Particle

logger = logging.getLogger(__name__)

class VLLMGenerator:
    MODEL_MAX_LENGTHS = {
        'Qwen/Qwen2.5-Math-7B-Instruct': 131072, 
        'Qwen/Qwen2.5-Math-1.5B-Instruct': 131072,
        'meta-llama/Llama-3.1-8B-Instruct': 131072,  
        'meta-llama/Llama-3.1-70B-Instruct': 131072,
        'deepseek-ai/deepseek-math-7b-instruct': 4096,
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B': 131072,
        'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B': 131072,
        'deepseek-ai/DeepSeek-R1-Distill-Llama-8B': 131072,
    }

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct",
                 tensor_parallel_size: int = 1, max_model_len: Optional[int] = None,
                 batch_size: int = 4, **vllm_kwargs):
        logger.info(f"Loading {model_name}...")

        if max_model_len is None:
            max_model_len = self._get_optimal_max_length(model_name)
            logger.info(f"Auto-detected max_model_len: {max_model_len}")

        defaults = {
            'gpu_memory_utilization': 0.9,
            'enable_prefix_caching': True,
            'trust_remote_code': True,
            'max_model_len': max_model_len
        }
        defaults.update(vllm_kwargs)

        self.llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, **defaults)
        self.tokenizer = self.llm.get_tokenizer()
        self.vocab_size = len(self.tokenizer)
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.batch_size = batch_size
        logger.info(f"Vocab size: {self.vocab_size}, Max model length: {self.max_model_len}, Batch size: {self.batch_size}")

    def _get_optimal_max_length(self, model_name: str) -> int:
        if model_name in self.MODEL_MAX_LENGTHS:
            return self.MODEL_MAX_LENGTHS[model_name]

        for key, length in self.MODEL_MAX_LENGTHS.items():
            if key.split('/')[-1].split('-')[0] in model_name:
                logger.info(f"Matched model family: {key}, using max_len={length}")
                return length

        logger.warning(f"Unknown model {model_name}, defaulting to 32768")
        return 32768

    def generate_batch(self, prompts: List[str], max_tokens: int = 100,
                      temperature: float = 0.8, top_p: float = 0.95,
                      stop: Optional[List[str]] = None,
                      batch_size: Optional[int] = None) -> List[Particle]:
        if batch_size is None:
            batch_size = self.batch_size

        stop = stop or ['####', '<|im_end|>', '<|endoftext|>']
        safe_prompts = []
        max_prompt_len = self.max_model_len - max_tokens - 100  # Reserve space for generation

        for prompt in prompts:
            tokens = self.tokenizer.encode(prompt)
            if len(tokens) > max_prompt_len:
                truncated_tokens = tokens[-max_prompt_len:]
                prompt = self.tokenizer.decode(truncated_tokens)
            safe_prompts.append(prompt)

        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            logprobs=1,
            prompt_logprobs=None,  # Changed from 0 to None to avoid sampler issues
            skip_special_tokens=True
        )

        all_particles = []
        try:
            for i in range(0, len(safe_prompts), batch_size):
                batch_prompts = safe_prompts[i:i + batch_size]

                try:
                    outputs = self.llm.generate(batch_prompts, params, use_tqdm=False)
                    for out in outputs:
                        new_text = out.outputs[0].text
                        full_text = out.prompt + new_text

                        all_particles.append(Particle(
                            text=full_text,
                            logprobs=out.outputs[0].logprobs or [],
                            finished=self._is_finished(out.outputs[0])
                        ))
                except AssertionError as ae:
                    # Handle vLLM sampler assertion errors
                    logger.warning(f"vLLM sampler assertion error in batch {i//batch_size}: {ae}")
                    logger.warning(f"Retrying batch with adjusted parameters...")

                    # Retry with simpler parameters
                    fallback_params = SamplingParams(
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        stop=stop,
                        logprobs=None,  # Disable logprobs for fallback
                        prompt_logprobs=None
                    )

                    outputs = self.llm.generate(batch_prompts, fallback_params, use_tqdm=False)
                    for out in outputs:
                        new_text = out.outputs[0].text
                        full_text = out.prompt + new_text

                        all_particles.append(Particle(
                            text=full_text,
                            logprobs=[],  # Empty logprobs for fallback
                            finished=self._is_finished(out.outputs[0])
                        ))

                if i + batch_size < len(safe_prompts):
                    self.clear_cache(aggressive=False)

            return all_particles

        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Clear cache on error
            self.clear_cache(aggressive=True)
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
            output.finish_reason == 'length' or  # Also stop at max length
            '####' in output.text or
            '<|endoftext|>' in output.text or
            '<|im_end|>' in output.text or
            output.text.strip().endswith('####')  # Check if ends with answer marker
        )

    def clear_cache(self, aggressive: bool = True):
        if aggressive:
            try:
                if hasattr(self.llm, 'llm_engine'):
                    engine = self.llm.llm_engine

                    if hasattr(engine, 'scheduler'):
                        schedulers = engine.scheduler if isinstance(engine.scheduler, list) else [engine.scheduler]
                        for scheduler in schedulers:
                            if hasattr(scheduler, 'free_finished_seq_groups'):
                                scheduler.free_finished_seq_groups()
                            if hasattr(scheduler, 'running'):
                                scheduler.running.clear()
                            if hasattr(scheduler, 'waiting'):
                                scheduler.waiting.clear()

                    if hasattr(engine, 'cache_engine'):
                        for cache in engine.cache_engine:
                            if hasattr(cache, 'gpu_cache'):
                                cache.gpu_cache = []

                    if hasattr(engine, 'model_executor'):
                        if hasattr(engine.model_executor, 'driver_worker'):
                            worker = engine.model_executor.driver_worker
                            if hasattr(worker, 'cache_engine'):
                                if hasattr(worker.cache_engine, 'gpu_cache'):
                                    pass

                logger.debug("Cleared vLLM internal cache")
            except Exception as e:
                logger.warning(f"Failed to clear vLLM cache: {e}")

        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("Cleared CUDA cache")

    def get_memory_stats(self) -> dict:
        if not torch.cuda.is_available():
            return {}

        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
            'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3
        }
