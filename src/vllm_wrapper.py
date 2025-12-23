from typing import List, Optional
import logging
from vllm import LLM, SamplingParams
from .persistent_smc import Particle

logger = logging.getLogger(__name__)


class vLLMGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct",
                 tensor_parallel_size: int = 1, **vllm_kwargs):
        logger.info(f"Loading {model_name}...")

        defaults = {
            'gpu_memory_utilization': 0.9,
            'enable_prefix_caching': True,
            'trust_remote_code': True
        }
        defaults.update(vllm_kwargs)

        self.llm = LLM(model=model_name, tensor_parallel_size=tensor_parallel_size, **defaults)
        self.tokenizer = self.llm.get_tokenizer()
        self.vocab_size = len(self.tokenizer)
        self.model_name = model_name
        logger.info(f"Ready. Vocab size: {self.vocab_size}")

    def generate_batch(self, prompts: List[str], max_tokens: int = 100,
                      temperature: float = 0.8, top_p: float = 0.95,
                      stop: Optional[List[str]] = None) -> List[Particle]:

        stop = stop or ['\n\n', '####']
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            logprobs=1,
            prompt_logprobs=0
        )

        outputs = self.llm.generate(prompts, params)
        return [
            Particle(
                text=out.prompt + out.outputs[0].text,
                logprobs=out.outputs[0].logprobs or [],
                finished=self._is_finished(out.outputs[0])
            )
            for out in outputs
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
