"""
vLLM wrappers for LLM generation and reward model scoring.

Provides:
- LLMGenerator: Wrapper for text generation with logprob extraction
- RewardModel: Wrapper for PRM scoring (task="reward")

Features:
- Batch generation with TokensPrompt
- Cumulative logprob extraction from vLLM output
- Efficient PRM scoring without re-encoding
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .config import BaseLLMConfig, GenerationConfig, PRMConfig


logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class GenerationOutput:
    """Output from a single generation."""
    token_ids: List[int]  # Generated token IDs
    text: str  # Decoded text
    logprobs: List[float]  # Per-token log probabilities
    cum_logprob: float  # Cumulative log probability
    finish_reason: str  # "stop", "length", etc.
    
    @property
    def is_finished(self) -> bool:
        """Whether generation finished (EOS or stop)."""
        return self.finish_reason == "stop"


@dataclass
class BatchGenerationOutput:
    """Output from batch generation."""
    outputs: List[GenerationOutput]
    
    def __len__(self):
        return len(self.outputs)
    
    def __getitem__(self, idx) -> GenerationOutput:
        return self.outputs[idx]


# ============================================================================
# LLM Generator
# ============================================================================

class LLMGenerator:
    """
    vLLM-based LLM generator with logprob extraction.
    
    Features:
    - Batch generation from token prompts
    - Per-token logprob extraction
    - Cumulative logprob computation
    - Prefix caching support
    """
    
    def __init__(
        self,
        model_config: BaseLLMConfig,
        generation_config: GenerationConfig,
    ):
        self.model_config = model_config
        self.generation_config = generation_config
        self._llm = None
        self._tokenizer = None
        self._sampling_params_cls = None
        self._tokens_prompt_cls = None
    
    def initialize(self):
        """Initialize vLLM LLM instance."""
        if self._llm is not None:
            return
        
        from vllm import LLM, SamplingParams, TokensPrompt
        
        logger.info(
            f"Loading LLM: {self.model_config.model_path} "
            f"(device={self.model_config.device}, "
            f"tp={self.model_config.tensor_parallel_size}, "
            f"mem={self.model_config.gpu_memory_utilization})"
        )
        
        self._llm = LLM(
            model=self.model_config.model_path,
            tensor_parallel_size=self.model_config.tensor_parallel_size,
            gpu_memory_utilization=self.model_config.gpu_memory_utilization,
            device=self.model_config.device,
            trust_remote_code=True,
            enable_prefix_caching=self.model_config.enable_prefix_caching,
        )
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling_params_cls = SamplingParams
        self._tokens_prompt_cls = TokensPrompt
        
        logger.info("LLM loaded successfully")
    
    @property
    def llm(self):
        """Lazy-loaded LLM instance."""
        if self._llm is None:
            self.initialize()
        return self._llm
    
    @property
    def tokenizer(self):
        """Lazy-loaded tokenizer."""
        if self._tokenizer is None:
            self.initialize()
        return self._tokenizer
    
    @property
    def eos_token_id(self) -> int:
        """EOS token ID."""
        return self.tokenizer.eos_token_id
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.tokenizer.encode(text)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def format_chat_prompt(
        self,
        problem: str,
        instruction: str = "Please reason step by step, and put your final answer within \\boxed{}."
    ) -> Tuple[str, str]:
        """
        Format a chat-style prompt.
        
        Args:
            problem: The problem text
            instruction: Instruction to append
        
        Returns:
            Tuple of (formatted prompt string, original query for scoring)
        """
        user_content = f"{problem}\n{instruction}"
        
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_content}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = user_content
        
        return prompt, problem
    
    def _create_sampling_params(
        self,
        n: int = 1,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """Create SamplingParams for generation."""
        self.initialize()  # Ensure initialized
        
        config = self.generation_config
        
        return self._sampling_params_cls(
            n=n,
            max_tokens=max_tokens or config.max_tokens_per_step,
            temperature=temperature or config.temperature,
            top_p=config.top_p,
            top_k=config.top_k if config.top_k > 0 else -1,
            stop=config.stop_strings if config.stop_strings else None,
            stop_token_ids=config.stop_token_ids if config.stop_token_ids else None,
            logprobs=config.logprobs_k if config.return_logprobs else None,
        )
    
    def _extract_logprobs(self, output) -> Tuple[List[float], float]:
        """
        Extract per-token logprobs from vLLM output.
        
        Args:
            output: vLLM CompletionOutput
        
        Returns:
            Tuple of (list of per-token logprobs, cumulative logprob)
        """
        logprobs = []
        
        if output.logprobs is not None:
            for logprob_dict in output.logprobs:
                if logprob_dict:
                    # Get logprob of the sampled token
                    # logprob_dict maps token_id -> Logprob object
                    # We need the logprob of the token that was actually sampled
                    sampled_logprob = list(logprob_dict.values())[0].logprob
                    logprobs.append(sampled_logprob)
                else:
                    logprobs.append(0.0)
        
        # Use cumulative_logprob from output if available, otherwise sum
        if hasattr(output, 'cumulative_logprob') and output.cumulative_logprob is not None:
            cum_logprob = output.cumulative_logprob
        else:
            cum_logprob = sum(logprobs) if logprobs else 0.0
        
        return logprobs, cum_logprob
    
    def generate_batch(
        self,
        token_ids_list: List[List[int]],
        n_per_prompt: int = 1,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[BatchGenerationOutput]:
        """
        Generate completions for a batch of token prompts.
        
        Args:
            token_ids_list: List of token ID lists (prompts)
            n_per_prompt: Number of completions per prompt
            max_tokens: Override max tokens
            temperature: Override temperature
        
        Returns:
            List of BatchGenerationOutput (one per prompt)
        """
        self.initialize()
        
        params = self._create_sampling_params(
            n=n_per_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        
        # Create TokensPrompt for each
        prompts = [
            self._tokens_prompt_cls(prompt_token_ids=tids)
            for tids in token_ids_list
        ]
        
        # Generate
        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=params,
            use_tqdm=False
        )
        
        # Process outputs
        results = []
        for req_output in outputs:
            gen_outputs = []
            for out in req_output.outputs:
                logprobs, cum_logprob = self._extract_logprobs(out)
                gen_outputs.append(GenerationOutput(
                    token_ids=list(out.token_ids),
                    text=out.text,
                    logprobs=logprobs,
                    cum_logprob=cum_logprob,
                    finish_reason=out.finish_reason or "unknown",
                ))
            results.append(BatchGenerationOutput(outputs=gen_outputs))
        
        return results
    
    def generate_single(
        self,
        token_ids: List[int],
        n: int = 1,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> BatchGenerationOutput:
        """
        Generate completions for a single prompt.
        
        Args:
            token_ids: Token IDs of the prompt
            n: Number of completions
            max_tokens: Override max tokens
            temperature: Override temperature
        
        Returns:
            BatchGenerationOutput containing n completions
        """
        results = self.generate_batch(
            token_ids_list=[token_ids],
            n_per_prompt=n,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return results[0]
    
    def continue_generation_batch(
        self,
        token_ids_list: List[List[int]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[GenerationOutput]:
        """
        Continue generation for a batch of partial sequences.
        
        Each prompt gets exactly 1 continuation.
        
        Args:
            token_ids_list: List of token ID lists (partial sequences)
            max_tokens: Override max tokens
            temperature: Override temperature
        
        Returns:
            List of GenerationOutput (one per prompt)
        """
        results = self.generate_batch(
            token_ids_list=token_ids_list,
            n_per_prompt=1,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return [r.outputs[0] for r in results]


# ============================================================================
# Reward Model (PRM)
# ============================================================================

class RewardModel:
    """
    vLLM-based Process Reward Model (PRM) scorer.
    
    Uses vLLM task="reward" for efficient batch scoring.
    
    Output format: [[prob_wrong, prob_correct]]
    Score = prob_correct (index 1)
    """
    
    def __init__(
        self,
        config: PRMConfig,
        step_separator: str = "<extra_0>",
        system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}.",
    ):
        self.config = config
        self.step_separator = step_separator
        self.system_prompt = system_prompt
        self._llm = None
        self._tokenizer = None
    
    @property
    def enabled(self) -> bool:
        """Whether PRM is enabled in config."""
        return self.config.enabled
    
    def initialize(self):
        """Initialize vLLM LLM instance for reward scoring."""
        if self._llm is not None:
            return
        
        if not self.enabled:
            logger.warning("PRM is disabled in config, skipping initialization")
            return
        
        from vllm import LLM
        
        logger.info(
            f"Loading PRM: {self.config.model_path} "
            f"(device={self.config.device}, "
            f"tp={self.config.tensor_parallel_size}, "
            f"mem={self.config.gpu_memory_utilization})"
        )
        
        self._llm = LLM(
            model=self.config.model_path,
            task="reward",
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            device=self.config.device,
            trust_remote_code=True,
        )
        self._tokenizer = self._llm.get_tokenizer()
        
        logger.info("PRM loaded successfully")
    
    @property
    def tokenizer(self):
        """Lazy-loaded tokenizer."""
        if self._tokenizer is None:
            self.initialize()
        return self._tokenizer
    
    def _format_for_prm(self, query: str, response_text: str) -> str:
        """
        Format input for PRM evaluation.
        
        Args:
            query: Original query/problem
            response_text: Model's response
        
        Returns:
            Formatted string for PRM
        """
        sep = self.step_separator
        
        # Split response into steps by double newlines
        steps = [s.strip() for s in re.split(r'\n\n+', response_text) if s.strip()]
        if not steps:
            steps = [response_text.strip() or "..."]
        
        formatted_response = sep.join(steps) + sep
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
            {"role": "assistant", "content": formatted_response},
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    
    def score_batch(
        self,
        queries: List[str],
        responses: List[str]
    ) -> List[float]:
        """
        Score a batch of query-response pairs.
        
        Args:
            queries: List of queries/problems
            responses: List of model responses
        
        Returns:
            List of scores (prob_correct, in [0, 1])
        """
        if not self.enabled:
            raise RuntimeError("PRM is not enabled")
        
        if not responses:
            return []
        
        self.initialize()
        
        # Format inputs
        conversations = [
            self._format_for_prm(q, r)
            for q, r in zip(queries, responses)
        ]
        
        # Encode and get rewards
        outputs = self._llm.encode(conversations)
        
        all_scores = []
        for output in outputs:
            try:
                data = output.outputs.data
                
                if data is not None and len(data) > 0:
                    step_scores = []
                    for score in data:
                        # score is typically [prob_wrong, prob_correct]
                        if hasattr(score, '__len__') and len(score) > 1:
                            val = score[1]  # prob_correct
                            step_scores.append(
                                float(val.item()) if hasattr(val, 'item') else float(val)
                            )
                        elif hasattr(score, 'item'):
                            step_scores.append(float(score.item()))
                        else:
                            step_scores.append(float(score))
                    
                    if step_scores:
                        # Average over steps
                        all_scores.append(float(np.mean(step_scores)))
                    else:
                        all_scores.append(0.5)
                else:
                    all_scores.append(0.5)
            except Exception as e:
                logger.warning(f"Error scoring with PRM: {e}")
                all_scores.append(0.5)
        
        return all_scores
    
    def score_single(self, query: str, response: str) -> float:
        """Score a single query-response pair."""
        scores = self.score_batch([query], [response])
        return scores[0]
