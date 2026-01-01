"""
Scoring strategies for Particle Sampling Algorithm
"""
import re
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)


class BaseScorer(ABC):
    @abstractmethod
    def score_batch(self, queries: List[str], responses: List[str], **kwargs) -> List[float]:
        pass

    def prepare_for_scoring(self, tokenizer, token_ids: List[int]) -> str:
        return tokenizer.decode(token_ids, skip_special_tokens=True)


class PRMScorer(BaseScorer):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.9,
        system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}.",
        step_separator: str = "<extra_0>",
    ):
        self.model_name = model_name
        self.step_separator = step_separator
        self.system_prompt = system_prompt

        from vllm import LLM
        logger.info(
            f"Loading PRM: {model_name} "
            f"(device={device}, tp={tensor_parallel_size}, mem={gpu_memory_utilization})"
        )
        self.llm = LLM(
            model=model_name,
            task="reward",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            device=device,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        logger.info("PRM loaded successfully")

    def _format_for_prm(self, query: str, response_text: str) -> str:
        sep = self.step_separator
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

    def score_batch(self, queries: List[str], responses: List[str], **kwargs) -> List[float]:
        if not responses:
            return []

        conversations = []
        for query, response in zip(queries, responses):
            conv = self._format_for_prm(query, response)
            conversations.append(conv)

        outputs = self.llm.encode(conversations)
        all_scores = []
        for output in outputs:
            try:
                data = output.outputs.data
                if data is not None and len(data) > 0:
                    step_scores = []
                    for score in data:
                        if hasattr(score, '__len__') and len(score) > 1:
                            val = score[1]
                            step_scores.append(float(val) if hasattr(val, 'item') else float(val))
                        elif hasattr(score, 'item'):
                            step_scores.append(float(score.item()))
                        else:
                            step_scores.append(float(score))

                    if step_scores:
                        all_scores.append(float(np.mean(step_scores)))
                    else:
                        all_scores.append(0.5)
                else:
                    all_scores.append(0.5)
            except Exception as e:
                logger.warning(f"Error scoring: {e}")
                all_scores.append(0.5)

        return all_scores


class LogProbScorer(BaseScorer):
    def __init__(
        self,
        normalize: bool = True,
        use_negative: bool = False,
        temperature: float = 1.0,
    ):
        self.normalize = normalize
        self.use_negative = use_negative
        self.temperature = temperature

    def score_batch(
        self,
        queries: List[str],
        responses: List[str],
        log_probs: Optional[List[float]] = None,
        sequence_lengths: Optional[List[int]] = None,
        **kwargs
    ) -> List[float]:
        if log_probs is None:
            return [0.0] * len(responses)

        scores = []
        for i, log_prob in enumerate(log_probs):
            # Tính trung bình cộng log_prob của tất cả các token từ đầu đến hiện tại
            if sequence_lengths is not None and sequence_lengths[i] > 0:
                mean_log_prob = log_prob / sequence_lengths[i]
            else:
                mean_log_prob = log_prob

            # Áp dụng temperature và lấy e mũ
            score = np.exp(mean_log_prob / self.temperature)

            if self.use_negative:
                score = -score

            scores.append(float(score))

        return scores


class CombinedScorer(BaseScorer):
    def __init__(self, scorers: List[tuple], weights: Optional[List[float]] = None):
        self.scorers = scorers
        if weights is None:
            weights = [1.0 / len(scorers)] * len(scorers)
        self.weights = weights

    def score_batch(self, queries: List[str], responses: List[str], **kwargs) -> List[float]:
        all_scores = []
        for (scorer, name), weight in zip(self.scorers, self.weights):
            scores = scorer.score_batch(queries, responses, **kwargs)
            all_scores.append((scores, weight))
            logger.debug(f"Scorer {name}: {scores[:3]}")

        combined = []
        for i in range(len(responses)):
            score = sum(scores[i] * weight for scores, weight in all_scores)
            combined.append(score)

        return combined
