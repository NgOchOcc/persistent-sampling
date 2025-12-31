import re
import logging

from typing import Optional
from math_verify import parse, verify
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

logger = logging.getLogger(__name__)


def extract_boxed_answer(text: str) -> Optional[str]:
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def verify_answer(response: str, ground_truth: str, use_math_verify: bool = True) -> bool:
    if use_math_verify:
        gold_parsed = parse(
            f"\\boxed{{{ground_truth}}}",
            extraction_config=[LatexExtractionConfig()]
        )
        pred_parsed = parse(
            response,
            extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()]
        )
        return verify(gold_parsed, pred_parsed)

    pred_answer = extract_boxed_answer(response)
    if pred_answer is None:
        return False
    return pred_answer.strip() == ground_truth.strip()


class AnnealingSchedule:
    @staticmethod
    def linear(t: int, T: int = 3, alpha: float = 0.5) -> float:
        return (t / T) ** alpha

    @staticmethod
    def power(t: int, T: int, gamma: float = 2.0) -> float:
        return min(1.0, (t / T) ** gamma)

    @staticmethod
    def saturating(t: int, kappa: float = 0.1) -> float:
        return 1.0 - np.exp(-kappa * t)

    @staticmethod
    def ess_targeted(sc_scores: np.ndarray, prev_beta: float,
                     target_ess: float, max_iter: int = 20) -> float:
        beta_min, beta_max = prev_beta, 10.0
        for _ in range(max_iter):
            beta_mid = (beta_min + beta_max) / 2
            weights = np.exp(beta_mid * sc_scores)
            ess = 1.0 / np.sum((weights / weights.sum()) ** 2)

            beta_min, beta_max = (beta_min, beta_mid) if ess < target_ess else (beta_mid, beta_max)

        return max(prev_beta, beta_mid)