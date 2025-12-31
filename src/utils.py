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
