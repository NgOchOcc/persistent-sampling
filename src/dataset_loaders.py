import json
import logging
from typing import List, Dict, Optional
from math_verify import (
    parse,
    verify,
    LatexExtractionConfig,
    ExprExtractionConfig,
)

logger = logging.getLogger(__name__)

class AnswerExtractor:
    @staticmethod
    def extract_with_math_verify(text: str, target_type: str = "latex") -> Optional[str]:
        if target_type == "latex":
            config = [LatexExtractionConfig()]
        elif target_type == "expr":
            config = [ExprExtractionConfig()]
        else:
            config = [LatexExtractionConfig(), ExprExtractionConfig()]

        # Parse with timeout
        parsed = parse(
            text,
            extraction_config=config,
            fallback_mode="first_match",
            extraction_mode="any_match",
            parsing_timeout=3,
            raise_on_error=False
        )

        return str(parsed[0]) if parsed else None


    @staticmethod
    def boxed(text: str) -> Optional[str]:
        return AnswerExtractor.extract_with_math_verify(text, "latex")

    @staticmethod
    def integer(text: str) -> Optional[int]:
        result = AnswerExtractor.extract_with_math_verify(text, "expr")
        if result:
            num = int(float(result))
            if 0 <= num <= 999:
                return num


    @staticmethod
    def generic(text: str) -> Optional[str]:
        return AnswerExtractor.extract_with_math_verify(text, "both")

    @staticmethod
    def normalize(answer: str) -> str:
        replacements = {
            '\\frac': 'frac', '\\pi': 'pi', '\\sqrt': 'sqrt',
            '\\pm': '+-', '\\text': '', '\\mathrm': '',
            '\\,': '', '\\!': '', '{': '', '}': '', ' ': ''
        }

        for old, new in replacements.items():
            answer = answer.replace(old, new)
        return answer.lower()

    @staticmethod
    def verify_equivalence(answer1: str, answer2: str, precision: int = 6) -> bool:
        parsed1 = parse(answer1, raise_on_error=False)
        parsed2 = parse(answer2, raise_on_error=False)
        if not parsed1 or not parsed2:
            return AnswerExtractor.normalize(answer1) == AnswerExtractor.normalize(answer2)

        return verify(
            gold=parsed2[0],
            target=parsed1[0],
            float_rounding=precision,
            strict=False, 
            raise_on_error=False
        )


class DatasetLoader:
    @staticmethod
    def load_math500(path: str) -> List[Dict]:
        with open(path) as f:
            data = json.load(f)
        return data

    @staticmethod
    def load_aime24(path: str) -> List[Dict]:
        with open(path) as f:
            data = json.load(f)
        return data

    @staticmethod
    def load_gsm8k(path: str) -> List[Dict]:
        with open(path) as f:
            data = json.load(f)
        return data

