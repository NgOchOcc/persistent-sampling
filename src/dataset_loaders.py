import json
import re
from typing import List, Dict, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AnswerExtractor:
    @staticmethod
    def boxed(text: str) -> Optional[str]:
        matches = re.findall(r'\\boxed\{([^}]*)\}', text)
        return matches[-1].strip() if matches else None

    @staticmethod
    def integer(text: str) -> Optional[int]:
        """Extract integer answer (for AIME)"""
        # Try #### format
        if "####" in text:
            after = text.split("####")[-1].strip()
            nums = re.findall(r'\d+', after.split('\n')[0])
            if nums:
                return int(nums[0])

        # Try common patterns
        patterns = [
            r'[Tt]he answer is (\d+)',
            r'[Ff]inal answer:?\s*(\d+)',
            r'= (\d+)$'
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return int(matches[-1])

        # Last resort: find final number
        for line in reversed(text.strip().split('\n')):
            match = re.search(r'\b(\d{1,3})\b', line)
            if match:
                num = int(match.group(1))
                if 0 <= num <= 999:
                    return num

        return None

    @staticmethod
    def generic(text: str) -> Optional[str]:
        """Generic extraction trying multiple methods"""
        # Try boxed first
        boxed = AnswerExtractor.boxed(text)
        if boxed:
            return boxed

        # Try #### format
        if "####" in text:
            return text.split("####")[-1].strip().split('\n')[0].strip()

        # Try common patterns
        patterns = [
            r'[Tt]he answer is:?\s*(.+?)(?:\.|$)',
            r'[Ff]inal answer:?\s*(.+?)(?:\.|$)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                return matches[-1].strip()

        return None

    @staticmethod
    def normalize(answer: str) -> str:
        """Normalize for comparison"""
        if not answer:
            return ""

        # Remove common LaTeX/formatting
        replacements = {
            '\\frac': 'frac', '\\pi': 'pi', '\\sqrt': 'sqrt',
            '\\pm': '+-', '\\text': '', '\\mathrm': '',
            '\\,': '', '\\!': '', '{': '', '}': '', ' ': ''
        }

        for old, new in replacements.items():
            answer = answer.replace(old, new)

        return answer.lower()


class DatasetLoader:
    """Unified dataset loading"""

    @staticmethod
    def load_math500(path: str, num_samples: Optional[int] = None,
                     difficulty: Optional[str] = None, problem_type: Optional[str] = None) -> List[Dict]:
        """Load MATH500 dataset"""
        logger.info(f"Loading MATH500 from {path}")

        if not Path(path).exists():
            logger.warning(f"File not found, attempting HuggingFace download...")
            return DatasetLoader._load_from_hf(num_samples, difficulty, problem_type)

        with open(path) as f:
            data = json.load(f)

        # Apply filters
        if difficulty:
            data = [d for d in data if d.get('level') == difficulty]
        if problem_type:
            data = [d for d in data if d.get('type') == problem_type]
        if num_samples:
            data = data[:num_samples]

        logger.info(f"Loaded {len(data)} problems")
        return data

    @staticmethod
    def load_aime24(path: str) -> List[Dict]:
        """Load AIME24 dataset"""
        logger.info(f"Loading AIME24 from {path}")

        if not Path(path).exists():
            logger.warning(f"File not found, using sample data")
            return DatasetLoader._create_sample_aime()

        with open(path) as f:
            data = json.load(f)

        logger.info(f"Loaded {len(data)} AIME problems")
        return data

    @staticmethod
    def _load_from_hf(num_samples: Optional[int] = None,
                      difficulty: Optional[str] = None,
                      problem_type: Optional[str] = None) -> List[Dict]:
        """Load from HuggingFace"""
        try:
            from datasets import load_dataset

            dataset = load_dataset("hendrycks/competition_math", split="test")

            data = [
                {
                    'problem': item['problem'],
                    'solution': item['solution'],
                    'level': item.get('level', 'Unknown'),
                    'type': item.get('type', 'Unknown'),
                    'answer': AnswerExtractor.boxed(item['solution'])
                }
                for item in dataset
            ]

            # Apply filters
            if difficulty:
                data = [d for d in data if d['level'] == difficulty]
            if problem_type:
                data = [d for d in data if d['type'] == problem_type]

            data = data[:500]  # MATH500 subset

            if num_samples:
                data = data[:num_samples]

            logger.info(f"Downloaded {len(data)} problems from HuggingFace")
            return data

        except ImportError:
            logger.error("datasets library not installed. Install with: pip install datasets")
            return DatasetLoader._create_sample_math()

    @staticmethod
    def _create_sample_math() -> List[Dict]:
        """Sample MATH problems"""
        return [
            {
                "problem": "What is the value of $\\sqrt{49}$?",
                "solution": "The square root of 49 is 7. \\boxed{7}",
                "level": "Level 1",
                "type": "Algebra",
                "answer": "7"
            },
            {
                "problem": "If $x + 2 = 5$, what is the value of $x$?",
                "solution": "Subtracting 2: $x = 5 - 2 = 3$. \\boxed{3}",
                "level": "Level 1",
                "type": "Algebra",
                "answer": "3"
            },
            {
                "problem": "How many prime numbers are there between 1 and 10?",
                "solution": "The primes are 2, 3, 5, 7. That's 4 primes. \\boxed{4}",
                "level": "Level 2",
                "type": "Number Theory",
                "answer": "4"
            }
        ]

    @staticmethod
    def _create_sample_aime() -> List[Dict]:
        """Sample AIME problems"""
        return [
            {
                "problem": "What is the sum of all positive integers less than 1000 divisible by 7 but not by 11?",
                "answer": "60658"
            },
            {
                "problem": "A box has integer edges. Sum of edges is 48, sum of face areas is 96. What is the volume?",
                "answer": "64"
            }
        ]


# Convenience functions
def load_math500(*args, **kwargs):
    return DatasetLoader.load_math500(*args, **kwargs)


def load_aime24(*args, **kwargs):
    return DatasetLoader.load_aime24(*args, **kwargs)


def extract_final_answer_math(text: str) -> Optional[str]:
    return AnswerExtractor.generic(text)


def extract_integer_answer(text: str) -> Optional[int]:
    return AnswerExtractor.integer(text)


def extract_boxed_answer(text: str) -> Optional[str]:
    return AnswerExtractor.boxed(text)


def normalize_answer(text: str) -> str:
    return AnswerExtractor.normalize(text)


def create_sample_math500():
    return DatasetLoader._create_sample_math()


def create_sample_aime24():
    return DatasetLoader._create_sample_aime()
