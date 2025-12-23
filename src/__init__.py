"""
Persistent Sequential Monte Carlo for LLM Reasoning
Based on Karamanis & Seljak (arXiv:2407.20722)
"""

from .persistent_smc import PersistentSMC, Particle
from .vllm_wrapper import vLLMGenerator
from .dataset_loaders import load_math500, load_aime24
from .evaluator import MathEvaluator

__version__ = "0.1.0"
__all__ = [
    "PersistentSMC",
    "Particle",
    "vLLMGenerator",
    "load_math500",
    "load_aime24",
    "MathEvaluator",
]
