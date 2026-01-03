"""
Utility functions for logging, seeding, numerics, and timing.
"""

import logging
import os
import random
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================================
# Logging setup
# ============================================================================

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    name: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging with optional file output.
    
    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        name: Logger name (None for root logger)
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# Seeding
# ============================================================================

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across numpy, random, and torch.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # For full reproducibility (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


# ============================================================================
# Numerical utilities (log-space operations)
# ============================================================================

def logsumexp(a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    """
    Compute log(sum(exp(a))) in a numerically stable way.
    
    Uses the max-subtraction trick for stability.
    
    Args:
        a: Input array (float64 recommended)
        axis: Axis to sum over
        keepdims: Whether to keep reduced dimensions
    
    Returns:
        Log-sum-exp result
    """
    a = np.asarray(a, dtype=np.float64)
    
    if a.size == 0:
        return np.array(-np.inf, dtype=np.float64)
    
    a_max = np.max(a, axis=axis, keepdims=True)
    
    # Handle -inf max (all elements are -inf)
    a_max = np.where(np.isfinite(a_max), a_max, 0)
    
    result = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    
    if not keepdims:
        result = np.squeeze(result, axis=axis)
    
    return result


def logmeanexp(a: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
    """
    Compute log(mean(exp(a))) in a numerically stable way.
    
    Args:
        a: Input array
        axis: Axis to compute over
        keepdims: Whether to keep reduced dimensions
    
    Returns:
        Log-mean-exp result
    """
    a = np.asarray(a, dtype=np.float64)
    
    if axis is None:
        n = a.size
    else:
        n = a.shape[axis]
    
    return logsumexp(a, axis=axis, keepdims=keepdims) - np.log(n)


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute log-softmax in a numerically stable way.
    
    Args:
        logits: Input logits
        axis: Axis to normalize over
    
    Returns:
        Log probabilities
    """
    logits = np.asarray(logits, dtype=np.float64)
    return logits - logsumexp(logits, axis=axis, keepdims=True)


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax in a numerically stable way.
    
    Args:
        logits: Input logits
        axis: Axis to normalize over
    
    Returns:
        Probabilities
    """
    log_probs = log_softmax(logits, axis=axis)
    return np.exp(log_probs)


def normalize_log_weights(log_weights: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Normalize log weights to probabilities and return log normalizing constant.
    
    Args:
        log_weights: Unnormalized log weights
    
    Returns:
        Tuple of (normalized probabilities, log normalizing constant)
    """
    log_weights = np.asarray(log_weights, dtype=np.float64)
    log_Z = logsumexp(log_weights)
    probs = np.exp(log_weights - log_Z)
    
    # Ensure probs sum to 1 due to numerical precision
    probs = probs / probs.sum()
    
    return probs, float(log_Z)


def compute_ess(weights: np.ndarray) -> float:
    """
    Compute Effective Sample Size from normalized weights.
    
    ESS = 1 / sum(w_i^2)
    
    Args:
        weights: Normalized probability weights (should sum to 1)
    
    Returns:
        Effective sample size
    """
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure normalized
    weights = weights / weights.sum()
    
    return 1.0 / np.sum(weights ** 2)


def compute_ess_from_log_weights(log_weights: np.ndarray) -> float:
    """
    Compute ESS directly from log weights.
    
    Args:
        log_weights: Unnormalized log weights
    
    Returns:
        Effective sample size
    """
    probs, _ = normalize_log_weights(log_weights)
    return compute_ess(probs)


def clip_values(
    values: np.ndarray,
    min_val: float = -2000.0,
    max_val: float = 2000.0
) -> np.ndarray:
    """
    Clip values to avoid overflow in exp operations.
    
    Args:
        values: Input array
        min_val: Minimum value
        max_val: Maximum value
    
    Returns:
        Clipped array
    """
    return np.clip(values, min_val, max_val)


# ============================================================================
# Timing utilities
# ============================================================================

class Timer:
    """Context manager and decorator for timing code blocks."""
    
    def __init__(self, name: str = "timer", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start_time
        if self.logger:
            self.logger.debug(f"{self.name}: {self.elapsed:.4f}s")
    
    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result
        return wrapper


class TimingStats:
    """Accumulate timing statistics across multiple phases."""
    
    def __init__(self):
        self.times: Dict[str, List[float]] = {}
    
    @contextmanager
    def time(self, name: str):
        """Context manager to time a phase."""
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(elapsed)
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all timed phases."""
        stats = {}
        for name, times in self.times.items():
            times_arr = np.array(times)
            stats[name] = {
                "count": len(times),
                "total": float(np.sum(times_arr)),
                "mean": float(np.mean(times_arr)),
                "std": float(np.std(times_arr)),
                "min": float(np.min(times_arr)),
                "max": float(np.max(times_arr)),
            }
        return stats
    
    def summary(self) -> str:
        """Get formatted summary string."""
        stats = self.get_stats()
        lines = ["Timing Summary:"]
        for name, s in sorted(stats.items(), key=lambda x: -x[1]["total"]):
            lines.append(
                f"  {name}: total={s['total']:.3f}s, "
                f"mean={s['mean']:.4f}s, count={s['count']}"
            )
        return "\n".join(lines)
    
    def reset(self):
        """Reset all timing statistics."""
        self.times.clear()


# ============================================================================
# Data structure helpers
# ============================================================================

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Nested dictionary
        parent_key: Prefix for keys
        sep: Separator between key levels
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d: Dict[str, Any], sep: str = ".") -> Dict[str, Any]:
    """
    Unflatten a dictionary with dot-notation keys.
    
    Args:
        d: Flattened dictionary
        sep: Separator between key levels
    
    Returns:
        Nested dictionary
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return result


# ============================================================================
# Answer extraction utilities
# ============================================================================

import re


def extract_boxed_answer(text: str) -> Optional[str]:
    """
    Extract answer from \\boxed{...} format.
    
    Handles nested braces.
    
    Args:
        text: Input text containing boxed answer
    
    Returns:
        Extracted answer string or None
    """
    # Pattern handles one level of nested braces
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_assistant_response(text: str) -> str:
    """
    Extract assistant response from chat-formatted text.
    
    Args:
        text: Full decoded text including chat markers
    
    Returns:
        Assistant's response portion
    """
    if "\nassistant\n" in text:
        return text.split("\nassistant\n")[-1]
    elif "assistant\n" in text:
        return text.split("assistant\n")[-1]
    elif "<|assistant|>" in text:
        return text.split("<|assistant|>")[-1]
    return text
