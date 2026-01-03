"""
Resampling methods for Persistent Sampling.

Methods:
- topN: Deterministic selection of top N particles by weight
- systematic: Systematic resampling (low variance)
- multinomial: Multinomial sampling with replacement

Features:
- Time-adaptive alpha threshold for ESS-triggered resampling
- Vectorized implementations
"""

from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from .config import ESSThresholdConfig, ESSThresholdMode, ResampleConfig, ResampleMethod
from .utils import compute_ess, normalize_log_weights


# ============================================================================
# Resampling Functions
# ============================================================================

def resample_topN(
    weights: np.ndarray,
    n: int,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Deterministic top-N selection.
    
    Uses argpartition for O(M) selection of top N indices.
    
    Args:
        weights: Normalized probability weights (shape: [M])
        n: Number of particles to select
        rng: Random generator (unused, for API consistency)
    
    Returns:
        Selected indices (shape: [n])
    """
    weights = np.asarray(weights, dtype=np.float64)
    m = len(weights)
    
    if n >= m:
        return np.arange(m)
    
    # Use argpartition for O(M) selection
    # Negate weights to get top (largest) values
    idx = np.argpartition(-weights, n)[:n]
    
    # Sort selected indices by weight (descending) for determinism
    sorted_order = np.argsort(-weights[idx])
    return idx[sorted_order]


def resample_systematic(
    weights: np.ndarray,
    n: int,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Systematic resampling (low-variance resampling).
    
    Uses a single random offset and evenly spaced points on the CDF.
    
    Args:
        weights: Normalized probability weights (shape: [M])
        n: Number of particles to select
        rng: Random generator (uses default if None)
    
    Returns:
        Selected indices (shape: [n])
    """
    if rng is None:
        rng = np.random.default_rng()
    
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()  # Ensure normalized
    
    m = len(weights)
    
    # Compute cumulative sum (CDF)
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0  # Ensure exactly 1.0
    
    # Generate systematic positions
    u = rng.uniform(0, 1.0 / n)
    positions = u + np.arange(n) / n
    
    # Select indices using searchsorted
    indices = np.searchsorted(cdf, positions)
    
    # Clip to valid range
    indices = np.clip(indices, 0, m - 1)
    
    return indices


def resample_multinomial(
    weights: np.ndarray,
    n: int,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Multinomial resampling (sampling with replacement).
    
    Args:
        weights: Normalized probability weights (shape: [M])
        n: Number of particles to select
        rng: Random generator (uses default if None)
    
    Returns:
        Selected indices (shape: [n])
    """
    if rng is None:
        rng = np.random.default_rng()
    
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()  # Ensure normalized
    
    m = len(weights)
    indices = rng.choice(m, size=n, replace=True, p=weights)
    
    return indices


def resample(
    weights: np.ndarray,
    n: int,
    method: ResampleMethod = ResampleMethod.TOP_N,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generic resampling function.
    
    Args:
        weights: Normalized probability weights (shape: [M])
        n: Number of particles to select
        method: Resampling method
        rng: Random generator
    
    Returns:
        Selected indices (shape: [n])
    """
    if method == ResampleMethod.TOP_N:
        return resample_topN(weights, n, rng)
    elif method == ResampleMethod.SYSTEMATIC:
        return resample_systematic(weights, n, rng)
    elif method == ResampleMethod.MULTINOMIAL:
        return resample_multinomial(weights, n, rng)
    else:
        raise ValueError(f"Unknown resample method: {method}")


# ============================================================================
# ESS Threshold / Resample Triggering
# ============================================================================

def compute_alpha_threshold(
    t: int,
    T: int,
    config: ESSThresholdConfig
) -> float:
    """
    Compute time-adaptive alpha threshold for resampling.
    
    For mode=alpha_schedule:
        α_t = α_start + (α_end - α_start) * (t / T)
    
    For mode=fixed_rho:
        Returns rho (constant)
    
    Args:
        t: Current time step
        T: Maximum time steps
        config: ESS threshold configuration
    
    Returns:
        Alpha threshold value
    """
    if config.mode == ESSThresholdMode.FIXED_RHO:
        return config.rho
    elif config.mode == ESSThresholdMode.ALPHA_SCHEDULE:
        ratio = t / max(T, 1)
        alpha = config.alpha_start + (config.alpha_end - config.alpha_start) * ratio
        return alpha
    else:
        raise ValueError(f"Unknown ESS threshold mode: {config.mode}")


def should_resample(
    ess: float,
    n_alive: int,
    t: int,
    T: int,
    config: ESSThresholdConfig
) -> Tuple[bool, float]:
    """
    Determine if resampling should be triggered based on ESS.
    
    Trigger condition: ESS < α_t * N_alive
    
    Args:
        ess: Current effective sample size
        n_alive: Number of alive particles
        t: Current time step
        T: Maximum time steps
        config: ESS threshold configuration
    
    Returns:
        Tuple of (should_resample, threshold_value)
    """
    alpha = compute_alpha_threshold(t, T, config)
    threshold = alpha * n_alive
    
    return ess < threshold, threshold


# ============================================================================
# Resampler Class (Stateful)
# ============================================================================

class Resampler:
    """
    Stateful resampler with configurable method and threshold.
    
    Handles:
    - Method selection (topN, systematic, multinomial)
    - ESS-based triggering
    - Time-adaptive threshold scheduling
    """
    
    def __init__(
        self,
        config: ResampleConfig,
        max_steps: int,
        seed: Optional[int] = None
    ):
        self.config = config
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
    
    def compute_ess_threshold(self, t: int) -> float:
        """Compute ESS threshold for current step."""
        return compute_alpha_threshold(t, self.max_steps, self.config.ess_threshold)
    
    def should_resample(self, ess: float, n_alive: int, t: int) -> Tuple[bool, float]:
        """Check if resampling should be triggered."""
        return should_resample(
            ess=ess,
            n_alive=n_alive,
            t=t,
            T=self.max_steps,
            config=self.config.ess_threshold
        )
    
    def resample(
        self,
        weights: np.ndarray,
        n: int
    ) -> np.ndarray:
        """
        Perform resampling using configured method.
        
        Args:
            weights: Normalized probability weights
            n: Number of particles to select
        
        Returns:
            Selected indices
        """
        return resample(
            weights=weights,
            n=n,
            method=self.config.method,
            rng=self.rng
        )
    
    def resample_from_log_weights(
        self,
        log_weights: np.ndarray,
        n: int
    ) -> Tuple[np.ndarray, float]:
        """
        Resample from log weights.
        
        Args:
            log_weights: Unnormalized log weights
            n: Number of particles to select
        
        Returns:
            Tuple of (selected indices, log normalizing constant)
        """
        probs, log_Z = normalize_log_weights(log_weights)
        indices = self.resample(probs, n)
        return indices, log_Z


# ============================================================================
# Utility Functions
# ============================================================================

def count_unique_particles(indices: np.ndarray) -> int:
    """Count number of unique particles after resampling."""
    return len(np.unique(indices))


def compute_resampling_stats(
    weights: np.ndarray,
    indices: np.ndarray
) -> dict:
    """
    Compute statistics about resampling.
    
    Args:
        weights: Original weights
        indices: Selected indices
    
    Returns:
        Dictionary with statistics
    """
    n_unique = count_unique_particles(indices)
    n_selected = len(indices)
    
    # Count how many times each particle was selected
    unique, counts = np.unique(indices, return_counts=True)
    max_copies = int(np.max(counts))
    
    # Weight statistics
    selected_weights = weights[indices]
    
    return {
        "n_unique": n_unique,
        "n_selected": n_selected,
        "diversity": n_unique / n_selected,
        "max_copies": max_copies,
        "selected_weight_mean": float(np.mean(selected_weights)),
        "selected_weight_std": float(np.std(selected_weights)),
    }
