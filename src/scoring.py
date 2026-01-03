"""
Scoring functions and length priors for Persistent Sampling.

Score types:
- logprob_power: τ * cumulative logprob
- logprob_avg: τ * (cum_logprob / length)
- prm_reward: PRM score from reward model
- hybrid_prm_logprob: α * PRM + β * logprob
- entropy_confidence: Confidence based on token entropy
- self_eval_prompted: Self-evaluation (placeholder)

Length priors:
- none: No length prior
- poisson: Poisson distribution
- geometric: Geometric distribution
- lognormal: Log-normal distribution
- linear_penalty: Linear penalty on length
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .config import (
    LengthPriorConfig,
    LengthPriorType,
    ScoreType,
    ScoringConfig,
)
from .utils import clip_values


# ============================================================================
# Length Prior Classes
# ============================================================================

class LengthPrior(ABC):
    """Abstract base class for length priors."""
    
    @abstractmethod
    def log_prob(self, length: int) -> float:
        """Compute log probability or log penalty for given length."""
        pass
    
    def __call__(self, length: int) -> float:
        return self.log_prob(length)


class NoLengthPrior(LengthPrior):
    """No length prior (returns 0)."""
    
    def log_prob(self, length: int) -> float:
        return 0.0


class PoissonLengthPrior(LengthPrior):
    """
    Poisson length prior.
    
    log P(k) = k * log(λ) - λ - log(k!)
    """
    
    def __init__(self, lambda_param: float = 50.0):
        self.lambda_param = lambda_param
        # Pre-compute log(λ) and λ
        self._log_lambda = np.log(lambda_param)
    
    def log_prob(self, length: int) -> float:
        if length < 0:
            return -np.inf
        # log P(k) = k * log(λ) - λ - log(k!)
        # Use Stirling approximation for log(k!) when k is large
        k = length
        if k == 0:
            return -self.lambda_param
        
        # log(k!) ≈ k*log(k) - k + 0.5*log(2πk) for large k
        # For small k, compute exactly
        if k <= 20:
            log_factorial = float(np.sum(np.log(np.arange(1, k + 1))))
        else:
            log_factorial = k * np.log(k) - k + 0.5 * np.log(2 * np.pi * k)
        
        return k * self._log_lambda - self.lambda_param - log_factorial


class GeometricLengthPrior(LengthPrior):
    """
    Geometric length prior.
    
    log P(k) = log(p) + k * log(1-p)
    """
    
    def __init__(self, p: float = 0.05):
        self.p = p
        self._log_p = np.log(p)
        self._log_1mp = np.log(1 - p)
    
    def log_prob(self, length: int) -> float:
        if length < 0:
            return -np.inf
        return self._log_p + length * self._log_1mp


class LognormalLengthPrior(LengthPrior):
    """
    Log-normal length prior.
    
    log P(k) ∝ -0.5 * ((log(k) - μ) / σ)^2 - log(k) - log(σ)
    """
    
    def __init__(self, mu: float = 3.5, sigma: float = 0.5):
        self.mu = mu
        self.sigma = sigma
        self._norm_const = -0.5 * np.log(2 * np.pi) - np.log(sigma)
    
    def log_prob(self, length: int) -> float:
        if length <= 0:
            return -np.inf
        
        log_k = np.log(length)
        z = (log_k - self.mu) / self.sigma
        
        return self._norm_const - 0.5 * z ** 2 - log_k


class LinearPenaltyLengthPrior(LengthPrior):
    """
    Linear penalty on length.
    
    score = slope * length
    """
    
    def __init__(self, slope: float = -0.01):
        self.slope = slope
    
    def log_prob(self, length: int) -> float:
        return self.slope * length


def create_length_prior(config: LengthPriorConfig) -> LengthPrior:
    """Factory function to create length prior from config."""
    prior_type = config.type
    params = config.params
    
    if prior_type == LengthPriorType.NONE:
        return NoLengthPrior()
    elif prior_type == LengthPriorType.POISSON:
        return PoissonLengthPrior(lambda_param=params.poisson_lambda)
    elif prior_type == LengthPriorType.GEOMETRIC:
        return GeometricLengthPrior(p=params.geometric_p)
    elif prior_type == LengthPriorType.LOGNORMAL:
        return LognormalLengthPrior(mu=params.lognormal_mu, sigma=params.lognormal_sigma)
    elif prior_type == LengthPriorType.LINEAR_PENALTY:
        return LinearPenaltyLengthPrior(slope=params.linear_penalty_slope)
    else:
        raise ValueError(f"Unknown length prior type: {prior_type}")


# ============================================================================
# Score Functions
# ============================================================================

@dataclass
class ParticleScoreData:
    """Data required for scoring a particle."""
    cum_logprob: float  # Cumulative log probability from LLM
    length: int  # Number of generated tokens (or steps)
    prm_score: Optional[float] = None  # PRM score if available
    token_entropies: Optional[List[float]] = None  # Per-token entropies
    
    # Additional fields for score caching
    score_value: Optional[float] = None
    logL: Optional[float] = None  # log likelihood value for PS weights


class ScoreFunction(ABC):
    """Abstract base class for scoring functions."""
    
    def __init__(
        self,
        length_prior: LengthPrior,
        lambda_len: float = 0.1,
        clip_min: float = -2000.0,
        clip_max: float = 2000.0,
    ):
        self.length_prior = length_prior
        self.lambda_len = lambda_len
        self.clip_min = clip_min
        self.clip_max = clip_max
    
    @abstractmethod
    def compute_base_score(self, data: ParticleScoreData) -> float:
        """Compute base score (before length prior)."""
        pass
    
    def compute_score(self, data: ParticleScoreData) -> float:
        """Compute total score including length prior."""
        base_score = self.compute_base_score(data)
        length_log_prob = self.length_prior.log_prob(data.length)
        total_score = base_score + self.lambda_len * length_log_prob
        return float(clip_values(np.array([total_score]), self.clip_min, self.clip_max)[0])
    
    def compute_batch(self, data_list: List[ParticleScoreData]) -> np.ndarray:
        """Compute scores for a batch of particles."""
        scores = np.array([self.compute_score(d) for d in data_list], dtype=np.float64)
        return scores
    
    def __call__(self, data: ParticleScoreData) -> float:
        return self.compute_score(data)


class LogprobPowerScore(ScoreFunction):
    """
    Score = τ * cumulative_logprob
    
    This directly uses the LLM's own likelihood.
    """
    
    def __init__(
        self,
        tau: float = 1.0,
        length_prior: LengthPrior = None,
        lambda_len: float = 0.1,
        clip_min: float = -2000.0,
        clip_max: float = 2000.0,
    ):
        super().__init__(
            length_prior=length_prior or NoLengthPrior(),
            lambda_len=lambda_len,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        self.tau = tau
    
    def compute_base_score(self, data: ParticleScoreData) -> float:
        return self.tau * data.cum_logprob


class LogprobAvgScore(ScoreFunction):
    """
    Score = τ * (cumulative_logprob / length)
    
    Average log probability per token.
    """
    
    def __init__(
        self,
        tau: float = 1.0,
        length_prior: LengthPrior = None,
        lambda_len: float = 0.1,
        clip_min: float = -2000.0,
        clip_max: float = 2000.0,
    ):
        super().__init__(
            length_prior=length_prior or NoLengthPrior(),
            lambda_len=lambda_len,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        self.tau = tau
    
    def compute_base_score(self, data: ParticleScoreData) -> float:
        if data.length == 0:
            return 0.0
        return self.tau * (data.cum_logprob / data.length)


class PRMRewardScore(ScoreFunction):
    """
    Score = prm_scale * PRM_score
    
    Uses external PRM for scoring.
    """
    
    def __init__(
        self,
        prm_scale: float = 1.0,
        length_prior: LengthPrior = None,
        lambda_len: float = 0.1,
        clip_min: float = -2000.0,
        clip_max: float = 2000.0,
    ):
        super().__init__(
            length_prior=length_prior or NoLengthPrior(),
            lambda_len=lambda_len,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        self.prm_scale = prm_scale
    
    def compute_base_score(self, data: ParticleScoreData) -> float:
        if data.prm_score is None:
            raise ValueError("PRM score required but not provided")
        return self.prm_scale * data.prm_score


class HybridPRMLogprobScore(ScoreFunction):
    """
    Score = prm_scale * PRM_score + logprob_scale * cumulative_logprob
    
    Combines PRM and logprob scoring.
    """
    
    def __init__(
        self,
        prm_scale: float = 1.0,
        logprob_scale: float = 0.1,
        length_prior: LengthPrior = None,
        lambda_len: float = 0.1,
        clip_min: float = -2000.0,
        clip_max: float = 2000.0,
    ):
        super().__init__(
            length_prior=length_prior or NoLengthPrior(),
            lambda_len=lambda_len,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        self.prm_scale = prm_scale
        self.logprob_scale = logprob_scale
    
    def compute_base_score(self, data: ParticleScoreData) -> float:
        if data.prm_score is None:
            raise ValueError("PRM score required but not provided")
        return self.prm_scale * data.prm_score + self.logprob_scale * data.cum_logprob


class EntropyConfidenceScore(ScoreFunction):
    """
    Score = -mean_entropy (lower entropy = higher confidence)
    
    Uses per-token entropy as a confidence measure.
    """
    
    def __init__(
        self,
        tau: float = 1.0,
        length_prior: LengthPrior = None,
        lambda_len: float = 0.1,
        clip_min: float = -2000.0,
        clip_max: float = 2000.0,
    ):
        super().__init__(
            length_prior=length_prior or NoLengthPrior(),
            lambda_len=lambda_len,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        self.tau = tau
    
    def compute_base_score(self, data: ParticleScoreData) -> float:
        if data.token_entropies is None or len(data.token_entropies) == 0:
            # Fallback to logprob-based proxy
            return self.tau * data.cum_logprob
        mean_entropy = np.mean(data.token_entropies)
        return -self.tau * mean_entropy


class SelfEvalPromptedScore(ScoreFunction):
    """
    Placeholder for self-evaluation via prompting.
    
    This would require the LLM to evaluate its own response.
    """
    
    def __init__(
        self,
        tau: float = 1.0,
        length_prior: LengthPrior = None,
        lambda_len: float = 0.1,
        clip_min: float = -2000.0,
        clip_max: float = 2000.0,
    ):
        super().__init__(
            length_prior=length_prior or NoLengthPrior(),
            lambda_len=lambda_len,
            clip_min=clip_min,
            clip_max=clip_max,
        )
        self.tau = tau
    
    def compute_base_score(self, data: ParticleScoreData) -> float:
        # Placeholder - would need LLM evaluation
        # For now, fall back to logprob
        return self.tau * data.cum_logprob


def create_score_function(config: ScoringConfig) -> ScoreFunction:
    """Factory function to create score function from config."""
    # Create length prior first
    length_prior = create_length_prior(config.length_prior)
    
    score_type = config.type
    params = config.params
    
    common_args = {
        "length_prior": length_prior,
        "lambda_len": config.length_prior.lambda_len,
        "clip_min": params.score_clip_min,
        "clip_max": params.score_clip_max,
    }
    
    if score_type == ScoreType.LOGPROB_POWER:
        return LogprobPowerScore(tau=params.tau, **common_args)
    elif score_type == ScoreType.LOGPROB_AVG:
        return LogprobAvgScore(tau=params.tau, **common_args)
    elif score_type == ScoreType.PRM_REWARD:
        return PRMRewardScore(prm_scale=params.prm_scale, **common_args)
    elif score_type == ScoreType.HYBRID_PRM_LOGPROB:
        return HybridPRMLogprobScore(
            prm_scale=params.prm_scale,
            logprob_scale=params.logprob_scale,
            **common_args
        )
    elif score_type == ScoreType.ENTROPY_CONFIDENCE:
        return EntropyConfidenceScore(tau=params.tau, **common_args)
    elif score_type == ScoreType.SELF_EVAL_PROMPTED:
        return SelfEvalPromptedScore(tau=params.tau, **common_args)
    else:
        raise ValueError(f"Unknown score type: {score_type}")


# ============================================================================
# Utility functions
# ============================================================================

def compute_length(
    num_tokens: int,
    num_steps: int,
    measure_by: str = "tokens"
) -> int:
    """
    Compute length based on measurement type.
    
    Args:
        num_tokens: Number of generated tokens
        num_steps: Number of generation steps
        measure_by: "tokens" or "steps"
    
    Returns:
        Length value
    """
    if measure_by == "tokens":
        return num_tokens
    elif measure_by == "steps":
        return num_steps
    else:
        raise ValueError(f"Unknown length measure: {measure_by}")
