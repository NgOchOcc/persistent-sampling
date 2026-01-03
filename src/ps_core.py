"""
Persistent Sampling (PS) Algorithm 2 implementation for LLM test-time scaling.

Core PS concepts:
- Maintain a growing pool of particles across time (persistent pool)
- Use mixture importance density with balance-heuristic weights
- Adaptive temperature schedule via ESS bisection
- Resample from ALL historical particles when degeneracy occurs

Key PS equations implemented:
- Eq. 7: Adaptive β_t via ESS threshold bisection
- Eq. 13: Mixture importance density over past distributions
- Eq. 16: Z-hat estimator update
- Eq. 17: ESS over persistent weights
- Algorithm 2: Main PS loop

State space mapping for LLM:
- Each particle is a variable-length token sequence
- Each "time step" is a generation chunk (up to max_tokens_per_step tokens)
- Scores are cached cumulative logprobs or PRM scores
"""

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import (
    Config,
    PSConfig,
    ResamplePolicy,
    TemperatureScheduleMode,
    ScoreType,
)
from .resampling import Resampler, compute_resampling_stats
from .scoring import ParticleScoreData, ScoreFunction, create_score_function
from .utils import (
    TimingStats,
    compute_ess,
    extract_assistant_response,
    extract_boxed_answer,
    logmeanexp,
    logsumexp,
    normalize_log_weights,
)
from .vllm_wrappers import GenerationOutput, LLMGenerator, RewardModel


logger = logging.getLogger(__name__)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Particle:
    """
    A particle representing a partial or complete generation trajectory.
    
    Caches all values needed for PS weight computation to avoid recomputation.
    """
    particle_id: int
    token_ids: List[int]  # Full token sequence (prompt + generated)
    prompt_len: int  # Length of prompt tokens
    
    # Generation state
    step: int = 0  # Current generation step
    alive: bool = True  # Whether particle is still generating
    
    # Cached score values (critical for PS efficiency)
    cum_logprob: float = 0.0  # Cumulative log probability of generated tokens
    score_value: float = 0.0  # Total score including length prior
    logL: float = 0.0  # Log-likelihood for PS weights (= score_value typically)
    prm_score: Optional[float] = None  # PRM score if computed
    
    # Token-level data for potential move steps
    token_logprobs: List[float] = field(default_factory=list)
    
    @property
    def generated_tokens(self) -> List[int]:
        """Token IDs of generated content (excluding prompt)."""
        return self.token_ids[self.prompt_len:]
    
    @property
    def num_generated_tokens(self) -> int:
        """Number of generated tokens."""
        return len(self.token_ids) - self.prompt_len
    
    def copy(self, new_id: Optional[int] = None) -> "Particle":
        """Create a copy of this particle."""
        return Particle(
            particle_id=new_id if new_id is not None else self.particle_id,
            token_ids=self.token_ids.copy(),
            prompt_len=self.prompt_len,
            step=self.step,
            alive=self.alive,
            cum_logprob=self.cum_logprob,
            score_value=self.score_value,
            logL=self.logL,
            prm_score=self.prm_score,
            token_logprobs=self.token_logprobs.copy(),
        )


@dataclass
class Snapshot:
    """
    A snapshot of a particle at a specific time step.
    
    Stored in the persistent pool for PS resampling.
    Critical: logL must be cached at creation time.
    """
    particle_id: int
    token_ids: List[int]
    prompt_len: int
    step: int
    logL: float  # Cached log-likelihood
    cum_logprob: float
    alive: bool
    
    # Additional cached values
    score_value: float = 0.0
    prm_score: Optional[float] = None
    token_logprobs: List[float] = field(default_factory=list)
    
    def to_particle(self, new_id: int) -> Particle:
        """Convert snapshot back to particle."""
        return Particle(
            particle_id=new_id,
            token_ids=self.token_ids.copy(),
            prompt_len=self.prompt_len,
            step=self.step,
            alive=self.alive,
            cum_logprob=self.cum_logprob,
            score_value=self.score_value,
            logL=self.logL,
            prm_score=self.prm_score,
            token_logprobs=self.token_logprobs.copy() if self.token_logprobs else [],
        )


class SnapshotPool:
    """
    Persistent pool of particle snapshots.
    
    Stores snapshots efficiently:
    - Numpy arrays for scalar values (logL, step, etc.)
    - List of token_ids lists
    - Supports pruning strategies for memory management
    """
    
    def __init__(self, max_size: Optional[int] = None, keep_last_steps: Optional[int] = None):
        self.max_size = max_size
        self.keep_last_steps = keep_last_steps
        
        # Storage
        self.snapshots: List[Snapshot] = []
        
        # Cached numpy arrays for vectorized weight computation
        self._logL_cache: Optional[np.ndarray] = None
        self._step_cache: Optional[np.ndarray] = None
        self._dirty = True  # Cache invalidation flag
    
    def add(self, snapshot: Snapshot):
        """Add a snapshot to the pool."""
        self.snapshots.append(snapshot)
        self._dirty = True
    
    def add_batch(self, snapshots: List[Snapshot]):
        """Add multiple snapshots."""
        self.snapshots.extend(snapshots)
        self._dirty = True
    
    def _rebuild_cache(self):
        """Rebuild numpy caches from snapshots."""
        if not self._dirty:
            return
        
        if self.snapshots:
            self._logL_cache = np.array(
                [s.logL for s in self.snapshots], dtype=np.float64
            )
            self._step_cache = np.array(
                [s.step for s in self.snapshots], dtype=np.int32
            )
        else:
            self._logL_cache = np.array([], dtype=np.float64)
            self._step_cache = np.array([], dtype=np.int32)
        
        self._dirty = False
    
    @property
    def logL_array(self) -> np.ndarray:
        """Get cached logL values as numpy array."""
        self._rebuild_cache()
        return self._logL_cache
    
    @property
    def step_array(self) -> np.ndarray:
        """Get cached step values as numpy array."""
        self._rebuild_cache()
        return self._step_cache
    
    def __len__(self) -> int:
        return len(self.snapshots)
    
    def __getitem__(self, idx) -> Snapshot:
        return self.snapshots[idx]
    
    def prune(self, current_step: int):
        """
        Prune pool based on configured strategy.
        
        Args:
            current_step: Current generation step
        """
        if not self.snapshots:
            return
        
        # Prune by keeping only last K steps
        if self.keep_last_steps is not None:
            min_step = current_step - self.keep_last_steps
            self.snapshots = [s for s in self.snapshots if s.step >= min_step]
            self._dirty = True
        
        # Prune by max size (FIFO)
        if self.max_size is not None and len(self.snapshots) > self.max_size:
            # Keep most recent
            self.snapshots = self.snapshots[-self.max_size:]
            self._dirty = True
    
    def clear(self):
        """Clear the pool."""
        self.snapshots = []
        self._dirty = True
    
    def get_by_indices(self, indices: np.ndarray) -> List[Snapshot]:
        """Get snapshots by indices."""
        return [self.snapshots[i] for i in indices]


# ============================================================================
# PS Temperature Schedule
# ============================================================================

class TemperatureSchedule:
    """
    Manages the annealing temperature schedule β_t for PS.
    
    Modes:
    - adaptive_ess: Bisection search for β_t such that ESS >= α * N (Eq. 7)
    - fixed_linear: Linear schedule from beta_min to beta_max
    - fixed_cosine: Cosine annealing schedule
    """
    
    def __init__(self, config: PSConfig, max_steps: int):
        self.config = config.temperature_schedule
        self.max_steps = max_steps
        
        # History of betas
        self.betas: List[float] = []
        
        # For Z-hat estimation (Eq. 16)
        self.logZ_hats: List[float] = []
    
    @property
    def current_beta(self) -> float:
        """Current beta value."""
        return self.betas[-1] if self.betas else self.config.beta_min
    
    @property
    def beta_array(self) -> np.ndarray:
        """All betas as numpy array."""
        return np.array(self.betas, dtype=np.float64)
    
    @property
    def logZ_array(self) -> np.ndarray:
        """All logZ_hats as numpy array."""
        return np.array(self.logZ_hats, dtype=np.float64)
    
    def compute_fixed_beta(self, t: int) -> float:
        """Compute beta for fixed schedules."""
        if self.config.mode == TemperatureScheduleMode.FIXED_LINEAR:
            # Linear interpolation
            ratio = t / max(self.max_steps - 1, 1)
            return self.config.beta_min + ratio * (self.config.beta_max - self.config.beta_min)
        
        elif self.config.mode == TemperatureScheduleMode.FIXED_COSINE:
            # Cosine annealing
            ratio = t / max(self.max_steps - 1, 1)
            cos_ratio = 0.5 * (1 - np.cos(np.pi * ratio))
            return self.config.beta_min + cos_ratio * (self.config.beta_max - self.config.beta_min)
        
        else:
            return self.config.beta_min
    
    def compute_adaptive_beta(
        self,
        pool_logL: np.ndarray,
        n_particles: int,
        prev_beta: float,
    ) -> float:
        """
        Compute adaptive beta via ESS bisection (Eq. 7).
        
        Find β_t = inf { β ∈ [β_{t-1}, β_max] : ESS_t(β) >= α * N }
        
        Args:
            pool_logL: Log-likelihood values for all pool particles
            n_particles: Number of particles N
            prev_beta: Previous beta value β_{t-1}
        
        Returns:
            New beta value
        """
        if len(pool_logL) == 0:
            return prev_beta
        
        alpha = self.config.ess_alpha
        target_ess = alpha * n_particles
        
        beta_lo = prev_beta
        beta_hi = self.config.beta_max
        
        # If alpha >= 1, ESS target might be >= N
        # In this case, beta can stay at beta_lo initially
        
        def compute_ess_at_beta(beta: float) -> float:
            """Compute ESS for given beta using current pool."""
            log_weights = beta * pool_logL
            probs, _ = normalize_log_weights(log_weights)
            return compute_ess(probs)
        
        # Check if current beta already satisfies
        ess_lo = compute_ess_at_beta(beta_lo)
        if ess_lo >= target_ess:
            return beta_lo
        
        # Check if max beta satisfies
        ess_hi = compute_ess_at_beta(beta_hi)
        if ess_hi < target_ess:
            # Can't achieve target ESS, use max beta
            return beta_hi
        
        # Bisection search
        for _ in range(self.config.bisection_max_iter):
            if beta_hi - beta_lo < self.config.bisection_tol:
                break
            
            beta_mid = (beta_lo + beta_hi) / 2
            ess_mid = compute_ess_at_beta(beta_mid)
            
            if ess_mid >= target_ess:
                beta_hi = beta_mid
            else:
                beta_lo = beta_mid
        
        return beta_hi
    
    def update(
        self,
        t: int,
        pool_logL: np.ndarray,
        n_particles: int,
        logZ: float,
    ):
        """
        Update schedule for step t.
        
        Args:
            t: Current step (1-indexed)
            pool_logL: Log-likelihood values for pool
            n_particles: Number of alive particles
            logZ: Log normalizing constant estimate
        """
        if self.config.mode == TemperatureScheduleMode.ADAPTIVE_ESS:
            prev_beta = self.current_beta
            new_beta = self.compute_adaptive_beta(pool_logL, n_particles, prev_beta)
        else:
            new_beta = self.compute_fixed_beta(t)
        
        self.betas.append(new_beta)
        self.logZ_hats.append(logZ)


# ============================================================================
# PS Weight Computation
# ============================================================================

def compute_ps_weights_vectorized(
    pool_logL: np.ndarray,
    betas: np.ndarray,
    logZ_hats: np.ndarray,
    current_beta: float,
) -> Tuple[np.ndarray, float]:
    """
    Compute PS balance-heuristic weights (Algorithm 2, vectorized).
    
    For each snapshot i at historical step s:
        log w_i = β_t * logL_i - log( (1/(t-1)) * Σ_{s=1..t-1} exp(β_s * logL_i - logZ_s) )
    
    Vectorized implementation:
        A[s, i] = β_s * logL_i - logZ_s
        denom_i = logsumexp(A[:, i]) - log(t-1)
        log_w_i = β_t * logL_i - denom_i
    
    Args:
        pool_logL: Shape [M] - log-likelihood for each snapshot
        betas: Shape [t-1] - historical beta values
        logZ_hats: Shape [t-1] - historical log normalizing constants
        current_beta: Current beta_t
    
    Returns:
        Tuple of (normalized probability weights, log normalizing constant)
    """
    M = len(pool_logL)
    T = len(betas)
    
    if M == 0:
        return np.array([]), 0.0
    
    if T == 0:
        # First step: uniform weights
        probs = np.ones(M, dtype=np.float64) / M
        return probs, 0.0
    
    # Compute A[s, i] = β_s * logL_i - logZ_s
    # Shape: [T, M]
    A = betas[:, np.newaxis] * pool_logL[np.newaxis, :] - logZ_hats[:, np.newaxis]
    
    # Compute denominator via logsumexp over s (axis=0)
    # denom_i = logsumexp(A[:, i]) - log(T)
    denom = logsumexp(A, axis=0) - np.log(T)
    
    # Compute log weights
    # log_w_i = β_t * logL_i - denom_i
    log_weights = current_beta * pool_logL - denom
    
    # Normalize to probabilities
    probs, logZ = normalize_log_weights(log_weights)
    
    return probs, logZ


def compute_ps_ess(
    pool_logL: np.ndarray,
    betas: np.ndarray,
    logZ_hats: np.ndarray,
    current_beta: float,
) -> float:
    """
    Compute ESS over persistent weights (Eq. 17).
    
    Args:
        pool_logL: Log-likelihood for each snapshot
        betas: Historical beta values
        logZ_hats: Historical log normalizing constants
        current_beta: Current beta
    
    Returns:
        Effective sample size
    """
    probs, _ = compute_ps_weights_vectorized(
        pool_logL, betas, logZ_hats, current_beta
    )
    
    if len(probs) == 0:
        return 0.0
    
    return compute_ess(probs)


# ============================================================================
# PS Engine
# ============================================================================

@dataclass
class SamplingResult:
    """Result of PS sampling."""
    response: str
    answer: Optional[str]
    particles: List[Dict[str, Any]]
    resample_count: int
    timing: Dict[str, float]
    betas: List[float]
    ess_history: List[float]


class PersistentSampler:
    """
    Persistent Sampling engine for LLM test-time scaling.
    
    Implements PS Algorithm 2 adapted for discrete text generation:
    1. Initialize N particles with first generation step
    2. Main loop:
       a. Generate next step for alive particles
       b. Score and cache logL for each particle
       c. Add snapshots to persistent pool
       d. Update temperature schedule
       e. Compute persistent weights
       f. Check resample condition (ESS-based or always)
       g. If resample: select from entire pool, optionally move/rejuvenate
    3. Return results with majority voting
    """
    
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize components
        self.llm_generator = LLMGenerator(
            model_config=config.models.base_llm,
            generation_config=config.generation,
        )
        
        self.prm: Optional[RewardModel] = None
        if config.models.prm.enabled:
            self.prm = RewardModel(config=config.models.prm)
        
        self.score_fn = create_score_function(config.scoring)
        self.resampler = Resampler(
            config=config.ps.resample,
            max_steps=config.generation.max_steps,
            seed=config.system.seed,
        )
        
        # Timing
        self.timing = TimingStats()
    
    def initialize(self):
        """Initialize LLM and optionally PRM."""
        self.llm_generator.initialize()
        if self.prm is not None:
            self.prm.initialize()
    
    def _score_particle(
        self,
        particle: Particle,
        query: str,
        use_prm: bool = False,
    ) -> float:
        """
        Compute score for a particle.
        
        Args:
            particle: The particle to score
            query: Original query for PRM
            use_prm: Whether to use PRM scoring
        
        Returns:
            Score value
        """
        # Prepare score data
        length = particle.num_generated_tokens
        
        # Get PRM score if needed
        prm_score = None
        if use_prm and self.prm is not None:
            text = self.llm_generator.decode(particle.token_ids)
            response = extract_assistant_response(text)
            prm_score = self.prm.score_single(query, response)
            particle.prm_score = prm_score
        
        score_data = ParticleScoreData(
            cum_logprob=particle.cum_logprob,
            length=length,
            prm_score=prm_score or particle.prm_score,
        )
        
        return self.score_fn.compute_score(score_data)
    
    def _score_particles_batch(
        self,
        particles: List[Particle],
        query: str,
    ) -> List[float]:
        """
        Score a batch of particles.
        
        Uses PRM if enabled and score type requires it.
        """
        use_prm = (
            self.config.scoring.type in [ScoreType.PRM_REWARD, ScoreType.HYBRID_PRM_LOGPROB]
            and self.prm is not None
        )
        
        if use_prm:
            # Batch PRM scoring
            texts = [self.llm_generator.decode(p.token_ids) for p in particles]
            responses = [extract_assistant_response(t) for t in texts]
            queries = [query] * len(responses)
            prm_scores = self.prm.score_batch(queries, responses)
            
            for p, score in zip(particles, prm_scores):
                p.prm_score = score
        
        # Compute scores
        scores = []
        for p in particles:
            score = self._score_particle(p, query, use_prm=False)  # PRM already computed
            p.score_value = score
            p.logL = score  # Use score as logL for PS
            scores.append(score)
        
        return scores
    
    def _create_snapshots(self, particles: List[Particle]) -> List[Snapshot]:
        """Create snapshots from current particle states."""
        snapshots = []
        for p in particles:
            snapshots.append(Snapshot(
                particle_id=p.particle_id,
                token_ids=p.token_ids.copy(),
                prompt_len=p.prompt_len,
                step=p.step,
                logL=p.logL,
                cum_logprob=p.cum_logprob,
                alive=p.alive,
                score_value=p.score_value,
                prm_score=p.prm_score,
                token_logprobs=p.token_logprobs.copy() if p.token_logprobs else [],
            ))
        return snapshots
    
    def _generate_step(self, particles: List[Particle]) -> List[Particle]:
        """
        Generate one step for all alive particles.
        
        Updates particles in-place with new tokens and logprobs.
        """
        alive_particles = [p for p in particles if p.alive]
        if not alive_particles:
            return particles
        
        # Batch generate
        token_ids_list = [p.token_ids for p in alive_particles]
        outputs = self.llm_generator.continue_generation_batch(token_ids_list)
        
        # Update particles
        for p, output in zip(alive_particles, outputs):
            # Extend token IDs
            p.token_ids = p.token_ids + output.token_ids
            
            # Update logprobs
            p.token_logprobs.extend(output.logprobs)
            p.cum_logprob += output.cum_logprob
            
            # Update step
            p.step += 1
            
            # Check termination
            if output.is_finished or p.step >= self.config.generation.max_steps:
                p.alive = False
        
        return particles
    
    def _resample_from_pool(
        self,
        pool: SnapshotPool,
        n_select: int,
        weights: np.ndarray,
    ) -> Tuple[List[Particle], List[int]]:
        """
        Resample particles from the persistent pool.
        
        Args:
            pool: Snapshot pool
            n_select: Number of particles to select
            weights: Normalized probability weights
        
        Returns:
            Tuple of (new particles, selected indices)
        """
        indices = self.resampler.resample(weights, n_select)
        
        new_particles = []
        for new_id, idx in enumerate(indices):
            snapshot = pool[idx]
            new_particles.append(snapshot.to_particle(new_id))
        
        return new_particles, indices.tolist()
    
    def _majority_vote(self, particles: List[Particle]) -> Tuple[str, Optional[str]]:
        """
        Select final answer via majority voting.
        
        Tie-break by highest score.
        """
        answers = []
        responses = []
        scores = []
        
        for p in particles:
            text = self.llm_generator.decode(p.token_ids, skip_special_tokens=True)
            response = extract_assistant_response(text)
            answer = extract_boxed_answer(response)
            
            answers.append(answer or "")
            responses.append(response)
            scores.append(p.score_value)
        
        # Count non-empty answers
        answer_counts = Counter([a for a in answers if a])
        
        if answer_counts:
            most_common = answer_counts.most_common()
            max_votes = most_common[0][1]
            best_answers = [ans for ans, count in most_common if count == max_votes]
            
            # Tie-break by score
            best_idx = -1
            best_score = float('-inf')
            for i, (ans, score) in enumerate(zip(answers, scores)):
                if ans in best_answers and score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx >= 0:
                return responses[best_idx], answers[best_idx]
        
        # Fallback: best score
        best_idx = int(np.argmax(scores))
        return responses[best_idx], answers[best_idx] if answers[best_idx] else None
    
    def sample(self, prompt: str, query: str) -> SamplingResult:
        """
        Main sampling function implementing PS Algorithm 2.
        
        Args:
            prompt: Formatted prompt string
            query: Original query for scoring
        
        Returns:
            SamplingResult with response, answer, and diagnostics
        """
        config = self.config
        n_particles = config.generation.n_particles
        max_steps = config.generation.max_steps
        
        # Initialize
        self.timing.reset()
        
        with self.timing.time("initialization"):
            self.initialize()
            prompt_token_ids = self.llm_generator.encode(prompt)
            prompt_len = len(prompt_token_ids)
        
        # Create pool and temperature schedule
        pool = SnapshotPool(
            max_size=config.ps.pool.max_size,
            keep_last_steps=config.ps.pool.keep_last_steps,
        )
        temp_schedule = TemperatureSchedule(config.ps, max_steps)
        
        # Generate initial particles
        with self.timing.time("initial_generation"):
            init_output = self.llm_generator.generate_single(
                prompt_token_ids,
                n=n_particles,
            )
            
            particles = []
            for i, out in enumerate(init_output.outputs):
                particles.append(Particle(
                    particle_id=i,
                    token_ids=prompt_token_ids + out.token_ids,
                    prompt_len=prompt_len,
                    step=1,
                    alive=not out.is_finished,
                    cum_logprob=out.cum_logprob,
                    token_logprobs=out.logprobs.copy(),
                ))
        
        # Initial scoring
        with self.timing.time("scoring"):
            self._score_particles_batch(particles, query)
        
        # Add initial snapshots to pool
        alive_particles = [p for p in particles if p.alive]
        pool.add_batch(self._create_snapshots(alive_particles))
        
        # Initialize temperature schedule
        if alive_particles:
            logZ = float(logsumexp(pool.logL_array))
            temp_schedule.update(1, pool.logL_array, len(alive_particles), logZ)
        
        # Tracking
        resample_count = 0
        ess_history = []
        
        logger.info(
            f"Initial: {len(particles)} particles, "
            f"{len(alive_particles)} alive, pool={len(pool)}"
        )
        
        # Main loop
        iteration = 0
        max_iterations = max_steps * 2
        
        while iteration < max_iterations:
            iteration += 1
            
            n_alive = sum(1 for p in particles if p.alive)
            if n_alive == 0:
                logger.info("All particles completed, stopping")
                break
            
            # Single particle case: generate to completion
            if n_alive == 1:
                with self.timing.time("single_completion"):
                    alive_p = next(p for p in particles if p.alive)
                    remaining = max_steps - alive_p.step
                    max_new_tokens = min(
                        remaining * config.generation.max_tokens_per_step,
                        1024
                    )
                    
                    output = self.llm_generator.generate_single(
                        alive_p.token_ids,
                        n=1,
                        max_tokens=max_new_tokens,
                    ).outputs[0]
                    
                    alive_p.token_ids = alive_p.token_ids + output.token_ids
                    alive_p.cum_logprob += output.cum_logprob
                    alive_p.token_logprobs.extend(output.logprobs)
                    alive_p.alive = False
                    
                    self._score_particles_batch([alive_p], query)
                
                break
            
            # Compute PS weights and ESS
            with self.timing.time("weight_computation"):
                ps_weights, logZ = compute_ps_weights_vectorized(
                    pool.logL_array,
                    temp_schedule.beta_array,
                    temp_schedule.logZ_array,
                    temp_schedule.current_beta,
                )
                
                ess = compute_ess(ps_weights) if len(ps_weights) > 0 else 0.0
                ess_history.append(ess)
            
            # Log state
            alive_steps = [p.step for p in particles if p.alive]
            alive_scores = [p.score_value for p in particles if p.alive]
            
            if config.system.debug:
                logger.info(
                    f"Iter {iteration}: alive={n_alive}, "
                    f"steps={alive_steps}, "
                    f"scores={[f'{s:.3f}' for s in alive_scores]}, "
                    f"ESS={ess:.2f}/{n_alive}, "
                    f"β={temp_schedule.current_beta:.4f}, "
                    f"pool={len(pool)}"
                )
            
            # Check resample condition
            should_resample_now = False
            resample_policy = config.ps.resample.policy
            
            if resample_policy == ResamplePolicy.ALWAYS:
                should_resample_now = True
            elif resample_policy == ResamplePolicy.ESS:
                threshold = self.resampler.compute_ess_threshold(iteration)
                should_resample_now = ess < threshold * n_alive
            # ResamplePolicy.NEVER: never resample
            
            # Resample from persistent pool
            if should_resample_now and len(pool) > 0:
                with self.timing.time("resampling"):
                    resample_count += 1
                    
                    logger.info(
                        f">>> RESAMPLE #{resample_count} "
                        f"(ESS={ess:.2f}, pool={len(pool)})"
                    )
                    
                    new_particles, selected_indices = self._resample_from_pool(
                        pool, n_alive, ps_weights
                    )
                    
                    # Keep dead particles
                    dead_particles = [p for p in particles if not p.alive]
                    particles = new_particles + dead_particles
                    
                    if config.system.debug:
                        stats = compute_resampling_stats(ps_weights, np.array(selected_indices))
                        logger.info(f"  Resample stats: {stats}")
            
            # Generate next step
            with self.timing.time("generation"):
                particles = self._generate_step(particles)
            
            # Score updated particles
            with self.timing.time("scoring"):
                alive_particles = [p for p in particles if p.alive]
                if alive_particles:
                    self._score_particles_batch(alive_particles, query)
            
            # Add new snapshots to pool
            new_snapshots = self._create_snapshots(alive_particles)
            pool.add_batch(new_snapshots)
            
            # Update temperature schedule
            if alive_particles:
                logZ = float(logsumexp(pool.logL_array))
                temp_schedule.update(
                    iteration + 1,
                    pool.logL_array,
                    len(alive_particles),
                    logZ,
                )
            
            # Prune pool if needed
            pool.prune(iteration)
        
        # Final scoring for all particles
        with self.timing.time("final_scoring"):
            self._score_particles_batch(particles, query)
        
        # Majority vote
        with self.timing.time("voting"):
            response, answer = self._majority_vote(particles)
        
        # Prepare results
        particles_info = []
        for p in particles:
            text = self.llm_generator.decode(p.token_ids, skip_special_tokens=True)
            particles_info.append({
                "particle_id": p.particle_id,
                "step": p.step,
                "alive": p.alive,
                "score": p.score_value,
                "cum_logprob": p.cum_logprob,
                "answer": extract_boxed_answer(extract_assistant_response(text)) or "",
                "num_tokens": p.num_generated_tokens,
            })
        
        timing_stats = self.timing.get_stats()
        timing_summary = {k: v["total"] for k, v in timing_stats.items()}
        
        logger.info(f"Final answer: {answer}")
        logger.info(f"Total resamples: {resample_count}")
        logger.info(self.timing.summary())
        
        return SamplingResult(
            response=response,
            answer=answer,
            particles=particles_info,
            resample_count=resample_count,
            timing=timing_summary,
            betas=temp_schedule.betas,
            ess_history=ess_history,
        )
    
    def format_prompt(
        self,
        problem: str,
        instruction: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Format a problem into a prompt."""
        return self.llm_generator.format_chat_prompt(problem, instruction)
