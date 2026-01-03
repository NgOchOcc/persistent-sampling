"""
Configuration management using Pydantic for type validation.

Loads YAML config and supports CLI overrides.
"""

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Enums for config options
# ============================================================================

class ScoreType(str, Enum):
    LOGPROB_POWER = "logprob_power"
    LOGPROB_AVG = "logprob_avg"
    PRM_REWARD = "prm_reward"
    HYBRID_PRM_LOGPROB = "hybrid_prm_logprob"
    ENTROPY_CONFIDENCE = "entropy_confidence"
    SELF_EVAL_PROMPTED = "self_eval_prompted"


class LengthPriorType(str, Enum):
    NONE = "none"
    POISSON = "poisson"
    GEOMETRIC = "geometric"
    LOGNORMAL = "lognormal"
    LINEAR_PENALTY = "linear_penalty"


class TemperatureScheduleMode(str, Enum):
    ADAPTIVE_ESS = "adaptive_ess"
    FIXED_LINEAR = "fixed_linear"
    FIXED_COSINE = "fixed_cosine"


class ResamplePolicy(str, Enum):
    ESS = "ess"
    ALWAYS = "always"
    NEVER = "never"


class ResampleMethod(str, Enum):
    TOP_N = "topN"
    SYSTEMATIC = "systematic"
    MULTINOMIAL = "multinomial"


class ESSThresholdMode(str, Enum):
    FIXED_RHO = "fixed_rho"
    ALPHA_SCHEDULE = "alpha_schedule"


class MoveType(str, Enum):
    TRUNCATE_RESAMPLE = "truncate_resample"
    MH_TRUNCATE_RESAMPLE = "mh_truncate_resample"


class PruneStrategy(str, Enum):
    NONE = "none"
    FIFO = "fifo"
    LOWEST_WEIGHT = "lowest_weight"
    OLDEST_STEP = "oldest_step"


class TuningSearch(str, Enum):
    GRID = "grid"
    RANDOM = "random"


# ============================================================================
# Config sub-models
# ============================================================================

class SystemConfig(BaseModel):
    """System-level configuration."""
    seed: int = 42
    dtype_weights: str = "float64"
    log_level: str = "INFO"
    output_dir: str = "./outputs"
    debug: bool = False


class BaseLLMConfig(BaseModel):
    """Base LLM configuration."""
    model_path: str = "Qwen/Qwen2.5-7B-Instruct"
    device: str = "cuda:0"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = False


class PRMConfig(BaseModel):
    """Process Reward Model configuration."""
    enabled: bool = False
    model_path: str = "Qwen/Qwen2.5-Math-PRM-7B"
    device: str = "cuda:1"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9


class ModelsConfig(BaseModel):
    """Models configuration."""
    base_llm: BaseLLMConfig = Field(default_factory=BaseLLMConfig)
    prm: PRMConfig = Field(default_factory=PRMConfig)


class GenerationConfig(BaseModel):
    """Generation parameters."""
    n_particles: int = 8
    max_steps: int = 20
    max_tokens_per_step: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    stop_strings: List[str] = Field(default_factory=list)
    stop_token_ids: List[int] = Field(default_factory=list)
    return_logprobs: bool = True
    logprobs_k: int = 1  # Number of top logprobs to return


class TemperatureScheduleConfig(BaseModel):
    """PS temperature schedule configuration."""
    mode: TemperatureScheduleMode = TemperatureScheduleMode.ADAPTIVE_ESS
    ess_alpha: float = 0.9  # PS Eq.7 alpha (can be >1)
    beta_min: float = 0.0
    beta_max: float = 1.0
    bisection_tol: float = 1e-4
    bisection_max_iter: int = 50


class ESSThresholdConfig(BaseModel):
    """ESS threshold configuration."""
    mode: ESSThresholdMode = ESSThresholdMode.FIXED_RHO
    rho: float = 0.5
    alpha_start: float = 1.2
    alpha_end: float = 0.5


class ResampleConfig(BaseModel):
    """Resampling configuration."""
    policy: ResamplePolicy = ResamplePolicy.ESS
    method: ResampleMethod = ResampleMethod.TOP_N
    ess_threshold: ESSThresholdConfig = Field(default_factory=ESSThresholdConfig)
    relabel_particles: bool = True


class MoveConfig(BaseModel):
    """Move/rejuvenation step configuration."""
    enabled: bool = False
    type: MoveType = MoveType.TRUNCATE_RESAMPLE
    truncate_k_tokens: int = 32
    n_moves: int = 1
    proposal_temperature: float = 1.0


class PoolConfig(BaseModel):
    """Snapshot pool configuration."""
    max_size: Optional[int] = None
    keep_last_steps: Optional[int] = None
    prune_strategy: PruneStrategy = PruneStrategy.NONE


class PSConfig(BaseModel):
    """Persistent Sampling algorithm configuration."""
    temperature_schedule: TemperatureScheduleConfig = Field(default_factory=TemperatureScheduleConfig)
    resample: ResampleConfig = Field(default_factory=ResampleConfig)
    move: MoveConfig = Field(default_factory=MoveConfig)
    pool: PoolConfig = Field(default_factory=PoolConfig)


class ScoringParamsConfig(BaseModel):
    """Scoring function parameters."""
    tau: float = 1.0  # Power/scale for logprob
    prm_scale: float = 1.0
    logprob_scale: float = 1.0
    score_clip_min: float = -2000.0
    score_clip_max: float = 2000.0


class LengthPriorParamsConfig(BaseModel):
    """Length prior parameters."""
    # Poisson
    poisson_lambda: float = 50.0
    # Geometric
    geometric_p: float = 0.05
    # Lognormal
    lognormal_mu: float = 3.5
    lognormal_sigma: float = 0.5
    # Linear penalty
    linear_penalty_slope: float = -0.01
    # Length measurement
    measure_by: str = "tokens"  # "tokens" or "steps"


class LengthPriorConfig(BaseModel):
    """Length prior configuration."""
    type: LengthPriorType = LengthPriorType.NONE
    lambda_len: float = 0.1  # Weight of length prior in total score
    params: LengthPriorParamsConfig = Field(default_factory=LengthPriorParamsConfig)


class ScoringConfig(BaseModel):
    """Scoring configuration."""
    type: ScoreType = ScoreType.LOGPROB_POWER
    params: ScoringParamsConfig = Field(default_factory=ScoringParamsConfig)
    length_prior: LengthPriorConfig = Field(default_factory=LengthPriorConfig)


class ValidationDataConfig(BaseModel):
    """Validation data configuration placeholder."""
    path: Optional[str] = None
    num_samples: Optional[int] = None
    split: str = "validation"


class TuningConfig(BaseModel):
    """Tuning configuration."""
    enabled: bool = False
    search: TuningSearch = TuningSearch.GRID
    budget: int = 50
    metric: str = "accuracy"
    validation_data: ValidationDataConfig = Field(default_factory=ValidationDataConfig)
    math500_path: str = "./data/math500.json"


class EvaluationConfig(BaseModel):
    """Evaluation configuration."""
    use_math_verify: bool = True
    output_file: str = "math500_results.jsonl"
    num_samples: Optional[int] = None  # None = all


# ============================================================================
# Root config
# ============================================================================

class Config(BaseModel):
    """Root configuration."""
    system: SystemConfig = Field(default_factory=SystemConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    ps: PSConfig = Field(default_factory=PSConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """Load config from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data) if data else cls()
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save config to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
    
    def apply_overrides(self, overrides: Dict[str, Any]) -> "Config":
        """
        Apply dot-notation overrides to config.
        
        Example: {"ps.resample.method": "systematic"}
        """
        data = self.model_dump()
        
        for key, value in overrides.items():
            parts = key.split(".")
            d = data
            for part in parts[:-1]:
                d = d[part]
            d[parts[-1]] = value
        
        return Config(**data)


def parse_cli_overrides(override_strings: List[str]) -> Dict[str, Any]:
    """
    Parse CLI override strings like "ps.resample.method=systematic".
    
    Handles type conversion for common cases.
    """
    overrides = {}
    
    for override in override_strings:
        if "=" not in override:
            continue
        
        key, value = override.split("=", 1)
        key = key.strip()
        value = value.strip()
        
        # Type conversion
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.lower() == "none":
            value = None
        else:
            # Try numeric conversion
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string
        
        overrides[key] = value
    
    return overrides


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[List[str]] = None
) -> Config:
    """
    Load config from file with optional CLI overrides.
    
    Args:
        config_path: Path to YAML config file (optional)
        overrides: List of override strings like "ps.resample.method=systematic"
    
    Returns:
        Validated Config object
    """
    if config_path is not None:
        config = Config.from_yaml(config_path)
    else:
        config = Config()
    
    if overrides:
        override_dict = parse_cli_overrides(overrides)
        config = config.apply_overrides(override_dict)
    
    return config
