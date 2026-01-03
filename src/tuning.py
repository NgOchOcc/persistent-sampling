"""
Hyperparameter tuning module for Persistent Sampling.

Provides:
- Grid search
- Random search
- Validation-based metric tracking
- Best config selection

Tunable parameters:
A) Scoring: type, tau, score_clip
B) Length prior: lambda_len, type, params
C) Resampling: policy, method, ESS threshold
D) Move: enabled, truncate_k_tokens, proposal_temperature
"""

import itertools
import json
import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from .config import (
    Config,
    TuningConfig,
    TuningSearch,
    ScoreType,
    LengthPriorType,
    ResamplePolicy,
    ResampleMethod,
    ESSThresholdMode,
)


logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    """Result from a single tuning run."""
    config_overrides: Dict[str, Any]
    metrics: Dict[str, float]
    
    # Tracking
    run_id: int = 0
    num_samples: int = 0
    total_time: float = 0.0
    
    @property
    def primary_metric(self) -> float:
        """Get primary metric (accuracy by default)."""
        return self.metrics.get("accuracy", 0.0)


@dataclass
class TuningState:
    """State of the tuning process."""
    results: List[TuningResult] = field(default_factory=list)
    best_result: Optional[TuningResult] = None
    best_metric: float = 0.0
    
    def update(self, result: TuningResult, metric_name: str = "accuracy"):
        """Update state with new result."""
        self.results.append(result)
        
        metric = result.metrics.get(metric_name, 0.0)
        if metric > self.best_metric:
            self.best_metric = metric
            self.best_result = result
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of tuning state."""
        return {
            "total_runs": len(self.results),
            "best_metric": self.best_metric,
            "best_config": self.best_result.config_overrides if self.best_result else None,
        }


# ============================================================================
# Hyperparameter Space
# ============================================================================

# Default search space
DEFAULT_SEARCH_SPACE = {
    # Scoring
    "scoring.type": [
        ScoreType.LOGPROB_POWER.value,
        ScoreType.LOGPROB_AVG.value,
    ],
    "scoring.params.tau": [0.5, 1.0, 2.0, 4.0],
    
    # Length prior
    "scoring.length_prior.lambda_len": [0.0, 0.05, 0.1, 0.2],
    "scoring.length_prior.type": [
        LengthPriorType.NONE.value,
        LengthPriorType.LINEAR_PENALTY.value,
    ],
    
    # Resampling
    "ps.resample.policy": [
        ResamplePolicy.ESS.value,
        ResamplePolicy.ALWAYS.value,
    ],
    "ps.resample.method": [
        ResampleMethod.TOP_N.value,
        ResampleMethod.SYSTEMATIC.value,
        ResampleMethod.MULTINOMIAL.value,
    ],
    "ps.resample.ess_threshold.rho": [0.3, 0.5, 0.7],
    
    # Move (optional)
    "ps.move.enabled": [False],  # Start with disabled
}

# Extended search space with move step
EXTENDED_SEARCH_SPACE = {
    **DEFAULT_SEARCH_SPACE,
    "ps.move.enabled": [True, False],
    "ps.move.truncate_k_tokens": [16, 32, 64],
    "ps.move.proposal_temperature": [0.7, 1.0, 1.3],
}


def generate_grid_configs(
    search_space: Dict[str, List[Any]],
    base_config: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Generate all configurations from grid search space.
    
    Args:
        search_space: Dict mapping param names to list of values
        base_config: Base config overrides to include in all configs
    
    Yields:
        Config override dictionaries
    """
    base = base_config or {}
    
    keys = list(search_space.keys())
    values = list(search_space.values())
    
    for combination in itertools.product(*values):
        config = dict(base)
        for key, value in zip(keys, combination):
            config[key] = value
        yield config


def generate_random_configs(
    search_space: Dict[str, List[Any]],
    n_samples: int,
    base_config: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """
    Generate random configurations from search space.
    
    Args:
        search_space: Dict mapping param names to list of values
        n_samples: Number of configs to generate
        base_config: Base config overrides
        seed: Random seed
    
    Yields:
        Config override dictionaries
    """
    if seed is not None:
        random.seed(seed)
    
    base = base_config or {}
    
    for _ in range(n_samples):
        config = dict(base)
        for key, values in search_space.items():
            config[key] = random.choice(values)
        yield config


def estimate_grid_size(search_space: Dict[str, List[Any]]) -> int:
    """Estimate total number of configurations in grid."""
    total = 1
    for values in search_space.values():
        total *= len(values)
    return total


# ============================================================================
# Tuning Runner
# ============================================================================

class TuningRunner:
    """
    Hyperparameter tuning runner.
    
    Supports:
    - Grid search
    - Random search
    - Validation metric tracking
    - Result logging and best config selection
    """
    
    def __init__(
        self,
        base_config: Config,
        eval_fn: Callable[[Config], Dict[str, float]],
        output_dir: str = "./tuning_results",
        metric_name: str = "accuracy",
    ):
        """
        Initialize tuning runner.
        
        Args:
            base_config: Base configuration
            eval_fn: Function that takes Config and returns metrics dict
            output_dir: Directory for saving results
            metric_name: Primary metric name for comparison
        """
        self.base_config = base_config
        self.eval_fn = eval_fn
        self.output_dir = Path(output_dir)
        self.metric_name = metric_name
        
        self.state = TuningState()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _apply_overrides(self, overrides: Dict[str, Any]) -> Config:
        """Apply overrides to base config."""
        return self.base_config.apply_overrides(overrides)
    
    def _run_single(self, overrides: Dict[str, Any], run_id: int) -> TuningResult:
        """Run evaluation with single config."""
        logger.info(f"Run {run_id}: {overrides}")
        
        config = self._apply_overrides(overrides)
        
        import time
        start_time = time.time()
        
        try:
            metrics = self.eval_fn(config)
        except Exception as e:
            logger.error(f"Run {run_id} failed: {e}")
            metrics = {self.metric_name: 0.0, "error": str(e)}
        
        elapsed = time.time() - start_time
        
        result = TuningResult(
            config_overrides=overrides,
            metrics=metrics,
            run_id=run_id,
            total_time=elapsed,
        )
        
        logger.info(
            f"Run {run_id} completed: "
            f"{self.metric_name}={metrics.get(self.metric_name, 0.0):.4f}, "
            f"time={elapsed:.1f}s"
        )
        
        return result
    
    def run_grid_search(
        self,
        search_space: Optional[Dict[str, List[Any]]] = None,
        budget: Optional[int] = None,
    ) -> TuningState:
        """
        Run grid search over search space.
        
        Args:
            search_space: Search space (uses default if None)
            budget: Maximum number of runs (None = all combinations)
        
        Returns:
            TuningState with all results
        """
        space = search_space or DEFAULT_SEARCH_SPACE
        
        total_configs = estimate_grid_size(space)
        logger.info(f"Grid search: {total_configs} total configurations")
        
        if budget:
            logger.info(f"Budget limit: {budget} runs")
        
        run_id = 0
        for overrides in generate_grid_configs(space):
            if budget and run_id >= budget:
                break
            
            result = self._run_single(overrides, run_id)
            self.state.update(result, self.metric_name)
            
            # Save intermediate results
            self._save_results()
            
            run_id += 1
        
        logger.info(f"Grid search complete: {run_id} runs")
        logger.info(f"Best {self.metric_name}: {self.state.best_metric:.4f}")
        
        return self.state
    
    def run_random_search(
        self,
        search_space: Optional[Dict[str, List[Any]]] = None,
        budget: int = 50,
        seed: Optional[int] = None,
    ) -> TuningState:
        """
        Run random search over search space.
        
        Args:
            search_space: Search space (uses default if None)
            budget: Number of random configurations to try
            seed: Random seed
        
        Returns:
            TuningState with all results
        """
        space = search_space or DEFAULT_SEARCH_SPACE
        
        logger.info(f"Random search: {budget} configurations")
        
        configs = list(generate_random_configs(space, budget, seed=seed))
        
        for run_id, overrides in enumerate(configs):
            result = self._run_single(overrides, run_id)
            self.state.update(result, self.metric_name)
            
            # Save intermediate results
            self._save_results()
        
        logger.info(f"Random search complete: {budget} runs")
        logger.info(f"Best {self.metric_name}: {self.state.best_metric:.4f}")
        
        return self.state
    
    def run(
        self,
        search_type: TuningSearch = TuningSearch.GRID,
        search_space: Optional[Dict[str, List[Any]]] = None,
        budget: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> TuningState:
        """
        Run tuning with specified search type.
        
        Args:
            search_type: Grid or random search
            search_space: Custom search space
            budget: Maximum runs
            seed: Random seed for random search
        
        Returns:
            TuningState with results
        """
        if search_type == TuningSearch.GRID:
            return self.run_grid_search(search_space, budget)
        elif search_type == TuningSearch.RANDOM:
            return self.run_random_search(search_space, budget or 50, seed)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
    
    def _save_results(self):
        """Save current results to file."""
        results_file = self.output_dir / "tuning_results.jsonl"
        
        with open(results_file, "w") as f:
            for result in self.state.results:
                f.write(json.dumps({
                    "run_id": result.run_id,
                    "config_overrides": result.config_overrides,
                    "metrics": result.metrics,
                    "total_time": result.total_time,
                }, ensure_ascii=False) + "\n")
        
        # Save best config
        if self.state.best_result:
            best_file = self.output_dir / "best_config.json"
            with open(best_file, "w") as f:
                json.dump({
                    "config_overrides": self.state.best_result.config_overrides,
                    "metrics": self.state.best_result.metrics,
                    "run_id": self.state.best_result.run_id,
                }, f, indent=2)
    
    def print_summary(self):
        """Print tuning summary."""
        summary = self.state.summary()
        
        print("\n" + "=" * 60)
        print("Tuning Summary")
        print("=" * 60)
        print(f"Total runs: {summary['total_runs']}")
        print(f"Best {self.metric_name}: {summary['best_metric']:.4f}")
        print(f"Best config: {summary['best_config']}")
        print("=" * 60)
        
        # Print top 5 results
        sorted_results = sorted(
            self.state.results,
            key=lambda r: r.metrics.get(self.metric_name, 0.0),
            reverse=True
        )
        
        print("\nTop 5 configurations:")
        for i, result in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {self.metric_name}={result.metrics.get(self.metric_name, 0.0):.4f}")
            print(f"     Config: {result.config_overrides}")


def create_tuning_eval_fn(
    sampler_class,
    dataset_loader,
    verify_fn: Callable,
    num_samples: Optional[int] = None,
) -> Callable[[Config], Dict[str, float]]:
    """
    Create evaluation function for tuning.
    
    Args:
        sampler_class: PersistentSampler class
        dataset_loader: Dataset loader
        verify_fn: Answer verification function
        num_samples: Number of samples to evaluate
    
    Returns:
        Evaluation function
    """
    def eval_fn(config: Config) -> Dict[str, float]:
        """Evaluate config on validation set."""
        import time
        
        sampler = sampler_class(config)
        
        correct = 0
        total = 0
        total_time = 0.0
        total_resamples = 0
        
        samples = list(dataset_loader)
        if num_samples:
            samples = samples[:num_samples]
        
        for problem in samples:
            prompt, query = sampler.format_prompt(problem.problem)
            
            start = time.time()
            result = sampler.sample(prompt, query)
            elapsed = time.time() - start
            
            is_correct = verify_fn(result.response, problem.answer)
            
            if is_correct:
                correct += 1
            total += 1
            total_time += elapsed
            total_resamples += result.resample_count
        
        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "avg_time": total_time / total if total > 0 else 0.0,
            "avg_resamples": total_resamples / total if total > 0 else 0.0,
        }
    
    return eval_fn
