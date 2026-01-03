"""
Evaluation script for Persistent Sampling on MATH-500 benchmark.

Config-driven evaluation pipeline:
- Loads config from YAML with CLI overrides
- Runs PS sampling on MATH-500
- Verifies answers with math_verify (fallback to string match)
- Saves results incrementally to JSONL

Usage:
    python -m src.eval_math500 --config config.yaml
    python -m src.eval_math500 --config config.yaml --override "ps.resample.method=systematic"
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .config import Config, load_config
from .data import Math500Loader
from .ps_core import PersistentSampler
from .utils import extract_boxed_answer, set_seed, setup_logging


logger = logging.getLogger(__name__)


# ============================================================================
# Answer Verification
# ============================================================================

def verify_answer_math_verify(response: str, ground_truth: str) -> bool:
    """
    Verify answer using math_verify library.
    
    Args:
        response: Model response containing boxed answer
        ground_truth: Ground truth answer
    
    Returns:
        Whether answer is correct
    """
    try:
        from math_verify import parse, verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
        
        gold_parsed = parse(
            f"\\boxed{{{ground_truth}}}",
            extraction_config=[LatexExtractionConfig()]
        )
        pred_parsed = parse(
            response,
            extraction_config=[ExprExtractionConfig(), LatexExtractionConfig()]
        )
        return verify(gold_parsed, pred_parsed)
    except Exception as e:
        logger.debug(f"math_verify failed: {e}")
        return False


def verify_answer_string_match(response: str, ground_truth: str) -> bool:
    """
    Verify answer using simple string matching.
    
    Args:
        response: Model response containing boxed answer
        ground_truth: Ground truth answer
    
    Returns:
        Whether answer matches
    """
    pred_answer = extract_boxed_answer(response)
    if pred_answer is None:
        return False
    return pred_answer.strip() == ground_truth.strip()


def create_verifier(use_math_verify: bool = True):
    """
    Create answer verification function.
    
    Args:
        use_math_verify: Whether to use math_verify library
    
    Returns:
        Verification function
    """
    # Check if math_verify is available
    math_verify_available = False
    if use_math_verify:
        try:
            from math_verify import parse, verify
            math_verify_available = True
            logger.info("Using math_verify for answer verification")
        except ImportError:
            logger.warning(
                "math_verify not available. Using string matching. "
                "Install with: pip install math-verify[antlr4_13_2]"
            )
    
    def verify_fn(response: str, ground_truth: str) -> bool:
        if math_verify_available:
            if verify_answer_math_verify(response, ground_truth):
                return True
            # Fallback to string match
            return verify_answer_string_match(response, ground_truth)
        return verify_answer_string_match(response, ground_truth)
    
    return verify_fn


# ============================================================================
# Evaluation Pipeline
# ============================================================================

def evaluate_math500(config: Config) -> dict:
    """
    Run evaluation on MATH-500 benchmark.
    
    Args:
        config: Configuration object
    
    Returns:
        Evaluation results dictionary
    """
    # Setup
    set_seed(config.system.seed)
    os.makedirs(config.system.output_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"Loading MATH-500 from: {config.tuning.math500_path}")
    dataset = Math500Loader(
        path=config.tuning.math500_path,
        num_samples=config.evaluation.num_samples,
    )
    
    logger.info(f"Total samples: {len(dataset)}")
    
    # Initialize sampler
    logger.info("Initializing Persistent Sampler...")
    sampler = PersistentSampler(config)
    sampler.initialize()
    
    # Log configuration summary
    logger.info(f"Model: {config.models.base_llm.model_path}")
    logger.info(f"PRM: {config.models.prm.model_path} (enabled={config.models.prm.enabled})")
    logger.info(f"N particles: {config.generation.n_particles}")
    logger.info(f"Max steps: {config.generation.max_steps}")
    logger.info(f"Score type: {config.scoring.type.value}")
    logger.info(f"Resample policy: {config.ps.resample.policy.value}")
    logger.info(f"Resample method: {config.ps.resample.method.value}")
    logger.info(f"Temperature schedule: {config.ps.temperature_schedule.mode.value}")
    
    # Create verifier
    verify_fn = create_verifier(config.evaluation.use_math_verify)
    
    # Output file
    output_file = Path(config.system.output_dir) / config.evaluation.output_file
    
    # Evaluation loop
    correct = 0
    total = 0
    total_time = 0.0
    total_resamples = 0
    
    with open(output_file, "w", encoding="utf-8") as f:
        for problem in tqdm(dataset, desc="Evaluating"):
            # Format prompt
            prompt, query = sampler.format_prompt(problem.problem)
            
            # Generate response
            start_time = time.time()
            result = sampler.sample(prompt, query)
            elapsed = time.time() - start_time
            
            # Verify answer
            is_correct = verify_fn(result.response, problem.answer)
            
            if is_correct:
                correct += 1
            total += 1
            total_time += elapsed
            total_resamples += result.resample_count
            
            # Prepare result record
            pred_answer = extract_boxed_answer(result.response)
            
            record = {
                "idx": problem.idx,
                "problem": problem.problem,
                "ground_truth": problem.answer,
                "response": result.response,
                "pred_answer": pred_answer or "",
                "is_correct": is_correct,
                "particles": result.particles,
                "resample_count": result.resample_count,
                "timing": result.timing,
                "betas": result.betas,
                "ess_history": result.ess_history,
                "elapsed_time": elapsed,
            }
            
            # Save incrementally
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            
            # Log progress
            if (total) % 10 == 0:
                acc = correct / total * 100
                avg_time = total_time / total
                logger.info(
                    f"[{total}/{len(dataset)}] "
                    f"Accuracy: {acc:.2f}% ({correct}/{total}), "
                    f"Avg time: {avg_time:.2f}s"
                )
    
    # Final results
    final_acc = correct / total * 100 if total > 0 else 0.0
    avg_time = total_time / total if total > 0 else 0.0
    avg_resamples = total_resamples / total if total > 0 else 0.0
    
    results = {
        "accuracy": final_acc,
        "correct": correct,
        "total": total,
        "avg_time_per_sample": avg_time,
        "avg_resamples": avg_resamples,
        "total_time": total_time,
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("MATH-500 Evaluation Results (Persistent Sampling)")
    print("=" * 60)
    print(f"Model: {config.models.base_llm.model_path}")
    print(f"PRM enabled: {config.models.prm.enabled}")
    print(f"N particles: {config.generation.n_particles}")
    print(f"Max steps: {config.generation.max_steps}")
    print(f"Score type: {config.scoring.type.value}")
    print(f"Resample policy: {config.ps.resample.policy.value}")
    print(f"Temperature schedule: {config.ps.temperature_schedule.mode.value}")
    print("-" * 60)
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {final_acc:.2f}%")
    print(f"Avg time per sample: {avg_time:.2f}s")
    print(f"Avg resamples: {avg_resamples:.2f}")
    print("=" * 60)
    print(f"Results saved to: {output_file}")
    
    # Save summary
    summary_file = Path(config.system.output_dir) / "evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            **results,
            "config": {
                "model": config.models.base_llm.model_path,
                "prm_enabled": config.models.prm.enabled,
                "n_particles": config.generation.n_particles,
                "max_steps": config.generation.max_steps,
                "score_type": config.scoring.type.value,
                "resample_policy": config.ps.resample.policy.value,
                "temperature_schedule": config.ps.temperature_schedule.mode.value,
            }
        }, f, indent=2)
    
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Persistent Sampling Evaluation on MATH-500"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--override", "-o",
        type=str,
        action="append",
        default=[],
        help="Config overrides (e.g., 'ps.resample.method=systematic')"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Load config
    config = load_config(args.config, args.override)
    
    # Apply CLI overrides
    if args.output_dir:
        config = config.apply_overrides({"system.output_dir": args.output_dir})
    if args.num_samples:
        config = config.apply_overrides({"evaluation.num_samples": args.num_samples})
    
    # Run evaluation
    evaluate_math500(config)


if __name__ == "__main__":
    main()
