#!/usr/bin/env python3
"""
Run Persistent SMC evaluation on MATH500 dataset
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persistent_smc import PersistentSMC
from vllm_wrapper import vLLMGenerator
from dataset_loaders import load_math500, create_sample_math500
from evaluator import MathEvaluator
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Persistent SMC on MATH500")

    # Dataset args
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/math500.json",
        help="Path to MATH500 dataset"
    )
    parser.add_argument(
        "--num_problems",
        type=int,
        default=None,
        help="Number of problems to evaluate (default: all)"
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default=None,
        help="Filter by difficulty level (e.g., 'Level 5')"
    )
    parser.add_argument(
        "--use_sample",
        action="store_true",
        help="Use sample data for testing"
    )

    # Model args
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Math-7B-Instruct",
        help="Model name from HuggingFace"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )

    # SMC args
    parser.add_argument(
        "--N",
        type=int,
        default=16,
        help="Number of particles"
    )
    parser.add_argument(
        "--k_max",
        type=int,
        default=10,
        help="Maximum sliding window size"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.33,
        help="ESS threshold for resampling"
    )
    parser.add_argument(
        "--target_ess_ratio",
        type=float,
        default=0.7,
        help="Target ESS ratio for ESS-targeted annealing"
    )
    parser.add_argument(
        "--annealing",
        type=str,
        default="ess_targeted",
        choices=["linear", "power", "saturating", "ess_targeted"],
        help="Annealing method"
    )
    parser.add_argument(
        "--T_anneal",
        type=int,
        default=20,
        help="Annealing timescale"
    )
    parser.add_argument(
        "--transform_sc",
        type=str,
        default="centering",
        choices=["none", "centering", "clipping"],
        help="Self-certainty transformation"
    )

    # Generation args
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50,
        help="Maximum generation steps"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature"
    )

    # Output args
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="math500_results.json",
        help="Output filename"
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_dir) / args.output_name

    logger.info("=" * 60)
    logger.info("Persistent SMC Evaluation on MATH500")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Particles: {args.N}")
    logger.info(f"Window size: {args.k_max}")
    logger.info(f"Annealing: {args.annealing}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)

    # Load dataset
    if args.use_sample:
        logger.info("Using sample data for testing...")
        problems = create_sample_math500()
    else:
        problems = load_math500(
            args.data_path,
            num_samples=args.num_problems,
            difficulty_level=args.difficulty
        )

    if not problems:
        logger.error("No problems loaded. Exiting.")
        return

    # Initialize vLLM
    logger.info("Initializing vLLM...")
    llm_generator = vLLMGenerator(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size
    )

    # Initialize Persistent SMC solver
    logger.info("Initializing Persistent SMC solver...")
    solver = PersistentSMC(
        llm_generator=llm_generator,
        N=args.N,
        k_max=args.k_max,
        tau=args.tau,
        target_ess_ratio=args.target_ess_ratio,
        annealing_method=args.annealing,
        T_anneal=args.T_anneal,
        transform_sc=args.transform_sc,
        verbose=True
    )

    # Format problems with prompts
    formatted_problems = []
    for p in problems:
        problem_text = p['problem']
        formatted_prompt = llm_generator.format_math_prompt(problem_text)

        formatted_problems.append({
            'problem': formatted_prompt,
            'answer': p.get('answer', extract_boxed_answer(p.get('solution', ''))),
            'level': p.get('level', 'Unknown'),
            'type': p.get('type', 'Unknown')
        })

    # Initialize evaluator
    logger.info("Initializing evaluator...")
    evaluator = MathEvaluator(
        solver=solver,
        answer_aggregation="majority_vote"
    )

    # Run evaluation
    logger.info(f"Starting evaluation on {len(formatted_problems)} problems...")
    results = evaluator.evaluate_dataset(
        problems=formatted_problems,
        dataset_name="MATH500",
        save_path=str(output_path),
        max_steps=args.max_steps,
        temperature=args.temperature
    )

    logger.info(f"\nEvaluation complete! Results saved to {output_path}")
    logger.info(f"Final accuracy: {results['accuracy']:.2%}")


def extract_boxed_answer(text):
    """Helper to extract boxed answer"""
    import re
    pattern = r'\\boxed\{([^}]*)\}'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None


if __name__ == "__main__":
    main()
