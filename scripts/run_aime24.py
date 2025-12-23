#!/usr/bin/env python3
"""
Run Persistent SMC evaluation on AIME24 dataset
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persistent_smc import PersistentSMC
from vllm_wrapper import vLLMGenerator
from dataset_loaders import load_aime24
from evaluator import MathEvaluator
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Persistent SMC on AIME24")

    # Dataset args
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/aime24.json",
        help="Path to AIME24 dataset"
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
        default=24,  # More particles for harder problems
        help="Number of particles"
    )
    parser.add_argument(
        "--k_max",
        type=int,
        default=15,  # Larger window for AIME
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
        default=30,  # Slower annealing for AIME
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
        default=80,  # More steps for AIME
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
        default="aime24_results.json",
        help="Output filename"
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_dir) / args.output_name

    logger.info("=" * 60)
    logger.info("Persistent SMC Evaluation on AIME24")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Particles: {args.N}")
    logger.info(f"Window size: {args.k_max}")
    logger.info(f"Annealing: {args.annealing}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)

    # Load dataset
    problems = load_aime24(args.data_path)

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

        # AIME-specific system prompt
        system_prompt = (
            "You are a mathematical problem solver specializing in competition mathematics. "
            "Solve the following AIME problem step by step. "
            "AIME answers are always integers from 000 to 999. "
            "End your solution with #### followed by the numerical answer."
        )

        formatted_prompt = llm_generator.format_math_prompt(
            problem_text,
            system_prompt=system_prompt
        )

        formatted_problems.append({
            'problem': formatted_prompt,
            'answer': p['answer']
        })

    # Initialize evaluator
    logger.info("Initializing evaluator...")
    evaluator = MathEvaluator(
        solver=solver,
        answer_aggregation="majority_vote"
    )

    # Run evaluation
    logger.info(f"Starting evaluation on {len(formatted_problems)} AIME problems...")
    results = evaluator.evaluate_dataset(
        problems=formatted_problems,
        dataset_name="AIME24",
        save_path=str(output_path),
        max_steps=args.max_steps,
        temperature=args.temperature
    )

    logger.info(f"\nEvaluation complete! Results saved to {output_path}")
    logger.info(f"Final accuracy: {results['accuracy']:.2%}")
    logger.info(f"Correct: {results['correct_count']}/{results['num_problems']}")


if __name__ == "__main__":
    main()
