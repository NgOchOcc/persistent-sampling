#!/usr/bin/env python3
"""
Unified evaluation script for MATH500 and AIME24
Replaces separate run_math500.py and run_aime24.py
"""

import sys
import argparse
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persistent_smc import PersistentSMC
from vllm_wrapper import vLLMGenerator
from dataset_loaders import load_math500, load_aime24, extract_boxed_answer
from evaluator import MathEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Persistent SMC Evaluation")

    # Dataset
    parser.add_argument("--dataset", type=str, required=True, choices=["math500", "aime24"],
                       help="Dataset to evaluate")
    parser.add_argument("--data_path", type=str, help="Path to dataset (auto if not provided)")
    parser.add_argument("--num_problems", type=int, help="Number of problems (default: all)")
    parser.add_argument("--difficulty", type=str, help="Filter by difficulty (MATH500 only)")
    parser.add_argument("--use_sample", action="store_true", help="Use sample data")

    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    # SMC - use **config pattern
    parser.add_argument("--N", type=int, default=16, help="Number of particles")
    parser.add_argument("--k_max", type=int, default=10, help="Window size")
    parser.add_argument("--tau", type=float, default=0.33, help="ESS threshold")
    parser.add_argument("--target_ess_ratio", type=float, default=0.7)
    parser.add_argument("--annealing", type=str, default="ess_targeted",
                       choices=["linear", "power", "saturating", "ess_targeted"])
    parser.add_argument("--T_anneal", type=int, default=20)
    parser.add_argument("--transform_sc", type=str, default="centering",
                       choices=["none", "centering", "clipping"])

    # Generation
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)

    # Output
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--output_name", type=str, help="Output filename (auto if not provided)")

    return parser.parse_args()


def load_dataset(args):
    """Load dataset based on args"""
    if args.use_sample:
        from dataset_loaders import create_sample_math500, create_sample_aime24
        return create_sample_math500() if args.dataset == "math500" else create_sample_aime24()

    data_path = args.data_path or f"data/{args.dataset}.json"

    if args.dataset == "math500":
        data = load_math500(data_path, args.num_problems, args.difficulty)
    else:
        data = load_aime24(data_path)
        if args.num_problems:
            data = data[:args.num_problems]

    return data


def format_problems(problems, dataset_type, llm):
    """Format problems with prompts"""
    formatted = []

    for p in problems:
        if dataset_type == "aime24":
            system_prompt = (
                "You are a competition math solver. "
                "AIME answers are integers 000-999. "
                "End with #### followed by the answer."
            )
        else:
            system_prompt = None

        prompt = llm.format_math_prompt(p['problem'], system_prompt)

        formatted.append({
            'problem': prompt,
            'answer': p.get('answer', extract_boxed_answer(p.get('solution', ''))),
            **{k: v for k, v in p.items() if k not in ['problem', 'answer', 'solution']}
        })

    return formatted


def main():
    args = parse_args()

    # Setup output
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"{args.dataset}_results.json"
    output_path = Path(args.output_dir) / output_name

    logger.info("=" * 60)
    logger.info(f"Persistent SMC Evaluation: {args.dataset.upper()}")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Particles: {args.N}, Window: {args.k_max}")
    logger.info(f"Annealing: {args.annealing}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)

    # Load data
    problems = load_dataset(args)
    if not problems:
        logger.error("No problems loaded. Exiting.")
        return

    # Initialize
    llm = vLLMGenerator(args.model, args.tensor_parallel_size)

    # Format problems
    formatted = format_problems(problems, args.dataset, llm)

    # Create solver with config dict
    smc_config = {
        'N': args.N,
        'k_max': args.k_max,
        'tau': args.tau,
        'target_ess_ratio': args.target_ess_ratio,
        'annealing_method': args.annealing,
        'T_anneal': args.T_anneal,
        'transform_sc': args.transform_sc,
        'verbose': True
    }
    solver = PersistentSMC(llm, **smc_config)

    # Evaluate
    evaluator = MathEvaluator(solver, aggregation="majority_vote")
    results = evaluator.evaluate_dataset(
        formatted,
        dataset_name=args.dataset.upper(),
        save_path=str(output_path),
        max_steps=args.max_steps,
        temperature=args.temperature
    )

    logger.info(f"\nDone! Accuracy: {results['accuracy']:.2%}")


if __name__ == "__main__":
    main()
