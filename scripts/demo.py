#!/usr/bin/env python3
"""
Simple demo script to test Persistent SMC implementation
Runs on a single problem to verify everything works
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from persistent_smc import PersistentSMC
from vllm_wrapper import vLLMGenerator
from dataset_loaders import create_sample_math500, extract_final_answer_math
from collections import Counter
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("Persistent SMC Demo")
    logger.info("=" * 60)

    # Get a sample problem
    problems = create_sample_math500()
    problem = problems[0]  # Simple sqrt problem

    logger.info(f"\nProblem: {problem['problem']}")
    logger.info(f"Ground truth: {problem['answer']}")

    # Initialize vLLM with small model for demo
    logger.info("\nInitializing vLLM...")
    logger.info("Note: This will download the model if not already cached")

    try:
        llm = vLLMGenerator(
            model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",  # Smaller model for demo
            tensor_parallel_size=1
        )
    except Exception as e:
        logger.error(f"Failed to initialize vLLM: {e}")
        logger.info("\nIf you don't have GPU or vLLM installed, this is expected.")
        logger.info("Install vLLM with: pip install vllm")
        return

    # Initialize Persistent SMC
    logger.info("\nInitializing Persistent SMC solver...")
    solver = PersistentSMC(
        llm_generator=llm,
        N=8,  # Small number for demo
        k_max=5,
        tau=0.33,
        annealing_method="ess_targeted",
        verbose=True
    )

    # Format prompt
    formatted_prompt = llm.format_math_prompt(problem['problem'])

    # Solve
    logger.info("\nSolving with Persistent SMC...")
    logger.info("=" * 60)

    solutions = solver.solve(
        formatted_prompt,
        max_steps=20,
        temperature=0.8
    )

    # Extract answers
    logger.info("\n" + "=" * 60)
    logger.info("Results")
    logger.info("=" * 60)

    answers = []
    for i, sol in enumerate(solutions):
        ans = extract_final_answer_math(sol.text)
        if ans:
            answers.append(ans)
            logger.info(f"\nSolution {i+1}:")
            logger.info(f"  Answer: {ans}")
            logger.info(f"  Self-certainty: {sol.self_certainty:.4f}")

    # Majority vote
    if answers:
        final_answer = Counter(answers).most_common(1)[0][0]
        logger.info(f"\n{'='*60}")
        logger.info(f"Final answer (majority vote): {final_answer}")
        logger.info(f"Ground truth: {problem['answer']}")
        logger.info(f"Correct: {final_answer == problem['answer']}")
    else:
        logger.warning("No answers extracted from solutions!")

    # Show statistics
    stats = solver.get_statistics()
    logger.info(f"\n{'='*60}")
    logger.info("Solver Statistics")
    logger.info(f"{'='*60}")
    logger.info(f"Resampling occurred at steps: {stats['resample_steps']}")
    logger.info(f"Final ESS: {stats['ess_history'][-1]:.2f}")
    logger.info(f"Final beta: {stats['beta_history'][-1]:.4f}")

    logger.info(f"\n{'='*60}")
    logger.info("Demo complete!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
