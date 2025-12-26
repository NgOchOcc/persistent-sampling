import sys
import argparse
from pathlib import Path


from src.persistent_smc import PersistentSMC
from src.vllm_wrapper import VLLMGenerator
from src.dataset_loaders import DatasetLoader, AnswerExtractor
from src.evaluator import MathEvaluator
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
    )

    # Model args
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Model name from HuggingFace"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for generation (smaller = less memory, default=4)"
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
        default=5,
        help="Maximum sliding window size"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.5,
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
        default=3,
        help="Annealing timescale"
    )
    parser.add_argument(
        "--transform_sc",
        type=str,
        default="clipping",
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
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum tokens per generation step (increased from 512 for better reasoning)"
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

    problems = DatasetLoader.load_math500(args.data_path)
    llm_generator = VLLMGenerator(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size
    )

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

    formatted_problems = []
    for p in problems:
        problem_text = p['problem']
        formatted_prompt = llm_generator.format_math_prompt(problem_text)
        formatted_problems.append({
            'problem': formatted_prompt,
            'answer': p.get('answer', AnswerExtractor.boxed(p.get('solution', ''))),
            'level': p.get('level', 'Unknown'),
            'type': p.get('type', 'Unknown')
        })

    evaluator = MathEvaluator(
        solver=solver,
        aggregation="majority_vote"
    )

    results = evaluator.evaluate_dataset(
        problems=formatted_problems,
        dataset_name="MATH500",
        save_path=str(output_path),
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    logger.info(f"Final accuracy: {results['accuracy']:.2%}")

if __name__ == "__main__":
    main()
