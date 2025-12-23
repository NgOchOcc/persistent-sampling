import logging
import argparse
from pathlib import Path

from src.persistent_smc import PersistentSMC
from src.vllm_wrapper import VLLMGenerator
from src.dataset_loaders import DatasetLoader, AnswerExtractor
from src.evaluator import MathEvaluator

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Persistent SMC Evaluation")

    parser.add_argument("--dataset", type=str, required=True, choices=["math500", "aime24", "gsm8k"])
    parser.add_argument("--data_path", type=str)

    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-7B-Instruct")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    # SMC 
    parser.add_argument("--N", type=int, default=16, help="Number of particles")
    parser.add_argument("--k_max", type=int, default=10, help="Window size")
    parser.add_argument("--tau", type=float, default=0.33, help="ESS threshold")
    parser.add_argument("--target_ess_ratio", type=float, default=0.7)
    parser.add_argument("--annealing", type=str, default="ess_targeted", choices=["linear", "power", "saturating", "ess_targeted"])
    parser.add_argument("--T_anneal", type=int, default=20)
    parser.add_argument("--transform_sc", type=str, default="centering", choices=["none", "centering", "clipping"])

    # Generation
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)

    # Output
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--output_name", type=str, help="Output filename (auto if not provided)")
    return parser.parse_args()


def load_dataset(args):
    data_path = args.data_path or f"data/{args.dataset}.json"
    if args.dataset == "math500":
        data = DatasetLoader.load_math500(data_path)
    elif args.dataset == "aime24":
        data = DatasetLoader.load_aime24(data_path)
    else:  
        data = DatasetLoader.load_gsm8k(data_path)
        if args.num_problems:
            data = data[:args.num_problems]

    return data


def format_problems(problems, dataset_type, llm):
    formatted = []

    for p in problems:
        if dataset_type == "aime24":
            system_prompt = (
                "You are a competition math solver. "
                "AIME answers are integers 000-999. "
                "End with #### followed by the answer."
            )
        elif dataset_type == "gsm8k":
            system_prompt = (
                "You are a grade school math problem solver. "
                "Solve step by step and end with #### followed by the numerical answer."
            )
        else:
            system_prompt = None

        prompt = llm.format_math_prompt(p['problem'], system_prompt)
        formatted.append({
            'problem': prompt,
            'answer': p.get('answer', AnswerExtractor.boxed(p.get('solution', ''))),
            **{k: v for k, v in p.items() if k not in ['problem', 'answer', 'solution']}
        })

    return formatted


def main():
    args = parse_args()

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

    problems = load_dataset(args)
    llm = VLLMGenerator(args.model, args.tensor_parallel_size)
    formatted = format_problems(problems, args.dataset, llm)
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
