import argparse
import logging
from typing import Optional

from src import ParticleSampler, PRMScorer, LogProbScorer
from src.evaluator import evaluate_math500, print_evaluation_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_evaluation(
    model_path: str = "Qwen/Qwen2.5-7B",
    prm_model_path: Optional[str] = None,
    n_particles: int = 8,
    max_steps: int = 20,
    max_tokens_per_step: int = 256,
    temperature: float = 0.7,
    num_samples: Optional[int] = None,
    output_file: str = "math500_results.jsonl",
    dataset_path: str = "data/math500.json",
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 2,
    scorer_type: str = "prm",    # prm or logprob
    prm_device: str = "cuda:0",  # PRM dùng GPU 0,1
    llm_device: str = "cuda:2",  # LLM dùng GPU 2,3

    # ESS params
    alpha: float = 0.5,
    beta: float = 1.0,
    rho: float = 0.5,
    # LogProb scorer params
    logprob_normalize: bool = True,
):
    if scorer_type == "prm":
        scorer = PRMScorer(
            model_name=prm_model_path,
            device=prm_device,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        scorer_name = f"PRM ({prm_model_path})"
        enable_logprobs = False

    elif scorer_type == "logprob":
        scorer = LogProbScorer(
            normalize=logprob_normalize,
            use_negative=False,
        )
        scorer_name = f"LogProb (normalize={logprob_normalize})"
        enable_logprobs = True

    else:
        raise ValueError(f"Unknown scorer_type: {scorer_type}. Use 'prm' or 'logprob'")

    # Initialize sampler
    logger.info(f"Loading model: {model_path}")
    logger.info(f"Config: n_particles={n_particles}, max_steps={max_steps}, "
                f"alpha={alpha}, beta={beta}, rho={rho}")
    logger.info(f"GPU config: llm_device={llm_device}, tp={tensor_parallel_size}, mem={gpu_memory_utilization}")

    sampler = ParticleSampler(
        model_path=model_path,
        scorer=scorer,
        n_particles=n_particles,
        max_steps=max_steps,
        max_tokens_per_step=max_tokens_per_step,
        temperature=temperature,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        device=llm_device,
        alpha=alpha,
        beta=beta,
        rho=rho,
        enable_logprobs=enable_logprobs,
    )

    # Run evaluation
    accuracy = evaluate_math500(
        sampler=sampler,
        dataset_path=dataset_path,
        num_samples=num_samples,
        output_file=output_file,
        use_math_verify=True,
    )

    # Print summary
    print_evaluation_summary(
        model_path=model_path,
        scorer_name=scorer_name,
        n_particles=n_particles,
        max_steps=max_steps,
        alpha=alpha,
        beta=beta,
        rho=rho,
        temperature=temperature,
        accuracy=accuracy,
        output_file=output_file,
    )

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Particle Sampling Evaluation on MATH-500 with Configurable Scorers"
    )
    parser.add_argument(
        "--model_path", type=str,
        default="Qwen/Qwen2.5-7B",
        help="LLM model path"
    )
    parser.add_argument(
        "--scorer_type", type=str, default="prm",
        choices=["prm", "logprob"],
        help="Scorer type: 'prm' (PRM-based) or 'logprob' (log probability)"
    )
    parser.add_argument(
        "--prm_model_path", type=str,
        default=None,
        help="PRM model path (required if --scorer_type=prm)"
    )
    parser.add_argument(
        "--logprob_normalize", action="store_true",
        help="Normalize log probs by sequence length (for logprob scorer)"
    )
    parser.add_argument(
        "--dataset_path", type=str,
        default="/home/anhld48/Working/icml/sampling_tts/persistent-sampling/data/math500.json",
        help="Path to MATH-500 dataset"
    )
    parser.add_argument(
        "--n_particles", type=int, default=8,
        help="Number of particles"
    )
    parser.add_argument(
        "--max_steps", type=int, default=20,
        help="Maximum number of steps (T)"
    )
    parser.add_argument(
        "--max_tokens_per_step", type=int, default=256,
        help="Maximum tokens per step"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Number of samples to evaluate (None = all 500)"
    )
    parser.add_argument(
        "--output_file", type=str, default="math500_prm_results.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--gpu_mem", type=float, default=0.9,
        help="GPU memory utilization per model"
    )
    parser.add_argument(
        "--tp", type=int, default=2,
        help="Tensor parallel size (2 = use 2 GPUs per model)"
    )
    parser.add_argument(
        "--prm_device", type=str, default="cuda:0",
        help="Device for PRM (e.g., 'cuda:0' → uses GPU 0,1 with tp=2)"
    )
    parser.add_argument(
        "--llm_device", type=str, default="cuda:2",
        help="Device for LLM (e.g., 'cuda:2' → uses GPU 2,3 with tp=2)"
    )
    # ESS params
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Step weight exponent: score = PRM × (t/T)^α"
    )
    parser.add_argument(
        "--beta", type=float, default=1.0,
        help="Softmax temperature: weights = softmax(β × score)"
    )
    parser.add_argument(
        "--rho", type=float, default=0.5,
        help="ESS threshold ratio: resample if ESS < ρ × N"
    )
    
    args = parser.parse_args()
    run_evaluation(
        model_path=args.model_path,
        prm_model_path=args.prm_model_path,
        scorer_type=args.scorer_type,
        n_particles=args.n_particles,
        max_steps=args.max_steps,
        max_tokens_per_step=args.max_tokens_per_step,
        temperature=args.temperature,
        num_samples=args.num_samples,
        output_file=args.output_file,
        dataset_path=args.dataset_path,
        gpu_memory_utilization=args.gpu_mem,
        tensor_parallel_size=args.tp,
        prm_device=args.prm_device,
        llm_device=args.llm_device,
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
        logprob_normalize=args.logprob_normalize,
    )


'''
Example Usage:

# 1. Using PRM Scorer (GPU 0,1 for PRM, GPU 2,3 for LLM)
python eval_math500.py \
    --scorer_type prm \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --prm_model_path Qwen/Qwen2.5-Math-PRM-7B \
    --n_particles 8 \
    --max_steps 25 \
    --max_tokens_per_step 128 \
    --temperature 0.7 \
    --output_file math500_prm_results.jsonl \
    --prm_device cuda:0 \
    --llm_device cuda:2 \
    --tp 2 \
    --gpu_mem 0.9 \
    --alpha 0.5 \
    --beta 1.0 \
    --rho 0.5

# 2. Using LogProb Scorer (only LLM on GPU 0,1)
python eval_math500.py \
    --scorer_type logprob \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --n_particles 8 \
    --max_steps 25 \
    --max_tokens_per_step 128 \
    --temperature 0.7 \
    --output_file math500_logprob_results.jsonl \
    --llm_device cuda:0 \
    --tp 2 \
    --gpu_mem 0.9 \
    --alpha 0.5 \
    --beta 1.0 \
    --rho 0.5 \
    --logprob_normalize

# 3. Quick test with LogProb scorer (single GPU)
python eval_math500.py \
    --scorer_type logprob \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --n_particles 4 \
    --max_steps 10 \
    --num_samples 10 \
    --llm_device cuda:0 \
    --tp 1 \
    --logprob_normalize
'''
