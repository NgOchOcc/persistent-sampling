"""
Evaluation script for PRM-based Particle Sampling on MATH-500 benchmark
(vLLM direct mode - no server)

GPU Layout (dùng device parameter):
- PRM: GPU 0,1 với tp=2, device="cuda:0"
- LLM: GPU 2,3 với tp=2, device="cuda:2"
"""
import argparse
import json
import logging
from typing import Optional

from tqdm import tqdm

from particle_sampler import ParticleSampler, extract_boxed_answer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_answer(response: str, ground_truth: str, use_math_verify: bool = True):
    """Verify answer using math_verify or simple matching"""
    if use_math_verify:
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
        except Exception:
            pass
    
    # Fallback: simple string matching
    pred_answer = extract_boxed_answer(response)
    if pred_answer is None:
        return False
    return pred_answer.strip() == ground_truth.strip()


def evaluate_math500(
    model_path: str = "Qwen/Qwen2.5-7B",
    prm_model_path: str = "Qwen/Qwen2.5-Math-PRM-7B",
    n_particles: int = 8,
    max_steps: int = 20,
    max_tokens_per_step: int = 256,
    temperature: float = 0.7,
    num_samples: Optional[int] = None,
    output_file: str = "math500_results.jsonl",
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 2,
    prm_device: str = "cuda:0",  # PRM dùng GPU 0,1
    llm_device: str = "cuda:2",  # LLM dùng GPU 2,3
    # ESS params
    alpha: float = 0.5,
    beta: float = 1.0,
    rho: float = 0.5,
):
    """Chạy evaluation trên MATH-500 benchmark"""
    
    # Check math_verify
    try:
        from math_verify import parse, verify
        use_math_verify = True
        logger.info("Using math_verify for evaluation")
    except ImportError:
        use_math_verify = False
        logger.warning("math_verify not installed. Using simple string matching.")
        logger.info("Install with: pip install math-verify")
    
    # Load MATH-500 dataset from local JSON
    logger.info("Loading MATH-500 dataset...")
    math500_path = "/home/anhld48/Working/icml/sampling_tts/persistent-sampling/data/math500.json"
    with open(math500_path, "r") as f:
        dataset = json.load(f)
    
    if num_samples is not None:
        dataset = dataset[:min(num_samples, len(dataset))]
    
    logger.info(f"Total samples: {len(dataset)}")
    
    # Initialize sampler
    logger.info(f"Loading model: {model_path}")
    logger.info(f"Loading PRM: {prm_model_path}")
    logger.info(f"Config: n_particles={n_particles}, max_steps={max_steps}, "
                f"alpha={alpha}, beta={beta}, rho={rho}")
    logger.info(f"GPU config: prm_device={prm_device}, llm_device={llm_device}, tp={tensor_parallel_size}, mem={gpu_memory_utilization}")
    
    sampler = ParticleSampler(
        model_path=model_path,
        prm_model_path=prm_model_path,
        n_particles=n_particles,
        max_steps=max_steps,
        max_tokens_per_step=max_tokens_per_step,
        temperature=temperature,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        prm_device=prm_device,
        llm_device=llm_device,
        alpha=alpha,
        beta=beta,
        rho=rho,
    )
    
    correct = 0
    total = 0
    
    with open(output_file, "w") as f:
        for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
            problem = sample["problem"]
            ground_truth = sample["answer"]
            
            # Format prompt
            prompt, query = sampler.format_prompt(problem)
            
            # Generate response
            result_dict = sampler.sample(prompt, query)
            response = result_dict["response"]
            particles_info = result_dict["particles"]
            resample_count = result_dict.get("resample_count", 0)
            
            # Verify
            is_correct = verify_answer(response, ground_truth, use_math_verify)
            
            if is_correct:
                correct += 1
            total += 1
            
            pred_answer = extract_boxed_answer(response)
            
            result = {
                "idx": idx,
                "problem": problem,
                "ground_truth": ground_truth,
                "response": response,
                "pred_answer": pred_answer if pred_answer else "",
                "is_correct": is_correct,
                "particles": particles_info,
                "resample_count": resample_count,
            }
            
            # Save incrementally
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            
            if (idx + 1) % 10 == 0:
                acc = correct / total * 100
                logger.info(f"\n[{idx+1}/{len(dataset)}] Accuracy: {acc:.2f}% ({correct}/{total})")
    
    # Final results
    final_acc = correct / total * 100
    print("\n" + "=" * 60)
    print("MATH-500 Evaluation Results (vLLM Direct Mode)")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"PRM: {prm_model_path}")
    print(f"N particles: {n_particles}")
    print(f"Max steps: {max_steps}")
    print(f"Alpha: {alpha}, Beta: {beta}, Rho: {rho}")
    print(f"Temperature: {temperature}")
    print("-" * 60)
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {final_acc:.2f}%")
    print("=" * 60)
    print(f"Results saved to: {output_file}")
    
    return final_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="vLLM PRM Particle Sampling Evaluation on MATH-500"
    )
    parser.add_argument(
        "--model_path", type=str, 
        default="Qwen/Qwen2.5-7B",
        help="Model path"
    )
    parser.add_argument(
        "--prm_model_path", type=str,
        default="Qwen/Qwen2.5-Math-PRM-7B",
        help="PRM model path"
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
    
    evaluate_math500(
        model_path=args.model_path,
        prm_model_path=args.prm_model_path,
        n_particles=args.n_particles,
        max_steps=args.max_steps,
        max_tokens_per_step=args.max_tokens_per_step,
        temperature=args.temperature,
        num_samples=args.num_samples,
        output_file=args.output_file,
        gpu_memory_utilization=args.gpu_mem,
        tensor_parallel_size=args.tp,
        prm_device=args.prm_device,
        llm_device=args.llm_device,
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
    )


'''
# GPU Layout: PRM on GPU 0,1 | LLM on GPU 2,3

python eval_math500.py \
    --model_path Qwen/Qwen2.5-7B-Instruct \
    --prm_model_path Qwen/Qwen2.5-Math-PRM-7B \
    --n_particles 8 \
    --max_steps 25 \
    --max_tokens_per_step 128 \
    --temperature 0.7 \
    --output_file math500_prm_results.jsonl \
    --prm_device cuda:0 \
    --llm_device cuda:1 \
    --tp 1 \
    --gpu_mem 0.9 \
    --alpha 0.2 \
    --beta 10.0 \
    --rho 1
'''
