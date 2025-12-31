import json
import logging
from typing import Optional
from tqdm import tqdm

from .sampler import ParticleSampler
from .utils import verify_answer, extract_boxed_answer
from math_verify import parse, verify
logger = logging.getLogger(__name__)


def evaluate_math500(
    sampler: ParticleSampler,
    dataset_path: str = "data/math500.json",
    num_samples: Optional[int] = None,
    output_file: str = "math500_results.jsonl",
    use_math_verify: bool = True,
) -> float:
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    if num_samples is not None:
        dataset = dataset[:min(num_samples, len(dataset))]

    correct = 0
    total = 0

    with open(output_file, "w") as f:
        for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
            problem = sample["problem"]
            ground_truth = sample["answer"]

            prompt, query = sampler.format_prompt(problem)
            result_dict = sampler.sample(prompt, query)
            response = result_dict["response"]
            particles_info = result_dict["particles"]
            resample_count = result_dict.get("resample_count", 0)
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

            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            if (idx + 1) % 10 == 0:
                acc = correct / total * 100
                logger.info(f"\n[{idx+1}/{len(dataset)}] Accuracy: {acc:.2f}% ({correct}/{total})")

    final_acc = correct / total * 100
    logger.info("\n" + "=" * 60)
    logger.info("MATH-500 Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"Total: {total}")
    logger.info(f"Correct: {correct}")
    logger.info(f"Accuracy: {final_acc:.2f}%")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_file}")

    return final_acc


def print_evaluation_summary(
    model_path: str,
    scorer_name: str,
    n_particles: int,
    max_steps: int,
    alpha: float,
    beta: float,
    rho: float,
    temperature: float,
    accuracy: float,
    output_file: str,
):
    print("\n" + "=" * 60)
    print("MATH-500 Evaluation Results")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Scorer: {scorer_name}")
    print(f"N particles: {n_particles}")
    print(f"Max steps: {max_steps}")
    print(f"Alpha: {alpha}, Beta: {beta}, Rho: {rho}")
    print(f"Temperature: {temperature}")
    print("-" * 60)
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 60)
    print(f"Results saved to: {output_file}")
