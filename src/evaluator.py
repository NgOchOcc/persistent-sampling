import json
import numpy as np
from typing import List, Dict, Optional, Callable
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import logging

from .persistent_smc import PersistentSMC
from .dataset_loaders import AnswerExtractor

logger = logging.getLogger(__name__)


class MathEvaluator:
    def __init__(self, solver: PersistentSMC, aggregation: str = "majority_vote"):
        self.solver = solver
        self.aggregation = aggregation

        self.extractors = {
            'math': AnswerExtractor.generic,
            'aime': AnswerExtractor.integer
        }

    def evaluate_problem(self, problem: str, ground_truth: str,
                        problem_type: str = "math", **solve_kwargs) -> Dict:
        # Solve
        solutions = self.solver.solve(problem, **solve_kwargs)

        # Extract answers
        extractor = self.extractors[problem_type]
        answers_with_scores = [
            (extractor(sol.text), sol.self_certainty)
            for sol in solutions
        ]
        answers_with_scores = [(a, s) for a, s in answers_with_scores if a is not None]

        # Aggregate
        final_answer = self._aggregate(answers_with_scores)

        # Check correctness
        is_correct = self._check_correctness(final_answer, ground_truth, problem_type)

        # Compute pass@k
        all_answers = [a for a, _ in answers_with_scores]
        pass_at_k = {
            f"pass@{k}": self._check_pass_k(all_answers[:k], ground_truth, problem_type)
            for k in [1, 2, 4, 8, 16] if k <= len(all_answers)
        }

        return {
            'final_answer': str(final_answer) if final_answer else None,
            'ground_truth': str(ground_truth),
            'correct': is_correct,
            'num_solutions': len(solutions),
            'unique_answers': len(set(all_answers)),
            'all_answers': all_answers,
            'pass_at_k': pass_at_k
        }

    def evaluate_dataset(self, problems: List[Dict], dataset_name: str = "math",
                        save_path: Optional[str] = None, **solve_kwargs) -> Dict:
        """Evaluate full dataset"""
        results = []
        correct = 0

        problem_type = 'aime' if 'aime' in dataset_name.lower() else 'math'

        logger.info(f"Evaluating {len(problems)} problems from {dataset_name}")

        for i, prob in enumerate(tqdm(problems, desc=f"Evaluating {dataset_name}")):
            result = self.evaluate_problem(
                prob['problem'],
                prob.get('answer', prob.get('ground_truth', '')),
                problem_type,
                **solve_kwargs
            )

            result.update({'problem_id': i, 'problem': prob['problem'], 'dataset': dataset_name})
            results.append(result)

            if result['correct']:
                correct += 1

            # Save intermediate
            if save_path and (i + 1) % 10 == 0:
                self._save_interim(results, correct, i + 1, save_path)

        # Compute metrics
        accuracy = correct / len(problems) if problems else 0.0
        pass_at_k_metrics = self._compute_pass_k_metrics(results, problems)

        output = {
            'dataset': dataset_name,
            'num_problems': len(problems),
            'accuracy': accuracy,
            'correct_count': correct,
            'pass_at_k': pass_at_k_metrics,
            'avg_unique_answers': np.mean([r['unique_answers'] for r in results]),
            'results': results,
            'solver_stats': self.solver.get_statistics()
        }

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(output, f, indent=2)
            logger.info(f"Results saved to {save_path}")

        self._print_summary(output)

        return output

    def _aggregate(self, answers_with_scores: List[tuple]) -> Optional[str]:
        """Aggregate answers"""
        if not answers_with_scores:
            return None

        answers, scores = zip(*answers_with_scores)

        if self.aggregation == "majority_vote":
            return Counter(answers).most_common(1)[0][0]

        elif self.aggregation == "weighted_vote":
            weights = {}
            for ans, score in answers_with_scores:
                weights[ans] = weights.get(ans, 0) + score
            return max(weights.items(), key=lambda x: x[1])[0]

        else:  # first
            return answers[0]

    def _check_correctness(self, predicted: Optional[str], ground_truth: str,
                          problem_type: str) -> bool:
        """Check if answer is correct"""
        if predicted is None:
            return False

        if problem_type == 'aime':
            try:
                return int(predicted) == int(ground_truth)
            except (ValueError, TypeError):
                return False

        return AnswerExtractor.normalize(str(predicted)) == AnswerExtractor.normalize(str(ground_truth))

    def _check_pass_k(self, answers: List, ground_truth: str, problem_type: str) -> bool:
        """Check if any answer in top-k is correct"""
        return any(self._check_correctness(ans, ground_truth, problem_type) for ans in answers)

    def _compute_pass_k_metrics(self, results: List[Dict], problems: List[Dict]) -> Dict:
        """Compute pass@k across all problems"""
        metrics = {}
        for k in [1, 2, 4, 8, 16]:
            count = sum(1 for r in results if r['pass_at_k'].get(f"pass@{k}", False))
            metrics[f"pass@{k}"] = count / len(problems) if problems else 0.0
        return metrics

    def _save_interim(self, results: List[Dict], correct: int, total: int, path: str):
        """Save intermediate results"""
        with open(path, 'w') as f:
            json.dump({
                'accuracy': correct / total,
                'correct_count': correct,
                'total': total,
                'results': results
            }, f, indent=2)

    def _print_summary(self, results: Dict):
        """Print summary"""
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATION SUMMARY: {results['dataset']}")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy: {results['accuracy']:.2%} ({results['correct_count']}/{results['num_problems']})")
        logger.info(f"Avg unique answers: {results['avg_unique_answers']:.2f}")

        logger.info("\nPass@k metrics:")
        for k, v in results['pass_at_k'].items():
            logger.info(f"  {k}: {v:.2%}")

        stats = results['solver_stats']
        if stats.get('resample_steps'):
            logger.info(f"\nResamples: {len(stats['resample_steps'])}")
            logger.info(f"Avg ESS: {np.mean(stats['ess_history']):.2f}")
            logger.info(f"Final β: {stats['beta_history'][-1]:.4f}")

        logger.info(f"{'='*60}\n")
