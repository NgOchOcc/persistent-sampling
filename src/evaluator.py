import json
import logging
import numpy as np

from tqdm import tqdm
from collections import Counter
from typing import List, Dict, Optional

from src.persistent_smc import PersistentSMC
from src.dataset_loaders import AnswerExtractor

logger = logging.getLogger(__name__)


class MathEvaluator:
    def __init__(self, solver: PersistentSMC, aggregation: str = "majority_vote",
                 clear_cache_after_problem: bool = True):
        self.solver = solver
        self.aggregation = aggregation
        self.clear_cache_after_problem = clear_cache_after_problem
        self.extractors = {
            'math': AnswerExtractor.generic,
            'aime': AnswerExtractor.integer
        }

    def evaluate_problem(self, problem: str, ground_truth: str,
                        problem_type: str = "math", **solve_kwargs) -> Dict:
        solutions = self.solver.solve(problem, **solve_kwargs)
        extractor = self.extractors[problem_type]

        answers_with_scores = [
            (extractor(sol.text), sol.self_certainty, sol.text)
            for sol in solutions
        ]
        valid_answers = [(a, s, t) for a, s, t in answers_with_scores if a is not None]
        answers_scores_only = [(a, s) for a, s, _ in valid_answers]
        final_answer = self._aggregate(answers_scores_only)
        is_correct = self._check_correctness(final_answer, ground_truth, problem_type)
        all_answers = [a for a, _, _ in valid_answers]
        all_solutions_text = [t for _, _, t in valid_answers]
        pass_at_k = {
            f"pass@{k}": self._check_pass_k(all_answers[:k], ground_truth, problem_type)
            for k in [1, 2, 4, 8, 16, 32] if k <= len(all_answers)
        }

        return {
            'final_answer': str(final_answer) if final_answer else None,
            'ground_truth': str(ground_truth),
            'correct': is_correct,
            'num_solutions': len(solutions),
            'unique_answers': len(set(all_answers)),
            'all_answers': all_answers,
            'all_solutions_text': all_solutions_text,
            'pass_at_k': pass_at_k
        }

    def evaluate_dataset(self, problems: List[Dict], dataset_name: str = "math",
                        save_path: Optional[str] = None, **solve_kwargs) -> Dict:
        results = []
        correct = 0
        problem_type = 'aime' if 'aime' in dataset_name.lower() else 'math'

        original_verbose = self.solver.cfg.get('verbose', False)
        self.solver.cfg['verbose'] = False

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

            if self.clear_cache_after_problem:
                if hasattr(self.solver.llm, 'clear_cache'):
                    self.solver.llm.clear_cache(aggressive=True)

            if (i + 1) % 10 == 0:
                if hasattr(self.solver.llm, 'get_memory_stats'):
                    mem_stats = self.solver.llm.get_memory_stats()
                    if mem_stats:
                        logger.info(f"Memory: {mem_stats['allocated_gb']:.2f}GB allocated, "
                                  f"{mem_stats['reserved_gb']:.2f}GB reserved")

            if save_path and (i + 1) % 10 == 0:
                self._save_interim(results, correct, i + 1, save_path)

        self.solver.cfg['verbose'] = original_verbose
        accuracy = correct / len(problems) if problems else 0.0
        pass_at_k_metrics = self._compute_pass_k_metrics(results, problems)
        output = {
            'dataset': dataset_name,
            'num_problems': len(problems),
            'accuracy': accuracy,
            'correct_count': correct,
            'pass_at_k': pass_at_k_metrics,
            'avg_unique_answers': np.mean([r['unique_answers'] for r in results]) if results else 0.0,
            'avg_num_solutions': np.mean([r['num_solutions'] for r in results]) if results else 0.0,
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

        else:
            return answers[0]

    def _check_correctness(self, predicted: Optional[str], ground_truth: str,
                          problem_type: str) -> bool:
        if predicted is None:
            return False

        if problem_type == 'aime':
            try:
                return int(predicted) == int(ground_truth)
            except (ValueError, TypeError):
                return False
        return AnswerExtractor.verify_equivalence(str(predicted), str(ground_truth))

    def _check_pass_k(self, answers: List, ground_truth: str, problem_type: str) -> bool:
        return any(self._check_correctness(ans, ground_truth, problem_type) for ans in answers)

    def _compute_pass_k_metrics(self, results: List[Dict], problems: List[Dict]) -> Dict:
        metrics = {}
        for k in [1, 2, 4, 8, 16, 32]:
            count = sum(1 for r in results if r['pass_at_k'].get(f"pass@{k}", False))
            metrics[f"pass@{k}"] = count / len(problems) if problems else 0.0
        return metrics

    def _save_interim(self, results: List[Dict], correct: int, total: int, path: str):
        with open(path, 'w') as f:
            json.dump({
                'accuracy': correct / total,
                'correct_count': correct,
                'total': total,
                'results': results
            }, f, indent=2)

    def _print_summary(self, results: Dict):
        logger.info(f"\n{'='*60}")
        logger.info(f"EVALUATION SUMMARY: {results['dataset']}")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy: {results['accuracy']:.2%} ({results['correct_count']}/{results['num_problems']})")
        logger.info(f"Avg unique answers: {results['avg_unique_answers']:.2f}")
        logger.info(f"Avg num solutions: {results['avg_num_solutions']:.2f}")

        # Print pass@k metrics
        logger.info("\nPass@k metrics:")
        for k, v in sorted(results['pass_at_k'].items(), key=lambda x: int(x[0].split('@')[1])):
            logger.info(f"  {k}: {v:.2%}")

        stats = results['solver_stats']
        if stats.get('resample_steps'):
            logger.info(f"\nResamples: {len(stats['resample_steps'])}")
            logger.info(f"Avg ESS: {np.mean(stats['ess_history']):.2f}")
            logger.info(f"Final β: {stats['beta_history'][-1]:.4f}")

        logger.info(f"{'='*60}\n")
