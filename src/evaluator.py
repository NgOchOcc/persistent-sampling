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
    def __init__(self, solver: PersistentSMC, aggregation: str = "majority_vote"):
        self.solver = solver
        self.aggregation = aggregation
        self.extractors = {
            'math': AnswerExtractor.generic,
            'aime': AnswerExtractor.integer
        }

    def evaluate_problem(self, problem: str, ground_truth: str,
                        problem_type: str = "math", **solve_kwargs) -> Dict:
        solutions = self.solver.solve(problem, **solve_kwargs)
        extractor = self.extractors[problem_type]
        answers_with_scores = [
            (extractor(sol.text), sol.self_certainty)
            for sol in solutions
        ]
        answers_with_scores = [(a, s) for a, s in answers_with_scores if a is not None]

        final_answer = self._aggregate(answers_with_scores)
        is_correct = self._check_correctness(final_answer, ground_truth, problem_type)
        all_answers = [a for a, _ in answers_with_scores]
        return {
            'final_answer': str(final_answer) if final_answer else None,
            'ground_truth': str(ground_truth),
            'correct': is_correct,
            'unique_answers': len(set(all_answers)),
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

            if save_path and (i + 1) % 10 == 0:
                self._save_interim(results, correct, i + 1, save_path)

        self.solver.cfg['verbose'] = original_verbose
        accuracy = correct / len(problems) if problems else 0.0
        output = {
            'dataset': dataset_name,
            'num_problems': len(problems),
            'accuracy': accuracy,
            'correct_count': correct,
            'avg_unique_answers': np.mean([r['unique_answers'] for r in results]) if results else 0.0,
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
        stats = results['solver_stats']
        if stats.get('resample_steps'):
            logger.info(f"\nResamples: {len(stats['resample_steps'])}")
            logger.info(f"Avg ESS: {np.mean(stats['ess_history']):.2f}")
            logger.info(f"Final β: {stats['beta_history'][-1]:.4f}")

        logger.info(f"{'='*60}\n")
