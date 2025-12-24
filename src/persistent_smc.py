import logging
import numpy as np

from copy import deepcopy
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Particle:
    text: str
    logprobs: List[Dict[int, float]] = field(default_factory=list)
    finished: bool = False
    self_certainty: float = 0.0
    weight: float = 1.0
    timestep: int = 0

    def copy(self) -> 'Particle':
        return deepcopy(self)


class AnnealingSchedule:
    @staticmethod
    def linear(t: int, T: int) -> float:
        return min(1.0, t / T)

    @staticmethod
    def power(t: int, T: int, gamma: float = 2.0) -> float:
        return min(1.0, (t / T) ** gamma)

    @staticmethod
    def saturating(t: int, kappa: float = 0.1) -> float:
        return 1.0 - np.exp(-kappa * t)

    @staticmethod
    def ess_targeted(sc_scores: np.ndarray, prev_beta: float,
                     target_ess: float, max_iter: int = 20) -> float:
        beta_min, beta_max = prev_beta, 10.0
        for _ in range(max_iter):
            beta_mid = (beta_min + beta_max) / 2
            weights = np.exp(beta_mid * sc_scores)
            ess = 1.0 / np.sum((weights / weights.sum()) ** 2)

            beta_min, beta_max = (beta_min, beta_mid) if ess < target_ess else (beta_mid, beta_max)

        return max(prev_beta, beta_mid)


class SlidingWindow:
    def __init__(self, k_max: int = 10, adaptive: str = "constant"):
        self.k_max = k_max
        self.adaptive = adaptive
        self.window = []

    def add(self, particles: List[Particle], weights: np.ndarray, t: int, N: int):
        """Add and auto-trim window"""
        self.window.append({
            'particles': [p.copy() for p in particles],
            'weights': weights.copy(),
            't': t
        })

        k_t = self._compute_size(t, N)
        while len(self.window) > k_t:
            self.window.pop(0)

    def get_all(self) -> Tuple[List[Particle], np.ndarray]:
        if not self.window:
            return [], np.array([])

        particles = [p for entry in self.window for p in entry['particles']]
        weights = np.concatenate([entry['weights'] for entry in self.window])
        return particles, weights

    def _compute_size(self, t: int, N: int) -> int:
        sizes = {
            "constant": lambda: self.k_max,
            "time": lambda: min(self.k_max, t),
            "N": lambda: min(self.k_max, int(np.ceil(2 * np.log(max(N, 1)))), t)
        }
        return sizes.get(self.adaptive, sizes["constant"])()

    def clear(self):
        self.window = []

    def __len__(self):
        return len(self.window)


class PersistentSMC:
    def __init__(self, llm_generator, **config):
        self.llm = llm_generator
        self.cfg = {
            'N': 16,
            'k_max': 10,
            'tau': 0.33,
            'target_ess_ratio': 0.7,
            'annealing_method': 'ess_targeted',
            'T_anneal': 20,
            'transform_sc': 'centering',
            'reset_after_resample': True,
            'verbose': True
        }
        self.cfg.update(config)
        self.stats = {'ess': [], 'beta': [], 'n_alive': [], 'resamples': []}

    def solve(self, prompt: str, max_steps: int = 50, temperature: float = 0.8,
              max_tokens: int = 100) -> List[Particle]:
        particles = self._initialize(prompt, temperature, max_tokens)
        window = SlidingWindow(self.cfg['k_max'])
        results = []
        t, beta = 1, 0.0

        # Main loop
        while particles and t < max_steps:
            sc_scores = self._compute_sc(particles)
            sc_scores = self._transform_sc(sc_scores)
            beta = self._compute_beta(t, beta, sc_scores, len(particles))
            weights = np.exp(beta * sc_scores)

            window.add(particles, weights, t, len(particles))
            all_particles, all_weights = window.get_all()
            ess = self._compute_ess(all_weights)
            self._log_step(t, len(particles), ess, beta, sc_scores)

            if ess < self.cfg['tau'] * len(particles):
                particles = self._resample(all_particles, all_weights, len(particles))
                if self.cfg['reset_after_resample']:
                    window.clear()
                self.stats['resamples'].append(t)

            particles = self._generate_next(particles, temperature, max_tokens)
            still_alive = []
            for p in particles:
                p.timestep = t
                (results if p.finished else still_alive).append(p)
            particles = still_alive
            t += 1

        for p in particles:
            p.finished = True
            results.append(p)

        if self.cfg['verbose']:
            logger.info(f"\nCompleted: {len(results)} solutions, {len(self.stats['resamples'])} resamples")

        return results

    def _initialize(self, prompt: str, temp: float, max_tokens: int = 100) -> List[Particle]:
        if self.cfg['verbose']:
            logger.info(f"Initializing {self.cfg['N']} particles...")
        return self.llm.generate_batch([prompt] * self.cfg['N'], temperature=temp, max_tokens=max_tokens)

    def _compute_sc(self, particles: List[Particle]) -> np.ndarray:
        scores = []
        for p in particles:
            if not p.logprobs:
                scores.append(0.0)
                continue

            # avg_logprob = np.mean([max(d.values()) if d else -10.0 for d in p.logprobs])
            avg_logprob = np.mean([
                max(lp.logprob for lp in d.values()) if d else -10.0
                for d in p.logprobs
            ])
            sc = -avg_logprob / np.log(self.llm.vocab_size)
            scores.append(sc)
            p.self_certainty = sc

        return np.array(scores)

    def _transform_sc(self, scores: np.ndarray) -> np.ndarray:
        transforms = {
            'centering': lambda s: s - s.mean(),
            'clipping': lambda s: np.clip(s, -5, 5),
            'none': lambda s: s
        }
        return transforms.get(self.cfg['transform_sc'], transforms['none'])(scores)

    def _compute_beta(self, t: int, prev_beta: float, sc_scores: np.ndarray, N: int) -> float:
        method = self.cfg['annealing_method']

        if method == 'ess_targeted':
            target_ess = self.cfg['target_ess_ratio'] * N
            return AnnealingSchedule.ess_targeted(sc_scores, prev_beta, target_ess)

        schedules = {
            'linear': lambda: AnnealingSchedule.linear(t, self.cfg['T_anneal']),
            'power': lambda: AnnealingSchedule.power(t, self.cfg['T_anneal']),
            'saturating': lambda: AnnealingSchedule.saturating(t)
        }

        return max(prev_beta, schedules.get(method, schedules['linear'])())

    def _compute_ess(self, weights: np.ndarray) -> float:
        if len(weights) == 0:
            return 0.0
        w_norm = weights / weights.sum()
        return 1.0 / (w_norm ** 2).sum()

    def _resample(self, particles: List[Particle], weights: np.ndarray, N: int) -> List[Particle]:
        if not particles:
            return []

        w_norm = weights / weights.sum()
        positions = (np.arange(N) + np.random.uniform()) / N
        cumsum = np.cumsum(w_norm)
        resampled = []
        i = j = 0
        while i < N and j < len(particles):
            if positions[i] < cumsum[j]:
                resampled.append(particles[j].copy())
                i += 1
            else:
                j += 1

        # Handle edge case
        while len(resampled) < N:
            resampled.append(particles[-1].copy())

        return resampled

    def _generate_next(self, particles: List[Particle], temp: float, max_tokens: int = 100) -> List[Particle]:
        prompts = [p.text for p in particles]
        return self.llm.generate_batch(prompts, temperature=temp, max_tokens=max_tokens)

    def _log_step(self, t: int, N: int, ess: float, beta: float, sc: np.ndarray):
        self.stats['ess'].append(ess)
        self.stats['beta'].append(beta)
        self.stats['n_alive'].append(N)

        if self.cfg['verbose']:
            logger.info(f"Step {t}: N={N}, ESS={ess:.1f} ({ess/N:.0%}), β={beta:.3f}, "
                       f"SC={sc.mean():.2f}±{sc.std():.2f}")

    def get_statistics(self) -> Dict:
        return {
            'ess_history': self.stats['ess'],
            'beta_history': self.stats['beta'],
            'n_alive_history': self.stats['n_alive'],
            'resample_steps': self.stats['resamples']
        }
