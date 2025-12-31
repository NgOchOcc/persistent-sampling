import logging
import numpy as np
from typing import List, Tuple, Optional
from collections import Counter

from src.models import Particle, Snapshot
from src.scorers import BaseScorer
from src.utils import extract_boxed_answer

logger = logging.getLogger(__name__)


class ParticleSampler:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B",
        scorer: Optional[BaseScorer] = None,
        n_particles: int = 8,
        max_steps: int = 50,
        max_tokens_per_step: int = 128,
        temperature: float = 0.7,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 2,
        device: str = "cuda:0",
        # ESS-based resample parameters
        alpha: float = 0.5,
        beta: float = 1.0,
        rho: float = 0.5,
        # LogProb options
        enable_logprobs: bool = False,
    ):
        self.n_particles = n_particles
        self.max_steps = max_steps
        self.max_tokens_per_step = max_tokens_per_step
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.scorer = scorer
        self.enable_logprobs = enable_logprobs

        from vllm import LLM, SamplingParams, TokensPrompt
        self._SamplingParams = SamplingParams
        self._TokensPrompt = TokensPrompt

        # Load LLM
        logger.info(
            f"Loading LLM: {model_path} "
            f"(device={device}, tp={tensor_parallel_size}, mem={gpu_memory_utilization})"
        )
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            device=device,
            trust_remote_code=True,
            enable_prefix_caching=False,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.eos_token_id = self.tokenizer.eos_token_id
        logger.info("LLM loaded successfully")

    def set_scorer(self, scorer: BaseScorer):
        self.scorer = scorer

    def _generate_one_step_batch(self, particles: List[Particle]) -> List[Particle]:
        alive_particles = [p for p in particles if p.alive]
        if not alive_particles:
            return particles

        params = self._SamplingParams(
            max_tokens=self.max_tokens_per_step,
            temperature=self.temperature,
            logprobs=1 if self.enable_logprobs else None,
        )

        # Generate using TokensPrompt
        prompts = [self._TokensPrompt(prompt_token_ids=p.token_ids) for p in alive_particles]
        outputs = self.llm.generate(
            prompts=prompts,
            sampling_params=params,
            use_tqdm=False
        )

        alive_idx = 0
        for p in particles:
            if not p.alive:
                continue

            output = outputs[alive_idx]
            new_token_ids = output.outputs[0].token_ids
            p.token_ids = p.token_ids + list(new_token_ids)
            p.current_step += 1
            if self.enable_logprobs and output.outputs[0].logprobs:
                step_log_prob = 0.0
                for token_logprob_dict in output.outputs[0].logprobs:
                    if token_logprob_dict:
                        for token_id, logprob_obj in token_logprob_dict.items():
                            step_log_prob += logprob_obj.logprob
                            break 
                p.log_prob += step_log_prob

            hit_eos = output.outputs[0].finish_reason == "stop"
            if hit_eos or p.current_step >= self.max_steps:
                p.alive = False

            alive_idx += 1

        return particles

    def _score_particles(self, particles: List[Particle], query: str) -> List[Particle]:
        alive_particles = [p for p in particles if p.alive]
        if not alive_particles:
            return particles

        responses = [self.tokenizer.decode(p.token_ids, skip_special_tokens=True) for p in alive_particles]
        queries = [query] * len(responses)
        scorer_kwargs = {}
        if self.enable_logprobs:
            scorer_kwargs['log_probs'] = [p.log_prob for p in alive_particles]
            scorer_kwargs['sequence_lengths'] = [len(p.token_ids) for p in alive_particles]

        base_scores = self.scorer.score_batch(queries, responses, **scorer_kwargs)
        alive_idx = 0
        raw_scores = []
        weighted_scores = []
        for p in particles:
            if not p.alive:
                continue

            base_score = base_scores[alive_idx]
            step_ratio = p.current_step / self.max_steps
            weighted_score = base_score * (step_ratio ** self.alpha)

            p.current_score = weighted_score
            p.prm_raw = base_score

            raw_scores.append(f"{base_score:.3f}")
            weighted_scores.append(f"{weighted_score:.3f}")
            alive_idx += 1

        logger.info(f"  Base scores: {raw_scores}")
        logger.info(f"  Weighted: {weighted_scores}")

        return particles

    def _compute_ess(self, particles: List[Particle]) -> Tuple[float, List[float]]:
        alive_particles = [p for p in particles if p.alive]
        if len(alive_particles) < 2:
            return float('inf'), []

        scores = np.array([p.current_score for p in alive_particles])
        s = scores - scores.max()
        w = np.exp(self.beta * s)
        w = w / w.sum()
        ess = 1.0 / np.sum(w ** 2)

        return ess, w.tolist()

    def _check_resample_condition(self, particles: List[Particle]) -> bool:
        alive_particles = [p for p in particles if p.alive]
        if len(alive_particles) < 2:
            return False

        if all(p.current_step == 1 for p in alive_particles):
            return False

        ess, _ = self._compute_ess(particles)
        n_alive = len(alive_particles)
        threshold = self.rho * n_alive
        return ess < threshold

    def _resample(
        self,
        snapshot_pool: List[Snapshot],
        n_select: int,
    ) -> Tuple[List[Particle], List[Snapshot]]:
        if not snapshot_pool:
            return [], []

        sorted_pool = sorted(snapshot_pool, key=lambda x: x.score, reverse=True)
        selected = sorted_pool[:n_select]

        logger.info(f"  Pool size: {len(snapshot_pool)}")
        logger.info(f"  Selected: {[(s.particle_id, s.step, f'{s.score:.4f}') for s in selected]}")

        new_particles = []
        new_pool = []

        for new_pid, selected_snap in enumerate(selected):
            old_pid = selected_snap.particle_id
            selected_step = selected_snap.step
            for s in snapshot_pool:
                if s.particle_id == old_pid and s.step <= selected_step:
                    new_pool.append(Snapshot(
                        particle_id=new_pid,
                        token_ids=s.token_ids.copy(),
                        step=s.step,
                        score=s.score,
                        alive=s.alive,
                        log_prob=s.log_prob
                    ))

            new_particles.append(Particle(
                particle_id=new_pid,
                token_ids=selected_snap.token_ids.copy(),
                current_step=selected_step,
                alive=selected_snap.alive,
                current_score=selected_snap.score,
                log_prob=selected_snap.log_prob
            ))

        logger.info(f"  Created {len(new_particles)} particles, pool size: {len(new_pool)}")
        return new_particles, new_pool

    def _majority_vote(self, particles: List[Particle]) -> Tuple[str, str]:
        answers = []
        responses = []
        scores = []

        for p in particles:
            full_text = self.tokenizer.decode(p.token_ids, skip_special_tokens=True)
            if "\nassistant\n" in full_text:
                text = full_text.split("\nassistant\n")[-1]
            elif "assistant\n" in full_text:
                text = full_text.split("assistant\n")[-1]
            else:
                text = full_text

            answer = extract_boxed_answer(text)
            answers.append(answer if answer else "")
            responses.append(text)
            scores.append(p.current_score)

        answer_counts = Counter([a for a in answers if a])
        if answer_counts:
            most_common = answer_counts.most_common()
            max_votes = most_common[0][1]
            best_answers = [ans for ans, count in most_common if count == max_votes]
            best_idx = -1
            best_score = float('-inf')
            for i, (ans, score) in enumerate(zip(answers, scores)):
                if ans in best_answers and score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx >= 0:
                return responses[best_idx], answers[best_idx]

        best_idx = int(np.argmax(scores))
        return responses[best_idx], answers[best_idx]

    def sample(self, prompt: str, query: str) -> dict:
        init_token_ids = self.tokenizer.encode(prompt)
        snapshot_pool: List[Snapshot] = []
        particles = []
        params = self._SamplingParams(
            n=self.n_particles,
            max_tokens=self.max_tokens_per_step,
            temperature=self.temperature,
            logprobs=1 if self.enable_logprobs else None,
        )

        outputs = self.llm.generate(
            prompts=[self._TokensPrompt(prompt_token_ids=init_token_ids)],
            sampling_params=params,
        )
        for particle_id, out in enumerate(outputs[0].outputs):
            new_token_ids = list(out.token_ids)
            full_token_ids = init_token_ids + new_token_ids
            hit_eos = out.finish_reason == "stop"
            log_prob = 0.0
            if self.enable_logprobs and out.logprobs:
                for token_logprob_dict in out.logprobs:
                    if token_logprob_dict:
                        for token_id, logprob_obj in token_logprob_dict.items():
                            log_prob += logprob_obj.logprob
                            break

            logger.info(f"  Particle {particle_id}: {len(new_token_ids)} tokens, alive={not hit_eos}")
            particles.append(Particle(
                particle_id=particle_id,
                token_ids=full_token_ids,
                current_step=1,
                alive=not hit_eos,
                current_score=0.5,
                log_prob=log_prob
            ))

        n_alive_init = sum(1 for p in particles if p.alive)
        logger.info(f"Initial: {len(particles)} particles, {n_alive_init} alive")
        particles = self._score_particles(particles, query)
        for p in particles:
            if p.alive:
                snapshot_pool.append(Snapshot(
                    particle_id=p.particle_id,
                    token_ids=p.token_ids.copy(),
                    step=p.current_step,
                    score=p.current_score,
                    alive=p.alive,
                    log_prob=p.log_prob
                ))

        iteration = 0
        max_iterations = self.max_steps * 2
        resample_count = 0
        while iteration < max_iterations:
            iteration += 1
            n_alive = sum(1 for p in particles if p.alive)
            if n_alive == 0:
                logger.info("All particles dead, stopping")
                break

            if n_alive == 1:
                logger.info("Only 1 particle alive, generating remaining...")
                alive_p = next(p for p in particles if p.alive)

                remaining_steps = self.max_steps - alive_p.current_step
                max_new_tokens = min(remaining_steps * self.max_tokens_per_step, 1024)

                params = self._SamplingParams(
                    n=1,
                    temperature=self.temperature,
                    max_tokens=max_new_tokens,
                    logprobs=1 if self.enable_logprobs else None,
                )
                outputs = self.llm.generate(
                    prompts=[self._TokensPrompt(prompt_token_ids=alive_p.token_ids)],
                    sampling_params=params,
                )
                new_tokens = list(outputs[0].outputs[0].token_ids)
                alive_p.token_ids.extend(new_tokens)
                # Update log prob
                if self.enable_logprobs and outputs[0].outputs[0].logprobs:
                    for token_logprob_dict in outputs[0].outputs[0].logprobs:
                        if token_logprob_dict:
                            for token_id, logprob_obj in token_logprob_dict.items():
                                alive_p.log_prob += logprob_obj.logprob
                                break

                alive_p.alive = False

                particles = self._score_particles(particles, query)
                break

            alive_particles = [p for p in particles if p.alive]
            steps = [p.current_step for p in alive_particles]
            scores = [p.current_score for p in alive_particles]
            ess, _ = self._compute_ess(particles)

            logger.info(
                f"Iter {iteration}: alive={n_alive}, "
                f"steps={steps}, "
                f"scores={[f'{s:.3f}' for s in scores]}, "
                f"ESS={ess:.2f}/{n_alive}, "
                f"pool={len(snapshot_pool)}"
            )

            if self._check_resample_condition(particles):
                threshold = self.rho * n_alive
                logger.info(f"\n>>> RESAMPLE triggered (ESS={ess:.2f} < {threshold:.2f})")
                resample_count += 1

                new_particles, snapshot_pool = self._resample(
                    snapshot_pool=snapshot_pool,
                    n_select=n_alive,
                )

                if new_particles:
                    dead_particles = [p for p in particles if not p.alive]
                    particles = new_particles + dead_particles
                    logger.info(f"  Resampled {len(new_particles)} particles\n")

            particles = self._generate_one_step_batch(particles)
            particles = self._score_particles(particles, query)
            for p in particles:
                if p.alive:
                    snapshot_pool.append(Snapshot(
                        particle_id=p.particle_id,
                        token_ids=p.token_ids.copy(),
                        step=p.current_step,
                        score=p.current_score,
                        alive=p.alive,
                        log_prob=p.log_prob
                    ))

        all_particles_info = []
        for p in particles:
            text = self.tokenizer.decode(p.token_ids, skip_special_tokens=True)
            answer = extract_boxed_answer(text)
            all_particles_info.append({
                "particle_id": p.particle_id,
                "step": p.current_step,
                "alive": p.alive,
                "score": p.current_score,
                "log_prob": p.log_prob,
                "answer": answer if answer else "",
                "response": text,
            })

        response, answer = self._majority_vote(particles)

        logger.info(f"Final answer: {answer}")
        logger.info(f"Total resamples: {resample_count}")

        return {
            "response": response,
            "answer": answer,
            "particles": all_particles_info,
            "resample_count": resample_count
        }

    def format_prompt(self, problem: str, instruction: str = None) -> Tuple[str, str]:
        user_content = f"{problem}\n{instruction}"
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_content}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = user_content

        return prompt, problem
