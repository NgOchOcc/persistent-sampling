"""
PRM-based Particle Sampling Algorithm (vLLM direct mode)

GPU Allocation (dùng device parameter):
- PRM: GPU 0,1 với tp=2, device="cuda:0"
- LLM: GPU 2,3 với tp=2, device="cuda:2"

Thuat toan:
1. Khoi tao N particles, sinh step dau tien
2. Vong lap chinh:
   - Sinh step tiep theo cho moi particle alive
   - Score bang PRM (vLLM task="reward")
   - Luu snapshot vao pool
   - Check dieu kien resample: ESS < rho * N
   - Neu resample: chon top N tu toan bo pool
   - Neu EOS: particle chet
3. Ket qua: majority voting
"""

import os
import re
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class Particle:
    """Particle dai dien cho mot trajectory"""
    particle_id: int
    token_ids: List[int]
    current_step: int = 0
    alive: bool = True
    current_score: float = 0.5
    prm_raw: float = 0.5
    
    def copy(self, new_particle_id: int = None):
        return Particle(
            particle_id=new_particle_id if new_particle_id is not None else self.particle_id,
            token_ids=self.token_ids.copy(),
            current_step=self.current_step,
            alive=self.alive,
            current_score=self.current_score,
            prm_raw=self.prm_raw
        )


@dataclass 
class Snapshot:
    """Snapshot cua particle tai mot thoi diem"""
    particle_id: int
    token_ids: List[int]
    step: int
    score: float
    alive: bool = True
    
    def to_particle(self, new_particle_id: int):
        return Particle(
            particle_id=new_particle_id,
            token_ids=self.token_ids.copy(),
            current_step=self.step,
            alive=self.alive,
            current_score=self.score
        )


class PRMScorerVLLM:
    """
    PRM Scorer su dung vLLM with task="reward"
    
    Output format: [[prob_wrong, prob_correct]] 
    Score = prob_correct (gia tri thu 2)
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        tensor_parallel_size: int = 2,
        gpu_memory_utilization: float = 0.9,
    ):
        self.model_name = model_name
        self.step_separator = "<extra_0>"
        self.system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        
        from vllm import LLM
        
        logger.info(f"Loading PRM with vLLM: {model_name} (device={device}, tp={tensor_parallel_size}, mem={gpu_memory_utilization})")
        self.llm = LLM(
            model=model_name,
            task="reward",
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            device=device,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        logger.info("PRM loaded successfully")
    
    def _format_for_prm(self, query: str, response_text: str) -> str:
        """Format input cho PRM"""
        sep = self.step_separator
        
        steps = [s.strip() for s in re.split(r'\n\n+', response_text) if s.strip()]
        if not steps:
            steps = [response_text.strip() or "..."]
        
        formatted_response = sep.join(steps) + sep
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
            {"role": "assistant", "content": formatted_response},
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    
    def score_batch(self, queries: List[str], responses: List[str]) -> List[float]:
        """Batch score"""
        if not responses:
            return []
        
        conversations = []
        for query, response in zip(queries, responses):
            conv = self._format_for_prm(query, response)
            conversations.append(conv)
        
        outputs = self.llm.encode(conversations)
        
        all_scores = []
        for output in outputs:
            try:
                data = output.outputs.data
                # data có thể là list hoặc tensor
                if data is not None and len(data) > 0:
                    step_scores = []
                    for score in data:
                        # score có thể là list/tuple [prob_wrong, prob_correct] hoặc tensor
                        if hasattr(score, '__len__') and len(score) > 1:
                            # Lấy prob_correct (index 1)
                            val = score[1]
                            step_scores.append(float(val) if hasattr(val, 'item') else float(val))
                        elif hasattr(score, 'item'):
                            step_scores.append(float(score.item()))
                        else:
                            step_scores.append(float(score))
                    
                    if step_scores:
                        all_scores.append(float(np.mean(step_scores)))
                    else:
                        all_scores.append(0.5)
                else:
                    all_scores.append(0.5)
            except Exception as e:
                logger.warning(f"Error scoring: {e}")
                all_scores.append(0.5)
        
        return all_scores


def extract_boxed_answer(text: str) -> Optional[str]:
    """Trich xuat cau tra loi tu boxed"""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


class ParticleSampler:
    """
    PRM-based Particle Sampler (vLLM direct mode)
    
    GPU Layout (dùng device parameter):
    - PRM: GPU 0,1 với tp=2, device="cuda:0"
    - LLM: GPU 2,3 với tp=2, device="cuda:2"
    """
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-7B",
        prm_model_path: str = "Qwen/Qwen2.5-Math-PRM-7B",
        n_particles: int = 8,
        max_steps: int = 50,
        max_tokens_per_step: int = 128,
        temperature: float = 0.7,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 2,
        prm_device: str = "cuda:0",  # PRM dùng GPU 0,1
        llm_device: str = "cuda:2",  # LLM dùng GPU 2,3
        # ESS-based resample parameters
        alpha: float = 0.5,
        beta: float = 1.0,
        rho: float = 0.5,
    ):
        self.n_particles = n_particles
        self.max_steps = max_steps
        self.max_tokens_per_step = max_tokens_per_step
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        
        from vllm import LLM, SamplingParams, TokensPrompt
        self._SamplingParams = SamplingParams
        self._TokensPrompt = TokensPrompt
        
        # Load PRM on GPU 0,1 (tp=2, device="cuda:0")
        logger.info(f"Loading PRM: {prm_model_path} (device={prm_device}, tp={tensor_parallel_size}, mem={gpu_memory_utilization})")
        self.prm = PRMScorerVLLM(
            model_name=prm_model_path,
            device=prm_device,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        
        # Load LLM on GPU 2,3 (tp=2, device="cuda:2")
        logger.info(f"Loading LLM: {model_path} (device={llm_device}, tp={tensor_parallel_size}, mem={gpu_memory_utilization})")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            device=llm_device,
            trust_remote_code=True,
            enable_prefix_caching=False,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.eos_token_id = self.tokenizer.eos_token_id
        logger.info("LLM loaded successfully")
    
    def _generate_one_step_batch(self, particles: List[Particle]) -> List[Particle]:
        """Generate mot step moi cho tat ca alive particles"""
        alive_particles = [p for p in particles if p.alive]
        if not alive_particles:
            return particles
        
        params = self._SamplingParams(
            max_tokens=self.max_tokens_per_step,
            temperature=self.temperature,
        )
        
        # Dùng TokensPrompt thay vì prompt_token_ids (deprecated)
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
            
            hit_eos = output.outputs[0].finish_reason == "stop"
            
            if hit_eos or p.current_step >= self.max_steps:
                p.alive = False
            
            alive_idx += 1
        
        return particles
    
    def _score_particles(self, particles: List[Particle], query: str) -> List[Particle]:
        """Score particles: score = PRM × (t/T)^α"""
        alive_particles = [p for p in particles if p.alive]
        if not alive_particles:
            return particles
        
        responses = [self.tokenizer.decode(p.token_ids, skip_special_tokens=True) for p in alive_particles]
        queries = [query] * len(responses)
        prm_scores = self.prm.score_batch(queries, responses)
        
        alive_idx = 0
        prm_raw_list = []
        weighted_list = []
        for p in particles:
            if not p.alive:
                continue
            prm_score = prm_scores[alive_idx]
            step_ratio = p.current_step / self.max_steps
            weighted_score = prm_score * (step_ratio ** self.alpha)
            p.current_score = weighted_score
            p.prm_raw = prm_score
            prm_raw_list.append(f"{prm_score:.3f}")
            weighted_list.append(f"{weighted_score:.3f}")
            alive_idx += 1
        
        logger.info(f"  PRM raw: {prm_raw_list}")
        logger.info(f"  Weighted: {weighted_list}")
        
        return particles
    
    def _compute_ess(self, particles: List[Particle]) -> Tuple[float, List[float]]:
        """Compute ESS using beta-softmax weights"""
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
        """Resample if ESS < ρ × N"""
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
        """Resample: Select top N from pool"""
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
                        alive=s.alive
                    ))
            
            new_particles.append(Particle(
                particle_id=new_pid,
                token_ids=selected_snap.token_ids.copy(),
                current_step=selected_step,
                alive=selected_snap.alive,
                current_score=selected_snap.score
            ))
        
        logger.info(f"  Created {len(new_particles)} particles, pool size: {len(new_pool)}")
        
        return new_particles, new_pool
    
    def _majority_vote(self, particles: List[Particle]) -> Tuple[str, str]:
        """Majority voting"""
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
        """Main sampling function"""
        init_token_ids = self.tokenizer.encode(prompt)
        
        snapshot_pool: List[Snapshot] = []
        particles = []
        
        params = self._SamplingParams(
            n=self.n_particles,
            max_tokens=self.max_tokens_per_step,
            temperature=self.temperature,
        )
        
        outputs = self.llm.generate(
            prompts=[self._TokensPrompt(prompt_token_ids=init_token_ids)],
            sampling_params=params,
        )
        
        for particle_id, out in enumerate(outputs[0].outputs):
            new_token_ids = list(out.token_ids)
            full_token_ids = init_token_ids + new_token_ids
            hit_eos = out.finish_reason == "stop"
            logger.info(f"  Particle {particle_id}: {len(new_token_ids)} tokens, alive={not hit_eos}")
            particles.append(Particle(
                particle_id=particle_id,
                token_ids=full_token_ids,
                current_step=1,
                alive=not hit_eos,
                current_score=0.5
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
                    alive=p.alive
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
                )
                outputs = self.llm.generate(
                    prompts=[self._TokensPrompt(prompt_token_ids=alive_p.token_ids)],
                    sampling_params=params,
                )
                new_tokens = list(outputs[0].outputs[0].token_ids)
                alive_p.token_ids.extend(new_tokens)
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
                        alive=p.alive
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
        """Format prompt"""
        if instruction is None:
            instruction = "Please reason step by step, and put your final answer within \\boxed{}."
        
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # PRM: GPU 0,1 (device="cuda:0", tp=2)
    # LLM: GPU 2,3 (device="cuda:2", tp=2)
    sampler = ParticleSampler(
        n_particles=8,
        max_steps=20,
        prm_device="cuda:0",
        llm_device="cuda:2",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
    )
    
    problem = "What is 123 + 456?"
    prompt, query = sampler.format_prompt(problem)
    
    result = sampler.sample(prompt, query)
    print("Result:", result)
