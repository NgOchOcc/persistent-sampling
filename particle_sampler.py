"""
PRM-based Particle Sampling Algorithm

Thuat toan:
1. Khoi tao N particles, sinh step dau tien
2. Vong lap chinh:
   - Sinh step tiep theo cho moi particle alive (particles co the o steps khac nhau)
   - Score bang PRM (score ca doan tu step 1 den step hien tai)
   - Luu snapshot vao pool (luu tat ca, khong gioi han)
   - Check dieu kien resample: max_score - min_score < threshold (chi xet N particles hien tai)
   - Neu resample: chon top N tu toan bo pool, co the backtrack ve step cu
   - Xoa cac snapshots sau step duoc chon (vi da backtrack)
   - Neu EOS: particle chet
3. Ket qua: majority voting
"""

import re
import logging
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from vllm import LLM, SamplingParams
from transformers import AutoModel, AutoTokenizer
from collections import Counter
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Particle:
    """Particle dai dien cho mot trajectory"""
    particle_id: int  # ID de track particle
    token_ids: List[int]  # Token IDs tu prompt den hien tai
    current_step: int = 0
    alive: bool = True
    current_score: float = 0.5  # Weighted score: PRM × (t/T)^α
    prm_raw: float = 0.5  # Raw PRM score (khong nhan step weight)
    
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
    """Snapshot cua particle tai mot thoi diem - dung cho resample"""
    particle_id: int  # ID cua particle goc
    token_ids: List[int]  # Token IDs tai thoi diem nay
    step: int
    score: float  # PRM score tai thoi diem nay (ca doan tu step 1 den step nay)
    alive: bool = True  # FIX: Track alive state
    
    def to_particle(self, new_particle_id: int):
        return Particle(
            particle_id=new_particle_id,
            token_ids=self.token_ids.copy(),
            current_step=self.step,
            alive=self.alive,  # FIX: Preserve alive state, don't force True
            current_score=self.score
        )


class PRMScorer:
    """PRM Scorer su dung Qwen2.5-Math-PRM"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None
        self._step_sep_id = None
        self.step_separator = "<extra_0>"
        self.system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    
    def _load_model(self):
        from transformers import AutoModel, AutoTokenizer
        
        logger.info(f"Loading PRM: {self.model_name}")
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        self._model = AutoModel.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
        ).eval()
        
        self._step_sep_id = self._tokenizer.encode(self.step_separator)[0]
        logger.info(f"PRM loaded on {self.device}")
    
    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    def _format_for_prm(self, query: str, response_text: str) -> str:
        """Format input cho PRM"""
        sep = self.step_separator
        
        # Split by common delimiters va join voi separator
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
    
    @staticmethod
    def _make_step_rewards(logits, token_masks):
        """Extract step-level rewards tu PRM logits"""
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)
        
        all_scores = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
            all_scores.append(positive_probs.cpu().tolist())
        return all_scores
    
    @torch.no_grad()
    def score_single(self, query: str, response: str) -> float:
        """Score mot response, tra ve aggregated score"""
        conversation_str = self._format_for_prm(query, response)
        input_ids = self.tokenizer.encode(
            conversation_str,
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model(input_ids=input_ids)
        token_masks = (input_ids == self._step_sep_id)
        step_rewards = self._make_step_rewards(outputs[0], token_masks)
        
        if step_rewards and step_rewards[0]:
            return float(np.mean(step_rewards[0]))
        return 0.5
    
    @torch.no_grad()
    def score_batch(self, queries: List[str], responses: List[str]) -> List[float]:
        """Batch score nhieu responses"""
        if not responses:
            return []
        
        conversations = []
        for query, response in zip(queries, responses):
            conv = self._format_for_prm(query, response)
            conversations.append(conv)
        
        encodings = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.model.device)
        
        outputs = self.model(**encodings)
        
        all_scores = []
        for i in range(len(responses)):
            input_ids = encodings['input_ids'][i]
            token_mask = (input_ids == self._step_sep_id).unsqueeze(0)
            logits = outputs[0][i:i+1]
            
            step_rewards = self._make_step_rewards(logits, token_mask)
            if step_rewards and step_rewards[0]:
                score = float(np.mean(step_rewards[0]))
            else:
                score = 0.5
            all_scores.append(score)
        
        return all_scores
    
    def clear_cache(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def extract_boxed_answer(text: str) -> Optional[str]:
    """Trich xuat cau tra loi tu boxed"""
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


class ParticleSampler:
    """
    PRM-based Particle Sampler
    
    Thuat toan:
    1. Khoi tao N particles
    2. Moi iteration: sinh them 1 step cho moi particle alive, score bang PRM
    3. Luu snapshot vao pool (tat ca snapshots tu dau)
    4. Khi max_score - min_score < threshold (cua N particles hien tai): resample
    5. Resample: chon top N tu toan bo pool, co the backtrack ve step cu
    6. Xoa cac snapshots sau step duoc chon (vi da backtrack)
    7. Particle gap EOS -> chet
    8. Final: majority vote
    """
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-Math-7B-Instruct",
        prm_model_path: str = "Qwen/Qwen2.5-Math-PRM-7B",
        n_particles: int = 8,
        max_steps: int = 50,
        max_tokens_per_step: int = 128,
        temperature: float = 0.7,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.85,
        prm_device: str = "cuda:0",
        llm_gpu_ids: str = None,
        # ESS-based resample parameters
        alpha: float = 0.5,  # step weight exponent: score = PRM × (t/T)^α
        beta: float = 1.0,   # softmax temperature: weights = softmax(β × score)
        rho: float = 0.5,    # ESS threshold ratio: resample if ESS < ρ × N
    ):
        self.n_particles = n_particles
        self.max_steps = max_steps
        self.max_tokens_per_step = max_tokens_per_step
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        
        # Load PRM first on specified device
        logger.info(f"Loading PRM: {prm_model_path} on {prm_device}")
        self.prm = PRMScorer(model_name=prm_model_path, device=prm_device)
        # Force load PRM model now
        _ = self.prm.model
        
        # Set CUDA_VISIBLE_DEVICES for vLLM if specified
        import os
        if llm_gpu_ids:
            logger.info(f"Setting CUDA_VISIBLE_DEVICES={llm_gpu_ids} for LLM")
            os.environ["CUDA_VISIBLE_DEVICES"] = llm_gpu_ids
        
        # Load LLM (vLLM will use available GPUs after CUDA_VISIBLE_DEVICES is set)
        logger.info(f"Loading LLM: {model_path} with tensor_parallel_size={tensor_parallel_size}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enable_prefix_caching=False,  # Disable to prevent cross-sample contamination
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.eos_token_id = self.tokenizer.eos_token_id
        
        # Step delimiter - dung de split steps
        self.step_delimiter = "\n\n"
    
    def _generate_one_step_batch(self, particles: List[Particle]) -> List[Particle]:
        """Generate mot step moi cho tat ca alive particles (batch)"""
        alive_particles = [p for p in particles if p.alive]
        if not alive_particles:
            return particles
        
        params = SamplingParams(
            max_tokens=self.max_tokens_per_step,
            temperature=self.temperature,
        )
        
        # DEBUG: Log input for first particle to file
        if alive_particles:
            p0 = alive_particles[0]
            input_text = self.tokenizer.decode(p0.token_ids, skip_special_tokens=False)
            debug_file = "full_debug_trace.txt"
            with open(debug_file, "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Generate Step {p0.current_step} -> {p0.current_step + 1}\n")
                f.write(f"Input token count: {len(p0.token_ids)}\n")
                f.write(f"{'='*60}\n")
                f.write(f"{input_text}\n")
                f.write(f"{'='*60}\n\n")
        
        # Generate using token_ids
        outputs = self.llm.generate(
            prompt_token_ids=[p.token_ids for p in alive_particles],
            sampling_params=params,
            use_tqdm=False
        )
        
        # Update particles
        alive_idx = 0
        for p in particles:
            if not p.alive:
                continue
            
            output = outputs[alive_idx]
            new_token_ids = output.outputs[0].token_ids
            new_text = output.outputs[0].text
            
            # Append new tokens
            p.token_ids = p.token_ids + list(new_token_ids)
            p.current_step += 1
            
            # Check EOS - particle chet khi gap EOS token
            hit_eos = output.outputs[0].finish_reason == "stop"
            
            # Particle dead khi: EOS hoac dat max_steps
            if hit_eos or p.current_step >= self.max_steps:
                p.alive = False
            
            alive_idx += 1
        
        return particles
    
    def _score_particles(self, particles: List[Particle], query: str) -> List[Particle]:
        """Score particles: score = PRM × (t/T)^α"""
        alive_particles = [p for p in particles if p.alive]
        if not alive_particles:
            return particles
        
        # Decode token_ids to text for PRM scoring
        responses = [self.tokenizer.decode(p.token_ids, skip_special_tokens=True) for p in alive_particles]
        queries = [query] * len(responses)
        prm_scores = self.prm.score_batch(queries, responses)
        
        # score = PRM × (step / max_step)^α
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
            p.prm_raw = prm_score  # Store raw PRM score
            prm_raw_list.append(f"{prm_score:.3f}")
            weighted_list.append(f"{weighted_score:.3f}")
            alive_idx += 1
        
        logger.info(f"  PRM raw: {prm_raw_list}")
        logger.info(f"  Weighted: {weighted_list}")
        
        return particles
    
    def _compute_ess(self, particles: List[Particle]) -> Tuple[float, List[float]]:
        """
        Compute ESS using beta-softmax weights
        Returns: (ESS, normalized_weights)
        """
        alive_particles = [p for p in particles if p.alive]
        if len(alive_particles) < 2:
            return float('inf'), []
        
        scores = np.array([p.current_score for p in alive_particles])
        
        # Beta-softmax: w = exp(β × (s - max(s)))
        s = scores - scores.max()
        w = np.exp(self.beta * s)
        w = w / w.sum()  # Normalize
        
        # ESS = 1 / sum(w²)
        ess = 1.0 / np.sum(w ** 2)
        
        return ess, w.tolist()
    
    def _check_resample_condition(self, particles: List[Particle]) -> bool:
        """
        Resample condition based on ESS:
        - Skip step 1
        - Resample if ESS < ρ × N
        """
        alive_particles = [p for p in particles if p.alive]
        if len(alive_particles) < 2:
            return False
        
        # Skip step 1
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
        """
        Resample: Select top N from entire pool (can backtrack to earlier steps)
        """
        if not snapshot_pool:
            return [], []
        
        # Sort by score va chon top N
        sorted_pool = sorted(snapshot_pool, key=lambda x: x.score, reverse=True)
        selected = sorted_pool[:n_select]
        
        logger.info(f"  Pool size: {len(snapshot_pool)}")
        logger.info(f"  Selected: {[(s.particle_id, s.step, f'{s.score:.4f}') for s in selected]}")
        
        new_particles = []
        new_pool = []
        
        for new_pid, selected_snap in enumerate(selected):
            old_pid = selected_snap.particle_id
            selected_step = selected_snap.step
            
            # Copy all snapshots from step 1 to selected_step
            for s in snapshot_pool:
                if s.particle_id == old_pid and s.step <= selected_step:
                    new_pool.append(Snapshot(
                        particle_id=new_pid,
                        token_ids=s.token_ids.copy(),
                        step=s.step,
                        score=s.score,
                        alive=s.alive  # FIX: Preserve alive state
                    ))
            
            new_particles.append(Particle(
                particle_id=new_pid,
                token_ids=selected_snap.token_ids.copy(),
                current_step=selected_step,
                alive=selected_snap.alive,  # FIX: Use snapshot's alive state
                current_score=selected_snap.score
            ))
        
        logger.info(f"  Created {len(new_particles)} particles, pool size: {len(new_pool)}")
        
        return new_particles, new_pool
    
    def _majority_vote(self, particles: List[Particle]) -> Tuple[str, str]:
        """Majority voting de chon answer cuoi cung"""
        answers = []
        responses = []
        scores = []
        
        for p in particles:
            # Decode token_ids to text
            full_text = self.tokenizer.decode(p.token_ids, skip_special_tokens=True)
            
            # Extract only assistant response (strip system/user prefix)
            # Find last occurrence of "assistant" to get the actual response
            if "\nassistant\n" in full_text:
                text = full_text.split("\nassistant\n")[-1]
            elif "assistant\n" in full_text:
                text = full_text.split("assistant\n")[-1]
            else:
                text = full_text  # Fallback
            
            answer = extract_boxed_answer(text)
            answers.append(answer if answer else "")
            responses.append(text)  # Store only generated text
            scores.append(p.current_score)
        
        # Dem votes
        answer_counts = Counter([a for a in answers if a])
        
        if answer_counts:
            most_common = answer_counts.most_common()
            max_votes = most_common[0][1]
            best_answers = [ans for ans, count in most_common if count == max_votes]
            
            # Neu co tie, chon particle co score cao nhat
            best_idx = -1
            best_score = float('-inf')
            for i, (ans, score) in enumerate(zip(answers, scores)):
                if ans in best_answers and score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx >= 0:
                return responses[best_idx], answers[best_idx]
        
        # Fallback: chon particle co score cao nhat
        best_idx = int(np.argmax(scores))
        return responses[best_idx], answers[best_idx]
    
    def sample(self, prompt: str, query: str) -> str:
        """
        Main sampling function
        
        Args:
            prompt: Full prompt da format voi chat template
            query: Original query (dung cho PRM scoring)
        
        Returns:
            Generated response
        """
        # Comprehensive debug logging to file
        debug_file = "full_debug_trace.txt"
        with open(debug_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"NEW SAMPLE - {datetime.now()}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Query: {query}\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Formatted Prompt:\n{prompt}\n")
            f.write(f"{'='*80}\n\n")
        
        # Encode prompt to token_ids
        prompt_token_ids = self.tokenizer.encode(prompt)
        self._prompt_length = len(prompt_token_ids)  # Save for stripping later
        
        with open(debug_file, "a") as f:
            f.write(f"Prompt token count: {len(prompt_token_ids)}\n")
            f.write(f"{'-'*80}\n\n")
        
        # Snapshot pool - luu TAT CA snapshots de resample
        snapshot_pool: List[Snapshot] = []
        
        # Initialize N particles - generate step dau tien
        particles = []
        
        params = SamplingParams(
            n=self.n_particles,
            max_tokens=self.max_tokens_per_step,
            temperature=self.temperature,
        )
        
        outputs = self.llm.generate(
            prompt_token_ids=[prompt_token_ids],
            sampling_params=params,
        )
        
        for particle_id, out in enumerate(outputs[0].outputs):
            new_token_ids = list(out.token_ids)
            full_token_ids = prompt_token_ids + new_token_ids
            hit_eos = out.finish_reason == "stop"
            logger.info(f"  Particle {particle_id}: {len(new_token_ids)} tokens, finish={out.finish_reason}, alive={not hit_eos}")
            particles.append(Particle(
                particle_id=particle_id,
                token_ids=full_token_ids,
                current_step=1,
                alive=not hit_eos,
                current_score=0.5
            ))
        
        # Log initial particles
        n_alive_init = sum(1 for p in particles if p.alive)
        n_dead_init = len(particles) - n_alive_init
        logger.info(f"Initial: {len(particles)} particles, {n_alive_init} alive, {n_dead_init} hit EOS")
        
        # Score initial particles
        particles = self._score_particles(particles, query)
        
        # Save initial snapshots to pool (only alive particles)
        for p in particles:
            if p.alive:
                snapshot_pool.append(Snapshot(
                    particle_id=p.particle_id,
                    token_ids=p.token_ids.copy(),
                    step=p.current_step,
                    score=p.current_score,
                    alive=p.alive  # FIX: Store alive state
                ))
        
        # Main loop
        iteration = 0
        max_iterations = self.max_steps * 2  # Safety limit
        resample_count = 0  # Track number of resamples
        
        while iteration < max_iterations:
            iteration += 1
            
            n_alive = sum(1 for p in particles if p.alive)
            if n_alive == 0:
                logger.info("All particles dead, stopping")
                break
            
            # Fast path: only 1 alive → generate ALL remaining at once
            if n_alive == 1:
                logger.info("Only 1 particle alive, generating all remaining tokens...")
                alive_p = next(p for p in particles if p.alive)
                
                # Generate remaining tokens at once (capped to prevent over-generation)
                remaining_steps = self.max_steps - alive_p.current_step
                max_new_tokens = min(remaining_steps * self.max_tokens_per_step, 1024)
                
                params = SamplingParams(
                    n=1,
                    temperature=self.temperature,
                    max_tokens=max_new_tokens,
                )
                outputs = self.llm.generate(
                    prompt_token_ids=[alive_p.token_ids],
                    sampling_params=params,
                )
                new_tokens = list(outputs[0].outputs[0].token_ids)
                alive_p.token_ids.extend(new_tokens)
                alive_p.alive = False  # Done generating
                
                # Score once at the end
                particles = self._score_particles(particles, query)
                logger.info(f"Final generation complete, added {len(new_tokens)} tokens")
                break
            
            # Log current state
            alive_particles = [p for p in particles if p.alive]
            steps = [p.current_step for p in alive_particles]
            scores = [p.current_score for p in alive_particles]
            ess, weights = self._compute_ess(particles)
            
            logger.info(
                f"Iter {iteration}: alive={n_alive}, "
                f"steps={steps}, "
                f"scores={[f'{s:.3f}' for s in scores]}, "
                f"ESS={ess:.2f}/{n_alive}, "
                f"pool={len(snapshot_pool)}"
            )
            
            # Check resample condition (ESS < ρ × N)
            if self._check_resample_condition(particles):
                threshold = self.rho * n_alive
                logger.info(f"\n>>> RESAMPLE triggered (ESS={ess:.2f} < {threshold:.2f})")
                resample_count += 1  # Increment resample counter
                
                # Resample from pool
                new_particles, snapshot_pool = self._resample(
                    snapshot_pool=snapshot_pool,
                    n_select=n_alive,
                )
                
                if new_particles:
                    dead_particles = [p for p in particles if not p.alive]
                    particles = new_particles + dead_particles
                    logger.info(f"  Resampled {len(new_particles)} particles\n")
                    # Khong continue - tiep tuc generate step moi tu particles da resample
            
            # Generate next step cho moi alive particle
            particles = self._generate_one_step_batch(particles)
            
            # Score
            particles = self._score_particles(particles, query)
            
            # Save snapshots to pool
            for p in particles:
                if p.alive:
                    snapshot_pool.append(Snapshot(
                        particle_id=p.particle_id,
                        token_ids=p.token_ids.copy(),
                        step=p.current_step,
                        score=p.current_score,
                        alive=p.alive  # FIX: Store alive state
                    ))
        
        # Collect all particles info
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
        
        # Final: majority vote
        response, answer = self._majority_vote(particles)
        
        logger.info(f"Final answer: {answer}")
        logger.info(f"Returning {len(all_particles_info)} particles, response len={len(response)}")
        logger.info(f"Total resamples: {resample_count}")
        
        # Log resample count to external file
        with open("resample_log.txt", "a") as f:
            f.write(f"{query[:50]}... | Resamples: {resample_count}\n")
        
        # Log final results to debug file
        debug_file = "full_debug_trace.txt"
        with open(debug_file, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"FINAL RESULTS\n")
            f.write(f"{'='*80}\n")
            f.write(f"Selected answer: {answer}\n")
            f.write(f"Selected response length: {len(response)}\n")
            f.write(f"\nAll particles:\n")
            for p_info in all_particles_info:
                f.write(f"  P{p_info['particle_id']}: step={p_info['step']}, score={p_info['score']:.4f}, answer={p_info['answer']}\n")
            f.write(f"\nFull selected response:\n{response}\n")
            f.write(f"{'='*80}\n\n")
        
        return {
            "response": response,
            "answer": answer,
            "particles": all_particles_info,
            "resample_count": resample_count  # Add to output
        }
    
    def format_prompt(self, problem: str, instruction: str = None) -> Tuple[str, str]:
        """Format prompt voi chat template"""
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
            logger.info(f"Formatted prompt:\n{prompt[:500]}")  # Log first 500 chars
        else:
            prompt = user_content
        
        return prompt, problem


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    sampler = ParticleSampler(
        model_path="Qwen/Qwen2.5-Math-7B-Instruct",
        prm_model_path="Qwen/Qwen2.5-Math-PRM-7B",
        n_particles=8,
        max_steps=20,
        resample_threshold=0.12,
    )
    
    problem = "What is 123 + 456?"
    prompt, query = sampler.format_prompt(problem)
    
    result = sampler.sample(prompt, query)
    print("Result:", result)
