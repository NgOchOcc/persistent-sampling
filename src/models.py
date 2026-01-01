from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Particle:
    particle_id: int
    token_ids: List[int]
    current_step: int = 0
    alive: bool = True
    current_score: float = 0.5
    prm_raw: float = 0.5
    log_prob: float = 0.0  # Cumulative log probability
    num_generated_tokens: int = 0  # Number of tokens generated (excluding prompt)

    def copy(self, new_particle_id: Optional[int] = None):
        return Particle(
            particle_id=new_particle_id if new_particle_id is not None else self.particle_id,
            token_ids=self.token_ids.copy(),
            current_step=self.current_step,
            alive=self.alive,
            current_score=self.current_score,
            prm_raw=self.prm_raw,
            log_prob=self.log_prob,
            num_generated_tokens=self.num_generated_tokens
        )


@dataclass
class Snapshot:
    particle_id: int
    token_ids: List[int]
    step: int
    score: float
    alive: bool = True
    log_prob: float = 0.0
    num_generated_tokens: int = 0

    def to_particle(self, new_particle_id: int):
        return Particle(
            particle_id=new_particle_id,
            token_ids=self.token_ids.copy(),
            current_step=self.step,
            alive=self.alive,
            current_score=self.score,
            log_prob=self.log_prob,
            num_generated_tokens=self.num_generated_tokens
        )
