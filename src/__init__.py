# Persistent Sampling for LLM Test-Time Scaling
"""
Implementation of Persistent Sampling (PS) Algorithm 2 adapted for LLM inference.

Core idea:
- Maintain a growing pool of particles (partial generations) across time
- When degeneracy happens, resample from ALL historical steps (persistent pool)
- Use mixture importance density with balance-heuristic weights
"""

__version__ = "0.1.0"
