# Persistent Sampling for LLM Test-Time Scaling

Implementation of **Algorithm 2: Persistent Sampling** from "Persistent Sampling: Unleashing the Potential of Sequential Monte Carlo for Language Model Post-Training" adapted for LLM test-time scaling inference.

## Overview

Persistent Sampling (PS) is a Sequential Monte Carlo (SMC) method that maintains a pool of "snapshots" across generation steps, using balance-heuristic importance weights and adaptive temperature scheduling to efficiently explore the solution space during LLM inference.

### Key Features

- **Algorithm 2 Implementation**: Faithful implementation of the PS algorithm with balance-heuristic weights (Eq. 17)
- **Config-Driven**: All hyperparameters controlled via YAML config + CLI overrides
- **Multiple Scoring Functions**: `logprob_power`, `logprob_avg`, `prm_reward`, `hybrid_prm_logprob`
- **Length Priors**: `none`, `poisson`, `geometric`, `lognormal`, `linear_penalty`
- **Resampling Methods**: `topN` (deterministic), `systematic`, `multinomial`
- **Adaptive Temperature**: ESS-based bisection for optimal β schedule
- **MATH-500 Evaluation**: Built-in evaluation pipeline with `math_verify`

## Installation

### 1. Create Environment

```bash
# Create conda environment
conda create -n ps-sampling python=3.10 -y
conda activate ps-sampling

# Install dependencies
pip install -r requirements.txt
```

### 2. Dependencies

The main dependencies are:
- `vllm>=0.6.0` - High-throughput LLM inference
- `torch>=2.0` - PyTorch backend
- `pydantic>=2.0` - Config validation
- `pyyaml` - YAML config loading
- `datasets` - HuggingFace datasets
- `math-verify` - Math answer verification

## Quick Start

### Basic Evaluation

```bash
# Run MATH-500 evaluation with default config
python -m src.eval_math500 --config config.yaml

# Override parameters via CLI
python -m src.eval_math500 --config config.yaml \
    --generation.n_particles=16 \
    --ps.resample.method=systematic
```

### Smoke Test

```bash
# Run smoke test with minimal settings
python -m src.eval_math500 --config config.yaml \
    --smoke_test.enabled=true \
    --smoke_test.n_samples=2
```

## Configuration

All settings are in `config.yaml`. Key sections:

### Models

```yaml
models:
  generator:
    model_path: "Qwen/Qwen2.5-Math-1.5B-Instruct"
  prm:
    enabled: false  # Set to true to use PRM scoring
    model_path: "Qwen/Qwen2.5-Math-PRM-7B"
```

### Generation

```yaml
generation:
  n_particles: 8        # Number of particles (M)
  max_steps: 20         # Max generation steps (T)
  temperature: 0.7      # Sampling temperature
```

### Scoring

```yaml
scoring:
  type: logprob_power   # Score function type
  params:
    tau: 1.0            # Temperature for logprob_power
  length_prior:
    type: none          # Length regularization
```

### PS Algorithm

```yaml
ps:
  temperature_schedule:
    mode: adaptive_ess  # Adaptive β via ESS bisection
    target_ess: 0.5     # Target ESS ratio
  resample:
    method: topN        # Resampling method
    alpha_start: 0.5    # ESS threshold (start)
    alpha_end: 0.8      # ESS threshold (end)
```

## Algorithm Details

### Persistent Sampling (Algorithm 2)

The PS algorithm maintains a pool of snapshots (particle states) across steps:

1. **Initialize**: Sample M particles from prior (generate first tokens)
2. **For each step t**:
   - Generate next tokens for alive particles
   - Compute scores `logL` for each particle
   - Add snapshots to persistent pool
   - Compute balance-heuristic weights (Eq. 17):
     ```
     w_s^i ∝ ψ_t(x^i) / Σ_{s'≤t} M_{s'} · q_{s'}^{β_t}(x^i)
     ```
   - Compute ESS; if below threshold, resample
3. **Return**: Best particle (highest score) as final answer

### Scoring Functions

| Type | Formula | Use Case |
|------|---------|----------|
| `logprob_power` | `exp(logp / τ)` | General purpose |
| `logprob_avg` | `exp(logp / n_tokens)` | Length-normalized |
| `prm_reward` | PRM score | With reward model |
| `hybrid_prm_logprob` | `α·prm + (1-α)·logp` | Combined |

### Resampling Methods

| Method | Description | Pros |
|--------|-------------|------|
| `topN` | Keep top M by weight | Deterministic, stable |
| `systematic` | Systematic resampling | Low variance |
| `multinomial` | Multinomial resampling | Unbiased |

## Hyperparameter Tuning

### Grid Search

```bash
python -m src.tuning --config config.yaml \
    --tuning.enabled=true \
    --tuning.method=grid \
    --tuning.validation_samples=50
```

### Random Search

```bash
python -m src.tuning --config config.yaml \
    --tuning.enabled=true \
    --tuning.method=random \
    --tuning.n_trials=100
```

Tuning search space is defined in config:

```yaml
tuning:
  search_space:
    ps.resample.alpha_start: [0.3, 0.5, 0.7]
    ps.resample.alpha_end: [0.6, 0.8, 0.9]
    scoring.params.tau: [0.5, 1.0, 2.0]
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_resampling.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Output Format

Results are saved as JSONL with one result per line:

```json
{
  "problem_idx": 0,
  "problem": "What is 2 + 2?",
  "ground_truth": "4",
  "response": "The answer is \\boxed{4}.",
  "extracted_answer": "4",
  "correct": true,
  "timing": {"total": 2.5, "generation": 2.0, "scoring": 0.3},
  "resample_count": 3,
  "final_ess": 5.2
}
```

## Project Structure

```
persistent-sampling/
├── config.yaml           # Main configuration file
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── src/
│   ├── __init__.py
│   ├── config.py        # Pydantic config classes
│   ├── utils.py         # Utilities (logging, numerics)
│   ├── scoring.py       # Score functions & length priors
│   ├── resampling.py    # Resampling methods
│   ├── vllm_wrappers.py # vLLM LLM & PRM wrappers
│   ├── ps_core.py       # PS Algorithm 2 implementation
│   ├── data.py          # Dataset loaders
│   ├── tuning.py        # Hyperparameter tuning
│   └── eval_math500.py  # MATH-500 evaluation script
└── tests/
    ├── __init__.py
    ├── test_resampling.py
    ├── test_weights_ps.py
    ├── test_score_functions.py
    └── test_smoke_run.py
```

## Numerical Stability

The implementation uses several techniques for numerical stability:

- **Log-space computation**: All weight computations use `logsumexp`
- **float64 precision**: Weight arrays use numpy float64
- **Score clipping**: Scores are clipped to `[min_score, max_score]`
- **ESS monitoring**: ESS is tracked and logged at each step

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{persistent-sampling,
  title={Persistent Sampling: Unleashing the Potential of Sequential Monte Carlo for Language Model Post-Training},
  author={...},
  year={2024}
}
```

## License

MIT License
