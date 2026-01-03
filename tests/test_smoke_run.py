"""
Smoke tests for end-to-end pipeline.

Tests:
- Config loading
- Pipeline execution without errors
- Output format validation

Note: These tests require mocking vLLM or using a tiny model.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import Config, load_config, parse_cli_overrides
from src.utils import extract_boxed_answer, extract_assistant_response, set_seed


class TestConfigLoading:
    """Tests for config loading."""
    
    def test_default_config(self):
        """Test loading default config."""
        config = Config()
        
        assert config.system.seed == 42
        assert config.generation.n_particles == 8
        assert config.generation.max_steps == 20
    
    def test_yaml_config(self):
        """Test loading config from YAML."""
        yaml_content = """
system:
  seed: 123
  log_level: DEBUG
  
generation:
  n_particles: 4
  max_steps: 10
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            config = Config.from_yaml(f.name)
            
            assert config.system.seed == 123
            assert config.system.log_level == "DEBUG"
            assert config.generation.n_particles == 4
            assert config.generation.max_steps == 10
            
            os.unlink(f.name)
    
    def test_cli_overrides(self):
        """Test CLI override parsing."""
        overrides = parse_cli_overrides([
            "ps.resample.method=systematic",
            "generation.n_particles=4",
            "scoring.params.tau=2.0",
        ])
        
        assert overrides["ps.resample.method"] == "systematic"
        assert overrides["generation.n_particles"] == 4
        assert overrides["scoring.params.tau"] == 2.0
    
    def test_apply_overrides(self):
        """Test applying overrides to config."""
        config = Config()
        config = config.apply_overrides({
            "generation.n_particles": 4,
            "ps.resample.method": "systematic",
        })
        
        assert config.generation.n_particles == 4
        assert config.ps.resample.method.value == "systematic"


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_extract_boxed_answer(self):
        """Test boxed answer extraction."""
        # Simple case
        text = "The answer is \\boxed{42}."
        assert extract_boxed_answer(text) == "42"
        
        # Nested braces
        text = "\\boxed{\\frac{1}{2}}"
        assert extract_boxed_answer(text) == "\\frac{1}{2}"
        
        # Multiple boxed
        text = "First \\boxed{1}, then \\boxed{2}."
        assert extract_boxed_answer(text) == "2"  # Returns last
        
        # No boxed
        text = "The answer is 42."
        assert extract_boxed_answer(text) is None
    
    def test_extract_assistant_response(self):
        """Test assistant response extraction."""
        # With assistant marker
        text = "user: Hi\nassistant\nHello!"
        assert extract_assistant_response(text) == "Hello!"
        
        # No marker
        text = "Hello world!"
        assert extract_assistant_response(text) == "Hello world!"
    
    def test_set_seed(self):
        """Test seed setting for reproducibility."""
        set_seed(42)
        a = np.random.rand(5)
        
        set_seed(42)
        b = np.random.rand(5)
        
        np.testing.assert_array_equal(a, b)


class TestDataStructures:
    """Tests for data structures."""
    
    def test_particle_creation(self):
        """Test particle data structure."""
        from src.ps_core import Particle
        
        particle = Particle(
            particle_id=0,
            token_ids=[1, 2, 3, 4, 5],
            prompt_len=2,
            step=1,
            alive=True,
        )
        
        assert particle.particle_id == 0
        assert particle.generated_tokens == [3, 4, 5]
        assert particle.num_generated_tokens == 3
    
    def test_particle_copy(self):
        """Test particle copying."""
        from src.ps_core import Particle
        
        particle = Particle(
            particle_id=0,
            token_ids=[1, 2, 3],
            prompt_len=1,
            step=1,
            cum_logprob=-5.0,
        )
        
        copy = particle.copy(new_id=1)
        
        assert copy.particle_id == 1
        assert copy.token_ids == [1, 2, 3]
        assert copy.cum_logprob == -5.0
        
        # Ensure deep copy
        copy.token_ids.append(4)
        assert particle.token_ids == [1, 2, 3]
    
    def test_snapshot_pool(self):
        """Test snapshot pool operations."""
        from src.ps_core import Snapshot, SnapshotPool
        
        pool = SnapshotPool()
        
        # Add snapshots
        for i in range(5):
            pool.add(Snapshot(
                particle_id=i,
                token_ids=[1, 2, 3],
                prompt_len=1,
                step=i,
                logL=float(i),
                cum_logprob=-float(i),
                alive=True,
            ))
        
        assert len(pool) == 5
        
        # Check cached arrays
        np.testing.assert_array_equal(
            pool.logL_array,
            np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        )
    
    def test_snapshot_pool_pruning(self):
        """Test snapshot pool pruning."""
        from src.ps_core import Snapshot, SnapshotPool
        
        pool = SnapshotPool(max_size=3)
        
        for i in range(5):
            pool.add(Snapshot(
                particle_id=i,
                token_ids=[1, 2, 3],
                prompt_len=1,
                step=i,
                logL=float(i),
                cum_logprob=-float(i),
                alive=True,
            ))
        
        pool.prune(current_step=4)
        
        # Should keep only last 3
        assert len(pool) == 3


class TestTemperatureSchedule:
    """Tests for temperature schedule."""
    
    def test_fixed_linear(self):
        """Test fixed linear schedule."""
        from src.ps_core import TemperatureSchedule
        from src.config import PSConfig, TemperatureScheduleConfig, TemperatureScheduleMode
        
        ps_config = PSConfig(
            temperature_schedule=TemperatureScheduleConfig(
                mode=TemperatureScheduleMode.FIXED_LINEAR,
                beta_min=0.0,
                beta_max=1.0,
            )
        )
        
        schedule = TemperatureSchedule(ps_config, max_steps=10)
        
        # Check linear progression
        assert schedule.compute_fixed_beta(0) == pytest.approx(0.0)
        assert schedule.compute_fixed_beta(5) == pytest.approx(0.5, rel=0.1)
        assert schedule.compute_fixed_beta(9) == pytest.approx(1.0)
    
    def test_fixed_cosine(self):
        """Test fixed cosine schedule."""
        from src.ps_core import TemperatureSchedule
        from src.config import PSConfig, TemperatureScheduleConfig, TemperatureScheduleMode
        
        ps_config = PSConfig(
            temperature_schedule=TemperatureScheduleConfig(
                mode=TemperatureScheduleMode.FIXED_COSINE,
                beta_min=0.0,
                beta_max=1.0,
            )
        )
        
        schedule = TemperatureSchedule(ps_config, max_steps=10)
        
        # Cosine should start slow, accelerate in middle, slow at end
        betas = [schedule.compute_fixed_beta(t) for t in range(10)]
        
        # First beta should be 0
        assert betas[0] == pytest.approx(0.0)
        
        # Middle should be around 0.5
        assert 0.4 < betas[4] < 0.6
        
        # Should be monotonically increasing
        for i in range(len(betas) - 1):
            assert betas[i] <= betas[i + 1]


class TestMockedPipeline:
    """Tests with mocked vLLM."""
    
    @pytest.fixture
    def mock_vllm(self):
        """Create mocked vLLM components."""
        # Mock vLLM imports
        mock_llm = MagicMock()
        mock_tokenizer = MagicMock()
        
        # Mock tokenizer methods
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        mock_tokenizer.decode.return_value = "The answer is \\boxed{42}."
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        
        mock_llm.get_tokenizer.return_value = mock_tokenizer
        
        return mock_llm, mock_tokenizer
    
    def test_format_prompt(self, mock_vllm):
        """Test prompt formatting."""
        mock_llm, mock_tokenizer = mock_vllm
        
        with patch('src.vllm_wrappers.LLM', return_value=mock_llm):
            from src.vllm_wrappers import LLMGenerator
            from src.config import BaseLLMConfig, GenerationConfig
            
            generator = LLMGenerator(
                model_config=BaseLLMConfig(),
                generation_config=GenerationConfig(),
            )
            generator._llm = mock_llm
            generator._tokenizer = mock_tokenizer
            
            prompt, query = generator.format_chat_prompt("What is 2+2?")
            
            assert query == "What is 2+2?"
            mock_tokenizer.apply_chat_template.assert_called()


class TestIntegration:
    """Integration tests (require actual models or extensive mocking)."""
    
    @pytest.mark.skip(reason="Requires vLLM and models")
    def test_mini_eval(self):
        """
        Run mini evaluation on a few samples.
        
        To run this test:
        1. Install vLLM and required models
        2. Remove the skip decorator
        3. Update model paths in config
        """
        from src.ps_core import PersistentSampler
        from src.config import Config
        
        config = Config()
        config = config.apply_overrides({
            "generation.n_particles": 2,
            "generation.max_steps": 3,
        })
        
        sampler = PersistentSampler(config)
        
        prompt, query = sampler.format_prompt("What is 1 + 1?")
        result = sampler.sample(prompt, query)
        
        assert result.response is not None
        assert "particles" in result.__dict__


class TestOutputFormats:
    """Tests for output format validation."""
    
    def test_sampling_result_format(self):
        """Test SamplingResult structure."""
        from src.ps_core import SamplingResult
        
        result = SamplingResult(
            response="The answer is \\boxed{42}.",
            answer="42",
            particles=[
                {"particle_id": 0, "step": 5, "score": 0.8},
                {"particle_id": 1, "step": 4, "score": 0.6},
            ],
            resample_count=2,
            timing={"generation": 1.5, "scoring": 0.3},
            betas=[0.1, 0.3, 0.5, 0.7],
            ess_history=[4.0, 3.5, 2.8, 2.1],
        )
        
        assert result.response == "The answer is \\boxed{42}."
        assert result.answer == "42"
        assert len(result.particles) == 2
        assert result.resample_count == 2
    
    def test_json_serializable(self):
        """Test that results are JSON serializable."""
        result_dict = {
            "response": "The answer is \\boxed{42}.",
            "answer": "42",
            "particles": [
                {"particle_id": 0, "step": 5, "score": 0.8},
            ],
            "resample_count": 2,
            "timing": {"generation": 1.5},
            "betas": [0.1, 0.3, 0.5],
            "ess_history": [4.0, 3.5],
        }
        
        # Should not raise
        json_str = json.dumps(result_dict)
        loaded = json.loads(json_str)
        
        assert loaded["answer"] == "42"
