"""
Unit tests for resampling methods.

Tests:
- topN deterministic selection
- Systematic resampling correctness
- Multinomial sampling distribution
- ESS threshold computation
"""

import numpy as np
import pytest

from src.resampling import (
    resample_topN,
    resample_systematic,
    resample_multinomial,
    resample,
    compute_alpha_threshold,
    should_resample,
    count_unique_particles,
    compute_resampling_stats,
    Resampler,
)
from src.config import (
    ResampleConfig,
    ResampleMethod,
    ESSThresholdConfig,
    ESSThresholdMode,
)


class TestResampleTopN:
    """Tests for topN resampling."""
    
    def test_basic_selection(self):
        """Test basic top-N selection."""
        weights = np.array([0.1, 0.3, 0.2, 0.4])
        indices = resample_topN(weights, 2)
        
        # Should select indices 3 and 1 (highest weights)
        assert len(indices) == 2
        assert set(indices) == {1, 3}
    
    def test_deterministic(self):
        """Test that topN is deterministic."""
        weights = np.array([0.1, 0.3, 0.2, 0.4])
        
        indices1 = resample_topN(weights, 2)
        indices2 = resample_topN(weights, 2)
        
        np.testing.assert_array_equal(indices1, indices2)
    
    def test_select_all(self):
        """Test selecting all particles."""
        weights = np.array([0.1, 0.3, 0.2, 0.4])
        indices = resample_topN(weights, 4)
        
        assert len(indices) == 4
        assert set(indices) == {0, 1, 2, 3}
    
    def test_select_more_than_available(self):
        """Test selecting more than available particles."""
        weights = np.array([0.1, 0.3, 0.2, 0.4])
        indices = resample_topN(weights, 10)
        
        # Should return all available
        assert len(indices) == 4
    
    def test_sorted_by_weight(self):
        """Test that returned indices are sorted by weight descending."""
        weights = np.array([0.1, 0.3, 0.2, 0.4])
        indices = resample_topN(weights, 4)
        
        # First index should have highest weight
        assert indices[0] == 3


class TestResampleSystematic:
    """Tests for systematic resampling."""
    
    def test_output_size(self):
        """Test output has correct size."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        rng = np.random.default_rng(42)
        
        indices = resample_systematic(weights, 4, rng)
        assert len(indices) == 4
    
    def test_uniform_weights(self):
        """Test with uniform weights."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        rng = np.random.default_rng(42)
        
        indices = resample_systematic(weights, 4, rng)
        
        # With uniform weights, should get each index once
        assert len(set(indices)) == 4
    
    def test_concentrated_weight(self):
        """Test with weight concentrated on one particle."""
        weights = np.array([0.0, 0.0, 1.0, 0.0])
        rng = np.random.default_rng(42)
        
        indices = resample_systematic(weights, 4, rng)
        
        # All should select index 2
        assert all(i == 2 for i in indices)
    
    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        weights = np.array([0.1, 0.3, 0.2, 0.4])
        
        rng1 = np.random.default_rng(42)
        indices1 = resample_systematic(weights, 4, rng1)
        
        rng2 = np.random.default_rng(42)
        indices2 = resample_systematic(weights, 4, rng2)
        
        np.testing.assert_array_equal(indices1, indices2)
    
    def test_valid_indices(self):
        """Test that all indices are valid."""
        weights = np.array([0.1, 0.3, 0.2, 0.4])
        rng = np.random.default_rng(42)
        
        indices = resample_systematic(weights, 10, rng)
        
        assert all(0 <= i < len(weights) for i in indices)


class TestResampleMultinomial:
    """Tests for multinomial resampling."""
    
    def test_output_size(self):
        """Test output has correct size."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        rng = np.random.default_rng(42)
        
        indices = resample_multinomial(weights, 4, rng)
        assert len(indices) == 4
    
    def test_concentrated_weight(self):
        """Test with weight concentrated on one particle."""
        weights = np.array([0.0, 0.0, 1.0, 0.0])
        rng = np.random.default_rng(42)
        
        indices = resample_multinomial(weights, 4, rng)
        
        # All should select index 2
        assert all(i == 2 for i in indices)
    
    def test_distribution(self):
        """Test that sampling follows weight distribution."""
        weights = np.array([0.5, 0.5, 0.0, 0.0])
        rng = np.random.default_rng(42)
        
        # Sample many times
        n_samples = 10000
        indices = resample_multinomial(weights, n_samples, rng)
        
        # Count occurrences
        counts = np.bincount(indices, minlength=4)
        
        # Indices 0 and 1 should have roughly equal counts
        # Indices 2 and 3 should have 0 counts
        assert counts[2] == 0
        assert counts[3] == 0
        assert abs(counts[0] - counts[1]) < n_samples * 0.1  # Within 10%
    
    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        weights = np.array([0.1, 0.3, 0.2, 0.4])
        
        rng1 = np.random.default_rng(42)
        indices1 = resample_multinomial(weights, 4, rng1)
        
        rng2 = np.random.default_rng(42)
        indices2 = resample_multinomial(weights, 4, rng2)
        
        np.testing.assert_array_equal(indices1, indices2)


class TestResampleGeneric:
    """Tests for generic resample function."""
    
    def test_method_selection(self):
        """Test that correct method is called."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        rng = np.random.default_rng(42)
        
        # TopN
        indices = resample(weights, 2, ResampleMethod.TOP_N, rng)
        assert len(indices) == 2
        
        # Systematic
        indices = resample(weights, 4, ResampleMethod.SYSTEMATIC, rng)
        assert len(indices) == 4
        
        # Multinomial
        indices = resample(weights, 4, ResampleMethod.MULTINOMIAL, rng)
        assert len(indices) == 4
    
    def test_invalid_method(self):
        """Test error on invalid method."""
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        
        with pytest.raises(ValueError):
            resample(weights, 2, "invalid_method")


class TestAlphaThreshold:
    """Tests for ESS threshold computation."""
    
    def test_fixed_rho(self):
        """Test fixed rho threshold."""
        config = ESSThresholdConfig(
            mode=ESSThresholdMode.FIXED_RHO,
            rho=0.5,
        )
        
        # Should always return rho
        assert compute_alpha_threshold(0, 10, config) == 0.5
        assert compute_alpha_threshold(5, 10, config) == 0.5
        assert compute_alpha_threshold(10, 10, config) == 0.5
    
    def test_alpha_schedule(self):
        """Test time-varying alpha schedule."""
        config = ESSThresholdConfig(
            mode=ESSThresholdMode.ALPHA_SCHEDULE,
            alpha_start=1.2,
            alpha_end=0.5,
        )
        
        # At t=0, should be alpha_start
        alpha_0 = compute_alpha_threshold(0, 10, config)
        assert alpha_0 == pytest.approx(1.2, abs=0.01)
        
        # At t=T, should be alpha_end
        alpha_T = compute_alpha_threshold(10, 10, config)
        assert alpha_T == pytest.approx(0.5, abs=0.01)
        
        # Should decrease monotonically
        alphas = [compute_alpha_threshold(t, 10, config) for t in range(11)]
        for i in range(len(alphas) - 1):
            assert alphas[i] >= alphas[i + 1]
    
    def test_should_resample_logic(self):
        """Test resample trigger logic."""
        config = ESSThresholdConfig(
            mode=ESSThresholdMode.FIXED_RHO,
            rho=0.5,
        )
        
        # ESS = 4, N = 10, threshold = 5 -> should resample
        should, threshold = should_resample(4.0, 10, 0, 10, config)
        assert should is True
        assert threshold == 5.0
        
        # ESS = 6, N = 10, threshold = 5 -> should not resample
        should, threshold = should_resample(6.0, 10, 0, 10, config)
        assert should is False


class TestResampler:
    """Tests for Resampler class."""
    
    def test_initialization(self):
        """Test resampler initialization."""
        config = ResampleConfig(
            method=ResampleMethod.SYSTEMATIC,
        )
        resampler = Resampler(config, max_steps=10, seed=42)
        
        assert resampler.config == config
        assert resampler.max_steps == 10
    
    def test_resample_method(self):
        """Test resample method."""
        config = ResampleConfig(
            method=ResampleMethod.SYSTEMATIC,
        )
        resampler = Resampler(config, max_steps=10, seed=42)
        
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        indices = resampler.resample(weights, 4)
        
        assert len(indices) == 4


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_count_unique_particles(self):
        """Test unique particle counting."""
        indices = np.array([0, 1, 1, 2, 2, 2])
        assert count_unique_particles(indices) == 3
        
        indices = np.array([0, 0, 0, 0])
        assert count_unique_particles(indices) == 1
    
    def test_compute_resampling_stats(self):
        """Test resampling statistics computation."""
        weights = np.array([0.1, 0.3, 0.2, 0.4])
        indices = np.array([1, 1, 3, 3])
        
        stats = compute_resampling_stats(weights, indices)
        
        assert stats["n_unique"] == 2
        assert stats["n_selected"] == 4
        assert stats["diversity"] == 0.5
        assert stats["max_copies"] == 2
