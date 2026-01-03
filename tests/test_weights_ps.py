"""
Unit tests for PS weight computation.

Tests:
- Vectorized weight computation correctness
- Log-space stability
- Reference implementation comparison
- Edge cases (empty pool, single particle)
"""

import numpy as np
import pytest

from src.ps_core import (
    compute_ps_weights_vectorized,
    compute_ps_ess,
)
from src.utils import (
    logsumexp,
    logmeanexp,
    normalize_log_weights,
    compute_ess,
)


class TestLogsumexp:
    """Tests for logsumexp implementation."""
    
    def test_basic(self):
        """Test basic logsumexp computation."""
        a = np.array([1.0, 2.0, 3.0])
        result = logsumexp(a)
        
        expected = np.log(np.exp(1) + np.exp(2) + np.exp(3))
        assert result == pytest.approx(expected, rel=1e-10)
    
    def test_with_axis(self):
        """Test logsumexp with axis."""
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        result = logsumexp(a, axis=0)
        assert result.shape == (2,)
        
        result = logsumexp(a, axis=1)
        assert result.shape == (2,)
    
    def test_stability(self):
        """Test numerical stability with large values."""
        # Large positive values
        a = np.array([1000.0, 1001.0, 1002.0])
        result = logsumexp(a)
        
        # Should not overflow
        assert np.isfinite(result)
        
        # Large negative values
        a = np.array([-1000.0, -1001.0, -1002.0])
        result = logsumexp(a)
        
        assert np.isfinite(result)
    
    def test_empty_array(self):
        """Test with empty array."""
        a = np.array([])
        result = logsumexp(a)
        
        assert result == -np.inf
    
    def test_all_negative_inf(self):
        """Test with all -inf values."""
        a = np.array([-np.inf, -np.inf, -np.inf])
        result = logsumexp(a)
        
        assert result == -np.inf


class TestLogmeanexp:
    """Tests for logmeanexp implementation."""
    
    def test_basic(self):
        """Test basic logmeanexp computation."""
        a = np.array([1.0, 2.0, 3.0])
        result = logmeanexp(a)
        
        expected = np.log(np.mean(np.exp(a)))
        assert result == pytest.approx(expected, rel=1e-10)
    
    def test_relationship_to_logsumexp(self):
        """Test relationship: logmeanexp = logsumexp - log(n)."""
        a = np.array([1.0, 2.0, 3.0])
        
        result = logmeanexp(a)
        expected = logsumexp(a) - np.log(len(a))
        
        assert result == pytest.approx(expected, rel=1e-10)


class TestNormalizeLogWeights:
    """Tests for log weight normalization."""
    
    def test_probabilities_sum_to_one(self):
        """Test that output probabilities sum to 1."""
        log_weights = np.array([1.0, 2.0, 3.0])
        probs, _ = normalize_log_weights(log_weights)
        
        assert probs.sum() == pytest.approx(1.0, rel=1e-10)
    
    def test_log_normalizing_constant(self):
        """Test log normalizing constant computation."""
        log_weights = np.array([1.0, 2.0, 3.0])
        probs, log_Z = normalize_log_weights(log_weights)
        
        expected_Z = np.exp(1) + np.exp(2) + np.exp(3)
        assert log_Z == pytest.approx(np.log(expected_Z), rel=1e-10)
    
    def test_stability(self):
        """Test stability with extreme values."""
        log_weights = np.array([1000.0, 1001.0, 1002.0])
        probs, log_Z = normalize_log_weights(log_weights)
        
        assert np.all(np.isfinite(probs))
        assert np.isfinite(log_Z)
        assert probs.sum() == pytest.approx(1.0, rel=1e-10)


class TestComputeESS:
    """Tests for ESS computation."""
    
    def test_uniform_weights(self):
        """Test ESS with uniform weights equals N."""
        n = 10
        weights = np.ones(n) / n
        ess = compute_ess(weights)
        
        assert ess == pytest.approx(n, rel=1e-10)
    
    def test_concentrated_weight(self):
        """Test ESS with concentrated weight equals 1."""
        weights = np.array([1.0, 0.0, 0.0, 0.0])
        ess = compute_ess(weights)
        
        assert ess == pytest.approx(1.0, rel=1e-10)
    
    def test_bounds(self):
        """Test ESS is between 1 and N."""
        weights = np.array([0.1, 0.3, 0.2, 0.4])
        ess = compute_ess(weights)
        
        assert 1.0 <= ess <= len(weights)


class TestPSWeightsVectorized:
    """Tests for PS weight computation."""
    
    def test_empty_pool(self):
        """Test with empty pool."""
        pool_logL = np.array([])
        betas = np.array([0.5])
        logZ_hats = np.array([0.0])
        
        probs, logZ = compute_ps_weights_vectorized(pool_logL, betas, logZ_hats, 1.0)
        
        assert len(probs) == 0
        assert logZ == 0.0
    
    def test_no_history(self):
        """Test with no history (first step)."""
        pool_logL = np.array([1.0, 2.0, 3.0])
        betas = np.array([])  # No history
        logZ_hats = np.array([])
        
        probs, _ = compute_ps_weights_vectorized(pool_logL, betas, logZ_hats, 0.5)
        
        # Should return uniform weights
        assert len(probs) == 3
        assert probs.sum() == pytest.approx(1.0, rel=1e-10)
    
    def test_probabilities_sum_to_one(self):
        """Test output probabilities sum to 1."""
        pool_logL = np.array([1.0, 2.0, 3.0, 4.0])
        betas = np.array([0.2, 0.4, 0.6])
        logZ_hats = np.array([0.1, 0.2, 0.3])
        
        probs, _ = compute_ps_weights_vectorized(pool_logL, betas, logZ_hats, 0.8)
        
        assert probs.sum() == pytest.approx(1.0, rel=1e-10)
    
    def test_reference_implementation(self):
        """Test against slow reference implementation."""
        pool_logL = np.array([1.0, 2.0, 3.0])
        betas = np.array([0.2, 0.5])
        logZ_hats = np.array([0.1, 0.3])
        current_beta = 0.8
        
        # Reference implementation (slow)
        def compute_weights_reference(pool_logL, betas, logZ_hats, current_beta):
            M = len(pool_logL)
            T = len(betas)
            
            log_weights = np.zeros(M)
            
            for i in range(M):
                logL_i = pool_logL[i]
                
                # Compute denominator
                denom_terms = []
                for s in range(T):
                    term = betas[s] * logL_i - logZ_hats[s]
                    denom_terms.append(term)
                
                denom = logsumexp(np.array(denom_terms)) - np.log(T)
                
                # Compute log weight
                log_weights[i] = current_beta * logL_i - denom
            
            # Normalize
            probs, logZ = normalize_log_weights(log_weights)
            return probs, logZ
        
        # Vectorized
        probs_vec, logZ_vec = compute_ps_weights_vectorized(
            pool_logL, betas, logZ_hats, current_beta
        )
        
        # Reference
        probs_ref, logZ_ref = compute_weights_reference(
            pool_logL, betas, logZ_hats, current_beta
        )
        
        np.testing.assert_array_almost_equal(probs_vec, probs_ref, decimal=10)
        assert logZ_vec == pytest.approx(logZ_ref, rel=1e-10)
    
    def test_stability_large_values(self):
        """Test stability with large log-likelihood values."""
        pool_logL = np.array([100.0, 200.0, 300.0])
        betas = np.array([0.2, 0.5])
        logZ_hats = np.array([10.0, 50.0])
        
        probs, logZ = compute_ps_weights_vectorized(pool_logL, betas, logZ_hats, 0.8)
        
        assert np.all(np.isfinite(probs))
        assert np.isfinite(logZ)
        assert probs.sum() == pytest.approx(1.0, rel=1e-10)
    
    def test_stability_negative_values(self):
        """Test stability with negative log-likelihood values."""
        pool_logL = np.array([-100.0, -200.0, -300.0])
        betas = np.array([0.2, 0.5])
        logZ_hats = np.array([-10.0, -50.0])
        
        probs, logZ = compute_ps_weights_vectorized(pool_logL, betas, logZ_hats, 0.8)
        
        assert np.all(np.isfinite(probs))
        assert np.isfinite(logZ)
        assert probs.sum() == pytest.approx(1.0, rel=1e-10)
    
    def test_beta_effect(self):
        """Test that higher beta concentrates weight on high logL."""
        pool_logL = np.array([1.0, 2.0, 3.0])
        betas = np.array([0.1])
        logZ_hats = np.array([0.0])
        
        # Low beta
        probs_low, _ = compute_ps_weights_vectorized(pool_logL, betas, logZ_hats, 0.1)
        
        # High beta
        probs_high, _ = compute_ps_weights_vectorized(pool_logL, betas, logZ_hats, 2.0)
        
        # High beta should give more weight to highest logL (index 2)
        assert probs_high[2] > probs_low[2]


class TestPSESS:
    """Tests for PS ESS computation."""
    
    def test_basic(self):
        """Test basic PS ESS computation."""
        pool_logL = np.array([1.0, 2.0, 3.0])
        betas = np.array([0.2, 0.5])
        logZ_hats = np.array([0.1, 0.3])
        
        ess = compute_ps_ess(pool_logL, betas, logZ_hats, 0.8)
        
        assert 1.0 <= ess <= len(pool_logL)
    
    def test_empty_pool(self):
        """Test ESS with empty pool."""
        pool_logL = np.array([])
        betas = np.array([0.5])
        logZ_hats = np.array([0.0])
        
        ess = compute_ps_ess(pool_logL, betas, logZ_hats, 1.0)
        
        assert ess == 0.0
