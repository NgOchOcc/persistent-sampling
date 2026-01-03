"""
Unit tests for scoring functions.

Tests:
- Length prior computation
- Score function correctness
- Score clipping
- Factory functions
"""

import numpy as np
import pytest

from src.scoring import (
    # Length priors
    NoLengthPrior,
    PoissonLengthPrior,
    GeometricLengthPrior,
    LognormalLengthPrior,
    LinearPenaltyLengthPrior,
    create_length_prior,
    # Score functions
    ParticleScoreData,
    LogprobPowerScore,
    LogprobAvgScore,
    PRMRewardScore,
    HybridPRMLogprobScore,
    create_score_function,
)
from src.config import (
    ScoringConfig,
    ScoreType,
    LengthPriorConfig,
    LengthPriorType,
    LengthPriorParamsConfig,
    ScoringParamsConfig,
)


class TestNoLengthPrior:
    """Tests for NoLengthPrior."""
    
    def test_returns_zero(self):
        """Test that it always returns 0."""
        prior = NoLengthPrior()
        
        assert prior.log_prob(0) == 0.0
        assert prior.log_prob(100) == 0.0
        assert prior.log_prob(1000) == 0.0


class TestPoissonLengthPrior:
    """Tests for PoissonLengthPrior."""
    
    def test_basic(self):
        """Test basic Poisson log probability."""
        prior = PoissonLengthPrior(lambda_param=50.0)
        
        # log P(k) should be finite for reasonable k
        assert np.isfinite(prior.log_prob(50))
        assert np.isfinite(prior.log_prob(10))
        assert np.isfinite(prior.log_prob(100))
    
    def test_negative_length(self):
        """Test negative length returns -inf."""
        prior = PoissonLengthPrior(lambda_param=50.0)
        
        assert prior.log_prob(-1) == -np.inf
    
    def test_mode(self):
        """Test that mode is near lambda."""
        prior = PoissonLengthPrior(lambda_param=50.0)
        
        # Maximum should be near lambda
        probs = [prior.log_prob(k) for k in range(100)]
        mode = np.argmax(probs)
        
        assert abs(mode - 50) <= 1
    
    def test_zero_length(self):
        """Test zero length probability."""
        prior = PoissonLengthPrior(lambda_param=50.0)
        
        # P(0) = exp(-λ), log P(0) = -λ
        assert prior.log_prob(0) == pytest.approx(-50.0, rel=1e-10)


class TestGeometricLengthPrior:
    """Tests for GeometricLengthPrior."""
    
    def test_basic(self):
        """Test basic geometric log probability."""
        prior = GeometricLengthPrior(p=0.05)
        
        assert np.isfinite(prior.log_prob(0))
        assert np.isfinite(prior.log_prob(50))
        assert np.isfinite(prior.log_prob(100))
    
    def test_decreasing(self):
        """Test that probability decreases with length."""
        prior = GeometricLengthPrior(p=0.05)
        
        # Probabilities should decrease
        for k in range(10):
            assert prior.log_prob(k) > prior.log_prob(k + 1)
    
    def test_negative_length(self):
        """Test negative length returns -inf."""
        prior = GeometricLengthPrior(p=0.05)
        
        assert prior.log_prob(-1) == -np.inf


class TestLognormalLengthPrior:
    """Tests for LognormalLengthPrior."""
    
    def test_basic(self):
        """Test basic lognormal log probability."""
        prior = LognormalLengthPrior(mu=3.5, sigma=0.5)
        
        assert np.isfinite(prior.log_prob(10))
        assert np.isfinite(prior.log_prob(50))
    
    def test_zero_length(self):
        """Test zero length returns -inf."""
        prior = LognormalLengthPrior(mu=3.5, sigma=0.5)
        
        assert prior.log_prob(0) == -np.inf
    
    def test_negative_length(self):
        """Test negative length returns -inf."""
        prior = LognormalLengthPrior(mu=3.5, sigma=0.5)
        
        assert prior.log_prob(-1) == -np.inf


class TestLinearPenaltyLengthPrior:
    """Tests for LinearPenaltyLengthPrior."""
    
    def test_basic(self):
        """Test linear penalty computation."""
        prior = LinearPenaltyLengthPrior(slope=-0.01)
        
        assert prior.log_prob(0) == 0.0
        assert prior.log_prob(100) == pytest.approx(-1.0, rel=1e-10)
        assert prior.log_prob(1000) == pytest.approx(-10.0, rel=1e-10)
    
    def test_linearity(self):
        """Test that penalty is linear."""
        prior = LinearPenaltyLengthPrior(slope=-0.01)
        
        # Check linearity
        for k in range(10, 100, 10):
            diff = prior.log_prob(k + 10) - prior.log_prob(k)
            assert diff == pytest.approx(-0.1, rel=1e-10)


class TestCreateLengthPrior:
    """Tests for length prior factory."""
    
    def test_create_none(self):
        """Test creating no length prior."""
        config = LengthPriorConfig(type=LengthPriorType.NONE)
        prior = create_length_prior(config)
        
        assert isinstance(prior, NoLengthPrior)
    
    def test_create_poisson(self):
        """Test creating Poisson prior."""
        config = LengthPriorConfig(
            type=LengthPriorType.POISSON,
            params=LengthPriorParamsConfig(poisson_lambda=30.0)
        )
        prior = create_length_prior(config)
        
        assert isinstance(prior, PoissonLengthPrior)
        assert prior.lambda_param == 30.0
    
    def test_create_geometric(self):
        """Test creating geometric prior."""
        config = LengthPriorConfig(
            type=LengthPriorType.GEOMETRIC,
            params=LengthPriorParamsConfig(geometric_p=0.1)
        )
        prior = create_length_prior(config)
        
        assert isinstance(prior, GeometricLengthPrior)
        assert prior.p == 0.1


class TestParticleScoreData:
    """Tests for ParticleScoreData."""
    
    def test_basic(self):
        """Test basic data structure."""
        data = ParticleScoreData(
            cum_logprob=-10.0,
            length=100,
            prm_score=0.8,
        )
        
        assert data.cum_logprob == -10.0
        assert data.length == 100
        assert data.prm_score == 0.8


class TestLogprobPowerScore:
    """Tests for LogprobPowerScore."""
    
    def test_basic(self):
        """Test basic score computation."""
        score_fn = LogprobPowerScore(tau=1.0)
        data = ParticleScoreData(cum_logprob=-10.0, length=100)
        
        score = score_fn.compute_base_score(data)
        
        assert score == pytest.approx(-10.0, rel=1e-10)
    
    def test_tau_scaling(self):
        """Test tau scaling."""
        score_fn = LogprobPowerScore(tau=2.0)
        data = ParticleScoreData(cum_logprob=-10.0, length=100)
        
        score = score_fn.compute_base_score(data)
        
        assert score == pytest.approx(-20.0, rel=1e-10)
    
    def test_with_length_prior(self):
        """Test score with length prior."""
        length_prior = LinearPenaltyLengthPrior(slope=-0.01)
        score_fn = LogprobPowerScore(
            tau=1.0,
            length_prior=length_prior,
            lambda_len=1.0,
        )
        data = ParticleScoreData(cum_logprob=-10.0, length=100)
        
        score = score_fn.compute_score(data)
        
        # -10 + 1.0 * (-0.01 * 100) = -10 - 1 = -11
        assert score == pytest.approx(-11.0, rel=1e-10)


class TestLogprobAvgScore:
    """Tests for LogprobAvgScore."""
    
    def test_basic(self):
        """Test basic average score computation."""
        score_fn = LogprobAvgScore(tau=1.0)
        data = ParticleScoreData(cum_logprob=-10.0, length=100)
        
        score = score_fn.compute_base_score(data)
        
        assert score == pytest.approx(-0.1, rel=1e-10)
    
    def test_zero_length(self):
        """Test with zero length."""
        score_fn = LogprobAvgScore(tau=1.0)
        data = ParticleScoreData(cum_logprob=-10.0, length=0)
        
        score = score_fn.compute_base_score(data)
        
        assert score == 0.0


class TestPRMRewardScore:
    """Tests for PRMRewardScore."""
    
    def test_basic(self):
        """Test basic PRM score."""
        score_fn = PRMRewardScore(prm_scale=1.0)
        data = ParticleScoreData(cum_logprob=-10.0, length=100, prm_score=0.8)
        
        score = score_fn.compute_base_score(data)
        
        assert score == pytest.approx(0.8, rel=1e-10)
    
    def test_scaling(self):
        """Test PRM scaling."""
        score_fn = PRMRewardScore(prm_scale=2.0)
        data = ParticleScoreData(cum_logprob=-10.0, length=100, prm_score=0.8)
        
        score = score_fn.compute_base_score(data)
        
        assert score == pytest.approx(1.6, rel=1e-10)
    
    def test_missing_prm_score(self):
        """Test error when PRM score is missing."""
        score_fn = PRMRewardScore(prm_scale=1.0)
        data = ParticleScoreData(cum_logprob=-10.0, length=100)  # No PRM score
        
        with pytest.raises(ValueError):
            score_fn.compute_base_score(data)


class TestHybridPRMLogprobScore:
    """Tests for HybridPRMLogprobScore."""
    
    def test_basic(self):
        """Test hybrid score computation."""
        score_fn = HybridPRMLogprobScore(prm_scale=1.0, logprob_scale=0.1)
        data = ParticleScoreData(cum_logprob=-10.0, length=100, prm_score=0.8)
        
        score = score_fn.compute_base_score(data)
        
        # 1.0 * 0.8 + 0.1 * (-10.0) = 0.8 - 1.0 = -0.2
        assert score == pytest.approx(-0.2, rel=1e-10)


class TestScoreClipping:
    """Tests for score clipping."""
    
    def test_clip_high(self):
        """Test clipping high values."""
        score_fn = LogprobPowerScore(
            tau=1.0,
            clip_min=-100.0,
            clip_max=100.0,
        )
        data = ParticleScoreData(cum_logprob=1000.0, length=100)  # Very high
        
        score = score_fn.compute_score(data)
        
        assert score == 100.0
    
    def test_clip_low(self):
        """Test clipping low values."""
        score_fn = LogprobPowerScore(
            tau=1.0,
            clip_min=-100.0,
            clip_max=100.0,
        )
        data = ParticleScoreData(cum_logprob=-1000.0, length=100)  # Very low
        
        score = score_fn.compute_score(data)
        
        assert score == -100.0


class TestCreateScoreFunction:
    """Tests for score function factory."""
    
    def test_create_logprob_power(self):
        """Test creating logprob_power score function."""
        config = ScoringConfig(
            type=ScoreType.LOGPROB_POWER,
            params=ScoringParamsConfig(tau=2.0),
        )
        score_fn = create_score_function(config)
        
        assert isinstance(score_fn, LogprobPowerScore)
        assert score_fn.tau == 2.0
    
    def test_create_logprob_avg(self):
        """Test creating logprob_avg score function."""
        config = ScoringConfig(
            type=ScoreType.LOGPROB_AVG,
            params=ScoringParamsConfig(tau=1.5),
        )
        score_fn = create_score_function(config)
        
        assert isinstance(score_fn, LogprobAvgScore)
    
    def test_create_prm_reward(self):
        """Test creating prm_reward score function."""
        config = ScoringConfig(
            type=ScoreType.PRM_REWARD,
            params=ScoringParamsConfig(prm_scale=2.0),
        )
        score_fn = create_score_function(config)
        
        assert isinstance(score_fn, PRMRewardScore)
    
    def test_create_hybrid(self):
        """Test creating hybrid score function."""
        config = ScoringConfig(
            type=ScoreType.HYBRID_PRM_LOGPROB,
            params=ScoringParamsConfig(prm_scale=1.0, logprob_scale=0.1),
        )
        score_fn = create_score_function(config)
        
        assert isinstance(score_fn, HybridPRMLogprobScore)


class TestBatchScoring:
    """Tests for batch scoring."""
    
    def test_compute_batch(self):
        """Test batch score computation."""
        score_fn = LogprobPowerScore(tau=1.0)
        
        data_list = [
            ParticleScoreData(cum_logprob=-10.0, length=100),
            ParticleScoreData(cum_logprob=-20.0, length=200),
            ParticleScoreData(cum_logprob=-30.0, length=300),
        ]
        
        scores = score_fn.compute_batch(data_list)
        
        assert len(scores) == 3
        assert scores[0] == pytest.approx(-10.0, rel=1e-10)
        assert scores[1] == pytest.approx(-20.0, rel=1e-10)
        assert scores[2] == pytest.approx(-30.0, rel=1e-10)
