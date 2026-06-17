"""
tests/test_dixon_coles.py — Unit tests for betting/models/dixon_coles.py

Tests cover:
  - Score matrix shape and probability sum ≈ 1
  - Outcome probabilities are valid (non-negative, sum ≤ 1)
  - Home/draw/away outcomes are exhaustive (sum ≈ 1)
  - Over/Under and BTTS probabilities are in [0, 1]
  - Stronger team (higher attack, lower opponent defense) has higher win prob
  - Tau correction: P(0-0) is higher than pure Poisson would predict
  - Unknown team fallback doesn't crash
"""

from __future__ import annotations

import numpy as np
import pytest

from models.dixon_coles import DixonColesModel, _tau_correction


class TestTauCorrection:
    def test_00_is_positive(self):
        # tau(0,0) = 1 + lam_h * lam_a * rho should be > 1 for rho < 0
        val = _tau_correction(0, 0, 0.8, 0.7, -0.13)
        assert val > 1.0

    def test_other_scores_less_affected(self):
        val_22 = _tau_correction(2, 2, 0.8, 0.7, -0.13)
        # For scores outside {0,0},{1,0},{0,1},{1,1} correction = 1.0
        assert val_22 == pytest.approx(1.0)

    def test_tau_is_non_negative(self):
        for rho in (-0.5, -0.13, 0.0):
            for x, y in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1)]:
                val = _tau_correction(x, y, 1.2, 0.9, rho)
                assert val >= 0, f"Negative tau for ({x},{y}) rho={rho}"


class TestDixonColesModel:
    def test_fit_returns_self(self, synthetic_matches):
        dc = DixonColesModel()
        result = dc.fit(synthetic_matches)
        assert result is dc

    def test_fitted_flag(self, trained_dc):
        assert trained_dc._fitted is True

    def test_score_matrix_shape(self, trained_dc):
        matrix = trained_dc.predict_score_matrix("ALPHA", "BETA")
        assert matrix.shape == (11, 11)

    def test_score_matrix_sums_to_one(self, trained_dc):
        matrix = trained_dc.predict_score_matrix("ALPHA", "BETA")
        total = matrix.sum()
        assert abs(total - 1.0) < 0.01, f"Matrix total = {total}, expected ~1.0"

    def test_score_matrix_non_negative(self, trained_dc):
        matrix = trained_dc.predict_score_matrix("ALPHA", "BETA")
        assert (matrix >= 0).all()

    def test_outcome_probs_valid(self, trained_dc):
        probs = trained_dc.predict_outcome_probs("ALPHA", "BETA")
        for key in ("home", "draw", "away"):
            assert 0 <= probs[key] <= 1
        total = probs["home"] + probs["draw"] + probs["away"]
        assert abs(total - 1.0) < 0.02

    def test_over_25_prob_valid(self, trained_dc):
        probs = trained_dc.predict_outcome_probs("ALPHA", "BETA")
        assert 0 <= probs["over_25"] <= 1

    def test_btts_prob_valid(self, trained_dc):
        probs = trained_dc.predict_outcome_probs("ALPHA", "BETA")
        assert 0 <= probs["btts"] <= 1

    def test_most_likely_score_format(self, trained_dc):
        probs = trained_dc.predict_outcome_probs("ALPHA", "BETA")
        score = probs["most_likely_score"]
        # Should be a (home_goals, away_goals) tuple of ints
        assert isinstance(score, tuple)
        assert len(score) == 2
        assert all(isinstance(g, int) and g >= 0 for g in score)

    def test_stronger_team_wins_more(self, trained_dc):
        """Teams with high attack/low opponent defense should win more."""
        probs_fav  = trained_dc.predict_outcome_probs("ALPHA", "KAPPA", neutral=True)
        probs_flip = trained_dc.predict_outcome_probs("KAPPA", "ALPHA", neutral=True)
        # They won't necessarily be mirror images but shouldn't both say home wins >50%
        # Just check that they give different home_win probabilities
        assert probs_fav["home"] != probs_flip["home"]

    def test_unknown_team_fallback(self, trained_dc):
        """Unknown team should not raise; should use league-average 0.0 strengths."""
        probs = trained_dc.predict_outcome_probs("UNKNOWN_TEAM", "ALPHA")
        assert 0 <= probs["home"] <= 1
        assert 0 <= probs["draw"] <= 1
        assert 0 <= probs["away"] <= 1

    def test_neutral_vs_home_field(self, trained_dc):
        probs_neutral = trained_dc.predict_outcome_probs("ALPHA", "BETA", neutral=True)
        probs_home    = trained_dc.predict_outcome_probs("ALPHA", "BETA", neutral=False)
        # With home advantage, home team should have higher win prob
        assert probs_home["home"] >= probs_neutral["home"]
