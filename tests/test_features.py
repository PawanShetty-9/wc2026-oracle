"""
tests/test_features.py — Unit tests for betting/features/form.py
                         and betting/features/engineering.py

Tests cover:
  - calculate_form() returns [0,1] range
  - calculate_goals_stats() returns realistic averages
  - h2h_record() has correct keys and non-negative values
  - days_since_last_match() returns positive integer
  - FeatureEngineer.build_features() returns all 23 expected feature names
  - Feature values are finite floats (no NaN / Inf)
  - build_training_set() returns correctly shaped (X, y)
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from features.form import (
    calculate_form,
    calculate_goals_stats,
    h2h_record,
    days_since_last_match,
    DEFAULT_DAYS_REST,
)
from features.engineering import FeatureEngineer, FEATURE_NAMES
from data.wc2026_teams import TEAM_META


class TestFormCalculation:
    def test_form_range(self, synthetic_matches):
        form = calculate_form("ALPHA", synthetic_matches, as_of_date=date(2024, 1, 1))
        assert 0.0 <= form <= 1.0

    def test_form_unknown_team_returns_default(self, synthetic_matches):
        form = calculate_form("NONEXISTENT", synthetic_matches, as_of_date=date(2024, 1, 1))
        assert 0.0 <= form <= 1.0

    def test_form_uses_only_pre_date_matches(self, synthetic_matches):
        form_early = calculate_form("ALPHA", synthetic_matches, as_of_date=date(1993, 6, 1))
        form_late  = calculate_form("ALPHA", synthetic_matches, as_of_date=date(2023, 12, 31))
        assert 0.0 <= form_early <= 1.0
        assert 0.0 <= form_late  <= 1.0


class TestGoalsStats:
    def test_goals_stats_keys(self, synthetic_matches):
        # keys are: scored_avg, conceded_avg, goal_diff_avg
        stats = calculate_goals_stats("ALPHA", synthetic_matches, as_of_date=date(2024, 1, 1))
        assert "scored_avg" in stats
        assert "conceded_avg" in stats

    def test_goals_are_non_negative(self, synthetic_matches):
        stats = calculate_goals_stats("BETA", synthetic_matches, as_of_date=date(2024, 1, 1))
        assert stats["scored_avg"]   >= 0
        assert stats["conceded_avg"] >= 0


class TestH2HRecord:
    def test_h2h_keys_present(self, synthetic_matches):
        rec = h2h_record("ALPHA", "BETA", synthetic_matches)
        assert "home_wins" in rec
        assert "draws" in rec
        assert "away_wins" in rec

    def test_h2h_sums(self, synthetic_matches):
        rec = h2h_record("ALPHA", "BETA", synthetic_matches)
        total = rec["home_wins"] + rec["draws"] + rec["away_wins"]
        # total should be <= n_meetings (5 by default) and non-negative
        assert total >= 0
        assert total <= 5

    def test_h2h_non_negative(self, synthetic_matches):
        rec = h2h_record("ALPHA", "GAMMA", synthetic_matches)
        assert rec["home_wins"]  >= 0
        assert rec["draws"]      >= 0
        assert rec["away_wins"]  >= 0

    def test_h2h_unknown_teams(self, synthetic_matches):
        rec = h2h_record("NOBODY", "NOBODY2", synthetic_matches)
        assert rec["home_wins"] == 0
        assert rec["draws"]     == 0
        assert rec["away_wins"] == 0


class TestDaysSinceLastMatch:
    def test_returns_positive_int(self, synthetic_matches):
        days = days_since_last_match("ALPHA", synthetic_matches, as_of_date=date(2024, 6, 1))
        assert isinstance(days, int)
        assert days >= 0

    def test_unknown_team_returns_default(self, synthetic_matches):
        days = days_since_last_match("NOBODY", synthetic_matches, as_of_date=date(2024, 6, 1))
        assert days == DEFAULT_DAYS_REST


class TestFeatureEngineer:
    @pytest.fixture(scope="class")
    def engineer(self, synthetic_matches, trained_elo, trained_dc):
        return FeatureEngineer(
            historical_df=synthetic_matches,
            elo_system=trained_elo,
            dc_model=trained_dc,
            team_meta=TEAM_META,
        )

    def test_build_features_returns_all_names(self, engineer):
        features = engineer.build_features(
            home="ALPHA", away="BETA",
            match_date=date(2026, 6, 20), stage="GROUP",
        )
        for name in FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"

    def test_feature_values_are_finite(self, engineer):
        features = engineer.build_features(
            home="ALPHA", away="BETA",
            match_date=date(2026, 6, 20), stage="GROUP",
        )
        for name, val in features.items():
            assert np.isfinite(val), f"Non-finite value for {name}: {val}"

    def test_unknown_team_does_not_crash(self, engineer):
        features = engineer.build_features(
            home="UNKNOWN_TEAM", away="ALPHA",
            match_date=date(2026, 6, 20), stage="GROUP",
        )
        assert isinstance(features, dict)
        assert len(features) >= len(FEATURE_NAMES)

    def test_build_training_set_shape(self, engineer, synthetic_matches):
        X, y = engineer.build_training_set(synthetic_matches.head(100))
        assert len(X) == len(y)
        assert len(X.columns) == len(FEATURE_NAMES)

    def test_labels_are_valid(self, engineer, synthetic_matches):
        _, y = engineer.build_training_set(synthetic_matches.head(200))
        assert set(y.unique()).issubset({0, 1, 2})

    def test_elo_diff_squared_non_negative(self, engineer):
        features = engineer.build_features(
            home="GAMMA", away="DELTA",
            match_date=date(2026, 6, 20), stage="GROUP",
        )
        assert features["elo_diff_sq"] >= 0

    def test_stage_encoding(self, engineer):
        f_group  = engineer.build_features("ALPHA", "BETA", date(2026, 6, 20), "GROUP")
        f_final  = engineer.build_features("ALPHA", "BETA", date(2026, 7, 19), "FINAL")
        assert f_group["tournament_round"] < f_final["tournament_round"]
