"""
tests/test_elo.py — Unit tests for betting/models/elo.py

Tests cover:
  - ELO update direction (winner gains, loser loses)
  - Win probability order (stronger team should have higher win%)
  - Draw probability shape (peaks at ~0 ELO diff, decays with gap)
  - Probability triplet sums to 1.0
  - Neutral venue vs home advantage
  - train_on_history() modifies ratings
"""

from __future__ import annotations

import pytest
from models.elo import EloSystem, HOME_ADVANTAGE_ELO, INITIAL_RATING


def test_initial_rating_unknown_team():
    elo = EloSystem()
    assert elo.get_rating("NEWTEAM") == INITIAL_RATING


def test_win_updates_ratings_correctly():
    elo = EloSystem()
    elo.ratings["STRONG"] = 1800.0
    elo.ratings["WEAK"]   = 1400.0

    before_strong = elo.get_rating("STRONG")
    before_weak   = elo.get_rating("WEAK")

    elo.update("STRONG", "WEAK", 2, 0, is_wc=True, is_neutral=True)

    # Strong team wins → small gain (was expected to win)
    # Weak team loses → relatively small loss
    assert elo.get_rating("STRONG") > before_strong
    assert elo.get_rating("WEAK")   < before_weak


def test_upset_causes_larger_rating_change():
    elo = EloSystem()
    elo.ratings["FAVOURITE"] = 1800.0
    elo.ratings["UNDERDOG"]  = 1400.0

    elo_fav_before = elo.get_rating("FAVOURITE")
    elo_under_before = elo.get_rating("UNDERDOG")

    elo.update("UNDERDOG", "FAVOURITE", 1, 0, is_wc=True, is_neutral=True)

    gain_underdog = elo.get_rating("UNDERDOG") - elo_under_before
    loss_favourite = elo_fav_before - elo.get_rating("FAVOURITE")
    # Upset → large swings
    assert gain_underdog > 20
    assert loss_favourite > 20


def test_win_probability_stronger_team():
    elo = EloSystem()
    elo.ratings["TOP"]    = 2000.0
    elo.ratings["BOTTOM"] = 1500.0

    home_p, draw_p, away_p = elo.win_probability("TOP", "BOTTOM", is_neutral=True)

    assert home_p > away_p, "Stronger team should have higher win probability"
    assert home_p > 0.5


def test_probabilities_sum_to_one():
    elo = EloSystem()
    elo.ratings["A"] = 1750.0
    elo.ratings["B"] = 1650.0

    for neutral in (True, False):
        h, d, a = elo.win_probability("A", "B", is_neutral=neutral)
        assert abs(h + d + a - 1.0) < 1e-9, "Probabilities must sum to 1"
        assert all(0 <= p <= 1 for p in (h, d, a))


def test_draw_probability_peaks_near_zero_diff():
    elo = EloSystem()
    elo.ratings["X"] = 1700.0
    elo.ratings["Y"] = 1700.0
    elo.ratings["Z"] = 2200.0

    _, d_close, _ = elo.win_probability("X", "Y", is_neutral=True)
    _, d_far, _   = elo.win_probability("X", "Z", is_neutral=True)

    assert d_close > d_far, "Draw prob should be higher when ELOs are close"


def test_home_advantage():
    elo = EloSystem()
    elo.ratings["HOME"] = 1700.0
    elo.ratings["AWAY"] = 1700.0

    h_neutral, _, _ = elo.win_probability("HOME", "AWAY", is_neutral=True)
    h_home, _, _    = elo.win_probability("HOME", "AWAY", is_neutral=False)

    assert h_home > h_neutral, "Home team should have higher win prob at home"


def test_train_on_history_changes_ratings(synthetic_matches):
    elo = EloSystem()
    initial = {t: INITIAL_RATING for t in ["ALPHA", "BETA", "GAMMA"]}

    elo.train_on_history(synthetic_matches)

    # After training, at least some teams should differ from initial 1500
    changed = sum(
        1 for team in ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON"]
        if abs(elo.get_rating(team) - INITIAL_RATING) > 10
    )
    assert changed > 0


def test_trained_elo_top_n(trained_elo):
    top = trained_elo.top_n(5)
    assert len(top) == 5
    ratings = [r for _, r in top]
    assert ratings == sorted(ratings, reverse=True)


def test_elo_ratings_df(trained_elo):
    df = trained_elo.all_ratings_df()
    assert "team" in df.columns
    assert "elo" in df.columns
    assert len(df) >= 5
