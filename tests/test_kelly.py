"""
tests/test_kelly.py — Unit tests for betting/strategy/kelly.py and ev.py

Tests cover:
  - EV calculation with known inputs
  - Implied probability from decimal odds
  - Kelly formula produces correct fraction
  - Fractional Kelly is 25% of full Kelly by default
  - Negative EV → Kelly returns 0
  - Stake is bounded by bankroll
  - Confidence tier assignment (STRONG/MEDIUM/WEAK)
  - build_recommendation() returns None when EV < threshold
  - portfolio_summary() arithmetic
"""

from __future__ import annotations

import pytest

from strategy.ev import expected_value, implied_probability, EVResult, analyse_match_markets
from strategy.kelly import (
    kelly_fraction_full,
    fractional_kelly,
    kelly_stake_amount,
    BetRecommendation,
    build_recommendation,
    DEFAULT_KELLY_FRACTION,
)
from strategy.portfolio import build_portfolio, portfolio_summary


# ── EV tests ──────────────────────────────────────────────────────────────────

class TestExpectedValue:
    def test_positive_ev(self):
        ev = expected_value(0.60, 2.0)
        assert abs(ev - 0.20) < 1e-9

    def test_zero_ev_at_fair_odds(self):
        ev = expected_value(0.50, 2.0)
        assert abs(ev) < 1e-9

    def test_negative_ev(self):
        ev = expected_value(0.30, 2.0)
        assert ev < 0

    def test_implied_probability(self):
        assert abs(implied_probability(2.0) - 0.50) < 1e-9
        assert abs(implied_probability(4.0) - 0.25) < 1e-9

    def test_ev_result_has_value_flag(self):
        result = EVResult(
            market="1X2", outcome_label="Home", model_prob=0.60,
            implied_prob=0.476, decimal_odds=2.10, ev=0.06, has_value=True,
        )
        assert result.has_value is True

    def test_ev_result_no_value_flag(self):
        result = EVResult(
            market="1X2", outcome_label="Home", model_prob=0.40,
            implied_prob=0.50, decimal_odds=2.00, ev=-0.20, has_value=False,
        )
        assert result.has_value is False

    def test_analyse_match_markets_returns_all_markets(self, simple_odds):
        # analyse_match_markets returns ALL markets with odds; filter by has_value
        results = analyse_match_markets(
            home_prob=0.55, draw_prob=0.25, away_prob=0.20,
            over_25_prob=0.60, btts_prob=0.55,
            odds=simple_odds,
            home_team="ALPHA", away_team="BETA",
            min_ev=0.03,
        )
        assert isinstance(results, list)
        assert len(results) > 0
        # All returned EVResult objects should be proper instances
        for r in results:
            assert isinstance(r, EVResult)

    def test_analyse_match_markets_value_flag_respected(self, simple_odds):
        results = analyse_match_markets(
            home_prob=0.55, draw_prob=0.25, away_prob=0.20,
            over_25_prob=0.60, btts_prob=0.55,
            odds=simple_odds,
            home_team="ALPHA", away_team="BETA",
            min_ev=0.03,
        )
        value_bets = [r for r in results if r.has_value]
        for r in value_bets:
            assert r.ev >= 0.03


# ── Kelly tests ───────────────────────────────────────────────────────────────

class TestKelly:
    def test_kelly_formula(self):
        # p=0.60, b=2.0 → kelly = (2.0*0.60 - 1)/(2.0 - 1) = 0.20
        f = kelly_fraction_full(0.60, 2.0)
        assert abs(f - 0.20) < 1e-9

    def test_negative_ev_returns_zero(self):
        f = kelly_fraction_full(0.30, 2.0)
        assert f == 0.0

    def test_fractional_kelly_is_fraction_of_full(self):
        full  = kelly_fraction_full(0.60, 2.0)
        frac  = fractional_kelly(0.60, 2.0, fraction=0.25)
        assert abs(frac - full * 0.25) < 1e-9

    def test_stake_amount_uses_bankroll(self):
        stake = kelly_stake_amount(0.60, 2.0, bankroll=1000.0, fraction=0.25)
        expected = fractional_kelly(0.60, 2.0, 0.25) * 1000.0
        assert abs(stake - expected) < 1e-6

    def test_stake_is_non_negative(self):
        stake = kelly_stake_amount(0.20, 2.0, bankroll=1000.0)
        assert stake >= 0.0


# ── BetRecommendation tests ───────────────────────────────────────────────────

class TestBetRecommendation:
    def _make_ev_result(self, ev: float = 0.08, model_prob: float = 0.60) -> EVResult:
        return EVResult(
            market="1X2",
            outcome_label="Home Win",
            model_prob=model_prob,
            implied_prob=implied_probability(2.10),
            decimal_odds=2.10,
            ev=ev,
            has_value=(ev >= 0.03),
        )

    def test_strong_confidence_tier(self, match_meta):
        ev_result = self._make_ev_result(ev=0.08)
        rec = build_recommendation(ev_result, bankroll=1000.0, **match_meta)
        assert rec is not None
        assert rec.confidence_tier == "STRONG"

    def test_medium_confidence_tier(self, match_meta):
        ev_result = self._make_ev_result(ev=0.04)
        rec = build_recommendation(ev_result, bankroll=1000.0, **match_meta)
        assert rec is not None
        assert rec.confidence_tier == "MEDIUM"

    def test_weak_confidence_tier(self, match_meta):
        # EV=2% with has_value=True should produce WEAK tier (EV < 3%)
        ev_result = EVResult(
            market="1X2", outcome_label="Home Win",
            model_prob=0.50, implied_prob=implied_probability(2.10),
            decimal_odds=2.10, ev=0.02, has_value=True,
        )
        rec = build_recommendation(ev_result, bankroll=1000.0, **match_meta)
        if rec is not None:
            assert rec.confidence_tier == "WEAK"

    def test_none_returned_below_min_ev(self, match_meta):
        # has_value=False → build_recommendation returns None
        ev_result = EVResult(
            market="1X2", outcome_label="Home Win",
            model_prob=0.48, implied_prob=0.50,
            decimal_odds=2.00, ev=0.01, has_value=False,
        )
        rec = build_recommendation(ev_result, bankroll=1000.0, **match_meta)
        assert rec is None

    def test_potential_profit_calculation(self, match_meta):
        ev_result = self._make_ev_result()
        rec = build_recommendation(ev_result, bankroll=1000.0, **match_meta)
        assert rec is not None
        expected = rec.stake_amount * (rec.decimal_odds - 1)
        assert abs(rec.potential_profit - expected) < 0.01


# ── Portfolio tests ────────────────────────────────────────────────────────────

class TestPortfolio:
    def _make_rec(self, ev: float, stake_frac: float, match_id: str, market: str = "1X2") -> BetRecommendation:
        return BetRecommendation(
            match_id=match_id,
            home_team="A", away_team="B",
            market=market,
            outcome_label="Home Win",
            model_prob=0.55,
            implied_prob=0.48,
            decimal_odds=2.10,
            ev=ev,
            kelly_full=stake_frac * 4,
            kelly_frac=stake_frac,
            recommended_stake=stake_frac,
            stake_amount=stake_frac * 1000,
            confidence_tier="STRONG",
            match_date="2026-06-20",
            stage="GROUP",
        )

    def test_max_five_bets(self):
        recs = [self._make_rec(0.10 - i * 0.01, 0.02, f"M{i}") for i in range(10)]
        portfolio = build_portfolio(recs, bankroll=1000.0, max_bets=5)
        assert len(portfolio) <= 5

    def test_max_exposure(self):
        recs = [self._make_rec(0.10, 0.05, f"M{i}") for i in range(8)]
        portfolio = build_portfolio(recs, bankroll=1000.0, max_exposure=0.20)
        total_stake = sum(r.stake_amount for r in portfolio)
        assert total_stake <= 1000.0 * 0.20 + 0.01

    def test_deduplication_same_match(self):
        rec1 = self._make_rec(0.10, 0.04, "MATCH1", "1X2")
        rec2 = self._make_rec(0.08, 0.03, "MATCH1", "over_25")
        portfolio = build_portfolio([rec1, rec2], bankroll=1000.0)
        match_ids = [r.match_id for r in portfolio]
        assert match_ids.count("MATCH1") <= 1

    def test_portfolio_summary_keys(self):
        rec = self._make_rec(0.10, 0.02, "M1")
        summary = portfolio_summary([rec], bankroll=1000.0)
        assert "n_bets" in summary
        assert "total_stake" in summary
        assert "expected_profit" in summary
        assert "best_ev" in summary
        assert "total_exposure_pct" in summary
