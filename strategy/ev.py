"""
strategy/ev.py — Expected Value (EV) Calculator
================================================
Expected Value is the core metric for identifying profitable bets.

Formula:
    EV = model_probability × (decimal_odds − 1) − (1 − model_probability)
       = model_probability × decimal_odds − 1

A positive EV means the bet is profitable *in expectation*. For example:
    Model says Team A wins with 60% probability.
    Bookmaker offers 2.10 decimal odds (implied prob = 1/2.10 = 47.6%).
    EV = 0.60 × 2.10 − 1 = 0.26  →  +26% return per unit staked in expectation.

We only recommend bets where EV > MIN_EV_THRESHOLD (default 3%) to account
for model uncertainty and bookmaker overround.

HOW TO DEBUG:
    - If EV is always 0, check that model_prob is not equal to 1/decimal_odds
    - Negative EV means the bookmaker's implied prob > your model prob (no edge)
    - EV > 0.30 is suspicious — double-check odds and probability inputs
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Thresholds ───────────────────────────────────────────────────────────────
# Only surface bets above this minimum edge.
# 3% is the conservative default; raise to 5% for a tighter filter.
MIN_EV_THRESHOLD: float = 0.03


def expected_value(model_prob: float, decimal_odds: float) -> float:
    """Return the expected value per unit staked.

    Parameters
    ----------
    model_prob   : float — our model's estimated win probability (0.0–1.0)
    decimal_odds : float — bookmaker decimal odds (e.g. 2.50 means £2.50 back per £1)

    Returns
    -------
    float — EV per unit staked.
        > 0 : value bet (profitable in expectation)
        = 0 : break-even
        < 0 : no value (avoid)

    Examples
    --------
    >>> expected_value(0.60, 2.10)   # model says 60%, bookie implies 47.6%
    0.26
    >>> expected_value(0.40, 2.00)   # model says 40%, bookie implies 50% → no value
    -0.2
    """
    # Validate inputs to catch common bugs
    if not (0.0 < model_prob < 1.0):
        raise ValueError(f"model_prob must be in (0, 1), got {model_prob}")
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal_odds must be > 1.0, got {decimal_odds}")

    return model_prob * decimal_odds - 1.0


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to bookmaker-implied probability (including overround).

    The bookmaker builds a margin (overround) into their odds, so the sum of
    implied probabilities across all outcomes > 1.0. This raw implied prob
    is used only for display / comparison — we do NOT use it in EV calculation.

    >>> implied_probability(2.00)
    0.5
    >>> implied_probability(3.50)
    0.2857142857142857
    """
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal_odds must be > 1.0, got {decimal_odds}")
    return 1.0 / decimal_odds


@dataclass
class EVResult:
    """Container for expected value analysis of a single bet market."""
    market: str           # e.g. "home", "draw", "away", "over_25", "btts"
    outcome_label: str    # human-readable: "France Win", "Draw", "Over 2.5 Goals"
    model_prob: float     # our model's probability
    implied_prob: float   # bookmaker's implied probability
    decimal_odds: float   # bookmaker decimal odds
    ev: float             # expected value per unit staked
    has_value: bool       # True if ev > MIN_EV_THRESHOLD

    @property
    def edge_pct(self) -> float:
        """Edge = model_prob − implied_prob (how much we're beating the market)."""
        return self.model_prob - self.implied_prob

    @property
    def ev_pct(self) -> str:
        """EV formatted as a percentage string, e.g. '+12.4%'."""
        sign = "+" if self.ev >= 0 else ""
        return f"{sign}{self.ev * 100:.1f}%"


def analyse_match_markets(
    home_prob: float,
    draw_prob: float,
    away_prob: float,
    over_25_prob: float,
    btts_prob: float,
    odds: dict[str, float],
    home_team: str = "Home",
    away_team: str = "Away",
    min_ev: float = MIN_EV_THRESHOLD,
) -> list[EVResult]:
    """Analyse all available markets for a single match.

    Parameters
    ----------
    home_prob, draw_prob, away_prob : float — ensemble model probabilities
    over_25_prob, btts_prob         : float — derived from Dixon-Coles score matrix
    odds                            : dict  — keyed by market name:
                                       {"home": 2.10, "draw": 3.40, "away": 3.20,
                                        "over_25": 1.85, "btts": 1.75}
    home_team, away_team            : str   — for labelling
    min_ev                          : float — minimum EV to flag as has_value=True

    Returns
    -------
    List[EVResult] — one entry per market that has odds available.

    HOW TO DEBUG:
        - If list is empty, check that `odds` dict has the correct keys
        - If all EV is negative, the model may be under-confident vs the market
    """
    # Map market names to (our_prob, human_label) pairs
    market_map: dict[str, tuple[float, str]] = {
        "home":     (home_prob,     f"{home_team} Win"),
        "draw":     (draw_prob,     "Draw"),
        "away":     (away_prob,     f"{away_team} Win"),
        "over_25":  (over_25_prob,  "Over 2.5 Goals"),
        "btts":     (btts_prob,     "Both Teams Score"),
    }

    results: list[EVResult] = []

    for market, decimal_odds in odds.items():
        if market not in market_map:
            continue  # unknown market key — skip gracefully

        model_p, label = market_map[market]
        if model_p <= 0.0 or decimal_odds <= 1.0:
            continue  # degenerate values — skip

        ev = expected_value(model_p, decimal_odds)
        results.append(
            EVResult(
                market=market,
                outcome_label=label,
                model_prob=model_p,
                implied_prob=implied_probability(decimal_odds),
                decimal_odds=decimal_odds,
                ev=ev,
                has_value=ev >= min_ev,
            )
        )

    # Sort by EV descending so the best bets are first
    results.sort(key=lambda r: r.ev, reverse=True)
    return results
