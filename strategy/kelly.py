"""
strategy/kelly.py — Kelly Criterion Stake Sizing
=================================================
The Kelly Criterion calculates the optimal fraction of your bankroll to bet
so that long-run wealth growth is maximised.

Formula (for a binary bet):
    f* = (b·p − q) / b
       = (decimal_odds · model_prob − 1) / (decimal_odds − 1)

Where:
    b = decimal_odds − 1   (net profit per unit staked if you win)
    p = model probability of winning
    q = 1 − p              (probability of losing)

WHY FRACTIONAL KELLY?
    Full Kelly is theoretically optimal but practically dangerous:
    - It assumes perfect probability estimates (ours are never perfect)
    - It can suggest betting 30-50% of bankroll on a single bet
    - A string of losses at full Kelly causes severe drawdowns

    We default to 25% (quarter) Kelly as a proven compromise between
    growth and risk management. You can adjust this in the sidebar.

MINIMUM STAKE RULE:
    Bets below 1% of bankroll are ignored (transaction costs + noise).

HOW TO DEBUG:
    - Kelly returns 0.0 when EV ≤ 0 (bookmaker has the edge)
    - Very high Kelly fractions (>30%) usually signal either huge EV
      or a model bug — inspect the probabilities and odds carefully
    - The fractional Kelly stake should be ≤ full Kelly stake always
"""

from __future__ import annotations

from dataclasses import dataclass, field

from strategy.ev import EVResult, expected_value

# ── Configuration constants ───────────────────────────────────────────────────
DEFAULT_KELLY_FRACTION: float = 0.25  # 25% of full Kelly (conservative)
MIN_KELLY_THRESHOLD: float = 0.01     # Ignore bets under 1% of bankroll


def kelly_fraction_full(model_prob: float, decimal_odds: float) -> float:
    """Compute the FULL Kelly fraction (uncapped).

    Returns 0.0 if the bet has negative or zero expected value.

    Parameters
    ----------
    model_prob   : float — our probability estimate (0.0–1.0)
    decimal_odds : float — decimal odds offered by bookmaker (> 1.0)

    Returns
    -------
    float — fraction of bankroll to bet (0.0 = no bet)

    Examples
    --------
    >>> kelly_fraction_full(0.60, 2.00)   # 60% chance, evens → Kelly = 0.20
    0.2
    >>> kelly_fraction_full(0.40, 2.00)   # 40% chance, evens → no value
    0.0
    """
    if model_prob <= 0.0 or decimal_odds <= 1.0:
        return 0.0

    # b = net profit per unit won; q = probability of losing
    b = decimal_odds - 1.0
    q = 1.0 - model_prob

    kelly = (b * model_prob - q) / b

    # Negative Kelly means negative EV — do not bet
    return max(0.0, kelly)


def fractional_kelly(
    model_prob: float,
    decimal_odds: float,
    fraction: float = DEFAULT_KELLY_FRACTION,
) -> float:
    """Compute the fractional Kelly stake (as a proportion of bankroll).

    This is the primary function used by the UI and portfolio optimizer.

    Parameters
    ----------
    model_prob   : float — probability estimate (0–1)
    decimal_odds : float — bookmaker decimal odds (> 1)
    fraction     : float — Kelly fraction to apply (default 0.25 = 25%)

    Returns
    -------
    float — recommended proportion of bankroll to stake (0–1)

    Examples
    --------
    >>> fractional_kelly(0.60, 2.00, fraction=0.25)   # 25% of 0.20 = 0.05
    0.05
    """
    full = kelly_fraction_full(model_prob, decimal_odds)
    return fraction * full


def kelly_stake_amount(
    model_prob: float,
    decimal_odds: float,
    bankroll: float,
    fraction: float = DEFAULT_KELLY_FRACTION,
) -> float:
    """Return the recommended stake in currency units.

    Parameters
    ----------
    model_prob   : float — probability estimate
    decimal_odds : float — bookmaker decimal odds
    bankroll     : float — current total bankroll (e.g. 1000.0)
    fraction     : float — Kelly multiplier (default 0.25)

    Returns
    -------
    float — stake in same currency as bankroll; 0.0 if below MIN_KELLY_THRESHOLD

    Examples
    --------
    >>> kelly_stake_amount(0.60, 2.00, bankroll=1000.0, fraction=0.25)
    50.0   # 5% of £1000 = £50
    """
    frac = fractional_kelly(model_prob, decimal_odds, fraction)

    # Skip bets that are too small to matter
    if frac < MIN_KELLY_THRESHOLD:
        return 0.0

    return round(frac * bankroll, 2)


@dataclass
class BetRecommendation:
    """A fully-analysed bet recommendation for a single market.

    This is the primary data structure flowing from strategy → UI.
    All monetary fields are in the same currency as the bankroll input.

    Attributes
    ----------
    match_id         : Unique key, e.g. "ENG_vs_USA_2026-06-20"
    home_team        : str
    away_team        : str
    market           : One of "home" | "draw" | "away" | "over_25" | "btts"
    outcome_label    : Human-readable, e.g. "England Win" or "Over 2.5 Goals"
    model_prob       : Our model's probability estimate
    implied_prob     : Bookmaker's implied probability (1 / decimal_odds)
    decimal_odds     : Odds as offered by bookmaker
    ev               : Expected value per unit staked (positive = value)
    kelly_full       : Full Kelly fraction (theoretical maximum)
    kelly_frac       : Applied fractional Kelly (what we actually recommend)
    recommended_stake: Stake as a fraction of bankroll (e.g. 0.05 = 5%)
    stake_amount     : Stake in currency units
    confidence_tier  : "STRONG" (EV ≥ 5%) | "MEDIUM" (EV ≥ 3%) | "WEAK" (< 3%)
    """

    match_id: str
    home_team: str
    away_team: str
    market: str
    outcome_label: str
    model_prob: float
    implied_prob: float
    decimal_odds: float
    ev: float
    kelly_full: float
    kelly_frac: float
    recommended_stake: float   # fraction of bankroll
    stake_amount: float        # currency units
    confidence_tier: str       # "STRONG" | "MEDIUM" | "WEAK"
    match_date: str = ""
    stage: str = ""

    @property
    def potential_profit(self) -> float:
        """Profit if the bet wins (stake × net odds)."""
        return round(self.stake_amount * (self.decimal_odds - 1.0), 2)

    @property
    def ev_pct_str(self) -> str:
        """EV as a coloured string for display, e.g. '+8.4%'."""
        sign = "+" if self.ev >= 0 else ""
        return f"{sign}{self.ev * 100:.1f}%"

    @property
    def edge_pct_str(self) -> str:
        """Edge (model_prob − implied_prob) as a percentage string."""
        edge = self.model_prob - self.implied_prob
        sign = "+" if edge >= 0 else ""
        return f"{sign}{edge * 100:.1f}%"


def build_recommendation(
    ev_result: EVResult,
    match_id: str,
    home_team: str,
    away_team: str,
    bankroll: float,
    kelly_fraction: float = DEFAULT_KELLY_FRACTION,
    match_date: str = "",
    stage: str = "",
) -> BetRecommendation | None:
    """Convert an EVResult into a BetRecommendation with stake sizing.

    Returns None if the bet has no value or stake is below minimum.

    HOW TO DEBUG:
        - Returns None when ev_result.ev < MIN_EV_THRESHOLD
        - Returns None when the Kelly stake is below MIN_KELLY_THRESHOLD
        - Increase MIN_EV_THRESHOLD to see fewer, higher-confidence bets
    """
    # Only recommend bets with positive EV above threshold
    if not ev_result.has_value:
        return None

    full_kelly = kelly_fraction_full(ev_result.model_prob, ev_result.decimal_odds)
    frac_kelly = full_kelly * kelly_fraction
    stake_frac = frac_kelly

    # Skip bets that are too small
    if stake_frac < MIN_KELLY_THRESHOLD:
        return None

    stake_amount = round(stake_frac * bankroll, 2)

    # Determine confidence tier based on EV magnitude
    if ev_result.ev >= 0.05:
        tier = "STRONG"
    elif ev_result.ev >= 0.03:
        tier = "MEDIUM"
    else:
        tier = "WEAK"

    return BetRecommendation(
        match_id=match_id,
        home_team=home_team,
        away_team=away_team,
        market=ev_result.market,
        outcome_label=ev_result.outcome_label,
        model_prob=ev_result.model_prob,
        implied_prob=ev_result.implied_prob,
        decimal_odds=ev_result.decimal_odds,
        ev=ev_result.ev,
        kelly_full=full_kelly,
        kelly_frac=frac_kelly,
        recommended_stake=stake_frac,
        stake_amount=stake_amount,
        confidence_tier=tier,
        match_date=match_date,
        stage=stage,
    )
