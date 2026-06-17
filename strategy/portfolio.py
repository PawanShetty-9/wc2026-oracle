"""
strategy/portfolio.py — Multi-Bet Portfolio Optimisation
=========================================================
Manages the set of active bet recommendations to ensure:
  1. Total bankroll exposure stays below a safe maximum (default 20%)
  2. Correlated bets on the same match are de-duplicated
  3. The number of simultaneous open bets is capped (default 5)
  4. Stakes are adjusted down if total exposure would exceed the cap

WHY PORTFOLIO MANAGEMENT MATTERS:
    Kelly Criterion optimises ONE bet in isolation. When betting on multiple
    simultaneous matches, the stakes should be reduced because:
    - Results on different matches can correlate (e.g. two group-stage matches
      at the same time may both go to underdogs in an "upset day")
    - A simultaneous losing streak can wipe out more bankroll than Kelly assumes

A conservative portfolio approach sacrifices some mathematical optimality
in exchange for much better risk-adjusted returns in practice.

HOW TO DEBUG:
    - If no bets are returned, MAX_TOTAL_EXPOSURE may be set too low
    - If stakes seem too small, increase MAX_BETS or MAX_TOTAL_EXPOSURE
    - Check that incoming BetRecommendation.match_id is consistent
      (same format: "HOME_vs_AWAY_DATE") to detect correlated bets correctly
"""

from __future__ import annotations

from strategy.kelly import BetRecommendation

# ── Portfolio constraints (editable) ─────────────────────────────────────────
MAX_TOTAL_EXPOSURE: float = 0.20   # max 20% of bankroll at risk simultaneously
MAX_BETS: int = 5                  # max simultaneous open bets
# Minimum EV to include even after portfolio filtering
MIN_PORTFOLIO_EV: float = 0.03


def build_portfolio(
    recommendations: list[BetRecommendation],
    bankroll: float,
    max_exposure: float = MAX_TOTAL_EXPOSURE,
    max_bets: int = MAX_BETS,
) -> list[BetRecommendation]:
    """Select and size the best bets under portfolio constraints.

    Algorithm
    ---------
    1. Filter out WEAK tier bets (EV < 3%).
    2. Sort remaining bets by EV descending (highest edge first).
    3. Remove correlated bets: for each match, keep only the single
       highest-EV recommendation (e.g. if "France Win" and "Draw" for the
       same France vs Argentina match both have value, only keep the top one).
    4. Greedily add bets until either MAX_BETS or MAX_TOTAL_EXPOSURE is hit.
    5. If total exposure would exceed the cap, scale the last bet's stake down.

    Parameters
    ----------
    recommendations : list[BetRecommendation] — output from build_recommendation()
    bankroll        : float — current bankroll
    max_exposure    : float — fraction of bankroll cap (e.g. 0.20)
    max_bets        : int   — hard limit on simultaneous bets

    Returns
    -------
    list[BetRecommendation] — approved bets with (potentially adjusted) stakes

    HOW TO DEBUG:
        - Add print(f"Excluded {r.outcome_label}: correlated") to see removals
        - Set max_exposure=1.0 to see all bets without the exposure cap
    """
    # Step 1: Filter out bets below portfolio minimum EV
    candidates = [r for r in recommendations if r.ev >= MIN_PORTFOLIO_EV]

    # Step 2: Sort by EV descending
    candidates.sort(key=lambda r: r.ev, reverse=True)

    # Step 3: De-duplicate correlated bets (same match_id → keep highest EV only)
    seen_matches: set[str] = set()
    deduplicated: list[BetRecommendation] = []
    for rec in candidates:
        if rec.match_id not in seen_matches:
            seen_matches.add(rec.match_id)
            deduplicated.append(rec)

    # Step 4: Greedy selection under exposure cap
    portfolio: list[BetRecommendation] = []
    total_exposure: float = 0.0

    for rec in deduplicated:
        if len(portfolio) >= max_bets:
            break

        remaining_capacity = max_exposure - total_exposure
        if remaining_capacity <= 0.001:
            break

        if rec.recommended_stake <= remaining_capacity:
            # Bet fits entirely within remaining capacity
            portfolio.append(rec)
            total_exposure += rec.recommended_stake
        else:
            # Step 5: Scale the stake down to fit within remaining capacity
            scaled = _scale_stake(rec, remaining_capacity, bankroll)
            if scaled is not None:
                portfolio.append(scaled)
                total_exposure += scaled.recommended_stake
            break

    return portfolio


def _scale_stake(
    rec: BetRecommendation,
    max_stake_fraction: float,
    bankroll: float,
) -> BetRecommendation | None:
    """Return a copy of rec with the stake reduced to max_stake_fraction.

    Returns None if the scaled stake is too small to be meaningful (< 0.5%).
    """
    if max_stake_fraction < 0.005:  # below 0.5% — not worth betting
        return None

    # Create a new dataclass instance with adjusted stakes
    # (dataclasses are immutable-ish — we use a workaround here)
    import dataclasses
    scaled = dataclasses.replace(
        rec,
        recommended_stake=max_stake_fraction,
        stake_amount=round(max_stake_fraction * bankroll, 2),
    )
    return scaled


def parlay_ev(bets: list[BetRecommendation]) -> tuple[float, float]:
    """Compute the EV and combined odds of a multi-bet parlay (accumulator).

    A parlay multiplies all decimal odds and requires ALL bets to win.
    This is usually a bad bet (negative EV vs single bets) but some users
    ask for it. We surface it as information only, never as a recommendation.

    Returns
    -------
    (combined_ev, combined_odds) — EV per unit staked, combined decimal odds
    """
    if not bets:
        return 0.0, 1.0

    # Combined odds = product of all decimal odds
    combined_odds = 1.0
    combined_prob = 1.0
    for bet in bets:
        combined_odds *= bet.decimal_odds
        combined_prob *= bet.model_prob  # assumes independence

    combined_ev = combined_prob * combined_odds - 1.0
    return combined_ev, combined_odds


def portfolio_summary(portfolio: list[BetRecommendation], bankroll: float) -> dict:
    """Return a summary dict for display in the UI.

    Keys: total_exposure_pct, total_stake, expected_profit, n_bets,
          best_ev, avg_ev, potential_return
    """
    if not portfolio:
        return {
            "total_exposure_pct": 0.0,
            "total_stake": 0.0,
            "expected_profit": 0.0,
            "n_bets": 0,
            "best_ev": 0.0,
            "avg_ev": 0.0,
            "potential_return": 0.0,
        }

    total_stake = sum(b.stake_amount for b in portfolio)
    # Expected profit = Σ(stake × EV)
    expected_profit = sum(b.stake_amount * b.ev for b in portfolio)
    # Optimistic return = all bets win
    potential_return = sum(b.stake_amount * b.decimal_odds for b in portfolio)

    return {
        "total_exposure_pct": total_stake / bankroll * 100,
        "total_stake": total_stake,
        "expected_profit": round(expected_profit, 2),
        "n_bets": len(portfolio),
        "best_ev": max(b.ev for b in portfolio),
        "avg_ev": sum(b.ev for b in portfolio) / len(portfolio),
        "potential_return": round(potential_return, 2),
    }
