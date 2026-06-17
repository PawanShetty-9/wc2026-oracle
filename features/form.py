"""
features/form.py — Recent Form & Head-to-Head Calculations
===========================================================
Form features capture a team's recent momentum — something that ELO and
Dixon-Coles (which use all historical data) may be slow to pick up.

A team on a 5-match winning streak should be rated higher than their
long-term rating suggests, especially at the World Cup where squad
morale and tactical cohesion matter enormously.

WALK-FORWARD PRINCIPLE:
  All functions take an `as_of_date` parameter. They ONLY look at matches
  BEFORE that date. This prevents lookahead bias when building training
  datasets for XGBoost — a crucial correctness requirement.

  Without walk-forward validation, the model would "know" future results
  during training, leading to massively overoptimistic accuracy estimates.

HOW TO DEBUG:
  - If form_score is always 0.5 (no differentiation):
    → Check that df["home_team"] and df["away_team"] use UPPERCASE names
      matching what you're querying (team names must match exactly)
  - If h2h_record returns all zeros:
    → Teams may not have played each other in the historical dataset
    → Synthetic data uses made-up names; real names are in TEAM_META
  - days_since_last_match returning 7 (default) for all teams:
    → The as_of_date may be before any matches in df
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# Points awarded for each result (used in form score calculation)
POINTS_WIN:  float = 3.0
POINTS_DRAW: float = 1.0
POINTS_LOSS: float = 0.0
MAX_POINTS_PER_GAME = POINTS_WIN  # = 3.0

# Default rest days if no prior match is found
DEFAULT_DAYS_REST: int = 7


def calculate_form(
    team: str,
    df: pd.DataFrame,
    as_of_date: date,
    n_matches: int = 5,
) -> float:
    """Calculate normalised points-per-game form score for a team.

    Considers only the last `n_matches` matches BEFORE `as_of_date`.

    Parameters
    ----------
    team        : str  — team name (UPPERCASE, matches TEAM_META)
    df          : pd.DataFrame — historical matches with columns:
                    date (datetime or str), home_team, away_team,
                    home_score, away_score
    as_of_date  : date — only include matches before this date
    n_matches   : int  — how many recent matches to consider (default 5)

    Returns
    -------
    float — normalised form score in [0.0, 1.0]
        1.0 = won all n_matches
        0.5 = drew all (roughly)
        0.0 = lost all n_matches

    Example
    -------
    >>> form = calculate_form("BRAZIL", training_df, date(2026, 6, 18))
    >>> print(f"Brazil form (last 5): {form:.2f}")  # e.g. 0.80 = very good
    """
    recent = _get_team_matches(team, df, as_of_date, n_matches)

    if recent.empty:
        logger.debug("No matches found for %s before %s", team, as_of_date)
        return 0.5  # neutral form when no data

    points_earned = 0.0
    for _, row in recent.iterrows():
        result = _team_result(team, row)
        if result == "W":
            points_earned += POINTS_WIN
        elif result == "D":
            points_earned += POINTS_DRAW
        # Loss adds 0

    max_possible = len(recent) * MAX_POINTS_PER_GAME
    return points_earned / max_possible if max_possible > 0 else 0.5


def calculate_goals_stats(
    team: str,
    df: pd.DataFrame,
    as_of_date: date,
    n_matches: int = 10,
) -> dict[str, float]:
    """Calculate average goals scored and conceded over recent matches.

    Uses exponential weighting so the most recent matches count more.

    Parameters
    ----------
    team       : str  — team name
    df         : pd.DataFrame — historical matches
    as_of_date : date — only matches before this date
    n_matches  : int  — how many recent matches to look at (default 10)

    Returns
    -------
    dict with keys:
        scored_avg    : float — exponentially-weighted goals scored per game
        conceded_avg  : float — exponentially-weighted goals conceded per game
        goal_diff_avg : float — scored_avg − conceded_avg

    HOW TO DEBUG:
        If scored_avg = conceded_avg = 1.28 for all teams, the df
        likely contains synthetic/random data (expected λ=1.28 for both).
    """
    recent = _get_team_matches(team, df, as_of_date, n_matches)

    if recent.empty:
        return {"scored_avg": 1.2, "conceded_avg": 1.2, "goal_diff_avg": 0.0}

    # Exponential weights (most recent = highest weight)
    n = len(recent)
    weights = pd.Series([0.85 ** i for i in range(n - 1, -1, -1)])
    weights = weights / weights.sum()  # normalise

    scored: list[float] = []
    conceded: list[float] = []

    for _, row in recent.iterrows():
        if row["home_team"] == team:
            scored.append(float(row["home_score"]))
            conceded.append(float(row["away_score"]))
        else:
            scored.append(float(row["away_score"]))
            conceded.append(float(row["home_score"]))

    scored_s   = pd.Series(scored)
    conceded_s = pd.Series(conceded)

    scored_avg   = float((scored_s * weights.values).sum())
    conceded_avg = float((conceded_s * weights.values).sum())

    return {
        "scored_avg":    round(scored_avg, 3),
        "conceded_avg":  round(conceded_avg, 3),
        "goal_diff_avg": round(scored_avg - conceded_avg, 3),
    }


def h2h_record(
    home: str,
    away: str,
    df: pd.DataFrame,
    n_meetings: int = 5,
) -> dict[str, int]:
    """Return head-to-head record over the last N meetings.

    Does NOT filter by date (H2H uses all available history).
    Both directions of the fixture are included (home vs away, away vs home).

    Parameters
    ----------
    home, away  : str — team names
    df          : pd.DataFrame — historical matches
    n_meetings  : int — how many recent meetings to consider

    Returns
    -------
    dict with keys: home_wins, draws, away_wins
    (where "home" = the first team argument, not the actual home team)

    Example
    -------
    >>> h2h_record("BRAZIL", "ARGENTINA", df, n_meetings=10)
    {"home_wins": 3, "draws": 4, "away_wins": 3}
    """
    # Find all meetings between these two teams (either direction)
    mask = (
        ((df["home_team"] == home) & (df["away_team"] == away))
        | ((df["home_team"] == away) & (df["away_team"] == home))
    )
    meetings = df[mask].copy()

    if meetings.empty:
        return {"home_wins": 0, "draws": 0, "away_wins": 0}

    # Sort by date descending, take most recent N
    meetings["date"] = pd.to_datetime(meetings["date"])
    meetings = meetings.sort_values("date", ascending=False).head(n_meetings)

    home_wins = draws = away_wins = 0

    for _, row in meetings.iterrows():
        if row["home_score"] == row["away_score"]:
            draws += 1
        elif row["home_team"] == home:
            # home is playing in home position
            if row["home_score"] > row["away_score"]:
                home_wins += 1
            else:
                away_wins += 1
        else:
            # home is playing in away position
            if row["away_score"] > row["home_score"]:
                home_wins += 1
            else:
                away_wins += 1

    return {"home_wins": home_wins, "draws": draws, "away_wins": away_wins}


def days_since_last_match(
    team: str,
    df: pd.DataFrame,
    as_of_date: date,
) -> int:
    """Return number of days since the team's last match before as_of_date.

    Returns DEFAULT_DAYS_REST (7) if no prior match is found in df.
    This represents a typical international break between matches.

    Parameters
    ----------
    team       : str  — team name
    df         : pd.DataFrame — matches with 'date' column
    as_of_date : date — reference date

    Returns
    -------
    int — days of rest (minimum 0, default 7 if unknown)
    """
    team_matches = _get_team_matches(team, df, as_of_date, n=1)

    if team_matches.empty:
        return DEFAULT_DAYS_REST

    last_date = pd.to_datetime(team_matches["date"].iloc[-1]).date()
    days = (as_of_date - last_date).days
    return max(0, days)


# ─── Private helpers ─────────────────────────────────────────────────────────

def _get_team_matches(
    team: str,
    df: pd.DataFrame,
    before_date: date,
    n: int,
) -> pd.DataFrame:
    """Return up to n most recent matches for a team before before_date.

    Results are sorted in ASCENDING date order (oldest first within the
    returned window). Assumes df is already sorted ascending by date with
    datetime dtype (FeatureEngineer pre-processes this once at init time).
    """
    cutoff = pd.Timestamp(before_date)
    mask = (
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < cutoff)
    )
    # df is pre-sorted ascending, so tail(n) gives the n most-recent in order
    return df[mask].tail(n)


def _team_result(team: str, row: pd.Series) -> str:
    """Return 'W', 'D', or 'L' for a team in a given match row."""
    if row["home_team"] == team:
        if row["home_score"] > row["away_score"]:
            return "W"
        elif row["home_score"] < row["away_score"]:
            return "L"
        else:
            return "D"
    else:
        if row["away_score"] > row["home_score"]:
            return "W"
        elif row["away_score"] < row["home_score"]:
            return "L"
        else:
            return "D"
