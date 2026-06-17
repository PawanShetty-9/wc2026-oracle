"""
models/elo.py — ELO Rating System for National Teams
=====================================================
ELO is the oldest and most reliable football prediction baseline.
Originally designed for chess (Arpad Elo, 1960), adapted for football
by the World Football Elo Ratings project (eloratings.net).

HOW ELO WORKS:
  Every team has a rating (typically 1500–2200). After each match:
    new_rating = old_rating + K × (actual_score − expected_score)
  Where:
    - expected_score = 1 / (1 + 10^((opponent_elo − your_elo) / 400))
      (= probability you would beat opponent if results were probabilistic)
    - actual_score   = 1.0 for win, 0.5 for draw, 0.0 for loss
    - K              = sensitivity factor (higher = faster adaptation)

  For football:
    - K = 40 for standard international matches
    - K = 60 for World Cup matches (higher stakes, more meaningful)
    - All World Cup matches are neutral venue (home_advantage = 0)

DRAW PROBABILITY:
  Pure ELO only gives P(A > B). To get a 3-way split (home/draw/away)
  we use the James Curley empirical approximation:
    p_draw ≈ 0.30 × exp(−|elo_diff| / 200)
  This gives ~30% draw rate for evenly-matched teams, falling to near-zero
  for large ELO gaps. Then we normalise: home + draw + away = 1.0

HOW TO DEBUG:
  - If win probabilities seem wrong, check that update() was called
    for all RESULTS_SO_FAR matches in chronological order
  - If draw probabilities are 0, check that elo_diff is not being passed
    as absolute value squared (sign matters only for win/loss split)
  - Print elo_system.ratings to inspect all current team ratings
"""

from __future__ import annotations

import logging
import math
from copy import deepcopy

import pandas as pd

logger = logging.getLogger(__name__)

# ── ELO configuration ─────────────────────────────────────────────────────────
K_STANDARD: float = 40.0   # K-factor for friendly / qualification matches
K_WORLD_CUP: float = 60.0  # Higher K for World Cup (results matter more)
HOME_ADVANTAGE_ELO: float = 100.0  # ELO points equivalent for home advantage
                                    # (set to 0 for World Cup / neutral venue)
INITIAL_RATING: float = 1500.0  # Default for first-time teams


class EloSystem:
    """World Football Elo rating system.

    Usage
    -----
    # 1. Create with pre-tournament ratings from TEAM_META
    elo = EloSystem(initial_ratings={"BRAZIL": 2070, "FRANCE": 2075, ...})

    # 2. Update with historical / tournament results
    elo.train_on_history(historical_df)  # bulk update from DataFrame

    # 3. Predict a match
    p_home, p_draw, p_away = elo.win_probability("BRAZIL", "FRANCE")
    """

    def __init__(self, initial_ratings: dict[str, float] | None = None) -> None:
        """
        Parameters
        ----------
        initial_ratings : dict mapping team_name → starting ELO rating.
            Teams not in this dict start at INITIAL_RATING (1500).
        """
        # Deep copy to avoid mutating the caller's dict
        self.ratings: dict[str, float] = dict(initial_ratings or {})
        self._match_count: int = 0  # tracks how many updates have been applied

    def get_rating(self, team: str) -> float:
        """Return current ELO rating for a team (default 1500 if unknown)."""
        return self.ratings.get(team, INITIAL_RATING)

    def expected_score(self, team_a_elo: float, team_b_elo: float) -> float:
        """Probability that team_a beats team_b, given their ELO ratings.

        Formula: 1 / (1 + 10^((B - A) / 400))
        Returns a value in (0, 1).

        >>> elo = EloSystem()
        >>> round(elo.expected_score(2000, 1800), 3)  # 200-pt favourite
        0.76
        """
        return 1.0 / (1.0 + 10.0 ** ((team_b_elo - team_a_elo) / 400.0))

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        is_wc: bool = True,
        is_neutral: bool = True,
    ) -> tuple[float, float]:
        """Update both teams' ratings after a match result.

        Parameters
        ----------
        home_team  : str — team name (must match TEAM_META keys)
        away_team  : str
        home_score : int — goals scored by home team
        away_score : int — goals scored by away team
        is_wc      : bool — if True, use higher K-factor
        is_neutral : bool — if True, home advantage = 0 (all WC matches)

        Returns
        -------
        (new_home_rating, new_away_rating) as a tuple of floats.

        HOW TO DEBUG:
            Before: home=2000, away=1900, expected home win prob=0.64
            After home wins 3-0: home gains ~14 pts, away loses ~14 pts
            After upset (away wins): home loses ~25 pts, away gains ~25 pts
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)

        # Apply home advantage in ELO space (for non-neutral venues)
        home_elo_adjusted = home_elo + (0.0 if is_neutral else HOME_ADVANTAGE_ELO)

        # Expected scores: P(home wins), P(away wins)
        home_expected = self.expected_score(home_elo_adjusted, away_elo)
        away_expected = 1.0 - home_expected

        # Actual scores: 1.0 win, 0.5 draw, 0.0 loss
        if home_score > away_score:
            home_actual, away_actual = 1.0, 0.0
        elif home_score < away_score:
            home_actual, away_actual = 0.0, 1.0
        else:
            home_actual = away_actual = 0.5

        # K-factor (World Cup = more important)
        k = K_WORLD_CUP if is_wc else K_STANDARD

        # Goal margin multiplier (larger wins = bigger rating swings)
        # Inspired by FIFA/World Football ELO implementations
        goal_diff = abs(home_score - away_score)
        margin_mult = _goal_margin_multiplier(goal_diff)

        # Update ratings
        delta_home = k * margin_mult * (home_actual - home_expected)
        delta_away = k * margin_mult * (away_actual - away_expected)

        new_home = home_elo + delta_home
        new_away = away_elo + delta_away

        self.ratings[home_team] = new_home
        self.ratings[away_team] = new_away
        self._match_count += 1

        logger.debug(
            "%s %d–%d %s | Δ home=%.1f (%.0f→%.0f), Δ away=%.1f (%.0f→%.0f)",
            home_team, home_score, away_score, away_team,
            delta_home, home_elo, new_home,
            delta_away, away_elo, new_away,
        )

        return new_home, new_away

    def win_probability(
        self,
        home_team: str,
        away_team: str,
        is_neutral: bool = True,  # All WC matches = neutral
    ) -> tuple[float, float, float]:
        """Compute 3-way match outcome probabilities.

        Parameters
        ----------
        home_team  : str
        away_team  : str
        is_neutral : bool — True for all World Cup matches

        Returns
        -------
        (p_home_win, p_draw, p_away_win) — a tuple that sums to 1.0

        Algorithm
        ---------
        1. Compute p_home_over_away using ELO formula
        2. Estimate draw probability using James Curley approximation:
               p_draw ≈ 0.30 × exp(−|Δelo| / 200)
           This models draws as most likely when teams are evenly matched.
        3. Distribute remaining probability between home and away win
           proportionally to their head-to-head ELO ratio.
        """
        home_elo = self.get_rating(home_team)
        away_elo = self.get_rating(away_team)

        # Apply home advantage (zero for neutral/WC)
        elo_adj = home_elo + (0.0 if is_neutral else HOME_ADVANTAGE_ELO)
        elo_diff = elo_adj - away_elo

        # Raw head-to-head probability (ELO formula)
        p_home_over_away = self.expected_score(elo_adj, away_elo)

        # Draw probability: highest for equal teams, decays with ELO gap
        # Calibrated to give ~27% draws for international football
        p_draw = 0.30 * math.exp(-abs(elo_diff) / 200.0)

        # Remaining probability is split between outright home/away wins
        p_decisive = 1.0 - p_draw
        p_home_win = p_decisive * p_home_over_away
        p_away_win = p_decisive * (1.0 - p_home_over_away)

        # Normalise to handle floating-point rounding
        total = p_home_win + p_draw + p_away_win
        p_home_win /= total
        p_draw /= total
        p_away_win /= total

        return p_home_win, p_draw, p_away_win

    def train_on_history(self, df: pd.DataFrame) -> "EloSystem":
        """Bulk-update ratings from a historical match DataFrame.

        Processes matches in chronological order (oldest first).
        Uses column names matching load_training_data() output:
            date, home_team, away_team, home_score, away_score,
            tournament, neutral

        Returns self for method chaining.

        HOW TO DEBUG:
            Before training: elo.ratings = {}  (or initial values)
            After training:  elo.ratings should have ~40-80 teams
            If no ratings change: check df column names match expected
        """
        # Ensure chronological order
        df_sorted = df.sort_values("date")

        for _, row in df_sorted.iterrows():
            home = str(row["home_team"]).upper()
            away = str(row["away_team"]).upper()

            # Skip if scores are missing (NaN in some datasets)
            if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
                continue

            is_wc = "world cup" in str(row.get("tournament", "")).lower()
            neutral = bool(row.get("neutral", False))

            self.update(
                home_team=home,
                away_team=away,
                home_score=int(row["home_score"]),
                away_score=int(row["away_score"]),
                is_wc=is_wc,
                is_neutral=neutral,
            )

        logger.info(
            "ELO trained on %d matches. Rated %d teams.",
            self._match_count, len(self.ratings),
        )
        return self

    def top_n(self, n: int = 20) -> list[tuple[str, float]]:
        """Return top-N teams by current ELO rating (descending).

        Returns list of (team_name, elo_rating) tuples.
        """
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)[:n]

    def all_ratings_df(self) -> pd.DataFrame:
        """Return all ratings as a sorted DataFrame for UI display."""
        df = pd.DataFrame(
            list(self.ratings.items()), columns=["team", "elo"]
        )
        return df.sort_values("elo", ascending=False).reset_index(drop=True)


# ─── Helper ──────────────────────────────────────────────────────────────────

def _goal_margin_multiplier(goal_diff: int) -> float:
    """Scale K-factor by goal margin to reward convincing wins.

    Based on World Football ELO methodology:
        0 goals (draw) → 1.0
        1 goal         → 1.0
        2 goals        → 1.5
        3+ goals       → 1.75

    This prevents a 5-0 win from moving ratings too dramatically
    (avoids rating inflation from high-scoring mismatches).
    """
    if goal_diff <= 1:
        return 1.0
    elif goal_diff == 2:
        return 1.5
    else:
        return 1.75
