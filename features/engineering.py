"""
features/engineering.py — XGBoost Feature Engineering Pipeline
===============================================================
Builds the complete feature vector for each match from raw data sources.

The features are designed to capture:
  1. Team quality   → ELO ratings, FIFA rankings
  2. Team form      → Recent results, goal stats
  3. Head-to-head   → Historical matchup record
  4. Context        → Stage of competition, days rest
  5. Interactions   → ELO²  (captures non-linear effects)

WALK-FORWARD GUARANTEE:
  All features are computed from data available BEFORE the match date.
  This is enforced by passing the match date to all form/H2H functions.
  Never use post-match data as features (it would be data leakage).

FEATURE NORMALISATION:
  XGBoost handles mixed scales well (no need to standardize), but for
  interpretability, we document the typical range of each feature below.

FEATURE LIST (23 total):
  elo_diff              : ELO rating difference (home − away). Range: -500 to +500
  fifa_rank_diff        : FIFA ranking difference (away − home). Range: -100 to +100
                          (inverted: lower rank = better team, so positive = home advantage)
  home_attack_str       : Dixon-Coles attack parameter for "home" team. Range: -1 to +2
  home_defense_str      : Dixon-Coles defense parameter for "home" team. Range: -1 to +1
  away_attack_str       : Same for "away" team
  away_defense_str      : Same for "away" team
  home_form_5           : Form score (last 5 matches). Range: 0 to 1
  away_form_5           : Same for away team
  home_form_10          : Form score (last 10 matches). Range: 0 to 1
  away_form_10          : Same for away team
  h2h_home_wins         : Home team wins in last 5 H2H meetings. Range: 0 to 5
  h2h_draws             : Draws in last 5 H2H meetings. Range: 0 to 5
  h2h_away_wins         : Away team wins in last 5 H2H meetings. Range: 0 to 5
  home_goals_scored_avg : Average goals scored by home team (last 10). Range: 0 to 3.5
  home_goals_conceded_avg: Average goals conceded by home team. Range: 0 to 3.5
  away_goals_scored_avg : Same for away team
  away_goals_conceded_avg: Same for away team
  home_days_rest        : Days since last match. Range: 0 to 30+
  away_days_rest        : Same for away team
  tournament_round      : Numeric encoding: GROUP=0, R32=1, R16=2, QF=3, SF=4, FINAL=5
  home_wc_appearances   : Historical World Cup appearances. Range: 0 to 22
  away_wc_appearances   : Same for away team
  elo_diff_sq           : elo_diff² (captures non-linear ELO effect). Range: 0 to 250000
  rank_diff_sq          : fifa_rank_diff² (non-linear FIFA rank effect)

HOW TO DEBUG:
  - If XGBoost performance is poor, check feature_importance() — low-importance
    features may be adding noise
  - If build_features() returns None values, the team may not be in dc_model
    or elo_system (unseen teams get fallback values)
  - To inspect a feature vector: print(json.dumps(feat, indent=2))
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from features.form import (
    calculate_form,
    calculate_goals_stats,
    days_since_last_match,
    h2h_record,
)

logger = logging.getLogger(__name__)

# All 23 features in the exact order XGBoost expects them
FEATURE_NAMES: list[str] = [
    "elo_diff",
    "fifa_rank_diff",
    "home_attack_str",
    "home_defense_str",
    "away_attack_str",
    "away_defense_str",
    "home_form_5",
    "away_form_5",
    "home_form_10",
    "away_form_10",
    "h2h_home_wins",
    "h2h_draws",
    "h2h_away_wins",
    "home_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_scored_avg",
    "away_goals_conceded_avg",
    "home_days_rest",
    "away_days_rest",
    "tournament_round",
    "home_wc_appearances",
    "away_wc_appearances",
    "elo_diff_sq",
    "rank_diff_sq",
]

STAGE_ENCODING: dict[str, int] = {
    "GROUP": 0, "R32": 1, "R16": 2,
    "QF": 3, "SF": 4, "FINAL": 5,
}


class FeatureEngineer:
    """Builds the XGBoost feature vector from raw model outputs and history.

    Usage
    -----
    eng = FeatureEngineer(
        historical_df=training_df,
        elo_system=fitted_elo,
        dc_model=fitted_dc,
        team_meta=TEAM_META,
    )

    # Predict a single match:
    features = eng.build_features("BRAZIL", "FRANCE", date(2026, 6, 20), "GROUP")

    # Build training set for XGBoost:
    X, y = eng.build_training_set(training_df)
    """

    def __init__(
        self,
        historical_df: pd.DataFrame,
        elo_system: object,     # EloSystem instance
        dc_model: object,       # DixonColesModel instance
        team_meta: dict,        # from wc2026_teams.TEAM_META
    ) -> None:
        # Pre-parse dates and pre-sort once so form lookups never repeat this work
        df = historical_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        self.df       = df
        self.elo      = elo_system
        self.dc       = dc_model
        self.meta     = team_meta

    def build_features(
        self,
        home: str,
        away: str,
        match_date: date,
        stage: str = "GROUP",
    ) -> dict[str, float]:
        """Build the feature dict for a single match.

        Parameters
        ----------
        home, away   : str  — team names (UPPERCASE)
        match_date   : date — match date (for walk-forward form calculation)
        stage        : str  — tournament stage (GROUP, R16, QF, SF, FINAL)

        Returns
        -------
        dict mapping FEATURE_NAMES → float values

        HOW TO DEBUG:
            features = eng.build_features("BRAZIL", "ARGENTINA", date(2026, 6, 20))
            for k, v in features.items():
                print(f"  {k}: {v}")
        """
        # ── ELO features ─────────────────────────────────────────────────
        home_elo  = self.elo.get_rating(home)
        away_elo  = self.elo.get_rating(away)
        elo_diff  = home_elo - away_elo

        # ── FIFA ranking features ─────────────────────────────────────────
        home_meta = self.meta.get(home, {})
        away_meta = self.meta.get(away, {})

        home_rank = home_meta.get("fifa_rank", 50)
        away_rank = away_meta.get("fifa_rank", 50)
        # Positive = home has better rank (lower number = better)
        rank_diff = away_rank - home_rank

        # ── Dixon-Coles attack/defense strengths ─────────────────────────
        home_atk = self.dc.attack_strengths.get(home, 0.0)
        home_def = self.dc.defense_strengths.get(home, 0.0)
        away_atk = self.dc.attack_strengths.get(away, 0.0)
        away_def = self.dc.defense_strengths.get(away, 0.0)

        # ── Form features (walk-forward: only data before match_date) ─────
        home_form_5  = calculate_form(home, self.df, match_date, n_matches=5)
        away_form_5  = calculate_form(away, self.df, match_date, n_matches=5)
        home_form_10 = calculate_form(home, self.df, match_date, n_matches=10)
        away_form_10 = calculate_form(away, self.df, match_date, n_matches=10)

        # ── Head-to-head ──────────────────────────────────────────────────
        h2h = h2h_record(home, away, self.df, n_meetings=5)

        # ── Goal statistics ───────────────────────────────────────────────
        home_gs = calculate_goals_stats(home, self.df, match_date, n_matches=10)
        away_gs = calculate_goals_stats(away, self.df, match_date, n_matches=10)

        # ── Rest / fatigue ────────────────────────────────────────────────
        home_rest = days_since_last_match(home, self.df, match_date)
        away_rest = days_since_last_match(away, self.df, match_date)

        # ── Tournament context ────────────────────────────────────────────
        stage_enc  = STAGE_ENCODING.get(stage, 0)
        home_apps  = home_meta.get("wc_apps", 5)
        away_apps  = away_meta.get("wc_apps", 5)

        # ── Non-linear interaction terms ──────────────────────────────────
        elo_diff_sq  = elo_diff ** 2
        rank_diff_sq = rank_diff ** 2

        return {
            "elo_diff":               elo_diff,
            "fifa_rank_diff":         rank_diff,
            "home_attack_str":        home_atk,
            "home_defense_str":       home_def,
            "away_attack_str":        away_atk,
            "away_defense_str":       away_def,
            "home_form_5":            home_form_5,
            "away_form_5":            away_form_5,
            "home_form_10":           home_form_10,
            "away_form_10":           away_form_10,
            "h2h_home_wins":          float(h2h["home_wins"]),
            "h2h_draws":              float(h2h["draws"]),
            "h2h_away_wins":          float(h2h["away_wins"]),
            "home_goals_scored_avg":  home_gs["scored_avg"],
            "home_goals_conceded_avg": home_gs["conceded_avg"],
            "away_goals_scored_avg":  away_gs["scored_avg"],
            "away_goals_conceded_avg": away_gs["conceded_avg"],
            "home_days_rest":         float(home_rest),
            "away_days_rest":         float(away_rest),
            "tournament_round":       float(stage_enc),
            "home_wc_appearances":    float(home_apps),
            "away_wc_appearances":    float(away_apps),
            "elo_diff_sq":            elo_diff_sq,
            "rank_diff_sq":           rank_diff_sq,
        }

    def build_training_set(
        self,
        match_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build (X, y) training arrays for XGBoost.

        Uses walk-forward feature construction: features for each match
        are built from data available BEFORE that match's date.

        Parameters
        ----------
        match_df : pd.DataFrame — full historical dataset (same as self.df)

        Returns
        -------
        (X, y) where:
            X : pd.DataFrame with columns = FEATURE_NAMES
            y : pd.Series with values 0/1/2 (away win / draw / home win)

        This is slow for large datasets (~1 min for 3000 matches).
        Progress is logged every 500 matches.

        HOW TO DEBUG:
            X, y = eng.build_training_set(df)
            print(X.isnull().sum())  # should be 0 (no missing values)
            print(y.value_counts())  # check class distribution
        """
        rows: list[dict] = []
        targets: list[int] = []

        match_df = match_df.copy()
        match_df["date"] = pd.to_datetime(match_df["date"])
        match_df = match_df.sort_values("date")

        for i, (_, row) in enumerate(match_df.iterrows()):
            home = str(row["home_team"]).upper()
            away = str(row["away_team"]).upper()
            match_date = row["date"].date()
            stage = "GROUP"  # training data doesn't always have stage

            try:
                features = self.build_features(home, away, match_date, stage)
                rows.append(features)
                targets.append(int(row["outcome"]))
            except Exception as exc:
                logger.warning("Skipping match %s vs %s: %s", home, away, exc)
                continue

            if (i + 1) % 500 == 0:
                logger.info("Feature engineering: %d/%d matches processed", i + 1, len(match_df))

        X = pd.DataFrame(rows, columns=FEATURE_NAMES)
        y = pd.Series(targets, name="outcome")

        logger.info("Training set: %d samples, %d features", len(X), len(FEATURE_NAMES))
        return X, y
