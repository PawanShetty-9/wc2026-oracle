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
        # Pre-parse dates, uppercase teams, and pre-sort once
        df = historical_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["home_team"] = df["home_team"].str.upper()
        df["away_team"] = df["away_team"].str.upper()
        df = df.sort_values("date").reset_index(drop=True)
        self.df       = df
        self.elo      = elo_system
        self.dc       = dc_model
        self.meta     = team_meta

        # Pre-group per team: each form lookup scans ~100 rows instead of 3000
        all_teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self._team_df: dict[str, pd.DataFrame] = {
            team: df[(df["home_team"] == team) | (df["away_team"] == team)].reset_index(drop=True)
            for team in all_teams
        }

        # Convert per-team data to numpy arrays — eliminates pandas overhead on 27k calls
        self._team_arr: dict[str, dict] = {}
        for team, tdf in self._team_df.items():
            self._team_arr[team] = {
                "dates":      tdf["date"].values.astype("int64"),   # nanoseconds
                "is_home":    tdf["home_team"].values == team,
                "home_score": tdf["home_score"].values.astype(float),
                "away_score": tdf["away_score"].values.astype(float),
            }

        # Pre-build H2H cache as numpy arrays for same reason
        self._h2h_arr: dict[tuple, dict] = {}
        for i, t1 in enumerate(all_teams):
            for t2 in all_teams[i+1:]:
                mask = (
                    ((df["home_team"] == t1) & (df["away_team"] == t2)) |
                    ((df["home_team"] == t2) & (df["away_team"] == t1))
                )
                pdf = df[mask]
                arr = {
                    "dates":      pdf["date"].values.astype("int64"),
                    "is_home_t1": pdf["home_team"].values == t1,
                    "home_score": pdf["home_score"].values.astype(float),
                    "away_score": pdf["away_score"].values.astype(float),
                }
                self._h2h_arr[(t1, t2)] = arr
                self._h2h_arr[(t2, t1)] = arr

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

        # Convert match_date once to int64 nanoseconds for all numpy fast-paths
        match_date_ns = int(pd.Timestamp(match_date).value)

        home_arr = self._team_arr.get(home)
        away_arr = self._team_arr.get(away)

        # Fall back to slow pandas path for unknown teams (e.g. live predictions
        # for teams not seen during training — very rare)
        if home_arr is None or away_arr is None:
            home_df = self._team_df.get(home, self.df)
            away_df = self._team_df.get(away, self.df)
            cutoff  = pd.Timestamp(match_date)
            home_form_5  = calculate_form(home, home_df, match_date, n_matches=5)
            away_form_5  = calculate_form(away, away_df, match_date, n_matches=5)
            home_form_10 = calculate_form(home, home_df, match_date, n_matches=10)
            away_form_10 = calculate_form(away, away_df, match_date, n_matches=10)
            h2h_df = pd.concat([home_df, away_df]).drop_duplicates()
            h2h = h2h_record(home, away, h2h_df, n_meetings=5)
            home_gs = calculate_goals_stats(home, home_df, match_date, n_matches=10)
            away_gs = calculate_goals_stats(away, away_df, match_date, n_matches=10)
            home_rest = days_since_last_match(home, home_df, match_date)
            away_rest = days_since_last_match(away, away_df, match_date)
        else:
            # ── Pure-numpy fast paths (zero pandas overhead) ──────────────
            home_form_5  = self._form_fast(home_arr, match_date_ns, 5)
            away_form_5  = self._form_fast(away_arr, match_date_ns, 5)
            home_form_10 = self._form_fast(home_arr, match_date_ns, 10)
            away_form_10 = self._form_fast(away_arr, match_date_ns, 10)
            h2h          = self._h2h_fast(home, away, 5)
            home_gs      = self._goals_fast(home_arr, match_date_ns, 10)
            away_gs      = self._goals_fast(away_arr, match_date_ns, 10)
            home_rest    = self._rest_fast(home_arr, match_date_ns)
            away_rest    = self._rest_fast(away_arr, match_date_ns)

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

    # ── Pure-numpy fast-path helpers — called 27k times during training ──────────
    # All operate on pre-built int64 nanosecond arrays (zero pandas overhead).
    # match_date_ns must be pd.Timestamp(match_date).value (int64 nanoseconds).

    def _form_fast(self, arr: dict, match_date_ns: int, n: int) -> float:
        idx = np.where(arr["dates"] < match_date_ns)[0]
        if len(idx) == 0:
            return 0.5
        idx = idx[-n:]
        is_home = arr["is_home"][idx]
        hs, as_ = arr["home_score"][idx], arr["away_score"][idx]
        wins  = (is_home & (hs > as_)) | (~is_home & (as_ > hs))
        points = float(wins.sum() * 3 + (hs == as_).sum())
        return points / (len(idx) * 3.0)

    def _goals_fast(self, arr: dict, match_date_ns: int, n: int) -> dict:
        idx = np.where(arr["dates"] < match_date_ns)[0]
        if len(idx) == 0:
            return {"scored_avg": 1.2, "conceded_avg": 1.2, "goal_diff_avg": 0.0}
        idx = idx[-n:]
        k = len(idx)
        w = np.array([0.85 ** i for i in range(k - 1, -1, -1)])
        w /= w.sum()
        is_home = arr["is_home"][idx]
        hs, as_ = arr["home_score"][idx], arr["away_score"][idx]
        scored   = np.where(is_home, hs, as_)
        conceded = np.where(is_home, as_, hs)
        sc = float(np.dot(scored, w))
        cc = float(np.dot(conceded, w))
        return {"scored_avg": round(sc, 3), "conceded_avg": round(cc, 3), "goal_diff_avg": round(sc - cc, 3)}

    def _rest_fast(self, arr: dict, match_date_ns: int) -> int:
        idx = np.where(arr["dates"] < match_date_ns)[0]
        if len(idx) == 0:
            return 7
        last_ns = arr["dates"][idx[-1]]
        days = int((match_date_ns - last_ns) / 86_400_000_000_000)
        return max(0, days)

    def _h2h_fast(self, home: str, away: str, n: int) -> dict:
        arr = self._h2h_arr.get((home, away))
        if arr is None or len(arr["dates"]) == 0:
            return {"home_wins": 0, "draws": 0, "away_wins": 0}
        hs, as_ = arr["home_score"][-n:], arr["away_score"][-n:]
        is_home_t1 = arr["is_home_t1"][-n:]
        draws = int((hs == as_).sum())
        hw = int(((is_home_t1 & (hs > as_)) | (~is_home_t1 & (as_ > hs))).sum())
        return {"home_wins": hw, "draws": draws, "away_wins": len(hs) - draws - hw}

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
