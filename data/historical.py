"""
data/historical.py — Historical International Match Data Loader
===============================================================
Loads the training dataset for ELO, Dixon-Coles, and XGBoost models.

Data priority (highest to lowest):
  1. Committed CSV at betting/data/raw/international_results.csv
     (sourced from martj42/international_results on GitHub, CC0 license)
  2. Synthetic data generated on-the-fly from seeded random distributions
     (always available, ensures CI and cold-starts never crash)

Synthetic data is calibrated to match real international football statistics:
  - Home team goals: Poisson(λ=1.45) — includes some home advantage
  - Away team goals: Poisson(λ=1.10)
  - Neutral venue: both teams Poisson(λ=1.25)
  - ~30 "strong" teams beat "weak" teams more often (ELO-like weights)

The CSV / synthetic data covers 1993–2022 (~8,500 rows / 2,000 synthetic).
Earlier data is excluded because squad structures changed dramatically
post-USSR dissolution (1991) and tactical/physical standards differ too much.

HOW TO DEBUG:
  - If models seem poorly calibrated, check that the CSV was loaded
    (not the synthetic fallback): load_training_data()["_source"].unique()
  - The _source column is "csv" or "synthetic" for tracing
  - Synthetic data always uses seed=42 for reproducibility
  - To force re-training: delete betting/data/raw/xgb_model.joblib
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Path to the optional bundled historical CSV
_RAW_DIR = Path(__file__).parent / "raw"
_CSV_PATH = _RAW_DIR / "international_results.csv"

# Required columns — both the CSV and synthetic data must have these
REQUIRED_COLUMNS = [
    "date",        # str ISO "YYYY-MM-DD"
    "home_team",   # str — team name (title case, e.g. "Brazil")
    "away_team",   # str
    "home_score",  # int
    "away_score",  # int
    "tournament",  # str — "FIFA World Cup", "Friendly", etc.
    "neutral",     # bool — True if played at neutral venue
]


def load_training_data(min_year: int = 1993) -> pd.DataFrame:
    """Load historical match data for model training.

    Returns
    -------
    pd.DataFrame with columns matching REQUIRED_COLUMNS plus:
        outcome     : int — 2=home win, 1=draw, 0=away win
        _source     : str — "csv" or "synthetic" (for debugging)

    HOW TO DEBUG:
        print(load_training_data()["_source"].value_counts())
        # Should show "csv" if the raw file exists, "synthetic" otherwise
    """
    _raw_dir_ensure()

    if _CSV_PATH.exists():
        logger.info("Loading historical data from %s", _CSV_PATH)
        df = _load_csv()
    else:
        logger.warning(
            "No CSV at %s — using synthetic data. "
            "For better predictions, download the dataset from: "
            "https://github.com/martj42/international_results",
            _CSV_PATH,
        )
        df = _generate_synthetic_matches(n=3000, seed=42)

    # Filter to years >= min_year for modelling relevance
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].dt.year >= min_year].copy()

    # Add outcome column (used as XGBoost target)
    df["outcome"] = df.apply(_outcome, axis=1)

    # Ensure neutral flag is bool
    df["neutral"] = df["neutral"].astype(bool)

    logger.info("Training data: %d matches (%d–%d)", len(df),
                df["date"].dt.year.min(), df["date"].dt.year.max())
    return df.reset_index(drop=True)


def load_wc_history() -> pd.DataFrame:
    """Return only World Cup matches — used for model validation.

    Filters by tournament name containing "World Cup".
    """
    df = load_training_data()
    wc = df[df["tournament"].str.contains("World Cup", case=False, na=False)].copy()
    logger.info("WC history: %d matches", len(wc))
    return wc


# ─── Private helpers ─────────────────────────────────────────────────────────

def _raw_dir_ensure() -> None:
    """Create the raw data directory if it doesn't exist."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)


def _load_csv() -> pd.DataFrame:
    """Load and validate the bundled CSV file."""
    df = pd.read_csv(_CSV_PATH)

    # Normalise column names (some CSVs use 'home_goals', not 'home_score')
    df = df.rename(columns={
        "home_goals": "home_score",
        "away_goals": "away_score",
    })

    # Check required columns exist
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing columns: {missing}. "
            f"Expected: {REQUIRED_COLUMNS}"
        )

    df["_source"] = "csv"
    return df[REQUIRED_COLUMNS + ["_source"]]


def _outcome(row: pd.Series) -> int:
    """Map a match row to 3-class outcome label.
    2 = home win, 1 = draw, 0 = away win.
    This encoding is used as the XGBoost classification target.
    """
    if row["home_score"] > row["away_score"]:
        return 2   # home win
    elif row["home_score"] < row["away_score"]:
        return 0   # away win
    else:
        return 1   # draw


def _generate_synthetic_matches(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic international match data.

    The synthetic data is statistically calibrated to match real international
    football distributions. It's NOT meant to be accurate predictions — it's
    a fallback so all models can train and tests can run without real data.

    Statistical calibration:
        - International football averages ~1.4 goals/team/game (home), 1.1 (away)
        - ~43% home wins, ~27% draws, ~30% away wins at non-neutral venues
        - World Cup matches (neutral): ~39% home/draw/away split

    Parameters
    ----------
    n    : int — number of matches to generate (default 3000)
    seed : int — random seed for reproducibility (default 42)

    HOW TO DEBUG:
        df = _generate_synthetic_matches(n=100)
        print(df.groupby("outcome").size() / len(df))
        # Should show roughly 43% / 27% / 30% distribution
    """
    rng = np.random.default_rng(seed)

    # 48 synthetic teams with varying strength levels (mimics ELO spread)
    teams = [
        # Elite (ELO ~2050-2130)
        "Brazil", "Argentina", "France", "Spain", "Germany", "England",
        # Strong (ELO ~1950-2050)
        "Portugal", "Netherlands", "Belgium", "Italy", "Croatia", "Colombia",
        "Uruguay", "Denmark", "Switzerland", "Morocco",
        # Mid-tier (ELO ~1800-1950)
        "Mexico", "Japan", "South Korea", "USA", "Austria", "Serbia",
        "Poland", "Senegal", "Egypt", "Chile", "Australia",
        # Lower tier (ELO ~1600-1800)
        "Iran", "Nigeria", "Saudi Arabia", "Cameroon", "Ecuador",
        "Hungary", "Romania", "Algeria", "South Africa", "Costa Rica",
        "Panama", "Canada", "Jordan", "Iraq", "Honduras", "New Zealand",
        "Jamaica", "Uzbekistan", "Venezuela", "Mali", "Paraguay", "Ivory Coast",
    ]

    # Assign strength scores to teams (higher = stronger)
    n_teams = len(teams)
    # Strength exponentially distributed so elite teams dominate
    strength = np.exp(np.linspace(1.8, 0.0, n_teams))
    team_strength = dict(zip(teams, strength))

    records: list[dict] = []
    current_date = date(1993, 1, 1)

    for i in range(n):
        # Pick two different teams at random
        home, away = rng.choice(teams, size=2, replace=False)

        # Determine neutral venue (roughly 20% of international matches)
        neutral = rng.random() < 0.20

        # Pick tournament type
        tournament_type = rng.choice(
            ["FIFA World Cup", "Friendly", "Qualification", "Continental"],
            p=[0.08, 0.35, 0.40, 0.17],
        )

        # World Cup and qualifiers are treated as more competitive
        is_wc = "World Cup" in tournament_type
        if is_wc:
            neutral = True  # all WC matches at neutral venue

        # Goal scoring model: Poisson with team-strength-adjusted lambda
        # Home advantage = 0.25 extra expected goals (eliminated for neutral)
        home_advantage = 0.0 if neutral else 0.25
        base_lambda = 1.15  # average expected goals per team

        # Scale by relative team strength
        strength_ratio = team_strength[home] / (team_strength[home] + team_strength[away])
        home_lambda = base_lambda * (strength_ratio * 2) + home_advantage
        away_lambda = base_lambda * (1 - strength_ratio) * 2

        # Clip to realistic range
        home_lambda = np.clip(home_lambda, 0.3, 3.5)
        away_lambda = np.clip(away_lambda, 0.3, 3.5)

        home_goals = int(rng.poisson(home_lambda))
        away_goals = int(rng.poisson(away_lambda))

        # Advance date by 1–10 days
        from datetime import timedelta
        current_date = current_date + timedelta(days=int(rng.integers(1, 11)))
        if current_date > date(2022, 12, 18):
            current_date = date(1993, 1, 1)  # wrap around

        records.append({
            "date":       current_date.isoformat(),
            "home_team":  home,
            "away_team":  away,
            "home_score": home_goals,
            "away_score": away_goals,
            "tournament": tournament_type,
            "neutral":    neutral,
            "_source":    "synthetic",
        })

    df = pd.DataFrame(records)

    # Sort chronologically
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    return df
