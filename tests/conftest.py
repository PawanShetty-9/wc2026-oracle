"""
tests/conftest.py — Shared pytest fixtures for betting module tests.
All fixtures use synthetic data; no API keys or internet access required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.elo import EloSystem
from models.dixon_coles import DixonColesModel
from features.form import calculate_form


# ── Synthetic match dataset ────────────────────────────────────────────────────

def _make_match(home: str, away: str, hs: int, as_: int, date: str, tournament: str = "Friendly") -> dict:
    return {
        "date": date,
        "home_team": home,
        "away_team": away,
        "home_score": hs,
        "away_score": as_,
        "tournament": tournament,
        "neutral": False,
        "outcome": 2 if hs > as_ else (1 if hs == as_ else 0),
    }


@pytest.fixture(scope="session")
def synthetic_matches() -> pd.DataFrame:
    """2000-match synthetic dataset covering 10 teams, 1993-2024."""
    rng = np.random.default_rng(42)
    teams = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON",
             "ZETA", "ETA", "THETA", "IOTA", "KAPPA"]

    # Simple attack/defense strengths to generate realistic scorelines
    attack  = {t: rng.uniform(0.8, 1.6) for t in teams}
    defense = {t: rng.uniform(0.8, 1.4) for t in teams}

    rows = []
    year = 1993
    for i in range(2000):
        year += i // 200
        year = min(year, 2023)
        month = rng.integers(1, 13)
        day   = rng.integers(1, 28)
        date  = f"{year}-{month:02d}-{day:02d}"

        home, away = rng.choice(teams, size=2, replace=False)
        lam_h = attack[home] / defense[away] * 1.3
        lam_a = attack[away] / defense[home]
        hs = int(rng.poisson(lam_h))
        as_ = int(rng.poisson(lam_a))
        rows.append(_make_match(home, away, hs, as_, date))

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@pytest.fixture(scope="session")
def trained_elo(synthetic_matches) -> EloSystem:
    """ELO system trained on synthetic_matches."""
    elo = EloSystem()
    elo.train_on_history(synthetic_matches)
    return elo


@pytest.fixture(scope="session")
def trained_dc(synthetic_matches) -> DixonColesModel:
    """Dixon-Coles model fitted on synthetic_matches."""
    dc = DixonColesModel()
    dc.fit(synthetic_matches)
    return dc


@pytest.fixture
def simple_odds() -> dict:
    """Realistic decimal odds for a notional match."""
    return {
        "home_win": 2.10,
        "draw":     3.40,
        "away_win": 3.60,
        "over_25":  1.85,
        "btts_yes": 1.75,
    }


@pytest.fixture
def match_meta() -> dict:
    return {
        "match_id":   "TEST-001",
        "home_team":  "ALPHA",
        "away_team":  "BETA",
        "match_date": "2026-06-20",
        "stage":      "GROUP",
    }
