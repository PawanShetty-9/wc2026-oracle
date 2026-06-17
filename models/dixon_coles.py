"""
models/dixon_coles.py — Dixon-Coles Bivariate Poisson Goal Model
================================================================
The Dixon-Coles model (Dixon & Coles, 1997) is the gold standard for
football score prediction. It models the goals scored by each team as
independent Poisson processes with team-specific attack and defense parameters.

POISSON BASICS:
  If a team scores an average of λ goals per game, the probability of
  scoring exactly k goals is: P(k) = λ^k × e^(−λ) / k!

  Dixon-Coles extends this to bivariate (joint) probabilities, modelling
  both teams simultaneously with a dependency correction for low scores.

THE MODEL:
  Expected goals for home team: λ_h = exp(attack_h + defense_a + home_adv)
  Expected goals for away team: λ_a = exp(attack_a + defense_h)

  For World Cup (neutral venue): home_adv = 0

  The rho (ρ) correction adjusts for the empirical observation that
  0-0, 1-0, 0-1, 1-1 scorelines are over- or under-represented
  relative to independent Poisson predictions:
    τ(0,0) = 1 − λ_h × λ_a × ρ
    τ(1,0) = 1 + λ_a × ρ
    τ(0,1) = 1 + λ_h × ρ
    τ(1,1) = 1 − ρ
    τ(x,y) = 1 for x+y ≥ 2

TIME DECAY:
  Recent matches are weighted more heavily using exponential decay:
    weight = exp(−ξ × days_since_match)
  With ξ = 0.002, a match 1 year ago has weight = exp(−0.73) ≈ 0.48

OPTIMISATION:
  Parameters (attack, defense per team + home_adv + rho) are found by
  maximising the weighted log-likelihood via scipy.optimize.minimize
  with method='L-BFGS-B' (Limited-memory BFGS with Bound constraints).

HOW TO DEBUG:
  - If scipy raises "OptimizeWarning: Desired error not necessarily achieved",
    try increasing max_goals or adjusting the initial parameter guess
  - If all probabilities are ~0.33 (no differentiation), training may have
    failed — check that df has sufficient rows (>200 unique matches)
  - Print model.attack_strengths to see if teams have varied attack values
  - Negative rho is expected (typically -0.05 to -0.15)
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

logger = logging.getLogger(__name__)

# Time decay factor: ξ = 0.002 → half-weight at ~347 days (roughly 1 year)
XI: float = 0.002

# Maximum goals to consider in the score matrix (10 is plenty; >10 is rare)
MAX_GOALS: int = 10

# Reference date for time decay (today's date used during prediction)
_TODAY: date = date(2026, 6, 17)


class DixonColesModel:
    """Dixon-Coles bivariate Poisson football prediction model.

    After fitting, call predict_outcome_probs(home, away) to get
    {'home': p, 'draw': p, 'away': p, 'over_25': p, 'btts': p}.
    """

    def __init__(self, xi: float = XI) -> None:
        """
        Parameters
        ----------
        xi : float — time decay constant. Higher = forget history faster.
            0.002 ≈ half-weight at 1 year (recommended for international football)
            0.0   = no decay (treat all historical matches equally)
        """
        self.xi = xi

        # Model parameters (set by fit())
        self.attack_strengths: dict[str, float] = {}    # team → α
        self.defense_strengths: dict[str, float] = {}   # team → β (lower = better defense)
        self.home_advantage: float = 0.0
        self.rho: float = -0.1          # Dixon-Coles correction (usually slightly negative)
        self.teams_: list[str] = []     # teams seen during training (ordered)
        self._fitted: bool = False
        self._train_date: date | None = None

    # ─── Public API ─────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "DixonColesModel":
        """Fit the model on historical match data.

        Parameters
        ----------
        df : pd.DataFrame with columns:
            date, home_team, away_team, home_score, away_score, neutral (bool)

        Returns self for method chaining.

        Training time: ~5-15 seconds on 3,000 matches (CPU).

        HOW TO DEBUG:
            If fit() throws ConvergenceWarning or returns odd results:
            1. Reduce n_matches by filtering df to more recent data
            2. Add regularisation (lambda_reg parameter)
            3. Try method='Nelder-Mead' as fallback (slower but more robust)
        """
        # Normalise team names to uppercase to match TEAM_META keys
        df = df.copy()
        df["home_team"] = df["home_team"].str.upper()
        df["away_team"] = df["away_team"].str.upper()
        df["date"] = pd.to_datetime(df["date"])

        # Remove rows with missing scores
        df = df.dropna(subset=["home_score", "away_score"])

        # Collect all unique teams
        teams = sorted(
            set(df["home_team"].unique()) | set(df["away_team"].unique())
        )
        self.teams_ = teams
        n = len(teams)
        team_idx = {t: i for i, t in enumerate(teams)}

        logger.info("Fitting Dixon-Coles on %d matches, %d teams", len(df), n)

        # Compute time-decay weights (more recent = higher weight)
        ref_date = df["date"].max()
        df["days_ago"] = (ref_date - df["date"]).dt.days
        df["weight"] = np.exp(-self.xi * df["days_ago"])

        # ── Optimisation setup ────────────────────────────────────────────
        # Parameter vector layout:
        #   params[0:n]   → attack strengths (one per team)
        #   params[n:2n]  → defense strengths (one per team)
        #   params[2n]    → home advantage (scalar)
        #   params[2n+1]  → rho (Dixon-Coles correction)

        # Initial guess: all zeros (log-scale, so zero = average strength)
        x0 = np.zeros(2 * n + 2)
        x0[2 * n + 1] = -0.1  # small negative rho as starting point

        # Build match list for faster iteration in the objective function
        matches = [
            {
                "hi":     team_idx[row.home_team],
                "ai":     team_idx[row.away_team],
                "hg":     int(row.home_score),
                "ag":     int(row.away_score),
                "w":      row.weight,
                "neutral": bool(row.neutral),
            }
            for row in df.itertuples()
            if row.home_team in team_idx and row.away_team in team_idx
        ]

        def neg_log_likelihood(params: np.ndarray) -> float:
            """Negative weighted log-likelihood (minimised by L-BFGS-B)."""
            attack  = params[:n]
            defense = params[n:2*n]
            home_adv = params[2*n]
            rho      = params[2*n+1]

            total = 0.0
            for m in matches:
                hi, ai = m["hi"], m["ai"]
                hg, ag = m["hg"], m["ag"]
                neutral = m["neutral"]

                # Expected goals (log-linear model)
                log_lam_h = attack[hi] + defense[ai] + (0.0 if neutral else home_adv)
                log_lam_a = attack[ai] + defense[hi]

                lam_h = np.exp(log_lam_h)
                lam_a = np.exp(log_lam_a)

                # Poisson log-probability
                log_p_h = scipy.stats.poisson.logpmf(hg, lam_h)
                log_p_a = scipy.stats.poisson.logpmf(ag, lam_a)

                # Dixon-Coles tau correction for low scores
                tau = _tau_correction(hg, ag, lam_h, lam_a, rho)
                if tau <= 0:
                    # Degenerate: skip this match to avoid log(0)
                    continue

                log_likelihood = log_p_h + log_p_a + np.log(tau)
                total += m["w"] * log_likelihood

            # L2 regularisation to prevent extreme parameter values
            reg = 0.001 * np.sum(params[:2*n] ** 2)
            return -(total - reg)

        # Bounds: rho must be in (-1, 0] to keep tau > 0 for low scores
        bounds = (
            [(None, None)] * (2 * n)      # attack and defense: unbounded
            + [(None, None)]               # home_advantage: unbounded
            + [(-0.99, 0.0)]               # rho: must be in (-1, 0]
        )

        result = scipy.optimize.minimize(
            neg_log_likelihood,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-8},
        )

        if not result.success:
            logger.warning(
                "Dixon-Coles optimisation did not fully converge: %s. "
                "Predictions will still work but may be slightly off.",
                result.message,
            )

        # Extract fitted parameters
        opt = result.x
        self.attack_strengths  = dict(zip(teams, opt[:n]))
        self.defense_strengths = dict(zip(teams, opt[n:2*n]))
        self.home_advantage    = float(opt[2*n])
        self.rho               = float(opt[2*n+1])
        self._fitted           = True
        self._train_date       = ref_date.date() if hasattr(ref_date, 'date') else ref_date

        logger.info(
            "Dixon-Coles fitted. home_adv=%.3f, rho=%.3f",
            self.home_advantage, self.rho,
        )
        return self

    def predict_score_matrix(
        self,
        home: str,
        away: str,
        neutral: bool = True,  # All WC matches are neutral
        max_goals: int = MAX_GOALS,
    ) -> np.ndarray:
        """Compute the joint probability matrix P[home_goals, away_goals].

        Returns an (max_goals+1) × (max_goals+1) matrix where entry [i, j]
        is the probability of the home team scoring i goals and the away
        team scoring j goals.

        Example: matrix[2, 1] = P(2-1 scoreline)

        HOW TO DEBUG:
            matrix = model.predict_score_matrix("BRAZIL", "ARGENTINA")
            print(f"P(2-1) = {matrix[2,1]:.3f}")
            print(f"Sum = {matrix.sum():.6f}")  # should be ≈ 1.0
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_score_matrix()")

        lam_h, lam_a = self._expected_goals(home, away, neutral)
        matrix = np.zeros((max_goals + 1, max_goals + 1))

        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                p_h = scipy.stats.poisson.pmf(i, lam_h)
                p_a = scipy.stats.poisson.pmf(j, lam_a)
                tau = _tau_correction(i, j, lam_h, lam_a, self.rho)
                matrix[i, j] = p_h * p_a * tau

        # Renormalise to handle truncation (prob mass > max_goals is dropped)
        total = matrix.sum()
        if total > 0:
            matrix /= total

        return matrix

    def predict_outcome_probs(
        self,
        home: str,
        away: str,
        neutral: bool = True,
    ) -> dict[str, float]:
        """Return outcome probabilities for a single match.

        Returns
        -------
        dict with keys:
            home    : P(home team wins)
            draw    : P(draw / tie)
            away    : P(away team wins)
            over_25 : P(total goals > 2.5)
            btts    : P(both teams score at least 1 goal each)
            most_likely_score : (home_goals, away_goals) tuple

        HOW TO DEBUG:
            probs = model.predict_outcome_probs("BRAZIL", "ARGENTINA")
            assert abs(sum([probs["home"], probs["draw"], probs["away"]]) - 1.0) < 1e-5
        """
        matrix = self.predict_score_matrix(home, away, neutral)

        # 1X2 probabilities: sum over appropriate parts of the matrix
        p_home = float(np.tril(matrix, k=-1).sum())  # home_goals > away_goals
        p_away = float(np.triu(matrix, k=1).sum())   # away_goals > home_goals
        p_draw = float(np.diag(matrix).sum())         # home_goals = away_goals

        # Over 2.5 goals: total goals > 2 (i.e., ≥ 3)
        max_g = matrix.shape[0]
        p_over_25 = sum(
            matrix[i, j]
            for i in range(max_g)
            for j in range(max_g)
            if i + j >= 3
        )

        # Both Teams to Score (BTTS): both teams score ≥ 1 goal
        p_btts = 1.0 - (
            matrix[:, 0].sum()    # all rows where away = 0
            + matrix[0, :].sum()  # all cols where home = 0
            - matrix[0, 0]        # subtract 0-0 (counted twice)
        )

        # Most likely scoreline (argmax of matrix)
        idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        most_likely_score = (int(idx[0]), int(idx[1]))

        return {
            "home":               round(p_home, 4),
            "draw":               round(p_draw, 4),
            "away":               round(p_away, 4),
            "over_25":            round(float(p_over_25), 4),
            "btts":               round(float(p_btts), 4),
            "most_likely_score":  most_likely_score,
        }

    # ─── Private helpers ─────────────────────────────────────────────────────

    def _expected_goals(
        self,
        home: str,
        away: str,
        neutral: bool = True,
    ) -> tuple[float, float]:
        """Compute expected goals for both teams.

        For unknown teams (not seen during training), uses league-average
        attack (0.0 in log space) and defense (0.0 in log space).
        This is a safe fallback but predictions are less accurate.

        HOW TO DEBUG:
            lam_h, lam_a = model._expected_goals("BRAZIL", "ARGENTINA")
            # Typical range: 0.5 to 2.5 goals per team
            # If both are 1.28 (the mean), team might be unknown
        """
        # Default to league-average for unseen teams (0 in log scale)
        a_home = self.attack_strengths.get(home, 0.0)
        d_home = self.defense_strengths.get(home, 0.0)
        a_away = self.attack_strengths.get(away, 0.0)
        d_away = self.defense_strengths.get(away, 0.0)

        if home not in self.attack_strengths:
            logger.warning("Team '%s' not in Dixon-Coles model — using average", home)
        if away not in self.attack_strengths:
            logger.warning("Team '%s' not in Dixon-Coles model — using average", away)

        home_adv = 0.0 if neutral else self.home_advantage

        lam_home = np.exp(a_home + d_away + home_adv)
        lam_away = np.exp(a_away + d_home)

        return float(lam_home), float(lam_away)


# ─── Module-level helper ─────────────────────────────────────────────────────

def _tau_correction(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles tau correction for low-scoring matches.

    Adjusts for the fact that 0-0, 1-0, 0-1, 1-1 occur at different
    frequencies than a pure independent Poisson model would predict.

    Parameters
    ----------
    x, y   : int   — observed/predicted home and away goals (0, 1 only relevant)
    lam_h  : float — home team expected goals (λ_h)
    lam_a  : float — away team expected goals (λ_a)
    rho    : float — correction parameter (typically slightly negative)

    Returns
    -------
    float — multiplicative adjustment (≈ 1.0 for most scores, adjusted for 0/1)
    """
    if x == 0 and y == 0:
        return 1.0 - lam_h * lam_a * rho
    elif x == 1 and y == 0:
        return 1.0 + lam_a * rho
    elif x == 0 and y == 1:
        return 1.0 + lam_h * rho
    elif x == 1 and y == 1:
        return 1.0 - rho
    else:
        return 1.0  # no adjustment for higher-scoring games
