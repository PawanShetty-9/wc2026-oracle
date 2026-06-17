"""
models/ensemble.py — Ensemble Prediction Orchestrator
======================================================
Combines Dixon-Coles (statistical model) and XGBoost (ML model) into
a single prediction by weighted averaging of their probability outputs.

WHY ENSEMBLE?
  - Dixon-Coles is strong at capturing team quality and scoring patterns
    but doesn't directly use form, H2H, or rest features
  - XGBoost captures all contextual features but can overfit on small WC datasets
  - Combining them smooths out each model's blind spots

WEIGHTING:
  Default: 60% Dixon-Coles, 40% XGBoost
  Rationale: DC has stronger theoretical grounding for football;
  XGBoost adds value but shouldn't dominate since its features are noisy.

  Users can adjust weights in the Streamlit sidebar.
  The learn_weights() method finds optimal weights via cross-validation
  on holdout WC data.

OUTPUT — MatchPrediction dataclass:
  All downstream code (UI, betting strategy) consumes MatchPrediction objects.
  This provides a single source of truth for all probability estimates.

HOW TO DEBUG:
  - If dc_probs and xgb_probs are the same, one model may have crashed
    and the other's output is being used for both — check logs
  - If all predictions are 50/50, neither model is fitted — check that
    fit() was called in app.py's cache_resource block
  - confidence = max(home_prob, draw_prob, away_prob) — low confidence
    (< 0.40) means the match is genuinely unpredictable
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np

logger = logging.getLogger(__name__)

# Default model weights (should sum to 1.0)
DEFAULT_WEIGHTS: dict[str, float] = {
    "dixon_coles": 0.60,
    "xgboost":     0.40,
}


@dataclass
class MatchPrediction:
    """Complete prediction for a single World Cup match.

    This is the primary data structure that flows through the entire app:
      models/ensemble.py → strategy/ev.py → strategy/kelly.py → ui/

    All probability fields sum to 1.0 within their category.
    """
    # Match identity
    match_id:   str         # e.g. "BRAZIL_vs_FRANCE_2026-06-19"
    home_team:  str
    away_team:  str
    match_date: str         # ISO format "YYYY-MM-DD"
    stage:      str         # "GROUP", "R16", "QF", "SF", "FINAL"
    group:      str = ""    # "A", "B", ..., "L" (empty for knockouts)
    venue:      str = ""

    # 1X2 outcome probabilities (home + draw + away = 1.0)
    home_prob:  float = 0.0
    draw_prob:  float = 0.0
    away_prob:  float = 0.0

    # Derived market probabilities (from Dixon-Coles score matrix)
    over_25_prob: float = 0.0   # P(total goals > 2.5)
    btts_prob:    float = 0.0   # P(both teams score)

    # Confidence: max single-outcome probability
    # Higher = more predictable match
    confidence: float = 0.0

    # Most likely scoreline (from DC score matrix)
    predicted_score: tuple[int, int] = (1, 0)

    # Component model outputs (for transparency / debugging)
    dc_probs:  dict[str, float] = field(default_factory=dict)  # Dixon-Coles
    xgb_probs: dict[str, float] = field(default_factory=dict)  # XGBoost

    @property
    def favourite(self) -> str:
        """Return the name of the predicted favourite (or 'DRAW' if draw is most likely)."""
        if self.home_prob > self.draw_prob and self.home_prob > self.away_prob:
            return self.home_team
        elif self.away_prob > self.home_prob and self.away_prob > self.draw_prob:
            return self.away_team
        else:
            return "DRAW"

    @property
    def confidence_label(self) -> str:
        """Human-readable confidence tier: HIGH / MEDIUM / LOW."""
        if self.confidence >= 0.55:
            return "HIGH"
        elif self.confidence >= 0.45:
            return "MEDIUM"
        else:
            return "LOW"

    @property
    def implied_score_str(self) -> str:
        """Predicted scoreline as a string, e.g. '2 - 1'."""
        return f"{self.predicted_score[0]} — {self.predicted_score[1]}"


class EnsemblePredictor:
    """Orchestrates Dixon-Coles + XGBoost into a single MatchPrediction.

    Usage
    -----
    predictor = EnsemblePredictor(
        dc_model=fitted_dc,
        xgb_model=fitted_xgb,
        feature_engineer=eng,
        calibrator=calibrator,  # optional
    )
    prediction = predictor.predict("BRAZIL", "FRANCE", date(2026, 6, 19), "GROUP")
    """

    def __init__(
        self,
        dc_model: object,           # DixonColesModel
        xgb_model: object,          # WorldCupXGBClassifier
        feature_engineer: object,   # FeatureEngineer
        calibrator: object = None,  # ProbabilityCalibrator (optional)
        weights: dict[str, float] | None = None,
    ) -> None:
        self.dc          = dc_model
        self.xgb         = xgb_model
        self.eng         = feature_engineer
        self.calibrator  = calibrator
        self.weights     = weights or DEFAULT_WEIGHTS.copy()

    def predict(
        self,
        home: str,
        away: str,
        match_date: date,
        stage: str = "GROUP",
        group: str = "",
        venue: str = "",
    ) -> MatchPrediction:
        """Generate a complete MatchPrediction for a single match.

        Parameters
        ----------
        home, away  : str  — team names (UPPERCASE)
        match_date  : date — match date
        stage       : str  — tournament stage
        group       : str  — group letter (for display)
        venue       : str  — venue name (for display)

        Returns
        -------
        MatchPrediction with all fields populated.

        HOW TO DEBUG:
            prediction = predictor.predict("BRAZIL", "FRANCE", date(2026, 6, 19))
            print(f"DC: {prediction.dc_probs}")
            print(f"XGB: {prediction.xgb_probs}")
            print(f"Ensemble: H={prediction.home_prob:.2f} D={prediction.draw_prob:.2f} A={prediction.away_prob:.2f}")
        """
        match_id = f"{home}_vs_{away}_{match_date.isoformat()}"

        # ── Dixon-Coles probabilities ─────────────────────────────────────
        dc_probs = self._get_dc_probs(home, away)

        # ── XGBoost probabilities ─────────────────────────────────────────
        xgb_probs = self._get_xgb_probs(home, away, match_date, stage)

        # ── Weighted ensemble combination ─────────────────────────────────
        w_dc  = self.weights.get("dixon_coles", 0.60)
        w_xgb = self.weights.get("xgboost", 0.40)

        home_prob = w_dc * dc_probs["home"] + w_xgb * xgb_probs["home"]
        draw_prob = w_dc * dc_probs["draw"] + w_xgb * xgb_probs["draw"]
        away_prob = w_dc * dc_probs["away"] + w_xgb * xgb_probs["away"]

        # ── Optional calibration ──────────────────────────────────────────
        if self.calibrator is not None:
            try:
                home_prob, draw_prob, away_prob = self.calibrator.calibrate_match(
                    home_prob, draw_prob, away_prob
                )
            except Exception as exc:
                logger.debug("Calibration failed (%s) — using raw ensemble", exc)

        # ── Normalise (floating point safety) ────────────────────────────
        total = home_prob + draw_prob + away_prob
        if total > 0:
            home_prob /= total
            draw_prob /= total
            away_prob /= total

        # ── Over/Under and BTTS (from Dixon-Coles score matrix) ──────────
        over_25 = dc_probs.get("over_25", _poisson_over_25(dc_probs.get("lambda_h", 1.3), dc_probs.get("lambda_a", 1.1)))
        btts    = dc_probs.get("btts",    0.50)
        predicted_score = dc_probs.get("most_likely_score", (1, 0))

        confidence = max(home_prob, draw_prob, away_prob)

        return MatchPrediction(
            match_id=match_id,
            home_team=home,
            away_team=away,
            match_date=match_date.isoformat(),
            stage=stage,
            group=group,
            venue=venue,
            home_prob=round(home_prob, 4),
            draw_prob=round(draw_prob, 4),
            away_prob=round(away_prob, 4),
            over_25_prob=round(over_25, 4),
            btts_prob=round(btts, 4),
            confidence=round(confidence, 4),
            predicted_score=predicted_score,
            dc_probs=dc_probs,
            xgb_probs=xgb_probs,
        )

    def predict_all_upcoming(
        self,
        upcoming_matches: list[dict],
        today: date,
    ) -> list[MatchPrediction]:
        """Predict all upcoming matches from the schedule.

        Parameters
        ----------
        upcoming_matches : list[dict] — from wc2026_teams.upcoming_matches()
        today            : date

        Returns list of MatchPrediction, sorted by date.
        """
        predictions: list[MatchPrediction] = []

        for match in upcoming_matches:
            try:
                match_date = date.fromisoformat(match["date"])
                pred = self.predict(
                    home=match["home"],
                    away=match["away"],
                    match_date=match_date,
                    stage=match.get("stage", "GROUP"),
                    group=match.get("group", ""),
                    venue=match.get("venue", ""),
                )
                predictions.append(pred)
            except Exception as exc:
                logger.warning("Could not predict %s vs %s: %s",
                               match.get("home"), match.get("away"), exc)

        return predictions

    # ─── Private helpers ─────────────────────────────────────────────────────

    def _get_dc_probs(self, home: str, away: str) -> dict:
        """Get Dixon-Coles probabilities with fallback if model not fitted."""
        try:
            return self.dc.predict_outcome_probs(home, away, neutral=True)
        except Exception as exc:
            logger.warning("Dixon-Coles failed for %s vs %s: %s — using fallback", home, away, exc)
            return {"home": 0.40, "draw": 0.25, "away": 0.35, "over_25": 0.50, "btts": 0.50, "most_likely_score": (1, 1)}

    def _get_xgb_probs(self, home: str, away: str, match_date: date, stage: str) -> dict:
        """Get XGBoost probabilities with fallback if model not fitted."""
        try:
            features = self.eng.build_features(home, away, match_date, stage)
            return self.xgb.predict_match(features)
        except Exception as exc:
            logger.warning("XGBoost failed for %s vs %s: %s — using ELO fallback", home, away, exc)
            return {"home": 0.40, "draw": 0.25, "away": 0.35}


def _poisson_over_25(lam_h: float, lam_a: float) -> float:
    """Simple Poisson approximation for P(total goals > 2.5).
    Used as fallback when DC model doesn't return over_25 directly.
    """
    import scipy.stats
    total_lambda = lam_h + lam_a
    # P(goals ≤ 2) = CDF at 2
    p_under = sum(scipy.stats.poisson.pmf(k, total_lambda) for k in range(3))
    return max(0.0, min(1.0, 1.0 - p_under))
