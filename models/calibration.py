"""
models/calibration.py — Probability Calibration
================================================
Machine learning classifiers (including XGBoost) often produce
*miscalibrated* probabilities — the model may be confident (e.g. 90%)
when the true frequency is lower (e.g. 70%). This is called overconfidence.

Isotonic regression calibration fixes this by learning a monotone mapping
from raw model scores to calibrated probabilities using a validation set.

WHY CALIBRATION MATTERS FOR BETTING:
  Kelly Criterion is very sensitive to probability estimates. An
  overconfident model will recommend bets that are too large, leading to
  higher variance and potential ruin. Well-calibrated probabilities lead to
  more accurate Kelly stakes and better long-term profitability.

HOW TO DEBUG:
  - Plot the calibration curve: model_prob on x-axis, actual_win_rate on y-axis
    → A perfect calibration shows a 45-degree diagonal line
    → Overconfident model: curve bends BELOW the diagonal
    → Underconfident model: curve bends ABOVE the diagonal
  - If calibration has no effect (probs unchanged): ensure you're passing
    a different validation set than the training set
  - If calibrated probs don't sum to 1.0: check _renormalize() is called
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Isotonic regression calibrator for 3-class match outcome probabilities.

    Applies separate isotonic regression to each class (home/draw/away),
    then renormalises so the calibrated probabilities sum to 1.

    Usage
    -----
    # 1. Train calibrator on a holdout validation set
    calibrator = ProbabilityCalibrator()
    calibrator.fit(raw_probs, true_labels)   # raw_probs shape: (n, 3)

    # 2. Calibrate predictions
    cal_home, cal_draw, cal_away = calibrator.calibrate_match(0.65, 0.20, 0.15)
    """

    def __init__(self) -> None:
        # One IsotonicRegression per class (away=0, draw=1, home=2)
        self._calibrators = [None, None, None]
        self._fitted: bool = False

    def fit(
        self,
        raw_probs: np.ndarray,
        true_labels: np.ndarray,
    ) -> "ProbabilityCalibrator":
        """Fit the calibrator on a validation set.

        Parameters
        ----------
        raw_probs   : np.ndarray shape (n, 3) — raw model probabilities
                      columns: [P(away=0), P(draw=1), P(home=2)]
        true_labels : np.ndarray shape (n,)  — true outcome labels (0/1/2)

        Returns self for method chaining.

        HOW TO DEBUG:
            If fit() raises ValueError about monotone calibration,
            ensure raw_probs is a valid probability matrix (all rows sum to 1).
        """
        from sklearn.isotonic import IsotonicRegression

        n_samples = len(true_labels)
        if n_samples < 20:
            logger.warning(
                "Only %d samples for calibration — calibrator will have little effect. "
                "Consider using a larger validation set.",
                n_samples,
            )
            # Still fit, but the calibrator won't be reliable
            self._fitted = False
            return self

        for class_idx in range(3):
            # Binary labels: 1 if this match ended in class_idx, 0 otherwise
            binary_labels = (true_labels == class_idx).astype(float)

            # Get the raw probabilities for this class
            class_probs = raw_probs[:, class_idx]

            # Fit isotonic regression (monotonically increasing mapping)
            ir = IsotonicRegression(
                out_of_bounds="clip",  # clip predictions outside training range
                increasing=True,
            )
            ir.fit(class_probs, binary_labels)
            self._calibrators[class_idx] = ir

        self._fitted = True
        logger.info("Probability calibrator fitted on %d samples", n_samples)
        return self

    def calibrate(self, raw_probs: np.ndarray) -> np.ndarray:
        """Calibrate a batch of probability matrices.

        Parameters
        ----------
        raw_probs : np.ndarray shape (n, 3)

        Returns
        -------
        np.ndarray shape (n, 3) — calibrated, normalised probabilities

        HOW TO DEBUG:
            raw = np.array([[0.70, 0.15, 0.15]])
            cal = calibrator.calibrate(raw)
            print(cal.sum(axis=1))  # should be [1.0]
        """
        if not self._fitted:
            # Return raw probs unchanged if calibrator is not fitted
            logger.debug("Calibrator not fitted — returning raw probabilities")
            return raw_probs

        calibrated = np.zeros_like(raw_probs)
        for class_idx in range(3):
            calibrated[:, class_idx] = self._calibrators[class_idx].predict(
                raw_probs[:, class_idx]
            )

        # Renormalise rows so they sum to 1.0
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid division by zero
        calibrated = calibrated / row_sums

        return calibrated

    def calibrate_match(
        self,
        home_prob: float,
        draw_prob: float,
        away_prob: float,
    ) -> Tuple[float, float, float]:
        """Calibrate probabilities for a single match.

        Parameters
        ----------
        home_prob, draw_prob, away_prob : float — raw ensemble probabilities

        Returns
        -------
        (calibrated_home, calibrated_draw, calibrated_away) tuple

        Example
        -------
        >>> cal_home, cal_draw, cal_away = calibrator.calibrate_match(0.65, 0.20, 0.15)
        >>> print(f"Calibrated home win: {cal_home:.1%}")
        """
        # Pack into matrix format required by calibrate()
        raw = np.array([[away_prob, draw_prob, home_prob]])  # order: 0=away, 1=draw, 2=home
        cal = self.calibrate(raw)[0]
        return float(cal[2]), float(cal[1]), float(cal[0])  # return as home, draw, away
