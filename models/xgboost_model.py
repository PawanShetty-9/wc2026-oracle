"""
models/xgboost_model.py — XGBoost 3-Class Match Outcome Classifier
===================================================================
XGBoost (eXtreme Gradient Boosting) is an ensemble of decision trees
trained to maximise log-likelihood on the 3-class outcome problem:
    Class 2 = Home Win
    Class 1 = Draw
    Class 0 = Away Win

WHY XGBOOST?
  - Handles mixed feature scales without normalisation
  - Built-in regularisation (L1/L2) reduces overfitting on small datasets
  - Feature importance is interpretable
  - Consistently outperforms random forests on tabular sports data
  - Much faster to train than neural networks for this dataset size

HYPERPARAMETERS:
  - max_depth=4: shallow trees reduce overfitting (international football
    has high variance — deep trees memorise noise)
  - learning_rate=0.05: slow learning = more robust (compensated by more trees)
  - n_estimators=500: with early stopping, actual trees used ≈ 100-200
  - subsample=0.8, colsample_bytree=0.8: stochastic gradient boosting
    reduces variance

PERSISTENCE:
  Model is saved as a joblib file at betting/data/raw/xgb_model.joblib.
  If the file doesn't exist on cold start, the app auto-trains it.
  Training takes ~10-30 seconds on CPU.

HOW TO DEBUG:
  - If accuracy < 45%, the model may be overfitting — reduce max_depth to 3
  - If predict_match() returns uniform [0.33, 0.33, 0.33], the model may
    not have loaded correctly — check that FEATURE_NAMES match training
  - Feature importance plot shows which features contribute most —
    if elo_diff isn't near the top, something is wrong with ELO training
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Where to save/load the trained model
_MODEL_PATH = Path(__file__).parent.parent / "data" / "raw" / "xgb_model.joblib"

from features.engineering import FEATURE_NAMES


class WorldCupXGBClassifier:
    """XGBoost classifier for 3-way match outcome prediction.

    Labels: 0=away_win, 1=draw, 2=home_win
    (matches the 'outcome' column from historical.py)
    """

    def __init__(self) -> None:
        # Lazy import xgboost to avoid slow import at module level
        import xgboost as xgb
        self._model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            early_stopping_rounds=50,
            verbosity=0,
            n_jobs=-1,
        )
        self._fitted: bool = False

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_fraction: float = 0.15,
    ) -> "WorldCupXGBClassifier":
        """Train the XGBoost classifier.

        Parameters
        ----------
        X             : pd.DataFrame — feature matrix from FeatureEngineer
        y             : pd.Series   — outcome labels (0/1/2)
        eval_fraction : float       — fraction of data held out for early stopping

        Returns self for method chaining.

        HOW TO DEBUG:
            Call fit() and check self._model.best_iteration to see how many
            trees were actually used (should be 50-200 for typical WC data).
        """
        from sklearn.model_selection import train_test_split

        # Ensure features are in the correct order
        X = X[FEATURE_NAMES]

        # Hold out a validation set for early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=eval_fraction,
            random_state=42,
            stratify=y,       # maintain class distribution
        )

        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        self._fitted = True

        n_trees = getattr(self._model, "best_iteration", self._model.n_estimators)
        logger.info(
            "XGBoost trained: %d trees, val_logloss=%.4f",
            n_trees,
            self._model.best_score if hasattr(self._model, "best_score") else 0.0,
        )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability matrix of shape (n_samples, 3).

        Columns: [P(away_win), P(draw), P(home_win)]

        HOW TO DEBUG:
            proba = model.predict_proba(X)
            print(proba.shape)  # should be (n_samples, 3)
            print(proba.sum(axis=1))  # should all be ≈ 1.0
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_proba()")
        X = X[FEATURE_NAMES]
        return self._model.predict_proba(X)

    def predict_match(self, features: dict) -> dict[str, float]:
        """Predict a single match from a feature dict.

        Parameters
        ----------
        features : dict — output from FeatureEngineer.build_features()

        Returns
        -------
        dict with keys 'home', 'draw', 'away' → probability floats

        Example
        -------
        >>> probs = xgb_model.predict_match(features)
        >>> print(f"Home win: {probs['home']:.1%}")
        """
        X = pd.DataFrame([features])[FEATURE_NAMES]
        proba = self._model.predict_proba(X)[0]
        # XGBoost returns [P(0), P(1), P(2)] = [away_win, draw, home_win]
        return {
            "away": float(proba[0]),
            "draw": float(proba[1]),
            "home": float(proba[2]),
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importances as a sorted DataFrame.

        Uses XGBoost's 'gain' importance (how much each feature reduces
        training loss on average when used in a split).

        Returns pd.DataFrame with columns: feature, importance (sorted desc).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before get_feature_importance()")

        importances = self._model.feature_importances_
        df = pd.DataFrame({
            "feature":    FEATURE_NAMES,
            "importance": importances,
        })
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: Path = _MODEL_PATH) -> None:
        """Save the trained model to disk using joblib.

        HOW TO DEBUG:
            If save() fails with PermissionError, check that the
            betting/data/raw/ directory exists and is writable.
        """
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("XGBoost model saved to %s", path)

    @classmethod
    def load(cls, path: Path = _MODEL_PATH) -> "WorldCupXGBClassifier":
        """Load a previously saved model from disk.

        Raises FileNotFoundError if no saved model exists.
        Use load_or_train() for a safer version that auto-trains if missing.
        """
        import joblib
        obj = joblib.load(path)
        logger.info("XGBoost model loaded from %s", path)
        return obj


def load_or_train_xgb(
    feature_engineer: object,
    training_df: pd.DataFrame,
    force_retrain: bool = False,
) -> WorldCupXGBClassifier:
    """Load a saved model or train a new one if not found.

    This is the primary entry point called by app.py on startup.

    Parameters
    ----------
    feature_engineer : FeatureEngineer — already fitted to training data
    training_df      : pd.DataFrame   — same df used to fit ELO/DC models
    force_retrain    : bool           — if True, always retrain even if saved

    Returns
    -------
    WorldCupXGBClassifier — fitted and ready to predict

    HOW TO DEBUG:
        - If training takes > 60 seconds: reduce training_df to post-2010 data
        - If you suspect a stale/corrupted model: set force_retrain=True
          or delete betting/data/raw/xgb_model.joblib manually
    """
    # Try to load an existing saved model first (fast path)
    if not force_retrain and _MODEL_PATH.exists():
        try:
            return WorldCupXGBClassifier.load()
        except Exception as exc:
            logger.warning("Could not load saved model (%s) — retraining", exc)

    logger.info("Training XGBoost model from scratch...")

    # Build training features (walk-forward)
    X, y = feature_engineer.build_training_set(training_df)

    # Guard: need at least 3 samples per class for stratified split
    class_counts = y.value_counts()
    if class_counts.min() < 10:
        logger.warning(
            "Very few samples for some classes: %s. "
            "Model accuracy may be low.",
            class_counts.to_dict(),
        )

    model = WorldCupXGBClassifier()
    model.fit(X, y)
    model.save()

    return model
