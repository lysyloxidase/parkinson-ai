"""XGBoost-based tabular model for PD prediction."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from parkinson_ai.ml.evaluation import evaluate_binary_classifier
from parkinson_ai.ml.models.base import BasePDModel

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
except ImportError:  # pragma: no cover - optional dependency
    LogisticRegression = None
    StratifiedKFold = None


FloatArray = NDArray[np.float64]


def _load_module(name: str) -> ModuleType | None:
    """Load an optional dependency."""

    try:
        return importlib.import_module(name)
    except ImportError:  # pragma: no cover - optional dependency
        return None


xgboost_module = _load_module("xgboost")
shap_module = _load_module("shap")


class XGBoostPDModel(BasePDModel):
    """XGBoost for clinical, molecular, and genetic PD tabular features."""

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        random_state: int = 42,
        scale_pos_weight: float | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.scale_pos_weight = scale_pos_weight
        self.feature_names = feature_names
        self.model: Any | None = None
        self.scale_pos_weight_: float = 1.0

    def fit(self, features: np.ndarray, labels: np.ndarray) -> XGBoostPDModel:
        """Fit the XGBoost model."""

        self.train(features, labels)
        return self

    def train(self, features: np.ndarray, labels: np.ndarray) -> XGBoostPDModel:
        """Train the underlying classifier."""

        x_train = np.asarray(features, dtype=float)
        y_train = np.asarray(labels, dtype=int)
        positives = max(int(np.sum(y_train == 1)), 1)
        negatives = max(int(np.sum(y_train == 0)), 1)
        self.scale_pos_weight_ = self.scale_pos_weight or (negatives / positives)
        self.model = self._build_model()
        self.model.fit(x_train, y_train)
        if self.feature_names is None:
            self.feature_names = [f"feature_{index}" for index in range(x_train.shape[1])]
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict class labels."""

        probabilities = self.predict_proba(features)[:, 1]
        return (probabilities >= 0.5).astype(int)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""

        if self.model is None:
            raise RuntimeError("Model must be fit before prediction.")
        probabilities = self.model.predict_proba(np.asarray(features, dtype=float))
        return np.asarray(probabilities, dtype=float)

    def feature_importance(
        self,
        features: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> pd.DataFrame:
        """Return gain and SHAP feature importance scores."""

        if self.model is None:
            raise RuntimeError("Model must be fit before computing importance.")
        names = feature_names or self.feature_names
        if names is None:
            raise ValueError("feature_names must be provided.")
        gain_scores = self._gain_importance(names)
        shap_scores = self._shap_importance(np.asarray(features, dtype=float), names) if features is not None else {}
        frame = pd.DataFrame({"feature": names})
        frame["gain"] = frame["feature"].map(gain_scores).fillna(0.0)
        frame["shap_importance"] = frame["feature"].map(shap_scores).fillna(0.0)
        frame["importance_rank"] = frame["shap_importance"].rank(ascending=False, method="dense").fillna(frame["gain"].rank(ascending=False, method="dense")).astype(int)
        return frame.sort_values(["shap_importance", "gain"], ascending=False).reset_index(drop=True)

    def cross_validate(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        n_splits: int = 5,
    ) -> pd.DataFrame:
        """Run stratified cross-validation and return per-fold metrics."""

        if StratifiedKFold is None:
            raise RuntimeError("scikit-learn is required for cross-validation.")
        x_data = np.asarray(features, dtype=float)
        y_data = np.asarray(labels, dtype=int)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        rows: list[dict[str, float | int]] = []
        for fold, (train_index, test_index) in enumerate(splitter.split(x_data, y_data), start=1):
            model = XGBoostPDModel(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                scale_pos_weight=self.scale_pos_weight,
                feature_names=self.feature_names,
            )
            model.fit(x_data[train_index], y_data[train_index])
            probabilities = model.predict_proba(x_data[test_index])[:, 1]
            metrics = evaluate_binary_classifier(y_data[test_index], probabilities, n_bootstrap=64)
            rows.append(
                {
                    "fold": fold,
                    "auroc": float(metrics["auroc"]),
                    "auprc": float(metrics["auprc"]),
                    "sensitivity": float(metrics["sensitivity"]),
                    "specificity": float(metrics["specificity"]),
                    "brier_score": float(metrics["brier_score"]),
                }
            )
        summary = pd.DataFrame(rows)
        mean_row: dict[str, float | int] = {"fold": -1}
        for column in ("auroc", "auprc", "sensitivity", "specificity", "brier_score"):
            mean_row[column] = float(summary[column].mean())
        return pd.concat([summary, pd.DataFrame([mean_row])], ignore_index=True)

    def _build_model(self) -> Any:
        """Build the underlying estimator."""

        if xgboost_module is not None:
            return xgboost_module.XGBClassifier(
                objective="binary:logistic",
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                scale_pos_weight=self.scale_pos_weight_,
                random_state=self.random_state,
                eval_metric="logloss",
                tree_method="hist",
            )
        if LogisticRegression is None:
            raise RuntimeError("Neither xgboost nor scikit-learn is available.")
        return LogisticRegression(max_iter=500, random_state=self.random_state)

    def _gain_importance(self, feature_names: list[str]) -> dict[str, float]:
        """Return gain-based feature importance."""

        if self.model is None:
            raise RuntimeError("Model must be fit before computing importance.")
        model = self.model
        if xgboost_module is not None and hasattr(self.model, "get_booster"):
            booster = model.get_booster()
            raw_scores = booster.get_score(importance_type="gain")
            return {feature_names[int(key[1:])] if key.startswith("f") and key[1:].isdigit() else key: float(value) for key, value in raw_scores.items()}
        if hasattr(model, "coef_"):
            coefficients = np.abs(np.asarray(model.coef_).reshape(-1))
            return {feature_name: float(coefficients[index]) for index, feature_name in enumerate(feature_names)}
        return {feature_name: 0.0 for feature_name in feature_names}

    def _shap_importance(self, features: np.ndarray, feature_names: list[str]) -> dict[str, float]:
        """Return mean absolute SHAP values when SHAP is available."""

        if shap_module is None or self.model is None:
            return {feature_name: 0.0 for feature_name in feature_names}
        model = self.model
        try:
            explainer = shap_module.TreeExplainer(model)
            shap_values = explainer.shap_values(features)
        except Exception:  # pragma: no cover - SHAP backend differences
            return {feature_name: 0.0 for feature_name in feature_names}
        shap_array = np.asarray(shap_values)
        if shap_array.ndim == 3:
            shap_array = shap_array[..., 1]
        if shap_array.ndim == 1:
            shap_array = shap_array.reshape(1, -1)
        mean_abs = np.mean(np.abs(shap_array), axis=0)
        return {feature_names[index]: float(mean_abs[index]) for index in range(len(feature_names))}
