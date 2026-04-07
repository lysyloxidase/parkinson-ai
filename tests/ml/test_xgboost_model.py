"""Tabular model tests."""

from __future__ import annotations

import numpy as np

from parkinson_ai.ml.models.xgboost_model import XGBoostPDModel


def test_xgboost_model_fit_predict() -> None:
    """The tabular model should fit, cross-validate, and expose importances."""

    rng = np.random.default_rng(42)
    features = rng.normal(size=(40, 6))
    labels = (features[:, 0] + features[:, 1] > 0.0).astype(int)
    feature_names = [f"feature_{index}" for index in range(features.shape[1])]
    model = XGBoostPDModel(n_estimators=20, max_depth=3, feature_names=feature_names).fit(features, labels)
    predictions = model.predict_proba(features)
    importance = model.feature_importance(features=features, feature_names=feature_names)
    cv_results = model.cross_validate(features, labels, n_splits=5)
    assert predictions.shape == (40, 2)
    assert {"feature", "gain", "shap_importance"}.issubset(importance.columns)
    assert len(cv_results) == 6
