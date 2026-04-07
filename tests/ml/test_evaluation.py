"""Evaluation tests."""

from __future__ import annotations

import numpy as np

from parkinson_ai.ml.evaluation import compare_models, evaluate_binary_classifier, optimal_threshold


def test_evaluation_metrics() -> None:
    """Metrics should include calibration and confidence intervals."""

    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = evaluate_binary_classifier(y_true, y_score, n_bootstrap=32)
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert "brier_score" in metrics
    assert len(metrics["auroc_ci"]) == 2
    assert 0.0 <= optimal_threshold(y_true, y_score) <= 1.0


def test_compare_models() -> None:
    """Model comparison should produce a sorted data frame."""

    frame = compare_models(
        {
            "xgboost": {"auroc": 0.91, "auprc": 0.90},
            "fusion": {"auroc": 0.95, "auprc": 0.94},
        }
    )
    assert list(frame["model"]) == ["fusion", "xgboost"]
