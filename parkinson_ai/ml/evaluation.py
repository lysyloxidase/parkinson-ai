"""Comprehensive evaluation helpers for PD prediction models."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import torch


def _load_module(name: str) -> ModuleType | None:
    """Load an optional module."""

    try:
        return importlib.import_module(name)
    except ImportError:  # pragma: no cover - optional dependency
        return None


sklearn_metrics = _load_module("sklearn.metrics")
sklearn_calibration = _load_module("sklearn.calibration")


def optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Return the Youden-optimal decision threshold."""

    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    candidate_thresholds = np.unique(scores)
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in candidate_thresholds:
        predictions = (scores >= threshold).astype(int)
        tp = np.sum((predictions == 1) & (truth == 1))
        tn = np.sum((predictions == 0) & (truth == 0))
        fp = np.sum((predictions == 1) & (truth == 0))
        fn = np.sum((predictions == 0) & (truth == 1))
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        score = sensitivity + specificity - 1.0
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def calibration_statistics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Return Brier score and calibration curve arrays."""

    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    brier_score = float(np.mean((scores - truth) ** 2))
    if sklearn_calibration is not None:
        prob_true, prob_pred = sklearn_calibration.calibration_curve(truth, scores, n_bins=n_bins)
        return {
            "brier_score": brier_score,
            "calibration_curve_true": prob_true,
            "calibration_curve_pred": prob_pred,
        }
    return {
        "brier_score": brier_score,
        "calibration_curve_true": np.asarray([], dtype=float),
        "calibration_curve_pred": np.asarray([], dtype=float),
    }


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Estimate a confidence interval via bootstrap resampling."""

    rng = np.random.default_rng(seed)
    truth = np.asarray(y_true)
    scores = np.asarray(y_score)
    metrics: list[float] = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(truth), len(truth))
        sample_truth = truth[indices]
        sample_scores = scores[indices]
        if len(np.unique(sample_truth)) < 2:
            continue
        metrics.append(float(metric_fn(sample_truth, sample_scores)))
    if not metrics:
        value = float(metric_fn(truth, scores))
        return value, value
    lower_q = (1.0 - confidence_level) / 2.0
    upper_q = 1.0 - lower_q
    return float(np.quantile(metrics, lower_q)), float(np.quantile(metrics, upper_q))


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float | None = None,
    *,
    n_bins: int = 10,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute a comprehensive set of binary classification metrics."""

    truth = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    decision_threshold = threshold if threshold is not None else optimal_threshold(truth, scores)
    predictions = (scores >= decision_threshold).astype(int)
    tp = int(np.sum((predictions == 1) & (truth == 1)))
    tn = int(np.sum((predictions == 0) & (truth == 0)))
    fp = int(np.sum((predictions == 1) & (truth == 0)))
    fn = int(np.sum((predictions == 0) & (truth == 1)))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    metrics: dict[str, Any] = {
        "threshold": float(decision_threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
    }
    if sklearn_metrics is not None:
        metrics["auroc"] = float(sklearn_metrics.roc_auc_score(truth, scores))
        metrics["auprc"] = float(sklearn_metrics.average_precision_score(truth, scores))
    else:
        metrics["auroc"] = float((sensitivity + specificity) / 2.0)
        metrics["auprc"] = float(precision)
    metrics.update(calibration_statistics(truth, scores, n_bins=n_bins))
    metrics["auroc_ci"] = bootstrap_confidence_interval(
        truth,
        scores,
        lambda yt, ys: float(metrics["auroc"]) if sklearn_metrics is None else float(sklearn_metrics.roc_auc_score(yt, ys)),
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return metrics


def per_modality_ablation(
    model: torch.nn.Module,
    patient_data: dict[str, torch.Tensor],
    modality_mask: torch.Tensor,
    y_true: np.ndarray,
) -> pd.DataFrame:
    """Evaluate leave-one-modality-out ablation for fusion-style models."""

    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        baseline_outputs = model(patient_data, modality_mask)
    baseline_prob = _extract_probability(baseline_outputs)
    baseline_metrics = evaluate_binary_classifier(y_true, baseline_prob, n_bootstrap=64)
    rows.append({"modality": "all", "auroc": baseline_metrics["auroc"], "auprc": baseline_metrics["auprc"]})
    for index in range(modality_mask.shape[1]):
        ablated_mask = modality_mask.clone()
        ablated_mask[:, index] = 0.0
        with torch.no_grad():
            outputs = model(patient_data, ablated_mask)
        probabilities = _extract_probability(outputs)
        metrics = evaluate_binary_classifier(y_true, probabilities, n_bootstrap=64)
        rows.append({"modality": index, "auroc": metrics["auroc"], "auprc": metrics["auprc"]})
    return pd.DataFrame(rows)


def per_biomarker_importance(
    model: Any,
    *,
    features: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    attention: np.ndarray | None = None,
    modality_names: list[str] | None = None,
) -> pd.DataFrame:
    """Return biomarker or modality importance from model-specific explainers."""

    if hasattr(model, "feature_importance") and features is not None and feature_names is not None:
        return model.feature_importance(features=features, feature_names=feature_names)
    if attention is not None and modality_names is not None:
        mean_attention = np.mean(attention, axis=0)
        frame = pd.DataFrame({"feature": modality_names, "attention": mean_attention})
        return frame.sort_values("attention", ascending=False).reset_index(drop=True)
    return pd.DataFrame(columns=["feature", "importance"])


def compare_models(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Compare model metrics in a single sortable table."""

    rows: list[dict[str, Any]] = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)
    frame = pd.DataFrame(rows)
    if "auroc" in frame.columns:
        frame = frame.sort_values("auroc", ascending=False)
    return frame.reset_index(drop=True)


def _extract_probability(outputs: dict[str, torch.Tensor] | torch.Tensor) -> np.ndarray:
    """Extract a binary PD probability from fusion-style outputs."""

    if isinstance(outputs, dict):
        diagnosis = outputs["diagnosis_prob"]
        if diagnosis.shape[-1] == 1:
            probability = diagnosis.squeeze(-1)
        else:
            probability = diagnosis[:, -1]
    else:
        probability = outputs.squeeze(-1)
    return probability.detach().cpu().numpy().astype(float)
