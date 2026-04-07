"""Explainability utilities for PD prediction models."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure

from parkinson_ai.core.llm_client import OllamaClient


def _load_module(name: str) -> ModuleType | None:
    """Load an optional dependency."""

    try:
        return importlib.import_module(name)
    except ImportError:  # pragma: no cover - optional dependency
        return None


shap_module = _load_module("shap")


def tree_shap_explanation(
    model: Any,
    features: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return SHAP values for tree-based models."""

    if shap_module is None or not hasattr(model, "model"):
        return pd.DataFrame({"feature": feature_names, "shap_importance": np.zeros(len(feature_names))})
    explainer = shap_module.TreeExplainer(model.model)
    shap_values = explainer.shap_values(features)
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[..., 1]
    mean_abs = np.mean(np.abs(shap_array), axis=0)
    return pd.DataFrame({"feature": feature_names, "shap_importance": mean_abs}).sort_values(
        "shap_importance",
        ascending=False,
    )


def deep_shap_explanation(
    model: torch.nn.Module,
    background: torch.Tensor,
    samples: torch.Tensor,
) -> np.ndarray:
    """Return SHAP values for neural models when DeepExplainer is available."""

    if shap_module is None:
        return np.zeros_like(samples.detach().cpu().numpy())
    explainer = shap_module.DeepExplainer(model, background)
    shap_values = explainer.shap_values(samples)
    return np.asarray(shap_values)


def extract_attention_weights(
    model: torch.nn.Module,
    patient_data: dict[str, torch.Tensor],
    modality_mask: torch.Tensor,
) -> np.ndarray:
    """Extract modality attention from a fusion-style model."""

    model.eval()
    with torch.no_grad():
        outputs = model(patient_data, modality_mask)
    if not isinstance(outputs, dict) or "modality_attention" not in outputs:
        raise ValueError("Model does not expose modality_attention.")
    return np.asarray(outputs["modality_attention"].detach().cpu().numpy())


def modality_importance_ranking(
    attention_weights: np.ndarray,
    modality_names: list[str],
) -> list[tuple[str, float]]:
    """Rank modalities by mean attention."""

    mean_attention = np.mean(attention_weights, axis=0)
    ranking = list(zip(modality_names, mean_attention.tolist(), strict=False))
    ranking.sort(key=lambda item: item[1], reverse=True)
    return ranking


def biomarker_waterfall_plot(
    feature_names: list[str],
    contributions: np.ndarray,
    values: np.ndarray,
    *,
    top_k: int = 10,
    title: str = "Patient-level biomarker contributions",
) -> Figure:
    """Create a waterfall-style horizontal bar plot."""

    indices = np.argsort(np.abs(contributions))[::-1][:top_k]
    selected_names = [feature_names[index] for index in indices]
    selected_contributions = contributions[indices]
    selected_values = values[indices]
    labels = [f"{name} ({value:.2f})" for name, value in zip(selected_names, selected_values, strict=False)]
    figure, axis = plt.subplots(figsize=(8, 5))
    colors = ["#a54d2d" if value >= 0 else "#436e68" for value in selected_contributions]
    axis.barh(labels[::-1], selected_contributions[::-1], color=colors[::-1])
    axis.set_title(title)
    axis.set_xlabel("Contribution")
    figure.tight_layout()
    return figure


def generate_patient_explanation_text(
    feature_names: list[str],
    contributions: np.ndarray,
    values: np.ndarray,
    *,
    prediction_label: str,
    top_k: int = 5,
) -> str:
    """Generate a concise text explanation without an LLM."""

    indices = np.argsort(np.abs(contributions))[::-1][:top_k]
    parts = [f"{feature_names[index]}={values[index]:.2f} ({'up' if contributions[index] >= 0 else 'down'} impact {contributions[index]:.3f})" for index in indices]
    joined = "; ".join(parts)
    return f"Prediction: {prediction_label}. Top drivers: {joined}."


async def explain_patient_with_llm(
    feature_names: list[str],
    contributions: np.ndarray,
    values: np.ndarray,
    *,
    prediction_label: str,
    llm_client: OllamaClient | None = None,
    top_k: int = 5,
) -> str:
    """Generate a patient-level explanation using a local LLM when available."""

    base_text = generate_patient_explanation_text(
        feature_names,
        contributions,
        values,
        prediction_label=prediction_label,
        top_k=top_k,
    )
    if llm_client is None:
        return base_text
    response = await llm_client.generate(prompt=(f"Rewrite this PD model explanation in clear clinical research English without overclaiming:\n{base_text}"))
    return response.response.strip() or base_text
