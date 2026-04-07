"""Explainability tests."""

from __future__ import annotations

import numpy as np

from parkinson_ai.ml.explainability import generate_patient_explanation_text, modality_importance_ranking


def test_explainability_helpers() -> None:
    """Attention ranking and text explanations should be well-formed."""

    attention = np.array([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
    ranking = modality_importance_ranking(attention, ["molecular", "clinical", "digital"])
    text = generate_patient_explanation_text(
        ["nfl", "saa", "updrs"],
        np.array([0.4, -0.2, 0.1]),
        np.array([18.0, 1.0, 24.0]),
        prediction_label="PD",
    )
    assert ranking[0][0] == "molecular"
    assert "Prediction: PD." in text
