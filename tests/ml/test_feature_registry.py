"""Feature registry tests."""

from __future__ import annotations

import torch

from parkinson_ai.ml.features.feature_registry import PDFeatureRegistry


def test_registry_size_and_vector_building() -> None:
    """The registry should expose the requested 91-feature core vector."""

    registry = PDFeatureRegistry()
    patient = {
        "saa_result": True,
        "nfl_pg_ml": 18.0,
        "prs_score": 1.2,
        "age": 67,
        "left_putamen_sbr": 1.2,
        "right_putamen_sbr": 1.0,
        "updrs_part3": 18.0,
        "rbd_present": True,
        "prodromal_risk_score": 0.72,
    }
    vector = registry.build_vector(patient)
    mask = registry.get_modality_mask(patient)
    assert registry.count() == 91
    assert registry.count(include_extended=True) > registry.count()
    assert vector.shape == (91,)
    assert mask.shape == (7,)
    assert torch.sum(mask).item() >= 5
    assert "genetic_age_prs_interaction" in registry.get_available(patient)
