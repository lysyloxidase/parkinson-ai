"""Prodromal model tests."""

from __future__ import annotations

import numpy as np

from parkinson_ai.ml.models.prodromal_model import ProdromalConversionModel


def test_prodromal_model_survival_curve() -> None:
    """Prodromal survival model should return valid survival curves."""

    rng = np.random.default_rng(4)
    features = rng.normal(size=(32, 6)).astype(np.float32)
    durations = np.linspace(1.0, 10.0, num=32).astype(np.float32)
    events = np.array([0] * 16 + [1] * 16, dtype=int)
    model = ProdromalConversionModel(input_dim=6)
    model.fit_survival(features, durations=durations, events=events, method="deepsurv", epochs=30, lr=1e-2)
    survival = model.predict_survival_curve(features[:4], times=np.array([2.0, 5.0, 8.0]))
    probabilities = model.predict_proba(features[:4])
    assert survival.shape == (4, 3)
    assert probabilities.shape == (4, 2)
    assert np.all((survival >= 0.0) & (survival <= 1.0))
