"""Gait LSTM tests."""

from __future__ import annotations

import numpy as np
import torch

from parkinson_ai.ml.models.gait_lstm import GaitLSTM


def test_gait_lstm_shapes_and_loss_decreases() -> None:
    """Gait model should learn a simple separable pattern."""

    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    features = rng.normal(size=(32, 40, 16)).astype(np.float32)
    labels = np.repeat([0, 1], repeats=16).astype(np.float32)
    features[labels == 1, :, 0] += 1.2
    model = GaitLSTM(n_sensors=16)
    criterion = torch.nn.BCELoss()
    with torch.no_grad():
        initial = criterion(model(torch.tensor(features)), torch.tensor(labels).view(-1, 1)).item()
    model.fit(features, labels, epochs=4, lr=1e-2, batch_size=8)
    with torch.no_grad():
        final = criterion(model(torch.tensor(features)), torch.tensor(labels).view(-1, 1)).item()
    assert final < initial
    assert model.predict_proba(features).shape == (32, 2)
