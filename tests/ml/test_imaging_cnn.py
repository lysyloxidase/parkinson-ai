"""Imaging CNN tests."""

from __future__ import annotations

import numpy as np
import torch

from parkinson_ai.ml.models.imaging_cnn import ImagingCNN


def test_imaging_cnn_shapes_and_loss_decreases() -> None:
    """Imaging model should learn a simple synthetic signal."""

    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    features = rng.normal(size=(16, 3, 64, 64)).astype(np.float32)
    labels = np.repeat([0, 1], repeats=8).astype(np.float32)
    features[labels == 1, :, 24:40, 24:40] += 2.0
    model = ImagingCNN(use_pretrained=False)
    criterion = torch.nn.BCELoss()
    with torch.no_grad():
        initial = criterion(model(torch.tensor(features)), torch.tensor(labels).view(-1, 1)).item()
    model.fit(features, labels, epochs=2, lr=1e-2, batch_size=4)
    with torch.no_grad():
        final = criterion(model(torch.tensor(features)), torch.tensor(labels).view(-1, 1)).item()
    assert final < initial
    assert model.predict_proba(features).shape == (16, 2)
