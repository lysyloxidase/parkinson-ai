"""Genomic model tests."""

from __future__ import annotations

import numpy as np
import torch

from parkinson_ai.ml.models.genomic_model import GenomicRiskModel


def test_genomic_model_shapes_and_loss_decreases() -> None:
    """Genomic model should learn a simple PRS-plus-variant pattern."""

    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    prs = rng.normal(size=(48, 1))
    age = rng.normal(loc=65.0, scale=7.0, size=(48, 1))
    variants = rng.integers(0, 2, size=(48, 6))
    pathways = rng.normal(size=(48, 5))
    features = np.concatenate([prs, age, variants, pathways], axis=1).astype(np.float32)
    labels = (prs.squeeze() + variants[:, 0] > np.median(prs.squeeze() + variants[:, 0])).astype(np.float32)
    model = GenomicRiskModel(variant_dim=6, pathway_dim=5)
    criterion = torch.nn.BCELoss()
    tensor_features = torch.tensor(features)
    with torch.no_grad():
        initial = criterion(model(*model._split_flat_features(tensor_features)), torch.tensor(labels).view(-1, 1)).item()
    model.fit(features, labels, epochs=4, lr=1e-2, batch_size=8)
    with torch.no_grad():
        final = criterion(model(*model._split_flat_features(tensor_features)), torch.tensor(labels).view(-1, 1)).item()
    assert final < initial
    assert model.predict_proba(features).shape == (48, 2)
