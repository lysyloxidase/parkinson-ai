"""Imputation tests."""

from __future__ import annotations

import numpy as np
import torch

from parkinson_ai.ml.features.imputation import (
    knn_imputation,
    learned_imputation,
    mean_imputation,
    modality_dropout,
    zero_imputation,
)


def test_imputation_strategies() -> None:
    """Imputation strategies should remove NaNs."""

    data = np.array([[1.0, np.nan, 3.0], [2.0, 5.0, np.nan], [np.nan, 4.0, 6.0]], dtype=float)
    assert not np.isnan(zero_imputation(data)).any()
    assert not np.isnan(mean_imputation(data)).any()
    assert not np.isnan(knn_imputation(data, k=2)).any()
    result = learned_imputation(data, epochs=10, latent_dim=2, hidden_dim=8)
    assert not np.isnan(result.imputed).any()


def test_modality_dropout_preserves_at_least_one_modality() -> None:
    """Dropout should not erase every modality in a sample."""

    torch.manual_seed(0)
    mask = torch.ones((4, 7))
    dropped = modality_dropout(mask, p=0.9)
    assert torch.all(torch.sum(dropped, dim=1) >= 1)
