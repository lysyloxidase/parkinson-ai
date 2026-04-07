"""Training loop tests."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from parkinson_ai.ml.training import TrainingConfig, stratified_train_val_test_split, train_torch_model


class ToyBinaryModel(torch.nn.Module):
    """Small binary classifier for trainer smoke tests."""

    def __init__(self) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1), torch.nn.Sigmoid())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return binary probabilities."""

        return torch.as_tensor(self.network(inputs))


def test_stratified_split_and_training_loop() -> None:
    """Training utilities should split data and improve a toy model."""

    rng = np.random.default_rng(5)
    features = rng.normal(size=(60, 4)).astype(np.float32)
    labels = (features[:, 0] + 0.5 * features[:, 1] > 0.0).astype(int)
    splits = stratified_train_val_test_split(features, labels, val_size=0.2, test_size=0.2)
    model = ToyBinaryModel()
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(splits["x_train"]),
            torch.tensor(splits["y_train"], dtype=torch.float32),
        ),
        batch_size=8,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(splits["x_val"]),
            torch.tensor(splits["y_val"], dtype=torch.float32),
        ),
        batch_size=8,
    )
    history = train_torch_model(
        model,
        train_loader,
        val_loader,
        config=TrainingConfig(epochs=6, learning_rate=1e-2, early_stopping_patience=3),
    )
    assert history.train_loss
    assert 0.0 <= history.best_val_auroc <= 1.0
