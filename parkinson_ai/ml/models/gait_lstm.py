"""BiLSTM gait model for PD sensor time series."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from parkinson_ai.ml.models.base import BasePDModel


class TemporalAttention(nn.Module):
    """Temporal attention for sequence embeddings."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate sequence features with attention."""

        scores = self.projection(inputs).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(inputs * weights.unsqueeze(-1), dim=1)
        return context, weights


class GaitLSTM(nn.Module, BasePDModel):
    """BiLSTM for gait sensor time series."""

    def __init__(self, n_sensors: int = 16) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_sensors,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.attention = TemporalAttention(128)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.last_attention: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return gait-based PD probability."""

        recurrent, _ = self.lstm(inputs)
        context, weights = self.attention(recurrent)
        self.last_attention = weights
        logits = self.classifier(context)
        return torch.sigmoid(logits)

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        epochs: int = 5,
        lr: float = 1e-3,
        batch_size: int = 16,
        device: torch.device | None = None,
    ) -> GaitLSTM:
        """Train the gait network."""

        device = device or torch.device("cpu")
        x_train = torch.tensor(features, dtype=torch.float32)
        y_train = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        for _ in range(epochs):
            self.train()
            for batch_inputs, batch_labels in loader:
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                predictions = self(batch_inputs)
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()
        self.to(torch.device("cpu"))
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""

        self.eval()
        inputs = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            probabilities = self(inputs).squeeze(-1).cpu().numpy()
        return np.stack([1.0 - probabilities, probabilities], axis=1)


GaitLstm = GaitLSTM
