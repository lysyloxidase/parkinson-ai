"""1D CNN + BiLSTM model for PD voice prediction."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from parkinson_ai.ml.models.base import BasePDModel


class VoiceAttention(nn.Module):
    """Temporal attention over recurrent voice features."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_dim, 1)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a context vector and normalized attention weights."""

        scores = self.projection(inputs).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(inputs * weights.unsqueeze(-1), dim=1)
        return context, weights


class VoiceCNN(nn.Module, BasePDModel):
    """1D CNN + BiLSTM for raw voice features or MFCCs."""

    def __init__(self, input_channels: int = 13) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention = VoiceAttention(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
        self.last_attention: torch.Tensor | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the predicted PD probability."""

        features = self.pool(torch.relu(self.bn1(self.conv1(inputs))))
        features = self.pool(torch.relu(self.bn2(self.conv2(features))))
        sequence = features.transpose(1, 2)
        recurrent, _ = self.lstm(sequence)
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
    ) -> VoiceCNN:
        """Train the network on a small in-memory dataset."""

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


VoiceCnn = VoiceCNN
