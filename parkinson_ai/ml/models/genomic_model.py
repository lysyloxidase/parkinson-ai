"""Genomic PD risk model combining PRS and variant embeddings."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from parkinson_ai.ml.models.base import BasePDModel


class GenomicRiskModel(nn.Module, BasePDModel):
    """PRS computation and variant embedding model."""

    def __init__(self, variant_dim: int = 6, pathway_dim: int = 5, embed_dim: int = 32) -> None:
        super().__init__()
        self.variant_dim = variant_dim
        self.pathway_dim = pathway_dim
        self.variant_embedding = nn.Linear(variant_dim, embed_dim)
        self.pathway_embedding = nn.Linear(pathway_dim, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 3, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        prs: torch.Tensor,
        variant_features: torch.Tensor,
        pathway_scores: torch.Tensor,
        age: torch.Tensor,
    ) -> torch.Tensor:
        """Return genomic PD risk probability."""

        variant_embedding = torch.relu(self.variant_embedding(variant_features))
        pathway_embedding = torch.relu(self.pathway_embedding(pathway_scores))
        interaction = prs * age
        merged = torch.cat(
            [variant_embedding, pathway_embedding, prs, age, interaction],
            dim=1,
        )
        logits = self.mlp(merged)
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
    ) -> GenomicRiskModel:
        """Train the genomic model from flat features."""

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
                predictions = self(*self._split_flat_features(batch_inputs))
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()
        self.to(torch.device("cpu"))
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict class probabilities from flat features."""

        self.eval()
        inputs = torch.tensor(features, dtype=torch.float32)
        with torch.no_grad():
            probabilities = self(*self._split_flat_features(inputs)).squeeze(-1).cpu().numpy()
        return np.stack([1.0 - probabilities, probabilities], axis=1)

    def _split_flat_features(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split a flat genomic vector into semantic tensors."""

        prs = features[:, 0:1]
        age = features[:, 1:2]
        variants = features[:, 2 : 2 + self.variant_dim]
        pathways = features[:, 2 + self.variant_dim : 2 + self.variant_dim + self.pathway_dim]
        return prs, variants, pathways, age


GenomicModel = GenomicRiskModel
