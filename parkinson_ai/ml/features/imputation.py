"""Imputation utilities for missing multimodal biomarker data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.impute import KNNImputer
except ImportError:  # pragma: no cover - optional dependency
    KNNImputer = None


FloatArray = NDArray[np.float64]


def zero_imputation(data: FloatArray) -> FloatArray:
    """Replace missing values with zero."""

    return cast(FloatArray, np.asarray(np.nan_to_num(np.asarray(data, dtype=float), nan=0.0), dtype=np.float64))


def mean_imputation(data: FloatArray) -> FloatArray:
    """Replace missing values with column means."""

    array = np.asarray(data, dtype=float).copy()
    if array.ndim != 2:
        raise ValueError("data must be 2-dimensional")
    means = np.nanmean(array, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    row_indices, col_indices = np.where(np.isnan(array))
    array[row_indices, col_indices] = means[col_indices]
    return array


def knn_imputation(data: FloatArray, k: int = 5) -> FloatArray:
    """Replace missing values using k-nearest-neighbors imputation."""

    array = np.asarray(data, dtype=float)
    if KNNImputer is None:
        return mean_imputation(array)
    imputer = KNNImputer(n_neighbors=k)
    return np.asarray(imputer.fit_transform(array), dtype=float)


class BiomarkerVAEImputer(nn.Module):
    """Small VAE for correlated biomarker imputation."""

    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode inputs into latent distribution parameters."""

        hidden = self.encoder(inputs)
        return self.mu_layer(hidden), self.logvar_layer(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample a latent vector with the reparameterization trick."""

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run a forward pass through the VAE."""

        mu, logvar = self.encode(inputs)
        latent = self.reparameterize(mu, logvar)
        return self.decoder(latent), mu, logvar

    def fit(
        self,
        data: FloatArray,
        *,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: torch.device | None = None,
    ) -> BiomarkerVAEImputer:
        """Train the VAE on partially observed biomarker data."""

        device = device or torch.device("cpu")
        prepared = mean_imputation(data)
        mask = ~np.isnan(np.asarray(data, dtype=float))
        inputs = torch.tensor(prepared, dtype=torch.float32)
        observed_mask = torch.tensor(mask.astype(np.float32), dtype=torch.float32)
        loader = DataLoader(TensorDataset(inputs, observed_mask), batch_size=batch_size, shuffle=True)
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(epochs):
            self.train()
            for batch_inputs, batch_mask in loader:
                batch_inputs = batch_inputs.to(device)
                batch_mask = batch_mask.to(device)
                optimizer.zero_grad()
                recon, mu, logvar = self(batch_inputs)
                recon_loss = ((recon - batch_inputs) ** 2 * batch_mask).sum() / batch_mask.sum().clamp_min(1.0)
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + 0.01 * kl_loss
                loss.backward()
                optimizer.step()
        self.to(torch.device("cpu"))
        return self

    def transform(self, data: FloatArray) -> FloatArray:
        """Impute missing values in a dataset."""

        self.eval()
        prepared = mean_imputation(data)
        tensor = torch.tensor(prepared, dtype=torch.float32)
        with torch.no_grad():
            recon, _, _ = self(tensor)
        recon_array = recon.cpu().numpy().astype(float)
        mask = np.isnan(np.asarray(data, dtype=float))
        imputed = prepared.copy()
        imputed[mask] = recon_array[mask]
        return cast(FloatArray, np.asarray(imputed, dtype=np.float64))


@dataclass(slots=True)
class LearnedImputationResult:
    """Result from learned VAE-based imputation."""

    imputed: FloatArray
    model: BiomarkerVAEImputer


def learned_imputation(
    data: FloatArray,
    *,
    latent_dim: int = 16,
    hidden_dim: int = 64,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> LearnedImputationResult:
    """Impute missing values using a VAE trained on correlated biomarkers."""

    array = np.asarray(data, dtype=float)
    model = BiomarkerVAEImputer(input_dim=array.shape[1], latent_dim=latent_dim, hidden_dim=hidden_dim)
    model.fit(array, epochs=epochs, batch_size=batch_size, lr=lr, device=device)
    return LearnedImputationResult(imputed=model.transform(array), model=model)


def modality_dropout(modality_mask: torch.Tensor, p: float = 0.3) -> torch.Tensor:
    """Randomly drop modalities during training for robustness."""

    if p < 0.0 or p > 1.0:
        raise ValueError("p must be between 0 and 1")
    mask = modality_mask.float() if not modality_mask.dtype.is_floating_point else modality_mask.clone()
    random_mask = torch.bernoulli(torch.full_like(mask, 1.0 - p))
    dropped = mask * random_mask
    all_missing = dropped.sum(dim=1, keepdim=True) == 0
    if torch.any(all_missing):
        dropped = dropped.clone()
        fallback = torch.argmax(mask, dim=1, keepdim=True)
        dropped.scatter_(1, fallback, 1.0)
    return dropped
