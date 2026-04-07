"""Transfer-learning imaging model for DaTSCAN or MRI slices."""

from __future__ import annotations

import importlib
from types import ModuleType

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from parkinson_ai.ml.models.base import BasePDModel


def _load_torchvision() -> ModuleType | None:
    """Load torchvision when available."""

    try:
        return importlib.import_module("torchvision")
    except ImportError:  # pragma: no cover - optional dependency
        return None


torchvision_module = _load_torchvision()


class ImagingCNN(nn.Module, BasePDModel):
    """EfficientNet-B0 with a fallback lightweight CNN when torchvision is unavailable."""

    def __init__(self, *, use_pretrained: bool = False) -> None:
        super().__init__()
        self.backbone, self.classifier = self._build_model(use_pretrained=use_pretrained)

    def _build_model(self, *, use_pretrained: bool) -> tuple[nn.Module, nn.Module]:
        """Construct the imaging model backbone."""

        if torchvision_module is not None:
            models = torchvision_module.models
            weights = None
            if use_pretrained:
                try:
                    weights = models.EfficientNet_B0_Weights.DEFAULT
                except Exception:  # pragma: no cover - local torchvision state
                    weights = None
            backbone = models.efficientnet_b0(weights=weights)
            for parameter in backbone.parameters():
                parameter.requires_grad = False
            for block in list(backbone.features)[-2:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            in_features = int(backbone.classifier[1].in_features)
            backbone.classifier = nn.Identity()
            classifier = nn.Linear(in_features, 1)
            return backbone, classifier
        backbone = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        classifier = nn.Linear(64, 1)
        return backbone, classifier

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the predicted PD probability from imaging features."""

        features = self.backbone(inputs)
        if features.ndim == 4:
            features = features.flatten(start_dim=1)
        logits = self.classifier(features)
        return torch.sigmoid(logits)

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        epochs: int = 3,
        lr: float = 1e-3,
        batch_size: int = 8,
        device: torch.device | None = None,
    ) -> ImagingCNN:
        """Train the imaging model."""

        device = device or torch.device("cpu")
        x_train = torch.tensor(features, dtype=torch.float32)
        y_train = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        self.to(device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
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


ImagingCnn = ImagingCNN
