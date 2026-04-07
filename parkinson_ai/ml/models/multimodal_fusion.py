"""Multimodal PD fusion models."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from parkinson_ai.ml.features.feature_registry import PDFeatureRegistry
from parkinson_ai.ml.features.fusion import MultimodalFusion
from parkinson_ai.ml.features.imputation import modality_dropout


class MultiParkNetLite(nn.Module):
    """Simplified multimodal fusion model inspired by MultiParkNet."""

    def __init__(
        self,
        feature_registry: PDFeatureRegistry,
        embed_dim: int = 64,
        n_heads: int = 4,
        modality_dropout_p: float = 0.3,
    ) -> None:
        super().__init__()
        self.feature_registry = feature_registry
        self.modalities = feature_registry.MODALITIES
        self.feature_dims = feature_registry.feature_dims
        self.fusion = MultimodalFusion(
            feature_dims=self.feature_dims,
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=2,
        )
        fusion_dim = embed_dim * len(self.modalities)
        self.shared_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.diagnosis_head = nn.Linear(128, 3)
        self.stage_head = nn.Linear(128, 7)
        self.subtype_head = nn.Linear(128, 2)
        self.progression_head = nn.Linear(128, 1)
        self.modality_dropout_p = modality_dropout_p

    def forward(
        self,
        patient_data: dict[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return diagnosis, stage, subtype, progression, and attention outputs."""

        effective_mask = modality_mask
        if self.training and self.modality_dropout_p > 0.0:
            effective_mask = modality_dropout(modality_mask, p=self.modality_dropout_p)
        fusion_output = self.fusion.forward_with_attention(patient_data, effective_mask)
        shared = self.shared_head(fusion_output.embedding)
        return {
            "diagnosis_prob": torch.softmax(self.diagnosis_head(shared), dim=-1),
            "stage_prob": torch.softmax(self.stage_head(shared), dim=-1),
            "subtype_prob": torch.softmax(self.subtype_head(shared), dim=-1),
            "progression_rate": self.progression_head(shared),
            "modality_attention": fusion_output.attention_scores,
        }


class MultimodalFusionModel:
    """Backward-compatible lightweight fusion helper."""

    def __init__(self, modality_weights: dict[str, float] | None = None) -> None:
        self.modality_weights = modality_weights or {}

    def fuse(self, features: dict[str, np.ndarray]) -> np.ndarray:
        """Fuse modality matrices using fixed weights."""

        if not features:
            raise ValueError("features must not be empty")
        weighted: list[np.ndarray] = []
        total_weight = 0.0
        for modality, values in features.items():
            weight = self.modality_weights.get(modality, 1.0)
            weighted.append(values * weight)
            total_weight += weight
        fused = np.concatenate(weighted, axis=1)
        return fused / max(total_weight, 1e-8)

    def attention_summary(self, features: dict[str, np.ndarray]) -> dict[str, float]:
        """Return normalized fixed modality weights."""

        weights = {modality: self.modality_weights.get(modality, 1.0) for modality in features}
        total = sum(weights.values()) or 1.0
        return {modality: weight / total for modality, weight in weights.items()}

    def fit(self, _: dict[str, np.ndarray], __: np.ndarray | None = None) -> MultimodalFusionModel:
        """Keep a symmetric interface with other models."""

        return self
