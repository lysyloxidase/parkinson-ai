"""Neural multimodal fusion blocks for PD prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn


@dataclass(slots=True)
class FusionOutput:
    """Container for multimodal fusion outputs."""

    embedding: torch.Tensor
    attention_scores: torch.Tensor


class ModalityEncoder(nn.Module):
    """Small per-modality encoder."""

    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        hidden_dim = max(embed_dim, min(256, input_dim * 2))
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode a modality feature tensor."""

        return cast(torch.Tensor, self.network(inputs))


class FusionTransformerLayer(nn.Module):
    """A single residual cross-attention block."""

    def __init__(self, embed_dim: int, n_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform modality embeddings."""

        attended, _ = self.attention(inputs, inputs, inputs, need_weights=False)
        hidden = self.attention_norm(inputs + attended)
        return cast(torch.Tensor, self.ffn_norm(hidden + self.ffn(hidden)))


class MultimodalFusion(nn.Module):
    """Attention-based multimodal feature fusion for PD prediction."""

    def __init__(
        self,
        feature_dims: dict[str, int],
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.feature_dims = dict(feature_dims)
        self.embed_dim = embed_dim
        self.ordered_modalities = tuple(feature_dims.keys())
        self.encoders = nn.ModuleDict({modality: ModalityEncoder(input_dim=feature_dim, embed_dim=embed_dim) for modality, feature_dim in feature_dims.items()})
        self.mask_tokens = nn.Parameter(torch.randn(len(self.ordered_modalities), embed_dim) * 0.02)
        self.transformer_layers = nn.ModuleList([FusionTransformerLayer(embed_dim=embed_dim, n_heads=n_heads) for _ in range(n_layers)])
        self.attention_head = nn.Linear(embed_dim, 1)
        self.last_attention_scores: torch.Tensor | None = None

    def forward(self, features: dict[str, torch.Tensor], modality_mask: torch.Tensor) -> torch.Tensor:
        """Fuse per-modality features into a flattened multimodal representation."""

        output = self.forward_with_attention(features=features, modality_mask=modality_mask)
        return output.embedding

    def forward_with_attention(
        self,
        features: dict[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ) -> FusionOutput:
        """Fuse features and return the attention distribution."""

        if modality_mask.ndim != 2:
            raise ValueError("modality_mask must have shape (batch, n_modalities)")
        batch_size = modality_mask.shape[0]
        device = modality_mask.device
        encoded_modalities: list[torch.Tensor] = []
        for index, modality in enumerate(self.ordered_modalities):
            if modality not in features:
                raw = torch.zeros(batch_size, self.feature_dims[modality], device=device)
            else:
                raw = features[modality]
                if raw.ndim != 2:
                    raise ValueError(f"{modality} features must have shape (batch, feature_dim)")
            encoded = self.encoders[modality](torch.nan_to_num(raw, nan=0.0))
            mask_token = self.mask_tokens[index].expand(batch_size, -1)
            mask_column = modality_mask[:, index].unsqueeze(-1)
            encoded_modalities.append(torch.where(mask_column > 0.5, encoded, mask_token))
        hidden = torch.stack(encoded_modalities, dim=1)
        for layer in self.transformer_layers:
            hidden = layer(hidden)
        attention_logits = self.attention_head(hidden).squeeze(-1)
        attention_scores = torch.softmax(attention_logits, dim=1)
        weighted_hidden = hidden * attention_scores.unsqueeze(-1)
        fused_embedding = weighted_hidden.reshape(batch_size, -1)
        self.last_attention_scores = attention_scores
        return FusionOutput(embedding=fused_embedding, attention_scores=attention_scores)
