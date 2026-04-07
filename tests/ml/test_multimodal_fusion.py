"""Fusion model tests."""

from __future__ import annotations

import numpy as np
import torch

from parkinson_ai.ml.features.feature_registry import PDFeatureRegistry
from parkinson_ai.ml.features.fusion import MultimodalFusion
from parkinson_ai.ml.models.multimodal_fusion import MultimodalFusionModel, MultiParkNetLite


def test_attention_fusion_block_shapes() -> None:
    """The low-level fusion block should return a fused embedding per modality."""

    fusion = MultimodalFusion({"molecular": 3, "clinical": 2}, embed_dim=8, n_heads=2, n_layers=1)
    features = {
        "molecular": torch.ones((4, 3)),
        "clinical": torch.ones((4, 2)),
    }
    mask = torch.ones((4, 2))
    embedding = fusion(features, mask)
    assert embedding.shape == (4, 16)
    assert fusion.last_attention_scores is not None


def test_multiparknet_lite_output_shapes() -> None:
    """The multimodal fusion model should emit all task heads."""

    registry = PDFeatureRegistry()
    model = MultiParkNetLite(registry, embed_dim=8, n_heads=2, modality_dropout_p=0.0)
    patient_data = {modality: torch.randn(6, dimension) for modality, dimension in registry.feature_dims.items()}
    mask = torch.ones((6, len(registry.MODALITIES)))
    outputs = model(patient_data, mask)
    assert outputs["diagnosis_prob"].shape == (6, 3)
    assert outputs["stage_prob"].shape == (6, 7)
    assert outputs["subtype_prob"].shape == (6, 2)
    assert outputs["progression_rate"].shape == (6, 1)
    assert outputs["modality_attention"].shape == (6, len(registry.MODALITIES))


def test_legacy_fusion_wrapper() -> None:
    """The legacy NumPy fusion wrapper should remain usable."""

    model = MultimodalFusionModel({"molecular": 2.0, "clinical": 1.0})
    fused = model.fuse({"molecular": np.ones((2, 3)), "clinical": np.ones((2, 2))})
    attention = model.attention_summary({"molecular": fused[:, :3], "clinical": fused[:, 3:]})
    assert fused.shape == (2, 5)
    assert abs(sum(attention.values()) - 1.0) < 1e-8
