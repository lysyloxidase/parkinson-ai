"""CLI for training parkinson-ai models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    from parkinson_ai.core.utils import seed_everything
    from parkinson_ai.ml.evaluation import evaluate_binary_classifier
    from parkinson_ai.ml.features.feature_registry import PDFeatureRegistry
    from parkinson_ai.ml.models.gait_lstm import GaitLSTM
    from parkinson_ai.ml.models.genomic_model import GenomicRiskModel
    from parkinson_ai.ml.models.imaging_cnn import ImagingCNN
    from parkinson_ai.ml.models.multimodal_fusion import MultiParkNetLite
    from parkinson_ai.ml.models.prodromal_model import ProdromalConversionModel
    from parkinson_ai.ml.models.voice_cnn import VoiceCNN
    from parkinson_ai.ml.models.xgboost_model import XGBoostPDModel
    from parkinson_ai.ml.training import TrainingConfig, train_torch_model
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from parkinson_ai.core.utils import seed_everything
    from parkinson_ai.ml.evaluation import evaluate_binary_classifier
    from parkinson_ai.ml.features.feature_registry import PDFeatureRegistry
    from parkinson_ai.ml.models.gait_lstm import GaitLSTM
    from parkinson_ai.ml.models.genomic_model import GenomicRiskModel
    from parkinson_ai.ml.models.imaging_cnn import ImagingCNN
    from parkinson_ai.ml.models.multimodal_fusion import MultiParkNetLite
    from parkinson_ai.ml.models.prodromal_model import ProdromalConversionModel
    from parkinson_ai.ml.models.voice_cnn import VoiceCNN
    from parkinson_ai.ml.models.xgboost_model import XGBoostPDModel
    from parkinson_ai.ml.training import TrainingConfig, train_torch_model


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Train parkinson-ai models.")
    parser.add_argument("--model", choices=["xgboost", "voice", "gait", "imaging", "genomic", "fusion", "prodromal", "all"], default="xgboost")
    parser.add_argument("--data-dir", default="data", help="Directory containing training arrays (.npz) when available.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    """Run the model training CLI."""

    args = parse_args()
    seed_everything(42)
    registry = PDFeatureRegistry()
    models_to_run = ["xgboost", "voice", "gait", "imaging", "genomic", "fusion", "prodromal"] if args.model == "all" else [args.model]
    results: dict[str, Any] = {}
    for model_name in models_to_run:
        results[model_name] = run_training(
            model_name=model_name,
            data_dir=Path(args.data_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            registry=registry,
        )
    print(json.dumps(_json_safe(results), indent=2))


def run_training(
    *,
    model_name: str,
    data_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    registry: PDFeatureRegistry,
) -> dict[str, Any]:
    """Train a selected model on local or synthetic data."""

    if model_name == "xgboost":
        features, labels = _load_or_generate_tabular(data_dir / "xgboost_data.npz", registry.count())
        xgboost_model = XGBoostPDModel(feature_names=registry.feature_names)
        xgboost_model.fit(features, labels)
        probabilities = xgboost_model.predict_proba(features)[:, 1]
        return evaluate_binary_classifier(labels, probabilities, n_bootstrap=64)
    if model_name == "voice":
        features, labels = _generate_voice_data()
        voice_model = VoiceCNN(input_channels=features.shape[1])
        voice_model.fit(features, labels, epochs=epochs, lr=learning_rate, batch_size=batch_size)
        probabilities = voice_model.predict_proba(features)[:, 1]
        return evaluate_binary_classifier(labels, probabilities, n_bootstrap=64)
    if model_name == "gait":
        features, labels = _generate_gait_data()
        gait_model = GaitLSTM(n_sensors=features.shape[2])
        gait_model.fit(features, labels, epochs=epochs, lr=learning_rate, batch_size=batch_size)
        probabilities = gait_model.predict_proba(features)[:, 1]
        return evaluate_binary_classifier(labels, probabilities, n_bootstrap=64)
    if model_name == "imaging":
        features, labels = _generate_imaging_data()
        imaging_model = ImagingCNN(use_pretrained=False)
        imaging_model.fit(features, labels, epochs=max(1, min(epochs, 3)), lr=learning_rate, batch_size=batch_size)
        probabilities = imaging_model.predict_proba(features)[:, 1]
        return evaluate_binary_classifier(labels, probabilities, n_bootstrap=64)
    if model_name == "genomic":
        features, labels = _generate_genomic_data()
        genomic_model = GenomicRiskModel(variant_dim=6, pathway_dim=5)
        genomic_model.fit(features, labels, epochs=epochs, lr=learning_rate, batch_size=batch_size)
        probabilities = genomic_model.predict_proba(features)[:, 1]
        return evaluate_binary_classifier(labels, probabilities, n_bootstrap=64)
    if model_name == "fusion":
        patient_data, modality_mask, labels = _generate_fusion_data(registry, n_samples=64)
        fusion_model = MultiParkNetLite(registry)
        dataset = TensorDataset(
            patient_data["molecular"],
            patient_data["genetic"],
            patient_data["imaging"],
            patient_data["digital"],
            patient_data["clinical"],
            patient_data["nonmotor"],
            patient_data["prodromal"],
            modality_mask,
            torch.tensor(labels, dtype=torch.long),
        )
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=_fusion_collate)
        history = train_torch_model(
            fusion_model,
            train_loader,
            train_loader,
            config=TrainingConfig(epochs=epochs, learning_rate=learning_rate, output_key="diagnosis_prob"),
        )
        return {"best_val_auroc": history.best_val_auroc, "best_epoch": history.best_epoch}
    if model_name == "prodromal":
        features, durations, events = _generate_prodromal_data()
        prodromal_model = ProdromalConversionModel(input_dim=features.shape[1])
        prodromal_model.fit_survival(features, durations=durations, events=events, epochs=max(epochs * 20, 50), method="deepsurv")
        probabilities = prodromal_model.predict_proba(features)[:, 1]
        return evaluate_binary_classifier(events, probabilities, n_bootstrap=64)
    raise ValueError(f"Unsupported model: {model_name}")


def _load_or_generate_tabular(path: Path, feature_dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Load tabular data from disk or generate a synthetic fallback."""

    if path.exists():
        payload = np.load(path)
        return np.asarray(payload["features"], dtype=float), np.asarray(payload["labels"], dtype=int)
    rng = np.random.default_rng(42)
    features = rng.normal(size=(128, feature_dim))
    signal = features[:, 0] + 0.8 * features[:, 1] - 0.5 * features[:, 2]
    labels = (signal > np.median(signal)).astype(int)
    return features, labels


def _generate_voice_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple separable voice dataset."""

    rng = np.random.default_rng(42)
    features = rng.normal(size=(64, 13, 64)).astype(np.float32)
    labels = np.repeat([0, 1], repeats=32)
    features[labels == 1, 0, :] += 1.5
    return features, labels.astype(int)


def _generate_gait_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple separable gait dataset."""

    rng = np.random.default_rng(7)
    features = rng.normal(size=(64, 50, 16)).astype(np.float32)
    labels = np.repeat([0, 1], repeats=32)
    features[labels == 1, :, 0] += 1.0
    return features, labels.astype(int)


def _generate_imaging_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic imaging slices with a discriminative central region."""

    rng = np.random.default_rng(21)
    features = rng.normal(size=(32, 3, 64, 64)).astype(np.float32)
    labels = np.repeat([0, 1], repeats=16)
    features[labels == 1, :, 24:40, 24:40] += 2.0
    return features, labels.astype(int)


def _generate_genomic_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic genomic features."""

    rng = np.random.default_rng(12)
    prs = rng.normal(size=(80, 1))
    age = rng.normal(loc=65.0, scale=8.0, size=(80, 1))
    variants = rng.integers(0, 2, size=(80, 6))
    pathways = rng.normal(size=(80, 5))
    features = np.concatenate([prs, age, variants, pathways], axis=1).astype(np.float32)
    signal = prs.squeeze() + 0.7 * variants[:, 0] + 0.3 * pathways[:, 0]
    labels = (signal > np.median(signal)).astype(int)
    return features, labels


def _generate_fusion_data(
    registry: PDFeatureRegistry,
    *,
    n_samples: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, np.ndarray]:
    """Generate synthetic multimodal tabular inputs."""

    rng = np.random.default_rng(3)
    patient_data: dict[str, torch.Tensor] = {}
    labels = np.repeat([0, 1], repeats=n_samples // 2)
    modality_mask = torch.ones((n_samples, len(registry.MODALITIES)), dtype=torch.float32)
    for modality in registry.MODALITIES:
        feature_dim = registry.feature_dims[modality]
        array = rng.normal(size=(n_samples, feature_dim)).astype(np.float32)
        array[labels == 1, 0] += 1.0
        patient_data[modality] = torch.tensor(array, dtype=torch.float32)
    return patient_data, modality_mask, labels.astype(int)


def _generate_prodromal_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic prodromal survival data."""

    rng = np.random.default_rng(99)
    features = rng.normal(size=(96, 6)).astype(np.float32)
    risk = features[:, 0] + 0.8 * features[:, 1]
    durations = np.clip(8.0 - risk + rng.normal(scale=0.5, size=96), 0.5, 12.0)
    events = (risk > np.median(risk)).astype(int)
    return features, durations.astype(np.float32), events.astype(int)


def _fusion_collate(batch: list[tuple[torch.Tensor, ...]]) -> dict[str, Any]:
    """Collate multimodal tensors into the dict format expected by the trainer."""

    molecular, genetic, imaging, digital, clinical, nonmotor, prodromal, mask, targets = zip(*batch, strict=False)
    return {
        "features": {
            "molecular": torch.stack(list(molecular)),
            "genetic": torch.stack(list(genetic)),
            "imaging": torch.stack(list(imaging)),
            "digital": torch.stack(list(digital)),
            "clinical": torch.stack(list(clinical)),
            "nonmotor": torch.stack(list(nonmotor)),
            "prodromal": torch.stack(list(prodromal)),
        },
        "modality_mask": torch.stack(list(mask)),
        "targets": torch.stack(list(targets)),
    }


def _json_safe(value: Any) -> Any:
    """Convert numpy and tensor outputs into JSON-serializable Python values."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


if __name__ == "__main__":
    main()
