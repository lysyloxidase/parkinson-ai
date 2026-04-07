"""Unified training loop utilities for PD prediction models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from parkinson_ai.ml.evaluation import evaluate_binary_classifier
from parkinson_ai.ml.features.imputation import modality_dropout

try:
    from sklearn.model_selection import train_test_split
except ImportError:  # pragma: no cover - optional dependency
    train_test_split = None


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for neural model training."""

    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    gradient_clip_norm: float = 1.0
    modality_dropout_p: float = 0.3
    checkpoint_path: str | None = None
    device: str = "cpu"
    output_key: str | None = None


@dataclass(slots=True)
class TrainingHistory:
    """Training history for a torch model."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_auroc: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_auroc: float = 0.0


def stratified_train_val_test_split(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Create stratified train, validation, and test splits."""

    if train_test_split is None:
        raise RuntimeError("scikit-learn is required for stratified splitting.")
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )
    relative_val_size = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val_size,
        stratify=y_train_val,
        random_state=random_state,
    )
    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }


def train_torch_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader[Any],
    val_loader: torch.utils.data.DataLoader[Any],
    *,
    config: TrainingConfig | None = None,
) -> TrainingHistory:
    """Train a torch model with early stopping and cosine scheduling."""

    config = config or TrainingConfig()
    device = torch.device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config.epochs, 1),
    )
    history = TrainingHistory()
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    for epoch in range(config.epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            inputs, mask, targets = _unpack_batch(batch, device)
            if mask is not None:
                mask = modality_dropout(mask, p=config.modality_dropout_p)
            optimizer.zero_grad()
            outputs = _forward_model(model, inputs, mask)
            predictions = _extract_training_tensor(outputs, config.output_key)
            loss = _compute_loss(predictions, targets)
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        scheduler.step()
        val_loss, val_scores, val_truth = _evaluate_model(model, val_loader, device, config.output_key)
        metrics = evaluate_binary_classifier(np.asarray(val_truth), np.asarray(val_scores), n_bootstrap=64)
        history.train_loss.append(float(np.mean(epoch_losses)))
        history.val_loss.append(val_loss)
        history.val_auroc.append(float(metrics["auroc"]))
        if float(metrics["auroc"]) > history.best_val_auroc:
            history.best_val_auroc = float(metrics["auroc"])
            history.best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
            if config.checkpoint_path is not None:
                checkpoint_path = Path(config.checkpoint_path)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(torch.device("cpu"))
    return history


def _unpack_batch(
    batch: Any,
    device: torch.device,
) -> tuple[Any, torch.Tensor | None, torch.Tensor]:
    """Unpack supported batch formats."""

    if isinstance(batch, dict):
        inputs = batch["features"]
        mask = batch.get("modality_mask")
        targets = batch["targets"]
    elif isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            inputs, targets = batch
            mask = None
        elif len(batch) == 3:
            inputs, mask, targets = batch
        else:
            raise ValueError("Unsupported batch structure.")
    else:
        raise ValueError("Unsupported batch type.")
    if isinstance(inputs, dict):
        tensor_inputs = {key: value.to(device) for key, value in inputs.items()}
    else:
        tensor_inputs = inputs.to(device)
    tensor_mask = None if mask is None else mask.to(device)
    tensor_targets = targets.to(device)
    return tensor_inputs, tensor_mask, tensor_targets


def _forward_model(
    model: torch.nn.Module,
    inputs: Any,
    mask: torch.Tensor | None,
) -> Any:
    """Forward a batch through the model."""

    if isinstance(inputs, dict):
        if mask is None:
            raise ValueError("modality_mask is required for dict inputs")
        return model(inputs, mask)
    return model(inputs)


def _extract_training_tensor(outputs: Any, output_key: str | None) -> torch.Tensor:
    """Extract the tensor used for optimization."""

    if isinstance(outputs, dict):
        if output_key is not None:
            tensor = outputs[output_key]
        elif "diagnosis_prob" in outputs:
            tensor = outputs["diagnosis_prob"]
        else:
            first_key = next(iter(outputs))
            tensor = outputs[first_key]
    else:
        tensor = outputs
    return cast(torch.Tensor, tensor)


def _compute_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute a binary or multiclass loss depending on prediction shape."""

    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(-1)
    if predictions.shape[-1] == 1:
        return F.binary_cross_entropy(predictions.float(), targets.float().view_as(predictions))
    safe_predictions = torch.log(predictions.clamp_min(1e-8))
    return F.nll_loss(safe_predictions, targets.long())


def _evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader[Any],
    device: torch.device,
    output_key: str | None,
) -> tuple[float, list[float], list[int]]:
    """Evaluate a model on a validation loader."""

    model.eval()
    losses: list[float] = []
    scores: list[float] = []
    truth: list[int] = []
    with torch.no_grad():
        for batch in loader:
            inputs, mask, targets = _unpack_batch(batch, device)
            outputs = _forward_model(model, inputs, mask)
            predictions = _extract_training_tensor(outputs, output_key)
            loss = _compute_loss(predictions, targets)
            losses.append(float(loss.cpu().item()))
            if predictions.shape[-1] == 1:
                probabilities = predictions.squeeze(-1)
                scores.extend(probabilities.cpu().numpy().tolist())
                truth.extend(targets.long().cpu().numpy().tolist())
            else:
                probabilities = predictions[:, -1]
                scores.extend(probabilities.cpu().numpy().tolist())
                truth.extend(targets.long().cpu().numpy().tolist())
    return float(np.mean(losses)), scores, truth
