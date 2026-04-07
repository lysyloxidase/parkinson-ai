"""Utility helpers shared across the project."""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


@dataclass(slots=True)
class TimingResult:
    """Simple timing result container."""

    label: str
    seconds: float


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure a consistent logging format for CLI and API usage."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""

    return logging.getLogger(name)


def detect_device(preferred: str | None = None) -> str:
    """Return the preferred compute device if available."""

    if preferred and preferred != "auto":
        if preferred == "cuda" and torch is not None and torch.cuda.is_available():
            return "cuda"
        if preferred == "mps" and torch is not None and hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available():
                return "mps"
        if preferred == "cpu":
            return "cpu"
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch when available."""

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:  # pragma: no branch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


@contextmanager
def timed(label: str) -> Iterator[TimingResult]:
    """Measure the duration of a code block."""

    start = time.perf_counter()
    result = TimingResult(label=label, seconds=0.0)
    try:
        yield result
    finally:
        result.seconds = time.perf_counter() - start


def ensure_list(value: Any) -> list[Any]:
    """Normalize scalars and iterables into a list."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]
