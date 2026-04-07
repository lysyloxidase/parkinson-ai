"""Embedding management with graceful fallbacks for local development."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Sequence
from types import ModuleType
from typing import Any

import numpy as np

from parkinson_ai.config import get_settings
from parkinson_ai.core.utils import detect_device


def _load_sentence_transformers_module() -> ModuleType | None:
    """Load sentence-transformers when available."""

    try:
        return importlib.import_module("sentence_transformers")
    except ImportError:  # pragma: no cover - optional dependency
        return None


sentence_transformers_module = _load_sentence_transformers_module()


class EmbeddingManager:
    """Encode text using sentence-transformers or a deterministic fallback."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name or get_settings().MODEL_EMBEDDING
        self.device = detect_device(device or get_settings().DEVICE)
        self.normalize = normalize
        self._model: Any | None = self._load_model(self.model_name)

    def embed_query(self, text: str) -> list[float]:
        """Encode a single query string."""

        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode multiple texts."""

        items = [text.strip() for text in texts]
        if not items:
            return []
        if self._model is not None:
            vectors = self._model.encode(
                items,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [[float(value) for value in row] for row in np.asarray(vectors)]
        return [self._hash_embedding(item) for item in items]

    def _load_model(self, model_name: str) -> Any | None:
        """Load a dense encoder when available."""

        if sentence_transformers_module is None:
            return None
        aliases = {
            "medcpt": "ncbi/MedCPT-Query-Encoder",
            "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
            "nomic-embed-text": "sentence-transformers/all-MiniLM-L6-v2",
        }
        resolved = aliases.get(model_name.lower(), model_name)
        try:
            return sentence_transformers_module.SentenceTransformer(resolved, device=self.device)
        except Exception:  # pragma: no cover - depends on local model state
            return None

    def _hash_embedding(self, text: str, dimension: int = 64) -> list[float]:
        """Create a deterministic lightweight embedding without model dependencies."""

        values = np.zeros(dimension, dtype=np.float32)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        for index, byte in enumerate(digest):
            values[index % dimension] += (byte / 255.0) - 0.5
        norm = np.linalg.norm(values)
        if self.normalize and norm > 0:
            values /= norm
        return [float(item) for item in values]
