"""Abstract base model for PD prediction models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BasePDModel(ABC):
    """Common interface for prediction models."""

    @abstractmethod
    def fit(self, features: np.ndarray, labels: np.ndarray) -> BasePDModel:
        """Fit the model."""

    @abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
