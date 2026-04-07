"""Graph Model model stub."""

from __future__ import annotations

import numpy as np

from parkinson_ai.ml.models.base import BasePDModel


class GraphModel(BasePDModel):
    """Placeholder model implementation."""

    def fit(self, features: np.ndarray, labels: np.ndarray) -> GraphModel:
        """Fit the placeholder model."""

        raise NotImplementedError("GraphModel is not implemented yet.")

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict probabilities with the placeholder model."""

        raise NotImplementedError("GraphModel is not implemented yet.")
