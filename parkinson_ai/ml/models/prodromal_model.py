"""Prodromal survival models for PD conversion."""

from __future__ import annotations

import importlib
import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from parkinson_ai.ml.models.base import BasePDModel


def _load_lifelines() -> Any | None:
    """Load lifelines when available."""

    try:
        return importlib.import_module("lifelines")
    except ImportError:  # pragma: no cover - optional dependency
        return None


lifelines_module = _load_lifelines()


class DeepSurvNetwork(nn.Module):
    """Neural Cox model for prodromal conversion risk."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a Cox risk score."""

        return torch.as_tensor(self.network(inputs))


class ProdromalConversionModel(BasePDModel):
    """Survival model for prodromal to clinical PD conversion."""

    def __init__(self, input_dim: int, *, horizon_years: float = 8.0) -> None:
        self.input_dim = input_dim
        self.horizon_years = horizon_years
        self.cox_model = lifelines_module.CoxPHFitter(penalizer=0.1) if lifelines_module is not None else None
        self.deep_surv = DeepSurvNetwork(input_dim)
        self._baseline_hazard: np.ndarray | None = None
        self._event_times: np.ndarray | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> ProdromalConversionModel:
        """Fit a fallback survival model using rank-ordered pseudo-durations."""

        durations = np.linspace(1.0, float(len(labels)), num=len(labels))
        return self.fit_survival(features, durations=durations, events=labels)

    def fit_survival(
        self,
        features: np.ndarray,
        *,
        durations: np.ndarray,
        events: np.ndarray,
        method: str = "coxph",
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> ProdromalConversionModel:
        """Fit either CoxPH or DeepSurv."""

        x_data = np.asarray(features, dtype=float)
        durations_array = np.asarray(durations, dtype=float)
        events_array = np.asarray(events, dtype=int)
        if method == "coxph" and self.cox_model is not None:
            frame = pd.DataFrame(x_data, columns=[f"feature_{index}" for index in range(x_data.shape[1])])
            frame["duration"] = durations_array
            frame["event"] = events_array
            try:
                self.cox_model.fit(frame, duration_col="duration", event_col="event")
                return self
            except Exception as exc:
                warnings.warn(
                    f"CoxPH fit failed ({exc!s}); falling back to DeepSurv.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        self._fit_deep_surv(x_data, durations_array, events_array, epochs=epochs, lr=lr)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict conversion probability by the configured horizon."""

        survival_curve = self.predict_survival_curve(features, times=np.array([self.horizon_years]))
        conversion_probability = 1.0 - survival_curve[:, 0]
        return np.stack([1.0 - conversion_probability, conversion_probability], axis=1)

    def predict_survival_curve(
        self,
        features: np.ndarray,
        *,
        times: np.ndarray,
        method: str | None = None,
    ) -> np.ndarray:
        """Predict survival probabilities over time."""

        x_data = np.asarray(features, dtype=float)
        horizon_times = np.asarray(times, dtype=float)
        if method in {None, "coxph"} and self.cox_model is not None and hasattr(self.cox_model, "params_"):
            frame = pd.DataFrame(x_data, columns=[f"feature_{index}" for index in range(x_data.shape[1])])
            survival = self.cox_model.predict_survival_function(frame, times=horizon_times)
            return np.asarray(survival.to_numpy().T, dtype=float)
        if self._baseline_hazard is None or self._event_times is None:
            raise RuntimeError("DeepSurv model must be fit before prediction.")
        self.deep_surv.eval()
        inputs = torch.tensor(x_data, dtype=torch.float32)
        with torch.no_grad():
            risks = self.deep_surv(inputs).squeeze(-1).cpu().numpy()
        cumulative_hazard = np.interp(horizon_times, self._event_times, self._baseline_hazard, left=0.0, right=self._baseline_hazard[-1])
        survival = np.exp(-np.exp(risks)[:, None] * cumulative_hazard[None, :])
        return np.asarray(survival, dtype=float)

    def _fit_deep_surv(
        self,
        features: np.ndarray,
        durations: np.ndarray,
        events: np.ndarray,
        *,
        epochs: int,
        lr: float,
    ) -> None:
        """Fit the DeepSurv network."""

        inputs = torch.tensor(features, dtype=torch.float32)
        duration_tensor = torch.tensor(durations, dtype=torch.float32)
        event_tensor = torch.tensor(events, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.deep_surv.parameters(), lr=lr)
        for _ in range(epochs):
            self.deep_surv.train()
            optimizer.zero_grad()
            risks = self.deep_surv(inputs).squeeze(-1)
            loss = _cox_partial_log_likelihood(risks, duration_tensor, event_tensor)
            torch.autograd.backward(loss)
            optimizer.step()
        self._baseline_hazard, self._event_times = _breslow_baseline_hazard(
            self.deep_surv(inputs).detach().squeeze(-1).cpu().numpy(),
            durations,
            events,
        )


def _cox_partial_log_likelihood(
    risks: torch.Tensor,
    durations: torch.Tensor,
    events: torch.Tensor,
) -> torch.Tensor:
    """Compute the negative Cox partial log-likelihood."""

    order = torch.argsort(durations, descending=True)
    sorted_risks = risks[order]
    sorted_events = events[order]
    log_cumsum = torch.logcumsumexp(sorted_risks, dim=0)
    partial = sorted_risks - log_cumsum
    return -(partial * sorted_events).sum() / sorted_events.sum().clamp_min(1.0)


def _breslow_baseline_hazard(
    risks: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the Breslow baseline cumulative hazard."""

    unique_times = np.sort(np.unique(durations[events == 1]))
    cumulative_hazard: list[float] = []
    running = 0.0
    exp_risk = np.exp(risks)
    for time in unique_times:
        event_mask = (durations == time) & (events == 1)
        risk_set_mask = durations >= time
        hazard = np.sum(event_mask) / max(float(np.sum(exp_risk[risk_set_mask])), 1e-8)
        running += hazard
        cumulative_hazard.append(running)
    if not cumulative_hazard:
        return np.array([0.0]), np.array([0.0])
    return np.asarray(cumulative_hazard, dtype=float), unique_times.astype(float)


ProdromalModel = ProdromalConversionModel
