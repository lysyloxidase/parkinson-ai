"""Agent 7: multimodal prodromal risk assessment for Parkinson's disease."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from parkinson_ai.agents.base import AgentResult, BaseAgent
from parkinson_ai.agents.common import SupportsGenerate, system_prompt, try_generate_text
from parkinson_ai.biomarkers.prodromal.risk_calculator import process_risk_calculator
from parkinson_ai.config import get_settings
from parkinson_ai.core.llm_client import OllamaClient
from parkinson_ai.core.utils import seed_everything
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.ml.features.feature_registry import PDFeatureRegistry
from parkinson_ai.ml.models.multimodal_fusion import MultiParkNetLite
from parkinson_ai.ml.models.prodromal_model import ProdromalConversionModel
from parkinson_ai.rag.kg_context import KGContextExtractor


class RiskAssessorAgent(BaseAgent):
    """Blend multimodal ML, prodromal criteria, and KG context into a risk report."""

    def __init__(
        self,
        *,
        graph: PDKnowledgeGraph | None = None,
        llm_client: SupportsGenerate | None = None,
        feature_registry: PDFeatureRegistry | None = None,
    ) -> None:
        super().__init__("risk_assessor")
        settings = get_settings()
        seed_everything(settings.RANDOM_SEED)
        self.graph = graph or PDKnowledgeGraph()
        self.llm_client = llm_client or OllamaClient()
        self.feature_registry = feature_registry or PDFeatureRegistry()
        self.model = MultiParkNetLite(self.feature_registry)
        self.model.eval()
        self.kg_extractor = KGContextExtractor(self.graph)
        self.prodromal_model = ProdromalConversionModel(input_dim=6, horizon_years=8.0)
        self._fit_prodromal_reference_model()

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Assess prodromal or progression risk from partial multimodal data."""

        patient_data = self._normalize_patient_data(kwargs.get("patient_data"))
        modality_tensors = self.feature_registry.build_modality_tensors(patient_data)
        modality_mask = self.feature_registry.get_modality_mask(patient_data).unsqueeze(0)
        batched_features = {modality: torch.nan_to_num(values.unsqueeze(0), nan=0.0) for modality, values in modality_tensors.items()}
        with torch.no_grad():
            ml_outputs = self.model(batched_features, modality_mask)
        criteria = process_risk_calculator(patient_data=patient_data)
        prodromal_curve = self._prodromal_curve(patient_data)
        available_modalities: list[str] = [modality for modality, flag in zip(self.feature_registry.MODALITIES, modality_mask.squeeze(0).tolist(), strict=False) if flag > 0.5]
        kg_context = self._kg_reasoning(patient_data)
        combined_risk = self._combine_scores(ml_outputs=ml_outputs, criteria=criteria, kg_context=kg_context)
        confidence_interval = self._confidence_interval(combined_risk, len(available_modalities))
        recommendations = self._recommended_tests(available_modalities)
        monitoring_timeline = self._monitoring_timeline(combined_risk)
        prompt = self._build_prompt(
            task=task,
            patient_data=patient_data,
            combined_risk=combined_risk,
            confidence_interval=confidence_interval,
            criteria=criteria,
            recommendations=recommendations,
            monitoring_timeline=monitoring_timeline,
        )
        content = self._fallback_report(combined_risk, confidence_interval, criteria, recommendations, monitoring_timeline)
        generated = self._try_generate(prompt)
        if generated:
            content = generated
        return AgentResult(
            agent_name=self.name,
            content=content,
            metadata={
                "available_modalities": available_modalities,
                "criteria_score": criteria,
                "ml_outputs": {key: value.squeeze(0).detach().cpu().numpy().round(4).tolist() for key, value in ml_outputs.items()},
                "prodromal_curve": prodromal_curve,
                "combined_risk": combined_risk,
                "confidence_interval": confidence_interval,
                "kg_context": kg_context,
                "recommendations": recommendations,
                "monitoring_timeline": monitoring_timeline,
            },
        )

    def _normalize_patient_data(self, payload: object) -> dict[str, Any]:
        """Normalize patient data to a flat dictionary."""

        if isinstance(payload, dict):
            return {str(key): value for key, value in payload.items()}
        return {}

    def _fit_prodromal_reference_model(self) -> None:
        """Fit a tiny deterministic prodromal reference cohort."""

        features = np.array(
            [
                [1, 1, 1, 2.0, 22.0, 68],
                [1, 1, 0, 1.5, 16.0, 64],
                [1, 0, 0, 0.5, 12.0, 60],
                [0, 1, 0, 0.2, 10.0, 59],
                [0, 0, 0, -0.5, 8.0, 55],
                [1, 1, 1, 1.0, 28.0, 72],
            ],
            dtype=float,
        )
        durations = np.array([4.0, 6.0, 8.5, 10.0, 12.0, 3.5], dtype=float)
        events = np.array([1, 1, 0, 0, 0, 1], dtype=int)
        # This tiny synthetic cohort is only used to bootstrap demo predictions.
        # DeepSurv is more stable than CoxPH here because the sample size is much
        # smaller than the feature count and the values are intentionally correlated.
        self.prodromal_model.fit_survival(
            features,
            durations=durations,
            events=events,
            method="deepsurv",
            epochs=60,
            lr=5e-3,
        )

    def _prodromal_curve(self, patient_data: dict[str, Any]) -> dict[str, float]:
        """Predict a time-to-conversion curve when prodromal features are present."""

        feature_vector = np.asarray(
            [
                [
                    1.0 if bool(patient_data.get("rbd_present")) else 0.0,
                    1.0 if bool(patient_data.get("hyposmia")) else 0.0,
                    1.0 if bool(patient_data.get("saa_result")) else 0.0,
                    float(patient_data.get("prs_score", 0.0) or 0.0),
                    float(patient_data.get("nfl_pg_ml", 0.0) or 0.0),
                    float(patient_data.get("age", 65.0) or 65.0),
                ]
            ],
            dtype=float,
        )
        times = np.array([2.0, 5.0, 8.0], dtype=float)
        survival = self.prodromal_model.predict_survival_curve(feature_vector, times=times)[0]
        return {f"{int(time)}y_conversion_prob": round(float(1.0 - probability), 4) for time, probability in zip(times, survival, strict=False)}

    def _kg_reasoning(self, patient_data: dict[str, Any]) -> list[str]:
        """Extract concise KG context around the available risk factors."""

        query_parts: list[str] = []
        for key in ("genetic_variants", "rbd_present", "hyposmia", "saa_result", "nfl_pg_ml"):
            value = patient_data.get(key)
            if isinstance(value, list):
                query_parts.extend(str(item) for item in value)
            elif isinstance(value, bool) and value:
                query_parts.append(key.replace("_", " "))
            elif isinstance(value, (int, float)):
                query_parts.append(key.replace("_", " "))
        if not query_parts:
            return []
        return self.kg_extractor.build_context(" ".join(query_parts), max_triples=5)

    def _combine_scores(
        self,
        *,
        ml_outputs: dict[str, torch.Tensor],
        criteria: dict[str, Any],
        kg_context: list[str],
    ) -> float:
        """Blend ML, criteria, and KG evidence into a final risk score."""

        diagnosis_prob = ml_outputs["diagnosis_prob"].squeeze(0).detach().cpu().numpy()
        ml_risk = float(diagnosis_prob[1] * 0.6 + diagnosis_prob[2] * 0.4)
        criteria_score = float(criteria.get("score", 0.0))
        kg_bonus = min(0.15, len(kg_context) * 0.02)
        combined = 0.45 * ml_risk + 0.45 * criteria_score + 0.10 * kg_bonus
        return round(float(max(0.0, min(combined, 0.99))), 4)

    def _confidence_interval(self, risk: float, observed_modalities: int) -> tuple[float, float]:
        """Return a simple confidence interval that widens with missing modalities."""

        missing_modalities = len(self.feature_registry.MODALITIES) - observed_modalities
        margin = min(0.25, 0.08 + missing_modalities * 0.02)
        lower = round(max(0.0, risk - margin), 4)
        upper = round(min(0.99, risk + margin), 4)
        return lower, upper

    def _recommended_tests(self, available_modalities: Sequence[str]) -> list[str]:
        """Recommend the next tests that would most improve certainty."""

        suggestions: list[str] = []
        if "molecular" not in available_modalities:
            suggestions.append("alpha-synuclein SAA and blood NfL")
        if "genetic" not in available_modalities:
            suggestions.append("PD-focused genetic panel or PRS")
        if "imaging" not in available_modalities:
            suggestions.append("DaTSCAN or neuromelanin MRI")
        if "nonmotor" not in available_modalities:
            suggestions.append("Formal RBD and hyposmia assessment")
        if "digital" not in available_modalities:
            suggestions.append("Voice or gait digital biomarker testing")
        return suggestions

    def _monitoring_timeline(self, risk: float) -> str:
        """Return a monitoring interval recommendation."""

        if risk >= 0.7:
            return "Repeat multimodal assessment every 6 months."
        if risk >= 0.4:
            return "Repeat multimodal assessment every 12 months."
        return "Repeat multimodal assessment every 18 to 24 months."

    def _build_prompt(
        self,
        *,
        task: str,
        patient_data: dict[str, Any],
        combined_risk: float,
        confidence_interval: tuple[float, float],
        criteria: dict[str, Any],
        recommendations: list[str],
        monitoring_timeline: str,
    ) -> str:
        """Build the risk-report prompt."""

        return "\n".join(
            [
                f"Task: {task}",
                f"Patient data: {patient_data}",
                f"Combined risk: {combined_risk}",
                f"Confidence interval: {confidence_interval}",
                f"Criteria score: {criteria}",
                f"Recommendations: {recommendations}",
                f"Monitoring timeline: {monitoring_timeline}",
                "Write a concise PD risk report with confidence interval and next steps.",
            ]
        )

    def _fallback_report(
        self,
        combined_risk: float,
        confidence_interval: tuple[float, float],
        criteria: dict[str, Any],
        recommendations: list[str],
        monitoring_timeline: str,
    ) -> str:
        """Build a deterministic risk report."""

        recommendation_text = ", ".join(recommendations) if recommendations else "No immediate additional testing was recommended."
        return (
            f"Combined prodromal or progression risk is {combined_risk:.2f} "
            f"(95% CI {confidence_interval[0]:.2f}-{confidence_interval[1]:.2f}). "
            f"MDS-style criteria score is {float(criteria.get('score', 0.0)):.2f} with factors {criteria.get('factors', [])}. "
            f"Recommended next tests: {recommendation_text} {monitoring_timeline}"
        )

    def _try_generate(self, prompt: str) -> str | None:
        """Attempt LLM generation with graceful fallback."""

        return try_generate_text(self.llm_client, prompt, system=system_prompt(self.name))
