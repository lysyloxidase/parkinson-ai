"""Approximate MDS prodromal Parkinson's disease risk scoring."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProdromalRiskResult(BaseModel):
    """Structured prodromal risk output."""

    score: float
    factors: list[str] = Field(default_factory=list)
    evidence_level: str


def process_risk_calculator(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return an approximate prodromal risk score from multimodal evidence."""

    patient_data = kwargs.get("patient_data")
    if patient_data is None and args:
        patient_data = args[0]
    if not isinstance(patient_data, dict):
        patient_data = {}

    score = 0.02
    factors: list[str] = []

    if bool(patient_data.get("rbd_present")):
        score += 0.30
        factors.append("RBD")
    if bool(patient_data.get("hyposmia")) or (patient_data.get("upsit_score") is not None and float(patient_data["upsit_score"]) < 22.0):
        score += 0.16
        factors.append("Hyposmia")
    if bool(patient_data.get("saa_result")):
        score += 0.24
        factors.append("Positive alpha-synuclein assay")
    if bool(patient_data.get("family_history")):
        score += 0.10
        factors.append("Family history")
    prs_value = patient_data.get("prs_score")
    if isinstance(prs_value, (int, float)) and float(prs_value) >= 1.0:
        score += 0.08
        factors.append("Elevated PRS")
    nfl_value = patient_data.get("nfl_pg_ml")
    if isinstance(nfl_value, (int, float)) and float(nfl_value) >= 14.8:
        score += 0.05
        factors.append("Elevated blood NfL")
    if bool(patient_data.get("datscan_abnormal")):
        score += 0.18
        factors.append("Abnormal DaTSCAN")
    if bool(patient_data.get("constipation")) or bool(patient_data.get("constipation_binary")):
        score += 0.03
        factors.append("Constipation")
    caffeine = patient_data.get("caffeine_intake")
    if isinstance(caffeine, (int, float)) and float(caffeine) <= 1.0:
        score += 0.02
        factors.append("Low caffeine intake")

    capped_score = max(0.0, min(round(score, 3), 0.99))
    evidence_level = "high" if capped_score >= 0.65 else "moderate" if capped_score >= 0.35 else "low"
    result = ProdromalRiskResult(score=capped_score, factors=factors, evidence_level=evidence_level)
    return result.model_dump()
