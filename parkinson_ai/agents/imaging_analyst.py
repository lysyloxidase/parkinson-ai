"""Agent 3: neuroimaging interpretation for Parkinson's disease."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from parkinson_ai.agents.base import AgentResult, BaseAgent
from parkinson_ai.agents.common import SupportsGenerate, system_prompt, try_generate_text
from parkinson_ai.core.llm_client import OllamaClient
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.knowledge_graph.staging import NSDISSStaging, PatientData

_IMAGING_REFERENCE: dict[str, dict[str, float]] = {
    "datscan": {"expected_mean": 2.8, "sd": 0.6, "abnormal_cutoff": 2.0},
    "neuromelanin mri": {"expected_mean": 1.15, "sd": 0.15, "abnormal_cutoff": 0.9},
    "qsm": {"expected_mean": 110.0, "sd": 18.0, "abnormal_cutoff": 145.0},
    "fdg-pet": {"expected_mean": 0.0, "sd": 1.0, "abnormal_cutoff": 1.5},
}


class ImagingSummary(BaseModel):
    """Structured imaging interpretation summary."""

    modality: str
    abnormal: bool
    z_score: float | None = None
    laterality_index: float | None = None
    affected_regions: list[str] = Field(default_factory=list)
    differential: list[str] = Field(default_factory=list)


class ImagingAnalystAgent(BaseAgent):
    """Interpret PD neuroimaging studies and suggest differentials."""

    def __init__(
        self,
        *,
        graph: PDKnowledgeGraph | None = None,
        llm_client: SupportsGenerate | None = None,
    ) -> None:
        super().__init__("imaging_analyst")
        self.graph = graph or PDKnowledgeGraph()
        self.llm_client = llm_client or OllamaClient()
        self.staging = NSDISSStaging()

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Interpret a PD-oriented imaging result."""

        modality = self._resolve_modality(kwargs.get("modality"), task)
        values = self._normalize_values(kwargs.get("values"))
        patient = self._coerce_patient(kwargs.get("patient_data"))
        age = _coerce_float(kwargs.get("age"))
        if age is None and patient is not None and patient.age is not None:
            age = float(patient.age)
        summary = self._analyze_modality(modality, values=values, age=age, patient=patient)
        stage_implications = self._stage_implications(patient, summary)
        prompt = self._build_prompt(task, summary, values, stage_implications)
        content = self._fallback_report(summary, values, stage_implications)
        generated = self._try_generate(prompt)
        if generated:
            content = generated
        return AgentResult(
            agent_name=self.name,
            content=content,
            metadata={
                "modality": modality,
                "values": values,
                "summary": summary.model_dump(),
                "stage_implications": stage_implications,
            },
        )

    def _resolve_modality(self, modality: object, task: str) -> str:
        """Resolve the imaging modality from user input."""

        if isinstance(modality, str) and modality.strip():
            return modality.strip()
        lowered = task.lower()
        if "datscan" in lowered or "sbr" in lowered:
            return "DaTSCAN"
        if "neuromelanin" in lowered or "nm-mri" in lowered:
            return "Neuromelanin MRI"
        if "qsm" in lowered:
            return "QSM"
        if "fdg" in lowered or "pet" in lowered:
            return "FDG-PET"
        return "DaTSCAN"

    def _normalize_values(self, payload: object) -> dict[str, float]:
        """Normalize imaging value payloads."""

        if isinstance(payload, dict):
            return {str(key): float(value) for key, value in payload.items() if isinstance(value, (int, float))}
        return {}

    def _coerce_patient(self, payload: object) -> PatientData | None:
        """Coerce optional patient data to the staging model."""

        if isinstance(payload, PatientData):
            return payload
        if isinstance(payload, dict):
            return PatientData(**payload)
        return None

    def _analyze_modality(
        self,
        modality: str,
        *,
        values: dict[str, float],
        age: float | None,
        patient: PatientData | None,
    ) -> ImagingSummary:
        """Analyze a modality-specific imaging profile."""

        normalized_modality = modality.lower()
        if "datscan" in normalized_modality:
            return self._analyze_datscan(values=values, age=age, patient=patient)
        if "neuromelanin" in normalized_modality:
            return self._analyze_single_value(
                modality="Neuromelanin MRI",
                value=values.get("nm_signal") or values.get("signal") or values.get("contrast"),
                abnormal_if="low",
                affected_regions=self._affected_regions("Neuromelanin MRI"),
            )
        if "qsm" in normalized_modality:
            return self._analyze_single_value(
                modality="QSM",
                value=values.get("sn_iron") or values.get("qsm_sn_iron"),
                abnormal_if="high",
                affected_regions=self._affected_regions("QSM"),
            )
        return self._analyze_single_value(
            modality=modality,
            value=values.get("score") or values.get("pdrp_score"),
            abnormal_if="high",
            affected_regions=self._affected_regions(modality),
        )

    def _analyze_datscan(
        self,
        *,
        values: dict[str, float],
        age: float | None,
        patient: PatientData | None,
    ) -> ImagingSummary:
        """Interpret DaTSCAN SBR patterns."""

        left_putamen = values.get("left_putamen_sbr")
        right_putamen = values.get("right_putamen_sbr")
        left_caudate = values.get("left_caudate_sbr")
        right_caudate = values.get("right_caudate_sbr")

        if left_putamen is None and patient is not None and patient.datscan_sbr is not None:
            left_putamen = patient.datscan_sbr
        if right_putamen is None and patient is not None and patient.datscan_sbr is not None:
            right_putamen = patient.datscan_sbr

        measures = [value for value in (left_putamen, right_putamen, left_caudate, right_caudate) if value is not None]
        mean_sbr = sum(measures) / len(measures) if measures else None
        reference = _IMAGING_REFERENCE["datscan"]
        expected_mean = reference["expected_mean"] - max((age or 65.0) - 60.0, 0.0) * 0.02
        z_score = None if mean_sbr is None else round((mean_sbr - expected_mean) / reference["sd"], 3)
        laterality = None
        if left_putamen is not None and right_putamen is not None and (left_putamen + right_putamen) > 0:
            laterality = round(abs(left_putamen - right_putamen) / ((left_putamen + right_putamen) / 2.0), 3)

        abnormal = bool(patient.datscan_abnormal if patient is not None and patient.datscan_abnormal is not None else False)
        if mean_sbr is not None:
            abnormal = abnormal or mean_sbr < reference["abnormal_cutoff"]

        caudate_ratio = None
        if left_caudate is not None and right_caudate is not None and left_putamen is not None and right_putamen is not None:
            caudate_ratio = ((left_caudate + right_caudate) / 2.0) / max(((left_putamen + right_putamen) / 2.0), 1e-8)
        differential = self._datscan_differential(
            abnormal=abnormal,
            laterality=laterality,
            caudate_ratio=caudate_ratio,
        )
        return ImagingSummary(
            modality="DaTSCAN",
            abnormal=abnormal,
            z_score=z_score,
            laterality_index=laterality,
            affected_regions=self._affected_regions("DaTSCAN"),
            differential=differential,
        )

    def _analyze_single_value(
        self,
        *,
        modality: str,
        value: float | None,
        abnormal_if: str,
        affected_regions: list[str],
    ) -> ImagingSummary:
        """Interpret a single scalar imaging value against a seeded normal range."""

        reference = _IMAGING_REFERENCE.get(modality.lower(), {"expected_mean": 0.0, "sd": 1.0, "abnormal_cutoff": 1.0})
        z_score = None if value is None else round((value - reference["expected_mean"]) / reference["sd"], 3)
        abnormal = False
        if value is not None:
            if abnormal_if == "low":
                abnormal = value < reference["abnormal_cutoff"]
            else:
                abnormal = value > reference["abnormal_cutoff"]
        return ImagingSummary(
            modality=modality,
            abnormal=abnormal,
            z_score=z_score,
            affected_regions=affected_regions,
            differential=["Pattern is compatible with PD-related degeneration." if abnormal else "Imaging is not clearly abnormal."],
        )

    def _affected_regions(self, modality: str) -> list[str]:
        """Return graph-linked brain regions for an imaging modality."""

        regions: list[str] = []
        lowered = modality.lower()
        for node_id, payload in self.graph.graph.nodes(data=True):
            node_name = str(payload.get("name", "")).lower()
            if lowered != node_name and lowered not in node_name and node_name not in lowered:
                continue
            for _, target, attributes in self.graph.graph.out_edges(node_id, data=True):
                if str(attributes.get("type")) == "imaging_detects_change_in":
                    regions.append(str(self.graph.graph.nodes[target].get("name", target)))
        return sorted(set(regions))

    def _datscan_differential(
        self,
        *,
        abnormal: bool,
        laterality: float | None,
        caudate_ratio: float | None,
    ) -> list[str]:
        """Return a PD-focused differential diagnosis from DaTSCAN pattern features."""

        if not abnormal:
            return ["A normal DaTSCAN pattern is more consistent with essential tremor or non-degenerative tremor syndromes."]
        differential = ["Posterior putaminal dopaminergic deficit supports Parkinson disease."]
        if laterality is not None and laterality >= 0.2:
            differential.append("Marked asymmetry is more typical of idiopathic PD than MSA or PSP.")
        if caudate_ratio is not None and caudate_ratio < 1.1:
            differential.append("Prominent caudate involvement can overlap with MSA or PSP.")
        else:
            differential.append("Relative caudate sparing favors PD over more diffuse atypical parkinsonism.")
        return differential

    def _stage_implications(self, patient: PatientData | None, summary: ImagingSummary) -> list[str]:
        """Summarize staging implications from the imaging interpretation."""

        implications: list[str] = []
        if summary.abnormal:
            implications.append("Imaging abnormality supports biomarker evidence of neurodegeneration.")
        if patient is not None:
            stage = self.staging.classify(patient)
            implications.append(f"With the supplied patient context, NSD-ISS stage is {stage.stage}.")
        return implications

    def _build_prompt(
        self,
        task: str,
        summary: ImagingSummary,
        values: dict[str, float],
        stage_implications: list[str],
    ) -> str:
        """Build the imaging-interpretation prompt."""

        return "\n".join(
            [
                f"Task: {task}",
                f"Summary: {summary.model_dump()}",
                f"Quantitative values: {values}",
                f"Stage implications: {stage_implications}",
                "Write a concise radiology-style PD imaging impression with differential diagnosis.",
            ]
        )

    def _fallback_report(
        self,
        summary: ImagingSummary,
        values: dict[str, float],
        stage_implications: list[str],
    ) -> str:
        """Build a deterministic radiology-style report."""

        z_text = f" z-score {summary.z_score}." if summary.z_score is not None else ""
        laterality_text = f" Laterality index {summary.laterality_index}." if summary.laterality_index is not None else ""
        stage_text = " ".join(stage_implications) if stage_implications else "No direct staging implication was computed."
        regions = ", ".join(summary.affected_regions) if summary.affected_regions else "no specific graph-linked regions"
        differential = " ".join(summary.differential)
        return (f"{summary.modality} values {values}. Study is {'abnormal' if summary.abnormal else 'not clearly abnormal'}; affected regions include {regions}.{z_text}{laterality_text} {differential} {stage_text}").strip()

    def _try_generate(self, prompt: str) -> str | None:
        """Attempt LLM generation with graceful fallback."""

        return try_generate_text(self.llm_client, prompt, system=system_prompt(self.name))


def _coerce_float(value: object) -> float | None:
    """Convert a scalar value into float."""

    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None
