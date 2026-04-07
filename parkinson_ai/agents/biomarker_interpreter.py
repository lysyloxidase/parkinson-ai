"""Agent 1: biomarker interpretation in Parkinson's disease clinical context."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel

from parkinson_ai.agents.base import AgentResult, BaseAgent
from parkinson_ai.agents.common import SupportsGenerate, load_reference_json, system_prompt, try_generate_text
from parkinson_ai.core.llm_client import OllamaClient
from parkinson_ai.knowledge_graph.biomarker_nodes import BIOMARKER_LIBRARY, BiomarkerDefinition
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.knowledge_graph.staging import NSDISSStaging, PatientData


class BiomarkerComparison(BaseModel):
    """Structured biomarker comparison result."""

    status: str
    summary: str
    threshold: float | None = None
    delta_from_threshold: float | None = None
    pd_supportive: bool


class BiomarkerInterpreterAgent(BaseAgent):
    """Interpret a PD biomarker using literature ranges and KG context."""

    def __init__(
        self,
        *,
        graph: PDKnowledgeGraph | None = None,
        llm_client: SupportsGenerate | None = None,
    ) -> None:
        super().__init__("biomarker_interpreter")
        self.graph = graph or PDKnowledgeGraph()
        self.llm_client = llm_client or OllamaClient()
        reference_payload = load_reference_json("biomarker_reference_ranges.json")
        biomarker_block = reference_payload.get("biomarkers", {})
        self.reference_ranges = biomarker_block if isinstance(biomarker_block, dict) else {}
        self.staging = NSDISSStaging()

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Interpret a single biomarker result in PD context."""

        biomarker_name = self._resolve_biomarker_name(
            kwargs.get("biomarker_name"),
            task=task,
            biofluid=kwargs.get("biofluid"),
        )
        if biomarker_name is None:
            raise ValueError("biomarker_name is required")
        definition = self._lookup_definition(biomarker_name)
        if definition is None:
            raise ValueError(f"Unknown biomarker: {biomarker_name}")

        patient = self._coerce_patient(kwargs.get("patient_data"))
        value = kwargs.get("value")
        if value is None:
            value = _extract_value_from_text(task)
        comparison = self._compare_value(definition, value)
        stage_implications = self._stage_implications(biomarker_name, patient, comparison)
        related_nodes = self._related_graph_context(biomarker_name)
        prompt = self._build_prompt(
            biomarker_name=biomarker_name,
            definition=definition,
            value=value,
            comparison=comparison,
            stage_implications=stage_implications,
            related_nodes=related_nodes,
        )
        content = self._fallback_report(
            biomarker_name=biomarker_name,
            value=value,
            definition=definition,
            comparison=comparison,
            stage_implications=stage_implications,
            related_nodes=related_nodes,
        )
        generated = self._try_generate(prompt)
        if generated:
            content = generated
        return AgentResult(
            agent_name=self.name,
            content=content,
            metadata={
                "biomarker_name": biomarker_name,
                "value": value,
                "comparison": comparison.model_dump(),
                "definition": dict(definition),
                "stage_implications": stage_implications,
                "related_nodes": related_nodes,
                "patient_data_used": patient.model_dump() if patient is not None else {},
            },
        )

    def _resolve_biomarker_name(self, candidate: object, *, task: str, biofluid: object) -> str | None:
        """Resolve a biomarker name from explicit input or task text."""

        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        lowered_task = task.lower()
        preferred_biofluid = str(biofluid).lower() if isinstance(biofluid, str) else ""
        aliases = {
            "nfl": ["NfL (blood/serum)", "NfL (CSF)"],
            "saa": [
                "alpha-synuclein SAA (CSF)",
                "alpha-synuclein SAA (blood)",
                "alpha-synuclein SAA (skin)",
            ],
            "dj-1": ["DJ-1 (CSF)"],
            "gcase": ["GCase activity (PBMC)"],
            "mirna": ["miRNA 7-panel (CSF)", "miR-133b + miR-221-3p (plasma)"],
            "proteomics": ["8-protein panel (plasma)"],
            "metabolite": ["5-metabolite panel (serum)"],
            "exosome": ["Exosomal alpha-syn (blood)"],
        }
        for key, options in aliases.items():
            if key in lowered_task:
                if preferred_biofluid:
                    for option in options:
                        if preferred_biofluid in option.lower():
                            return option
                return options[0]
        for name in BIOMARKER_LIBRARY:
            if name.lower() in lowered_task:
                return name
        return None

    def _lookup_definition(self, biomarker_name: str) -> BiomarkerDefinition | None:
        """Look up a biomarker definition from the reference libraries."""

        if biomarker_name in BIOMARKER_LIBRARY:
            return BIOMARKER_LIBRARY[biomarker_name]
        if biomarker_name in self.reference_ranges:
            payload = self.reference_ranges[biomarker_name]
            if isinstance(payload, dict):
                return payload  # type: ignore[return-value]
        lowered = biomarker_name.lower()
        for name, definition in BIOMARKER_LIBRARY.items():
            if lowered == name.lower():
                return definition
            if lowered in name.lower() or name.lower() in lowered:
                return definition
        return None

    def _coerce_patient(self, payload: object) -> PatientData | None:
        """Normalize optional patient context."""

        if isinstance(payload, PatientData):
            return payload
        if isinstance(payload, dict):
            return PatientData(**payload)
        return None

    def _compare_value(self, definition: BiomarkerDefinition, value: object) -> BiomarkerComparison:
        """Compare a biomarker value with healthy and PD-oriented ranges."""

        healthy_range = definition.get("reference_range_healthy")
        pd_range = definition.get("pd_range")
        unit = definition.get("measurement_unit") or ""
        if value is None:
            return BiomarkerComparison(
                status="reference_only",
                summary=(
                    f"No patient-specific value was provided. "
                    f"Healthy range: {healthy_range}. PD-associated range: {pd_range}."
                ),
                pd_supportive=False,
            )
        numeric_value = _coerce_float(value)
        threshold = _extract_threshold(healthy_range or "")
        if unit == "binary" or isinstance(value, bool) or str(value).strip().lower() in {"positive", "negative"}:
            is_positive = _is_positive(value)
            status = "positive" if is_positive else "negative"
            supportive = is_positive
            summary = f"{definition['name']} is positive, which is aligned with PD-associated biology." if is_positive else f"{definition['name']} is negative and does not support PD biology on its own."
            return BiomarkerComparison(status=status, summary=summary, pd_supportive=supportive)

        if numeric_value is None:
            return BiomarkerComparison(
                status="uninterpretable",
                summary="The biomarker value could not be parsed into a numeric reference comparison.",
                pd_supportive=False,
            )

        if threshold is not None:
            if "<" in (healthy_range or "") and numeric_value > threshold:
                status = "elevated"
                supportive = True
            elif ">" in (healthy_range or "") and numeric_value < threshold:
                status = "reduced"
                supportive = True
            elif "-" in (healthy_range or "") and _outside_range(numeric_value, healthy_range or ""):
                status = "abnormal"
                supportive = True
            else:
                status = "within_reference"
                supportive = False
            delta = round(numeric_value - threshold, 3)
            summary = f"{definition['name']} {numeric_value:g} {definition.get('measurement_unit') or ''} is {status.replace('_', ' ')} relative to the healthy reference ({healthy_range}). PD-oriented range: {pd_range}."
            return BiomarkerComparison(
                status=status,
                summary=summary.strip(),
                threshold=threshold,
                delta_from_threshold=delta,
                pd_supportive=supportive,
            )

        supportive = any(token in str(pd_range).lower() for token in {"elevated", "reduced", "positive", "higher", "lower"})
        return BiomarkerComparison(
            status="contextual",
            summary=(f"{definition['name']} {numeric_value:g} {definition.get('measurement_unit') or ''} was compared qualitatively with healthy range '{healthy_range}' and PD range '{pd_range}'.").strip(),
            pd_supportive=supportive,
        )

    def _stage_implications(
        self,
        biomarker_name: str,
        patient: PatientData | None,
        comparison: BiomarkerComparison,
    ) -> list[str]:
        """Estimate staging implications from the biomarker and patient context."""

        implications: list[str] = []
        if patient is None:
            if comparison.pd_supportive and "saa" in biomarker_name.lower():
                implications.append("A positive alpha-synuclein assay supports at least NSD-ISS stage 1A if isolated.")
            if comparison.pd_supportive and "nfl" in biomarker_name.lower():
                implications.append("Elevated NfL supports biomarker evidence of neurodegeneration.")
            return implications
        stage_result = self.staging.classify(patient)
        implications.append(f"Given the supplied patient context, NSD-ISS classification is {stage_result.stage}.")
        implications.extend(stage_result.neurodegeneration_evidence)
        return implications

    def _related_graph_context(self, biomarker_name: str) -> list[str]:
        """Collect graph neighbors and stage links for a biomarker."""

        matches = []
        lowered = biomarker_name.lower()
        for node_id, attributes in self.graph.graph.nodes(data=True):
            node_name = str(attributes.get("name", "")).lower()
            if lowered == node_name or lowered in node_name or node_name in lowered:
                matches.append(str(node_id))
        context: list[str] = []
        for node_id in matches:
            for _, target, attributes in self.graph.graph.out_edges(node_id, data=True):
                target_name = str(self.graph.graph.nodes[target].get("name", target))
                context.append(f"{biomarker_name} {attributes.get('type', 'relates to')} {target_name}")
            for source, _, attributes in self.graph.graph.in_edges(node_id, data=True):
                source_name = str(self.graph.graph.nodes[source].get("name", source))
                context.append(f"{source_name} {attributes.get('type', 'relates to')} {biomarker_name}")
        return sorted(set(context))[:8]

    def _build_prompt(
        self,
        *,
        biomarker_name: str,
        definition: BiomarkerDefinition,
        value: object,
        comparison: BiomarkerComparison,
        stage_implications: list[str],
        related_nodes: list[str],
    ) -> str:
        """Build the LLM prompt."""

        return "\n".join(
            [
                f"Biomarker: {biomarker_name}",
                f"Observed value: {value}",
                f"Healthy range: {definition.get('reference_range_healthy')}",
                f"PD range: {definition.get('pd_range')}",
                f"Performance: sensitivity={definition.get('sensitivity')}, specificity={definition.get('specificity')}, auc={definition.get('auc')}",
                f"Comparison: {comparison.summary}",
                f"Stage implications: {stage_implications}",
                f"KG context: {related_nodes}",
                "Write a concise PD biomarker interpretation with confidence level.",
            ]
        )

    def _fallback_report(
        self,
        *,
        biomarker_name: str,
        value: object,
        definition: BiomarkerDefinition,
        comparison: BiomarkerComparison,
        stage_implications: list[str],
        related_nodes: list[str],
    ) -> str:
        """Build a deterministic report when no LLM is available."""

        stage_text = " ".join(stage_implications[:2]) if stage_implications else "No direct staging inference was available."
        context_text = " ".join(related_nodes[:2]) if related_nodes else "No additional KG relationships were found."
        performance = []
        sensitivity = definition.get("sensitivity")
        specificity = definition.get("specificity")
        auc = definition.get("auc")
        if sensitivity is not None:
            performance.append(f"sensitivity {float(sensitivity) * 100:.1f}%")
        if specificity is not None:
            performance.append(f"specificity {float(specificity) * 100:.1f}%")
        if auc is not None:
            performance.append(f"AUC {float(auc):.3f}")
        performance_text = ", ".join(performance) if performance else "performance metrics were not fully available"
        return (f"{biomarker_name} = {value} {definition.get('measurement_unit') or ''}. {comparison.summary} Published performance includes {performance_text}. {stage_text} {context_text}").strip()

    def _try_generate(self, prompt: str) -> str | None:
        """Attempt LLM generation and fall back silently on failure."""

        return try_generate_text(self.llm_client, prompt, system=system_prompt(self.name))


def _coerce_float(value: object) -> float | None:
    """Convert scalar values to float."""

    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def _extract_threshold(reference_text: str) -> float | None:
    """Extract a numeric threshold from a reference-range string."""

    match = re.search(r"-?\d+(?:\.\d+)?", reference_text)
    return float(match.group(0)) if match else None


def _outside_range(value: float, reference_text: str) -> bool:
    """Return whether a value lies outside a simple a-b interval."""

    bounds = re.findall(r"-?\d+(?:\.\d+)?", reference_text)
    if len(bounds) < 2:
        return False
    low, high = float(bounds[0]), float(bounds[1])
    return value < low or value > high


def _is_positive(value: object) -> bool:
    """Return whether a biomarker result is positive."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        return lowered in {"positive", "pos", "detected", "yes", "true", "1"}
    if isinstance(value, (int, float)):
        return float(value) > 0
    return False


def _extract_value_from_text(text: str) -> object | None:
    """Extract a scalar biomarker value or binary status from free text."""

    lowered = text.strip().lower()
    if any(token in lowered for token in {" positive", "positive ", " detected", " yes", "saa+"}):
        return True
    if any(token in lowered for token in {" negative", "negative ", " not detected", " no", "saa-"}):
        return False
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if match:
        return float(match.group(0))
    return None
