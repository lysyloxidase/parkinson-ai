"""Task decomposition helpers for Parkinson's disease multi-agent workflows."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from parkinson_ai.orchestration.state import Task

_BIOMARKER_KEYS = ("saa", "nfl", "dj-1", "gcase", "il-6", "tnf", "nlr", "urate")
_GENETIC_KEYS = ("lrrk2", "gba1", "snca", "park2", "pink1", "dj-1", "vps35", "prs")
_IMAGING_KEYS = ("datscan", "mri", "pet", "qsm", "neuromelanin")


class QueryDecomposer:
    """Heuristic decomposer for PD clinical and research queries."""

    def decompose(self, query: str, *, patient_data: dict[str, Any] | None = None) -> list[Task]:
        """Break a complex query into specialist-agent tasks."""

        lowered = query.lower()
        patient_payload = patient_data or {}
        tasks: list[Task] = []

        if patient_payload or any(token in lowered for token in ("assess this patient", "workup", "stage this patient")):
            biomarker_payloads = self._biomarker_payloads(query, patient_payload)
            for payload in biomarker_payloads:
                tasks.append(Task(agent="biomarker_interpreter", description=f"Interpret {payload['biomarker_name']}.", payload=payload))
            if self._has_genetic_context(lowered, patient_payload):
                tasks.append(Task(agent="genetic_counselor", description="Interpret PD genetic findings.", payload={"patient_data": patient_payload}))
            if self._has_imaging_context(lowered, patient_payload):
                tasks.append(Task(agent="imaging_analyst", description="Interpret neuroimaging results.", payload={"patient_data": patient_payload, "values": patient_payload}))
            tasks.append(Task(agent="staging_agent", description="Stage the patient with NSD-ISS and SynNeurGe.", payload={"patient_data": patient_payload}))
            tasks.append(Task(agent="risk_assessor", description="Estimate prodromal or progression risk.", payload={"patient_data": patient_payload}))

        if "pubmed" in lowered or "literature" in lowered or "citation" in lowered or "paper" in lowered:
            tasks.append(Task(agent="literature_agent", description="Retrieve and summarize PD literature.", payload={}))
        if "drug" in lowered or "trial" in lowered or "therapeutic" in lowered or "levodopa" in lowered or "prasinezumab" in lowered:
            tasks.append(Task(agent="drug_analyst", description="Analyze PD therapeutics.", payload={}))
        if "graph" in lowered or "connect" in lowered or "pathway" in lowered or "network" in lowered:
            tasks.append(Task(agent="kg_explorer", description="Explain PD KG connections.", payload={}))

        if not tasks:
            tasks.extend(self._simple_decomposition(query))
        return _deduplicate_tasks(tasks)

    def _biomarker_payloads(self, query: str, patient_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Return one biomarker task per available biomarker measurement."""

        payloads: list[dict[str, Any]] = []
        lowered = query.lower()
        if "saa" in lowered or "saa_result" in patient_data:
            payloads.append({"biomarker_name": "alpha-synuclein SAA (CSF)", "value": patient_data.get("saa_result"), "patient_data": patient_data})
        if "nfl" in lowered or "nfl_pg_ml" in patient_data:
            payloads.append({"biomarker_name": "NfL (blood/serum)", "value": patient_data.get("nfl_pg_ml"), "patient_data": patient_data})
        if "dj-1" in lowered or "dj1_ng_ml" in patient_data:
            payloads.append({"biomarker_name": "DJ-1 (CSF)", "value": patient_data.get("dj1_ng_ml"), "patient_data": patient_data})
        return payloads

    def _has_genetic_context(self, lowered_query: str, patient_data: dict[str, Any]) -> bool:
        """Return whether the query contains genetic information."""

        variants = patient_data.get("genetic_variants")
        return bool(variants) or any(token in lowered_query for token in _GENETIC_KEYS)

    def _has_imaging_context(self, lowered_query: str, patient_data: dict[str, Any]) -> bool:
        """Return whether the query contains imaging information."""

        return any(key in patient_data for key in ("datscan_abnormal", "datscan_sbr", "nm_mri_abnormal")) or any(token in lowered_query for token in _IMAGING_KEYS)

    def _simple_decomposition(self, query: str) -> list[Task]:
        """Return a sentence-level fallback decomposition."""

        tasks: list[Task] = []
        for sentence in [part.strip() for part in query.split(".") if part.strip()]:
            tasks.append(Task(agent="router", description=sentence, payload={}))
        return tasks or [Task(agent="router", description=query, payload={})]


def decompose_query(query: str, patient_data: dict[str, Any] | None = None) -> list[str]:
    """Return a human-readable list of decomposed task descriptions."""

    return [task.description for task in QueryDecomposer().decompose(query, patient_data=patient_data)]


def _deduplicate_tasks(tasks: Sequence[Task]) -> list[Task]:
    """Deduplicate tasks by agent and description."""

    seen: set[tuple[str, str]] = set()
    output: list[Task] = []
    for task in tasks:
        key = (task.agent, task.description)
        if key not in seen:
            seen.add(key)
            output.append(task)
    return output
