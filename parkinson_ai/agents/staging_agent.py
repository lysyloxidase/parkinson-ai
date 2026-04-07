"""Agent 6: combined NSD-ISS and SynNeurGe staging reports."""

from __future__ import annotations

from typing import Any

from parkinson_ai.agents.base import AgentResult, BaseAgent
from parkinson_ai.agents.common import SupportsGenerate, system_prompt, try_generate_text
from parkinson_ai.core.llm_client import OllamaClient
from parkinson_ai.knowledge_graph.staging import (
    NSDISSStaging,
    PatientData,
    ProgressionPrediction,
    SynNeurGeStaging,
)


class StagingAgent(BaseAgent):
    """Specialist agent for NSD-ISS and SynNeurGe classification."""

    def __init__(self, *, llm_client: SupportsGenerate | None = None) -> None:
        super().__init__("staging_agent")
        self.nsd_iss = NSDISSStaging()
        self.synneurge = SynNeurGeStaging()
        self.llm_client = llm_client or OllamaClient()

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Run staging on the provided patient payload."""

        raw_patient = kwargs.get("patient_data")
        if raw_patient is None:
            content = self._overview_report()
            generated = self._try_generate(
                "\n".join(
                    [
                        f"Task: {task}",
                        "Explain the differences between NSD-ISS and SynNeurGe for Parkinson's disease.",
                    ]
                )
            )
            if generated:
                content = generated
            return AgentResult(
                agent_name=self.name,
                content=content,
                metadata={
                    "mode": "overview",
                    "task": task,
                },
            )

        patient = self._coerce_patient(raw_patient)
        nsd = self.nsd_iss.classify(patient)
        syn = self.synneurge.classify(patient)
        progression = self.nsd_iss.predict_progression(nsd.stage, patient.model_dump())
        agreements, discrepancies = self._compare_results(nsd.stage, syn.label)
        additional_tests = self._recommended_tests(patient)
        content = self._fallback_report(nsd, syn, progression, agreements, discrepancies, additional_tests)
        prompt = self._build_prompt(task, patient, nsd.model_dump(), syn.model_dump(), progression.model_dump())
        generated = self._try_generate(prompt)
        if generated:
            content = generated
        return AgentResult(
            agent_name=self.name,
            content=content,
            metadata={
                "nsd_iss": nsd.model_dump(),
                "synneurge": syn.model_dump(),
                "progression": progression.model_dump(),
                "agreements": agreements,
                "discrepancies": discrepancies,
                "recommended_tests": additional_tests,
                "task": task,
            },
        )

    def _coerce_patient(self, payload: object) -> PatientData:
        """Normalize input payload to PatientData."""

        if isinstance(payload, PatientData):
            return payload
        if isinstance(payload, dict):
            return PatientData(**payload)
        raise ValueError("patient_data is required")

    def _compare_results(self, nsd_stage: str, syn_label: str) -> tuple[list[str], list[str]]:
        """Highlight agreements and discrepancies across the staging systems."""

        agreements: list[str] = []
        discrepancies: list[str] = []
        if nsd_stage in {"3", "4", "5", "6"} and syn_label.startswith("S1N1"):
            agreements.append("Both systems support clinically established synuclein-positive PD.")
        if nsd_stage in {"1A", "1B", "2A", "2B"} and syn_label.startswith("S1N0"):
            agreements.append("Both systems place the case before clear motor disability.")
        if syn_label.startswith("S0") and nsd_stage != "0":
            discrepancies.append("SynNeurGe allows SAA-negative genetic phenotypes that NSD-ISS does not stage as canonical NSD.")
        if nsd_stage == "0" and syn_label.endswith("G2"):
            discrepancies.append("NSD-ISS stage 0 can coexist with genetically driven SynNeurGe G2 classification.")
        return agreements, discrepancies

    def _recommended_tests(self, patient: PatientData) -> list[str]:
        """Recommend additional tests that would refine staging."""

        tests: list[str] = []
        if patient.saa_result is None:
            tests.append("alpha-synuclein SAA")
        if patient.datscan_abnormal is None and patient.datscan_sbr is None:
            tests.append("DaTSCAN")
        if patient.nfl_pg_ml is None:
            tests.append("blood NfL")
        if patient.nm_mri_abnormal is None:
            tests.append("Neuromelanin MRI")
        if not patient.genetic_variants:
            tests.append("PD-focused genetic testing")
        if patient.moca_score is None:
            tests.append("MoCA")
        return tests

    def _build_prompt(
        self,
        task: str,
        patient: PatientData,
        nsd: dict[str, Any],
        syn: dict[str, Any],
        progression: dict[str, Any],
    ) -> str:
        """Build the staging-report prompt."""

        return "\n".join(
            [
                f"Task: {task}",
                f"Patient: {patient.model_dump()}",
                f"NSD-ISS: {nsd}",
                f"SynNeurGe: {syn}",
                f"Progression: {progression}",
                "Explain the staging logic, agreements, discrepancies, and next tests.",
            ]
        )

    def _fallback_report(
        self,
        nsd: Any,
        syn: Any,
        progression: ProgressionPrediction,
        agreements: list[str],
        discrepancies: list[str],
        additional_tests: list[str],
    ) -> str:
        """Build a deterministic staging report."""

        agreement_text = " ".join(agreements) if agreements else "The two systems emphasize different aspects of disease biology."
        discrepancy_text = " ".join(discrepancies) if discrepancies else "No major discrepancy was detected."
        test_text = ", ".join(additional_tests) if additional_tests else "No additional tests were required for coarse staging."
        progression_text = (
            f"Estimated transition from stage {progression.current_stage} to {progression.next_stage} in {progression.estimated_years_to_next_stage} years."
            if progression.next_stage is not None
            else "No later NSD-ISS stage milestone is modeled."
        )
        return f"NSD-ISS stage {nsd.stage} and SynNeurGe {syn.label}. {agreement_text} {discrepancy_text} {progression_text} Additional tests that would refine staging: {test_text}."

    def _try_generate(self, prompt: str) -> str | None:
        """Attempt LLM report generation with silent fallback."""

        return try_generate_text(self.llm_client, prompt, system=system_prompt(self.name))

    def _overview_report(self) -> str:
        """Return a deterministic overview when no patient case is supplied."""

        return (
            "NSD-ISS is a stage-based framework centered on alpha-synuclein biology, "
            "neurodegeneration, prodromal features, and disability progression from stage 0 through 6. "
            "SynNeurGe is an axis-based framework that labels synuclein status (S), neurodegeneration (N), "
            "and genetic contribution (G), which makes it better suited to describing SAA-negative genetic PD phenotypes. "
            "In practice, NSD-ISS is useful for milestone-style staging, while SynNeurGe is useful for biological subtype description."
        )
