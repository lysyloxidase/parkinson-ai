"""Biomarker interpreter tests."""

from __future__ import annotations

from parkinson_ai.agents.biomarker_interpreter import BiomarkerInterpreterAgent
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.knowledge_graph.staging import PatientData
from tests.conftest import FakeLLMClient


def test_biomarker_interpreter_handles_nfl_with_patient_context(
    sample_pd_graph: PDKnowledgeGraph,
    fake_llm_client: FakeLLMClient,
) -> None:
    """Biomarker interpreter should compare NfL against the seeded cutoff."""

    agent = BiomarkerInterpreterAgent(graph=sample_pd_graph, llm_client=fake_llm_client)
    patient = PatientData(saa_result=True, datscan_abnormal=True, nfl_pg_ml=28.5, age=68)

    result = agent.run(
        "Interpret serum NfL in this PD patient.",
        biomarker_name="NfL (blood/serum)",
        value=28.5,
        patient_data=patient,
    )

    assert result.metadata["comparison"]["status"] == "elevated"
    assert "stage_implications" in result.metadata
    assert result.content == "biomarker report"


def test_biomarker_interpreter_parses_value_from_chat_query(
    sample_pd_graph: PDKnowledgeGraph,
    fake_llm_client: FakeLLMClient,
) -> None:
    """Free-text chat queries should contribute the biomarker value when present."""

    agent = BiomarkerInterpreterAgent(graph=sample_pd_graph, llm_client=fake_llm_client)

    result = agent.run("Interpret NfL level 28 pg/mL in a 65-year-old")

    assert result.metadata["biomarker_name"] == "NfL (blood/serum)"
    assert result.metadata["value"] == 28.0
    assert result.metadata["comparison"]["status"] == "elevated"
