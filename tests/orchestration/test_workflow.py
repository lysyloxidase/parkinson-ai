"""Workflow tests."""

from __future__ import annotations

from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.orchestration.workflow import build_workflow
from tests.conftest import FakeLLMClient


def test_workflow_runs_multispecialty_patient_assessment(
    sample_pd_graph: PDKnowledgeGraph,
    fake_llm_client: FakeLLMClient,
) -> None:
    """Workflow should decompose a patient-assessment query and aggregate results."""

    workflow = build_workflow(graph=sample_pd_graph, llm_client=fake_llm_client)

    state = workflow.run(
        "Assess this patient: SAA+, NfL 22, LRRK2 G2019S, UPDRS-III 28, RBD for 3 years, hyposmia.",
        patient_data={
            "saa_result": True,
            "nfl_pg_ml": 22.0,
            "genetic_variants": ["LRRK2 G2019S"],
            "updrs_part3": 28.0,
            "rbd_present": True,
            "hyposmia": True,
            "motor_signs": True,
            "functional_impairment": "mild",
            "age": 67,
        },
    )

    agent_names = [result.agent for result in state.results]
    assert state.route == "multi_agent"
    assert "biomarker_interpreter" in agent_names
    assert "genetic_counselor" in agent_names
    assert "staging_agent" in agent_names
    assert "risk_assessor" in agent_names
    assert state.final_report
    assert "Sentinel verification" in state.final_report


def test_workflow_routes_simple_biomarker_query_to_specialist(
    sample_pd_graph: PDKnowledgeGraph,
    fake_llm_client: FakeLLMClient,
) -> None:
    """Simple chat-style biomarker queries should execute the specialist, not only the router."""

    workflow = build_workflow(graph=sample_pd_graph, llm_client=fake_llm_client)

    state = workflow.run("What is alpha-synuclein SAA and how accurate is it?")

    agent_names = [result.agent for result in state.results]
    assert state.route == "biomarker_interpreter"
    assert "biomarker_interpreter" in agent_names
    assert "router" not in agent_names
    assert state.final_report


def test_workflow_routes_simple_staging_query_to_staging_agent(
    sample_pd_graph: PDKnowledgeGraph,
    fake_llm_client: FakeLLMClient,
) -> None:
    """General staging comparison questions should still return a staging answer."""

    workflow = build_workflow(graph=sample_pd_graph, llm_client=fake_llm_client)

    state = workflow.run("Compare NSD-ISS vs SynNeurGe staging")

    agent_names = [result.agent for result in state.results]
    assert state.route == "staging_agent"
    assert "staging_agent" in agent_names
    assert "router" not in agent_names
    assert state.final_report


def test_workflow_routes_symptom_query_to_kg_explorer(
    sample_pd_graph: PDKnowledgeGraph,
    fake_llm_client: FakeLLMClient,
) -> None:
    """Generic symptom questions should produce a graph-backed answer."""

    workflow = build_workflow(graph=sample_pd_graph, llm_client=fake_llm_client)

    state = workflow.run("First symptoms of PD?")

    agent_names = [result.agent for result in state.results]
    assert state.route == "kg_explorer"
    assert "kg_explorer" in agent_names
    assert "router" not in agent_names
