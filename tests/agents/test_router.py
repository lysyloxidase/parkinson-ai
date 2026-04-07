"""Router agent tests."""

from __future__ import annotations

from parkinson_ai.agents.router import RouterAgent


def test_router_classifies_staging_task() -> None:
    """Router should direct staging requests correctly."""

    routed = RouterAgent().classify_task("Please stage this patient with NSD-ISS.")
    assert routed.agent == "staging_agent"


def test_router_classifies_symptom_question_to_kg_explorer() -> None:
    """Symptom questions should route to graph-backed phenotype reasoning."""

    routed = RouterAgent().classify_task("First symptoms of PD?")
    assert routed.agent == "kg_explorer"
