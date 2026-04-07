"""Sentinel tests."""

from __future__ import annotations

from parkinson_ai.agents.base import AgentResult
from parkinson_ai.agents.sentinel import SentinelAgent
from parkinson_ai.knowledge_graph.staging import PatientData


def test_sentinel_flags_supported_content() -> None:
    """Cited content should pass sentinel inspection."""

    report = SentinelAgent().inspect("This may support PD biology. PMID:38181717")
    assert report["passes"] is True


def test_sentinel_flags_staging_mismatch(mock_patient_data: PatientData) -> None:
    """Sentinel should flag staging outputs that disagree with recomputed logic."""

    agent_results = [
        AgentResult(
            agent_name="staging_agent",
            content="NSD-ISS 1A; SynNeurGe S1N0G0",
            metadata={
                "nsd_iss": {"stage": "1A"},
                "synneurge": {"label": "S1N0G0"},
            },
        )
    ]

    report = SentinelAgent().inspect("NSD-ISS 1A", agent_results=agent_results, patient_data=mock_patient_data)

    assert report["passes"] is False
    assert report["issues"]
