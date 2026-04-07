"""Staging agent tests."""

from __future__ import annotations

from parkinson_ai.agents.staging_agent import StagingAgent
from parkinson_ai.knowledge_graph.staging import PatientData
from tests.conftest import FakeLLMClient


def test_staging_agent(mock_patient_data: PatientData, fake_llm_client: FakeLLMClient) -> None:
    """Staging agent should return both staging systems."""

    result = StagingAgent(llm_client=fake_llm_client).run("stage patient", patient_data=mock_patient_data)
    assert "nsd_iss" in result.metadata
    assert "synneurge" in result.metadata
    assert "progression" in result.metadata
    assert result.content == "staging report"
