"""Imaging analyst tests."""

from __future__ import annotations

from parkinson_ai.agents.imaging_analyst import ImagingAnalystAgent
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from tests.conftest import FakeLLMClient


def test_imaging_analyst_interprets_datscan_pattern(
    sample_pd_graph: PDKnowledgeGraph,
    fake_llm_client: FakeLLMClient,
) -> None:
    """Imaging analyst should compute laterality and PD-focused differential."""

    agent = ImagingAnalystAgent(graph=sample_pd_graph, llm_client=fake_llm_client)

    result = agent.run(
        "Interpret this DaTSCAN.",
        modality="DaTSCAN",
        values={
            "left_putamen_sbr": 0.9,
            "right_putamen_sbr": 1.8,
            "left_caudate_sbr": 2.0,
            "right_caudate_sbr": 2.4,
        },
        age=67,
    )

    summary = result.metadata["summary"]
    assert summary["abnormal"] is True
    assert summary["laterality_index"] is not None
    assert result.content == "imaging report"
