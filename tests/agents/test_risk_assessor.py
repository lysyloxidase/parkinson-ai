"""Risk assessor tests."""

from __future__ import annotations

from parkinson_ai.agents.risk_assessor import RiskAssessorAgent
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from tests.conftest import FakeLLMClient


def test_risk_assessor_combines_ml_and_prodromal_scores(
    sample_pd_graph: PDKnowledgeGraph,
    fake_llm_client: FakeLLMClient,
) -> None:
    """Risk assessor should return multimodal risk outputs and confidence bounds."""

    agent = RiskAssessorAgent(graph=sample_pd_graph, llm_client=fake_llm_client)

    result = agent.run(
        "Assess prodromal risk.",
        patient_data={
            "saa_result": True,
            "nfl_pg_ml": 22.0,
            "rbd_present": True,
            "hyposmia": True,
            "age": 67,
            "prs_score": 1.4,
            "genetic_variants": ["LRRK2 G2019S"],
        },
    )

    assert 0.0 <= result.metadata["combined_risk"] <= 0.99
    assert len(result.metadata["confidence_interval"]) == 2
    assert result.metadata["available_modalities"]
    assert result.content == "risk report"
