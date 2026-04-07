"""Genetic counselor tests."""

from __future__ import annotations

from parkinson_ai.agents.genetic_counselor import GeneticCounselorAgent
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from tests.conftest import FakeLLMClient


def test_genetic_counselor_interprets_lrrk2_variant(
    sample_pd_graph: PDKnowledgeGraph,
    fake_llm_client: FakeLLMClient,
) -> None:
    """Genetic counselor should classify LRRK2 G2019S with age-adjusted penetrance."""

    agent = GeneticCounselorAgent(graph=sample_pd_graph, llm_client=fake_llm_client)

    result = agent.run("Interpret LRRK2 G2019S.", variants=["LRRK2 G2019S"], age=72)

    variant = result.metadata["variants"][0]
    assert variant["gene"] == "LRRK2"
    assert variant["classification"] == "pathogenic"
    assert variant["penetrance_estimate"] is not None
    assert result.content == "genetics report"
