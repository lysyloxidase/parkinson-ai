"""KG explorer tests."""

from __future__ import annotations

from parkinson_ai.agents.kg_explorer import KGExplorerAgent
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from tests.conftest import FakeLLMClient


def test_kg_explorer_finds_short_pd_path(
    sample_pd_graph: PDKnowledgeGraph,
    fake_llm_client: FakeLLMClient,
) -> None:
    """KG explorer should return a path between LRRK2 and the gut microbiome signature."""

    agent = KGExplorerAgent(graph=sample_pd_graph, llm_client=fake_llm_client)

    result = agent.run("What connects LRRK2 to gut microbiome in PD?")

    assert result.metadata["paths"]
    assert result.metadata["selected_entities"]
    assert result.content == "kg report"


def test_kg_explorer_summarizes_pd_symptoms(
    sample_pd_graph: PDKnowledgeGraph,
    fake_llm_client: FakeLLMClient,
) -> None:
    """Symptom-oriented queries should produce a phenotype summary mode."""

    agent = KGExplorerAgent(graph=sample_pd_graph, llm_client=fake_llm_client)

    result = agent.run("First symptoms of PD?")

    assert result.metadata["mode"] == "phenotype_summary"
    assert "Parkinson disease" in result.metadata["selected_entities"]
    assert result.content == "kg report"
