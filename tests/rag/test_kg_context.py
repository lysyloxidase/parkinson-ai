"""KG context tests."""

from __future__ import annotations

from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.rag.kg_context import KGContextExtractor


def test_kg_context_extractor_builds_natural_language_triples(sample_pd_graph: PDKnowledgeGraph) -> None:
    """KG context should resolve entities and convert graph triples into sentences."""

    context = KGContextExtractor(sample_pd_graph).extract("DaTSCAN putamen alpha-synuclein", max_triples=5)

    assert context.entities
    assert any(match.node_name == "DaTSCAN" for match in context.entities)
    assert context.sentences
    assert any("Putamen" in sentence for sentence in context.sentences)
