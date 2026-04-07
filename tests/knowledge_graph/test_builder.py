"""Knowledge graph builder tests."""

from __future__ import annotations

from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph


def test_graph_statistics(sample_pd_graph: PDKnowledgeGraph) -> None:
    """Graph statistics should reflect the sample graph."""

    stats = sample_pd_graph.statistics()
    assert stats["node_count"] >= 50
    assert stats["edge_count"] >= 100


def test_graph_save_load(sample_pd_graph: PDKnowledgeGraph, tmp_graph_path: str) -> None:
    """Graphs should persist and reload cleanly."""

    sample_pd_graph.save(tmp_graph_path)
    loaded = PDKnowledgeGraph.load(tmp_graph_path)
    assert loaded.graph.number_of_nodes() == sample_pd_graph.graph.number_of_nodes()
