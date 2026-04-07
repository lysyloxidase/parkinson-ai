"""Knowledge graph query tests."""

from __future__ import annotations

from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.knowledge_graph.queries import PDGraphQueries


def test_genes_for_pathway(sample_pd_graph: PDKnowledgeGraph) -> None:
    """Mitophagy genes should be discoverable."""

    queries = PDGraphQueries(sample_pd_graph)
    genes = queries.genes_for_pathway("mitophagy")
    assert "SNCA" in genes or "LRRK2" in genes


def test_differential_biomarkers(sample_pd_graph: PDKnowledgeGraph) -> None:
    """Disease differential biomarkers should prefer PD-only markers."""

    queries = PDGraphQueries(sample_pd_graph)
    biomarkers = queries.differential_biomarkers("Parkinson", "PSP")
    assert "alpha-synuclein SAA (CSF)" in biomarkers
