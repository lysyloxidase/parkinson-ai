"""Retriever tests."""

from __future__ import annotations

from parkinson_ai.core.vector_store import VectorDocument
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.rag.retriever import HybridPDRetriever


def test_hybrid_pd_retriever_fuses_sparse_dense_and_kg(sample_pd_graph: PDKnowledgeGraph) -> None:
    """Retriever should return documents, query expansions, and KG context."""

    retriever = HybridPDRetriever(graph=sample_pd_graph)
    retriever.index_documents(
        [
            VectorDocument(
                "1",
                "Alpha-synuclein seed amplification assay reached 90 percent sensitivity in Parkinson disease.",
                {
                    "pmid": "39990002",
                    "title": "Blood alpha-synuclein assay for early PD",
                    "authors": "Anna Kluge",
                    "year": 2024,
                    "citation": "Kluge 2024",
                },
            ),
            VectorDocument(
                "2",
                "DaTSCAN imaging detects putaminal dopaminergic deficit in parkinsonism.",
                {
                    "pmid": "38181717",
                    "title": "Biological staging of PD",
                    "authors": "Silke Simuni",
                    "year": 2024,
                    "citation": "Simuni 2024",
                },
            ),
        ]
    )

    result = retriever.retrieve("PD alpha syn biomarker", top_k=2)

    assert result.documents
    assert result.documents[0].doc_id == "1"
    assert any("Parkinson disease" in expanded for expanded in result.expanded_queries)
    assert result.kg_context
    assert "alpha-synuclein" in " ".join(result.query_entities).lower()
