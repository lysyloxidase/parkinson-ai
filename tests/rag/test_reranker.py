"""Reranker tests."""

from __future__ import annotations

from parkinson_ai.core.vector_store import RetrievedDocument
from parkinson_ai.rag.reranker import CrossEncoderReranker


def test_cross_encoder_reranker_fallback_prefers_relevant_text() -> None:
    """Fallback reranking should promote the document with the strongest lexical match."""

    reranker = CrossEncoderReranker(model_name="__missing_model__")
    documents = [
        RetrievedDocument(
            doc_id="1",
            text="General neurodegeneration review.",
            metadata={"pmid": "1", "citation": "Smith 2024", "year": 2024},
            dense_score=0.2,
            sparse_score=0.1,
            combined_score=0.15,
        ),
        RetrievedDocument(
            doc_id="2",
            text="Alpha-synuclein seed amplification assay improves Parkinson disease diagnosis.",
            metadata={"pmid": "2", "citation": "Kluge 2024", "year": 2024},
            dense_score=0.1,
            sparse_score=0.3,
            combined_score=0.2,
        ),
    ]

    reranked = reranker.rerank("alpha-synuclein assay for Parkinson disease", documents, top_k=2)

    assert reranked[0].doc_id == "2"
    assert reranked[0].metadata["rerank_score"] >= reranked[1].metadata["rerank_score"]
