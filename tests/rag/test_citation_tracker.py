"""Citation tracker tests."""

from __future__ import annotations

from parkinson_ai.core.vector_store import RetrievedDocument
from parkinson_ai.rag.citation_tracker import CitationTracker


def test_citation_tracker_verifies_author_year_labels() -> None:
    """Citation tracker should map citations back to PMIDs."""

    tracker = CitationTracker()
    documents = [
        RetrievedDocument(
            doc_id="pmid:1:sent:1",
            text="Blood alpha-synuclein assay showed 90 percent sensitivity.",
            metadata={
                "pmid": "39990002",
                "title": "Blood alpha-synuclein assay for early PD",
                "authors": "Anna Kluge",
                "year": 2024,
                "journal": "Brain",
                "citation": "Kluge 2024",
            },
            dense_score=0.9,
            sparse_score=0.8,
            combined_score=0.85,
        )
    ]

    verification = tracker.verify(
        "Blood alpha-synuclein assays showed strong diagnostic performance [Kluge 2024].",
        documents,
    )

    assert verification.all_valid is True
    assert verification.verified["Kluge 2024"].pmid == "39990002"
