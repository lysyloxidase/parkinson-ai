"""PubMed indexer tests."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from parkinson_ai.data.pubmed import PubMedArticle
from parkinson_ai.rag.pubmed_indexer import PubMedIndexer


class FakePubMedClient:
    """Small fake PubMed client for indexer testing."""

    async def search(self, query: str, *, retmax: int = 20, sort: str = "relevance") -> list[str]:
        assert query
        assert retmax > 0
        assert sort
        return ["1001", "1002"]

    async def fetch_articles(self, pmids: Sequence[str]) -> list[PubMedArticle]:
        assert list(pmids) == ["1001", "1002"]
        return [
            PubMedArticle(
                pmid="1001",
                title="Blood alpha-synuclein biomarker",
                abstract="Alpha-synuclein SAA showed strong accuracy. It supports early biological diagnosis.",
                journal="Brain",
                authors=["Anna Kluge"],
                publication_year=2024,
                mesh_terms=["Parkinson Disease", "Biomarkers"],
            ),
            PubMedArticle(
                pmid="1002",
                title="Prodromal Parkinson risk factors",
                abstract="RBD and hyposmia increase future conversion risk.",
                journal="Movement Disorders",
                authors=["Silke Simuni"],
                publication_year=2024,
                mesh_terms=["REM Sleep Behavior Disorder"],
            ),
        ]


def test_pubmed_indexer_fetches_chunks_and_indexes() -> None:
    """Indexer should fetch PubMed articles, chunk abstracts, and persist them."""

    indexer = PubMedIndexer(pubmed_client=FakePubMedClient())

    summary = asyncio.run(indexer.fetch_and_index(["Parkinson disease biomarker"], retmax=2))

    assert summary.article_count == 2
    assert summary.chunk_count >= 3
    assert len(indexer.store._documents) == summary.chunk_count  # noqa: SLF001
    assert indexer.store._documents[0].metadata["pmid"] == "1001"  # noqa: SLF001
