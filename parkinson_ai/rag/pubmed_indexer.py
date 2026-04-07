"""Index Parkinson's disease PubMed abstracts into the local vector store."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from parkinson_ai.core.embedding import EmbeddingManager
from parkinson_ai.core.vector_store import HybridVectorStore, VectorDocument
from parkinson_ai.data.pubmed import PubMedArticle, PubMedClient
from parkinson_ai.rag.citation_tracker import CitationTracker


def build_default_pd_queries() -> list[str]:
    """Return the default PD literature queries requested for the indexer."""

    start_year = date.today().year - 4
    last_five_years = f'("{start_year}/01/01"[Date - Publication] : "3000"[Date - Publication])'
    return [
        f'("Parkinson disease biomarker"[Title/Abstract]) AND {last_five_years}',
        "alpha synuclein seed amplification assay",
        "Parkinson machine learning prediction",
        "prodromal Parkinson risk factors",
        "Parkinson disease treatment clinical trial 2024",
    ]


@dataclass(slots=True)
class IndexingSummary:
    """Summary of an indexing run."""

    query_count: int
    article_count: int
    chunk_count: int
    pmids: list[str]


class SupportsPubMedClient(Protocol):
    """Protocol used by the indexer for PubMed-like clients."""

    async def search(
        self,
        query: str,
        *,
        retmax: int = 20,
        sort: str = "relevance",
    ) -> list[str]:
        """Search PubMed and return PMIDs."""

    async def fetch_articles(self, pmids: Sequence[str]) -> list[PubMedArticle]:
        """Fetch structured PubMed articles."""


class PubMedIndexer:
    """Fetch PubMed abstracts, sentence-chunk them, and index them locally."""

    def __init__(
        self,
        store: HybridVectorStore | None = None,
        *,
        pubmed_client: SupportsPubMedClient | None = None,
        citation_tracker: CitationTracker | None = None,
    ) -> None:
        self.store = store or HybridVectorStore(embedding_manager=EmbeddingManager(model_name="medcpt"))
        self.pubmed_client = pubmed_client or PubMedClient()
        self.citation_tracker = citation_tracker or CitationTracker()
        self._indexed_doc_ids: set[str] = set()

    async def fetch_articles_for_queries(
        self,
        queries: Sequence[str] | None = None,
        *,
        retmax: int = 20,
        sort: str = "relevance",
    ) -> list[PubMedArticle]:
        """Fetch PubMed articles for one or more PD-focused queries."""

        effective_queries = list(queries or build_default_pd_queries())
        pmids: list[str] = []
        seen_pmids: set[str] = set()

        for query in effective_queries:
            result_pmids = await self.pubmed_client.search(query, retmax=retmax, sort=sort)
            for pmid in result_pmids:
                if pmid and pmid not in seen_pmids:
                    seen_pmids.add(pmid)
                    pmids.append(pmid)

        return await self.pubmed_client.fetch_articles(pmids)

    def chunk_article(self, article: PubMedArticle) -> list[VectorDocument]:
        """Split a PubMed abstract into sentence-level chunks."""

        sentences = _split_sentences(article.abstract or article.title)
        if not sentences:
            return []

        documents: list[VectorDocument] = []
        metadata = self._build_metadata(article)
        for index, sentence in enumerate(sentences, start=1):
            doc_id = f"pmid:{article.pmid}:sent:{index}"
            if doc_id in self._indexed_doc_ids:
                continue
            documents.append(
                VectorDocument(
                    doc_id=doc_id,
                    text=sentence,
                    metadata={**metadata, "sentence_index": index},
                )
            )
        return documents

    def index_articles(self, articles: Sequence[PubMedArticle]) -> IndexingSummary:
        """Chunk and store PubMed articles in the hybrid vector index."""

        documents: list[VectorDocument] = []
        pmids: list[str] = []
        for article in articles:
            chunks = self.chunk_article(article)
            if chunks:
                pmids.append(article.pmid)
                documents.extend(chunks)
        self.store.add_documents(documents)
        self._indexed_doc_ids.update(document.doc_id for document in documents)
        return IndexingSummary(
            query_count=0,
            article_count=len(pmids),
            chunk_count=len(documents),
            pmids=pmids,
        )

    async def fetch_and_index(
        self,
        queries: Sequence[str] | None = None,
        *,
        retmax: int = 20,
        sort: str = "relevance",
    ) -> IndexingSummary:
        """Fetch PD literature from PubMed and add it to the local store."""

        effective_queries = list(queries or build_default_pd_queries())
        articles = await self.fetch_articles_for_queries(effective_queries, retmax=retmax, sort=sort)
        summary = self.index_articles(articles)
        return IndexingSummary(
            query_count=len(effective_queries),
            article_count=summary.article_count,
            chunk_count=summary.chunk_count,
            pmids=summary.pmids,
        )

    def _build_metadata(self, article: PubMedArticle) -> dict[str, object]:
        """Build Chroma-safe scalar metadata for an article chunk."""

        metadata: dict[str, object] = {
            "pmid": article.pmid,
            "title": article.title,
            "year": article.publication_year,
            "journal": article.journal or "",
            "mesh": "; ".join(article.mesh_terms),
            "authors": "; ".join(article.authors),
        }
        citation = self.citation_tracker.render_label(metadata)
        metadata["citation"] = citation
        return metadata


def _split_sentences(text: str) -> list[str]:
    """Split an abstract into trimmed sentence chunks."""

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]
