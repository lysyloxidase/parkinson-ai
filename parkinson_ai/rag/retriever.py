"""Hybrid retrieval specialized for Parkinson's disease literature."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from parkinson_ai.config import get_settings
from parkinson_ai.core.embedding import EmbeddingManager
from parkinson_ai.core.vector_store import HybridVectorStore, RetrievedDocument, VectorDocument
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.rag.kg_context import KGContextExtractor, KGContextResult
from parkinson_ai.rag.reranker import CrossEncoderReranker

_PD_SYNONYMS: dict[str, tuple[str, ...]] = {
    "pd": ("Parkinson disease", "Parkinson's disease", "parkinsonism"),
    "parkinson disease": ("Parkinson's disease", "parkinsonism", "synucleinopathy"),
    "parkinson": ("Parkinson disease", "parkinsonism"),
    "alpha synuclein": ("alpha-synuclein", "SNCA", "Lewy pathology"),
    "alpha-synuclein": ("alpha synuclein", "SNCA", "Lewy pathology"),
    "saa": ("seed amplification assay", "RT-QuIC", "alpha-synuclein seeding"),
    "nfl": ("neurofilament light", "serum NfL", "CSF NfL"),
    "rbd": ("REM sleep behavior disorder",),
    "datscan": ("dopamine transporter imaging", "striatal dopaminergic deficit"),
}


@dataclass(slots=True)
class RetrievalResult:
    """Hybrid retrieval result enriched with graph context and rank traces."""

    documents: list[RetrievedDocument]
    kg_context: list[str] = field(default_factory=list)
    query_entities: list[str] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)
    sparse_results: list[RetrievedDocument] = field(default_factory=list)
    dense_results: list[RetrievedDocument] = field(default_factory=list)


class HybridPDRetriever:
    """Combine BM25, dense search, KG context, and cross-encoder reranking."""

    def __init__(
        self,
        store: HybridVectorStore | None = None,
        *,
        graph: PDKnowledgeGraph | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.store = store or HybridVectorStore(embedding_manager=EmbeddingManager(model_name="medcpt"))
        self.graph = graph
        self.kg_extractor = KGContextExtractor(graph) if graph is not None else None
        self.reranker = reranker or CrossEncoderReranker()

    def index_documents(self, documents: Sequence[VectorDocument]) -> None:
        """Add PubMed chunks or other PD text documents to the hybrid store."""

        self.store.add_documents(list(documents))

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        candidate_k: int | None = None,
        use_reranker: bool = True,
    ) -> RetrievalResult:
        """Retrieve PD literature with sparse+dense fusion and KG grounding."""

        limit = top_k or get_settings().RAG_TOP_K
        candidates = max(candidate_k or limit * 3, limit)
        expanded_queries = self.expand_query(query)

        sparse_results = self._sparse_search(expanded_queries, top_k=candidates)
        dense_results = self._dense_search(expanded_queries, top_k=candidates)
        fused_results = self._reciprocal_rank_fusion((sparse_results, dense_results), top_k=candidates)

        kg_context_result = KGContextResult()
        if self.kg_extractor is not None:
            kg_context_result = self.kg_extractor.extract(query, max_triples=get_settings().KG_CONTEXT_MAX_TRIPLES)

        if use_reranker and fused_results:
            fused_results = self.reranker.rerank(
                query=query,
                documents=fused_results,
                top_k=limit,
                kg_context=kg_context_result.sentences,
            )
        else:
            fused_results = fused_results[:limit]

        return RetrievalResult(
            documents=fused_results,
            kg_context=kg_context_result.sentences,
            query_entities=[match.node_name for match in kg_context_result.entities],
            expanded_queries=expanded_queries,
            sparse_results=sparse_results[:limit],
            dense_results=dense_results[:limit],
        )

    def expand_query(self, query: str) -> list[str]:
        """Expand a query with PD-specific synonyms and biomarker aliases."""

        normalized_query = query.strip()
        lowered = normalized_query.lower()
        expanded: list[str] = [normalized_query]
        for key, synonyms in _PD_SYNONYMS.items():
            if re.search(rf"\b{re.escape(key)}\b", lowered):
                expanded.extend(f"{normalized_query} {synonym}" for synonym in synonyms)
        if "parkinson" not in lowered and "pd" not in lowered:
            expanded.append(f"Parkinson disease {normalized_query}")
        if all("parkinson" not in candidate.lower() for candidate in expanded):
            expanded.append(f"{normalized_query} parkinsonism")
        return _deduplicate(expanded)

    def _sparse_search(self, expanded_queries: Sequence[str], *, top_k: int) -> list[RetrievedDocument]:
        """Run BM25-style sparse ranking across expanded queries."""

        if not self.store._documents:  # noqa: SLF001
            return []
        score_map: dict[str, tuple[VectorDocument, float]] = {}
        for query in expanded_queries:
            sparse_scores = self.store._sparse_scores(query)  # noqa: SLF001
            for index, document in enumerate(self.store._documents):  # noqa: SLF001
                sparse_score = sparse_scores[index]
                current = score_map.get(document.doc_id)
                if current is None or sparse_score > current[1]:
                    score_map[document.doc_id] = (document, sparse_score)
        results = [
            RetrievedDocument(
                doc_id=document.doc_id,
                text=document.text,
                metadata=document.metadata,
                dense_score=0.0,
                sparse_score=score,
                combined_score=score,
            )
            for document, score in score_map.values()
        ]
        results.sort(key=lambda item: item.sparse_score, reverse=True)
        return results[:top_k]

    def _dense_search(self, expanded_queries: Sequence[str], *, top_k: int) -> list[RetrievedDocument]:
        """Run MedCPT-style dense ranking across expanded queries."""

        if not self.store._documents:  # noqa: SLF001
            return []
        score_map: dict[str, tuple[VectorDocument, float]] = {}
        for query in expanded_queries:
            query_embedding = self.store.embedding_manager.embed_query(query)
            for index, document in enumerate(self.store._documents):  # noqa: SLF001
                embedding = self.store._embeddings[index]  # noqa: SLF001
                dense_score = self.store._cosine_similarity(query_embedding, embedding)  # noqa: SLF001
                current = score_map.get(document.doc_id)
                if current is None or dense_score > current[1]:
                    score_map[document.doc_id] = (document, dense_score)
        results = [
            RetrievedDocument(
                doc_id=document.doc_id,
                text=document.text,
                metadata=document.metadata,
                dense_score=score,
                sparse_score=0.0,
                combined_score=score,
            )
            for document, score in score_map.values()
        ]
        results.sort(key=lambda item: item.dense_score, reverse=True)
        return results[:top_k]

    def _reciprocal_rank_fusion(
        self,
        rankings: Sequence[Sequence[RetrievedDocument]],
        *,
        top_k: int,
        constant: int = 60,
    ) -> list[RetrievedDocument]:
        """Merge sparse and dense rankings using reciprocal rank fusion."""

        scores: defaultdict[str, float] = defaultdict(float)
        best_document: dict[str, RetrievedDocument] = {}
        for ranking in rankings:
            for rank, document in enumerate(ranking, start=1):
                scores[document.doc_id] += 1.0 / (constant + rank)
                current = best_document.get(document.doc_id)
                if current is None or document.combined_score > current.combined_score:
                    best_document[document.doc_id] = document

        fused_results: list[RetrievedDocument] = []
        for doc_id, fused_score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]:
            document = best_document[doc_id]
            fused_results.append(
                RetrievedDocument(
                    doc_id=document.doc_id,
                    text=document.text,
                    metadata=document.metadata,
                    dense_score=document.dense_score,
                    sparse_score=document.sparse_score,
                    combined_score=fused_score,
                )
            )
        return fused_results


HybridRetriever = HybridPDRetriever


def _deduplicate(items: Sequence[str]) -> list[str]:
    """Return a list with duplicates removed while preserving order."""

    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_items.append(normalized)
    return unique_items
