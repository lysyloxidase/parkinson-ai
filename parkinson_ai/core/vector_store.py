"""Hybrid vector store combining BM25 and dense similarity."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import sqrt
from typing import Any

from parkinson_ai.config import get_settings
from parkinson_ai.core.embedding import EmbeddingManager

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - optional dependency
    BM25Okapi = None

try:
    import chromadb
except ImportError:  # pragma: no cover - optional dependency
    chromadb = None


@dataclass(slots=True)
class VectorDocument:
    """Document stored in the hybrid retrieval index."""

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedDocument:
    """Ranked retrieval result."""

    doc_id: str
    text: str
    metadata: dict[str, Any]
    dense_score: float
    sparse_score: float
    combined_score: float


class HybridVectorStore:
    """Local in-memory hybrid retriever with optional Chroma persistence."""

    def __init__(
        self,
        embedding_manager: EmbeddingManager | None = None,
        *,
        persist_directory: str | None = None,
        collection_name: str = "pd_documents",
    ) -> None:
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.persist_directory = persist_directory or get_settings().CHROMA_PERSIST_DIR
        self.collection_name = collection_name
        self._documents: list[VectorDocument] = []
        self._embeddings: list[list[float]] = []
        self._tokenized: list[list[str]] = []
        self._bm25: Any | None = None
        self._collection = self._init_collection()

    def add_documents(self, documents: Sequence[VectorDocument]) -> None:
        """Insert documents into the index."""

        if not documents:
            return
        self._documents.extend(documents)
        self._embeddings.extend(self.embedding_manager.embed_texts([doc.text for doc in documents]))
        self._tokenized.extend(_tokenize(doc.text) for doc in documents)
        self._bm25 = BM25Okapi(self._tokenized) if BM25Okapi is not None and self._tokenized else None
        if self._collection is not None:  # pragma: no cover - persistence is optional
            self._collection.add(
                ids=[doc.doc_id for doc in documents],
                documents=[doc.text for doc in documents],
                metadatas=[doc.metadata for doc in documents],
                embeddings=self._embeddings[-len(documents) :],
            )

    def query(self, query: str, *, top_k: int | None = None) -> list[RetrievedDocument]:
        """Retrieve documents using dense and sparse ranking."""

        limit = top_k or get_settings().RAG_TOP_K
        if not self._documents:
            return []
        query_embedding = self.embedding_manager.embed_query(query)
        sparse_scores = self._sparse_scores(query)
        dense_scores = [self._cosine_similarity(query_embedding, embedding) for embedding in self._embeddings]
        results: list[RetrievedDocument] = []
        for index, document in enumerate(self._documents):
            dense_score = dense_scores[index]
            sparse_score = sparse_scores[index]
            combined_score = (dense_score + sparse_score) / 2.0
            results.append(
                RetrievedDocument(
                    doc_id=document.doc_id,
                    text=document.text,
                    metadata=document.metadata,
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                    combined_score=combined_score,
                )
            )
        results.sort(key=lambda item: item.combined_score, reverse=True)
        return results[:limit]

    def _init_collection(self) -> Any | None:
        """Create an optional Chroma collection."""

        if chromadb is None:
            return None
        try:  # pragma: no cover - depends on local chroma runtime
            client = chromadb.PersistentClient(path=self.persist_directory)
            return client.get_or_create_collection(name=self.collection_name)
        except Exception:
            return None

    def _sparse_scores(self, query: str) -> list[float]:
        """Return BM25 or token-overlap scores for each document."""

        query_tokens = _tokenize(query)
        if self._bm25 is not None:
            scores = self._bm25.get_scores(query_tokens)
            max_score = max(float(score) for score in scores) if len(scores) else 1.0
            return [float(score) / max(max_score, 1e-8) for score in scores]
        if not self._tokenized:
            return [0.0 for _ in self._documents]
        return [len(set(query_tokens).intersection(tokens)) / max(len(set(query_tokens).union(tokens)), 1) for tokens in self._tokenized]

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        """Compute cosine similarity without NumPy dependency."""

        numerator = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = sqrt(sum(a * a for a in left))
        right_norm = sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)


def _tokenize(text: str) -> list[str]:
    """Tokenize a document for sparse ranking."""

    return [token.strip(".,;:!?()[]{}").lower() for token in text.split() if token.strip()]
