"""Cross-encoder reranking for Parkinson's disease literature retrieval."""

from __future__ import annotations

import importlib
from math import log1p
from typing import Any

from parkinson_ai.core.utils import get_logger
from parkinson_ai.core.vector_store import RetrievedDocument

logger = get_logger(__name__)


def _load_sentence_transformers() -> Any | None:
    """Load sentence-transformers lazily when available."""

    try:
        return importlib.import_module("sentence_transformers")
    except ImportError:  # pragma: no cover - optional dependency
        return None


_sentence_transformers = _load_sentence_transformers()


class CrossEncoderReranker:
    """Rerank retrieved abstracts with a cross-encoder or lexical fallback."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        self.model_name = model_name
        self._model = self._load_model(model_name)

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        *,
        top_k: int | None = None,
        kg_context: list[str] | None = None,
    ) -> list[RetrievedDocument]:
        """Return reranked documents, preserving retrieval metadata."""

        if not documents:
            return []
        enriched_query = self._enrich_query(query=query, kg_context=kg_context or [])
        scores = self._score_documents(query=enriched_query, documents=documents)
        reranked: list[tuple[RetrievedDocument, float]] = list(zip(documents, scores, strict=False))
        reranked.sort(key=lambda item: item[1], reverse=True)

        limit = top_k if top_k is not None else len(reranked)
        output: list[RetrievedDocument] = []
        for document, score in reranked[:limit]:
            document.metadata["rerank_score"] = round(score, 6)
            document.metadata["reranker_model"] = self.model_name
            document.combined_score = (document.combined_score + score) / 2.0
            output.append(document)
        return output

    def _load_model(self, model_name: str) -> Any | None:
        """Load the cross-encoder when available locally."""

        if _sentence_transformers is None:
            return None
        cross_encoder = getattr(_sentence_transformers, "CrossEncoder", None)
        if cross_encoder is None:
            return None
        try:
            return cross_encoder(model_name)
        except Exception:  # pragma: no cover - model availability depends on runtime
            logger.warning("Falling back to lexical reranking", exc_info=True)
            return None

    def _score_documents(self, *, query: str, documents: list[RetrievedDocument]) -> list[float]:
        """Return relevance scores for the provided query-document pairs."""

        if self._model is not None:
            pairs = [(query, document.text) for document in documents]
            raw_scores = self._model.predict(pairs, show_progress_bar=False)
            return [float(score) for score in raw_scores]
        return [self._fallback_score(query, document) for document in documents]

    def _enrich_query(self, *, query: str, kg_context: list[str]) -> str:
        """Append compact graph evidence to the reranker query."""

        if not kg_context:
            return query
        snippets = " ".join(kg_context[:3])
        return f"{query}\nRelevant PD graph facts: {snippets}"

    def _fallback_score(self, query: str, document: RetrievedDocument) -> float:
        """Compute a deterministic lexical score when no cross-encoder is available."""

        query_tokens = _tokenize(query)
        document_tokens = _tokenize(document.text)
        overlap = len(query_tokens.intersection(document_tokens))
        density = overlap / max(len(query_tokens), 1)
        metadata_bonus = 0.0
        citation = str(document.metadata.get("citation", ""))
        if citation:
            metadata_bonus += 0.05
        year = document.metadata.get("year")
        if isinstance(year, int):
            metadata_bonus += min(log1p(max(year - 2000, 0)) / 20.0, 0.15)
        return density + metadata_bonus + max(document.combined_score, 0.0) * 0.5


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase alphanumeric terms."""

    cleaned = "".join(character.lower() if character.isalnum() else " " for character in text)
    return {token for token in cleaned.split() if token}
