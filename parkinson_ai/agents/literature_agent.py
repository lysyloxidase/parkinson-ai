"""PubMed-grounded literature agent specialized for Parkinson's disease."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Protocol

from parkinson_ai.agents.base import AgentResult, BaseAgent
from parkinson_ai.core.llm_client import LLMResponse, OllamaClient
from parkinson_ai.core.vector_store import RetrievedDocument
from parkinson_ai.data.pubmed import PubMedClient
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.knowledge_graph.schema import EdgeType, NodeType, PublicationNode
from parkinson_ai.rag.citation_tracker import CitationTracker
from parkinson_ai.rag.kg_context import KGContextExtractor
from parkinson_ai.rag.pubmed_indexer import PubMedIndexer, SupportsPubMedClient
from parkinson_ai.rag.retriever import HybridPDRetriever, RetrievalResult

LITERATURE_AGENT_SYSTEM_PROMPT = (
    "You are a Parkinson's disease research assistant. Answer using ONLY "
    "information from the provided PubMed abstracts. Cite every claim as "
    "[Author Year]. Mention biomarker performance metrics (sensitivity, "
    "specificity, AUC) when available. If information is insufficient, say so."
)


class SupportsGenerate(Protocol):
    """Protocol for the local LLM client used by the literature agent."""

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a non-streaming response."""


class LiteratureAgent(BaseAgent):
    """Agent 4: PubMed-grounded PD literature Q&A with citation verification."""

    def __init__(
        self,
        *,
        graph: PDKnowledgeGraph | None = None,
        pubmed_client: SupportsPubMedClient | None = None,
        llm_client: SupportsGenerate | None = None,
        retriever: HybridPDRetriever | None = None,
        indexer: PubMedIndexer | None = None,
        citation_tracker: CitationTracker | None = None,
    ) -> None:
        super().__init__("literature_agent")
        self.graph = graph or PDKnowledgeGraph()
        self.pubmed_client = pubmed_client or PubMedClient()
        self.citation_tracker = citation_tracker or CitationTracker()
        self._retriever = retriever
        self._indexer = indexer
        self.llm_client = llm_client or OllamaClient()
        self.kg_context = KGContextExtractor(self.graph)

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Execute the PubMed-grounded PD literature workflow synchronously."""

        return asyncio.run(self.arun(task, **kwargs))

    async def arun(self, task: str, **kwargs: Any) -> AgentResult:
        """Execute the literature workflow asynchronously."""

        top_k = int(kwargs.get("top_k", 5))
        search_terms = self.extract_search_terms(task)
        pubmed_query = self._build_pubmed_query(task, search_terms)
        pmids = await self.pubmed_client.search(pubmed_query, retmax=30, sort="relevance")
        articles = await self.pubmed_client.fetch_articles(pmids)
        indexer = self._get_indexer()
        retriever = self._get_retriever()
        indexer.index_articles(articles)

        retrieval = retriever.retrieve(task, top_k=top_k, candidate_k=max(top_k * 6, 30))
        answer = await self._generate_answer(task=task, retrieval=retrieval)
        verification = self.citation_tracker.verify(answer, retrieval.documents)
        if not verification.all_valid:
            answer = self._fallback_answer(task=task, documents=retrieval.documents)
            verification = self.citation_tracker.verify(answer, retrieval.documents)

        updated_publications = self._update_graph_from_documents(retrieval.documents)
        return AgentResult(
            agent_name=self.name,
            content=answer,
            metadata={
                "task": task,
                "pubmed_query": pubmed_query,
                "search_terms": search_terms,
                "retrieved_pmids": [document.metadata.get("pmid", "") for document in retrieval.documents],
                "expanded_queries": retrieval.expanded_queries,
                "kg_context": retrieval.kg_context,
                "query_entities": retrieval.query_entities,
                "citations_valid": verification.all_valid,
                "verified_citations": {label: record.pmid for label, record in verification.verified.items()},
                "updated_publications": updated_publications,
            },
        )

    def extract_search_terms(self, task: str) -> list[str]:
        """Extract PD-specific terms from the task and the KG."""

        kg_matches = self.kg_context.extract_entities(task)
        terms = [match.node_name for match in kg_matches[:8]]
        lowered = task.lower()
        heuristic_terms = [
            keyword
            for keyword in (
                "alpha-synuclein",
                "SAA",
                "NfL",
                "DaTSCAN",
                "LRRK2",
                "GBA1",
                "RBD",
                "hyposmia",
                "levodopa",
                "prasinezumab",
            )
            if keyword.lower() in lowered
        ]
        if not any(term.lower().startswith("parkinson") for term in terms):
            heuristic_terms.insert(0, "Parkinson disease")
        return _deduplicate(terms + heuristic_terms)

    def _build_pubmed_query(self, task: str, search_terms: Sequence[str]) -> str:
        """Build a PubMed search query focused on Parkinson's disease."""

        if search_terms:
            quoted_terms = [f'"{term}"' if " " in term else term for term in search_terms[:8]]
            return " AND ".join(quoted_terms)
        if "parkinson" not in task.lower():
            return f"Parkinson disease {task}"
        return task

    async def _generate_answer(self, *, task: str, retrieval: RetrievalResult) -> str:
        """Generate a grounded answer using Ollama when available."""

        if not retrieval.documents:
            return "I could not find sufficient PubMed abstracts to answer this question."
        prompt = self._build_prompt(task=task, retrieval=retrieval)
        try:
            response = await asyncio.wait_for(
                self.llm_client.generate(prompt, system=LITERATURE_AGENT_SYSTEM_PROMPT),
                timeout=5.0,
            )
        except Exception:
            return self._fallback_answer(task=task, documents=retrieval.documents)
        if not response.response.strip():
            return self._fallback_answer(task=task, documents=retrieval.documents)
        return response.response.strip()

    def _build_prompt(self, *, task: str, retrieval: RetrievalResult) -> str:
        """Assemble the LLM prompt from reranked abstracts and KG evidence."""

        abstract_blocks = []
        for index, document in enumerate(retrieval.documents, start=1):
            citation = self.citation_tracker.format_inline(document.metadata)
            abstract_blocks.append(
                "\n".join(
                    [
                        f"Abstract {index}: {citation}",
                        f"PMID: {document.metadata.get('pmid', '')}",
                        f"Title: {document.metadata.get('title', '')}",
                        f"Abstract sentence: {document.text}",
                    ]
                )
            )
        context_block = "\n".join(retrieval.kg_context[:5]) or "No matching PD knowledge-graph context."
        return "\n\n".join(
            [
                f"Question: {task}",
                f"KG context:\n{context_block}",
                "PubMed evidence:",
                "\n\n".join(abstract_blocks),
                "Answer with only supported statements and author-year citations.",
            ]
        )

    def _fallback_answer(self, *, task: str, documents: Sequence[RetrievedDocument]) -> str:
        """Produce a citation-grounded extractive answer without an LLM."""

        if not documents:
            return "I could not find sufficient PubMed abstracts to answer this question."
        sentences: list[str] = [f"Evidence for '{task}' is limited to the retrieved PubMed abstracts."]
        for document in documents[:3]:
            citation = self.citation_tracker.format_inline(document.metadata)
            snippet = document.text.strip().rstrip(".")
            title = str(document.metadata.get("title", "")).strip()
            if title:
                sentences.append(f"{title}: {snippet} {citation}.")
            else:
                sentences.append(f"{snippet} {citation}.")
        return " ".join(sentences)

    def _update_graph_from_documents(self, documents: Sequence[RetrievedDocument]) -> list[str]:
        """Add publication nodes and biomarker-reporting edges to the PD KG."""

        updated_publications: list[str] = []
        for document in documents:
            pmid = str(document.metadata.get("pmid", "")).strip()
            if not pmid.isdigit():
                continue
            node_id = f"publication:pmid:{pmid}"
            if self.graph.get_node(node_id) is None:
                citation = self.citation_tracker.render_label(document.metadata)
                self.graph.add_node(
                    PublicationNode(
                        id=node_id,
                        name=citation or f"PMID {pmid}",
                        pmid=pmid,
                        title=str(document.metadata.get("title", "")).strip() or None,
                        year=_coerce_int(document.metadata.get("year")),
                        journal=str(document.metadata.get("journal", "")).strip() or None,
                    )
                )
            updated_publications.append(node_id)
            entity_matches = self.kg_context.extract_entities(f"{document.metadata.get('title', '')} {document.text}")
            for match in entity_matches:
                if match.node_type == NodeType.BIOMARKER.value:
                    self.graph.connect(node_id, match.node_id, EdgeType.PUBLICATION_REPORTS)
        return _deduplicate(updated_publications)

    def _get_retriever(self) -> HybridPDRetriever:
        """Build or return the cached hybrid retriever."""

        if self._retriever is None:
            self._retriever = HybridPDRetriever(graph=self.graph)
        return self._retriever

    def _get_indexer(self) -> PubMedIndexer:
        """Build or return the cached PubMed indexer."""

        if self._indexer is None:
            self._indexer = PubMedIndexer(
                store=self._get_retriever().store,
                pubmed_client=self.pubmed_client,
                citation_tracker=self.citation_tracker,
            )
        return self._indexer


def _deduplicate(items: Sequence[str]) -> list[str]:
    """Return a de-duplicated list while preserving order."""

    seen: set[str] = set()
    unique_items: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique_items.append(item)
    return unique_items


def _coerce_int(value: object) -> int | None:
    """Convert graph metadata into an integer when possible."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
