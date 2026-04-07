"""Literature agent tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from parkinson_ai.agents.literature_agent import LiteratureAgent
from parkinson_ai.core.llm_client import LLMResponse
from parkinson_ai.data.pubmed import PubMedArticle
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph


class FakePubMedClient:
    """Fake PubMed client for literature-agent testing."""

    async def search(self, query: str, *, retmax: int = 20, sort: str = "relevance") -> list[str]:
        assert "Parkinson" in query or "parkinson" in query
        assert retmax == 30
        assert sort == "relevance"
        return ["2001", "2002"]

    async def fetch_articles(self, pmids: Sequence[str]) -> list[PubMedArticle]:
        assert list(pmids) == ["2001", "2002"]
        return [
            PubMedArticle(
                pmid="2001",
                title="Blood alpha-synuclein assay for early PD",
                abstract="Blood alpha-synuclein SAA showed 90 percent sensitivity and high specificity in Parkinson disease.",
                journal="Brain",
                authors=["Anna Kluge"],
                publication_year=2024,
                mesh_terms=["Parkinson Disease", "alpha-Synuclein"],
            ),
            PubMedArticle(
                pmid="2002",
                title="DaTSCAN and biological staging",
                abstract="DaTSCAN abnormality supports biological evidence of nigrostriatal degeneration.",
                journal="Lancet Neurology",
                authors=["Silke Simuni"],
                publication_year=2024,
                mesh_terms=["Parkinson Disease", "DaTSCAN"],
            ),
        ]


@dataclass(slots=True)
class FakeLLMClient:
    """Fake LLM returning a grounded answer."""

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, object] | None = None,
    ) -> LLMResponse:
        assert "PubMed evidence" in prompt
        assert system is not None
        assert options is None
        return LLMResponse(
            model=model or "fake-model",
            response=("Blood alpha-synuclein SAA showed strong diagnostic performance in Parkinson disease [Kluge 2024]. DaTSCAN abnormalities supported nigrostriatal degeneration evidence [Simuni 2024]."),
            raw={},
        )


def test_literature_agent_runs_grounded_pipeline(sample_pd_graph: PDKnowledgeGraph) -> None:
    """Literature agent should retrieve evidence, verify citations, and update the KG."""

    agent = LiteratureAgent(
        graph=sample_pd_graph,
        pubmed_client=FakePubMedClient(),
        llm_client=FakeLLMClient(),
    )

    result = agent.run("Summarize PD biomarker evidence for alpha-synuclein and DaTSCAN.")

    assert result.metadata["citations_valid"] is True
    assert "Kluge 2024" in result.content
    assert "publication:pmid:2001" in result.metadata["updated_publications"]
    assert sample_pd_graph.get_node("publication:pmid:2001") is not None
