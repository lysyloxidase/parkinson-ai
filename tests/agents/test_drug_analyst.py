"""Drug analyst tests."""

from __future__ import annotations

from parkinson_ai.agents.drug_analyst import DrugAnalystAgent
from parkinson_ai.data.clinicaltrials import ClinicalTrialRecord
from parkinson_ai.data.open_targets import OpenTargetAssociation
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from tests.conftest import FakeLLMClient


class FakeOpenTargetsClient:
    """Deterministic Open Targets stub."""

    async def fetch_pd_targets(self, *, size: int = 20) -> list[OpenTargetAssociation]:
        assert size == 10
        return [OpenTargetAssociation(target_id="ENSG00000145335", target_symbol="SNCA", score=0.92, approved_drugs=1)]


class FakeClinicalTrialsClient:
    """Deterministic ClinicalTrials stub."""

    async def search_pd_trials(self, query: str, *, max_studies: int = 10) -> list[ClinicalTrialRecord]:
        assert "prasinezumab" in query.lower()
        assert max_studies == 5
        return [
            ClinicalTrialRecord(
                nct_id="NCT03100149",
                title="PASADENA",
                status="Completed",
                phase="Phase 2",
                interventions=["Prasinezumab"],
            )
        ]


def test_drug_analyst_reports_targets_and_trials(
    sample_pd_graph: PDKnowledgeGraph,
    fake_llm_client: FakeLLMClient,
) -> None:
    """Drug analyst should combine KG, Open Targets, and trial data."""

    agent = DrugAnalystAgent(
        graph=sample_pd_graph,
        llm_client=fake_llm_client,
        open_targets_client=FakeOpenTargetsClient(),
        clinical_trials_client=FakeClinicalTrialsClient(),
    )

    result = agent.run("What about prasinezumab in PD?")

    assert result.metadata["drug_name"] == "Prasinezumab"
    assert result.metadata["kg_context"]["targets"]
    assert result.metadata["clinical_trials"][0]["title"] == "PASADENA"
    assert result.content == "drug report"
