"""Agent 8: PD therapeutics analysis across the KG and public sources."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, cast

from parkinson_ai.agents.base import AgentResult, BaseAgent
from parkinson_ai.agents.common import SupportsGenerate, run_coro, system_prompt, try_generate_text
from parkinson_ai.core.llm_client import OllamaClient
from parkinson_ai.data.clinicaltrials import ClinicalTrialRecord, ClinicalTrialsClient
from parkinson_ai.data.open_targets import OpenTargetAssociation, OpenTargetsClient
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph


class SupportsTargetLookup(Protocol):
    """Protocol for Open Targets-like clients used by the drug analyst."""

    async def fetch_pd_targets(self, *, size: int = 20) -> list[OpenTargetAssociation]:
        """Fetch top PD targets."""


class SupportsTrialLookup(Protocol):
    """Protocol for ClinicalTrials-like clients used by the drug analyst."""

    async def search_pd_trials(self, query: str, *, max_studies: int = 10) -> list[ClinicalTrialRecord]:
        """Search PD clinical trials."""


class DrugAnalystAgent(BaseAgent):
    """Analyze PD therapeutics, targets, and trial status."""

    def __init__(
        self,
        *,
        graph: PDKnowledgeGraph | None = None,
        llm_client: SupportsGenerate | None = None,
        open_targets_client: SupportsTargetLookup | None = None,
        clinical_trials_client: SupportsTrialLookup | None = None,
    ) -> None:
        super().__init__("drug_analyst")
        self.graph = graph or PDKnowledgeGraph()
        self.llm_client = llm_client or OllamaClient()
        self.open_targets_client = open_targets_client or OpenTargetsClient()
        self.clinical_trials_client = clinical_trials_client or ClinicalTrialsClient()

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Return a PD therapeutic report for a drug or target query."""

        drug_name = self._resolve_drug_name(kwargs.get("drug_name"), task)
        kg_context = self._graph_context(drug_name)
        pd_targets = self._fetch_targets()
        trials = self._fetch_trials(drug_name)
        prompt = self._build_prompt(task, drug_name, kg_context, pd_targets, trials)
        content = self._fallback_report(drug_name, kg_context, pd_targets, trials)
        generated = self._try_generate(prompt)
        if generated:
            content = generated
        return AgentResult(
            agent_name=self.name,
            content=content,
            metadata={
                "drug_name": drug_name,
                "kg_context": kg_context,
                "open_targets": [item.model_dump() for item in pd_targets],
                "clinical_trials": [item.model_dump() for item in trials],
            },
        )

    def _resolve_drug_name(self, candidate: object, task: str) -> str:
        """Resolve a drug name from explicit input or task text."""

        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        lowered = task.lower()
        for _node_id, payload in self.graph.graph.nodes(data=True):
            if str(payload.get("type")) != "Drug":
                continue
            name = str(payload.get("name", ""))
            if name.lower() in lowered:
                return name
        if "gcase" in lowered or "gba1" in lowered:
            return "Ambroxol"
        if "alpha-syn" in lowered or "synuclein" in lowered:
            return "Prasinezumab"
        return task.strip()

    def _graph_context(self, drug_name: str) -> dict[str, list[str]]:
        """Collect KG target, pathway, and trial context for a drug."""

        targets: list[str] = []
        pathways: list[str] = []
        trials: list[str] = []
        node_ids = [str(node_id) for node_id, payload in self.graph.graph.nodes(data=True) if str(payload.get("type")) == "Drug" and str(payload.get("name", "")).lower() == drug_name.lower()]
        for drug_node_id in node_ids:
            for _, target, attributes in self.graph.graph.out_edges(drug_node_id, data=True):
                if str(attributes.get("type")) == "drug_targets_protein":
                    protein_name = str(self.graph.graph.nodes[target].get("name", target))
                    targets.append(protein_name)
                    encoded_gene = str(self.graph.graph.nodes[target].get("encoded_by", ""))
                    if encoded_gene:
                        for gene_id, payload in self.graph.graph.nodes(data=True):
                            if str(payload.get("type")) == "Gene" and str(payload.get("symbol", "")).upper() == encoded_gene.upper():
                                for _, pathway_id, gene_edge in self.graph.graph.out_edges(gene_id, data=True):
                                    if str(gene_edge.get("type")) == "gene_in_pathway":
                                        pathways.append(str(self.graph.graph.nodes[pathway_id].get("name", pathway_id)))
            for source, _, attributes in self.graph.graph.in_edges(drug_node_id, data=True):
                if str(attributes.get("type")) == "trial_investigates_drug":
                    trials.append(str(self.graph.graph.nodes[source].get("name", source)))
        return {
            "targets": _unique(targets),
            "pathways": _unique(pathways),
            "trials": _unique(trials),
        }

    def _fetch_targets(self) -> list[OpenTargetAssociation]:
        """Fetch top PD targets from Open Targets with safe fallback."""

        try:
            payload = run_coro(self.open_targets_client.fetch_pd_targets(size=10))
        except Exception:
            return []
        return cast(list[OpenTargetAssociation], payload) if isinstance(payload, list) else []

    def _fetch_trials(self, drug_name: str) -> list[ClinicalTrialRecord]:
        """Fetch drug-specific PD trials with safe fallback."""

        try:
            payload = run_coro(self.clinical_trials_client.search_pd_trials(drug_name, max_studies=5))
        except Exception:
            return []
        return cast(list[ClinicalTrialRecord], payload) if isinstance(payload, list) else []

    def _build_prompt(
        self,
        task: str,
        drug_name: str,
        kg_context: dict[str, list[str]],
        pd_targets: Sequence[OpenTargetAssociation],
        trials: Sequence[ClinicalTrialRecord],
    ) -> str:
        """Build the drug-analysis prompt."""

        return "\n".join(
            [
                f"Task: {task}",
                f"Drug: {drug_name}",
                f"KG context: {kg_context}",
                f"Open Targets associations: {[item.model_dump() for item in pd_targets]}",
                f"Clinical trials: {[item.model_dump() for item in trials]}",
                "Write a concise PD therapeutics report with mechanism, trial status, and biomarker stratification.",
            ]
        )

    def _fallback_report(
        self,
        drug_name: str,
        kg_context: dict[str, list[str]],
        pd_targets: Sequence[OpenTargetAssociation],
        trials: Sequence[ClinicalTrialRecord],
    ) -> str:
        """Build a deterministic drug report."""

        target_text = ", ".join(kg_context["targets"]) if kg_context["targets"] else "no direct KG target was found"
        pathway_text = ", ".join(kg_context["pathways"]) if kg_context["pathways"] else "no direct pathway was recovered"
        kg_trial_text = ", ".join(kg_context["trials"]) if kg_context["trials"] else "no KG trial link"
        external_trial_text = ", ".join(trial.title for trial in trials) if trials else "no additional ClinicalTrials.gov records"
        top_targets = ", ".join(association.target_symbol for association in pd_targets[:5]) if pd_targets else "no Open Targets context"
        biomarker_stratification = "SAA-positive enrichment may be relevant for alpha-synuclein-directed programs." if "alpha-synuclein" in target_text.lower() else "Biomarker stratification depends on the mechanism and trial design."
        return f"{drug_name}: targets {target_text}; pathways {pathway_text}. KG-linked trials: {kg_trial_text}. ClinicalTrials.gov: {external_trial_text}. Open Targets PD pipeline context: {top_targets}. {biomarker_stratification}"

    def _try_generate(self, prompt: str) -> str | None:
        """Attempt LLM generation with silent fallback."""

        return try_generate_text(self.llm_client, prompt, system=system_prompt(self.name))


def _unique(values: Sequence[str]) -> list[str]:
    """Deduplicate strings while preserving order."""

    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output
