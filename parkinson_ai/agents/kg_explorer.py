"""Agent 5: Parkinson's disease knowledge-graph exploration and reasoning."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field

from parkinson_ai.agents.base import AgentResult, BaseAgent
from parkinson_ai.agents.common import SupportsGenerate, system_prompt, try_generate_text
from parkinson_ai.core.llm_client import OllamaClient
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.knowledge_graph.schema import EdgeType
from parkinson_ai.rag.kg_context import KGContextExtractor, KGEntityMatch


class PathExplanation(BaseModel):
    """A readable KG path explanation."""

    nodes: list[str] = Field(default_factory=list)
    edge_types: list[str] = Field(default_factory=list)
    summary: str


class KGExplorerAgent(BaseAgent):
    """Traverse the PD KG to connect genes, biomarkers, drugs, and phenotypes."""

    def __init__(
        self,
        *,
        graph: PDKnowledgeGraph | None = None,
        llm_client: SupportsGenerate | None = None,
    ) -> None:
        super().__init__("kg_explorer")
        self.graph = graph or PDKnowledgeGraph()
        self.llm_client = llm_client or OllamaClient()
        self.extractor = KGContextExtractor(self.graph)

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Explain graph paths between PD entities."""

        ordered_entities = self._ordered_entity_matches(task)
        symptom_report = self._phenotype_summary(task, ordered_entities)
        if symptom_report is not None:
            content, metadata = symptom_report
            generated = self._try_generate(
                "\n".join(
                    [
                        f"Task: {task}",
                        f"Phenotype summary: {metadata}",
                        "Explain the likely early or characteristic PD symptoms using only the graph context.",
                    ]
                )
            )
            if generated:
                content = generated
            return AgentResult(agent_name=self.name, content=content, metadata=metadata)
        selected = self._select_entity_pair(ordered_entities)
        explanations = self._explain_paths(selected[0], selected[1]) if len(selected) == 2 else []
        prompt = self._build_prompt(task, ordered_entities, explanations)
        content = self._fallback_report(selected, explanations)
        generated = self._try_generate(prompt)
        if generated:
            content = generated
        return AgentResult(
            agent_name=self.name,
            content=content,
            metadata={
                "entities": [asdict(match) for match in ordered_entities],
                "selected_entities": [match.node_name for match in selected],
                "paths": [item.model_dump() for item in explanations],
            },
        )

    def _ordered_entity_matches(self, task: str) -> list[KGEntityMatch]:
        """Return entity matches in query order."""

        matches = self.extractor.extract_entities(task)
        lower_task = task.lower()
        return sorted(
            matches,
            key=lambda match: (
                lower_task.find(match.matched_alias.lower()) if match.matched_alias.lower() in lower_task else 10_000,
                -match.score,
            ),
        )

    def _select_entity_pair(self, matches: Sequence[KGEntityMatch]) -> list[KGEntityMatch]:
        """Select the two entities that should be connected."""

        if len(matches) >= 2:
            non_disease = [match for match in matches if match.node_type != "Disease"]
            if len(non_disease) >= 2:
                return non_disease[:2]
            return list(matches[:2])
        if len(matches) == 1:
            pd_match = next((match for match in matches if "parkinson" in match.node_name.lower()), None)
            if pd_match is not None:
                return [matches[0], pd_match]
            synthetic = KGEntityMatch(
                node_id="disease:pd",
                node_name="Parkinson disease",
                node_type="Disease",
                matched_alias="parkinson disease",
                score=1.0,
            )
            return [matches[0], synthetic]
        return []

    def _phenotype_summary(
        self,
        task: str,
        matches: Sequence[KGEntityMatch],
    ) -> tuple[str, dict[str, Any]] | None:
        """Summarize disease phenotypes for symptom-oriented questions."""

        if not re.search(r"\b(symptom|symptoms|sign|signs|phenotype|phenotypes|early)\b", task.lower()):
            return None
        disease_match = next((match for match in matches if match.node_type == "Disease"), None)
        if disease_match is None and re.search(r"\b(pd|parkinson)\b", task.lower()):
            disease_match = KGEntityMatch(
                node_id="disease:pd",
                node_name="Parkinson disease",
                node_type="Disease",
                matched_alias="parkinson disease",
                score=1.0,
            )
        if disease_match is None or disease_match.node_id not in self.graph.graph:
            return None

        phenotypes: list[str] = []
        for _, target, attributes in self.graph.graph.out_edges(disease_match.node_id, data=True):
            if str(attributes.get("type", "")) != EdgeType.DISEASE_HAS_PHENOTYPE.value:
                continue
            phenotypes.append(str(self.graph.graph.nodes[target].get("name", target)))

        phenotypes = sorted(set(phenotypes))
        if not phenotypes:
            return None

        early_priority = ["REM sleep behavior disorder", "Hyposmia", "Constipation"]
        early = [name for name in early_priority if name in phenotypes]
        characteristic = [name for name in phenotypes if name not in early]
        if early:
            content = (
                f"In the seeded PD knowledge graph, early or prodromal features linked to {disease_match.node_name} "
                f"include {', '.join(early)}. Other characteristic features represented in the graph are "
                f"{', '.join(characteristic) if characteristic else 'not yet expanded beyond the prodromal set'}."
            )
        else:
            content = (
                f"In the seeded PD knowledge graph, {disease_match.node_name} is linked to the following represented "
                f"phenotypes: {', '.join(phenotypes)}."
            )
        metadata = {
            "mode": "phenotype_summary",
            "selected_entities": [disease_match.node_name],
            "phenotypes": phenotypes,
            "early_features": early,
        }
        return content, metadata

    def _explain_paths(self, left: KGEntityMatch, right: KGEntityMatch, *, max_paths: int = 3, max_hops: int = 4) -> list[PathExplanation]:
        """Find short explanatory paths between two graph nodes."""

        if left.node_id not in self.graph.graph or right.node_id not in self.graph.graph:
            return []
        graph = self.graph.graph.to_undirected(as_view=True)
        try:
            raw_paths = list(nx.all_simple_paths(graph, left.node_id, right.node_id, cutoff=max_hops))
        except nx.NetworkXNoPath:
            return []
        explanations: list[PathExplanation] = []
        for path in sorted(raw_paths, key=len)[:max_paths]:
            node_names = [str(self.graph.graph.nodes[node_id].get("name", node_id)) for node_id in path]
            edge_types: list[str] = []
            clauses: list[str] = []
            for source_id, target_id in zip(path, path[1:], strict=False):
                edge_payload = self.graph.graph.get_edge_data(source_id, target_id) or self.graph.graph.get_edge_data(target_id, source_id) or {}
                if edge_payload:
                    first_payload = next(iter(edge_payload.values()))
                    relation = str(first_payload.get("type", "related_to"))
                else:
                    relation = "related_to"
                edge_types.append(relation)
                source_name = str(self.graph.graph.nodes[source_id].get("name", source_id))
                target_name = str(self.graph.graph.nodes[target_id].get("name", target_id))
                clauses.append(f"{source_name} {relation.replace('_', ' ')} {target_name}")
            explanations.append(
                PathExplanation(
                    nodes=node_names,
                    edge_types=edge_types,
                    summary="; ".join(clauses) + ".",
                )
            )
        return explanations

    def _build_prompt(
        self,
        task: str,
        matches: Sequence[KGEntityMatch],
        explanations: Sequence[PathExplanation],
    ) -> str:
        """Build the prompt used for LLM-based biological reasoning."""

        return "\n".join(
            [
                f"Task: {task}",
                f"Matched entities: {[asdict(item) for item in matches]}",
                f"Graph paths: {[item.model_dump() for item in explanations]}",
                "Explain the biological rationale linking these PD entities.",
            ]
        )

    def _fallback_report(
        self,
        selected: Sequence[KGEntityMatch],
        explanations: Sequence[PathExplanation],
    ) -> str:
        """Produce a deterministic path explanation."""

        if len(selected) < 2:
            return "I could not resolve two PD knowledge-graph entities to connect."
        if not explanations:
            return f"I could not find a short path connecting {selected[0].node_name} and {selected[1].node_name} in the current PD KG."
        headline = f"KG path from {selected[0].node_name} to {selected[1].node_name}: "
        return headline + " ".join(item.summary for item in explanations)

    def _try_generate(self, prompt: str) -> str | None:
        """Attempt LLM generation with silent fallback."""

        return try_generate_text(self.llm_client, prompt, system=system_prompt(self.name))


KgExplorerAgent = KGExplorerAgent
