"""PD-specific graph queries."""

from __future__ import annotations

from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.knowledge_graph.schema import EdgeType, NodeType


class PDGraphQueries:
    """High-level graph queries used by retrieval and agents."""

    def __init__(self, graph: PDKnowledgeGraph) -> None:
        self.graph = graph

    def biomarkers_for_stage(self, stage: str) -> list[str]:
        """Return biomarkers explicitly linked to a stage."""

        stage_tokens = {stage.lower(), f"stage {stage}".lower()}
        biomarkers: list[str] = []
        for source, target, attributes in self.graph.graph.edges(data=True):
            if attributes.get("type") != EdgeType.BIOMARKER_PREDICTS_STAGE.value:
                continue
            target_name = str(self.graph.graph.nodes[target].get("name", "")).lower()
            if target_name in stage_tokens or any(token in target_name for token in stage_tokens):
                biomarkers.append(str(self.graph.graph.nodes[source].get("name", source)))
        return sorted(set(biomarkers))

    def genes_for_pathway(self, pathway_name: str) -> list[str]:
        """Return genes participating in a given pathway."""

        pathway_name = pathway_name.lower()
        gene_names: list[str] = []
        for source, target, attributes in self.graph.graph.edges(data=True):
            if attributes.get("type") != EdgeType.GENE_IN_PATHWAY.value:
                continue
            target_label = str(self.graph.graph.nodes[target].get("name", "")).lower()
            if pathway_name in target_label:
                gene_names.append(str(self.graph.graph.nodes[source].get("name", source)))
        return sorted(set(gene_names))

    def differential_biomarkers(self, disease_a: str, disease_b: str) -> list[str]:
        """Return biomarkers linked to disease A but not disease B."""

        disease_a_id = self._resolve_disease(disease_a)
        disease_b_id = self._resolve_disease(disease_b)
        if disease_a_id is None or disease_b_id is None:
            return []
        linked_a = self._linked_biomarkers(disease_a_id)
        linked_b = self._linked_biomarkers(disease_b_id)
        return sorted(linked_a.difference(linked_b))

    def _resolve_disease(self, name: str) -> str | None:
        """Resolve a disease name to a node id."""

        lowered = name.lower()
        for node_id, attrs in self.graph.graph.nodes(data=True):
            if attrs.get("type") != NodeType.DISEASE.value:
                continue
            node_name = str(attrs.get("name", ""))
            acronym = "".join(word[0] for word in node_name.split() if word and word[0].isalpha()).lower()
            haystacks = {node_name.lower(), str(node_id).lower(), acronym}
            if any(lowered in candidate or candidate == lowered for candidate in haystacks):
                return str(node_id)
        return None

    def _linked_biomarkers(self, disease_id: str) -> set[str]:
        """Collect biomarkers indicating a disease."""

        biomarkers: set[str] = set()
        for source, target, attributes in self.graph.graph.edges(data=True):
            if target == disease_id and attributes.get("type") == EdgeType.BIOMARKER_INDICATES.value:
                biomarkers.add(str(self.graph.graph.nodes[source].get("name", source)))
        return biomarkers
