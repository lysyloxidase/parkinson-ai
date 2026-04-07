"""Import helpers for integrating external biomedical knowledge graphs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.knowledge_graph.schema import BaseNode, EdgeType, NodeType


class PrimeKGImporter:
    """Import a Parkinson-focused subset of PrimeKG."""

    def import_subset(self, nodes_path: str | Path, edges_path: str | Path) -> PDKnowledgeGraph:
        """Load CSVs and extract the PD-relevant subgraph."""

        nodes = pd.read_csv(nodes_path)
        edges = pd.read_csv(edges_path)
        return self.extract_subset(nodes, edges)

    def extract_subset(self, nodes: pd.DataFrame, edges: pd.DataFrame) -> PDKnowledgeGraph:
        """Extract rows whose endpoints are related to Parkinson disease."""

        graph = PDKnowledgeGraph()
        node_map: dict[str, str] = {}
        for record in nodes.to_dict(orient="records"):
            identifier = str(record.get("id", record.get("node_id", "")))
            name = str(record.get("name", ""))
            category = str(record.get("category", record.get("node_type", "Disease")))
            if "parkinson" not in name.lower() and category.lower() not in {"gene", "drug", "pathway"}:
                continue
            node_type = _map_node_type(category)
            node = BaseNode(id=identifier, type=node_type, name=name, description=str(record.get("source", "")))
            graph.add_node(node)
            node_map[identifier] = name
        for record in edges.to_dict(orient="records"):
            source = str(record.get("x_id", record.get("source", "")))
            target = str(record.get("y_id", record.get("target", "")))
            if source not in node_map and target not in node_map:
                continue
            relation = str(record.get("relation", "associated_with"))
            graph.connect(
                source,
                target,
                _map_edge_type(relation),
                evidence=[str(record.get("display_relation", relation))],
            )
        return graph


def _map_node_type(category: str) -> NodeType:
    """Map a PrimeKG category label onto the local schema."""

    normalized = category.lower()
    if "gene" in normalized:
        return NodeType.GENE
    if "drug" in normalized:
        return NodeType.DRUG
    if "pathway" in normalized:
        return NodeType.PATHWAY
    if "phenotype" in normalized:
        return NodeType.PHENOTYPE
    return NodeType.DISEASE


def _map_edge_type(relation: str) -> EdgeType:
    """Map a PrimeKG relationship label onto the local schema."""

    normalized = relation.lower()
    if "treat" in normalized:
        return EdgeType.DRUG_TREATS_DISEASE
    if "target" in normalized:
        return EdgeType.DRUG_TARGETS_PROTEIN
    if "pathway" in normalized:
        return EdgeType.GENE_IN_PATHWAY
    return EdgeType.DISEASE_DIFFERENTIAL
