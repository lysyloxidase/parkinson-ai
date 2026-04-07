"""Graph analytics for the PD knowledge graph."""

from __future__ import annotations

from typing import Any

import networkx as nx

from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.knowledge_graph.schema import NodeType


class GraphStatistics:
    """Compute graph-level and biomarker-centric statistics."""

    def __init__(self, graph: PDKnowledgeGraph) -> None:
        self.graph = graph.graph

    def hub_biomarkers(self, *, top_n: int = 10) -> list[tuple[str, float]]:
        """Return top biomarker hubs by degree centrality."""

        centrality = nx.degree_centrality(self.graph)
        biomarkers = [(str(self.graph.nodes[node_id].get("name", node_id)), score) for node_id, score in centrality.items() if self.graph.nodes[node_id].get("type") == NodeType.BIOMARKER.value]
        biomarkers.sort(key=lambda item: item[1], reverse=True)
        return biomarkers[:top_n]

    def connectivity_analysis(self) -> dict[str, Any]:
        """Return core connectivity metrics."""

        undirected = self.graph.to_undirected()
        components = list(nx.connected_components(undirected)) if self.graph.number_of_nodes() else []
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0.0,
            "connected_components": len(components),
            "largest_component_size": max((len(component) for component in components), default=0),
        }

    def shortest_path(self, source: str, target: str) -> list[str]:
        """Return the shortest path between two nodes when it exists."""

        try:
            return [str(node) for node in nx.shortest_path(self.graph.to_undirected(), source, target)]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
