"""Typed builder and persistence utilities for the PD knowledge graph."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import networkx as nx

from parkinson_ai.config import get_settings
from parkinson_ai.knowledge_graph.schema import BaseNode, EdgeType, GraphEdge, NodeType


class PDKnowledgeGraph:
    """High-level wrapper over a typed multi-directed graph."""

    def __init__(self, graph: nx.MultiDiGraph | None = None) -> None:
        self.graph = graph or nx.MultiDiGraph()

    def add_node(self, node: BaseNode) -> None:
        """Add a typed node to the graph."""

        self.graph.add_node(node.id, **node.model_dump(mode="json"))

    def add_edge(self, edge: GraphEdge) -> None:
        """Add a typed edge to the graph."""

        payload = edge.model_dump(mode="json")
        edge_type = payload.pop("type")
        self.graph.add_edge(edge.source, edge.target, key=edge_type, type=edge_type, **payload)

    def connect(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        *,
        weight: float = 1.0,
        evidence: list[str] | None = None,
        **metadata: str | int | float | bool | None,
    ) -> None:
        """Convenience wrapper for building edges."""

        self.add_edge(
            GraphEdge(
                source=source,
                target=target,
                type=edge_type,
                weight=weight,
                evidence=evidence or [],
                metadata=metadata,
            )
        )

    def bulk_add_nodes(self, nodes: Sequence[BaseNode]) -> None:
        """Insert multiple nodes."""

        for node in nodes:
            self.add_node(node)

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Return a node payload by id."""

        if node_id not in self.graph:
            return None
        return dict(self.graph.nodes[node_id])

    def node_ids_by_type(self, node_type: NodeType) -> list[str]:
        """Return all node ids for a specific node type."""

        return [str(node_id) for node_id, attributes in self.graph.nodes(data=True) if attributes.get("type") == node_type.value]

    def neighbors_by_type(self, node_id: str, node_type: NodeType) -> list[str]:
        """Return neighbors filtered by node type."""

        result: list[str] = []
        for neighbor in self.graph.neighbors(node_id):
            if self.graph.nodes[neighbor].get("type") == node_type.value:
                result.append(str(neighbor))
        return result

    def subgraph_from_nodes(self, node_ids: list[str]) -> PDKnowledgeGraph:
        """Return a view of a subset of nodes."""

        return PDKnowledgeGraph(self.graph.subgraph(node_ids).copy())

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the graph to disk."""

        target = Path(path or get_settings().GRAPH_PERSIST_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self.graph, handle)
        return target

    @classmethod
    def load(cls, path: str | Path) -> PDKnowledgeGraph:
        """Load a persisted graph."""

        with Path(path).open("rb") as handle:
            graph = pickle.load(handle)
        if not isinstance(graph, nx.MultiDiGraph):
            raise TypeError("Persisted object is not a MultiDiGraph")
        return cls(graph)

    def statistics(self) -> dict[str, Any]:
        """Return simple graph statistics."""

        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "node_type_counts": {node_type.value: len(self.node_ids_by_type(node_type)) for node_type in NodeType},
        }
