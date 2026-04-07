"""CLI for building a seed PD knowledge graph."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from parkinson_ai.knowledge_graph.biomarker_nodes import iter_biomarker_nodes
    from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
    from parkinson_ai.knowledge_graph.schema import BaseNode, EdgeType, NodeType, StagingNode
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from parkinson_ai.knowledge_graph.biomarker_nodes import iter_biomarker_nodes
    from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
    from parkinson_ai.knowledge_graph.schema import BaseNode, EdgeType, NodeType, StagingNode


def build_graph(reference_dir: Path) -> PDKnowledgeGraph:
    """Build a seed graph from bundled reference data."""

    graph = PDKnowledgeGraph()
    graph.bulk_add_nodes(
        [
            BaseNode(id="disease:pd", type=NodeType.DISEASE, name="Parkinson disease"),
            BaseNode(id="disease:msa", type=NodeType.DISEASE, name="Multiple system atrophy"),
            BaseNode(id="disease:psp", type=NodeType.DISEASE, name="Progressive supranuclear palsy"),
            BaseNode(id="biofluid:csf", type=NodeType.BIOFLUID, name="CSF"),
            BaseNode(id="biofluid:blood", type=NodeType.BIOFLUID, name="blood"),
        ]
    )
    graph.bulk_add_nodes(list(iter_biomarker_nodes()))
    nsd_criteria = json.loads((reference_dir / "nsd_iss_criteria.json").read_text(encoding="utf-8"))
    for stage, stage_data in nsd_criteria["stages"].items():
        graph.add_node(
            StagingNode(
                id=f"stage:nsdiss:{stage}",
                name=f"NSD-ISS stage {stage}",
                system="NSD_ISS",
                stage=stage,
                criteria=stage_data["criteria"],
                description=stage_data["description"],
            )
        )
    for biomarker in iter_biomarker_nodes():
        if biomarker.biofluid:
            target_id = f"biofluid:{biomarker.biofluid.lower()}"
            if target_id not in graph.graph:
                graph.add_node(BaseNode(id=target_id, type=NodeType.BIOFLUID, name=biomarker.biofluid))
            graph.connect(biomarker.id, target_id, EdgeType.BIOMARKER_MEASURES)
        if biomarker.category in {"molecular", "clinical", "nonmotor"}:
            graph.connect(biomarker.id, "disease:pd", EdgeType.BIOMARKER_INDICATES)
    return graph


def main() -> None:
    """Run the graph-building CLI."""

    parser = argparse.ArgumentParser(description="Build the parkinson-ai seed knowledge graph.")
    parser.add_argument(
        "--output",
        default="data/pd_knowledge_graph.gpickle",
        help="Path where the graph pickle should be written.",
    )
    parser.add_argument(
        "--reference-dir",
        default="data/reference",
        help="Directory containing bundled reference JSON files.",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    graph = build_graph(project_root / args.reference_dir)
    saved = graph.save(project_root / args.output)
    print(f"Saved graph with {graph.graph.number_of_nodes()} nodes and {graph.graph.number_of_edges()} edges to {saved}")


if __name__ == "__main__":
    main()
