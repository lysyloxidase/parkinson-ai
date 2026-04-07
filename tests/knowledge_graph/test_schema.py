"""Schema tests."""

from __future__ import annotations

from parkinson_ai.knowledge_graph.schema import BiomarkerNode, EdgeType, GeneNode, NodeType


def test_enum_sizes() -> None:
    """Schema should expose the requested number of node and edge types."""

    assert len(NodeType) == 15
    assert len(EdgeType) == 22


def test_gene_and_biomarker_models() -> None:
    """Typed nodes should preserve required fields."""

    gene = GeneNode(id="gene:SNCA", name="SNCA", symbol="SNCA", chromosome="4")
    biomarker = BiomarkerNode(
        id="biomarker:saa",
        name="alpha-synuclein SAA (CSF)",
        category="molecular",
        biofluid="CSF",
        measurement_unit="binary",
    )
    assert gene.type == NodeType.GENE
    assert biomarker.category == "molecular"
