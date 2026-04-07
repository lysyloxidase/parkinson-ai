"""Typed schema definitions for the Parkinson's disease knowledge graph."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, Field


class NodeType(StrEnum):
    """Supported node types in the PD knowledge graph."""

    GENE = "Gene"
    PROTEIN = "Protein"
    VARIANT = "Variant"
    DISEASE = "Disease"
    DRUG = "Drug"
    CLINICAL_TRIAL = "ClinicalTrial"
    BIOMARKER = "Biomarker"
    PATHWAY = "Pathway"
    PHENOTYPE = "Phenotype"
    BRAIN_REGION = "BrainRegion"
    IMAGING_MODALITY = "ImagingModality"
    BIOFLUID = "Biofluid"
    GWAS_LOCUS = "GWASLocus"
    STAGING_SYSTEM = "StagingSystem"
    PUBLICATION = "Publication"


class EdgeType(StrEnum):
    """Supported edge types in the PD knowledge graph."""

    GENE_CAUSES_DISEASE = "gene_causes_disease"
    VARIANT_IN_GENE = "variant_in_gene"
    VARIANT_RISK_FOR = "variant_increases_risk_for"
    GENE_IN_PATHWAY = "gene_in_pathway"
    DRUG_TARGETS_PROTEIN = "drug_targets_protein"
    DRUG_TREATS_DISEASE = "drug_treats_disease"
    BIOMARKER_MEASURES = "biomarker_measured_in"
    BIOMARKER_INDICATES = "biomarker_indicates"
    DISEASE_HAS_PHENOTYPE = "disease_has_phenotype"
    PHENOTYPE_IN_STAGE = "phenotype_appears_in_stage"
    BRAIN_REGION_AFFECTED = "brain_region_affected_in"
    IMAGING_DETECTS = "imaging_detects_change_in"
    PROTEIN_AGGREGATES_IN = "protein_aggregates_in"
    GENE_INTERACTS = "gene_interacts_with"
    PATHWAY_CROSSTALK = "pathway_crosstalks_with"
    LOCUS_NEAR_GENE = "locus_near_gene"
    PUBLICATION_REPORTS = "publication_reports_biomarker"
    TRIAL_INVESTIGATES = "trial_investigates_drug"
    BIOMARKER_PREDICTS_STAGE = "biomarker_predicts_stage"
    DRUG_CONTRAINDICATED = "drug_contraindicated_in"
    PHENOTYPE_PRECEDES = "phenotype_precedes_phenotype"
    DISEASE_DIFFERENTIAL = "disease_differential_with"


class BaseNode(BaseModel):
    """Base graph node."""

    id: str
    type: NodeType
    name: str
    description: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)


class GeneNode(BaseNode):
    """Gene node."""

    type: Literal[NodeType.GENE] = NodeType.GENE
    ensembl_id: str | None = None
    symbol: str
    chromosome: str | None = None
    inheritance_pattern: str | None = None


class ProteinNode(BaseNode):
    """Protein node."""

    type: Literal[NodeType.PROTEIN] = NodeType.PROTEIN
    uniprot_id: str | None = None
    encoded_by: str | None = None
    subcellular_location: str | None = None


class VariantNode(BaseNode):
    """Variant node."""

    type: Literal[NodeType.VARIANT] = NodeType.VARIANT
    rsid: str | None = None
    hgvs: str | None = None
    allele_frequency: float | None = None
    odds_ratio: float | None = None
    penetrance: float | None = None
    pathogenicity: str | None = None


class DiseaseNode(BaseNode):
    """Disease node."""

    type: Literal[NodeType.DISEASE] = NodeType.DISEASE
    ontology_id: str | None = None
    disease_family: str | None = None


class DrugNode(BaseNode):
    """Drug node."""

    type: Literal[NodeType.DRUG] = NodeType.DRUG
    drugbank_id: str | None = None
    approval_status: str | None = None
    mechanism: str | None = None


class ClinicalTrialNode(BaseNode):
    """Clinical trial node."""

    type: Literal[NodeType.CLINICAL_TRIAL] = NodeType.CLINICAL_TRIAL
    trial_id: str | None = None
    phase: str | None = None
    status: str | None = None


class BiomarkerNode(BaseNode):
    """Biomarker node."""

    type: Literal[NodeType.BIOMARKER] = NodeType.BIOMARKER
    category: str
    biofluid: str | None = None
    measurement_unit: str | None = None
    reference_range: str | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    auc: float | None = None
    source_paper: str | None = None
    clinical_utility: str | None = None


class PathwayNode(BaseNode):
    """Pathway node."""

    type: Literal[NodeType.PATHWAY] = NodeType.PATHWAY
    pathway_id: str | None = None
    database: str | None = None


class PhenotypeNode(BaseNode):
    """Phenotype node."""

    type: Literal[NodeType.PHENOTYPE] = NodeType.PHENOTYPE
    hpo_id: str | None = None
    onset_pattern: str | None = None


class BrainRegionNode(BaseNode):
    """Brain region node."""

    type: Literal[NodeType.BRAIN_REGION] = NodeType.BRAIN_REGION
    atlas: str | None = None


class ImagingModalityNode(BaseNode):
    """Imaging modality node."""

    type: Literal[NodeType.IMAGING_MODALITY] = NodeType.IMAGING_MODALITY
    modality_family: str | None = None


class BiofluidNode(BaseNode):
    """Biofluid node."""

    type: Literal[NodeType.BIOFLUID] = NodeType.BIOFLUID
    matrix_type: str | None = None


class GWASLocusNode(BaseNode):
    """GWAS locus node."""

    type: Literal[NodeType.GWAS_LOCUS] = NodeType.GWAS_LOCUS
    rsid: str | None = None
    chromosome: str | None = None
    position: int | None = None
    nearest_gene: str | None = None
    odds_ratio: float | None = None
    p_value: float | None = None
    consortium: str | None = None


class StagingNode(BaseNode):
    """Staging system node."""

    type: Literal[NodeType.STAGING_SYSTEM] = NodeType.STAGING_SYSTEM
    system: str
    stage: str
    criteria: list[str] = Field(default_factory=list)
    description: str


class PublicationNode(BaseNode):
    """Publication node."""

    type: Literal[NodeType.PUBLICATION] = NodeType.PUBLICATION
    pmid: str | None = None
    title: str | None = None
    year: int | None = None
    journal: str | None = None


KnowledgeGraphNode: TypeAlias = Annotated[
    GeneNode | ProteinNode | VariantNode | DiseaseNode | DrugNode | ClinicalTrialNode | BiomarkerNode | PathwayNode | PhenotypeNode | BrainRegionNode | ImagingModalityNode | BiofluidNode | GWASLocusNode | StagingNode | PublicationNode,
    Field(discriminator="type"),
]


class GraphEdge(BaseModel):
    """Typed directed multi-edge."""

    source: str
    target: str
    type: EdgeType
    weight: float = 1.0
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


NODE_TYPE_TO_MODEL: dict[NodeType, type[BaseNode]] = {
    NodeType.GENE: GeneNode,
    NodeType.PROTEIN: ProteinNode,
    NodeType.VARIANT: VariantNode,
    NodeType.DISEASE: DiseaseNode,
    NodeType.DRUG: DrugNode,
    NodeType.CLINICAL_TRIAL: ClinicalTrialNode,
    NodeType.BIOMARKER: BiomarkerNode,
    NodeType.PATHWAY: PathwayNode,
    NodeType.PHENOTYPE: PhenotypeNode,
    NodeType.BRAIN_REGION: BrainRegionNode,
    NodeType.IMAGING_MODALITY: ImagingModalityNode,
    NodeType.BIOFLUID: BiofluidNode,
    NodeType.GWAS_LOCUS: GWASLocusNode,
    NodeType.STAGING_SYSTEM: StagingNode,
    NodeType.PUBLICATION: PublicationNode,
}
