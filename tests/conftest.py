"""Shared fixtures for the parkinson-ai test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from parkinson_ai.api.app import app
from parkinson_ai.core.llm_client import LLMResponse
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.knowledge_graph.schema import (
    BaseNode,
    BiomarkerNode,
    BrainRegionNode,
    ClinicalTrialNode,
    DiseaseNode,
    DrugNode,
    EdgeType,
    GeneNode,
    GWASLocusNode,
    ImagingModalityNode,
    NodeType,
    PathwayNode,
    PhenotypeNode,
    ProteinNode,
    PublicationNode,
    StagingNode,
    VariantNode,
)
from parkinson_ai.knowledge_graph.staging import PatientData


@pytest.fixture()
def sample_pd_graph() -> PDKnowledgeGraph:
    """Return a graph with 50+ nodes and 100+ edges spanning all schema types."""

    graph = PDKnowledgeGraph()
    diseases = [
        DiseaseNode(id="disease:pd", name="Parkinson disease", ontology_id="MONDO:0005180"),
        DiseaseNode(id="disease:msa", name="Multiple system atrophy"),
        DiseaseNode(id="disease:psp", name="Progressive supranuclear palsy"),
        DiseaseNode(id="disease:dlb", name="Dementia with Lewy bodies"),
        DiseaseNode(id="disease:cbd", name="Corticobasal degeneration"),
    ]
    genes = [
        GeneNode(id="gene:SNCA", name="SNCA", symbol="SNCA", chromosome="4", inheritance_pattern="AD"),
        GeneNode(id="gene:LRRK2", name="LRRK2", symbol="LRRK2", chromosome="12", inheritance_pattern="AD"),
        GeneNode(id="gene:GBA1", name="GBA1", symbol="GBA1", chromosome="1", inheritance_pattern="risk"),
        GeneNode(id="gene:PINK1", name="PINK1", symbol="PINK1", chromosome="1", inheritance_pattern="AR"),
        GeneNode(id="gene:PARK2", name="PARK2", symbol="PARK2", chromosome="6", inheritance_pattern="AR"),
        GeneNode(id="gene:MAPT", name="MAPT", symbol="MAPT", chromosome="17", inheritance_pattern="risk"),
        GeneNode(id="gene:TMEM175", name="TMEM175", symbol="TMEM175", chromosome="4", inheritance_pattern="risk"),
        GeneNode(id="gene:VPS35", name="VPS35", symbol="VPS35", chromosome="16", inheritance_pattern="AD"),
    ]
    proteins = [
        ProteinNode(id="protein:alpha_syn", name="alpha-synuclein", uniprot_id="P37840", encoded_by="SNCA"),
        ProteinNode(id="protein:lrrk2", name="LRRK2 kinase", uniprot_id="Q5S007", encoded_by="LRRK2"),
        ProteinNode(id="protein:gcase", name="glucocerebrosidase", uniprot_id="P04062", encoded_by="GBA1"),
        ProteinNode(id="protein:parkin", name="parkin", encoded_by="PARK2"),
        ProteinNode(id="protein:tau", name="tau", encoded_by="MAPT"),
    ]
    variants = [
        VariantNode(id="variant:lrrk2_g2019s", name="LRRK2 G2019S", rsid="rs34637584", odds_ratio=3.0),
        VariantNode(id="variant:gba1_n370s", name="GBA1 N370S", odds_ratio=3.0),
        VariantNode(id="variant:gba1_l444p", name="GBA1 L444P", odds_ratio=10.0),
        VariantNode(id="variant:snca_dup", name="SNCA duplication"),
        VariantNode(id="variant:snca_trip", name="SNCA triplication"),
        VariantNode(id="variant:vps35_d620n", name="VPS35 D620N"),
    ]
    drugs = [
        DrugNode(id="drug:levodopa", name="Levodopa", mechanism="dopamine replacement"),
        DrugNode(id="drug:prasinezumab", name="Prasinezumab", mechanism="anti-alpha-syn antibody"),
        DrugNode(id="drug:ambroxol", name="Ambroxol", mechanism="GCase enhancer"),
    ]
    trials = [
        ClinicalTrialNode(id="trial:pasadena", name="PASADENA", trial_id="NCT03100149", phase="2"),
        ClinicalTrialNode(id="trial:light", name="LIGHTHOUSE", trial_id="NCT05424369", phase="2"),
        ClinicalTrialNode(id="trial:ambroxol", name="Ambroxol Trial", trial_id="NCT02914366", phase="2"),
    ]
    pathways = [
        PathwayNode(id="pathway:mitophagy", name="Mitophagy", pathway_id="R-HSA-5205647", database="Reactome"),
        PathwayNode(id="pathway:lysosome", name="Lysosomal function", pathway_id="R-HSA-123456", database="Reactome"),
        PathwayNode(id="pathway:synaptic", name="Synaptic vesicle cycle", pathway_id="hsa04721", database="KEGG"),
        PathwayNode(id="pathway:inflammation", name="Neuroinflammation", database="Reactome"),
    ]
    phenotypes = [
        PhenotypeNode(id="phenotype:tremor", name="Resting tremor", hpo_id="HP:0002322"),
        PhenotypeNode(id="phenotype:bradykinesia", name="Bradykinesia", hpo_id="HP:0002063"),
        PhenotypeNode(id="phenotype:rigidity", name="Rigidity", hpo_id="HP:0002064"),
        PhenotypeNode(id="phenotype:rbd", name="REM sleep behavior disorder", hpo_id="HP:0004409"),
        PhenotypeNode(id="phenotype:hyposmia", name="Hyposmia", hpo_id="HP:0004408"),
        PhenotypeNode(id="phenotype:constipation", name="Constipation", hpo_id="HP:0002014"),
    ]
    brain_regions = [
        BrainRegionNode(id="region:sn", name="Substantia nigra"),
        BrainRegionNode(id="region:putamen", name="Putamen"),
        BrainRegionNode(id="region:caudate", name="Caudate"),
        BrainRegionNode(id="region:lc", name="Locus coeruleus"),
        BrainRegionNode(id="region:cortex", name="Cortex"),
    ]
    modalities = [
        ImagingModalityNode(id="imaging:datscan", name="DaTSCAN"),
        ImagingModalityNode(id="imaging:nm_mri", name="Neuromelanin MRI"),
        ImagingModalityNode(id="imaging:qsm", name="QSM"),
        ImagingModalityNode(id="imaging:fdg_pet", name="FDG-PET"),
    ]
    biofluids = [
        BaseNode(id="biofluid:csf", type=NodeType.BIOFLUID, name="CSF"),
        BaseNode(id="biofluid:blood", type=NodeType.BIOFLUID, name="blood"),
        BaseNode(id="biofluid:skin", type=NodeType.BIOFLUID, name="skin"),
        BaseNode(id="biofluid:stool", type=NodeType.BIOFLUID, name="stool"),
    ]
    loci = [
        GWASLocusNode(id="locus:rs356182", name="rs356182", rsid="rs356182", chromosome="4", position=90626111, nearest_gene="SNCA", odds_ratio=1.34, p_value=1e-12, consortium="GP2"),
        GWASLocusNode(id="locus:rs76904798", name="rs76904798", rsid="rs76904798", chromosome="12", position=40252984, nearest_gene="LRRK2", odds_ratio=1.25, p_value=1e-10, consortium="GP2"),
        GWASLocusNode(id="locus:rs2230288", name="rs2230288", rsid="rs2230288", chromosome="1", position=155205634, nearest_gene="GBA1", odds_ratio=2.2, p_value=1e-14, consortium="GP2"),
        GWASLocusNode(id="locus:rs34311866", name="rs34311866", rsid="rs34311866", chromosome="16", position=30947987, nearest_gene="TMEM175", odds_ratio=1.14, p_value=1e-9, consortium="GP2"),
    ]
    stages = [
        StagingNode(id="stage:2b", name="Stage 2B", system="NSD_ISS", stage="2B", criteria=["Subtle signs", "Neurodegeneration"], description="Prodromal with degeneration"),
        StagingNode(id="stage:3", name="Stage 3", system="NSD_ISS", stage="3", criteria=["Motor signs", "Functional impact"], description="Clinically established PD"),
        StagingNode(id="stage:s1n1g0", name="S1N1G0", system="SynNeurGe", stage="S1N1G0", criteria=["S1", "N1", "G0"], description="Typical sporadic PD"),
    ]
    publications = [
        PublicationNode(id="pub:simuni2024", name="Simuni 2024", pmid="38181717", year=2024, journal="Lancet Neurology"),
        PublicationNode(id="pub:hoglinger2024", name="Hoglinger 2024", pmid="38181718", year=2024, journal="Lancet Neurology"),
        PublicationNode(id="pub:bartl2024", name="Bartl 2024", pmid="39990001", year=2024, journal="Nature Aging"),
        PublicationNode(id="pub:kluge2024", name="Kluge 2024", pmid="39990002", year=2024, journal="Brain"),
    ]
    biomarkers = [
        BiomarkerNode(id="biomarker:saa_csf", name="alpha-synuclein SAA (CSF)", category="molecular", biofluid="CSF", measurement_unit="binary", source_paper="Meta-analysis 2025", clinical_utility="Biological diagnosis"),
        BiomarkerNode(id="biomarker:nfl", name="NfL (blood/serum)", category="molecular", biofluid="blood", measurement_unit="pg/mL", source_paper="Serum NfL studies", clinical_utility="Progression marker"),
        BiomarkerNode(id="biomarker:datscan", name="DaTSCAN SBR", category="imaging", measurement_unit="SBR", source_paper="DaTSCAN meta-analyses", clinical_utility="Nigrostriatal degeneration"),
        BiomarkerNode(id="biomarker:nm_mri", name="Neuromelanin MRI", category="imaging", measurement_unit="contrast ratio", source_paper="NM-MRI studies", clinical_utility="Nigral degeneration"),
        BiomarkerNode(id="biomarker:voice", name="Voice biomarker panel", category="digital", measurement_unit="model score", source_paper="Voice studies", clinical_utility="Remote screening"),
        BiomarkerNode(id="biomarker:gait", name="Gait variability index", category="digital", measurement_unit="score", source_paper="Gait studies", clinical_utility="Remote monitoring"),
        BiomarkerNode(id="biomarker:updrs3", name="MDS-UPDRS Part III", category="clinical", measurement_unit="points", source_paper="PPMI", clinical_utility="Motor severity"),
        BiomarkerNode(id="biomarker:moca", name="MoCA", category="clinical", measurement_unit="points", source_paper="MoCA validation", clinical_utility="Cognition"),
        BiomarkerNode(id="biomarker:rbd", name="REM sleep behavior disorder", category="nonmotor", measurement_unit="present/absent", source_paper="iRBD meta-analysis", clinical_utility="Prodromal risk"),
        BiomarkerNode(id="biomarker:hyposmia", name="Hyposmia (UPSIT)", category="nonmotor", measurement_unit="UPSIT score", source_paper="UPSIT studies", clinical_utility="Prodromal risk"),
        BiomarkerNode(id="biomarker:microbiome", name="Gut microbiome signature", category="emerging", biofluid="stool", measurement_unit="classifier score", source_paper="Microbiome studies", clinical_utility="Emerging stratification"),
        BiomarkerNode(id="biomarker:oct", name="Retinal OCT RNFL thinning", category="emerging", measurement_unit="micrometers", source_paper="OCT studies", clinical_utility="Retinal biomarker"),
    ]

    for group in [diseases, genes, proteins, variants, drugs, trials, pathways, phenotypes, brain_regions, modalities, biofluids, loci, stages, publications, biomarkers]:
        graph.bulk_add_nodes(group)  # type: ignore[arg-type]

    for variant, gene in zip(variants, genes[: len(variants)], strict=False):
        graph.connect(variant.id, gene.id, EdgeType.VARIANT_IN_GENE)
        graph.connect(variant.id, "disease:pd", EdgeType.VARIANT_RISK_FOR)
    for gene in genes:
        graph.connect(gene.id, "disease:pd", EdgeType.GENE_CAUSES_DISEASE)
    for gene, pathway in zip(genes, pathways * 2, strict=False):
        graph.connect(gene.id, pathway.id, EdgeType.GENE_IN_PATHWAY)
    for left in genes:
        for right in genes:
            if left.id != right.id and (left.id, right.id) < (right.id, left.id):
                graph.connect(left.id, right.id, EdgeType.GENE_INTERACTS)
    for biomarker in biomarkers:
        graph.connect(biomarker.id, "disease:pd", EdgeType.BIOMARKER_INDICATES)
        if biomarker.biofluid:
            graph.connect(biomarker.id, f"biofluid:{biomarker.biofluid.lower()}", EdgeType.BIOMARKER_MEASURES)
    graph.connect("biomarker:datscan", "stage:2b", EdgeType.BIOMARKER_PREDICTS_STAGE)
    graph.connect("biomarker:updrs3", "stage:3", EdgeType.BIOMARKER_PREDICTS_STAGE)
    for phenotype in phenotypes:
        graph.connect("disease:pd", phenotype.id, EdgeType.DISEASE_HAS_PHENOTYPE)
    graph.connect("phenotype:rbd", "stage:2b", EdgeType.PHENOTYPE_IN_STAGE)
    graph.connect("phenotype:hyposmia", "stage:2b", EdgeType.PHENOTYPE_IN_STAGE)
    graph.connect("phenotype:hyposmia", "phenotype:bradykinesia", EdgeType.PHENOTYPE_PRECEDES)
    for region in brain_regions:
        graph.connect(region.id, "disease:pd", EdgeType.BRAIN_REGION_AFFECTED)
    graph.connect("imaging:datscan", "region:putamen", EdgeType.IMAGING_DETECTS)
    graph.connect("imaging:nm_mri", "region:sn", EdgeType.IMAGING_DETECTS)
    graph.connect("protein:alpha_syn", "region:sn", EdgeType.PROTEIN_AGGREGATES_IN)
    graph.connect("protein:alpha_syn", "region:cortex", EdgeType.PROTEIN_AGGREGATES_IN)
    graph.connect("drug:prasinezumab", "protein:alpha_syn", EdgeType.DRUG_TARGETS_PROTEIN)
    graph.connect("drug:levodopa", "disease:pd", EdgeType.DRUG_TREATS_DISEASE)
    graph.connect("drug:ambroxol", "disease:pd", EdgeType.DRUG_TREATS_DISEASE)
    graph.connect("trial:pasadena", "drug:prasinezumab", EdgeType.TRIAL_INVESTIGATES)
    graph.connect("trial:light", "drug:levodopa", EdgeType.TRIAL_INVESTIGATES)
    graph.connect("trial:ambroxol", "drug:ambroxol", EdgeType.TRIAL_INVESTIGATES)
    for publication in publications:
        for biomarker in biomarkers[:4]:
            graph.connect(publication.id, biomarker.id, EdgeType.PUBLICATION_REPORTS)
    for locus, gene in zip(loci, genes[:4], strict=False):
        graph.connect(locus.id, gene.id, EdgeType.LOCUS_NEAR_GENE)
    graph.connect("pathway:mitophagy", "pathway:lysosome", EdgeType.PATHWAY_CROSSTALK)
    graph.connect("pathway:lysosome", "pathway:inflammation", EdgeType.PATHWAY_CROSSTALK)
    graph.connect("drug:levodopa", "disease:psp", EdgeType.DRUG_CONTRAINDICATED)
    graph.connect("disease:pd", "disease:msa", EdgeType.DISEASE_DIFFERENTIAL)
    graph.connect("disease:pd", "disease:psp", EdgeType.DISEASE_DIFFERENTIAL)
    graph.connect("disease:pd", "disease:dlb", EdgeType.DISEASE_DIFFERENTIAL)
    graph.connect("disease:pd", "disease:cbd", EdgeType.DISEASE_DIFFERENTIAL)

    assert graph.graph.number_of_nodes() >= 50
    assert graph.graph.number_of_edges() >= 100
    return graph


@pytest.fixture()
def mock_patient_data() -> PatientData:
    """Return a representative PD patient payload."""

    return PatientData(
        saa_result=True,
        datscan_abnormal=True,
        nfl_pg_ml=18.2,
        motor_signs=True,
        functional_impairment="mild",
        updrs_total=34,
        updrs_part3=18,
        hoehn_yahr=2.0,
        rbd_present=True,
        hyposmia=True,
        age=67,
    )


@pytest.fixture()
def mock_ollama_transport() -> httpx.MockTransport:
    """Return a mock Ollama transport."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/api/tags"):
            return httpx.Response(200, json={"models": [{"name": "llama3.2:3b"}]})
        return httpx.Response(200, json={"model": "llama3.2:3b", "response": "ok", "done": True})

    return httpx.MockTransport(handler)


@pytest.fixture()
def api_client() -> TestClient:
    """Return a FastAPI test client."""

    return TestClient(app)


@pytest.fixture()
def tmp_graph_path(tmp_path: Path) -> Path:
    """Return a temporary graph path."""

    return tmp_path / "graph.gpickle"


class FakeLLMClient:
    """Deterministic async fake LLM for agent testing."""

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del prompt, options
        marker = "generic"
        if isinstance(system, str):
            lowered = system.lower()
            if "biomarker interpreter" in lowered:
                marker = "biomarker"
            elif "genetic counselor" in lowered:
                marker = "genetics"
            elif "imaging analyst" in lowered:
                marker = "imaging"
            elif "staging specialist" in lowered:
                marker = "staging"
            elif "risk assessor" in lowered:
                marker = "risk"
            elif "therapeutics analyst" in lowered:
                marker = "drug"
            elif "knowledge-graph explorer" in lowered:
                marker = "kg"
            elif "research assistant" in lowered:
                marker = "literature"
        return LLMResponse(model=model or "fake-llm", response=f"{marker} report", raw={})


@pytest.fixture()
def fake_llm_client() -> FakeLLMClient:
    """Return a deterministic fake LLM client."""

    return FakeLLMClient()
