"""FastAPI application and SPA-facing endpoints for ParkinsonAI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, WebSocket
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from parkinson_ai.agents.biomarker_interpreter import BiomarkerInterpreterAgent
from parkinson_ai.agents.genetic_counselor import GeneticCounselorAgent
from parkinson_ai.agents.risk_assessor import RiskAssessorAgent
from parkinson_ai.agents.router import RouterAgent
from parkinson_ai.agents.staging_agent import StagingAgent
from parkinson_ai.api.websocket import stream_agent_chat
from parkinson_ai.knowledge_graph.biomarker_nodes import BIOMARKER_CATEGORY_COUNTS, BIOMARKER_LIBRARY, iter_biomarker_nodes
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.knowledge_graph.schema import (
    BaseNode,
    DiseaseNode,
    DrugNode,
    EdgeType,
    GeneNode,
    NodeType,
    PathwayNode,
    PhenotypeNode,
    ProteinNode,
    StagingNode,
)
from parkinson_ai.knowledge_graph.staging import NSDISSStaging, PatientData, SynNeurGeStaging
from parkinson_ai.orchestration.workflow import PDMultiAgentWorkflow

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_INDEX_PATH = PROJECT_ROOT / "web" / "index.html"


class PatientAssessmentRequest(BaseModel):
    """Full patient-assessment payload used by the SPA."""

    saa_result: bool | None = None
    saa_biofluid: str | None = None
    nfl_pg_ml: float | None = None
    dj1_ng_ml: float | None = None
    gcase_activity_pct: float | None = None
    genetic_variants: list[str] = Field(default_factory=list)
    prs_score: float | None = None
    datscan_abnormal: bool | None = None
    datscan_sbr: float | None = None
    nm_mri_abnormal: bool | None = None
    qsm_sn_iron: float | None = None
    voice_upload_name: str | None = None
    gait_upload_name: str | None = None
    updrs_part_i: float | None = None
    updrs_part_ii: float | None = None
    updrs_part3: float | None = None
    updrs_part_iv: float | None = None
    updrs_total: float | None = None
    hoehn_yahr: float | None = None
    moca_score: float | None = None
    mmse_score: float | None = None
    motor_subtype: str | None = None
    age: int | None = None
    sex: str | None = None
    disease_duration: float | None = None
    rbd_present: bool | None = None
    hyposmia: bool | None = None
    upsit_score: float | None = None
    constipation: bool | None = None
    depression_score: float | None = None
    anxiety_score: float | None = None
    orthostatic_hypotension: bool | None = None
    family_history: bool | None = None
    age_at_rbd_onset: float | None = None
    years_since_rbd: float | None = None
    caffeine_intake: float | None = None
    motor_signs: bool | None = None
    functional_impairment: str | None = None

    def to_patient_data(self) -> PatientData:
        """Convert the request into the staging model."""

        return PatientData(
            saa_result=self.saa_result,
            saa_biofluid=self.saa_biofluid,
            datscan_abnormal=self.datscan_abnormal,
            datscan_sbr=self.datscan_sbr,
            nfl_pg_ml=self.nfl_pg_ml,
            nm_mri_abnormal=self.nm_mri_abnormal,
            motor_signs=self.motor_signs,
            functional_impairment=self.functional_impairment,
            genetic_variants=self.genetic_variants,
            updrs_total=self.updrs_total,
            updrs_part3=self.updrs_part3,
            hoehn_yahr=self.hoehn_yahr,
            moca_score=self.moca_score,
            rbd_present=self.rbd_present,
            hyposmia=self.hyposmia,
            age=self.age,
        )

    def to_feature_payload(self) -> dict[str, Any]:
        """Return a flattened multimodal payload for downstream agents."""

        payload = self.model_dump()
        payload["prodromal_risk_score"] = _heuristic_prodromal_score(payload)
        return payload


app = FastAPI(title="ParkinsonAI", version="0.2.0")


def _build_api_graph() -> PDKnowledgeGraph:
    """Build a graph with enough breadth for the UI, agents, and demo endpoints."""

    graph = PDKnowledgeGraph()
    core_nodes: list[BaseNode] = [
        DiseaseNode(id="disease:pd", name="Parkinson disease", ontology_id="MONDO:0005180"),
        DiseaseNode(id="disease:msa", name="Multiple system atrophy"),
        DiseaseNode(id="disease:psp", name="Progressive supranuclear palsy"),
        DiseaseNode(id="disease:et", name="Essential tremor"),
        GeneNode(id="gene:SNCA", name="SNCA", symbol="SNCA", chromosome="4", inheritance_pattern="AD"),
        GeneNode(id="gene:LRRK2", name="LRRK2", symbol="LRRK2", chromosome="12", inheritance_pattern="AD"),
        GeneNode(id="gene:GBA1", name="GBA1", symbol="GBA1", chromosome="1", inheritance_pattern="risk"),
        GeneNode(id="gene:PINK1", name="PINK1", symbol="PINK1", chromosome="1", inheritance_pattern="AR"),
        GeneNode(id="gene:PARK2", name="PARK2", symbol="PARK2", chromosome="6", inheritance_pattern="AR"),
        ProteinNode(id="protein:alpha_syn", name="alpha-synuclein", encoded_by="SNCA"),
        ProteinNode(id="protein:lrrk2", name="LRRK2 kinase", encoded_by="LRRK2"),
        ProteinNode(id="protein:gcase", name="glucocerebrosidase", encoded_by="GBA1"),
        DrugNode(id="drug:levodopa", name="Levodopa", mechanism="dopamine replacement"),
        DrugNode(id="drug:prasinezumab", name="Prasinezumab", mechanism="anti-alpha-synuclein antibody"),
        DrugNode(id="drug:ambroxol", name="Ambroxol", mechanism="GCase enhancement"),
        PathwayNode(id="pathway:mitophagy", name="Mitophagy", database="Reactome"),
        PathwayNode(id="pathway:lysosome", name="Lysosomal function", database="Reactome"),
        PathwayNode(id="pathway:synaptic", name="Synaptic vesicle cycle", database="KEGG"),
        PhenotypeNode(id="phenotype:tremor", name="Resting tremor"),
        PhenotypeNode(id="phenotype:bradykinesia", name="Bradykinesia"),
        PhenotypeNode(id="phenotype:rbd", name="REM sleep behavior disorder"),
        PhenotypeNode(id="phenotype:hyposmia", name="Hyposmia"),
        PhenotypeNode(id="phenotype:constipation", name="Constipation"),
        BaseNode(id="biofluid:csf", type=NodeType.BIOFLUID, name="CSF"),
        BaseNode(id="biofluid:blood", type=NodeType.BIOFLUID, name="blood"),
        BaseNode(id="biofluid:skin", type=NodeType.BIOFLUID, name="skin"),
        BaseNode(id="biofluid:plasma", type=NodeType.BIOFLUID, name="plasma"),
        StagingNode(
            id="stage:nsd:1a",
            name="NSD-ISS 1A",
            system="NSD_ISS",
            stage="1A",
            criteria=["SAA positivity only"],
            description="Biological synucleinopathy without degeneration or symptoms.",
        ),
        StagingNode(
            id="stage:nsd:1b",
            name="NSD-ISS 1B",
            system="NSD_ISS",
            stage="1B",
            criteria=["SAA positivity", "Neurodegeneration biomarker"],
            description="Biological synucleinopathy with biomarker evidence of degeneration.",
        ),
        StagingNode(
            id="stage:nsd:2b",
            name="NSD-ISS 2B",
            system="NSD_ISS",
            stage="2B",
            criteria=["Subtle signs", "Neurodegeneration"],
            description="Prodromal synucleinopathy with biomarker degeneration.",
        ),
        StagingNode(
            id="stage:nsd:3",
            name="NSD-ISS 3",
            system="NSD_ISS",
            stage="3",
            criteria=["Motor signs", "Early functional impact"],
            description="Clinically established Parkinson disease.",
        ),
    ]
    graph.bulk_add_nodes(core_nodes)
    graph.bulk_add_nodes(list(iter_biomarker_nodes()))
    graph.connect("gene:SNCA", "disease:pd", EdgeType.GENE_CAUSES_DISEASE)
    graph.connect("gene:LRRK2", "disease:pd", EdgeType.GENE_CAUSES_DISEASE)
    graph.connect("gene:GBA1", "disease:pd", EdgeType.GENE_CAUSES_DISEASE)
    graph.connect("gene:PINK1", "disease:pd", EdgeType.GENE_CAUSES_DISEASE)
    graph.connect("gene:PARK2", "disease:pd", EdgeType.GENE_CAUSES_DISEASE)
    graph.connect("gene:PINK1", "pathway:mitophagy", EdgeType.GENE_IN_PATHWAY)
    graph.connect("gene:PARK2", "pathway:mitophagy", EdgeType.GENE_IN_PATHWAY)
    graph.connect("gene:GBA1", "pathway:lysosome", EdgeType.GENE_IN_PATHWAY)
    graph.connect("gene:SNCA", "pathway:synaptic", EdgeType.GENE_IN_PATHWAY)
    graph.connect("gene:LRRK2", "pathway:lysosome", EdgeType.GENE_IN_PATHWAY)
    graph.connect("gene:SNCA", "gene:LRRK2", EdgeType.GENE_INTERACTS)
    graph.connect("gene:SNCA", "gene:GBA1", EdgeType.GENE_INTERACTS)
    graph.connect("gene:LRRK2", "gene:GBA1", EdgeType.GENE_INTERACTS)
    graph.connect("drug:levodopa", "disease:pd", EdgeType.DRUG_TREATS_DISEASE)
    graph.connect("drug:ambroxol", "protein:gcase", EdgeType.DRUG_TARGETS_PROTEIN)
    graph.connect("drug:prasinezumab", "protein:alpha_syn", EdgeType.DRUG_TARGETS_PROTEIN)
    graph.connect("drug:prasinezumab", "disease:pd", EdgeType.DRUG_TREATS_DISEASE)
    graph.connect("disease:pd", "phenotype:tremor", EdgeType.DISEASE_HAS_PHENOTYPE)
    graph.connect("disease:pd", "phenotype:bradykinesia", EdgeType.DISEASE_HAS_PHENOTYPE)
    graph.connect("disease:pd", "phenotype:rbd", EdgeType.DISEASE_HAS_PHENOTYPE)
    graph.connect("disease:pd", "phenotype:hyposmia", EdgeType.DISEASE_HAS_PHENOTYPE)
    graph.connect("disease:pd", "phenotype:constipation", EdgeType.DISEASE_HAS_PHENOTYPE)
    graph.connect("phenotype:rbd", "stage:nsd:2b", EdgeType.PHENOTYPE_IN_STAGE)
    graph.connect("phenotype:hyposmia", "stage:nsd:2b", EdgeType.PHENOTYPE_IN_STAGE)
    graph.connect("phenotype:hyposmia", "phenotype:bradykinesia", EdgeType.PHENOTYPE_PRECEDES)
    graph.connect("disease:pd", "disease:msa", EdgeType.DISEASE_DIFFERENTIAL)
    graph.connect("disease:pd", "disease:psp", EdgeType.DISEASE_DIFFERENTIAL)
    graph.connect("disease:pd", "disease:et", EdgeType.DISEASE_DIFFERENTIAL)
    for biomarker in iter_biomarker_nodes():
        if biomarker.biofluid:
            target_id = f"biofluid:{biomarker.biofluid.lower().replace(' ', '_')}"
            if target_id not in graph.graph:
                graph.add_node(BaseNode(id=target_id, type=NodeType.BIOFLUID, name=biomarker.biofluid))
            graph.connect(biomarker.id, target_id, EdgeType.BIOMARKER_MEASURES)
        graph.connect(biomarker.id, "disease:pd", EdgeType.BIOMARKER_INDICATES)
        name_lower = biomarker.name.lower()
        if "saa" in name_lower:
            graph.connect(biomarker.id, "stage:nsd:1a", EdgeType.BIOMARKER_PREDICTS_STAGE)
        if "nfl" in name_lower or "datscan" in name_lower or "neuromelanin" in name_lower:
            graph.connect(biomarker.id, "stage:nsd:1b", EdgeType.BIOMARKER_PREDICTS_STAGE)
        if "updrs" in name_lower or "rbd" in name_lower or "hyposmia" in name_lower:
            graph.connect(biomarker.id, "stage:nsd:2b", EdgeType.BIOMARKER_PREDICTS_STAGE)
    return graph


_graph = _build_api_graph()
_nsd = NSDISSStaging()
_syn = SynNeurGeStaging()
_router = RouterAgent()
_biomarker_agent = BiomarkerInterpreterAgent(graph=_graph)
_genetic_agent = GeneticCounselorAgent(graph=_graph)
_staging_agent = StagingAgent()
_risk_agent = RiskAssessorAgent(graph=_graph)
_workflow = PDMultiAgentWorkflow(graph=_graph)


@app.get("/")
def index() -> FileResponse:
    """Serve the production single-page application."""

    return FileResponse(WEB_INDEX_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    """Service health endpoint."""

    return {"status": "ok"}


@app.get("/graph/stats")
def graph_stats() -> dict[str, object]:
    """Return lightweight graph stats."""

    return {
        "node_count": _graph.graph.number_of_nodes(),
        "edge_count": _graph.graph.number_of_edges(),
        "biomarker_categories": BIOMARKER_CATEGORY_COUNTS,
    }


@app.get("/graph/network")
def graph_network(
    search: str | None = Query(default=None),
    category: str | None = Query(default=None),
    stage_relevant: bool = Query(default=False),
    limit: int = Query(default=120, ge=10, le=300),
) -> dict[str, object]:
    """Return graph nodes and edges for the SPA explorer."""

    search_text = search.strip().lower() if isinstance(search, str) else ""
    category_text = category.strip().lower() if isinstance(category, str) else ""
    ranked_nodes: list[tuple[str, dict[str, Any], int]] = []
    for node_id, payload in _graph.graph.nodes(data=True):
        node_type = str(payload.get("type", ""))
        biomarker_category = str(payload.get("category", ""))
        stage_links = _stage_relevant_node_ids()
        if stage_relevant and str(node_id) not in stage_links:
            continue
        if category_text and category_text not in {node_type.lower(), biomarker_category.lower()}:
            continue
        name = str(payload.get("name", node_id))
        if search_text and search_text not in name.lower() and search_text not in str(node_id).lower():
            continue
        ranked_nodes.append((str(node_id), dict(payload), _graph.graph.degree(node_id)))
    ranked_nodes.sort(key=lambda item: (-item[2], str(item[1].get("name", item[0]))))
    selected_ids = {node_id for node_id, _, _ in ranked_nodes[:limit]}
    nodes = [
        {
            "id": node_id,
            "label": payload.get("name", node_id),
            "type": payload.get("type", ""),
            "category": payload.get("category", ""),
            "degree": degree,
            "stage_relevant": node_id in _stage_relevant_node_ids(),
            "details": payload,
        }
        for node_id, payload, degree in ranked_nodes[:limit]
    ]
    edges = []
    for source, target, key, payload in _graph.graph.edges(keys=True, data=True):
        if str(source) not in selected_ids or str(target) not in selected_ids:
            continue
        edges.append(
            {
                "id": f"{source}->{target}:{key}",
                "source": str(source),
                "target": str(target),
                "type": str(payload.get("type", key)),
            }
        )
    return {
        "nodes": nodes,
        "edges": edges,
        "autocomplete": sorted(str(node["label"]) for node in nodes),
    }


@app.get("/graph/node/{node_id:path}")
def graph_node_details(node_id: str) -> dict[str, object]:
    """Return details and immediate neighbors for a single graph node."""

    payload = _graph.get_node(node_id)
    if payload is None:
        return {"node": None, "neighbors": []}
    neighbors: list[dict[str, str]] = []
    for source, _target, attributes in _graph.graph.in_edges(node_id, data=True):
        neighbors.append(
            {
                "direction": "in",
                "neighbor_id": str(source),
                "neighbor_name": str(_graph.graph.nodes[source].get("name", source)),
                "edge_type": str(attributes.get("type", "")),
            }
        )
    for _source, target, attributes in _graph.graph.out_edges(node_id, data=True):
        neighbors.append(
            {
                "direction": "out",
                "neighbor_id": str(target),
                "neighbor_name": str(_graph.graph.nodes[target].get("name", target)),
                "edge_type": str(attributes.get("type", "")),
            }
        )
    return {"node": payload, "neighbors": neighbors[:24]}


@app.get("/references/biomarkers")
def biomarker_references() -> dict[str, object]:
    """Return biomarker definitions for the SPA explorer."""

    entries = [
        {
            "name": name,
            **definition,
        }
        for name, definition in sorted(BIOMARKER_LIBRARY.items(), key=lambda item: item[0].lower())
    ]
    return {"biomarkers": entries, "count": len(entries)}


@app.post("/staging/nsdiss")
def classify_nsdiss(patient: PatientData) -> dict[str, object]:
    """Classify a patient using NSD-ISS."""

    return _nsd.classify(patient).model_dump()


@app.post("/staging/synneurge")
def classify_synneurge(patient: PatientData) -> dict[str, object]:
    """Classify a patient using SynNeurGe."""

    return _syn.classify(patient).model_dump()


@app.post("/agents/route")
def route_task(payload: dict[str, str]) -> dict[str, object]:
    """Route a task to a specialist agent."""

    task = payload.get("task", "")
    return _router.classify_task(task).model_dump()


@app.post("/assessment")
def assess_patient(payload: PatientAssessmentRequest) -> dict[str, object]:
    """Run a multimodal patient assessment for the SPA."""

    patient_data = payload.to_patient_data()
    feature_payload = payload.to_feature_payload()
    nsd_result = _nsd.classify(patient_data)
    syn_result = _syn.classify(patient_data)
    progression = _nsd.predict_progression(nsd_result.stage, patient_data.model_dump())
    staging_report = _staging_agent.run("Stage this Parkinson patient", patient_data=patient_data)
    risk_report = _risk_agent.run("Assess Parkinson risk and progression", patient_data=feature_payload)
    biomarker_sections: list[dict[str, Any]] = []
    if payload.saa_result is not None:
        saa_agent = _biomarker_agent.run(
            "Interpret alpha-synuclein SAA",
            biomarker_name=_saa_biomarker_name(payload.saa_biofluid),
            value=payload.saa_result,
            patient_data=patient_data,
        )
        biomarker_sections.append({"agent": saa_agent.agent_name, "content": saa_agent.content, "metadata": saa_agent.metadata})
    if payload.nfl_pg_ml is not None:
        nfl_agent = _biomarker_agent.run(
            "Interpret NfL level",
            biomarker_name="NfL (blood/serum)",
            value=payload.nfl_pg_ml,
            patient_data=patient_data,
        )
        biomarker_sections.append({"agent": nfl_agent.agent_name, "content": nfl_agent.content, "metadata": nfl_agent.metadata})

    genetics_section: dict[str, Any] | None = None
    if payload.genetic_variants or payload.prs_score is not None:
        genetics = _genetic_agent.run(
            "Interpret PD genetic findings",
            patient_data=feature_payload,
            variants=payload.genetic_variants,
            prs_value=payload.prs_score,
            age=payload.age,
        )
        genetics_section = {"agent": genetics.agent_name, "content": genetics.content, "metadata": genetics.metadata}

    modality_importance = _modality_importance_from_metadata(
        risk_report.metadata.get("ml_outputs"),
        fallback_payload=feature_payload,
    )
    top_biomarkers = _derive_top_contributors(feature_payload)
    criteria_panels = {
        "nsd_iss": _build_nsd_criteria_panel(payload, nsd_result),
        "synneurge": _build_synneurge_panel(payload, syn_result),
    }
    workflow_report = "\n\n".join(section["content"] for section in biomarker_sections + ([genetics_section] if genetics_section is not None else []))
    workflow_report = " ".join(
        part
        for part in [
            workflow_report,
            staging_report.content,
            risk_report.content,
        ]
        if part
    )
    return {
        "nsd_iss": nsd_result.model_dump(),
        "synneurge": syn_result.model_dump(),
        "progression": progression.model_dump(),
        "criteria_panels": criteria_panels,
        "risk": {
            "score": risk_report.metadata.get("combined_risk"),
            "confidence_interval": risk_report.metadata.get("confidence_interval"),
            "modality_importance": modality_importance,
            "top_biomarkers": top_biomarkers,
            "recommended_tests": risk_report.metadata.get("recommendations", []),
            "stage_badge": f"NSD-ISS {nsd_result.stage} / {syn_result.label}",
        },
        "agent_summaries": {
            "biomarkers": biomarker_sections,
            "genetics": genetics_section,
            "staging": {"agent": staging_report.agent_name, "content": staging_report.content, "metadata": staging_report.metadata},
            "risk": {"agent": risk_report.agent_name, "content": risk_report.content, "metadata": risk_report.metadata},
        },
        "report": workflow_report,
    }


@app.websocket("/ws/chat")
async def chat_socket(websocket: WebSocket) -> None:
    """Stream multi-agent chat events to the SPA."""

    await stream_agent_chat(websocket, workflow=_workflow, router=_router)


def _stage_relevant_node_ids() -> set[str]:
    """Return the set of nodes directly connected to staging nodes."""

    stage_ids = {str(node_id) for node_id, payload in _graph.graph.nodes(data=True) if str(payload.get("type", "")) == NodeType.STAGING_SYSTEM.value}
    related = set(stage_ids)
    for source, target, _key in _graph.graph.edges(keys=True):
        if str(source) in stage_ids:
            related.add(str(target))
        if str(target) in stage_ids:
            related.add(str(source))
    return related


def _saa_biomarker_name(biofluid: str | None) -> str:
    """Resolve the preferred SAA biomarker name by biofluid."""

    lowered = (biofluid or "").strip().lower()
    if "blood" in lowered:
        return "alpha-synuclein SAA (blood)"
    if "skin" in lowered:
        return "alpha-synuclein SAA (skin)"
    return "alpha-synuclein SAA (CSF)"


def _heuristic_prodromal_score(payload: dict[str, Any]) -> float:
    """Create a deterministic criteria-style prodromal score for the UI."""

    score = 0.02
    if bool(payload.get("saa_result")):
        score += 0.24
    if isinstance(payload.get("nfl_pg_ml"), (int, float)) and float(payload["nfl_pg_ml"]) >= 14.8:
        score += 0.14
    if payload.get("genetic_variants"):
        score += 0.12
    if bool(payload.get("rbd_present")):
        score += 0.16
    if bool(payload.get("hyposmia")):
        score += 0.10
    if isinstance(payload.get("datscan_sbr"), (int, float)) and float(payload["datscan_sbr"]) < 2.0:
        score += 0.12
    if isinstance(payload.get("updrs_part3"), (int, float)) and float(payload["updrs_part3"]) >= 15:
        score += 0.10
    return round(min(score, 0.98), 4)


def _modality_importance_from_metadata(
    ml_outputs: object,
    *,
    fallback_payload: dict[str, Any],
) -> list[dict[str, float | str]]:
    """Convert model attention or fallback heuristics into UI-ready modality weights."""

    modality_names = ["molecular", "genetic", "imaging", "digital", "clinical", "nonmotor", "prodromal"]
    if isinstance(ml_outputs, dict):
        attention = ml_outputs.get("modality_attention")
        if isinstance(attention, list) and len(attention) == len(modality_names):
            total = sum(float(value) for value in attention) or 1.0
            return [{"modality": modality, "weight": round(float(value) / total, 4)} for modality, value in zip(modality_names, attention, strict=False)]
    fallback_weights = {
        "molecular": 0.0,
        "genetic": 0.0,
        "imaging": 0.0,
        "digital": 0.0,
        "clinical": 0.0,
        "nonmotor": 0.0,
        "prodromal": 0.0,
    }
    if fallback_payload.get("saa_result") is not None or fallback_payload.get("nfl_pg_ml") is not None:
        fallback_weights["molecular"] += 1.0
    if fallback_payload.get("genetic_variants") or fallback_payload.get("prs_score") is not None:
        fallback_weights["genetic"] += 1.0
    if fallback_payload.get("datscan_sbr") is not None or fallback_payload.get("nm_mri_abnormal") is not None:
        fallback_weights["imaging"] += 1.0
    if fallback_payload.get("voice_upload_name") or fallback_payload.get("gait_upload_name"):
        fallback_weights["digital"] += 1.0
    if fallback_payload.get("updrs_part3") is not None or fallback_payload.get("hoehn_yahr") is not None:
        fallback_weights["clinical"] += 1.0
    if fallback_payload.get("rbd_present") is not None or fallback_payload.get("hyposmia") is not None:
        fallback_weights["nonmotor"] += 1.0
    if fallback_payload.get("family_history") is not None or fallback_payload.get("years_since_rbd") is not None:
        fallback_weights["prodromal"] += 1.0
    total = sum(fallback_weights.values()) or 1.0
    return [{"modality": modality, "weight": round(weight / total, 4)} for modality, weight in fallback_weights.items()]


def _derive_top_contributors(payload: dict[str, Any]) -> list[dict[str, str | float]]:
    """Build a deterministic top-biomarker list for the prediction panel."""

    contributions: list[tuple[str, float, str]] = []
    if bool(payload.get("saa_result")):
        contributions.append(("alpha-synuclein SAA", 0.24, "Positive synuclein seeding signal"))
    if isinstance(payload.get("nfl_pg_ml"), (int, float)):
        nfl = float(payload["nfl_pg_ml"])
        if nfl >= 14.8:
            contributions.append(("Blood NfL", 0.18, "Above the 14.8 pg/mL neurodegeneration cutoff"))
    variants = payload.get("genetic_variants")
    if isinstance(variants, list) and variants:
        contributions.append(("PD-associated variant burden", 0.14, f"Detected variants: {', '.join(str(item) for item in variants[:3])}"))
    if isinstance(payload.get("datscan_sbr"), (int, float)) and float(payload["datscan_sbr"]) < 2.0:
        contributions.append(("DaTSCAN SBR", 0.16, "Reduced presynaptic dopaminergic signal"))
    if bool(payload.get("rbd_present")):
        contributions.append(("REM sleep behavior disorder", 0.11, "Strong prodromal synucleinopathy marker"))
    if bool(payload.get("hyposmia")):
        contributions.append(("Hyposmia", 0.08, "Established early non-motor PD feature"))
    if isinstance(payload.get("updrs_part3"), (int, float)) and float(payload["updrs_part3"]) >= 15:
        contributions.append(("UPDRS Part III", 0.09, "Clinically meaningful motor burden"))
    contributions.sort(key=lambda item: item[1], reverse=True)
    return [{"name": name, "contribution": round(score, 4), "detail": detail} for name, score, detail in contributions[:5]]


def _build_nsd_criteria_panel(
    payload: PatientAssessmentRequest,
    nsd_result: Any,
) -> list[dict[str, str]]:
    """Return traffic-light NSD-ISS criteria for the UI."""

    return [
        _criterion("SAA positivity", payload.saa_result, "alpha-synuclein assay status"),
        _criterion("Neurodegeneration biomarker", _has_neurodegeneration(payload), "DaTSCAN, NfL, or NM-MRI evidence"),
        _criterion("Subtle prodromal features", bool(payload.rbd_present) or bool(payload.hyposmia), "RBD or hyposmia"),
        _criterion("Motor signs", payload.motor_signs or _at_least(payload.updrs_part3, 15.0), "Motor examination burden"),
        _criterion("Functional impact", payload.functional_impairment not in {None, "", "none"}, "Daily-living impairment"),
        {"label": "Assigned stage", "status": "met", "detail": f"NSD-ISS {nsd_result.stage}"},
    ]


def _build_synneurge_panel(
    payload: PatientAssessmentRequest,
    syn_result: Any,
) -> list[dict[str, str]]:
    """Return traffic-light SynNeurGe criteria for the UI."""

    return [
        {"label": "Synuclein axis", "status": "met" if syn_result.synuclein_axis == "S1" else "not_met", "detail": syn_result.synuclein_axis},
        {
            "label": "Neurodegeneration axis",
            "status": "met" if syn_result.neurodegeneration_axis in {"N1", "N2"} else "not_met",
            "detail": syn_result.neurodegeneration_axis,
        },
        {"label": "Genetic axis", "status": "met" if syn_result.genetic_axis != "G0" else "unknown", "detail": syn_result.genetic_axis},
        _criterion("DaTSCAN/NM-MRI/NfL evidence", _has_neurodegeneration(payload), "Biomarker neurodegeneration evidence"),
        _criterion("Clinical parkinsonism", payload.motor_signs or _at_least(payload.updrs_part3, 15.0), "Manifest motor syndrome"),
    ]


def _criterion(label: str, value: bool | None, detail: str) -> dict[str, str]:
    """Build a traffic-light criterion row."""

    if value is None:
        status = "unknown"
    else:
        status = "met" if value else "not_met"
    return {"label": label, "status": status, "detail": detail}


def _has_neurodegeneration(payload: PatientAssessmentRequest) -> bool | None:
    """Return whether the payload contains neurodegeneration evidence."""

    evidence = [
        payload.datscan_abnormal,
        payload.nm_mri_abnormal,
        _at_least(payload.nfl_pg_ml, 14.8),
        _below(payload.datscan_sbr, 2.0),
    ]
    if all(item is None for item in evidence):
        return None
    return any(bool(item) for item in evidence if item is not None)


def _at_least(value: float | None, threshold: float) -> bool | None:
    """Return whether a numeric value is at least a threshold."""

    if value is None:
        return None
    return float(value) >= threshold


def _below(value: float | None, threshold: float) -> bool | None:
    """Return whether a numeric value is below a threshold."""

    if value is None:
        return None
    return float(value) < threshold
