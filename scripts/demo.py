"""Rich interactive demo for ParkinsonAI."""

from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

try:
    from parkinson_ai.agents.genetic_counselor import GeneticCounselorAgent
    from parkinson_ai.agents.risk_assessor import RiskAssessorAgent
    from parkinson_ai.agents.staging_agent import StagingAgent
    from parkinson_ai.knowledge_graph.biomarker_nodes import iter_biomarker_nodes
    from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
    from parkinson_ai.knowledge_graph.schema import BaseNode, EdgeType, GeneNode, NodeType
    from parkinson_ai.knowledge_graph.staging import PatientData
    from parkinson_ai.ml.models.xgboost_model import XGBoostPDModel
    from parkinson_ai.rag.pubmed_indexer import IndexingSummary, PubMedIndexer
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from parkinson_ai.agents.genetic_counselor import GeneticCounselorAgent
    from parkinson_ai.agents.risk_assessor import RiskAssessorAgent
    from parkinson_ai.agents.staging_agent import StagingAgent
    from parkinson_ai.knowledge_graph.biomarker_nodes import iter_biomarker_nodes
    from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
    from parkinson_ai.knowledge_graph.schema import BaseNode, EdgeType, GeneNode, NodeType
    from parkinson_ai.knowledge_graph.staging import PatientData
    from parkinson_ai.ml.models.xgboost_model import XGBoostPDModel
    from parkinson_ai.rag.pubmed_indexer import IndexingSummary, PubMedIndexer

console = Console()

UCI_VOICE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

EXAMPLE_PATIENTS: list[dict[str, Any]] = [
    {
        "label": "Prodromal synucleinopathy",
        "payload": {
            "saa_result": True,
            "saa_biofluid": "CSF",
            "nfl_pg_ml": 11.2,
            "rbd_present": True,
            "hyposmia": True,
            "age": 63,
        },
    },
    {
        "label": "Biological PD with degeneration",
        "payload": {
            "saa_result": True,
            "datscan_abnormal": True,
            "datscan_sbr": 1.6,
            "nfl_pg_ml": 22.0,
            "age": 67,
        },
    },
    {
        "label": "Genetic PD",
        "payload": {
            "saa_result": True,
            "nfl_pg_ml": 18.4,
            "genetic_variants": ["LRRK2 G2019S"],
            "updrs_part3": 19.0,
            "motor_signs": True,
            "functional_impairment": "mild",
            "age": 69,
        },
    },
    {
        "label": "Advanced clinical PD",
        "payload": {
            "saa_result": True,
            "datscan_abnormal": True,
            "nfl_pg_ml": 31.5,
            "updrs_total": 61.0,
            "updrs_part3": 32.0,
            "motor_signs": True,
            "functional_impairment": "moderate",
            "hoehn_yahr": 3.0,
            "age": 74,
        },
    },
    {
        "label": "GBA1-enriched prodromal risk",
        "payload": {
            "saa_result": True,
            "genetic_variants": ["GBA1 N370S"],
            "prs_score": 1.2,
            "rbd_present": True,
            "years_since_rbd": 4.0,
            "age": 61,
        },
    },
]


def build_mini_graph() -> PDKnowledgeGraph:
    """Build a compact graph used for the interactive demo."""

    graph = PDKnowledgeGraph()
    graph.bulk_add_nodes(
        [
            BaseNode(id="disease:pd", type=NodeType.DISEASE, name="Parkinson disease"),
            BaseNode(id="disease:msa", type=NodeType.DISEASE, name="Multiple system atrophy"),
            GeneNode(id="gene:SNCA", name="SNCA", symbol="SNCA", chromosome="4", inheritance_pattern="AD"),
            GeneNode(id="gene:LRRK2", name="LRRK2", symbol="LRRK2", chromosome="12", inheritance_pattern="AD"),
            GeneNode(id="gene:GBA1", name="GBA1", symbol="GBA1", chromosome="1", inheritance_pattern="risk"),
            BaseNode(id="drug:levodopa", type=NodeType.DRUG, name="Levodopa"),
            BaseNode(id="drug:prasinezumab", type=NodeType.DRUG, name="Prasinezumab"),
            BaseNode(id="phenotype:rbd", type=NodeType.PHENOTYPE, name="REM sleep behavior disorder"),
            BaseNode(id="phenotype:hyposmia", type=NodeType.PHENOTYPE, name="Hyposmia"),
        ]
    )
    graph.bulk_add_nodes(list(iter_biomarker_nodes())[:18])
    graph.connect("gene:SNCA", "disease:pd", EdgeType.GENE_CAUSES_DISEASE)
    graph.connect("gene:LRRK2", "disease:pd", EdgeType.GENE_CAUSES_DISEASE)
    graph.connect("gene:GBA1", "disease:pd", EdgeType.GENE_CAUSES_DISEASE)
    graph.connect("drug:levodopa", "disease:pd", EdgeType.DRUG_TREATS_DISEASE)
    graph.connect("drug:prasinezumab", "disease:pd", EdgeType.DRUG_TREATS_DISEASE)
    graph.connect("disease:pd", "phenotype:rbd", EdgeType.DISEASE_HAS_PHENOTYPE)
    graph.connect("disease:pd", "phenotype:hyposmia", EdgeType.DISEASE_HAS_PHENOTYPE)
    for biomarker in list(iter_biomarker_nodes())[:18]:
        graph.connect(biomarker.id, "disease:pd", EdgeType.BIOMARKER_INDICATES)
    return graph


async def build_pubmed_demo_index() -> IndexingSummary | None:
    """Index the default 100-abstract PubMed demo corpus."""

    indexer = PubMedIndexer()
    try:
        return await indexer.fetch_and_index(retmax=20, sort="relevance")
    except Exception:
        return None


def run_assessments(graph: PDKnowledgeGraph) -> None:
    """Run five example multimodal assessments and print a summary table."""

    staging_agent = StagingAgent()
    risk_agent = RiskAssessorAgent(graph=graph)
    genetics_agent = GeneticCounselorAgent(graph=graph)

    table = Table(title="Example Patient Assessments")
    table.add_column("Scenario")
    table.add_column("NSD-ISS")
    table.add_column("SynNeurGe")
    table.add_column("Risk")
    table.add_column("Highlights")

    for example in EXAMPLE_PATIENTS:
        payload = dict(example["payload"])
        patient = PatientData(**{key: value for key, value in payload.items() if key in PatientData.model_fields})
        staging = staging_agent.run(example["label"], patient_data=patient)
        risk = risk_agent.run(example["label"], patient_data=payload)
        genetics_summary = ""
        if payload.get("genetic_variants"):
            genetics = genetics_agent.run(
                "Interpret genetic profile",
                patient_data=payload,
                variants=payload["genetic_variants"],
                prs_value=payload.get("prs_score"),
                age=payload.get("age"),
            )
            genetics_summary = genetics.content
        nsd_stage = staging.metadata["nsd_iss"]["stage"]
        syn_label = staging.metadata["synneurge"]["label"]
        risk_score = float(risk.metadata["combined_risk"])
        table.add_row(
            str(example["label"]),
            str(nsd_stage),
            str(syn_label),
            f"{risk_score:.2f}",
            genetics_summary or str(staging.content),
        )

    console.print(table)


def benchmark_uci_voice() -> dict[str, float] | None:
    """Benchmark the tabular XGBoost baseline on the UCI voice dataset."""

    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            response = client.get(UCI_VOICE_URL)
            response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text))
    except Exception:
        return None

    features = frame.drop(columns=["name", "status"]).to_numpy(dtype=float)
    labels = frame["status"].to_numpy(dtype=int)
    model = XGBoostPDModel(feature_names=[str(column) for column in frame.columns if column not in {"name", "status"}])
    metrics = model.cross_validate(features, labels, n_splits=5)
    summary = metrics.iloc[-1]
    return {
        "auroc": float(summary["auroc"]),
        "auprc": float(summary["auprc"]),
        "sensitivity": float(summary["sensitivity"]),
        "specificity": float(summary["specificity"]),
    }


def render_voice_metrics(metrics: dict[str, float] | None) -> None:
    """Render the UCI voice benchmark results."""

    if metrics is None:
        console.print(Panel("UCI voice benchmark could not be completed because the dataset was unavailable.", title="Voice Benchmark"))
        return
    table = Table(title="UCI Voice Baseline")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in metrics.items():
        table.add_row(key.upper(), f"{value:.3f}")
    console.print(table)


def render_graph_summary(graph: PDKnowledgeGraph) -> None:
    """Print a compact graph summary."""

    stats = graph.statistics()
    console.print(
        Panel(
            f"Nodes: {stats['node_count']}\nEdges: {stats['edge_count']}\nTypes: {', '.join(f'{key}={value}' for key, value in stats['node_type_counts'].items() if value)}",
            title="Mini PD Knowledge Graph",
        )
    )


def main() -> None:
    """Run the end-to-end ParkinsonAI demo."""

    console.print(
        Panel(
            "Build a compact PD graph, index PubMed abstracts, stage five example patients, and benchmark a voice baseline.",
            title="ParkinsonAI Demo",
        )
    )

    graph = build_mini_graph()
    render_graph_summary(graph)

    summary = asyncio.run(build_pubmed_demo_index())
    if summary is None:
        console.print(Panel("PubMed indexing skipped because the network or NCBI endpoint was unavailable.", title="PubMed Index"))
    else:
        console.print(
            Panel(
                f"Queries: {summary.query_count}\nArticles: {summary.article_count}\nChunks: {summary.chunk_count}",
                title="PubMed Index",
            )
        )

    run_assessments(graph)
    render_voice_metrics(benchmark_uci_voice())


if __name__ == "__main__":
    main()
