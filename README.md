# ParkinsonAI

> The first open-source platform for comprehensive Parkinson's disease prediction integrating 100+ biomarkers, a PD-specific knowledge graph, multimodal ML, PubMed RAG, and local multi-agent LLM support, with NSD-ISS and SynNeurGe staging.

ParkinsonAI is a research platform for building biologically grounded Parkinson's disease tooling across data ingestion, graph reasoning, multimodal prediction, literature retrieval, and local agent orchestration.

## Why ParkinsonAI?

1. It unifies seven biomarker families in one open codebase instead of splitting molecular, genetic, imaging, digital, and staging work across unrelated repositories.
2. It treats Parkinson's disease as a biological and computational systems problem by combining a PD-specific knowledge graph with multimodal machine learning and PubMed-grounded retrieval.
3. It implements both 2024 biological staging frameworks, NSD-ISS and SynNeurGe, so new α-synuclein-first classification logic is available in open source.
4. It is designed for local-first research workflows with Ollama-backed agents, a FastAPI service, CLIs, and a browser SPA.

## Architecture

```text
                +--------------------------------------+
                |      External Data and Literature    |
                | PubMed, PPMI, AMP-PD, UCI, OT, etc.  |
                +-------------------+------------------+
                                    |
                                    v
   +--------------------+   +-------+--------+   +-----------------------+
   | Biomarker Modules  |-->| PD Knowledge   |<->| RAG Retrieval Layer    |
   | 7 categories       |   | Graph          |   | BM25 + dense + KG      |
   +--------------------+   | 15 node types  |   +-----------------------+
             |              | 22 edge types  |               |
             v              +-------+--------+               v
   +--------------------+           |            +------------------------+
   | Feature Registry   |-----------+----------->| ML Models              |
   | 91 core features   |                       | modality + fusion + Cox |
   +--------------------+                       +------------------------+
             |                                                |
             +----------------------+-------------------------+
                                    v
                     +----------------------------------+
                     | 10-Agent Orchestration Layer     |
                     | Router -> specialists -> sentinel|
                     +----------------+-----------------+
                                      |
                                      v
                   +---------------------------------------+
                   | FastAPI API + WebSocket + SPA + CLIs  |
                   +---------------------------------------+
```

## Biomarker Coverage

| Category | Representative markers | Role in platform |
| --- | --- | --- |
| Molecular | α-synuclein SAA, NfL, DJ-1, GCase, cytokines, miRNA panels | Biological diagnosis, degeneration, progression |
| Genetic | LRRK2, GBA1, SNCA, PRS, pathway-level risk scores | Risk, penetrance, genotype-stratified analysis |
| Imaging | DaTSCAN SBR, NM-MRI, QSM, PET, swallow tail sign | Nigrostriatal and network-level degeneration |
| Digital | Voice, gait, tremor, tapping, typing, handwriting | Remote screening and longitudinal monitoring |
| Clinical | MDS-UPDRS, Hoehn and Yahr, MoCA, MMSE, motor subtype | Clinical severity and phenotype context |
| Non-motor / Prodromal | RBD, hyposmia, constipation, OH, HRV, color discrimination | Pre-diagnostic enrichment and conversion risk |
| Emerging | Blood immunoassay SAA, retinal OCT, microbiome, EEG, sebum | Research-stage expansion layer |

The bundled graph biomarker library includes 100+ entries with ranges, performance metrics, source-paper labels, and clinical utility notes.

## PD Knowledge Graph

### Node types

| Node type | Examples |
| --- | --- |
| Gene | `SNCA`, `LRRK2`, `GBA1`, `PINK1` |
| Protein | α-synuclein, LRRK2 kinase, GCase |
| Variant | `G2019S`, `N370S`, `rs2230288` |
| Disease | PD, MSA, PSP, DLB |
| Drug | Levodopa, prasinezumab, ambroxol |
| ClinicalTrial | PASADENA, LIGHTHOUSE |
| Biomarker | blood NfL, DaTSCAN SBR, RBD |
| Pathway | mitophagy, lysosome, synaptic vesicle cycle |
| Phenotype | tremor, bradykinesia, hyposmia |
| BrainRegion | substantia nigra, putamen, caudate |
| ImagingModality | DaTSCAN, NM-MRI, QSM, PET |
| Biofluid | CSF, blood, skin, plasma |
| GWASLocus | GP2 lead loci |
| StagingSystem | NSD-ISS stages, SynNeurGe axes |
| Publication | PubMed articles and staging papers |

### Edge types

The graph currently models 22 typed relationships such as `gene_causes_disease`, `variant_in_gene`, `drug_targets_protein`, `biomarker_predicts_stage`, `publication_reports_biomarker`, `pathway_crosstalks_with`, and `disease_differential_with`.

### Example graph questions

- What connects `LRRK2` to lysosomal dysfunction?
- Which biomarkers support NSD-ISS `1B` versus `2B`?
- Which drugs target α-synuclein-related pathways?
- What differentiates PD from MSA and PSP in the graph?

## Biological Staging

### NSD-ISS

NSD-ISS stages Parkinson-related synuclein disease from risk states through severe disability:

- `0`: at-risk, such as fully penetrant `SNCA` variants with unknown SAA status
- `1A`: SAA-positive only
- `1B`: SAA-positive plus neurodegeneration biomarker evidence
- `2A`: subtle signs without degeneration
- `2B`: subtle signs plus degeneration
- `3`: clinically established disease with early functional impact
- `4-6`: increasing disability and dependence

### SynNeurGe

SynNeurGe classifies patients along three axes:

- `S`: synuclein biology (`S0`/`S1`)
- `N`: neurodegeneration (`N0`/`N1`/`N2`)
- `G`: genetic contribution (`G0`/`G1`/`G2`)

### Decision sketch

```text
SAA positive?
  yes -> NSD-ISS 1A or higher
        neurodegeneration biomarker positive?
          yes -> 1B or 2B or 3+
          no  -> 1A or 2A
  no  -> SynNeurGe can still capture genetic PD via S0N1G2 or S0N0G2
```

## ML Models

| Model | Modality | Purpose |
| --- | --- | --- |
| `XGBoostPDModel` | Tabular clinical + molecular + genetics | Strong baseline for interpretable biomarker prediction |
| `VoiceCNN` | Voice / MFCC time series | Acoustic PD detection |
| `GaitLSTM` | Gait sensor sequences | Temporal gait-pattern classification |
| `ImagingCNN` | DaTSCAN / MRI slices | Transfer-learning imaging classifier |
| `GenomicRiskModel` | PRS + variants + pathway scores | Genotype-aware risk estimation |
| `GraphModel` | PD graph / GNN scaffold | Future heterograph learning |
| `MultiParkNetLite` | Seven modality groups | Missing-modality-tolerant fusion model |
| `ProdromalConversionModel` | Prodromal survival features | Time-to-conversion modeling |
| `Federated` | Cross-site stub | Flower-ready collaboration stub |

## RAG Pipeline

ParkinsonAI literature retrieval uses:

1. PD-specific query expansion with synonyms like `PD`, `Parkinson's disease`, and `parkinsonism`.
2. Sparse retrieval via BM25-like scoring over sentence-chunked abstracts.
3. Dense retrieval via local embeddings with MedCPT-style configuration and Chroma-compatible persistence.
4. Reciprocal rank fusion across sparse and dense candidates.
5. Knowledge-graph context extraction around genes, drugs, biomarkers, and phenotypes.
6. Cross-encoder reranking for top candidates.
7. Citation verification against PMID-backed metadata.

## Agents

| Agent | Purpose |
| --- | --- |
| Router | Task classification and decomposition |
| Biomarker Interpreter | Range-aware biomarker interpretation |
| Genetic Counselor | Variant classification, penetrance, genotype context |
| Imaging Analyst | Radiology-style PD imaging interpretation |
| Literature Agent | PubMed-grounded Q&A with citation checks |
| KG Explorer | Path tracing and biological rationale |
| Staging Agent | NSD-ISS + SynNeurGe comparison |
| Risk Assessor | Multimodal prodromal and progression risk |
| Drug Analyst | Targets, mechanisms, pipeline, trials |
| Sentinel | Verification and hallucination screening |

## Datasets

| Dataset | Role | Access notes |
| --- | --- | --- |
| PPMI | Core biomarker, imaging, clinical, progression data | Registration required at PPMI |
| AMP-PD | Omics and genotype-rich PD data | Controlled access |
| mPower | Smartphone digital biomarkers | Research access through source program |
| PhysioNet gait datasets | Gait and wearable signals | PhysioNet credentials may be required |
| UCI Parkinson's datasets | Voice and telemonitoring baselines | Open download |
| PubMed | Literature grounding | Open via E-utilities |
| Open Targets | Target and drug-context enrichment | Open GraphQL API |

## Comparison

| Capability | Typical open repos | ParkinsonAI |
| --- | --- | --- |
| PD-specific knowledge graph | Rare or absent | Yes |
| 7-category biomarker stack | Usually partial | Yes |
| NSD-ISS and SynNeurGe implementation | Usually absent | Yes |
| PubMed RAG with KG context | Usually generic | Yes |
| Local multi-agent PD specialists | Rare | Yes |
| End-to-end UI + API + scripts | Usually fragmented | Yes |

No equivalent open-source project currently combines all of these layers in one Parkinson-focused codebase.

## Hardware

| Profile | Suggested hardware |
| --- | --- |
| API, tests, graph, docs | 8 CPU threads, 16 GB RAM |
| Embeddings and hybrid RAG | 16-32 GB RAM, optional GPU |
| Local 8B Ollama workflow | 32 GB RAM recommended |
| Local 14B reasoning workflow | 64 GB RAM recommended |
| Multimodal deep learning | CUDA GPU strongly recommended |

## Quickstart

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .[dev]
ruff check .
mypy --strict .
pytest -v
uvicorn parkinson_ai.api:app --reload
```

### Useful scripts

```bash
python scripts/build_pd_graph.py
python scripts/index_pubmed.py
python scripts/train_models.py --model all
python scripts/demo.py
```

## License

MIT.

## References

Selected foundational papers and reviews used to shape the platform:

1. Simuni et al., 2024, *Lancet Neurology* — NSD-ISS staging.
2. Höglinger et al., 2024, *Lancet Neurology* — SynNeurGe framework.
3. Dam et al., 2024, *npj Parkinson's Disease* — NSD-ISS validation.
4. Bartl et al., 2024 — plasma proteomic premotor panel.
5. Kluge et al., 2024 — blood α-synuclein detection.
6. Gibbons et al., 2024 — skin α-synuclein detection.
7. Marques et al., 2017 — CSF miRNA diagnostic panel.
8. Han et al., 2017 — serum metabolomics panel.
9. Nalls et al., 2019 — Parkinson polygenic risk benchmarks.
10. PPMI consortium publications — longitudinal biomarker and clinical trajectories.
11. GP2 meta-GWAS preprint, 2025 — 134 PD loci.
12. Ruskey et al., 2024, *Movement Disorders* — GBA1 counseling guidance.
13. Hong et al., 2010 — DJ-1 CSF biomarker signal.
14. Serum NfL meta-analysis papers — atypical versus typical parkinsonism.
15. DaTSCAN meta-analyses — diagnostic sensitivity and specificity.
16. Neuromelanin MRI case-control studies — nigral signal loss.
17. QSM subtype papers — nigral iron and phenotype separation.
18. FDOPA PET longitudinal studies — putaminal decline.
19. PDRP FDG-PET network studies.
20. Transcranial sonography meta-analyses.
21. Nigrosome / swallow tail sign MRI studies.
22. NeuroQWERTY publications.
23. WATCH-PD digital biomarker publications.
24. PhysioNet gait classification studies.
25. mPower speech and tapping studies.
26. Hyposmia / UPSIT prodromal studies.
27. iRBD conversion cohort and meta-analysis papers.
28. Autonomic dysfunction and orthostatic hypotension prodromal papers.
29. Retinal OCT PD biomarker studies.
30. Gut microbiome PD classifier studies.
31. EEG motor severity studies in PD.
32. Sebum metabolomics PD signature studies.
33. Open Targets platform methodology.
34. DisGeNET gene-disease integration papers.
35. STRING and Reactome systems biology resources.

## Important Note

ParkinsonAI is a research tool. It is not a medical device, is not intended for diagnosis or treatment decisions, and should not replace clinical judgement or source-dataset governance.
