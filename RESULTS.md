# Results

This document is the benchmark ledger for ParkinsonAI. Replace the placeholders below with
reproducible run outputs as models, retrieval indices, and staging cohorts mature.

## Per-Modality Model Performance

| Model | Dataset | AUROC | AUPRC | Sensitivity | Specificity | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| XGBoost tabular | PPMI clinical+molecular | TBD | TBD | TBD | TBD | Baseline for mixed biomarker panels |
| Voice CNN-BiLSTM | UCI Parkinson speech | TBD | TBD | TBD | TBD | Raw/MFCC audio model |
| Gait BiLSTM | PhysioNet gait | TBD | TBD | TBD | TBD | Time-series foot sensor model |
| Imaging CNN | PPMI DaTSCAN / MRI ROI | TBD | TBD | TBD | TBD | EfficientNet-B0 transfer learning |
| Genomic MLP | AMP-PD / GP2-derived features | TBD | TBD | TBD | TBD | PRS + monogenic variant embeddings |
| Prodromal Cox / DeepSurv | iRBD conversion cohort | TBD | TBD | TBD | TBD | Time-to-conversion calibration |
| MultiParkNetLite fusion | Multimodal internal cohort | TBD | TBD | TBD | TBD | Seven-modality masked fusion model |

## Fusion Model Leave-One-Modality-Out Ablation

| Removed Modality | Delta AUROC | Delta AUPRC | Observed Failure Mode | Notes |
| --- | ---: | ---: | --- | --- |
| None (full model) | 0.000 | 0.000 | Baseline | Reference condition |
| Molecular | TBD | TBD | Reduced biological PD discrimination | SAA/NfL removed |
| Genetic | TBD | TBD | Lower hereditary subtype recall | PRS and monogenic burden removed |
| Imaging | TBD | TBD | Weaker degeneration staging | DaTSCAN / NM-MRI removed |
| Digital | TBD | TBD | Lower remote-screening sensitivity | Voice / gait / tremor removed |
| Clinical | TBD | TBD | Lower motor severity alignment | UPDRS / H&Y removed |
| Non-motor | TBD | TBD | Lower prodromal recall | RBD / hyposmia removed |
| Prodromal | TBD | TBD | Lower pre-diagnostic forecasting | Family history / exposure context removed |

## Staging Accuracy Versus PPMI Gold Standard

| Staging System | Cohort Split | Accuracy | Macro F1 | Cohen's kappa | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| NSD-ISS | Validation | TBD | TBD | TBD | Compared against adjudicated biological stage |
| SynNeurGe | Validation | TBD | TBD | TBD | Compared against expert axis assignments |
| Concordance NSD-ISS vs SynNeurGe | Validation | TBD | TBD | TBD | Agreement across biological phenotypes |

## Top-20 SHAP Biomarker Importance Ranking

| Rank | Biomarker / Feature | Modality | Mean |SHAP| | Directionality |
| ---: | --- | --- | ---: | --- |
| 1 | alpha-synuclein SAA | Molecular | TBD | Higher positive signal increases PD probability |
| 2 | Blood NfL | Molecular | TBD | Higher values increase neurodegeneration probability |
| 3 | DaTSCAN SBR | Imaging | TBD | Lower values increase PD probability |
| 4 | MDS-UPDRS Part III | Clinical | TBD | Higher values increase motor-stage probability |
| 5 | LRRK2 / GBA1 burden | Genetic | TBD | Variant carriage increases genetic-risk signal |
| 6 | REM sleep behavior disorder | Non-motor | TBD | Presence increases prodromal probability |
| 7 | Hyposmia / UPSIT | Non-motor | TBD | Lower smell function increases prodromal risk |
| 8 | Neuromelanin MRI signal | Imaging | TBD | Lower signal increases degeneration probability |
| 9 | Voice MFCC composite | Digital | TBD | Abnormal speech pattern increases PD probability |
| 10 | Gait asymmetry | Digital | TBD | Higher asymmetry increases PD probability |
| 11 | PRS | Genetic | TBD | Higher PRS increases lifetime risk |
| 12 | MoCA | Clinical | TBD | Lower cognition increases advanced-stage risk |
| 13 | Finger tapping speed | Digital | TBD | Slower tapping increases PD probability |
| 14 | Constipation | Non-motor | TBD | Presence increases prodromal probability |
| 15 | QSM substantia nigra iron | Imaging | TBD | Higher iron burden increases subtype severity |
| 16 | GCase activity | Molecular | TBD | Lower activity increases GBA1-like risk |
| 17 | DJ-1 | Molecular | TBD | Context-dependent signal |
| 18 | Free-water SN | Imaging | TBD | Higher free-water increases degeneration probability |
| 19 | Motor subtype | Clinical | TBD | PIGD weighting alters progression rate |
| 20 | Age x PRS interaction | Genetic | TBD | Older age amplifies polygenic effect |

## RAG Retrieval Quality

| Retriever Configuration | Precision@5 | Recall@10 | MRR | Citation Validity | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| BM25 only | TBD | TBD | TBD | TBD | Sparse baseline |
| Dense only (MedCPT/embedding fallback) | TBD | TBD | TBD | TBD | Semantic baseline |
| BM25 + dense RRF | TBD | TBD | TBD | TBD | Hybrid fusion baseline |
| Hybrid + KG context | TBD | TBD | TBD | TBD | Entity-aware augmentation |
| Hybrid + KG + reranker | TBD | TBD | TBD | TBD | Production PD literature stack |

## Reproducibility Notes

- Record dataset versions, split definitions, and random seeds beside each benchmark update.
- Report confidence intervals for all clinical metrics.
- Keep literature retrieval benchmarks tied to a frozen PubMed snapshot date and query set.
