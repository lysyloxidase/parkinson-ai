# Changelog

All notable changes to ParkinsonAI are documented in this file.

## 0.2.0 - 2026-04-07

- Delivered the production-facing ParkinsonAI SPA in [web/index.html](./web/index.html) with a three-panel layout for PD knowledge-graph exploration, streaming chat, multimodal patient assessment, staging visualization, and biomarker reference browsing.
- Expanded the FastAPI surface with graph-network, node-details, biomarker-reference, assessment, and WebSocket streaming endpoints to support the new UI.
- Added the final-phase Rich demo in [scripts/demo.py](./scripts/demo.py) to build a compact PD graph, index PubMed literature, stage example patients, and benchmark the UCI voice baseline.
- Replaced placeholder genetic reference catalogs with source-backed literature data in [data/reference/gwas_134_loci.json](./data/reference/gwas_134_loci.json) and [data/reference/gba1_variants.json](./data/reference/gba1_variants.json).
- Hardened package exports by defining `__all__` across package `__init__.py` files.
- Upgraded project documentation with a production README, structured benchmark tables in `RESULTS.md`, and final-phase validation tests for the new API and reference-data surface.

## 0.1.0

- Created the initial `parkinson-ai` scaffold with configuration, PD knowledge-graph schema, staging logic, core RAG plumbing, and FastAPI service foundations.
- Added typed stubs and incremental implementations for biomarkers, multimodal ML, orchestration, agents, visualization, and data-ingestion modules.
