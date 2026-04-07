"""Agent 9: verification and hallucination detection for PD outputs."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from parkinson_ai.agents.base import AgentResult, BaseAgent
from parkinson_ai.agents.common import load_reference_json
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.knowledge_graph.staging import NSDISSStaging, PatientData, SynNeurGeStaging

_BIOMARKER_VALUE_PATTERN = re.compile(r"\b(NfL|DJ-1|MoCA|MMSE|UPDRS(?:-III| Part III)?)\s*[:=]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)
_STAGE_PATTERN = re.compile(r"NSD-ISS(?: stage)?\s*([0-6](?:[AB])?)", re.IGNORECASE)
_CITATION_PATTERN = re.compile(r"PMID[: ]?(\d+)|\[[A-Z][A-Za-z-]+ \d{4}\]")


class SentinelAgent(BaseAgent):
    """Cross-check biomarker, genetic, citation, and staging claims."""

    def __init__(self, *, graph: PDKnowledgeGraph | None = None) -> None:
        super().__init__("sentinel")
        self.graph = graph or PDKnowledgeGraph()
        self.biomarker_references = load_reference_json("biomarker_reference_ranges.json")
        self.gba1_catalog = load_reference_json("gba1_variants.json")
        self.monogenic_catalog = load_reference_json("pd_genes_monogenic.json")
        self.nsd_iss = NSDISSStaging()
        self.synneurge = SynNeurGeStaging()
        self.known_genes = self._load_known_genes()
        self.known_variants = self._load_known_variants()

    def inspect(
        self,
        content: str,
        *,
        agent_results: Sequence[AgentResult] | None = None,
        patient_data: PatientData | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Inspect content and optional upstream agent outputs."""

        claims = self._extract_claims(content, agent_results)
        issues: list[str] = []
        issues.extend(self._check_biomarker_claims(content, agent_results))
        issues.extend(self._check_genetic_claims(content, agent_results))
        issues.extend(self._check_staging_claims(content, agent_results, patient_data))
        issues.extend(self._check_citations(agent_results))
        confidence_score = max(0.05, round(1.0 - len(issues) * 0.18, 3))
        pmids = re.findall(r"PMID[: ]?(\d+)", content, flags=re.IGNORECASE)
        has_citation_style = bool(_CITATION_PATTERN.search(content))
        has_uncertainty = any(token in content.lower() for token in ["may", "suggest", "likely", "uncertain"])
        passes = not issues or (confidence_score >= 0.5 and (has_citation_style or has_uncertainty or bool(pmids)))
        return {
            "claims": claims,
            "issues": issues,
            "confidence_score": confidence_score,
            "pmid_count": len(pmids),
            "has_uncertainty_language": has_uncertainty,
            "passes": passes,
        }

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Verify a candidate answer or a set of agent outputs."""

        report = self.inspect(
            task,
            agent_results=kwargs.get("agent_results"),
            patient_data=kwargs.get("patient_data"),
        )
        content = "verified" if report["passes"] else "needs-review"
        return AgentResult(agent_name=self.name, content=content, metadata=report)

    def _extract_claims(self, content: str, agent_results: Sequence[AgentResult] | None) -> list[str]:
        """Extract simple sentence-level claims."""

        claims = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", content) if segment.strip()]
        if agent_results is not None:
            claims.extend(result.content.strip() for result in agent_results if result.content.strip())
        return claims

    def _check_biomarker_claims(self, content: str, agent_results: Sequence[AgentResult] | None) -> list[str]:
        """Cross-check explicit biomarker values against seeded cutoffs."""

        issues: list[str] = []
        for biomarker, value_text in _BIOMARKER_VALUE_PATTERN.findall(content):
            value = float(value_text)
            if biomarker.lower() == "nfl" and value < 14.8 and "elevated" in content.lower():
                issues.append("NfL is described as elevated despite being below the 14.8 pg/mL cutoff.")
        if agent_results is None:
            return issues
        for result in agent_results:
            if result.agent_name != "biomarker_interpreter":
                continue
            metadata = result.metadata
            comparison = metadata.get("comparison")
            if not isinstance(comparison, dict):
                continue
            observed_value = metadata.get("value")
            threshold_value = comparison.get("threshold")
            status = str(comparison.get("status", ""))
            if isinstance(observed_value, (int, float)) and isinstance(threshold_value, (int, float)):
                if status == "elevated" and float(observed_value) <= float(threshold_value):
                    issues.append(f"{metadata.get('biomarker_name', 'Biomarker')} was labeled elevated below its threshold.")
                if status == "within_reference" and float(observed_value) > float(threshold_value):
                    issues.append(f"{metadata.get('biomarker_name', 'Biomarker')} was labeled within reference above its threshold.")
        return issues

    def _check_genetic_claims(self, content: str, agent_results: Sequence[AgentResult] | None) -> list[str]:
        """Verify gene and variant names against the seeded PD catalogs."""

        issues: list[str] = []
        uppercase_tokens = re.findall(r"\b[A-Z0-9-]{3,}\b", content)
        for token in uppercase_tokens:
            if token in {"PD", "NSD", "ISS", "PMID", "SAA", "MRI", "PET"}:
                continue
            if re.fullmatch(r"[A-Z]+\d+[A-Z]", token):
                if token not in self.known_variants:
                    issues.append(f"Variant token '{token}' was not found in the seeded PD catalogs.")
            elif token.isalpha() and token not in self.known_genes and len(token) >= 4:
                continue
        if agent_results is None:
            return issues
        for result in agent_results:
            if result.agent_name != "genetic_counselor":
                continue
            variants = result.metadata.get("variants")
            if not isinstance(variants, list):
                continue
            for entry in variants:
                if not isinstance(entry, dict):
                    continue
                gene = str(entry.get("gene", "")).upper()
                variant = str(entry.get("variant", "")).upper()
                if gene != "PRS" and gene not in self.known_genes:
                    issues.append(f"Gene '{gene}' is not present in the seeded PD gene catalog.")
                if gene == "GBA1" and variant not in self.known_variants:
                    issues.append(f"GBA1 variant '{variant}' is not present in the seeded GBA1 catalog.")
        return issues

    def _check_staging_claims(
        self,
        content: str,
        agent_results: Sequence[AgentResult] | None,
        patient_data: PatientData | dict[str, Any] | None,
    ) -> list[str]:
        """Check staging claims against the staging logic."""

        issues: list[str] = []
        patient: PatientData | None
        if isinstance(patient_data, PatientData):
            patient = patient_data
        elif isinstance(patient_data, dict):
            patient = PatientData(**patient_data)
        else:
            patient = None
        if patient is None:
            return issues
        expected_nsd = self.nsd_iss.classify(patient).stage
        expected_syn = self.synneurge.classify(patient).label
        for stage in _STAGE_PATTERN.findall(content):
            if stage != expected_nsd:
                issues.append(f"Claimed NSD-ISS stage {stage} does not match recomputed stage {expected_nsd}.")
        if agent_results is not None:
            for result in agent_results:
                if result.agent_name != "staging_agent":
                    continue
                nsd_meta = result.metadata.get("nsd_iss")
                syn_meta = result.metadata.get("synneurge")
                if isinstance(nsd_meta, dict) and str(nsd_meta.get("stage", "")) != expected_nsd:
                    issues.append("Staging agent NSD-ISS output does not match recomputed staging logic.")
                if isinstance(syn_meta, dict) and str(syn_meta.get("label", "")) != expected_syn:
                    issues.append("Staging agent SynNeurGe output does not match recomputed staging logic.")
        return issues

    def _check_citations(self, agent_results: Sequence[AgentResult] | None) -> list[str]:
        """Verify agent-level citation flags when available."""

        if agent_results is None:
            return []
        issues: list[str] = []
        for result in agent_results:
            if result.agent_name == "literature_agent":
                valid = result.metadata.get("citations_valid")
                if valid is False:
                    issues.append("Literature agent returned unverified citations.")
        return issues

    def _load_known_genes(self) -> set[str]:
        """Load known PD-associated genes from the graph and reference catalog."""

        genes: set[str] = set()
        for _, payload in self.graph.graph.nodes(data=True):
            if str(payload.get("type")) == "Gene":
                symbol = str(payload.get("symbol", payload.get("name", ""))).upper()
                if symbol:
                    genes.add(symbol)
        catalog_genes = self.monogenic_catalog.get("genes", [])
        if isinstance(catalog_genes, list):
            for entry in catalog_genes:
                if isinstance(entry, dict):
                    gene = str(entry.get("gene", "")).upper()
                    if gene:
                        genes.add(gene)
        return genes

    def _load_known_variants(self) -> set[str]:
        """Load known PD-associated variants from the graph and reference catalogs."""

        variants: set[str] = set()
        for _, payload in self.graph.graph.nodes(data=True):
            if str(payload.get("type")) == "Variant":
                name = str(payload.get("name", "")).upper()
                rsid = str(payload.get("rsid", "")).upper()
                if name:
                    variants.add(name)
                if rsid:
                    variants.add(rsid)
        catalog_variants = self.gba1_catalog.get("variants", [])
        if isinstance(catalog_variants, list):
            for entry in catalog_variants:
                if isinstance(entry, dict):
                    variant = str(entry.get("variant", "")).upper()
                    if variant:
                        variants.add(variant)
        return variants
