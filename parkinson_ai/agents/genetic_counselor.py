"""Agent 2: genetic counseling and Parkinson's disease variant interpretation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from parkinson_ai.agents.base import AgentResult, BaseAgent
from parkinson_ai.agents.common import SupportsGenerate, load_reference_json, system_prompt, try_generate_text
from parkinson_ai.core.llm_client import OllamaClient
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph

_CAUSATIVE_GENES = frozenset({"LRRK2", "SNCA", "PARK2", "PINK1", "DJ-1", "VPS35", "ATP13A2", "FBXO7", "SYNJ1"})
_RISK_GENE = "GBA1"
_THERAPEUTIC_HINTS: dict[str, list[str]] = {
    "GBA1": ["Ambroxol", "GCase activators"],
    "LRRK2": ["LRRK2 inhibitors"],
    "SNCA": ["Prasinezumab", "alpha-synuclein antibodies"],
}


class VariantInterpretation(BaseModel):
    """Interpretation for a single PD-associated variant."""

    gene: str
    variant: str
    classification: str
    penetrance_estimate: float | None = None
    severity: str | None = None
    notes: list[str] = Field(default_factory=list)


class GeneticCounselorAgent(BaseAgent):
    """Interpret PD genetic findings and contextualize therapy links."""

    def __init__(
        self,
        *,
        graph: PDKnowledgeGraph | None = None,
        llm_client: SupportsGenerate | None = None,
    ) -> None:
        super().__init__("genetic_counselor")
        self.graph = graph or PDKnowledgeGraph()
        self.llm_client = llm_client or OllamaClient()
        self.gba1_catalog = load_reference_json("gba1_variants.json")
        self.monogenic_catalog = load_reference_json("pd_genes_monogenic.json")

    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Interpret one or more variants or a PRS value."""

        variants = self._normalize_variants(kwargs.get("variants") or kwargs.get("variant_name"))
        patient_data = kwargs.get("patient_data")
        if not variants and isinstance(patient_data, dict):
            variants = self._normalize_variants(patient_data.get("genetic_variants"))
        prs_value = _coerce_float(kwargs.get("prs_value"))
        age = _coerce_int(kwargs.get("age"))
        if age is None and isinstance(patient_data, dict):
            age = _coerce_int(patient_data.get("age"))
        interpretations = [self._interpret_variant(variant, age=age) for variant in variants]
        if prs_value is not None:
            interpretations.append(self._interpret_prs(prs_value=prs_value, age=age))
        therapies, trials, modifiers = self._knowledge_graph_context(interpretations)
        prompt = self._build_prompt(task, interpretations, prs_value, therapies, trials, modifiers)
        content = self._fallback_report(interpretations, prs_value, therapies, trials, modifiers)
        generated = self._try_generate(prompt)
        if generated:
            content = generated
        return AgentResult(
            agent_name=self.name,
            content=content,
            metadata={
                "variants": [item.model_dump() for item in interpretations],
                "prs_value": prs_value,
                "therapies": therapies,
                "clinical_trials": trials,
                "modifiers": modifiers,
            },
        )

    def _normalize_variants(self, variants: object) -> list[str]:
        """Normalize incoming variant payloads into a list of strings."""

        if variants is None:
            return []
        if isinstance(variants, str):
            text = variants.strip()
            if not text:
                return []
            if "," in text:
                return [part.strip() for part in text.split(",") if part.strip()]
            return [text]
        if isinstance(variants, Sequence) and not isinstance(variants, (bytes, bytearray)):
            return [str(item).strip() for item in variants if str(item).strip()]
        return []

    def _interpret_variant(self, raw_variant: str, *, age: int | None) -> VariantInterpretation:
        """Interpret a PD-associated variant with penetrance estimates."""

        gene, variant = self._split_variant(raw_variant)
        notes: list[str] = []
        severity = self._lookup_gba1_severity(gene, variant)
        if gene == _RISK_GENE and severity is not None:
            classification = "risk_factor"
            notes.append(f"GBA1 {variant} is categorized as {severity}.")
        elif gene in _CAUSATIVE_GENES:
            classification = "pathogenic" if variant else "likely_pathogenic"
        elif gene == _RISK_GENE:
            classification = "risk_factor"
        else:
            classification = "VUS"
            notes.append("Variant was not found in the seeded PD monogenic catalog.")

        penetrance = self._estimate_penetrance(gene=gene, variant=variant, severity=severity, age=age)
        if penetrance is not None:
            notes.append(f"Estimated age-adjusted penetrance is {penetrance * 100:.1f}%.")
        inheritance = self._lookup_inheritance(gene)
        if inheritance:
            notes.append(f"Cataloged inheritance: {inheritance}.")
        return VariantInterpretation(
            gene=gene,
            variant=variant or raw_variant,
            classification=classification,
            penetrance_estimate=penetrance,
            severity=severity,
            notes=notes,
        )

    def _interpret_prs(self, *, prs_value: float, age: int | None) -> VariantInterpretation:
        """Interpret a standardized PD polygenic risk score."""

        classification = "risk_factor" if prs_value >= 1.0 else "low_risk" if prs_value <= -1.0 else "intermediate_risk"
        penetrance = max(0.02, min(0.35, 0.08 + max(prs_value, 0.0) * 0.09))
        if age is not None and prs_value > 0:
            penetrance = min(0.5, penetrance + max(age - 60, 0) * 0.004)
        notes = [f"PRS {prs_value:.2f} is classified as {classification.replace('_', ' ')}."]
        return VariantInterpretation(
            gene="PRS",
            variant="polygenic_risk_score",
            classification=classification,
            penetrance_estimate=round(penetrance, 3),
            notes=notes,
        )

    def _knowledge_graph_context(
        self,
        interpretations: Sequence[VariantInterpretation],
    ) -> tuple[list[str], list[str], list[str]]:
        """Collect therapies, trials, and modifiers linked to the genes in the KG."""

        therapies: list[str] = []
        trials: list[str] = []
        modifiers: list[str] = []
        for interpretation in interpretations:
            if interpretation.gene == "PRS":
                modifiers.append("Polygenic background may modify penetrance across monogenic carriers.")
                continue
            therapies.extend(_THERAPEUTIC_HINTS.get(interpretation.gene, []))
            gene_node_id = self._find_gene_node(interpretation.gene)
            if gene_node_id is None:
                continue
            for _, target, attributes in self.graph.graph.out_edges(gene_node_id, data=True):
                if str(attributes.get("type")) == "gene_in_pathway":
                    modifiers.append(str(self.graph.graph.nodes[target].get("name", target)))
            encoded_proteins = [str(node_id) for node_id, payload in self.graph.graph.nodes(data=True) if str(payload.get("type")) == "Protein" and str(payload.get("encoded_by", "")).upper() == interpretation.gene]
            for protein_id in encoded_proteins:
                for source, _, attributes in self.graph.graph.in_edges(protein_id, data=True):
                    if str(attributes.get("type")) == "drug_targets_protein":
                        drug_name = str(self.graph.graph.nodes[source].get("name", source))
                        therapies.append(drug_name)
                        trials.extend(self._trials_for_drug_node(str(source)))
        return _unique(therapies), _unique(trials), _unique(modifiers)

    def _find_gene_node(self, symbol: str) -> str | None:
        """Resolve a gene symbol into a graph node id."""

        for node_id, payload in self.graph.graph.nodes(data=True):
            if str(payload.get("type")) != "Gene":
                continue
            if str(payload.get("symbol", payload.get("name", ""))).upper() == symbol:
                return str(node_id)
        return None

    def _trials_for_drug_node(self, drug_node_id: str) -> list[str]:
        """Return trial names connected to a drug node."""

        names: list[str] = []
        for source, _, attributes in self.graph.graph.in_edges(drug_node_id, data=True):
            if str(attributes.get("type")) == "trial_investigates_drug":
                names.append(str(self.graph.graph.nodes[source].get("name", source)))
        return names

    def _build_prompt(
        self,
        task: str,
        interpretations: Sequence[VariantInterpretation],
        prs_value: float | None,
        therapies: Sequence[str],
        trials: Sequence[str],
        modifiers: Sequence[str],
    ) -> str:
        """Build the LLM prompt for a counseling-style summary."""

        return "\n".join(
            [
                f"Task: {task}",
                f"Interpretations: {[item.model_dump() for item in interpretations]}",
                f"PRS value: {prs_value}",
                f"Therapies: {list(therapies)}",
                f"Trials: {list(trials)}",
                f"Modifiers: {list(modifiers)}",
                "Write a concise counseling-style PD genetics interpretation with risk quantification.",
            ]
        )

    def _fallback_report(
        self,
        interpretations: Sequence[VariantInterpretation],
        prs_value: float | None,
        therapies: Sequence[str],
        trials: Sequence[str],
        modifiers: Sequence[str],
    ) -> str:
        """Build a deterministic counseling-style report."""

        variant_lines = []
        for item in interpretations:
            penetrance_text = f"estimated penetrance {item.penetrance_estimate * 100:.1f}%" if item.penetrance_estimate is not None else "penetrance depends on broader context"
            severity_text = f", severity {item.severity}" if item.severity else ""
            variant_lines.append(f"{item.gene} {item.variant}: {item.classification.replace('_', ' ')}{severity_text}; {penetrance_text}.")
        therapy_text = ", ".join(therapies) if therapies else "No therapy links were recovered from the current KG."
        trial_text = ", ".join(trials) if trials else "No linked PD trials were found in the current KG."
        modifier_text = ", ".join(modifiers[:4]) if modifiers else "No clear genetic modifiers were recovered."
        prs_text = f" PRS={prs_value:.2f}." if prs_value is not None else ""
        return " ".join(variant_lines) + prs_text + f" Therapy context: {therapy_text}. Trial context: {trial_text}. Modifiers: {modifier_text}."

    def _try_generate(self, prompt: str) -> str | None:
        """Attempt LLM generation and fall back silently on failure."""

        return try_generate_text(self.llm_client, prompt, system=system_prompt(self.name))

    def _split_variant(self, raw_variant: str) -> tuple[str, str]:
        """Split a free-text variant string into gene and variant tokens."""

        parts = raw_variant.replace("(", " ").replace(")", " ").split()
        if not parts:
            return raw_variant.upper(), ""
        gene = parts[0].upper()
        variant = " ".join(parts[1:]).strip()
        return gene, variant

    def _lookup_inheritance(self, gene: str) -> str | None:
        """Look up inheritance mode from the seeded monogenic catalog."""

        genes = self.monogenic_catalog.get("genes", [])
        if not isinstance(genes, list):
            return None
        for entry in genes:
            if isinstance(entry, dict) and str(entry.get("gene", "")).upper() == gene:
                inheritance = entry.get("inheritance")
                return str(inheritance) if inheritance is not None else None
        return None

    def _lookup_gba1_severity(self, gene: str, variant: str) -> str | None:
        """Look up GBA1 severity from the reference catalog."""

        if gene != _RISK_GENE:
            return None
        variants = self.gba1_catalog.get("variants", [])
        if not isinstance(variants, list):
            return None
        normalized = variant.upper()
        for entry in variants:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("variant", "")).upper() == normalized:
                severity = entry.get("severity")
                return str(severity) if severity is not None else None
        return None

    def _estimate_penetrance(
        self,
        *,
        gene: str,
        variant: str,
        severity: str | None,
        age: int | None,
    ) -> float | None:
        """Estimate a rough age-adjusted penetrance for major PD variants."""

        effective_age = age or 65
        if gene == "LRRK2" and variant.upper() == "G2019S":
            if effective_age < 60:
                return 0.18
            if effective_age < 70:
                return 0.28
            if effective_age < 80:
                return 0.36
            return 0.425
        if gene == "SNCA" and "triplication" in variant.lower():
            return 0.95
        if gene == "SNCA" and "duplication" in variant.lower():
            return 0.80
        if gene == "GBA1":
            if severity == "severe":
                return 0.28 if effective_age >= 70 else 0.20
            if severity == "mild":
                return 0.14 if effective_age >= 70 else 0.10
            if severity == "risk":
                return 0.08
            return 0.10
        if gene in {"PARK2", "PINK1", "DJ-1", "ATP13A2", "FBXO7", "SYNJ1"}:
            return 0.65
        if gene == "VPS35":
            return 0.50
        return None


def _coerce_float(value: object) -> float | None:
    """Convert a value to float when possible."""

    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_int(value: object) -> int | None:
    """Convert a value to int when possible."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _unique(values: Sequence[str]) -> list[str]:
    """Deduplicate a sequence while preserving order."""

    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output
