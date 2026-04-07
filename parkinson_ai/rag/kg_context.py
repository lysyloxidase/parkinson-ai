"""Knowledge graph context extraction specialized for Parkinson's disease RAG."""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from parkinson_ai.config import get_settings
from parkinson_ai.knowledge_graph.builder import PDKnowledgeGraph
from parkinson_ai.knowledge_graph.schema import EdgeType, NodeType

_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        NodeType.GENE.value,
        NodeType.DRUG.value,
        NodeType.BIOMARKER.value,
        NodeType.BRAIN_REGION.value,
        NodeType.PHENOTYPE.value,
        NodeType.DISEASE.value,
        NodeType.IMAGING_MODALITY.value,
    }
)

_EDGE_TEXT: dict[str, str] = {
    EdgeType.GENE_CAUSES_DISEASE.value: "is causally linked to",
    EdgeType.VARIANT_IN_GENE.value: "is a variant in",
    EdgeType.VARIANT_RISK_FOR.value: "increases risk for",
    EdgeType.GENE_IN_PATHWAY.value: "participates in",
    EdgeType.DRUG_TARGETS_PROTEIN.value: "targets",
    EdgeType.DRUG_TREATS_DISEASE.value: "treats",
    EdgeType.BIOMARKER_MEASURES.value: "is measured in",
    EdgeType.BIOMARKER_INDICATES.value: "indicates",
    EdgeType.DISEASE_HAS_PHENOTYPE.value: "has phenotype",
    EdgeType.PHENOTYPE_IN_STAGE.value: "appears in",
    EdgeType.BRAIN_REGION_AFFECTED.value: "is affected in",
    EdgeType.IMAGING_DETECTS.value: "detects change in",
    EdgeType.PROTEIN_AGGREGATES_IN.value: "aggregates in",
    EdgeType.GENE_INTERACTS.value: "interacts with",
    EdgeType.PATHWAY_CROSSTALK.value: "cross-talks with",
    EdgeType.LOCUS_NEAR_GENE.value: "is near",
    EdgeType.PUBLICATION_REPORTS.value: "reports",
    EdgeType.TRIAL_INVESTIGATES.value: "investigates",
    EdgeType.BIOMARKER_PREDICTS_STAGE.value: "predicts",
    EdgeType.DRUG_CONTRAINDICATED.value: "is contraindicated in",
    EdgeType.PHENOTYPE_PRECEDES.value: "precedes",
    EdgeType.DISEASE_DIFFERENTIAL.value: "has differential diagnosis with",
}

_NORMALIZATION_RULES: tuple[tuple[str, str], ...] = (
    ("alpha-synuclein", "alpha synuclein"),
    ("alpha syn", "alpha synuclein"),
    ("a-syn", "alpha synuclein"),
    ("a syn", "alpha synuclein"),
    ("parkinson's", "parkinson"),
    ("pd", "parkinson disease"),
    ("rbd", "rem sleep behavior disorder"),
    ("nfl", "neurofilament light"),
    ("nm-mri", "neuromelanin mri"),
    ("dat-scan", "datscan"),
)


@dataclass(slots=True)
class KGEntityMatch:
    """Entity matched between a free-text query and the PD knowledge graph."""

    node_id: str
    node_name: str
    node_type: str
    matched_alias: str
    score: float


@dataclass(slots=True)
class KGTriple:
    """A graph triple represented as both raw ids and readable text."""

    source_id: str
    source_name: str
    edge_type: str
    target_id: str
    target_name: str
    sentence: str


@dataclass(slots=True)
class KGContextResult:
    """Structured KG context returned for a PD literature query."""

    entities: list[KGEntityMatch] = field(default_factory=list)
    triples: list[KGTriple] = field(default_factory=list)

    @property
    def sentences(self) -> list[str]:
        """Return human-readable triples only."""

        return [triple.sentence for triple in self.triples]

    @property
    def node_ids(self) -> list[str]:
        """Return all participating graph node ids."""

        seen: set[str] = set()
        ordered: list[str] = []
        for entity in self.entities:
            if entity.node_id not in seen:
                seen.add(entity.node_id)
                ordered.append(entity.node_id)
        for triple in self.triples:
            for node_id in (triple.source_id, triple.target_id):
                if node_id not in seen:
                    seen.add(node_id)
                    ordered.append(node_id)
        return ordered


class KGContextExtractor:
    """Extract PD-specific entities and local graph evidence for a query."""

    def __init__(self, graph: PDKnowledgeGraph) -> None:
        self.graph = graph

    def extract(
        self,
        query: str,
        *,
        depth: int | None = None,
        max_triples: int | None = None,
    ) -> KGContextResult:
        """Extract matched entities and an N-hop context neighborhood."""

        matches = self.extract_entities(query)
        node_ids = [match.node_id for match in matches]
        triples = self._collect_triples(
            node_ids=node_ids,
            depth=depth or get_settings().KG_CONTEXT_DEPTH,
            max_triples=max_triples or get_settings().KG_CONTEXT_MAX_TRIPLES,
        )
        return KGContextResult(entities=matches, triples=triples)

    def extract_entities(self, text: str) -> list[KGEntityMatch]:
        """Recognize PD-relevant KG entities in free text."""

        normalized_text = _normalize_text(text)
        token_set = set(normalized_text.split())
        matches: list[KGEntityMatch] = []
        seen_ids: set[str] = set()

        for node_id, attributes in self.graph.graph.nodes(data=True):
            node_type = str(attributes.get("type", ""))
            if node_type not in _ENTITY_TYPES:
                continue
            aliases = self._node_aliases(node_id=str(node_id), attributes=attributes)
            best_match = self._best_alias_match(normalized_text, token_set, aliases)
            if best_match is None or str(node_id) in seen_ids:
                continue
            seen_ids.add(str(node_id))
            matches.append(
                KGEntityMatch(
                    node_id=str(node_id),
                    node_name=str(attributes.get("name", node_id)),
                    node_type=node_type,
                    matched_alias=best_match[0],
                    score=best_match[1],
                )
            )

        matches.sort(key=lambda item: (-item.score, item.node_name))
        return matches

    def build_context(
        self,
        query: str,
        *,
        depth: int | None = None,
        max_triples: int | None = None,
    ) -> list[str]:
        """Return natural-language KG sentences for backward compatibility."""

        return self.extract(query, depth=depth, max_triples=max_triples).sentences

    def _collect_triples(self, node_ids: Sequence[str], *, depth: int, max_triples: int) -> list[KGTriple]:
        """Collect an N-hop subgraph around the matched entity nodes."""

        if not node_ids:
            return []

        expanded_nodes = self._expand_nodes(node_ids=node_ids, depth=depth)
        matched_nodes = set(node_ids)
        selected_edges: list[tuple[int, KGTriple]] = []
        seen_keys: set[tuple[str, str, str]] = set()

        for source, target, key, attributes in self.graph.graph.edges(keys=True, data=True):
            source_id = str(source)
            target_id = str(target)
            if source_id not in expanded_nodes and target_id not in expanded_nodes:
                continue

            edge_type = str(attributes.get("type", key))
            dedupe_key = (source_id, target_id, edge_type)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            source_name = str(self.graph.graph.nodes[source].get("name", source_id))
            target_name = str(self.graph.graph.nodes[target].get("name", target_id))
            direct_hits = int(source_id in matched_nodes) + int(target_id in matched_nodes)
            selected_edges.append(
                (
                    direct_hits,
                    KGTriple(
                        source_id=source_id,
                        source_name=source_name,
                        edge_type=edge_type,
                        target_id=target_id,
                        target_name=target_name,
                        sentence=_triple_to_sentence(source_name, edge_type, target_name),
                    ),
                )
            )

        selected_edges.sort(
            key=lambda item: (
                -item[0],
                item[1].source_name,
                item[1].edge_type,
                item[1].target_name,
            )
        )
        return [triple for _, triple in selected_edges[:max_triples]]

    def _expand_nodes(self, node_ids: Sequence[str], *, depth: int) -> set[str]:
        """Expand matched nodes to an N-hop neighborhood."""

        undirected = self.graph.graph.to_undirected(as_view=True)
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque((node_id, 0) for node_id in node_ids)

        while queue:
            node_id, hops = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            if hops >= depth:
                continue
            for neighbor in undirected.neighbors(node_id):
                neighbor_id = str(neighbor)
                if neighbor_id not in visited:
                    queue.append((neighbor_id, hops + 1))
        return visited

    def _node_aliases(self, *, node_id: str, attributes: dict[str, object]) -> set[str]:
        """Generate searchable aliases from node attributes."""

        aliases = {str(attributes.get("name", node_id))}
        synonyms = attributes.get("synonyms", [])
        if isinstance(synonyms, Iterable) and not isinstance(synonyms, str):
            for synonym in synonyms:
                aliases.add(str(synonym))
        for key in ("symbol", "hpo_id", "rsid", "title", "pmid"):
            value = attributes.get(key)
            if value:
                aliases.add(str(value))
        cleaned_aliases: set[str] = set()
        for alias in aliases:
            normalized = _normalize_text(alias)
            if normalized:
                cleaned_aliases.add(normalized)
                if "(" in alias and ")" in alias:
                    cleaned_aliases.add(_normalize_text(re.sub(r"\(.*?\)", "", alias)))
        return cleaned_aliases

    def _best_alias_match(
        self,
        normalized_text: str,
        token_set: set[str],
        aliases: Iterable[str],
    ) -> tuple[str, float] | None:
        """Return the highest-scoring alias match for a query."""

        best: tuple[str, float] | None = None
        for alias in aliases:
            if not alias:
                continue
            score = 0.0
            if len(alias) <= 5 and alias in token_set:
                score = 1.0 + len(alias) / 10.0
            elif f" {alias} " in f" {normalized_text} ":
                score = 1.0 + len(alias.split()) * 0.25 + len(alias) / 100.0
            elif alias in normalized_text:
                score = 0.75 + len(alias.split()) * 0.2
            if best is None or score > best[1]:
                best = (alias, score)
        return best


KGContextBuilder = KGContextExtractor


def _normalize_text(text: str) -> str:
    """Normalize PD text for matching."""

    normalized = text.lower()
    normalized = normalized.replace("α", "alpha").replace("β", "beta")
    for source, target in _NORMALIZATION_RULES:
        normalized = normalized.replace(source, target)
    normalized = re.sub(r"[^a-z0-9\s-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def _triple_to_sentence(source_name: str, edge_type: str, target_name: str) -> str:
    """Convert a graph triple into a readable sentence."""

    relation_text = _EDGE_TEXT.get(edge_type, edge_type.replace("_", " "))
    return f"{source_name} {relation_text} {target_name}."
