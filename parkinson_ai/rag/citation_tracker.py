"""Citation tracking and verification for PubMed-grounded answers."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from parkinson_ai.core.vector_store import RetrievedDocument, VectorDocument

_CITATION_PATTERN = re.compile(r"\[([A-Z][A-Za-z-]+ \d{4})\]")


@dataclass(slots=True)
class CitationRecord:
    """Normalized citation metadata resolved to a PubMed identifier."""

    label: str
    pmid: str
    title: str
    journal: str | None
    year: int | None


@dataclass(slots=True)
class CitationVerification:
    """Verification outcome for a generated answer."""

    cited_labels: list[str] = field(default_factory=list)
    verified: dict[str, CitationRecord] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    @property
    def all_valid(self) -> bool:
        """Return whether all answer citations map to indexed PubMed records."""

        return bool(self.cited_labels) and not self.missing


class CitationTracker:
    """Verify author-year citations against retrieved PubMed documents."""

    def extract_citations(self, text: str) -> list[str]:
        """Extract unique `[Author Year]` citations in appearance order."""

        citations: list[str] = []
        seen: set[str] = set()
        for match in _CITATION_PATTERN.finditer(text):
            label = match.group(1).strip()
            if label not in seen:
                seen.add(label)
                citations.append(label)
        return citations

    def build_index(
        self,
        documents: Sequence[RetrievedDocument | VectorDocument | dict[str, Any]],
    ) -> dict[str, CitationRecord]:
        """Build a citation lookup from retrieved or indexed document metadata."""

        index: dict[str, CitationRecord] = {}
        for document in documents:
            metadata = self._metadata_from_document(document)
            pmid = str(metadata.get("pmid", "")).strip()
            if not pmid.isdigit():
                continue
            label = self.render_label(metadata)
            if not label:
                continue
            record = CitationRecord(
                label=label,
                pmid=pmid,
                title=str(metadata.get("title", "")).strip(),
                journal=str(metadata.get("journal", "")).strip() or None,
                year=_coerce_int(metadata.get("year")),
            )
            index[label] = record
        return index

    def verify(
        self,
        answer: str,
        documents: Sequence[RetrievedDocument | VectorDocument | dict[str, Any]],
    ) -> CitationVerification:
        """Verify that every answer citation maps to a valid PMID."""

        cited_labels = self.extract_citations(answer)
        index = self.build_index(documents)
        verified: dict[str, CitationRecord] = {}
        missing: list[str] = []
        for label in cited_labels:
            if label in index:
                verified[label] = index[label]
            else:
                missing.append(label)
        return CitationVerification(cited_labels=cited_labels, verified=verified, missing=missing)

    def render_label(self, metadata: dict[str, Any]) -> str:
        """Render a normalized `[Author Year]` label from article metadata."""

        explicit = str(metadata.get("citation", "")).strip()
        if explicit:
            return explicit
        year = _coerce_int(metadata.get("year"))
        if year is None:
            return ""
        author = self._first_author_last_name(metadata.get("authors"))
        if not author:
            author = self._first_author_last_name(metadata.get("author"))
        return f"{author} {year}".strip() if author else ""

    def format_inline(self, metadata: dict[str, Any]) -> str:
        """Return an inline citation like `[Kluge 2024]`."""

        label = self.render_label(metadata)
        return f"[{label}]" if label else "[Unknown]"

    def _metadata_from_document(
        self,
        document: RetrievedDocument | VectorDocument | dict[str, Any],
    ) -> dict[str, Any]:
        """Extract metadata regardless of document wrapper type."""

        if isinstance(document, dict):
            return document
        return document.metadata

    def _first_author_last_name(self, authors: object) -> str:
        """Resolve the first author's surname from list or string metadata."""

        if authors is None:
            return ""
        candidates: list[str] = []
        if isinstance(authors, str):
            candidates = [item.strip() for item in authors.split(";") if item.strip()]
        elif isinstance(authors, Iterable):
            candidates = [str(item).strip() for item in authors if str(item).strip()]
        if not candidates:
            return ""
        first_author = candidates[0]
        return first_author.split()[-1].strip(",. ")


def _coerce_int(value: object) -> int | None:
    """Convert a metadata field into an integer when possible."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None
