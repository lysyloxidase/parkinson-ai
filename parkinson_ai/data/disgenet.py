"""DisGeNET client for PD-associated genes."""

from __future__ import annotations

from pydantic import BaseModel

from parkinson_ai.data.base_client import AsyncAPIClient


class GeneDiseaseAssociation(BaseModel):
    """Association between a gene and Parkinson's disease."""

    gene_symbol: str
    disease_name: str
    score: float
    evidence_count: int | None = None


class DisGeNETClient(AsyncAPIClient):
    """Client for the DisGeNET API."""

    def __init__(self, token: str | None = None) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else None
        super().__init__("https://www.disgenet.org/api", headers=headers, rate_limit_per_second=1.0)

    async def fetch_pd_gene_associations(self, *, limit: int = 20) -> list[GeneDiseaseAssociation]:
        """Fetch PD gene-disease associations."""

        data = await self.get_json(
            "/gda/disease/C0030567",
            params={"limit": limit},
            headers={"Accept": "application/json"},
        )
        records = data.get("payload", data)
        if not isinstance(records, list):
            return []
        associations: list[GeneDiseaseAssociation] = []
        for record in records:
            if not isinstance(record, dict):
                continue
            associations.append(
                GeneDiseaseAssociation(
                    gene_symbol=str(record.get("gene_symbol", "")),
                    disease_name=str(record.get("disease_name", "Parkinson disease")),
                    score=float(record.get("score", 0.0)),
                    evidence_count=_coerce_int(record.get("num_pmids")),
                )
            )
        return associations


def _coerce_int(value: object) -> int | None:
    """Convert arbitrary JSON values into integers when possible."""

    try:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return int(value)
        return None
    except (TypeError, ValueError):
        return None
