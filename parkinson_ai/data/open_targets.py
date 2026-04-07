"""Open Targets GraphQL helpers for PD target discovery."""

from __future__ import annotations

from pydantic import BaseModel

from parkinson_ai.data.base_client import AsyncAPIClient

PD_DISEASE_ID = "MONDO_0005180"


class OpenTargetAssociation(BaseModel):
    """Disease target association record."""

    target_id: str
    target_symbol: str
    score: float
    approved_drugs: int = 0


class OpenTargetsClient(AsyncAPIClient):
    """Thin GraphQL wrapper for disease target lookups."""

    def __init__(self) -> None:
        super().__init__("https://api.platform.opentargets.org/api/v4", rate_limit_per_second=2.0)

    async def fetch_pd_targets(self, *, size: int = 20) -> list[OpenTargetAssociation]:
        """Fetch top PD-associated targets."""

        query = """
        query DiseaseTargets($diseaseId: String!, $size: Int!) {
          disease(efoId: $diseaseId) {
            associatedTargets(page: {index: 0, size: $size}) {
              rows {
                score
                target {
                  id
                  approvedSymbol
                  knownDrugs {
                    count
                  }
                }
              }
            }
          }
        }
        """
        payload = await self.post_json(
            "/graphql",
            json={"query": query, "variables": {"diseaseId": PD_DISEASE_ID, "size": size}},
        )
        rows = payload["data"]["disease"]["associatedTargets"]["rows"]
        return [
            OpenTargetAssociation(
                target_id=str(row["target"]["id"]),
                target_symbol=str(row["target"]["approvedSymbol"]),
                score=float(row["score"]),
                approved_drugs=int(row["target"]["knownDrugs"]["count"]),
            )
            for row in rows
        ]
