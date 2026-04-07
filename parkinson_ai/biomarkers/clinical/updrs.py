"""MDS-UPDRS scoring helpers."""

from __future__ import annotations

from pydantic import BaseModel


class UPDRSScore(BaseModel):
    """Structured MDS-UPDRS score."""

    part_i: float = 0.0
    part_ii: float = 0.0
    part_iii: float = 0.0
    part_iv: float = 0.0

    @property
    def total(self) -> float:
        """Return the full MDS-UPDRS total."""

        return self.part_i + self.part_ii + self.part_iii + self.part_iv

    @property
    def severity_band(self) -> str:
        """Return a simple severity band."""

        if self.total < 20:
            return "mild"
        if self.total < 60:
            return "moderate"
        return "advanced"


def score_updrs(part_i: float, part_ii: float, part_iii: float, part_iv: float) -> UPDRSScore:
    """Build a structured UPDRS score."""

    return UPDRSScore(
        part_i=max(part_i, 0.0),
        part_ii=max(part_ii, 0.0),
        part_iii=max(part_iii, 0.0),
        part_iv=max(part_iv, 0.0),
    )
