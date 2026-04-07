"""Neurofilament light chain processing."""

from __future__ import annotations

from pydantic import BaseModel


class NfLInterpretation(BaseModel):
    """Structured NfL interpretation."""

    level_pg_ml: float
    cutoff_pg_ml: float
    elevated: bool
    interpretation: str


def age_adjusted_nfl_cutoff(age: int | None) -> float:
    """Return a simple age-adjusted serum NfL cutoff."""

    if age is None:
        return 14.8
    if age < 50:
        return 10.0
    if age < 65:
        return 12.5
    if age < 75:
        return 14.8
    return 18.0


def interpret_nfl(level_pg_ml: float, *, age: int | None = None) -> NfLInterpretation:
    """Interpret a serum NfL measurement."""

    cutoff = age_adjusted_nfl_cutoff(age)
    elevated = level_pg_ml >= cutoff
    interpretation = "Elevated NfL suggests faster progression or atypical parkinsonism risk." if elevated else "NfL is within the expected age-adjusted range."
    return NfLInterpretation(
        level_pg_ml=level_pg_ml,
        cutoff_pg_ml=cutoff,
        elevated=elevated,
        interpretation=interpretation,
    )
