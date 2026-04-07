"""Neurofilament tests."""

from __future__ import annotations

from parkinson_ai.biomarkers.molecular.neurofilament import age_adjusted_nfl_cutoff, interpret_nfl


def test_age_adjusted_cutoff() -> None:
    """Older adults should have a higher NfL cutoff."""

    assert age_adjusted_nfl_cutoff(80) > age_adjusted_nfl_cutoff(45)


def test_interpret_nfl() -> None:
    """Elevated NfL should be flagged."""

    result = interpret_nfl(20.0, age=70)
    assert result.elevated is True
