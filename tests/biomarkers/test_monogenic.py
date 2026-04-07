"""Monogenic variant tests."""

from __future__ import annotations

from parkinson_ai.biomarkers.genetic.monogenic import classify_variant, is_causative_variant


def test_classify_variant() -> None:
    """Known variants should be classified from the catalog."""

    assessment = classify_variant("LRRK2 G2019S")
    assert assessment.gene == "LRRK2"


def test_is_causative_variant() -> None:
    """Causative variants should be detected."""

    assert is_causative_variant("SNCA duplication") is True
