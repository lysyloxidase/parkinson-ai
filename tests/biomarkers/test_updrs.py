"""UPDRS tests."""

from __future__ import annotations

from parkinson_ai.biomarkers.clinical.updrs import score_updrs


def test_updrs_total() -> None:
    """UPDRS totals should sum component parts."""

    score = score_updrs(5, 6, 12, 3)
    assert score.total == 26
    assert score.severity_band == "moderate"
