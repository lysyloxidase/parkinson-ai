"""Alpha-synuclein biomarker tests."""

from __future__ import annotations

from parkinson_ai.biomarkers.molecular.alpha_synuclein import interpret_saa_result


def test_interpret_saa_result() -> None:
    """Positive assays should return supportive interpretation."""

    result = interpret_saa_result("CSF", True)
    assert result.positive is True
    assert result.sensitivity == 0.89
