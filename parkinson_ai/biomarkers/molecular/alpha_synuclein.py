"""Alpha-synuclein seed amplification assay utilities."""

from __future__ import annotations

from pydantic import BaseModel


class AlphaSynSAAResult(BaseModel):
    """Interpreted alpha-synuclein assay result."""

    sample_type: str
    positive: bool
    sensitivity: float
    specificity: float
    interpretation: str


DEFAULT_PERFORMANCE = {
    "csf": (0.89, 0.93),
    "blood": (0.90, 0.91),
    "skin": (0.91, 0.91),
    "extracellular vesicles": (0.94, 1.00),
}


def interpret_saa_result(sample_type: str, positive: bool) -> AlphaSynSAAResult:
    """Interpret an alpha-synuclein seed amplification assay result."""

    normalized = sample_type.lower()
    sensitivity, specificity = DEFAULT_PERFORMANCE.get(normalized, (0.90, 0.90))
    interpretation = "Positive alpha-synuclein seeding signal supports biological PD classification." if positive else "Negative seeding signal lowers the likelihood of synucleinopathy."
    return AlphaSynSAAResult(
        sample_type=sample_type,
        positive=positive,
        sensitivity=sensitivity,
        specificity=specificity,
        interpretation=interpretation,
    )


def classify_seed_amplification(signal_ratio: float, cutoff: float = 1.0) -> bool:
    """Classify a raw amplification signal using a configurable cutoff."""

    return signal_ratio >= cutoff
