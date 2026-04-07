"""Reference-data integrity tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "data" / "reference"


def _load_reference(path: str) -> dict[str, Any]:
    """Load a reference JSON file."""

    with (REFERENCE_DIR / path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return cast(dict[str, Any], payload)


def test_gwas_134_loci_reference_has_expected_shape() -> None:
    """GWAS catalog should contain 134 representative loci with required fields."""

    payload = _load_reference("gwas_134_loci.json")
    loci = payload["loci"]
    assert payload["loci_count"] == 134
    assert len(loci) == 134
    required_keys = {
        "locus_number",
        "rsid",
        "chr",
        "pos",
        "nearest_gene",
        "odds_ratio",
        "p_value",
    }
    assert all(required_keys.issubset(entry) for entry in loci)
    assert any(entry["nearest_gene"] == "GBA1" for entry in loci)
    assert any(entry["nearest_gene"] == "LRRK2" for entry in loci)
    assert any(entry["nearest_gene"] == "SNCA" for entry in loci)


def test_gba1_variant_reference_has_key_pd_variants() -> None:
    """GBA1 reference should include key PD-enriched variants and fields."""

    payload = _load_reference("gba1_variants.json")
    variants = payload["variants"]
    assert payload["variant_count"] >= 10
    lookup = {entry["variant"]: entry for entry in variants}
    for variant_name in ("N370S", "L444P", "E326K", "T369M"):
        assert variant_name in lookup
        assert lookup[variant_name]["severity"] in {"mild", "severe", "risk"}
        assert "odds_ratio_range" in lookup[variant_name]
        assert "frequency_percent" in lookup[variant_name]
