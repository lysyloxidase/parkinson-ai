"""Monogenic Parkinson's disease variant interpretation."""

from __future__ import annotations

from pydantic import BaseModel


class VariantAssessment(BaseModel):
    """Interpretation for a monogenic PD variant."""

    variant_name: str
    gene: str
    category: str
    odds_ratio: tuple[float, float] | None = None
    penetrance: str | None = None
    interpretation: str


VARIANT_CATALOG: dict[str, VariantAssessment] = {
    "LRRK2 G2019S": VariantAssessment(
        variant_name="LRRK2 G2019S",
        gene="LRRK2",
        category="causative",
        odds_ratio=(2.1, 5.0),
        penetrance="25-42.5% by age 80",
        interpretation="Autosomal dominant PD variant with incomplete penetrance.",
    ),
    "GBA1 N370S": VariantAssessment(
        variant_name="GBA1 N370S",
        gene="GBA1",
        category="risk",
        odds_ratio=(2.2, 7.8),
        penetrance="Variable",
        interpretation="Mild GBA1 risk variant associated with increased PD susceptibility.",
    ),
    "GBA1 L444P": VariantAssessment(
        variant_name="GBA1 L444P",
        gene="GBA1",
        category="severe_risk",
        odds_ratio=(6.4, 30.4),
        penetrance="Variable",
        interpretation="Severe GBA1 variant with strong PD risk and faster progression tendency.",
    ),
    "SNCA duplication": VariantAssessment(
        variant_name="SNCA duplication",
        gene="SNCA",
        category="causative",
        interpretation="Highly penetrant copy-number gain associated with monogenic PD.",
    ),
    "SNCA triplication": VariantAssessment(
        variant_name="SNCA triplication",
        gene="SNCA",
        category="causative",
        interpretation="Severe copy-number gain associated with early onset and dementia risk.",
    ),
    "PARK2": VariantAssessment(
        variant_name="PARK2",
        gene="PARK2",
        category="causative",
        interpretation="Classic autosomal recessive early-onset PD genotype.",
    ),
    "PINK1": VariantAssessment(
        variant_name="PINK1",
        gene="PINK1",
        category="causative",
        interpretation="Autosomal recessive early-onset PD genotype.",
    ),
    "DJ-1": VariantAssessment(
        variant_name="DJ-1",
        gene="PARK7",
        category="causative",
        interpretation="Rare autosomal recessive PD genotype.",
    ),
    "VPS35 D620N": VariantAssessment(
        variant_name="VPS35 D620N",
        gene="VPS35",
        category="causative",
        interpretation="Rare autosomal dominant PD genotype.",
    ),
}


def classify_variant(variant_name: str) -> VariantAssessment:
    """Interpret a named monogenic PD variant."""

    if variant_name in VARIANT_CATALOG:
        return VARIANT_CATALOG[variant_name]
    return VariantAssessment(
        variant_name=variant_name,
        gene=variant_name.split()[0] if variant_name else "unknown",
        category="unknown",
        interpretation="Variant is not yet curated in the development catalog.",
    )


def is_causative_variant(variant_name: str) -> bool:
    """Return whether a variant is treated as causative."""

    return classify_variant(variant_name).category == "causative"
