"""Central registry of predictive Parkinson's disease features."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isnan
from typing import Any, Literal

import torch

FeatureDType = Literal["continuous", "binary", "categorical"]
FeatureCategory = Literal[
    "molecular",
    "genetic",
    "imaging",
    "digital",
    "clinical",
    "nonmotor",
    "prodromal",
]

MODALITY_ORDER: tuple[FeatureCategory, ...] = (
    "molecular",
    "genetic",
    "imaging",
    "digital",
    "clinical",
    "nonmotor",
    "prodromal",
)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Metadata for a predictive PD feature."""

    name: str
    category: FeatureCategory
    dtype: FeatureDType
    value_range: tuple[float, float] | None
    unit: str
    missing_rate: float
    importance_rank: int
    source_dataset: str
    aliases: tuple[str, ...] = ()
    encoder: Literal["identity", "binary", "sex", "motor_subtype", "gba1_severity"] = "identity"
    is_core: bool = True


def _continuous(
    name: str,
    category: FeatureCategory,
    *,
    value_range: tuple[float, float] | None,
    unit: str,
    missing_rate: float,
    importance_rank: int,
    source_dataset: str,
    aliases: Sequence[str] = (),
    is_core: bool = True,
) -> FeatureSpec:
    """Build a continuous feature spec."""

    return FeatureSpec(
        name=name,
        category=category,
        dtype="continuous",
        value_range=value_range,
        unit=unit,
        missing_rate=missing_rate,
        importance_rank=importance_rank,
        source_dataset=source_dataset,
        aliases=tuple(aliases),
        is_core=is_core,
    )


def _binary(
    name: str,
    category: FeatureCategory,
    *,
    importance_rank: int,
    source_dataset: str,
    aliases: Sequence[str] = (),
    is_core: bool = True,
) -> FeatureSpec:
    """Build a binary feature spec."""

    return FeatureSpec(
        name=name,
        category=category,
        dtype="binary",
        value_range=(0.0, 1.0),
        unit="binary",
        missing_rate=0.25,
        importance_rank=importance_rank,
        source_dataset=source_dataset,
        aliases=tuple(aliases),
        encoder="binary",
        is_core=is_core,
    )


def _categorical(
    name: str,
    category: FeatureCategory,
    *,
    encoder: Literal["sex", "motor_subtype", "gba1_severity"],
    importance_rank: int,
    source_dataset: str,
    aliases: Sequence[str] = (),
    is_core: bool = True,
) -> FeatureSpec:
    """Build a categorical feature spec."""

    return FeatureSpec(
        name=name,
        category=category,
        dtype="categorical",
        value_range=None,
        unit="encoded",
        missing_rate=0.15,
        importance_rank=importance_rank,
        source_dataset=source_dataset,
        aliases=tuple(aliases),
        encoder=encoder,
        is_core=is_core,
    )


def _build_feature_specs() -> list[FeatureSpec]:
    """Build the ordered feature catalog."""

    specs: list[FeatureSpec] = [
        _binary("molecular_saa_binary", "molecular", importance_rank=1, source_dataset="PPMI", aliases=("saa_binary", "saa_result")),
        _continuous("molecular_nfl_pg_ml", "molecular", value_range=(0.0, 120.0), unit="pg/mL", missing_rate=0.18, importance_rank=2, source_dataset="PPMI", aliases=("nfl_pg_ml",)),
        _continuous("molecular_dj1_ng_ml", "molecular", value_range=(0.0, 200.0), unit="ng/mL", missing_rate=0.35, importance_rank=24, source_dataset="PPMI", aliases=("dj1_ng_ml",)),
        _continuous("molecular_gcase_activity_pct", "molecular", value_range=(0.0, 200.0), unit="% baseline", missing_rate=0.42, importance_rank=28, source_dataset="AMP-PD", aliases=("gcase_activity_pct",)),
        _continuous("molecular_il6_pg_ml", "molecular", value_range=(0.0, 100.0), unit="pg/mL", missing_rate=0.31, importance_rank=54, source_dataset="PPMI", aliases=("il6_pg_ml",)),
        _continuous("molecular_tnf_alpha_pg_ml", "molecular", value_range=(0.0, 100.0), unit="pg/mL", missing_rate=0.31, importance_rank=57, source_dataset="PPMI", aliases=("tnf_alpha_pg_ml", "tnf_pg_ml")),
        _continuous("molecular_nlr_ratio", "molecular", value_range=(0.0, 20.0), unit="ratio", missing_rate=0.14, importance_rank=36, source_dataset="PPMI", aliases=("nlr_ratio", "nlr")),
        _continuous("molecular_mirna_1", "molecular", value_range=(-5.0, 5.0), unit="normalized expression", missing_rate=0.45, importance_rank=41, source_dataset="PPMI", aliases=("mirna_1",)),
        _continuous("molecular_mirna_2", "molecular", value_range=(-5.0, 5.0), unit="normalized expression", missing_rate=0.45, importance_rank=42, source_dataset="PPMI", aliases=("mirna_2",)),
        _continuous("molecular_mirna_3", "molecular", value_range=(-5.0, 5.0), unit="normalized expression", missing_rate=0.45, importance_rank=43, source_dataset="PPMI", aliases=("mirna_3",)),
        _continuous("molecular_mirna_4", "molecular", value_range=(-5.0, 5.0), unit="normalized expression", missing_rate=0.45, importance_rank=44, source_dataset="PPMI", aliases=("mirna_4",)),
        _continuous("molecular_mirna_5", "molecular", value_range=(-5.0, 5.0), unit="normalized expression", missing_rate=0.45, importance_rank=45, source_dataset="PPMI", aliases=("mirna_5",)),
        _continuous("molecular_mirna_6", "molecular", value_range=(-5.0, 5.0), unit="normalized expression", missing_rate=0.45, importance_rank=46, source_dataset="PPMI", aliases=("mirna_6",)),
        _continuous("molecular_mirna_7", "molecular", value_range=(-5.0, 5.0), unit="normalized expression", missing_rate=0.45, importance_rank=47, source_dataset="PPMI", aliases=("mirna_7",)),
        _continuous("molecular_proteomics_pc1", "molecular", value_range=(-6.0, 6.0), unit="PC score", missing_rate=0.37, importance_rank=16, source_dataset="Bartl 2024", aliases=("proteomics_pc1",)),
        _continuous("molecular_proteomics_pc2", "molecular", value_range=(-6.0, 6.0), unit="PC score", missing_rate=0.37, importance_rank=21, source_dataset="Bartl 2024", aliases=("proteomics_pc2",)),
        _continuous("molecular_metabolomics_pc1", "molecular", value_range=(-6.0, 6.0), unit="PC score", missing_rate=0.4, importance_rank=18, source_dataset="Han 2017", aliases=("metabolomics_pc1",)),
        _continuous("molecular_metabolomics_pc2", "molecular", value_range=(-6.0, 6.0), unit="PC score", missing_rate=0.4, importance_rank=22, source_dataset="Han 2017", aliases=("metabolomics_pc2",)),
        _continuous("molecular_exosomal_alpha_syn", "molecular", value_range=(0.0, 10000.0), unit="relative concentration", missing_rate=0.41, importance_rank=19, source_dataset="AMP-PD", aliases=("exosomal_alpha_syn",)),
        _binary("genetic_lrrk2_carrier", "genetic", importance_rank=9, source_dataset="PPMI genetics", aliases=("lrrk2_carrier",)),
        _categorical("genetic_gba1_severity", "genetic", encoder="gba1_severity", importance_rank=7, source_dataset="AMP-PD", aliases=("gba1_severity",)),
        _binary("genetic_snca_carrier", "genetic", importance_rank=12, source_dataset="AMP-PD", aliases=("snca_carrier",)),
        _binary("genetic_parkin_carrier", "genetic", importance_rank=29, source_dataset="AMP-PD", aliases=("parkin_carrier", "park2_carrier")),
        _binary("genetic_vps35_carrier", "genetic", importance_rank=38, source_dataset="AMP-PD", aliases=("vps35_carrier",)),
        _continuous("genetic_prs_score", "genetic", value_range=(-5.0, 5.0), unit="z-score", missing_rate=0.1, importance_rank=5, source_dataset="GP2", aliases=("prs_score", "prs")),
        _continuous("genetic_lysosomal_grs", "genetic", value_range=(-5.0, 5.0), unit="z-score", missing_rate=0.16, importance_rank=26, source_dataset="GP2", aliases=("lysosomal_grs",)),
        _continuous("genetic_mitochondrial_grs", "genetic", value_range=(-5.0, 5.0), unit="z-score", missing_rate=0.16, importance_rank=27, source_dataset="GP2", aliases=("mitochondrial_grs",)),
        _continuous("genetic_synaptic_grs", "genetic", value_range=(-5.0, 5.0), unit="z-score", missing_rate=0.16, importance_rank=31, source_dataset="GP2", aliases=("synaptic_grs",)),
        _continuous("genetic_immune_grs", "genetic", value_range=(-5.0, 5.0), unit="z-score", missing_rate=0.16, importance_rank=33, source_dataset="GP2", aliases=("immune_grs",)),
        _continuous("genetic_autophagy_grs", "genetic", value_range=(-5.0, 5.0), unit="z-score", missing_rate=0.16, importance_rank=34, source_dataset="GP2", aliases=("autophagy_grs",)),
        _continuous("genetic_age_prs_interaction", "genetic", value_range=(-500.0, 500.0), unit="interaction", missing_rate=0.12, importance_rank=15, source_dataset="GP2", aliases=("age_prs_interaction",)),
        _continuous("imaging_left_putamen_sbr", "imaging", value_range=(0.0, 6.0), unit="SBR", missing_rate=0.28, importance_rank=3, source_dataset="PPMI imaging", aliases=("left_putamen_sbr",)),
        _continuous("imaging_right_putamen_sbr", "imaging", value_range=(0.0, 6.0), unit="SBR", missing_rate=0.28, importance_rank=4, source_dataset="PPMI imaging", aliases=("right_putamen_sbr",)),
        _continuous("imaging_left_caudate_sbr", "imaging", value_range=(0.0, 6.0), unit="SBR", missing_rate=0.28, importance_rank=10, source_dataset="PPMI imaging", aliases=("left_caudate_sbr",)),
        _continuous("imaging_right_caudate_sbr", "imaging", value_range=(0.0, 6.0), unit="SBR", missing_rate=0.28, importance_rank=11, source_dataset="PPMI imaging", aliases=("right_caudate_sbr",)),
        _continuous("imaging_asymmetry_index", "imaging", value_range=(0.0, 2.0), unit="index", missing_rate=0.3, importance_rank=13, source_dataset="PPMI imaging", aliases=("asymmetry_index",)),
        _continuous("imaging_caudate_putamen_ratio", "imaging", value_range=(0.0, 4.0), unit="ratio", missing_rate=0.31, importance_rank=17, source_dataset="PPMI imaging", aliases=("caudate_putamen_ratio",)),
        _continuous("imaging_nm_mri_signal", "imaging", value_range=(0.0, 5.0), unit="contrast ratio", missing_rate=0.39, importance_rank=8, source_dataset="PPMI imaging", aliases=("nm_mri_signal",)),
        _continuous("imaging_qsm_sn_iron", "imaging", value_range=(0.0, 500.0), unit="ppb", missing_rate=0.44, importance_rank=32, source_dataset="PPMI imaging", aliases=("qsm_sn_iron",)),
        _continuous("imaging_sn_volume", "imaging", value_range=(0.0, 2000.0), unit="mm^3", missing_rate=0.42, importance_rank=37, source_dataset="PPMI imaging", aliases=("sn_volume",)),
        _continuous("imaging_free_water", "imaging", value_range=(0.0, 2.0), unit="fraction", missing_rate=0.41, importance_rank=35, source_dataset="PPMI imaging", aliases=("free_water",)),
        _binary("imaging_swallow_tail_sign", "imaging", importance_rank=48, source_dataset="PPMI imaging", aliases=("swallow_tail_sign",)),
        _continuous("imaging_pdrp_score", "imaging", value_range=(-6.0, 6.0), unit="network score", missing_rate=0.46, importance_rank=23, source_dataset="PPMI imaging", aliases=("pdrp_score",)),
        _continuous("imaging_motor_cortex_thickness", "imaging", value_range=(1.0, 6.0), unit="mm", missing_rate=0.35, importance_rank=49, source_dataset="PPMI imaging", aliases=("motor_cortex_thickness",)),
        _continuous("imaging_prefrontal_thickness", "imaging", value_range=(1.0, 6.0), unit="mm", missing_rate=0.35, importance_rank=50, source_dataset="PPMI imaging", aliases=("prefrontal_thickness",)),
        _continuous("imaging_temporal_thickness", "imaging", value_range=(1.0, 6.0), unit="mm", missing_rate=0.35, importance_rank=51, source_dataset="PPMI imaging", aliases=("temporal_thickness",)),
        _continuous("digital_jitter", "digital", value_range=(0.0, 1.0), unit="ratio", missing_rate=0.27, importance_rank=40, source_dataset="UCI voice", aliases=("jitter",)),
        _continuous("digital_shimmer", "digital", value_range=(0.0, 1.0), unit="ratio", missing_rate=0.27, importance_rank=39, source_dataset="UCI voice", aliases=("shimmer",)),
        _continuous("digital_hnr", "digital", value_range=(-10.0, 50.0), unit="dB", missing_rate=0.27, importance_rank=30, source_dataset="UCI voice", aliases=("hnr",)),
        _continuous("digital_mfcc_1", "digital", value_range=(-200.0, 200.0), unit="coefficient", missing_rate=0.29, importance_rank=58, source_dataset="mPower", aliases=("mfcc_1",)),
        _continuous("digital_mfcc_2", "digital", value_range=(-200.0, 200.0), unit="coefficient", missing_rate=0.29, importance_rank=59, source_dataset="mPower", aliases=("mfcc_2",)),
        _continuous("digital_mfcc_3", "digital", value_range=(-200.0, 200.0), unit="coefficient", missing_rate=0.29, importance_rank=60, source_dataset="mPower", aliases=("mfcc_3",)),
        _continuous("digital_mfcc_4", "digital", value_range=(-200.0, 200.0), unit="coefficient", missing_rate=0.29, importance_rank=61, source_dataset="mPower", aliases=("mfcc_4",)),
        _continuous("digital_mfcc_5", "digital", value_range=(-200.0, 200.0), unit="coefficient", missing_rate=0.29, importance_rank=62, source_dataset="mPower", aliases=("mfcc_5",)),
        _continuous("digital_mfcc_6", "digital", value_range=(-200.0, 200.0), unit="coefficient", missing_rate=0.29, importance_rank=63, source_dataset="mPower", aliases=("mfcc_6",)),
        _continuous("digital_mfcc_7", "digital", value_range=(-200.0, 200.0), unit="coefficient", missing_rate=0.29, importance_rank=64, source_dataset="mPower", aliases=("mfcc_7",)),
        _continuous("digital_mfcc_8", "digital", value_range=(-200.0, 200.0), unit="coefficient", missing_rate=0.29, importance_rank=65, source_dataset="mPower", aliases=("mfcc_8",)),
        _continuous("digital_mfcc_9", "digital", value_range=(-200.0, 200.0), unit="coefficient", missing_rate=0.29, importance_rank=66, source_dataset="mPower", aliases=("mfcc_9",)),
        _continuous("digital_mfcc_10", "digital", value_range=(-200.0, 200.0), unit="coefficient", missing_rate=0.29, importance_rank=67, source_dataset="mPower", aliases=("mfcc_10",)),
        _continuous("digital_mfcc_11", "digital", value_range=(-200.0, 200.0), unit="coefficient", missing_rate=0.29, importance_rank=68, source_dataset="mPower", aliases=("mfcc_11",)),
        _continuous("digital_mfcc_12", "digital", value_range=(-200.0, 200.0), unit="coefficient", missing_rate=0.29, importance_rank=69, source_dataset="mPower", aliases=("mfcc_12",)),
        _continuous("digital_mfcc_13", "digital", value_range=(-200.0, 200.0), unit="coefficient", missing_rate=0.29, importance_rank=70, source_dataset="mPower", aliases=("mfcc_13",)),
        _continuous("digital_f0_mean", "digital", value_range=(50.0, 350.0), unit="Hz", missing_rate=0.27, importance_rank=52, source_dataset="UCI voice", aliases=("f0_mean",)),
        _continuous("digital_stride_variability", "digital", value_range=(0.0, 1.0), unit="CV", missing_rate=0.33, importance_rank=20, source_dataset="PhysioNet gait", aliases=("stride_variability",)),
        _continuous("digital_gait_asymmetry", "digital", value_range=(0.0, 1.0), unit="index", missing_rate=0.33, importance_rank=25, source_dataset="PhysioNet gait", aliases=("gait_asymmetry",)),
        _continuous("digital_tremor_frequency", "digital", value_range=(0.0, 20.0), unit="Hz", missing_rate=0.36, importance_rank=53, source_dataset="mPower", aliases=("tremor_frequency", "tremor_freq")),
        _continuous("digital_tremor_amplitude", "digital", value_range=(0.0, 50.0), unit="m/s^2", missing_rate=0.36, importance_rank=71, source_dataset="mPower", aliases=("tremor_amplitude",), is_core=False),
        _continuous("digital_tapping_speed", "digital", value_range=(0.0, 20.0), unit="taps/s", missing_rate=0.34, importance_rank=55, source_dataset="WATCH-PD", aliases=("tapping_speed",), is_core=False),
        _continuous("digital_keystroke_latency", "digital", value_range=(0.0, 1000.0), unit="ms", missing_rate=0.52, importance_rank=56, source_dataset="NeuroQWERTY", aliases=("keystroke_latency",), is_core=False),
        _continuous("digital_f0_std", "digital", value_range=(0.0, 100.0), unit="Hz", missing_rate=0.27, importance_rank=72, source_dataset="UCI voice", aliases=("f0_std",), is_core=False),
        _continuous("clinical_updrs_part_i", "clinical", value_range=(0.0, 52.0), unit="points", missing_rate=0.08, importance_rank=14, source_dataset="PPMI", aliases=("updrs_part_i",)),
        _continuous("clinical_updrs_part_ii", "clinical", value_range=(0.0, 52.0), unit="points", missing_rate=0.08, importance_rank=14, source_dataset="PPMI", aliases=("updrs_part_ii",)),
        _continuous("clinical_updrs_part_iii", "clinical", value_range=(0.0, 132.0), unit="points", missing_rate=0.05, importance_rank=6, source_dataset="PPMI", aliases=("updrs_part3", "updrs_part_iii")),
        _continuous("clinical_updrs_part_iv", "clinical", value_range=(0.0, 24.0), unit="points", missing_rate=0.12, importance_rank=43, source_dataset="PPMI", aliases=("updrs_part_iv",)),
        _continuous("clinical_hoehn_yahr", "clinical", value_range=(0.0, 5.0), unit="stage", missing_rate=0.03, importance_rank=18, source_dataset="PPMI", aliases=("hoehn_yahr",)),
        _continuous("clinical_moca", "clinical", value_range=(0.0, 30.0), unit="points", missing_rate=0.13, importance_rank=28, source_dataset="PPMI", aliases=("moca_score", "moca")),
        _continuous("clinical_mmse", "clinical", value_range=(0.0, 30.0), unit="points", missing_rate=0.16, importance_rank=44, source_dataset="PPMI", aliases=("mmse_score", "mmse")),
        _continuous("clinical_age", "clinical", value_range=(18.0, 100.0), unit="years", missing_rate=0.0, importance_rank=11, source_dataset="PPMI", aliases=("age",)),
        _categorical("clinical_sex", "clinical", encoder="sex", importance_rank=41, source_dataset="PPMI", aliases=("sex",)),
        _continuous("clinical_disease_duration_years", "clinical", value_range=(0.0, 50.0), unit="years", missing_rate=0.07, importance_rank=27, source_dataset="PPMI", aliases=("disease_duration", "disease_duration_years")),
        _categorical("clinical_motor_subtype", "clinical", encoder="motor_subtype", importance_rank=35, source_dataset="PPMI", aliases=("motor_subtype",)),
        _continuous("clinical_ledd", "clinical", value_range=(0.0, 3000.0), unit="mg/day", missing_rate=0.26, importance_rank=63, source_dataset="PPMI", aliases=("ledd", "levodopa_equiv_daily_dose")),
        _binary("nonmotor_rbd_binary", "nonmotor", importance_rank=17, source_dataset="iRBD cohorts", aliases=("rbd_present", "rbd_binary")),
        _continuous("nonmotor_upsit_score", "nonmotor", value_range=(0.0, 40.0), unit="points", missing_rate=0.21, importance_rank=19, source_dataset="PPMI", aliases=("upsit_score", "upsit", "hyposmia_score")),
        _binary("nonmotor_constipation_binary", "nonmotor", importance_rank=42, source_dataset="PPMI", aliases=("constipation_binary", "constipation")),
        _continuous("nonmotor_depression_score", "nonmotor", value_range=(0.0, 63.0), unit="points", missing_rate=0.24, importance_rank=52, source_dataset="PPMI", aliases=("depression_score",)),
        _continuous("nonmotor_anxiety_score", "nonmotor", value_range=(0.0, 63.0), unit="points", missing_rate=0.24, importance_rank=58, source_dataset="PPMI", aliases=("anxiety_score",)),
        _binary("nonmotor_orthostatic_hypotension_binary", "nonmotor", importance_rank=53, source_dataset="PPMI", aliases=("orthostatic_hypotension_binary", "oh_binary")),
        _continuous("nonmotor_hrv_sdnn", "nonmotor", value_range=(0.0, 500.0), unit="ms", missing_rate=0.39, importance_rank=61, source_dataset="WATCH-PD", aliases=("hrv_sdnn",)),
        _continuous("nonmotor_color_discrimination", "nonmotor", value_range=(0.0, 100.0), unit="score", missing_rate=0.47, importance_rank=62, source_dataset="PPMI", aliases=("color_discrimination",)),
        _continuous("prodromal_risk_score", "prodromal", value_range=(0.0, 1.0), unit="probability", missing_rate=0.18, importance_rank=7, source_dataset="MDS prodromal", aliases=("prodromal_risk_score",)),
        _binary("prodromal_family_history", "prodromal", importance_rank=34, source_dataset="MDS prodromal", aliases=("family_history",)),
        _continuous("prodromal_age_at_rbd_onset", "prodromal", value_range=(0.0, 100.0), unit="years", missing_rate=0.55, importance_rank=36, source_dataset="iRBD cohorts", aliases=("age_at_rbd_onset",)),
        _continuous("prodromal_years_since_rbd", "prodromal", value_range=(0.0, 50.0), unit="years", missing_rate=0.51, importance_rank=22, source_dataset="iRBD cohorts", aliases=("years_since_rbd",)),
        _continuous("prodromal_caffeine_intake", "prodromal", value_range=(0.0, 10.0), unit="cups/day", missing_rate=0.34, importance_rank=70, source_dataset="PREDICT-PD", aliases=("caffeine_intake",)),
    ]
    return specs


class PDFeatureRegistry:
    """Central registry of PD predictive features across seven categories.

    The default vector contains 91 core features. A small set of extended digital
    features is also registered for experimentation, but excluded from the default
    benchmark vector to preserve a stable 91-feature core representation.
    """

    FEATURE_SPECS: tuple[FeatureSpec, ...] = tuple(_build_feature_specs())
    MODALITIES: tuple[FeatureCategory, ...] = MODALITY_ORDER

    def __init__(self) -> None:
        self._feature_map = {spec.name: spec for spec in self.FEATURE_SPECS}
        self._alias_map: dict[str, str] = {}
        for spec in self.FEATURE_SPECS:
            for alias in (spec.name, *spec.aliases):
                self._alias_map[alias] = spec.name

    @property
    def core_features(self) -> list[FeatureSpec]:
        """Return the ordered core feature list."""

        return [spec for spec in self.FEATURE_SPECS if spec.is_core]

    @property
    def extended_features(self) -> list[FeatureSpec]:
        """Return experimental registered features outside the 91-feature core."""

        return [spec for spec in self.FEATURE_SPECS if not spec.is_core]

    @property
    def feature_names(self) -> list[str]:
        """Return the ordered names of the core features."""

        return [spec.name for spec in self.core_features]

    @property
    def all_feature_names(self) -> list[str]:
        """Return the ordered names of all registered features."""

        return [spec.name for spec in self.FEATURE_SPECS]

    @property
    def feature_dims(self) -> dict[str, int]:
        """Return the number of core features per modality."""

        return {modality: len([spec for spec in self.core_features if spec.category == modality]) for modality in self.MODALITIES}

    def count(self, *, include_extended: bool = False) -> int:
        """Return the number of registered features."""

        return len(self.FEATURE_SPECS) if include_extended else len(self.core_features)

    def list_by_modality(
        self,
        modality: FeatureCategory,
        *,
        include_extended: bool = False,
    ) -> list[FeatureSpec]:
        """Return the ordered features for a modality."""

        specs = self.FEATURE_SPECS if include_extended else tuple(self.core_features)
        return [spec for spec in specs if spec.category == modality]

    def get(self, name: str) -> FeatureSpec | None:
        """Return a feature spec by canonical name or alias."""

        canonical = self._alias_map.get(name, name)
        return self._feature_map.get(canonical)

    def harmonize_patient_data(self, patient_data: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize nested or aliased patient data into canonical feature keys."""

        harmonized: dict[str, Any] = {}
        for spec in self.FEATURE_SPECS:
            raw = self._extract_raw_value(patient_data, spec)
            if raw is not None:
                harmonized[spec.name] = raw
        return harmonized

    def get_available(self, patient_data: Mapping[str, Any], *, include_extended: bool = False) -> list[str]:
        """Return the feature names that are present for a patient."""

        specs = self.FEATURE_SPECS if include_extended else tuple(self.core_features)
        available: list[str] = []
        for spec in specs:
            encoded = self._encode_feature_value(spec, self._extract_raw_value(patient_data, spec), patient_data)
            if encoded is not None and not isnan(encoded):
                available.append(spec.name)
        return available

    def build_vector(
        self,
        patient_data: Mapping[str, Any],
        *,
        include_extended: bool = False,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build an ordered feature vector using NaN for missing values."""

        specs = self.FEATURE_SPECS if include_extended else tuple(self.core_features)
        values = [self._encode_feature_value(spec, self._extract_raw_value(patient_data, spec), patient_data) for spec in specs]
        numeric_values = [float("nan") if value is None else value for value in values]
        return torch.tensor(numeric_values, dtype=torch.float32, device=device)

    def build_modality_tensors(
        self,
        patient_data: Mapping[str, Any],
        *,
        include_extended: bool = False,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build per-modality tensors from a patient record."""

        modality_tensors: dict[str, torch.Tensor] = {}
        for modality in self.MODALITIES:
            features = self.list_by_modality(modality, include_extended=include_extended)
            values = [self._encode_feature_value(spec, self._extract_raw_value(patient_data, spec), patient_data) for spec in features]
            numeric = [float("nan") if value is None else value for value in values]
            modality_tensors[modality] = torch.tensor(numeric, dtype=torch.float32, device=device)
        return modality_tensors

    def get_modality_mask(
        self,
        patient_data: Mapping[str, Any],
        *,
        include_extended: bool = False,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Return a binary modality mask indicating which modalities are present."""

        mask_values = []
        for modality in self.MODALITIES:
            available = self.list_by_modality(modality, include_extended=include_extended)
            has_any = any(self._encode_feature_value(spec, self._extract_raw_value(patient_data, spec), patient_data) is not None for spec in available)
            mask_values.append(1.0 if has_any else 0.0)
        return torch.tensor(mask_values, dtype=torch.float32, device=device)

    def _extract_raw_value(self, patient_data: Mapping[str, Any], spec: FeatureSpec) -> Any | None:
        """Extract a raw feature value from nested or flat patient data."""

        for key in (spec.name, *spec.aliases):
            if key in patient_data:
                return patient_data[key]
        category_block = patient_data.get(spec.category)
        if isinstance(category_block, Mapping):
            for key in (spec.name, *spec.aliases):
                if key in category_block:
                    return category_block[key]
        return None

    def _encode_feature_value(
        self,
        spec: FeatureSpec,
        raw_value: Any | None,
        patient_data: Mapping[str, Any],
    ) -> float | None:
        """Convert raw values into model-ready numeric features."""

        if spec.name == "genetic_age_prs_interaction" and raw_value is None:
            age_value = self._extract_raw_value(patient_data, self._feature_map["clinical_age"])
            prs_value = self._extract_raw_value(patient_data, self._feature_map["genetic_prs_score"])
            if age_value is not None and prs_value is not None:
                try:
                    return float(age_value) * float(prs_value)
                except (TypeError, ValueError):
                    return None
        if raw_value is None:
            return None
        if spec.encoder == "identity":
            try:
                return float(raw_value)
            except (TypeError, ValueError):
                return None
        if spec.encoder == "binary":
            return _encode_binary(raw_value)
        if spec.encoder == "sex":
            return _encode_sex(raw_value)
        if spec.encoder == "motor_subtype":
            return _encode_motor_subtype(raw_value)
        if spec.encoder == "gba1_severity":
            return _encode_gba1_severity(raw_value)
        return None


def _encode_binary(value: Any) -> float | None:
    """Encode truthy values into 0/1."""

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if float(value) > 0 else 0.0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "positive", "present", "male"}:
            return 1.0
        if lowered in {"0", "false", "no", "negative", "absent", "female"}:
            return 0.0
    return None


def _encode_sex(value: Any) -> float | None:
    """Encode sex values into a numeric representation."""

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"f", "female", "woman"}:
            return 0.0
        if lowered in {"m", "male", "man"}:
            return 1.0
    return _encode_binary(value)


def _encode_motor_subtype(value: Any) -> float | None:
    """Encode PD motor subtype values."""

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"td", "tremor-dominant", "tremor_dominant"}:
            return 0.0
        if lowered in {"pigd", "postural-instability-gait-disorder", "postural_instability_gait_disorder"}:
            return 1.0
        if lowered in {"indeterminate", "mixed"}:
            return 2.0
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _encode_gba1_severity(value: Any) -> float | None:
    """Encode GBA1 variant severity."""

    if isinstance(value, str):
        lowered = value.strip().lower()
        mapping = {"none": 0.0, "risk": 1.0, "mild": 2.0, "severe": 3.0}
        return mapping.get(lowered)
    if isinstance(value, (int, float)):
        return float(value)
    return None


FeatureRegistry = PDFeatureRegistry
