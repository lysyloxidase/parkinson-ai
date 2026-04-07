"""Implementation of NSD-ISS and SynNeurGe staging systems."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PatientData(BaseModel):
    """Unified patient data model for staging."""

    saa_result: bool | None = None
    saa_biofluid: str | None = None
    datscan_abnormal: bool | None = None
    datscan_sbr: float | None = None
    nfl_pg_ml: float | None = None
    nm_mri_abnormal: bool | None = None
    motor_signs: bool | None = None
    functional_impairment: str | None = None
    genetic_variants: list[str] = Field(default_factory=list)
    updrs_total: float | None = None
    updrs_part3: float | None = None
    hoehn_yahr: float | None = None
    moca_score: float | None = None
    rbd_present: bool | None = None
    hyposmia: bool | None = None
    age: int | None = None


class NSDStageResult(BaseModel):
    """NSD-ISS classification result."""

    stage: str
    rationale: list[str]
    neurodegeneration_evidence: list[str] = Field(default_factory=list)
    prodromal_features: list[str] = Field(default_factory=list)
    confidence: float = 0.8


class ProgressionPrediction(BaseModel):
    """Expected transition to the next stage."""

    current_stage: str
    next_stage: str | None
    estimated_years_to_next_stage: float | None
    rationale: list[str]


class SynNeurGeResult(BaseModel):
    """SynNeurGe axis result."""

    synuclein_axis: Literal["S0", "S1"]
    neurodegeneration_axis: Literal["N0", "N1", "N2"]
    genetic_axis: Literal["G0", "G1", "G2"]
    label: str
    rationale: list[str]


class NSDISSStaging:
    """Neuronal alpha-synuclein disease integrated staging system."""

    def classify(self, patient_data: PatientData) -> NSDStageResult:
        """Classify a patient into an NSD-ISS stage."""

        rationale: list[str] = []
        neurodegeneration_evidence = self._neurodegeneration_evidence(patient_data)
        prodromal_features = self._prodromal_features(patient_data)
        functional = (patient_data.functional_impairment or "none").lower()

        if not patient_data.saa_result:
            if self._has_fully_penetrant_snca_variant(patient_data) and patient_data.saa_result is None:
                rationale.append("Fully penetrant SNCA variant with unknown SAA supports stage 0.")
                return NSDStageResult(stage="0", rationale=rationale, confidence=0.7)
            rationale.append("Without SAA positivity the case remains outside canonical NSD-ISS staging.")
            return NSDStageResult(
                stage="0",
                rationale=rationale,
                neurodegeneration_evidence=neurodegeneration_evidence,
                prodromal_features=prodromal_features,
                confidence=0.5,
            )

        if patient_data.hoehn_yahr is not None and patient_data.hoehn_yahr >= 5:
            return NSDStageResult(stage="6", rationale=["Hoehn and Yahr 5 implies complete dependence."])
        if functional == "severe" or (patient_data.hoehn_yahr is not None and patient_data.hoehn_yahr >= 4):
            return NSDStageResult(stage="5", rationale=["Severe disability is consistent with stage 5."])
        if functional == "moderate" or (patient_data.hoehn_yahr is not None and patient_data.hoehn_yahr >= 3):
            return NSDStageResult(stage="4", rationale=["Moderate disability is consistent with stage 4."])

        subtle_signs = self._has_subtle_signs(patient_data)
        manifest_motor = bool(patient_data.motor_signs) or (patient_data.updrs_part3 or 0) >= 15
        has_neurodegeneration = bool(neurodegeneration_evidence)

        if manifest_motor and functional in {"mild", "none"}:
            rationale.append("Manifest parkinsonism with early functional effect supports stage 3.")
            return NSDStageResult(
                stage="3",
                rationale=rationale,
                neurodegeneration_evidence=neurodegeneration_evidence,
                prodromal_features=prodromal_features,
                confidence=0.85,
            )
        if subtle_signs and has_neurodegeneration:
            rationale.append("Subtle features plus biomarker neurodegeneration support stage 2B.")
            return NSDStageResult(
                stage="2B",
                rationale=rationale,
                neurodegeneration_evidence=neurodegeneration_evidence,
                prodromal_features=prodromal_features,
                confidence=0.85,
            )
        if subtle_signs:
            rationale.append("Subtle motor or non-motor signs without degeneration support stage 2A.")
            return NSDStageResult(
                stage="2A",
                rationale=rationale,
                neurodegeneration_evidence=neurodegeneration_evidence,
                prodromal_features=prodromal_features,
                confidence=0.8,
            )
        if has_neurodegeneration:
            rationale.append("Isolated SAA positivity plus neurodegeneration supports stage 1B.")
            return NSDStageResult(
                stage="1B",
                rationale=rationale,
                neurodegeneration_evidence=neurodegeneration_evidence,
                prodromal_features=prodromal_features,
                confidence=0.8,
            )
        return NSDStageResult(
            stage="1A",
            rationale=["SAA positivity without additional findings supports stage 1A."],
            prodromal_features=prodromal_features,
            confidence=0.8,
        )

    def predict_progression(
        self,
        current_stage: str,
        baseline_data: dict[str, float | int | str | bool | None],
    ) -> ProgressionPrediction:
        """Predict time to the next milestone using stage-specific priors."""

        defaults: dict[str, tuple[str | None, float | None]] = {
            "0": ("1A", 4.0),
            "1A": ("1B", 3.2),
            "1B": ("2A", 2.5),
            "2A": ("2B", 2.0),
            "2B": ("3", 8.3),
            "3": ("4", 5.9),
            "4": ("5", 2.4),
            "5": ("6", 1.5),
            "6": (None, None),
        }
        next_stage, years = defaults.get(current_stage, (None, None))
        rationale = ["Stage priors are based on published NSD-ISS milestone estimates."]
        if years is not None:
            multiplier = 1.0
            nfl = baseline_data.get("nfl_pg_ml")
            if isinstance(nfl, (float, int)) and float(nfl) >= 14.8:
                multiplier *= 0.85
                rationale.append("Elevated NfL suggests faster progression.")
            updrs = baseline_data.get("updrs_part3")
            if isinstance(updrs, (float, int)) and float(updrs) >= 20:
                multiplier *= 0.9
                rationale.append("Higher UPDRS Part III suggests faster transition.")
            years = round(years * multiplier, 2)
        return ProgressionPrediction(
            current_stage=current_stage,
            next_stage=next_stage,
            estimated_years_to_next_stage=years,
            rationale=rationale,
        )

    def _neurodegeneration_evidence(self, patient_data: PatientData) -> list[str]:
        """Collect neurodegeneration evidence."""

        evidence: list[str] = []
        if patient_data.datscan_abnormal:
            evidence.append("Abnormal DaTSCAN")
        if patient_data.nm_mri_abnormal:
            evidence.append("Abnormal neuromelanin MRI")
        if patient_data.nfl_pg_ml is not None and patient_data.nfl_pg_ml >= 14.8:
            evidence.append("Elevated blood NfL")
        if patient_data.datscan_sbr is not None and patient_data.datscan_sbr < 2.0:
            evidence.append("Low DaTSCAN SBR")
        return evidence

    def _prodromal_features(self, patient_data: PatientData) -> list[str]:
        """Collect prodromal features."""

        features: list[str] = []
        if patient_data.rbd_present:
            features.append("RBD")
        if patient_data.hyposmia:
            features.append("Hyposmia")
        if patient_data.moca_score is not None and patient_data.moca_score < 26:
            features.append("Cognitive symptoms")
        return features

    def _has_subtle_signs(self, patient_data: PatientData) -> bool:
        """Return whether subtle signs are present without clear disability."""

        if bool(patient_data.rbd_present) or bool(patient_data.hyposmia):
            return True
        if patient_data.updrs_part3 is not None and 5 <= patient_data.updrs_part3 < 15:
            return True
        if patient_data.updrs_total is not None and 8 <= patient_data.updrs_total < 30:
            return True
        return False

    def _has_fully_penetrant_snca_variant(self, patient_data: PatientData) -> bool:
        """Return whether a highly penetrant SNCA genotype is present."""

        variants = {variant.lower() for variant in patient_data.genetic_variants}
        return any(key in variants for key in {"snca triplication", "snca duplication"})


class SynNeurGeStaging:
    """SynNeurGe biological staging on synuclein, neurodegeneration, and genetics axes."""

    def classify(self, patient_data: PatientData) -> SynNeurGeResult:
        """Classify a patient on the SynNeurGe axes."""

        rationale: list[str] = []
        synuclein_axis: Literal["S0", "S1"] = "S1" if patient_data.saa_result else "S0"
        rationale.append("S1 assigned due to positive alpha-synuclein assay." if synuclein_axis == "S1" else "S0 assigned because alpha-synuclein evidence is absent.")

        neuro_axis: Literal["N0", "N1", "N2"]
        if self._has_biomarker_neurodegeneration(patient_data):
            neuro_axis = "N2"
            rationale.append("N2 assigned due to biomarker evidence of neurodegeneration.")
        elif bool(patient_data.motor_signs) or (patient_data.updrs_part3 or 0) >= 15:
            neuro_axis = "N1"
            rationale.append("N1 assigned due to clinical motor manifestations.")
        else:
            neuro_axis = "N0"
            rationale.append("N0 assigned because no neurodegeneration signal is documented.")

        genetic_axis = self._genetic_axis(patient_data.genetic_variants)
        rationale.append(f"{genetic_axis} assigned from the provided genetic variants.")
        label = f"{synuclein_axis}{neuro_axis}{genetic_axis}"
        return SynNeurGeResult(
            synuclein_axis=synuclein_axis,
            neurodegeneration_axis=neuro_axis,
            genetic_axis=genetic_axis,
            label=label,
            rationale=rationale,
        )

    def _has_biomarker_neurodegeneration(self, patient_data: PatientData) -> bool:
        """Return whether biomarker evidence supports neurodegeneration."""

        return bool(patient_data.datscan_abnormal or patient_data.nm_mri_abnormal or (patient_data.nfl_pg_ml is not None and patient_data.nfl_pg_ml >= 14.8) or (patient_data.datscan_sbr is not None and patient_data.datscan_sbr < 2.0))

    def _genetic_axis(self, variants: list[str]) -> Literal["G0", "G1", "G2"]:
        """Assign the genetic axis."""

        normalized = {variant.lower() for variant in variants}
        causative = {
            "lrrk2 g2019s",
            "gba1 l444p",
            "snca duplication",
            "snca triplication",
            "park2",
            "pink1",
            "dj-1",
            "vps35 d620n",
        }
        risk = {
            "gba1 n370s",
            "gba1 e326k",
            "gba1 t369m",
            "prs high",
            "high polygenic risk percentile",
        }
        if normalized.intersection(causative):
            return "G2"
        if normalized.intersection(risk):
            return "G1"
        return "G0"
