"""Staging tests."""

from __future__ import annotations

from parkinson_ai.knowledge_graph.staging import NSDISSStaging, PatientData, SynNeurGeStaging


def test_nsdiss_classification() -> None:
    """NSD-ISS should classify a prodromal biomarker-positive patient."""

    patient = PatientData(saa_result=True, datscan_abnormal=True, rbd_present=True, hyposmia=True)
    result = NSDISSStaging().classify(patient)
    assert result.stage == "2B"


def test_synneurge_classification() -> None:
    """SynNeurGe should recognize genetic PD without SAA positivity."""

    patient = PatientData(saa_result=False, motor_signs=True, genetic_variants=["LRRK2 G2019S"])
    result = SynNeurGeStaging().classify(patient)
    assert result.label == "S0N1G2"
