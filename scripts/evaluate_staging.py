"""CLI for evaluating PD staging logic on a bundled demo cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from parkinson_ai.knowledge_graph.staging import NSDISSStaging, PatientData, SynNeurGeStaging
except ModuleNotFoundError:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from parkinson_ai.knowledge_graph.staging import NSDISSStaging, PatientData, SynNeurGeStaging


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Evaluate NSD-ISS and SynNeurGe on a bundled demo cohort.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full case-level results as JSON.",
    )
    return parser.parse_args()


def demo_cohort() -> list[tuple[str, PatientData, str, str]]:
    """Return a small deterministic cohort with expected staging labels."""

    return [
        (
            "at_risk_snca",
            PatientData(saa_result=None, genetic_variants=["SNCA triplication"], age=46),
            "0",
            "S0N0G2",
        ),
        (
            "saa_only",
            PatientData(saa_result=True, saa_biofluid="CSF", age=61),
            "1A",
            "S1N0G0",
        ),
        (
            "biological_degeneration",
            PatientData(saa_result=True, datscan_abnormal=True, nfl_pg_ml=18.0, age=66),
            "1B",
            "S1N2G0",
        ),
        (
            "prodromal_features",
            PatientData(saa_result=True, rbd_present=True, hyposmia=True, age=64),
            "2A",
            "S1N0G0",
        ),
        (
            "clinical_pd",
            PatientData(
                saa_result=True,
                motor_signs=True,
                updrs_part3=24.0,
                functional_impairment="mild",
                age=69,
            ),
            "3",
            "S1N1G0",
        ),
        (
            "genetic_pd",
            PatientData(
                saa_result=True,
                motor_signs=True,
                datscan_abnormal=True,
                genetic_variants=["LRRK2 G2019S"],
                functional_impairment="mild",
                age=68,
            ),
            "3",
            "S1N2G2",
        ),
    ]


def evaluate_demo_cohort() -> dict[str, Any]:
    """Evaluate the staging systems on the bundled cohort."""

    nsd = NSDISSStaging()
    syn = SynNeurGeStaging()
    cases: list[dict[str, Any]] = []
    nsd_correct = 0
    syn_correct = 0

    for case_id, patient, expected_nsd, expected_syn in demo_cohort():
        nsd_result = nsd.classify(patient)
        syn_result = syn.classify(patient)
        if nsd_result.stage == expected_nsd:
            nsd_correct += 1
        if syn_result.label == expected_syn:
            syn_correct += 1
        cases.append(
            {
                "case_id": case_id,
                "expected_nsd_iss": expected_nsd,
                "predicted_nsd_iss": nsd_result.stage,
                "expected_synneurge": expected_syn,
                "predicted_synneurge": syn_result.label,
                "patient_data": patient.model_dump(),
            }
        )

    total = len(cases)
    return {
        "cohort": "bundled_demo_cohort",
        "note": "PPMI-based evaluation is not available yet because the PPMI loader remains a stub.",
        "cases": cases,
        "summary": {
            "case_count": total,
            "nsd_iss_accuracy": round(nsd_correct / total, 4),
            "synneurge_accuracy": round(syn_correct / total, 4),
        },
    }


def main() -> None:
    """Run the staging evaluation CLI."""

    args = parse_args()
    results = evaluate_demo_cohort()
    if args.json:
        print(json.dumps(results, indent=2))
        return
    summary = results["summary"]
    print("Staging evaluation completed.")
    print(f"Cohort: {results['cohort']}")
    print(results["note"])
    print(f"Cases: {summary['case_count']}")
    print(f"NSD-ISS accuracy: {summary['nsd_iss_accuracy']:.2%}")
    print(f"SynNeurGe accuracy: {summary['synneurge_accuracy']:.2%}")


if __name__ == "__main__":
    main()
