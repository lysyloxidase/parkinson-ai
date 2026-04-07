"""Prompt registry tests."""

from __future__ import annotations

from parkinson_ai.orchestration.prompts import SYSTEM_PROMPTS


def test_system_prompts_cover_all_ten_agents() -> None:
    """Prompt registry should contain every named agent in the ten-agent stack."""

    expected = {
        "router",
        "biomarker_interpreter",
        "genetic_counselor",
        "imaging_analyst",
        "literature_agent",
        "kg_explorer",
        "staging_agent",
        "risk_assessor",
        "drug_analyst",
        "sentinel",
    }
    assert expected.issubset(SYSTEM_PROMPTS)
