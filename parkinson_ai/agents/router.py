"""Task router for the multi-agent system."""

from __future__ import annotations

import re

from pydantic import BaseModel

from parkinson_ai.agents.base import AgentResult, BaseAgent


class RoutedTask(BaseModel):
    """A routed task assignment."""

    agent: str
    confidence: float
    reason: str
    suggested_agents: list[str] = []


class RouterAgent(BaseAgent):
    """Heuristic router that assigns tasks to specialist agents."""

    def __init__(self) -> None:
        super().__init__("router")

    def classify_task(self, task: str) -> RoutedTask:
        """Assign a task to the most relevant agent."""

        lowered = task.lower()
        rules = [
            ("staging_agent", ["nsd", "synneurge", "stage", "staging"]),
            ("genetic_counselor", ["gene", "variant", "lrrk2", "gba1", "prs"]),
            ("imaging_analyst", ["datscan", "mri", "pet", "imaging"]),
            ("literature_agent", ["paper", "pubmed", "citation", "literature"]),
            ("kg_explorer", ["graph", "knowledge graph", "pathway", "network", "symptom", "symptoms", "sign", "signs", "phenotype", "phenotypes"]),
            ("biomarker_interpreter", ["nfl", "alpha-syn", "alpha-synuclein", "saa", "biomarker", "blood test"]),
            ("risk_assessor", ["risk", "prodromal", "predict"]),
            ("drug_analyst", ["drug", "levodopa", "trial", "therapeutic"]),
        ]
        matched_agents = [agent for agent, keywords in rules if _matches_any(lowered, keywords)]
        if len(matched_agents) >= 2 or ("patient" in lowered and any(token in lowered for token in ["saa", "nfl", "lrrk2", "updrs", "rbd"])):
            return RoutedTask(
                agent="multi_agent",
                confidence=0.9,
                reason="The query spans multiple PD specialties and benefits from decomposition.",
                suggested_agents=matched_agents or ["biomarker_interpreter", "genetic_counselor", "staging_agent", "risk_assessor"],
            )
        for agent, keywords in rules:
            if _matches_any(lowered, keywords):
                return RoutedTask(
                    agent=agent,
                    confidence=0.85,
                    reason=f"Matched keywords for {agent}.",
                    suggested_agents=[agent],
                )
        if _matches_any(lowered, ["parkinson", "pd"]):
            return RoutedTask(
                agent="literature_agent",
                confidence=0.65,
                reason="General Parkinson disease question routed to literature grounding.",
                suggested_agents=["literature_agent"],
            )
        return RoutedTask(agent="sentinel", confidence=0.55, reason="Fallback verification route.", suggested_agents=["sentinel"])

    def run(self, task: str, **kwargs: object) -> AgentResult:
        """Route a task and return the assignment."""

        routed = self.classify_task(task)
        return AgentResult(agent_name=self.name, content=routed.agent, metadata=routed.model_dump())


def _matches_any(text: str, keywords: list[str]) -> bool:
    """Return whether any keyword matches as a word or phrase."""

    return any(re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in keywords)
