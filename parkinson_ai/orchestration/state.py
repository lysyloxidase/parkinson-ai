"""Workflow state models for the multi-agent orchestration layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


@dataclass(slots=True)
class Task:
    """A decomposed specialist-agent task."""

    agent: str
    description: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentExecution:
    """A completed agent execution."""

    agent: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkflowState:
    """High-level workflow state used by the orchestration wrapper."""

    query: str
    patient_data: dict[str, Any] = field(default_factory=dict)
    route: str = ""
    tasks: list[Task] = field(default_factory=list)
    results: list[AgentExecution] = field(default_factory=list)
    sentinel_report: dict[str, Any] = field(default_factory=dict)
    final_report: str = ""


class WorkflowRuntimeState(TypedDict, total=False):
    """LangGraph-compatible runtime state."""

    query: str
    patient_data: dict[str, Any]
    route: str
    router_reason: str
    tasks: list[Task]
    results: list[AgentExecution]
    sentinel_report: dict[str, Any]
    final_report: str
