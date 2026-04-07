"""Base agent abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AgentResult:
    """Standard agent response."""

    agent_name: str
    content: str
    metadata: dict[str, Any]


class BaseAgent(ABC):
    """Abstract agent interface."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def run(self, task: str, **kwargs: Any) -> AgentResult:
        """Execute an agent task."""
