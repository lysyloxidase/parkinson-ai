"""Shared utilities for Parkinson-AI specialist agents."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Protocol, TypeVar

from parkinson_ai.core.llm_client import LLMResponse
from parkinson_ai.orchestration.prompts import SYSTEM_PROMPTS

_T = TypeVar("_T")


class SupportsGenerate(Protocol):
    """Protocol for local LLM clients used by agents."""

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a non-streaming completion."""


def run_coro(coro: Any) -> Any:
    """Run an async coroutine from synchronous agent code."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError("Synchronous agent entrypoints cannot be called from a running event loop.")


def try_generate_text(
    llm_client: SupportsGenerate,
    prompt: str,
    *,
    system: str | None = None,
    timeout_seconds: float = 5.0,
) -> str | None:
    """Attempt bounded LLM generation and return text when available."""

    try:
        response = run_coro(
            asyncio.wait_for(
                llm_client.generate(prompt, system=system),
                timeout=timeout_seconds,
            )
        )
    except Exception:
        return None
    text = str(response.response).strip() if hasattr(response, "response") else ""
    return text or None


def repo_root() -> Path:
    """Return the project root path."""

    return Path(__file__).resolve().parents[2]


def reference_data_path(filename: str) -> Path:
    """Return the path to a reference-data file."""

    return repo_root() / "data" / "reference" / filename


def load_reference_json(filename: str) -> dict[str, Any]:
    """Load a JSON reference-data file."""

    with reference_data_path(filename).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {filename}")
    return data


def system_prompt(agent_name: str) -> str:
    """Return a registered system prompt for an agent."""

    return SYSTEM_PROMPTS.get(agent_name, "You are a Parkinson's disease specialist assistant.")
