"""Async Ollama client with retries and optional streaming support."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import BaseModel, Field
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_fixed

from parkinson_ai.config import get_settings
from parkinson_ai.core.utils import get_logger

logger = get_logger(__name__)


class ChatMessage(BaseModel):
    """A chat message sent to a local LLM."""

    role: str
    content: str


class LLMResponse(BaseModel):
    """Normalized response returned by the client."""

    model: str
    response: str
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    done: bool = True
    raw: dict[str, Any] = Field(default_factory=dict)


class OllamaClient:
    """Minimal async wrapper around the Ollama HTTP API."""

    def __init__(
        self,
        host: str | None = None,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.host = (host or get_settings().OLLAMA_HOST).rstrip("/")
        self._timeout = timeout
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def __aenter__(self) -> OllamaClient:
        """Enter the async context manager."""

        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the underlying HTTP client when owned."""

        await self.close()

    async def close(self) -> None:
        """Close the underlying transport if this instance created it."""

        if self._owns_client:
            await self._client.aclose()

    async def health_check(self) -> bool:
        """Return whether the Ollama server is reachable."""

        try:
            response = await self._client.get(f"{self.host}/api/tags")
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            logger.warning("Ollama health check failed", exc_info=True)
            return False

    async def list_models(self) -> list[str]:
        """Return the locally available model names."""

        response = await self._client.get(f"{self.host}/api/tags")
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        return [str(model["name"]) for model in models if "name" in model]

    async def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Generate a single non-streaming completion."""

        payload = {
            "model": model or get_settings().MODEL_REASONER,
            "prompt": prompt,
            "stream": False,
        }
        if system is not None:
            payload["system"] = system
        if options:
            payload["options"] = options

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_fixed(1),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        ):
            with attempt:
                response = await self._client.post(f"{self.host}/api/generate", json=payload)
                response.raise_for_status()
                data = response.json()
                return LLMResponse(
                    model=str(data.get("model", payload["model"])),
                    response=str(data.get("response", "")),
                    prompt_eval_count=_coerce_int(data.get("prompt_eval_count")),
                    eval_count=_coerce_int(data.get("eval_count")),
                    done=bool(data.get("done", True)),
                    raw=data,
                )
        raise RuntimeError("Ollama generation failed after retries")

    async def stream_generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream response fragments from the Ollama generate endpoint."""

        payload = {
            "model": model or get_settings().MODEL_REASONER,
            "prompt": prompt,
            "stream": True,
        }
        if system is not None:
            payload["system"] = system
        if options:
            payload["options"] = options

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_fixed(1),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        ):
            with attempt:
                async with self._client.stream(
                    "POST",
                    f"{self.host}/api/generate",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        data = json.loads(line)
                        chunk = str(data.get("response", ""))
                        if chunk:
                            yield chunk
                        if data.get("done", False):
                            return


def _coerce_int(value: Any) -> int | None:
    """Convert a JSON value to an integer when present."""

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
