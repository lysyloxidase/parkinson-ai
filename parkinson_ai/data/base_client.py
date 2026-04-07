"""Async API client base classes with retries and simple rate limiting."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from time import monotonic
from typing import Any

import httpx
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential


class AsyncAPIClient:
    """Shared async API client for public biomedical data sources."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 30.0,
        rate_limit_per_second: float = 3.0,
        headers: Mapping[str, str] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rate_limit_per_second = rate_limit_per_second
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url,
            headers=dict(headers or {}),
            timeout=timeout,
        )
        self._owns_client = client is None
        self._min_interval = 1.0 / rate_limit_per_second if rate_limit_per_second > 0 else 0.0
        self._last_request = 0.0

    async def __aenter__(self) -> AsyncAPIClient:
        """Enter the async context manager."""

        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the client when leaving the async context manager."""

        await self.close()

    async def close(self) -> None:
        """Close the underlying HTTP client."""

        if self._owns_client:
            await self._client.aclose()

    async def get_json(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Perform a GET request and decode the response as JSON."""

        response = await self._request("GET", path, params=params, headers=headers)
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError("Expected a JSON object response")
        return {str(key): value for key, value in data.items()}

    async def post_json(
        self,
        path: str,
        *,
        json: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Perform a POST request and decode the response as JSON."""

        response = await self._request("POST", path, json=json, headers=headers)
        data = response.json()
        if not isinstance(data, dict):
            raise TypeError("Expected a JSON object response")
        return {str(key): value for key, value in data.items()}

    async def get_text(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
    ) -> str:
        """Perform a GET request and return the response text."""

        response = await self._request("GET", path, params=params)
        return response.text

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        """Execute a rate-limited request with retries."""

        await self._throttle()
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4),
            retry=retry_if_exception_type(httpx.HTTPError),
            reraise=True,
        ):
            with attempt:
                response = await self._client.request(
                    method=method,
                    url=path,
                    params=params,
                    json=json,
                    headers=dict(headers or {}),
                )
                response.raise_for_status()
                return response
        raise RuntimeError("Request failed after retries")

    async def _throttle(self) -> None:
        """Sleep briefly to respect the configured rate limit."""

        elapsed = monotonic() - self._last_request
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        self._last_request = monotonic()
