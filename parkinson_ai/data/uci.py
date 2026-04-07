"""Uci data interface stub."""

from __future__ import annotations

from typing import Any


class UciLoader:
    """Placeholder external data client."""

    async def fetch(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Fetch remote data."""

        raise NotImplementedError("UciLoader is not implemented yet.")
