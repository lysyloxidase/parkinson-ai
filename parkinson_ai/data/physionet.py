"""Physionet data interface stub."""

from __future__ import annotations

from typing import Any


class PhysionetLoader:
    """Placeholder external data client."""

    async def fetch(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Fetch remote data."""

        raise NotImplementedError("PhysionetLoader is not implemented yet.")
