"""Rate limiter with exponential backoff for World Athletics GraphQL API."""

import asyncio
import time
from typing import Optional


class RateLimiter:
    """Token bucket rate limiter with exponential backoff."""

    def __init__(self, max_per_second: float = 3.0, max_retries: int = 5):
        self.max_per_second = max_per_second
        self.min_interval = 1.0 / max_per_second
        self.max_retries = max_retries
        self._last_request: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until we can make the next request."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
            self._last_request = time.monotonic()

    def get_backoff_delay(self, attempt: int) -> float:
        """Exponential backoff: 1s, 2s, 4s, 8s, 16s."""
        return min(2 ** attempt, 30)
