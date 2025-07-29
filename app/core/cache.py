# app/core/cache.py
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class InMemoryCache:
    """A simple, swappable in-memory cache for storing processed documents."""
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        logger.info("Initialized In-Memory Cache.")

    async def get(self, key: str) -> Any:
        return self._cache.get(key)

    async def set(self, key: str, value: Any):
        self._cache[key] = value

# Singleton instance to ensure one cache is used across the entire app
cache = InMemoryCache()