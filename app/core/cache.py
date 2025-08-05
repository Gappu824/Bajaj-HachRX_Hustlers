# app/core/cache.py - Enhanced with TTL and size limits
from typing import Dict, Any, Optional
import logging
import time
import sys
from collections import OrderedDict

logger = logging.getLogger(__name__)

class EnhancedInMemoryCache:
    """Enhanced in-memory cache with TTL and size limits"""
    def __init__(self, max_size_mb: int = 500, ttl_seconds: int = 3600):
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.current_size = 0
        logger.info(f"Initialized Enhanced Cache (max: {max_size_mb}MB, TTL: {ttl_seconds}s)")

    async def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            entry = self._cache[key]
            # Check TTL
            if time.time() - entry['timestamp'] > self.ttl_seconds:
                # Expired
                await self.delete(key)
                return None
            # Move to end (LRU)
            self._cache.move_to_end(key)
            return entry['value']
        return None

    async def set(self, key: str, value: Any):
        # Estimate size (rough)
        size = sys.getsizeof(value)
        
        # If adding this would exceed limit, remove oldest entries
        while self.current_size + size > self.max_size_bytes and self._cache:
            oldest_key = next(iter(self._cache))
            await self.delete(oldest_key)
        
        # Add/update entry
        if key in self._cache:
            self.current_size -= sys.getsizeof(self._cache[key]['value'])
        
        self._cache[key] = {
            'value': value,
            'timestamp': time.time(),
            'size': size
        }
        self.current_size += size
        
        # Move to end (LRU)
        self._cache.move_to_end(key)
        
        logger.debug(f"Cached {key} (size: {size/1024:.1f}KB, total: {self.current_size/1024/1024:.1f}MB)")

    async def delete(self, key: str):
        if key in self._cache:
            self.current_size -= self._cache[key]['size']
            del self._cache[key]

    def clear(self):
        self._cache.clear()
        self.current_size = 0

# Singleton instance with enhanced features
cache = EnhancedInMemoryCache(max_size_mb=500, ttl_seconds=3600)