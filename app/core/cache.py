# app/core/cache.py - Enhanced caching with larger size
import logging
import time
import sys
import pickle
import lz4.frame
from typing import Dict, Any, Optional
from collections import OrderedDict

logger = logging.getLogger(__name__)

class EnhancedInMemoryCache:
    """Enhanced cache with compression and better serialization"""
    
    def __init__(self, max_size_mb: int = 2000, ttl_seconds: int = 7200):
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.current_size = 0
        self.hit_count = 0
        self.miss_count = 0
        logger.info(f"Initialized Enhanced Cache (max: {max_size_mb}MB, TTL: {ttl_seconds}s)")

    async def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            entry = self._cache[key]
            # Check TTL
            if time.time() - entry['timestamp'] > self.ttl_seconds:
                await self.delete(key)
                self.miss_count += 1
                return None
            
            # Move to end (LRU)
            self._cache.move_to_end(key)
            self.hit_count += 1
            
            # Decompress if needed
            if entry.get('compressed', False):
                try:
                    decompressed = lz4.frame.decompress(entry['value'])
                    return pickle.loads(decompressed)
                except Exception as e:
                    logger.error(f"Failed to decompress cache entry {key}: {e}")
                    await self.delete(key)
                    return None
            
            return entry['value']
        
        self.miss_count += 1
        return None

    async def set(self, key: str, value: Any):
        # Try to compress large objects
        try:
            serialized = pickle.dumps(value)
            size = len(serialized)
            
            if size > 1024 * 1024:  # Compress if > 1MB
                compressed = lz4.frame.compress(serialized)
                if len(compressed) < size * 0.8:  # Only use if 20% smaller
                    value_to_store = compressed
                    size = len(compressed)
                    is_compressed = True
                else:
                    value_to_store = value
                    is_compressed = False
            else:
                value_to_store = value
                is_compressed = False
        except Exception as e:
            logger.warning(f"Failed to serialize/compress value: {e}")
            # Fallback to uncompressed
            value_to_store = value
            size = sys.getsizeof(value)
            is_compressed = False
        
        # Evict old entries if needed
        while self.current_size + size > self.max_size_bytes and self._cache:
            oldest_key = next(iter(self._cache))
            await self.delete(oldest_key)
        
        # Update entry
        if key in self._cache:
            old_size = self._cache[key].get('size', 0)
            self.current_size -= old_size
        
        self._cache[key] = {
            'value': value_to_store,
            'timestamp': time.time(),
            'size': size,
            'compressed': is_compressed
        }
        self.current_size += size
        self._cache.move_to_end(key)
        
        logger.debug(f"Cached {key} (size: {size/1024:.1f}KB, compressed: {is_compressed})")

    async def delete(self, key: str):
        if key in self._cache:
            self.current_size -= self._cache[key].get('size', 0)
            del self._cache[key]

    def clear(self):
        self._cache.clear()
        self.current_size = 0
        self.hit_count = 0
        self.miss_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "entries": len(self._cache),
            "size_mb": self.current_size / 1024 / 1024,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "max_size_mb": self.max_size_bytes / 1024 / 1024,
            "total_requests": total_requests
        }

# Create singleton with settings from config
from app.core.config import settings
cache = EnhancedInMemoryCache(
    max_size_mb=settings.CACHE_SIZE_MB, 
    ttl_seconds=settings.CACHE_TTL_SECONDS
)