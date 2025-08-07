# app/core/cache.py - Hybrid memory and disk cache
import logging
import time
import sys
import pickle
import lz4.frame
from typing import Dict, Any, Optional
from collections import OrderedDict
import diskcache
import hashlib
import os
from app.core.config import settings
from typing import Dict, Any, Optional, List
import asyncio

logger = logging.getLogger(__name__)

class HybridCache:
    """Hybrid cache using memory for small items and disk for large items"""
    
    def __init__(self, cache_dir: str = ".cache", memory_size_mb: int = 500, disk_size_mb: int = 2000):
        # Memory cache for small, frequently accessed items
        self.memory_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.memory_size_bytes = memory_size_mb * 1024 * 1024
        self.current_memory_size = 0
        
        # Disk cache for large items
        self.disk_cache = diskcache.Cache(
            cache_dir,
            size_limit=disk_size_mb * 1024 * 1024,
            eviction_policy='least-recently-used'
        )
        
        # Stats
        self.hits = 0
        self.misses = 0
        
        logger.info(f"Initialized Hybrid Cache (Memory: {memory_size_mb}MB, Disk: {disk_size_mb}MB)")

    # async def get_many(self, keys: List[str]) -> Dict[str, Any]:
    #     """Batch get multiple keys"""
    #     results = {}
        
    #     # Check memory cache first
    #     for key in keys:
    #         if key in self.memory_cache:
    #             self.memory_cache.move_to_end(key)  # LRU update
    #             results[key] = self.memory_cache[key]['value']
    #             self.hits += 1
        
    #     # Check disk cache for missing keys
    #     missing_keys = [k for k in keys if k not in results]
    #     if missing_keys and self.disk_cache:
    #         try:
    #             # DiskCache doesn't have get_many, use individual gets
    #             for key in missing_keys:
    #                 value = self.disk_cache.get(key)
    #                 if value is not None:
    #                     results[key] = value
    #                     self.hits += 1
                        
    #                     # Promote small items to memory
    #                     size = self._get_item_size(value)
    #                     if size < 1024 * 1024:  # < 1MB
    #                         await self._add_to_memory(key, value, size)
    #         except Exception as e:
    #             logger.warning(f"Batch disk cache error: {e}")
        
    #     # Track misses
    #     self.misses += len(keys) - len(results)
        
    #     return results
    # In HybridCache class in cache.py

# --- REPLACE THIS METHOD ---
    # async def get_many(self, keys: List[str]) -> Dict[str, Any]:
    #     """Efficiently get multiple keys from memory and then disk."""
    #     results = {}
        
    #     # First, check the in-memory cache
    #     missing_keys = []
    #     for key in keys:
    #         if key in self.memory_cache:
    #             self.memory_cache.move_to_end(key)
    #             results[key] = self.memory_cache[key]['value']
    #             self.hits += 1
    #         else:
    #             missing_keys.append(key)

    #     # Then, check the disk cache for any remaining keys
    #     if missing_keys:
    #         try:
    #             # diskcache does not have a native async get_many, so we run it in a thread
    #             disk_results = await asyncio.to_thread(self.disk_cache.get, missing_keys, default=None)
                
    #             items_to_promote = {}
    #             for i, key in enumerate(missing_keys):
    #                 value = disk_results[i]
    #                 if value is not None:
    #                     results[key] = value
    #                     self.hits += 1
    #                     # Check if the item is small enough to promote to the memory cache
    #                     if self._get_item_size(value) < 1024 * 1024: # 1MB threshold
    #                         items_to_promote[key] = value

    #             # Promote items to memory cache
    #             for key, value in items_to_promote.items():
    #                 await self._add_to_memory(key, value, self._get_item_size(value))

    #         except Exception as e:
    #             logger.warning(f"Batch disk cache get error: {e}")
                
    #     self.misses += len(missing_keys)
    #     return results
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Batch get multiple keys"""
        results = {}
        
        # Check memory cache first
        for key in keys:
            if key in self.memory_cache:
                self.memory_cache.move_to_end(key)  # LRU update
                results[key] = self.memory_cache[key]['value']
                self.hits += 1
        
        # Check disk cache for missing keys
        missing_keys = [k for k in keys if k not in results]
        if missing_keys and self.disk_cache:
            try:
                # DiskCache doesn't have get_many, use individual gets
                for key in missing_keys:
                    value = self.disk_cache.get(key)
                    if value is not None:
                        results[key] = value
                        self.hits += 1
                        
                        # Promote small items to memory
                        size = self._get_item_size(value)
                        if size < 1024 * 1024:  # < 1MB
                            await self._add_to_memory(key, value, size)
            except Exception as e:
                logger.warning(f"Batch disk cache error: {e}")
        
        # Track misses
        self.misses += len(keys) - len(results)
        
        return results
    
    async def set_many(self, items: Dict[str, Any], ttl: int = 3600):
        """Batch set multiple key-value pairs"""
        for key, value in items.items():
            await self.set(key, value, ttl)    
    # async def get_many(self, keys: List[str]) -> Dict[str, Any]:
    #     """Batch get multiple keys"""
    #     # NEW: Efficient batch retrieval
        
    #     results = {}
        
    #     # Check memory cache first
    #     for key in keys:
    #         if key in self.memory_cache:
    #             self.memory_cache.move_to_end(key)  # LRU update
    #             results[key] = self.memory_cache[key]['value']
    #             self.hits += 1
        
    #     # Check disk cache for missing keys
    #     missing_keys = [k for k in keys if k not in results]
    #     if missing_keys and self.disk_cache:
    #         try:
    #             # DiskCache supports batch get
    #             disk_results = self.disk_cache.get_many(missing_keys)
    #             for key, value in disk_results.items():
    #                 if value is not None:
    #                     results[key] = value
    #                     self.hits += 1
                        
    #                     # Promote small items to memory
    #                     size = self._get_item_size(value)
    #                     if size < 1024 * 1024:  # < 1MB
    #                         await self._add_to_memory(key, value, size)
    #         except Exception as e:
    #             logger.warning(f"Batch disk cache error: {e}")
    # async def set_many(self, items: Dict[str, Any], ttl: int = 3600):
    #     """Batch set multiple key-value pairs"""
    #     # NEW: Efficient batch setting
        
    #     for key, value in items.items():
    #         await self.set(key, value, ttl)            
        
    #     # Track misses
    #     self.misses += len(keys) - len(results)
        
    #     return results    
    
    def _get_item_size(self, value: Any) -> int:
        """Get approximate size of an item"""
        try:
            return len(pickle.dumps(value))
        except:
            return sys.getsizeof(value)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache (memory first, then disk)"""
        # Check memory cache
        if key in self.memory_cache:
            self.memory_cache.move_to_end(key)  # LRU
            self.hits += 1
            return self.memory_cache[key]['value']
        
        # Check disk cache
        try:
            value = self.disk_cache.get(key)
            if value is not None:
                self.hits += 1
                # Promote to memory if small enough
                size = self._get_item_size(value)
                if size < 1024 * 1024:  # < 1MB
                    await self._add_to_memory(key, value, size)
                return value
        except Exception as e:
            logger.warning(f"Disk cache error: {e}")
        
        self.misses += 1
        return None
    
    async def _add_to_memory(self, key: str, value: Any, size: int):
        """Add item to memory cache with LRU eviction"""
        # Evict if needed
        while self.current_memory_size + size > self.memory_size_bytes and self.memory_cache:
            oldest_key, oldest_val = self.memory_cache.popitem(last=False)
            self.current_memory_size -= self._get_item_size(oldest_val['value'])
        
        # Add to memory
        self.memory_cache[key] = {'value': value, 'size': size}
        self.current_memory_size += size
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set item in appropriate cache based on size"""
        size = self._get_item_size(value)
        
        if size < 1024 * 1024:  # < 1MB goes to memory
            await self._add_to_memory(key, value, size)
        else:  # Large items go to disk
            try:
                self.disk_cache.set(key, value, expire=ttl)
            except Exception as e:
                logger.error(f"Failed to cache to disk: {e}")
    
    async def delete(self, key: str):
        """Delete from both caches"""
        if key in self.memory_cache:
            val = self.memory_cache.pop(key)
            self.current_memory_size -= val['size']
        
        try:
            self.disk_cache.delete(key)
        except:
            pass
    
    # def clear(self):
    #     """Clear both caches"""
    #     self.memory_cache.clear()
    #     self.current_memory_size = 0
    #     self.disk_cache.clear()
    #     self.hits = 0
    #     self.misses = 0
    # def clear(self):
    #     """Clear memory and disk cache"""
    #     # OLD: No cleanup
    #     # NEW: Proper cleanup
        
    #     self.small_chunks.clear()
    #     self.large_chunks.clear()
    #     self.chunk_metadata.clear()
    #     self.chunk_to_large_mapping.clear()
        
    #     # Clean disk cache
    #     if os.path.exists(self.disk_cache_dir):
    #         shutil.rmtree(self.disk_cache_dir)
        
    #     logger.info("Vector store cleared")
    def clear(self):
        """Correctly clear the memory and disk caches."""
        # Clears the in-memory cache
        self.memory_cache.clear()
        self.current_memory_size = 0
        
        # Clears the on-disk cache provided by the diskcache library
        self.disk_cache.clear()
        
        self.hits = 0
        self.misses = 0
        logger.info("HybridCache cleared successfully.")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "memory_entries": len(self.memory_cache),
            "memory_size_mb": self.current_memory_size / 1024 / 1024,
            "disk_entries": len(self.disk_cache),
            "disk_size_mb": self.disk_cache.volume() / 1024 / 1024,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }

# Create singleton

cache = HybridCache(
    memory_size_mb=settings.CACHE_SIZE_MB // 2,
    disk_size_mb=settings.CACHE_SIZE_MB * 2
)