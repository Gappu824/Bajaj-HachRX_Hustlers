# app/core/cache.py - Enhanced hybrid memory and disk cache
import logging
import time
import sys
import pickle
import lz4.frame
import asyncio
from typing import Dict, Any, Optional, Union
from collections import OrderedDict
import diskcache
import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import msgpack

logger = logging.getLogger(__name__)

class HybridCache:
    """Enhanced hybrid cache using memory for small items and disk for large items"""
    
    def __init__(self, cache_dir: str = ".cache", memory_size_mb: int = 500, disk_size_mb: int = 2000):
        # Memory cache for small, frequently accessed items
        self.memory_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.memory_size_bytes = memory_size_mb * 1024 * 1024
        self.current_memory_size = 0
        self._memory_lock = threading.RLock()
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Enhanced disk cache configuration
        self.disk_cache = diskcache.Cache(
            cache_dir,
            size_limit=disk_size_mb * 1024 * 1024,
            eviction_policy='least-recently-used',
            cull_limit=0,  # Manual management for better control
            statistics=True  # Enable statistics tracking
        )
        
        # Thread pool for async disk operations
        self._thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cache_worker")
        
        # Enhanced statistics
        self.hits = 0
        self.misses = 0
        self.memory_hits = 0
        self.disk_hits = 0
        self.write_errors = 0
        self.read_errors = 0
        
        # Performance tracking
        self.total_read_time = 0.0
        self.total_write_time = 0.0
        self.read_count = 0
        self.write_count = 0
        
        logger.info(f"✅ Initialized Enhanced Hybrid Cache (Memory: {memory_size_mb}MB, Disk: {disk_size_mb}MB)")
    
    def _get_item_size(self, value: Any) -> int:
        """Get approximate size of an item with better estimation"""
        try:
            # Try msgpack first (faster and more accurate for our data types)
            if isinstance(value, (dict, list, str, int, float, bool)):
                return len(msgpack.packb(value))
            else:
                # Fallback to pickle for complex objects
                return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            # Ultimate fallback
            return sys.getsizeof(value)
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value with compression for large items"""
        try:
            # Use msgpack for simple types (faster)
            if isinstance(value, (dict, list, str, int, float, bool, type(None))):
                serialized = msgpack.packb(value)
            else:
                # Use pickle for complex objects
                serialized = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Compress if larger than 1KB
            if len(serialized) > 1024:
                return lz4.frame.compress(serialized)
            return serialized
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value with decompression"""
        try:
            # Try to decompress first
            try:
                decompressed = lz4.frame.decompress(data)
            except:
                decompressed = data
            
            # Try msgpack first
            try:
                return msgpack.unpackb(decompressed, raw=False)
            except:
                # Fallback to pickle
                return pickle.loads(decompressed)
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache (memory first, then disk) with performance tracking"""
        start_time = time.time()
        
        try:
            # Check memory cache first
            with self._memory_lock:
                if key in self.memory_cache:
                    self.memory_cache.move_to_end(key)  # LRU update
                    value = self.memory_cache[key]['value']
                    
                    self.hits += 1
                    self.memory_hits += 1
                    self.total_read_time += time.time() - start_time
                    self.read_count += 1
                    return value
            
            # Check disk cache asynchronously
            loop = asyncio.get_event_loop()
            value = await loop.run_in_executor(
                self._thread_pool, 
                self._get_from_disk, 
                key
            )
            
            if value is not None:
                self.hits += 1
                self.disk_hits += 1
                
                # Promote to memory if small enough
                size = self._get_item_size(value)
                if size < 1024 * 1024:  # < 1MB
                    await self._add_to_memory(key, value, size)
                
                self.total_read_time += time.time() - start_time
                self.read_count += 1
                return value
            
            self.misses += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key '{key}': {e}")
            self.read_errors += 1
            self.misses += 1
            return None
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Synchronous disk get operation"""
        try:
            return self.disk_cache.get(key)
        except Exception as e:
            logger.warning(f"Disk cache read error: {e}")
            return None
    
    async def _add_to_memory(self, key: str, value: Any, size: int):
        """Add item to memory cache with intelligent LRU eviction"""
        with self._memory_lock:
            # Evict items if needed to make room
            while (self.current_memory_size + size > self.memory_size_bytes 
                   and self.memory_cache):
                oldest_key, oldest_val = self.memory_cache.popitem(last=False)
                self.current_memory_size -= oldest_val['size']
            
            # Add to memory if there's still room
            if self.current_memory_size + size <= self.memory_size_bytes:
                self.memory_cache[key] = {
                    'value': value, 
                    'size': size,
                    'timestamp': time.time()
                }
                self.current_memory_size += size
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set item in appropriate cache based on size with performance tracking"""
        start_time = time.time()
        
        try:
            size = self._get_item_size(value)
            
            # Strategy: Small items go to memory, large items to disk
            if size < 2 * 1024 * 1024:  # < 2MB goes to memory first
                await self._add_to_memory(key, value, size)
                
                # Also store in disk for persistence if it's important data
                if size > 100 * 1024:  # > 100KB also goes to disk
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        self._thread_pool,
                        self._set_to_disk,
                        key, value, ttl
                    )
            else:
                # Large items go directly to disk
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self._thread_pool,
                    self._set_to_disk,
                    key, value, ttl
                )
            
            self.total_write_time += time.time() - start_time
            self.write_count += 1
            
        except Exception as e:
            logger.error(f"Cache set error for key '{key}': {e}")
            self.write_errors += 1
    
    def _set_to_disk(self, key: str, value: Any, ttl: int):
        """Synchronous disk set operation"""
        try:
            self.disk_cache.set(key, value, expire=ttl)
        except Exception as e:
            logger.error(f"Disk cache write error: {e}")
            raise
    
    async def delete(self, key: str):
        """Delete from both caches"""
        try:
            # Remove from memory
            with self._memory_lock:
                if key in self.memory_cache:
                    val = self.memory_cache.pop(key)
                    self.current_memory_size -= val['size']
            
            # Remove from disk asynchronously
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self._thread_pool,
                self._delete_from_disk,
                key
            )
        except Exception as e:
            logger.error(f"Cache delete error for key '{key}': {e}")
    
    def _delete_from_disk(self, key: str):
        """Synchronous disk delete operation"""
        try:
            self.disk_cache.delete(key)
        except Exception:
            pass  # Ignore errors for delete operations
    
    def clear(self):
        """Clear both caches"""
        try:
            with self._memory_lock:
                self.memory_cache.clear()
                self.current_memory_size = 0
            
            self.disk_cache.clear()
            
            # Reset statistics
            self.hits = 0
            self.misses = 0
            self.memory_hits = 0
            self.disk_hits = 0
            self.write_errors = 0
            self.read_errors = 0
            self.total_read_time = 0.0
            self.total_write_time = 0.0
            self.read_count = 0
            self.write_count = 0
            
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        try:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            memory_hit_rate = self.memory_hits / total_requests if total_requests > 0 else 0
            disk_hit_rate = self.disk_hits / total_requests if total_requests > 0 else 0
            
            avg_read_time = self.total_read_time / self.read_count if self.read_count > 0 else 0
            avg_write_time = self.total_write_time / self.write_count if self.write_count > 0 else 0
            
            with self._memory_lock:
                memory_entries = len(self.memory_cache)
                memory_size_mb = self.current_memory_size / 1024 / 1024
            
            try:
                disk_entries = len(self.disk_cache)
                disk_size_mb = self.disk_cache.volume() / 1024 / 1024
            except:
                disk_entries = 0
                disk_size_mb = 0
            
            return {
                "memory": {
                    "entries": memory_entries,
                    "size_mb": round(memory_size_mb, 2),
                    "max_size_mb": self.memory_size_bytes / 1024 / 1024,
                    "utilization": round(memory_size_mb / (self.memory_size_bytes / 1024 / 1024) * 100, 1)
                },
                "disk": {
                    "entries": disk_entries,
                    "size_mb": round(disk_size_mb, 2)
                },
                "performance": {
                    "hit_rate": round(hit_rate * 100, 1),
                    "memory_hit_rate": round(memory_hit_rate * 100, 1),
                    "disk_hit_rate": round(disk_hit_rate * 100, 1),
                    "total_requests": total_requests,
                    "avg_read_time_ms": round(avg_read_time * 1000, 2),
                    "avg_write_time_ms": round(avg_write_time * 1000, 2)
                },
                "errors": {
                    "read_errors": self.read_errors,
                    "write_errors": self.write_errors
                }
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self._thread_pool.shutdown(wait=True)
            self.disk_cache.close()
        except Exception as e:
            logger.error(f"Cache cleanup error: {e}")
    
    def evict_expired(self):
        """Manually evict expired items from memory cache"""
        current_time = time.time()
        expired_keys = []
        
        with self._memory_lock:
            for key, data in self.memory_cache.items():
                # Remove items older than 1 hour from memory
                if current_time - data.get('timestamp', 0) > 3600:
                    expired_keys.append(key)
            
            for key in expired_keys:
                val = self.memory_cache.pop(key, None)
                if val:
                    self.current_memory_size -= val['size']
        
        if expired_keys:
            logger.info(f"Evicted {len(expired_keys)} expired items from memory cache")

# Create singleton with enhanced configuration
def create_cache():
    """Create cache instance with proper error handling"""
    try:
        from app.core.config import settings
        cache_instance = HybridCache(
            cache_dir=".cache",
            memory_size_mb=getattr(settings, 'CACHE_SIZE_MB', 1000) // 2,
            disk_size_mb=getattr(settings, 'CACHE_SIZE_MB', 1000) * 2
        )
        return cache_instance
    except Exception as e:
        logger.error(f"Failed to create cache with settings, using defaults: {e}")
        return HybridCache(memory_size_mb=250, disk_size_mb=1000)

cache = create_cache()