# app/core/performance_monitor.py - Create this new file

import time
import psutil
import logging
from typing import Dict, Any, List
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor system performance and resource usage"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.request_times = deque(maxlen=max_history)
        self.memory_usage = deque(maxlen=max_history)
        self.start_time = time.time()
        self.total_requests = 0
        self.error_count = 0
        
    def record_request(self, duration: float, success: bool = True):
        """Record a request completion"""
        self.request_times.append(duration)
        self.total_requests += 1
        if not success:
            self.error_count += 1
        
        # Record memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)
        
        # Log if slow
        if duration > 30:
            logger.warning(f"Slow request: {duration:.2f}s")
        
        # Warn if high memory
        if memory_mb > 1500:
            logger.warning(f"High memory usage: {memory_mb:.2f}MB")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        uptime = time.time() - self.start_time
        
        stats = {
            'uptime_seconds': uptime,
            'total_requests': self.total_requests,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.total_requests, 1),
            'current_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.cpu_percent(interval=0.1),
        }
        
        if self.request_times:
            times_list = list(self.request_times)
            stats.update({
                'avg_request_time': sum(times_list) / len(times_list),
                'max_request_time': max(times_list),
                'min_request_time': min(times_list),
                'p95_request_time': sorted(times_list)[int(len(times_list) * 0.95)] if len(times_list) > 1 else times_list[0],
            })
        
        if self.memory_usage:
            stats.update({
                'avg_memory_mb': sum(self.memory_usage) / len(self.memory_usage),
                'max_memory_mb': max(self.memory_usage),
            })
        
        return stats