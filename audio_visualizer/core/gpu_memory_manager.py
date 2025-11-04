"""
GPU Memory Manager
Monitors and manages GPU VRAM usage with cleanup utilities
"""

import gc
import time
from typing import Dict, Optional
import logging

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """Manages GPU memory allocation, monitoring, and cleanup."""
    
    def __init__(self):
        self.enabled = HAS_CUPY
        if self.enabled:
            # Use memory pool for efficient allocation
            self.mempool = cp.get_default_memory_pool()
            self.pinned_mempool = cp.get_default_pinned_memory_pool()
        else:
            self.mempool = None
            self.pinned_mempool = None
        
        self.allocation_history = []
        self.peak_usage = 0
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage information."""
        if not self.enabled:
            return {
                'used_bytes': 0,
                'total_bytes': 0,
                'free_bytes': 0,
                'used_mb': 0,
                'total_mb': 0,
                'free_mb': 0,
                'utilization': 0.0
            }
        
        # Get memory info from CuPy
        mempool_used = self.mempool.used_bytes()
        mempool_total = self.mempool.total_bytes()
        
        # Get device memory info
        device = cp.cuda.Device()
        total_bytes = device.mem_info[1]  # Total memory
        free_bytes = device.mem_info[0]   # Free memory
        used_bytes = total_bytes - free_bytes
        
        # Update peak usage
        self.peak_usage = max(self.peak_usage, used_bytes)
        
        return {
            'used_bytes': used_bytes,
            'total_bytes': total_bytes,
            'free_bytes': free_bytes,
            'used_mb': used_bytes / (1024**2),
            'total_mb': total_bytes / (1024**2),
            'free_mb': free_bytes / (1024**2),
            'utilization': (used_bytes / total_bytes * 100) if total_bytes > 0 else 0,
            'mempool_used_mb': mempool_used / (1024**2),
            'mempool_total_mb': mempool_total / (1024**2),
            'peak_mb': self.peak_usage / (1024**2)
        }
    
    def cleanup(self, aggressive: bool = False):
        """Clean up GPU memory.
        
        Args:
            aggressive: If True, force free all cached memory
        """
        if not self.enabled:
            return
        
        logger.info("Cleaning up GPU memory...")
        
        # Synchronize GPU to ensure all operations are complete
        cp.cuda.Stream.null.synchronize()
        
        # Force Python garbage collection
        gc.collect()
        
        if aggressive:
            # Free all cached memory in the pool
            self.mempool.free_all_blocks()
            self.pinned_mempool.free_all_blocks()
            logger.info("Aggressive cleanup: freed all cached blocks")
        else:
            # Just free unused blocks
            self.mempool.free_all_free()
            self.pinned_mempool.free_all_free()
            logger.info("Normal cleanup: freed unused blocks")
        
        # Log memory status
        mem_info = self.get_memory_info()
        logger.info(f"GPU Memory after cleanup: {mem_info['used_mb']:.1f} MB used / "
                   f"{mem_info['total_mb']:.1f} MB total ({mem_info['utilization']:.1f}% used)")
    
    def check_available(self, required_bytes: int) -> bool:
        """Check if enough GPU memory is available.
        
        Args:
            required_bytes: Number of bytes required
            
        Returns:
            True if enough memory is available
        """
        if not self.enabled:
            return False
        
        mem_info = self.get_memory_info()
        available = mem_info['free_bytes']
        
        # Keep 10% buffer
        safe_available = available * 0.9
        
        return safe_available >= required_bytes
    
    def log_allocation(self, name: str, size_bytes: int):
        """Log a memory allocation for tracking.
        
        Args:
            name: Name/description of allocation
            size_bytes: Size in bytes
        """
        self.allocation_history.append({
            'name': name,
            'size_bytes': size_bytes,
            'size_mb': size_bytes / (1024**2),
            'timestamp': time.time()
        })
        
        # Keep only last 100 allocations
        if len(self.allocation_history) > 100:
            self.allocation_history.pop(0)
    
    def get_allocation_summary(self) -> str:
        """Get a summary of recent allocations."""
        if not self.allocation_history:
            return "No allocations recorded"
        
        recent = self.allocation_history[-10:]
        lines = ["Recent GPU allocations:"]
        for alloc in recent:
            lines.append(f"  - {alloc['name']}: {alloc['size_mb']:.1f} MB")
        
        total_mb = sum(a['size_mb'] for a in self.allocation_history)
        lines.append(f"Total tracked: {total_mb:.1f} MB")
        
        return "\n".join(lines)
    
    def reset_peak(self):
        """Reset peak usage tracking."""
        self.peak_usage = 0
    
    def get_device_info(self) -> Dict[str, any]:
        """Get GPU device information."""
        if not self.enabled:
            return {'available': False}
        
        device = cp.cuda.Device()
        props = device.attributes
        
        return {
            'available': True,
            'name': device.name,
            'compute_capability': f"{device.compute_capability[0]}.{device.compute_capability[1]}",
            'total_memory_mb': device.mem_info[1] / (1024**2),
            'multiprocessors': props.get('MultiProcessorCount', 'Unknown'),
            'max_threads_per_block': props.get('MaxThreadsPerBlock', 'Unknown'),
            'warp_size': props.get('WarpSize', 32)
        }


# Global instance
_gpu_memory_manager = None


def get_gpu_memory_manager() -> GPUMemoryManager:
    """Get the global GPU memory manager instance."""
    global _gpu_memory_manager
    if _gpu_memory_manager is None:
        _gpu_memory_manager = GPUMemoryManager()
    return _gpu_memory_manager

