"""
File Switch Manager - Aggressive Memory Cleanup Between Files
Provides maximum performance by ensuring clean state between file loads.
"""

import gc
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


class FileSwitchManager:
    """Manages aggressive cleanup when switching between audio files."""
    
    def __init__(self):
        self.enabled = True
        self.cleanup_stats = {
            'files_switched': 0,
            'memory_freed_mb': 0.0,
            'gpu_memory_freed_mb': 0.0
        }
    
    def cleanup_for_new_file(self, cache_manager=None, gpu_memory_manager=None, 
                            memory_pools=None, tile_cache=None, engines=None):
        """Perform aggressive cleanup before loading a new file.
        
        This ensures maximum available memory and prevents accumulation
        of data from previous files.
        
        Args:
            cache_manager: Main cache manager instance
            gpu_memory_manager: GPU memory manager instance
            memory_pools: Memory pool optimizer instance
            tile_cache: Tile cache instance
            engines: Dict of analysis engines to cleanup
        """
        if not self.enabled:
            return
        
        initial_memory = self._get_memory_info()
        logger.info("Starting aggressive cleanup for new file...")
        
        try:
            # 1. Clear all analysis engine caches
            if engines:
                self._cleanup_engines(engines)
            
            # 2. Clear main cache manager
            if cache_manager:
                self._cleanup_cache_manager(cache_manager)
            
            # 3. Clear tile cache completely
            if tile_cache:
                self._cleanup_tile_cache(tile_cache)
            
            # 4. Reset memory pools
            if memory_pools:
                self._cleanup_memory_pools(memory_pools)
            
            # 5. Aggressive GPU cleanup
            if gpu_memory_manager:
                self._cleanup_gpu_memory(gpu_memory_manager)
            
            # 6. Force Python garbage collection
            self._force_garbage_collection()
            
            # Update statistics
            final_memory = self._get_memory_info()
            self._update_cleanup_stats(initial_memory, final_memory)
            
            logger.info("File switch cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during file switch cleanup: {e}")
    
    def _cleanup_engines(self, engines: Dict[str, Any]):
        """Clean up all analysis engines."""
        logger.debug("Cleaning up analysis engines...")
        
        for engine_name, engine in engines.items():
            try:
                if hasattr(engine, 'cleanup'):
                    engine.cleanup()
                
                # Clear engine-specific caches
                if hasattr(engine, '_window_cache'):
                    engine._window_cache.clear()
                if hasattr(engine, '_mel_filterbank'):
                    engine._mel_filterbank = None
                if hasattr(engine, '_steering_vectors'):
                    engine._steering_vectors.clear()
                if hasattr(engine, 'plans'):
                    engine.plans.clear()
                
                logger.debug(f"Cleaned up {engine_name} engine")
                
            except Exception as e:
                logger.warning(f"Error cleaning up {engine_name} engine: {e}")
    
    def _cleanup_cache_manager(self, cache_manager):
        """Clear all cache manager data."""
        logger.debug("Clearing cache manager...")
        
        try:
            # Clear all view caches
            cache_manager.clear_all_caches()
            
            # Force eviction of any remaining items
            if hasattr(cache_manager, 'lru_cache'):
                cache_manager.lru_cache.clear()
            
            logger.debug("Cache manager cleared")
            
        except Exception as e:
            logger.warning(f"Error clearing cache manager: {e}")
    
    def _cleanup_tile_cache(self, tile_cache):
        """Clear tile cache completely."""
        logger.debug("Clearing tile cache...")
        
        try:
            # Clear memory cache
            tile_cache.clear()
            
            # Force disk cache cleanup if needed
            if hasattr(tile_cache, 'clear_disk_cache'):
                tile_cache.clear_disk_cache()
            
            logger.debug("Tile cache cleared")
            
        except Exception as e:
            logger.warning(f"Error clearing tile cache: {e}")
    
    def _cleanup_memory_pools(self, memory_pools):
        """Reset memory pools to free all workspace buffers."""
        logger.debug("Resetting memory pools...")
        
        try:
            # Clear all workspace pools
            memory_pools.cleanup()
            
            # Force reset of pool statistics
            if hasattr(memory_pools, 'reset_stats'):
                memory_pools.reset_stats()
            
            logger.debug("Memory pools reset")
            
        except Exception as e:
            logger.warning(f"Error resetting memory pools: {e}")
    
    def _cleanup_gpu_memory(self, gpu_memory_manager):
        """Aggressive GPU memory cleanup."""
        logger.debug("Performing aggressive GPU cleanup...")
        
        try:
            # Standard cleanup
            gpu_memory_manager.cleanup(aggressive=True)
            
            # Additional CuPy cleanup if available
            if HAS_CUPY:
                # Free all blocks in memory pool
                mempool = cp.get_default_memory_pool()
                pinned_mempool = cp.get_default_pinned_memory_pool()
                
                # Get current usage before cleanup
                used_before = mempool.used_bytes()
                
                # Free all cached memory
                mempool.free_all_blocks()
                pinned_mempool.free_all_blocks()
                
                # Get usage after cleanup
                used_after = mempool.used_bytes()
                freed_mb = (used_before - used_after) / (1024**2)
                
                logger.debug(f"GPU memory freed: {freed_mb:.1f} MB")
            
        except Exception as e:
            logger.warning(f"Error during GPU cleanup: {e}")
    
    def _force_garbage_collection(self):
        """Force Python garbage collection."""
        logger.debug("Forcing garbage collection...")
        
        try:
            # Multiple passes to ensure cleanup of circular references
            for i in range(3):
                collected = gc.collect()
                if collected == 0:
                    break
                logger.debug(f"GC pass {i+1}: collected {collected} objects")
            
            # Additional GPU-specific garbage collection
            if HAS_CUPY:
                # Force CuPy to release any remaining references
                cp._default_memory_pool.free_all_blocks()
                cp._default_pinned_memory_pool.free_all_blocks()
            
        except Exception as e:
            logger.debug(f"Error during garbage collection: {e}")
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information."""
        info = {
            'cpu_memory_mb': 0.0,
            'gpu_memory_mb': 0.0
        }
        
        try:
            # Get CPU memory info (simplified)
            import psutil
            process = psutil.Process()
            info['cpu_memory_mb'] = process.memory_info().rss / (1024**2)
        except ImportError:
            pass
        
        try:
            # Get GPU memory info
            if HAS_CUPY:
                device = cp.cuda.Device()
                free_bytes, total_bytes = device.mem_info
                used_bytes = total_bytes - free_bytes
                info['gpu_memory_mb'] = used_bytes / (1024**2)
        except Exception:
            pass
        
        return info
    
    def _update_cleanup_stats(self, before: Dict[str, float], after: Dict[str, float]):
        """Update cleanup statistics."""
        try:
            cpu_freed = before.get('cpu_memory_mb', 0) - after.get('cpu_memory_mb', 0)
            gpu_freed = before.get('gpu_memory_mb', 0) - after.get('gpu_memory_mb', 0)
            
            self.cleanup_stats['files_switched'] += 1
            self.cleanup_stats['memory_freed_mb'] += max(0, cpu_freed)
            self.cleanup_stats['gpu_memory_freed_mb'] += max(0, gpu_freed)
            
            logger.info(
                f"Cleanup completed - CPU: {cpu_freed:.1f}MB freed, "
                f"GPU: {gpu_freed:.1f}MB freed"
            )
            
        except Exception as e:
            logger.debug(f"Error updating cleanup stats: {e}")
    
    def get_cleanup_stats(self) -> Dict[str, float]:
        """Get cleanup statistics."""
        return self.cleanup_stats.copy()
    
    def enable_cleanup(self, enabled: bool = True):
        """Enable or disable file switch cleanup."""
        self.enabled = enabled
        logger.info(f"File switch cleanup {'enabled' if enabled else 'disabled'}")


# Global instance
_file_switch_manager = None

def get_file_switch_manager() -> FileSwitchManager:
    """Get the global file switch manager instance."""
    global _file_switch_manager
    if _file_switch_manager is None:
        _file_switch_manager = FileSwitchManager()
    return _file_switch_manager