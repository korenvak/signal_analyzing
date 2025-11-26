import numpy as np
import threading
import gc
import time
from collections import OrderedDict
from typing import Any, Optional, Tuple, Dict
import zarr
import tempfile
import os

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

class TileKey:
    """Immutable key for tile identification."""
    def __init__(self, view_type: str, time_start: float, time_end: float, 
                 freq_start: float, freq_end: float, resolution_level: int = 0):
        self.view_type = view_type
        self.time_start = time_start
        self.time_end = time_end
        self.freq_start = freq_start
        self.freq_end = freq_end
        self.resolution_level = resolution_level
        
    def __hash__(self):
        return hash((self.view_type, self.time_start, self.time_end, 
                    self.freq_start, self.freq_end, self.resolution_level))
    
    def __eq__(self, other):
        if not isinstance(other, TileKey):
            return False
        return (self.view_type == other.view_type and
                self.time_start == other.time_start and
                self.time_end == other.time_end and
                self.freq_start == other.freq_start and
                self.freq_end == other.freq_end and
                self.resolution_level == other.resolution_level)
    
    def __str__(self):
        return (f"TileKey({self.view_type}, t:{self.time_start:.2f}-{self.time_end:.2f}, "
                f"f:{self.freq_start:.0f}-{self.freq_end:.0f}, res:{self.resolution_level})")

class CachedTile:
    """Container for cached tile data with metadata."""
    def __init__(self, data: np.ndarray, key: TileKey, gpu_data: Optional[Any] = None):
        self.data = data
        self.key = key
        self.gpu_data = gpu_data
        self.access_time = time.time()
        self.size_bytes = data.nbytes
        self.is_dirty = False
        
    def touch(self):
        """Update access time for LRU tracking."""
        self.access_time = time.time()
        
    def move_to_gpu(self):
        """Move data to GPU memory if available."""
        if HAS_CUPY and self.gpu_data is None:
            try:
                self.gpu_data = cp.asarray(self.data)
                return True
            except Exception as e:
                print(f"Failed to move tile to GPU: {e}")
        return False
        
    def move_to_cpu(self):
        """Move data back to CPU and free GPU memory."""
        if self.gpu_data is not None:
            try:
                if HAS_CUPY and isinstance(self.gpu_data, cp.ndarray):
                    del self.gpu_data
                    cp.get_default_memory_pool().free_all_blocks()
                self.gpu_data = None
                return True
            except Exception as e:
                print(f"Failed to free GPU memory: {e}")
        return False

class LRUCache:
    """High-performance LRU cache with GPU memory management."""
    
    def __init__(self, max_memory_mb: int = 2048, max_gpu_memory_mb: int = 1024):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_gpu_memory_bytes = max_gpu_memory_mb * 1024 * 1024
        
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._current_memory = 0
        self._current_gpu_memory = 0
        
        self._temp_dir = tempfile.mkdtemp(prefix='audio_viz_cache_')
        self._disk_arrays = {}
        
    def get(self, key: TileKey) -> Optional[CachedTile]:
        """Get tile from cache, promoting to most recently used."""
        with self._lock:
            if key in self._cache:
                tile = self._cache[key]
                tile.touch()
                self._cache.move_to_end(key)
                return tile
            return None
    
    def put(self, key: TileKey, data: np.ndarray, 
            move_to_gpu: bool = False) -> CachedTile:
        """Add tile to cache with optional GPU placement."""
        with self._lock:
            if key in self._cache:
                old_tile = self._cache[key]
                self._current_memory -= old_tile.size_bytes
                if old_tile.gpu_data is not None:
                    old_tile.move_to_cpu()
                del self._cache[key]
            
            tile = CachedTile(data.copy(), key)
            
            if move_to_gpu:
                if tile.move_to_gpu():
                    self._current_gpu_memory += tile.size_bytes
            
            self._current_memory += tile.size_bytes
            self._cache[key] = tile
            
            self._evict_if_needed()
            return tile
    
    def _evict_if_needed(self):
        """Evict least recently used tiles if memory limits exceeded."""
        while (self._current_memory > self.max_memory_bytes or 
               self._current_gpu_memory > self.max_gpu_memory_bytes):
            
            if not self._cache:
                break
                
            oldest_key = next(iter(self._cache))
            oldest_tile = self._cache[oldest_key]
            
            if oldest_tile.gpu_data is not None:
                oldest_tile.move_to_cpu()
                self._current_gpu_memory -= oldest_tile.size_bytes
                
            if self._current_memory > self.max_memory_bytes:
                self._move_to_disk(oldest_key, oldest_tile)
                self._current_memory -= oldest_tile.size_bytes
                del self._cache[oldest_key]
            else:
                break
    
    def _move_to_disk(self, key: TileKey, tile: CachedTile):
        """Move tile data to disk storage."""
        try:
            filename = f"tile_{hash(key):x}.zarr"
            filepath = os.path.join(self._temp_dir, filename)
            
            zarr_array = zarr.open(filepath, mode='w', 
                                 shape=tile.data.shape, 
                                 dtype=tile.data.dtype,
                                 compression='blosc')
            zarr_array[:] = tile.data
            
            self._disk_arrays[key] = filepath
            
        except Exception as e:
            print(f"Failed to move tile to disk: {e}")
    
    def promote_to_gpu(self, key: TileKey) -> bool:
        """Promote tile to GPU memory if available."""
        with self._lock:
            if key in self._cache:
                tile = self._cache[key]
                if tile.gpu_data is None:
                    success = tile.move_to_gpu()
                    if success:
                        self._current_gpu_memory += tile.size_bytes
                        self._evict_if_needed()
                    return success
                return True
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        with self._lock:
            gpu_tiles = sum(1 for tile in self._cache.values() 
                          if tile.gpu_data is not None)
            
            return {
                'cpu_memory_mb': self._current_memory / (1024 * 1024),
                'gpu_memory_mb': self._current_gpu_memory / (1024 * 1024),
                'max_cpu_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'max_gpu_memory_mb': self.max_gpu_memory_bytes / (1024 * 1024),
                'total_tiles': len(self._cache),
                'gpu_tiles': gpu_tiles,
                'disk_tiles': len(self._disk_arrays)
            }
    
    def clear(self):
        """Clear all cached data."""
        with self._lock:
            for tile in self._cache.values():
                tile.move_to_cpu()
            
            self._cache.clear()
            self._current_memory = 0
            self._current_gpu_memory = 0
            
            for filepath in self._disk_arrays.values():
                try:
                    if os.path.exists(filepath):
                        os.remove(filepath)
                except OSError:
                    pass  # Ignore file removal errors during cleanup
            self._disk_arrays.clear()
            
            gc.collect()
            if HAS_CUPY:
                cp.get_default_memory_pool().free_all_blocks()
    
    def __del__(self):
        """Cleanup on destruction."""
        self.clear()
        try:
            if os.path.exists(self._temp_dir):
                os.rmdir(self._temp_dir)
        except OSError:
            pass  # Ignore directory removal errors during cleanup

class CacheManager:
    """High-level cache manager for multiple view types."""
    
    def __init__(self, max_memory_mb: int = 2048, max_gpu_memory_mb: int = 1024):
        self.cache = LRUCache(max_memory_mb, max_gpu_memory_mb)
        self._locks = {}
        
    def get_tile(self, view_type: str, time_range: Tuple[float, float],
                 freq_range: Tuple[float, float], 
                 resolution_level: int = 0) -> Optional[np.ndarray]:
        """Get tile data for specified parameters."""
        key = TileKey(view_type, time_range[0], time_range[1],
                     freq_range[0], freq_range[1], resolution_level)
        
        cached_tile = self.cache.get(key)
        if cached_tile:
            return cached_tile.data
        return None
    
    def store_tile(self, view_type: str, time_range: Tuple[float, float],
                   freq_range: Tuple[float, float], data: np.ndarray,
                   resolution_level: int = 0, use_gpu: bool = False) -> bool:
        """Store tile data in cache."""
        key = TileKey(view_type, time_range[0], time_range[1],
                     freq_range[0], freq_range[1], resolution_level)
        
        try:
            self.cache.put(key, data, move_to_gpu=use_gpu)
            return True
        except Exception as e:
            print(f"Failed to store tile: {e}")
            return False
    
    def prefetch_gpu(self, view_type: str, time_range: Tuple[float, float],
                     freq_range: Tuple[float, float], 
                     resolution_level: int = 0) -> bool:
        """Prefetch tile to GPU memory."""
        key = TileKey(view_type, time_range[0], time_range[1],
                     freq_range[0], freq_range[1], resolution_level)
        return self.cache.promote_to_gpu(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_memory_stats()
    
    def clear(self):
        """Clear all cached data."""
        self.cache.clear()
    
    def clear_view_cache(self, view_type: str):
        """Clear cache for specific view type."""
        keys_to_remove = []
        for key in self.cache._cache.keys():
            if key.view_type == view_type:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            with self.cache._lock:
                if key in self.cache._cache:
                    tile = self.cache._cache[key]
                    tile.move_to_cpu()
                    self.cache._current_memory -= tile.size_bytes
                    del self.cache._cache[key]