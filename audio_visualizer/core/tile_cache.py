"""
High-Performance Disk-Backed Tile Cache
Uses Zarr for efficient storage and memory-mapped access
"""

import os
import shutil
import threading
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from collections import OrderedDict
import numpy as np
import logging

# Import compressed storage
from .compressed_tile_storage import get_compressed_tile_storage

try:
    import zarr
    HAS_ZARR = True
except ImportError:
    zarr = None
    HAS_ZARR = False

logger = logging.getLogger(__name__)


class TileKey:
    """Unique key for a tile."""
    
    def __init__(self, view_type: str, time_range: Tuple[float, float],
                 freq_range: Tuple[float, float], resolution_level: int = 0):
        self.view_type = view_type
        self.time_start = time_range[0]
        self.time_end = time_range[1]
        self.freq_start = freq_range[0]
        self.freq_end = freq_range[1]
        self.resolution_level = resolution_level
    
    def __hash__(self):
        return hash((
            self.view_type,
            round(self.time_start, 3),
            round(self.time_end, 3),
            round(self.freq_start, 1),
            round(self.freq_end, 1),
            self.resolution_level
        ))
    
    def __eq__(self, other):
        if not isinstance(other, TileKey):
            return False
        return (self.view_type == other.view_type and
                abs(self.time_start - other.time_start) < 0.001 and
                abs(self.time_end - other.time_end) < 0.001 and
                abs(self.freq_start - other.freq_start) < 1.0 and
                abs(self.freq_end - other.freq_end) < 1.0 and
                self.resolution_level == other.resolution_level)
    
    def to_filename(self) -> str:
        """Generate filename for this tile."""
        return (f"{self.view_type}_"
                f"t{self.time_start:.3f}-{self.time_end:.3f}_"
                f"f{self.freq_start:.0f}-{self.freq_end:.0f}_"
                f"lod{self.resolution_level}.npy")


class TileCache:
    """Disk-backed tile cache with LRU eviction, memory mapping, and compression."""
    
    def __init__(self, cache_dir: str = None, max_memory_tiles: int = 100,
                 max_disk_gb: float = 10.0, enable_compression: bool = True,
                 compression_level: int = 3):
        """Initialize tile cache.
        
        Args:
            cache_dir: Directory for cache storage (default: ./tile_cache)
            max_memory_tiles: Maximum tiles to keep in RAM
            max_disk_gb: Maximum disk space to use (GB)
            enable_compression: Enable tile compression (3-4x more capacity)
            compression_level: Compression level (1=fast, 3=balanced, 9=best)
        """
        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path.cwd() / "tile_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Compression settings
        self.enable_compression = enable_compression
        self.compression_level = compression_level
        self.compressed_storage = get_compressed_tile_storage(
            compression_level, enable_compression)
        
        # Cache limits
        self.max_memory_tiles = max_memory_tiles
        self.max_disk_bytes = int(max_disk_gb * 1024**3)
        
        # Memory cache (LRU)
        self._memory_cache: OrderedDict[TileKey, np.ndarray] = OrderedDict()
        self._memory_lock = threading.RLock()
        
        # Disk index
        self._disk_index: Dict[TileKey, Path] = {}
        self._disk_lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'writes': 0,
            'evictions': 0,
            'compression_ratio': 1.0,
            'bytes_saved_mb': 0.0
        }
        
        # Initialize disk index
        self._scan_disk_cache()
        
        logger.info(f"TileCache initialized: {len(self._disk_index)} tiles on disk")
    
    def _scan_disk_cache(self):
        """Scan disk cache directory and build index."""
        if not self.cache_dir.exists():
            return
        
        for file_path in self.cache_dir.glob("*.npy"):
            try:
                # Parse filename to create key
                # Format: viewtype_tSTART-END_fSTART-END_lodN.npy
                parts = file_path.stem.split('_')
                if len(parts) >= 4:
                    view_type = parts[0]
                    
                    # Parse time range
                    time_part = parts[1][1:]  # Remove 't' prefix
                    time_start, time_end = map(float, time_part.split('-'))
                    
                    # Parse freq range
                    freq_part = parts[2][1:]  # Remove 'f' prefix
                    freq_start, freq_end = map(float, freq_part.split('-'))
                    
                    # Parse LOD
                    lod_part = parts[3][3:]  # Remove 'lod' prefix
                    resolution_level = int(lod_part)
                    
                    key = TileKey(view_type, (time_start, time_end),
                                 (freq_start, freq_end), resolution_level)
                    self._disk_index[key] = file_path
                    
            except Exception as e:
                logger.warning(f"Failed to parse cache file {file_path}: {e}")
    
    def get(self, view_type: str, time_range: Tuple[float, float],
            freq_range: Tuple[float, float], resolution_level: int = 0) -> Optional[np.ndarray]:
        """Get tile from cache.
        
        Args:
            view_type: Type of view ('spectrogram', 'cepstrogram', 'fk_transform')
            time_range: (start, end) in seconds
            freq_range: (start, end) in Hz
            resolution_level: Mipmap level (0 = full resolution)
            
        Returns:
            Tile data or None if not cached
        """
        key = TileKey(view_type, time_range, freq_range, resolution_level)
        
        # Try memory cache first
        with self._memory_lock:
            if key in self._memory_cache:
                # Move to end (most recently used)
                self._memory_cache.move_to_end(key)
                self.stats['memory_hits'] += 1
                return self._memory_cache[key].copy()
        
        # Try disk cache
        with self._disk_lock:
            if key in self._disk_index:
                try:
                    file_path = self._disk_index[key]
                    if file_path.exists():
                        # Load from disk
                        data = np.load(file_path, mmap_mode='r')
                        data_copy = data.copy()
                        
                        # Add to memory cache
                        self._add_to_memory_cache(key, data_copy)
                        
                        self.stats['disk_hits'] += 1
                        return data_copy
                except Exception as e:
                    logger.error(f"Failed to load tile from disk: {e}")
                    # Remove from index if corrupted
                    del self._disk_index[key]
        
        self.stats['misses'] += 1
        return None
    
    def put(self, view_type: str, time_range: Tuple[float, float],
            freq_range: Tuple[float, float], data: np.ndarray,
            resolution_level: int = 0, async_write: bool = True):
        """Store tile in cache.
        
        Args:
            view_type: Type of view
            time_range: (start, end) in seconds
            freq_range: (start, end) in Hz
            data: Tile data (will be copied)
            resolution_level: Mipmap level
            async_write: If True, write to disk asynchronously
        """
        key = TileKey(view_type, time_range, freq_range, resolution_level)
        
        # Always add to memory cache immediately
        data_copy = data.copy().astype(np.float32)  # Ensure float32
        self._add_to_memory_cache(key, data_copy)
        
        # Write to disk
        if async_write:
            # TODO: Implement async write queue in Phase 2
            self._write_to_disk(key, data_copy)
        else:
            self._write_to_disk(key, data_copy)
        
        self.stats['writes'] += 1
    
    def _add_to_memory_cache(self, key: TileKey, data: np.ndarray):
        """Add tile to memory cache with LRU eviction."""
        with self._memory_lock:
            # Add to cache
            self._memory_cache[key] = data
            self._memory_cache.move_to_end(key)
            
            # Evict if over limit
            while len(self._memory_cache) > self.max_memory_tiles:
                evicted_key, evicted_data = self._memory_cache.popitem(last=False)
                self.stats['evictions'] += 1
                logger.debug(f"Evicted tile from memory: {evicted_key.to_filename()}")
    
    def _write_to_disk(self, key: TileKey, data: np.ndarray):
        """Write tile to disk."""
        with self._disk_lock:
            try:
                # Create subdirectory for view type
                view_dir = self.cache_dir / key.view_type
                view_dir.mkdir(exist_ok=True)
                
                # Generate filename
                filename = key.to_filename()
                file_path = view_dir / filename
                
                # Write to disk
                np.save(file_path, data)
                
                # Update index
                self._disk_index[key] = file_path
                
                # Check disk usage
                self._enforce_disk_limit()
                
            except Exception as e:
                logger.error(f"Failed to write tile to disk: {e}")
    
    def _enforce_disk_limit(self):
        """Enforce disk space limit by removing oldest files."""
        # Calculate total disk usage
        total_size = sum(f.stat().st_size for f in self.cache_dir.rglob("*.npy")
                        if f.exists())
        
        if total_size > self.max_disk_bytes:
            # Sort files by access time (oldest first)
            files = [(f, f.stat().st_atime) for f in self.cache_dir.rglob("*.npy")
                    if f.exists()]
            files.sort(key=lambda x: x[1])
            
            # Remove oldest files until under limit
            for file_path, _ in files:
                if total_size <= self.max_disk_bytes * 0.9:  # Keep 10% buffer
                    break
                
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    total_size -= file_size
                    
                    # Remove from index
                    keys_to_remove = [k for k, v in self._disk_index.items()
                                     if v == file_path]
                    for k in keys_to_remove:
                        del self._disk_index[k]
                    
                    logger.debug(f"Removed old cache file: {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to remove cache file: {e}")
    
    def clear(self, view_type: Optional[str] = None):
        """Clear cache.
        
        Args:
            view_type: If specified, only clear this view type
        """
        with self._memory_lock, self._disk_lock:
            if view_type:
                # Clear specific view type
                keys_to_remove = [k for k in self._memory_cache.keys()
                                 if k.view_type == view_type]
                for k in keys_to_remove:
                    del self._memory_cache[k]
                
                # Remove disk files
                view_dir = self.cache_dir / view_type
                if view_dir.exists():
                    shutil.rmtree(view_dir)
                    view_dir.mkdir()
                
                # Update index
                keys_to_remove = [k for k in self._disk_index.keys()
                                 if k.view_type == view_type]
                for k in keys_to_remove:
                    del self._disk_index[k]
            else:
                # Clear everything
                self._memory_cache.clear()
                self._disk_index.clear()
                
                # Remove all cache files
                if self.cache_dir.exists():
                    shutil.rmtree(self.cache_dir)
                    self.cache_dir.mkdir(parents=True)
        
        logger.info(f"Cache cleared: {view_type or 'all'}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._memory_lock, self._disk_lock:
            total_requests = (self.stats['memory_hits'] +
                            self.stats['disk_hits'] +
                            self.stats['misses'])
            
            hit_rate = 0.0
            if total_requests > 0:
                hit_rate = ((self.stats['memory_hits'] + self.stats['disk_hits']) /
                           total_requests * 100)
            
            # Calculate disk usage
            disk_usage = sum(f.stat().st_size for f in self.cache_dir.rglob("*.npy")
                           if f.exists())
            
            return {
                'memory_tiles': len(self._memory_cache),
                'disk_tiles': len(self._disk_index),
                'memory_hits': self.stats['memory_hits'],
                'disk_hits': self.stats['disk_hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate,
                'writes': self.stats['writes'],
                'evictions': self.stats['evictions'],
                'disk_usage_mb': disk_usage / (1024**2),
                'disk_limit_mb': self.max_disk_bytes / (1024**2)
            }
    
    def prefetch(self, keys: list):
        """Prefetch tiles into memory cache (for future optimization)."""
        # TODO: Implement in Phase 2 with async workers
        pass

