"""
Waterfall Tile Manager for DAS multi-channel visualization.

Handles tile-based loading and caching for large DAS matrices:
- Divides large matrices into manageable tiles
- Multi-level cache (GPU → CPU → Disk)
- Multi-resolution mipmap pyramid for zoom
- Async tile loading with prefetching
"""

import numpy as np
import logging
import threading
import time
from typing import Optional, Dict, List, Tuple, Callable, Any
from dataclasses import dataclass, field
from collections import OrderedDict
from datetime import datetime
import gc

logger = logging.getLogger(__name__)

# Try to import GPU libraries
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    cp = None
    HAS_GPU = False


# ==================== Configuration ====================

@dataclass
class TileConfig:
    """Configuration for tile-based loading."""
    tile_width: int = 512       # Sensors per tile
    tile_height: int = 512      # Time samples per tile
    max_gpu_tiles: int = 64     # Max tiles in GPU memory (~64MB)
    max_cpu_tiles: int = 256    # Max tiles in CPU RAM (~256MB)
    prefetch_radius: int = 1    # Tiles to prefetch around visible
    mipmap_levels: int = 6      # Number of resolution levels (1x to 32x)
    async_threads: int = 4      # Background loader threads


# ==================== Data Structures ====================

@dataclass(frozen=True)
class TileCoord:
    """Immutable tile coordinate (for use as dict key)."""
    level: int      # Mipmap level (0 = full resolution)
    tile_x: int     # Tile X index (sensor direction)
    tile_y: int     # Tile Y index (time direction)

    def __hash__(self):
        return hash((self.level, self.tile_x, self.tile_y))


@dataclass
class TileData:
    """Cached tile data with metadata."""
    coord: TileCoord
    data: np.ndarray                    # Tile data (float32)
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    is_on_gpu: bool = False
    gpu_data: Any = None                # CuPy array if on GPU

    @property
    def size_bytes(self) -> int:
        return self.data.nbytes

    def touch(self):
        """Update access timestamp and count."""
        self.timestamp = time.time()
        self.access_count += 1


@dataclass
class ViewRect:
    """Visible region in data coordinates."""
    sensor_start: int
    sensor_end: int
    time_start: int      # Sample index
    time_end: int        # Sample index
    zoom_level: float    # Current zoom (1.0 = full view, >1 = zoomed in)


# ==================== LRU Cache ====================

class LRUTileCache:
    """LRU cache for tile data."""

    def __init__(self, max_tiles: int, name: str = "cache"):
        self.max_tiles = max_tiles
        self.name = name
        self._cache: OrderedDict[TileCoord, TileData] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, coord: TileCoord) -> Optional[TileData]:
        """Get tile from cache, updating access order."""
        with self._lock:
            if coord in self._cache:
                tile = self._cache[coord]
                tile.touch()
                # Move to end (most recently used)
                self._cache.move_to_end(coord)
                return tile
            return None

    def put(self, tile: TileData) -> List[TileData]:
        """
        Put tile in cache, evicting old tiles if needed.

        Returns:
            List of evicted tiles (for cleanup)
        """
        evicted = []
        with self._lock:
            # If already in cache, update and move to end
            if tile.coord in self._cache:
                self._cache[tile.coord] = tile
                self._cache.move_to_end(tile.coord)
                return evicted

            # Evict oldest tiles if at capacity
            while len(self._cache) >= self.max_tiles:
                oldest_coord, oldest_tile = self._cache.popitem(last=False)
                evicted.append(oldest_tile)
                logger.debug(f"{self.name}: Evicted tile {oldest_coord}")

            # Add new tile
            self._cache[tile.coord] = tile

        return evicted

    def remove(self, coord: TileCoord) -> Optional[TileData]:
        """Remove tile from cache."""
        with self._lock:
            return self._cache.pop(coord, None)

    def clear(self):
        """Clear all tiles."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, coord: TileCoord) -> bool:
        return coord in self._cache


# ==================== Mipmap Manager ====================

class MipmapManager:
    """
    Manages multi-resolution tile pyramid for smooth zooming.

    Level 0: Full resolution
    Level 1: 2x downsampled
    Level 2: 4x downsampled
    ...
    Level N: 2^N downsampled
    """

    def __init__(self, config: TileConfig):
        self.config = config
        self._level_caches: List[LRUTileCache] = []

        for level in range(config.mipmap_levels):
            cache = LRUTileCache(
                max_tiles=config.max_cpu_tiles // (level + 1),
                name=f"mipmap_L{level}"
            )
            self._level_caches.append(cache)

    def get_level_for_zoom(self, zoom: float) -> int:
        """
        Determine appropriate mipmap level for zoom factor.

        Args:
            zoom: Zoom level (1.0 = fit all, >1 = zoomed in)

        Returns:
            Mipmap level (0 = full res, higher = lower res)
        """
        if zoom >= 1.0:
            return 0  # Full resolution for zoomed-in view

        # Calculate level based on zoom
        # zoom 0.5 → level 1, zoom 0.25 → level 2, etc.
        import math
        level = int(math.log2(1.0 / zoom))
        return min(level, self.config.mipmap_levels - 1)

    def get_downsample_factor(self, level: int) -> int:
        """Get downsample factor for mipmap level."""
        return 2 ** level

    def get_tile(self, coord: TileCoord) -> Optional[TileData]:
        """Get tile from appropriate level cache."""
        if coord.level < len(self._level_caches):
            return self._level_caches[coord.level].get(coord)
        return None

    def put_tile(self, tile: TileData) -> List[TileData]:
        """Put tile in appropriate level cache."""
        if tile.coord.level < len(self._level_caches):
            return self._level_caches[tile.coord.level].put(tile)
        return []

    def clear(self):
        """Clear all mipmap caches."""
        for cache in self._level_caches:
            cache.clear()


# ==================== Main Tile Manager ====================

class WaterfallTileManager:
    """
    Main tile manager for waterfall visualization.

    Coordinates:
    - Tile loading from data provider
    - Multi-level caching (GPU/CPU)
    - Mipmap pyramid for zoom
    - Async prefetching
    """

    def __init__(
        self,
        config: Optional[TileConfig] = None,
        data_provider=None  # DASDataProvider instance
    ):
        self.config = config or TileConfig()
        self._provider = data_provider

        # Caching
        self._gpu_cache = LRUTileCache(self.config.max_gpu_tiles, "GPU")
        self._mipmap = MipmapManager(self.config)

        # Data dimensions (set when provider loaded)
        self._n_sensors: int = 0
        self._n_samples: int = 0
        self._sample_rate: float = 1000.0

        # Async loading
        self._pending_loads: Dict[TileCoord, threading.Thread] = {}
        self._load_lock = threading.Lock()
        self._shutdown = False

        # Statistics
        self._stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'tiles_loaded': 0,
            'gpu_uploads': 0
        }

        logger.info(f"WaterfallTileManager initialized: "
                   f"tile_size={self.config.tile_width}x{self.config.tile_height}, "
                   f"max_gpu={self.config.max_gpu_tiles}, "
                   f"max_cpu={self.config.max_cpu_tiles}")

    def set_data_provider(self, provider, metadata):
        """Set the data provider and update dimensions."""
        self._provider = provider

        if metadata:
            self._n_sensors = metadata.n_sensors
            self._n_samples = metadata.total_samples
            self._sample_rate = metadata.sample_rate

            logger.info(f"Data provider set: {self._n_sensors} sensors, "
                       f"{self._n_samples} samples @ {self._sample_rate}Hz")

    def set_dimensions(self, n_sensors: int, n_samples: int, sample_rate: float):
        """Manually set data dimensions."""
        self._n_sensors = n_sensors
        self._n_samples = n_samples
        self._sample_rate = sample_rate

    # ==================== Tile Coordinate Calculation ====================

    def get_n_tiles_x(self, level: int = 0) -> int:
        """Get number of tiles in sensor (X) direction for level."""
        downsample = self._mipmap.get_downsample_factor(level)
        effective_sensors = self._n_sensors // downsample
        return (effective_sensors + self.config.tile_width - 1) // self.config.tile_width

    def get_n_tiles_y(self, level: int = 0) -> int:
        """Get number of tiles in time (Y) direction for level."""
        downsample = self._mipmap.get_downsample_factor(level)
        effective_samples = self._n_samples // downsample
        return (effective_samples + self.config.tile_height - 1) // self.config.tile_height

    def get_visible_tiles(self, view: ViewRect) -> List[TileCoord]:
        """
        Calculate which tiles are visible for the current view.

        Args:
            view: ViewRect describing visible region

        Returns:
            List of TileCoord for visible tiles
        """
        level = self._mipmap.get_level_for_zoom(view.zoom_level)
        downsample = self._mipmap.get_downsample_factor(level)

        # Convert view coordinates to downsampled space
        sensor_start = view.sensor_start // downsample
        sensor_end = view.sensor_end // downsample
        time_start = view.time_start // downsample
        time_end = view.time_end // downsample

        # Calculate tile indices
        tile_x_start = sensor_start // self.config.tile_width
        tile_x_end = (sensor_end + self.config.tile_width - 1) // self.config.tile_width
        tile_y_start = time_start // self.config.tile_height
        tile_y_end = (time_end + self.config.tile_height - 1) // self.config.tile_height

        # Clamp to valid range
        tile_x_start = max(0, tile_x_start)
        tile_x_end = min(tile_x_end, self.get_n_tiles_x(level))
        tile_y_start = max(0, tile_y_start)
        tile_y_end = min(tile_y_end, self.get_n_tiles_y(level))

        tiles = []
        for ty in range(tile_y_start, tile_y_end):
            for tx in range(tile_x_start, tile_x_end):
                tiles.append(TileCoord(level=level, tile_x=tx, tile_y=ty))

        return tiles

    def get_prefetch_tiles(self, view: ViewRect) -> List[TileCoord]:
        """Get tiles to prefetch (neighbors of visible tiles)."""
        visible = set(self.get_visible_tiles(view))
        level = self._mipmap.get_level_for_zoom(view.zoom_level)

        prefetch = set()
        radius = self.config.prefetch_radius

        for tile in visible:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue

                    nx = tile.tile_x + dx
                    ny = tile.tile_y + dy

                    if 0 <= nx < self.get_n_tiles_x(level) and \
                       0 <= ny < self.get_n_tiles_y(level):
                        coord = TileCoord(level=level, tile_x=nx, tile_y=ny)
                        if coord not in visible:
                            prefetch.add(coord)

        return list(prefetch)

    # ==================== Tile Loading ====================

    def get_tile(self, coord: TileCoord) -> Optional[TileData]:
        """
        Get tile data, loading if necessary.

        Checks caches in order: GPU → CPU mipmap → load from provider
        """
        # Check GPU cache
        tile = self._gpu_cache.get(coord)
        if tile:
            self._stats['cache_hits'] += 1
            return tile

        # Check CPU mipmap cache
        tile = self._mipmap.get_tile(coord)
        if tile:
            self._stats['cache_hits'] += 1
            # Upload to GPU if available
            if HAS_GPU:
                self._upload_to_gpu(tile)
            return tile

        # Cache miss - need to load
        self._stats['cache_misses'] += 1
        return self._load_tile(coord)

    def _load_tile(self, coord: TileCoord) -> Optional[TileData]:
        """Load tile from data provider."""
        if self._provider is None:
            logger.warning("No data provider set")
            return None

        try:
            downsample = self._mipmap.get_downsample_factor(coord.level)

            # Calculate data coordinates
            sensor_start = coord.tile_x * self.config.tile_width * downsample
            sensor_end = min(sensor_start + self.config.tile_width * downsample, self._n_sensors)
            time_start = coord.tile_y * self.config.tile_height * downsample
            time_end = min(time_start + self.config.tile_height * downsample, self._n_samples)

            # Create request
            from .das_data_provider import DASDataRequest
            from datetime import timedelta

            # Convert sample indices to datetime (assuming base time is set in provider)
            metadata = self._provider.get_metadata()
            if metadata and metadata.time_ranges:
                base_time = metadata.time_ranges[0][0]
            else:
                base_time = datetime.now()

            time_start_dt = base_time + timedelta(seconds=time_start / self._sample_rate)
            time_end_dt = base_time + timedelta(seconds=time_end / self._sample_rate)

            request = DASDataRequest(
                sensor_start=sensor_start,
                sensor_end=sensor_end,
                time_start=time_start_dt,
                time_end=time_end_dt,
                downsample_factor=downsample
            )

            # Load data
            data = self._provider.get_data_chunk(request)

            # Create tile
            tile = TileData(
                coord=coord,
                data=data.astype(np.float32)
            )

            # Cache it
            self._mipmap.put_tile(tile)
            self._stats['tiles_loaded'] += 1

            logger.debug(f"Loaded tile {coord}: shape={data.shape}")

            return tile

        except Exception as e:
            logger.error(f"Failed to load tile {coord}: {e}")
            return None

    def load_tile_async(
        self,
        coord: TileCoord,
        callback: Optional[Callable[[TileCoord, Optional[TileData]], None]] = None
    ):
        """Load tile asynchronously in background thread."""
        with self._load_lock:
            if coord in self._pending_loads:
                return  # Already loading

            def worker():
                tile = self._load_tile(coord)
                with self._load_lock:
                    self._pending_loads.pop(coord, None)
                if callback:
                    callback(coord, tile)

            thread = threading.Thread(target=worker, daemon=True)
            self._pending_loads[coord] = thread
            thread.start()

    def prefetch_tiles(self, view: ViewRect):
        """Prefetch tiles around visible region."""
        for coord in self.get_prefetch_tiles(view):
            if coord not in self._mipmap._level_caches[coord.level]:
                self.load_tile_async(coord)

    # ==================== GPU Management ====================

    def _upload_to_gpu(self, tile: TileData):
        """Upload tile data to GPU."""
        if not HAS_GPU or tile.is_on_gpu:
            return

        try:
            tile.gpu_data = cp.asarray(tile.data)
            tile.is_on_gpu = True

            # Add to GPU cache
            evicted = self._gpu_cache.put(tile)

            # Free GPU memory for evicted tiles
            for evicted_tile in evicted:
                self._free_gpu_tile(evicted_tile)

            self._stats['gpu_uploads'] += 1

        except Exception as e:
            logger.warning(f"Failed to upload tile to GPU: {e}")

    def _free_gpu_tile(self, tile: TileData):
        """Free GPU memory for tile."""
        if tile.is_on_gpu and tile.gpu_data is not None:
            del tile.gpu_data
            tile.gpu_data = None
            tile.is_on_gpu = False

    # ==================== Composite Image Generation ====================

    def get_visible_image(self, view: ViewRect) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
        """
        Get composite image for visible region.

        Returns:
            Tuple of (image_data, (sensor_start, sensor_end, time_start, time_end))
            Image is in (time, sensors) orientation for waterfall display
        """
        tiles = self.get_visible_tiles(view)
        if not tiles:
            return None, (0, 0, 0, 0)

        level = tiles[0].level
        downsample = self._mipmap.get_downsample_factor(level)

        # Calculate output dimensions
        min_tx = min(t.tile_x for t in tiles)
        max_tx = max(t.tile_x for t in tiles)
        min_ty = min(t.tile_y for t in tiles)
        max_ty = max(t.tile_y for t in tiles)

        out_width = (max_tx - min_tx + 1) * self.config.tile_width
        out_height = (max_ty - min_ty + 1) * self.config.tile_height

        # Create output array
        output = np.zeros((out_height, out_width), dtype=np.float32)

        # Fill with tile data
        for coord in tiles:
            tile = self.get_tile(coord)
            if tile is None:
                continue

            # Calculate position in output
            x_offset = (coord.tile_x - min_tx) * self.config.tile_width
            y_offset = (coord.tile_y - min_ty) * self.config.tile_height

            # Copy tile data
            tile_h, tile_w = tile.data.shape
            output[y_offset:y_offset + tile_h, x_offset:x_offset + tile_w] = tile.data

        # Calculate actual data coordinates
        sensor_start = min_tx * self.config.tile_width * downsample
        sensor_end = (max_tx + 1) * self.config.tile_width * downsample
        time_start = min_ty * self.config.tile_height * downsample
        time_end = (max_ty + 1) * self.config.tile_height * downsample

        return output, (sensor_start, sensor_end, time_start, time_end)

    # ==================== Cleanup ====================

    def clear_cache(self):
        """Clear all caches."""
        # Free GPU tiles
        for coord in list(self._gpu_cache._cache.keys()):
            tile = self._gpu_cache.remove(coord)
            if tile:
                self._free_gpu_tile(tile)

        self._gpu_cache.clear()
        self._mipmap.clear()

        if HAS_GPU:
            cp.get_default_memory_pool().free_all_blocks()

        gc.collect()
        logger.info("Tile caches cleared")

    def shutdown(self):
        """Shutdown tile manager."""
        self._shutdown = True
        self.clear_cache()
        logger.info("WaterfallTileManager shutdown")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = 0
        total = self._stats['cache_hits'] + self._stats['cache_misses']
        if total > 0:
            hit_rate = self._stats['cache_hits'] / total * 100

        return {
            **self._stats,
            'hit_rate_percent': hit_rate,
            'gpu_tiles': len(self._gpu_cache),
            'cpu_tiles': sum(len(c) for c in self._mipmap._level_caches)
        }


# Global tile manager instance
_tile_manager: Optional[WaterfallTileManager] = None


def get_waterfall_tile_manager() -> WaterfallTileManager:
    """Get global waterfall tile manager."""
    global _tile_manager
    if _tile_manager is None:
        _tile_manager = WaterfallTileManager()
    return _tile_manager


def set_waterfall_tile_manager(manager: WaterfallTileManager):
    """Set global waterfall tile manager."""
    global _tile_manager
    _tile_manager = manager
