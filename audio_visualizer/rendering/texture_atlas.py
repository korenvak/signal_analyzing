"""
GPU Texture Atlas for Tiled Rendering
Manages a fixed-size GPU texture atlas for displaying large spectrograms
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from collections import OrderedDict
import logging

try:
    from vispy import scene
    from vispy.visuals import Visual
    HAS_VISPY = True
except ImportError:
    HAS_VISPY = False

logger = logging.getLogger(__name__)


class TileSlot:
    """Represents a slot in the texture atlas."""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.occupied = False
        self.tile_id = None
        self.last_used = 0.0
    
    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is inside this slot."""
        return (self.x <= x < self.x + self.width and
                self.y <= y < self.y + self.height)


class TextureAtlas:
    """
    GPU Texture Atlas for tiled spectrogram rendering.
    
    Manages a fixed-size GPU texture (e.g., 4096x4096) and loads
    tiles into it dynamically based on what's visible.
    """
    
    def __init__(self, atlas_size: int = 4096, tile_size: Tuple[int, int] = (512, 256)):
        """Initialize texture atlas.
        
        Args:
            atlas_size: Size of the square atlas texture (e.g., 4096)
            tile_size: (width, height) of each tile in pixels
        """
        if not HAS_VISPY:
            raise RuntimeError("VisPy not available for texture atlas")
        
        self.atlas_size = atlas_size
        self.tile_width, self.tile_height = tile_size
        
        # Calculate grid dimensions
        self.grid_cols = atlas_size // self.tile_width
        self.grid_rows = atlas_size // self.tile_height
        self.max_tiles = self.grid_cols * self.grid_rows
        
        # Initialize atlas texture data (grayscale float32)
        self.atlas_data = np.zeros((atlas_size, atlas_size), dtype=np.float32)
        
        # Create tile slots
        self.slots: List[TileSlot] = []
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x = col * self.tile_width
                y = row * self.tile_height
                slot = TileSlot(x, y, self.tile_width, self.tile_height)
                self.slots.append(slot)
        
        # Tile management
        self.tile_map: Dict[Tuple, int] = {}  # (time, freq, lod) -> slot_index
        self.lru_order: OrderedDict = OrderedDict()  # slot_index -> timestamp
        
        # Statistics
        self.stats = {
            'slots_used': 0,
            'tiles_loaded': 0,
            'tiles_evicted': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info(f"TextureAtlas initialized: {atlas_size}x{atlas_size}, "
                   f"{self.max_tiles} slots ({self.grid_cols}x{self.grid_rows})")
    
    def get_tile_id(self, time_center: float, freq_center: float, lod: int = 0) -> Tuple:
        """Generate tile ID from world coordinates."""
        # Quantize to tile grid
        tile_time = int(time_center / self.tile_width)
        tile_freq = int(freq_center / self.tile_height)
        return (tile_time, tile_freq, lod)
    
    def get_visible_tiles(self, time_range: Tuple[float, float],
                         freq_range: Tuple[float, float],
                         zoom_level: float) -> List[Tuple]:
        """Calculate which tiles are visible in the current view.
        
        Args:
            time_range: (start, end) in seconds
            freq_range: (start, end) in Hz  
            zoom_level: Current zoom level (for LOD selection)
            
        Returns:
            List of tile IDs that should be visible
        """
        # Select LOD based on zoom
        lod = self._select_lod(zoom_level)
        
        # Calculate tile grid coordinates
        time_start_tile = int(time_range[0] / self.tile_width)
        time_end_tile = int(time_range[1] / self.tile_width) + 1
        freq_start_tile = int(freq_range[0] / self.tile_height)
        freq_end_tile = int(freq_range[1] / self.tile_height) + 1
        
        # Generate list of visible tiles
        visible_tiles = []
        for t in range(time_start_tile, time_end_tile + 1):
            for f in range(freq_start_tile, freq_end_tile + 1):
                visible_tiles.append((t, f, lod))
        
        return visible_tiles
    
    def _select_lod(self, zoom_level: float) -> int:
        """Select appropriate LOD level based on zoom.
        
        Args:
            zoom_level: Current zoom level (1.0 = fit to screen, >1.0 = zoomed in, <1.0 = zoomed out)
        
        Returns:
            LOD level (0 = full resolution, higher = more downsampled)
        """
        # When zoomed out (zoom_level < 1.0), use lower resolution tiles
        # When zoomed in (zoom_level > 1.0), use full resolution
        if zoom_level >= 1.0:
            return 0  # Full resolution when zoomed in
        elif zoom_level >= 0.5:
            return 1  # 2x downsampled
        elif zoom_level >= 0.25:
            return 2  # 4x downsampled
        elif zoom_level >= 0.125:
            return 3  # 8x downsampled
        else:
            return 4  # 16x downsampled for very zoomed out views
    
    def load_tile(self, tile_id: Tuple, tile_data: np.ndarray) -> bool:
        """Load a tile into the atlas.
        
        Args:
            tile_id: (time_tile, freq_tile, lod)
            tile_data: Tile data array (height, width)
            
        Returns:
            True if loaded successfully
        """
        import time as time_module
        
        # Check if already loaded
        if tile_id in self.tile_map:
            slot_idx = self.tile_map[tile_id]
            self.lru_order.move_to_end(slot_idx)
            self.lru_order[slot_idx] = time_module.time()
            self.stats['cache_hits'] += 1
            return True
        
        # Find free slot or evict LRU
        slot_idx = self._find_free_slot()
        if slot_idx is None:
            slot_idx = self._evict_lru_tile()
            if slot_idx is None:
                logger.error("Failed to find slot for tile!")
                return False
        
        # Check if tile_data is valid
        slot = self.slots[slot_idx]
        if tile_data.size == 0 or tile_data.shape[0] == 0 or tile_data.shape[1] == 0:
            logger.warning(f"Empty tile data received: {tile_data.shape}, filling with zeros")
            tile_data = np.zeros((slot.height, slot.width), dtype=np.float32)
        elif tile_data.shape != (slot.height, slot.width):
            # Resize tile data to fit slot
            from scipy.ndimage import zoom
            scale_y = slot.height / max(1, tile_data.shape[0])
            scale_x = slot.width / max(1, tile_data.shape[1])
            if scale_y != 1.0 or scale_x != 1.0:
                tile_data = zoom(tile_data, (scale_y, scale_x), order=1)
        
        # Copy tile data into atlas
        y_start = slot.y
        y_end = slot.y + slot.height
        x_start = slot.x
        x_end = slot.x + slot.width
        
        self.atlas_data[y_start:y_end, x_start:x_end] = tile_data
        
        # Update slot
        slot.occupied = True
        slot.tile_id = tile_id
        slot.last_used = time_module.time()
        
        # Update mappings
        self.tile_map[tile_id] = slot_idx
        self.lru_order[slot_idx] = slot.last_used
        self.lru_order.move_to_end(slot_idx)
        
        # Update stats
        self.stats['slots_used'] = sum(1 for s in self.slots if s.occupied)
        self.stats['tiles_loaded'] += 1
        self.stats['cache_misses'] += 1
        
        return True
    
    def _find_free_slot(self) -> Optional[int]:
        """Find a free (unoccupied) slot."""
        for idx, slot in enumerate(self.slots):
            if not slot.occupied:
                return idx
        return None
    
    def _evict_lru_tile(self) -> Optional[int]:
        """Evict least recently used tile."""
        if not self.lru_order:
            return None
        
        # Get LRU slot
        slot_idx, _ = self.lru_order.popitem(last=False)
        slot = self.slots[slot_idx]
        
        # Remove from tile map
        if slot.tile_id in self.tile_map:
            del self.tile_map[slot.tile_id]
        
        # Mark as free
        slot.occupied = False
        slot.tile_id = None
        
        self.stats['tiles_evicted'] += 1
        
        return slot_idx
    
    def get_atlas_data(self) -> np.ndarray:
        """Get the current atlas texture data."""
        return self.atlas_data
    
    def get_tile_mapping(self) -> Dict[Tuple, Tuple[int, int, int, int]]:
        """Get mapping from tile IDs to atlas positions.
        
        Returns:
            Dict mapping tile_id -> (x, y, width, height) in atlas
        """
        mapping = {}
        for tile_id, slot_idx in self.tile_map.items():
            slot = self.slots[slot_idx]
            mapping[tile_id] = (slot.x, slot.y, slot.width, slot.height)
        return mapping
    
    def clear(self):
        """Clear all tiles from the atlas."""
        self.atlas_data.fill(0)
        for slot in self.slots:
            slot.occupied = False
            slot.tile_id = None
        self.tile_map.clear()
        self.lru_order.clear()
        self.stats['slots_used'] = 0
    
    def get_stats(self) -> Dict:
        """Get atlas statistics."""
        return {
            **self.stats,
            'atlas_size': self.atlas_size,
            'max_tiles': self.max_tiles,
            'tile_size': (self.tile_width, self.tile_height),
            'utilization': (self.stats['slots_used'] / self.max_tiles * 100)
        }


class AtlasRenderer:
    """
    Renders the texture atlas to screen.
    Handles coordinate mapping from world space to atlas space.
    """
    
    def __init__(self, atlas: TextureAtlas):
        """Initialize atlas renderer.
        
        Args:
            atlas: TextureAtlas instance to render
        """
        if not HAS_VISPY:
            raise RuntimeError("VisPy not available for atlas rendering")
        
        self.atlas = atlas
        self.image_visual = None  # Will be set by main window
    
    def update_display(self, image_visual):
        """Update the VisPy image visual with atlas data.
        
        Args:
            image_visual: VisPy Image visual to update
        """
        atlas_data = self.atlas.get_atlas_data()
        
        # Normalize for display
        if atlas_data.max() > atlas_data.min():
            display_data = (atlas_data - atlas_data.min()) / (atlas_data.max() - atlas_data.min())
        else:
            display_data = atlas_data
        
        image_visual.set_data(display_data)
        image_visual.clim = (0, 1)
    
    def world_to_atlas(self, time: float, freq: float) -> Tuple[int, int]:
        """Convert world coordinates to atlas pixel coordinates.
        
        Args:
            time: Time in seconds
            freq: Frequency in Hz
            
        Returns:
            (x, y) in atlas texture coordinates
        """
        # Get tile ID
        tile_id = self.atlas.get_tile_id(time, freq, lod=0)
        
        # Get slot position
        if tile_id not in self.atlas.tile_map:
            return (0, 0)  # Tile not loaded
        
        slot_idx = self.atlas.tile_map[tile_id]
        slot = self.atlas.slots[slot_idx]
        
        # Calculate position within tile
        tile_time = tile_id[0] * self.atlas.tile_width
        tile_freq = tile_id[1] * self.atlas.tile_height
        
        local_x = int((time - tile_time))
        local_y = int((freq - tile_freq))
        
        atlas_x = slot.x + local_x
        atlas_y = slot.y + local_y
        
        return (atlas_x, atlas_y)

