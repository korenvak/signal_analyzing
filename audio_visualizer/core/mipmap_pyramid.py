"""
Multi-Resolution Mipmap Pyramid
Provides level-of-detail (LOD) system for smooth zooming with full resolution
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from scipy.ndimage import zoom
import logging

from .tile_cache import TileCache

logger = logging.getLogger(__name__)


class MipmapPyramid:
    """
    Multi-resolution mipmap pyramid for spectrograms.
    
    Builds and manages multiple resolution levels:
    - LOD 0: Full resolution (11.6ms per frame)
    - LOD 1: 2x downsampled (23.2ms per frame)
    - LOD 2: 4x downsampled (46.4ms per frame)
    - LOD 3: 8x downsampled (92.8ms per frame)
    - etc.
    
    Automatically selects appropriate LOD based on zoom level.
    """
    
    def __init__(self, tile_cache: TileCache, max_levels: int = 6):
        """Initialize mipmap pyramid.
        
        Args:
            tile_cache: TileCache for storing pyramid levels
            max_levels: Maximum number of LOD levels
        """
        self.tile_cache = tile_cache
        self.max_levels = max_levels
        
        # LOD selection parameters
        self.lod_thresholds = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
        
        logger.info(f"MipmapPyramid initialized with {max_levels} levels")
    
    def select_lod(self, zoom_level: float, viewport_width: int,
                   data_width: int) -> int:
        """Select appropriate LOD level based on zoom and viewport.
        
        Args:
            zoom_level: Current zoom level (1.0 = fit to screen)
            viewport_width: Viewport width in pixels
            data_width: Full data width in samples/frames
            
        Returns:
            LOD level (0 = full resolution, higher = more downsampled)
        """
        # Calculate pixels per data sample
        pixels_per_sample = (viewport_width * zoom_level) / data_width
        
        # Select LOD based on detail needed
        # If pixels_per_sample < 1, we're zoomed out - use lower resolution
        # If pixels_per_sample > 1, we're zoomed in - use full resolution
        
        if pixels_per_sample >= 1.0:
            return 0  # Full resolution - zoomed in enough to see detail
        
        # Calculate appropriate LOD
        for lod, threshold in enumerate(self.lod_thresholds):
            if pixels_per_sample >= threshold:
                return min(lod, self.max_levels - 1)
        
        return self.max_levels - 1  # Most downsampled
    
    def build_level(self, base_data: np.ndarray, target_lod: int) -> np.ndarray:
        """Build a specific LOD level from base data.
        
        Args:
            base_data: Full resolution data (freq_bins, time_frames)
            target_lod: Target LOD level (0 = no change, 1 = 2x downsample, etc.)
            
        Returns:
            Downsampled data
        """
        if target_lod == 0:
            return base_data.astype(np.float32)
        
        # Calculate downsample factor
        factor = 2 ** target_lod
        
        # Downsample using simple indexing (fast)
        # For spectrogram: preserve frequency detail, downsample time
        downsampled = base_data[:, ::factor].astype(np.float32)
        
        logger.debug(f"Built LOD {target_lod}: {base_data.shape} -> {downsampled.shape} ({factor}x downsample)")
        
        return downsampled
    
    def build_pyramid(self, view_type: str, time_range: Tuple[float, float],
                     freq_range: Tuple[float, float], base_data: np.ndarray):
        """Build complete pyramid for a tile.
        
        Args:
            view_type: Type of view
            time_range: Time range
            freq_range: Frequency range
            base_data: Full resolution data (LOD 0)
        """
        # Store LOD 0 (full resolution)
        self.tile_cache.put(view_type, time_range, freq_range, base_data, resolution_level=0)
        
        # Build and store additional LOD levels
        for lod in range(1, self.max_levels):
            downsampled = self.build_level(base_data, lod)
            self.tile_cache.put(view_type, time_range, freq_range, downsampled, resolution_level=lod)
            
            logger.debug(f"Stored LOD {lod} for {view_type}: {downsampled.shape}")
    
    def get_tile_at_lod(self, view_type: str, time_range: Tuple[float, float],
                       freq_range: Tuple[float, float], lod: int) -> Optional[np.ndarray]:
        """Get tile at specific LOD level.
        
        Args:
            view_type: Type of view
            time_range: Time range
            freq_range: Frequency range
            lod: LOD level
            
        Returns:
            Tile data at requested LOD, or None if not cached
        """
        return self.tile_cache.get(view_type, time_range, freq_range, lod)
    
    def get_best_available_lod(self, view_type: str, time_range: Tuple[float, float],
                              freq_range: Tuple[float, float],
                              desired_lod: int) -> Tuple[Optional[np.ndarray], int]:
        """Get best available LOD (may be different from desired).
        
        Args:
            view_type: Type of view
            time_range: Time range
            freq_range: Frequency range
            desired_lod: Desired LOD level
            
        Returns:
            (data, actual_lod) - May return higher LOD if desired not available
        """
        # Try desired LOD first
        data = self.get_tile_at_lod(view_type, time_range, freq_range, desired_lod)
        if data is not None:
            return data, desired_lod
        
        # Try progressively lower quality (higher LOD numbers)
        for lod in range(desired_lod + 1, self.max_levels):
            data = self.get_tile_at_lod(view_type, time_range, freq_range, lod)
            if data is not None:
                logger.debug(f"Using LOD {lod} instead of {desired_lod}")
                return data, lod
        
        # Try progressively higher quality (lower LOD numbers)
        for lod in range(desired_lod - 1, -1, -1):
            data = self.get_tile_at_lod(view_type, time_range, freq_range, lod)
            if data is not None:
                logger.debug(f"Using LOD {lod} instead of {desired_lod}")
                return data, lod
        
        return None, -1


def downsample_2d(data: np.ndarray, factor: int) -> np.ndarray:
    """Fast 2D downsampling using averaging.
    
    Args:
        data: Input array (height, width)
        factor: Downsample factor
        
    Returns:
        Downsampled array
    """
    if factor == 1:
        return data
    
    # Use simple indexing for speed (could use averaging for better quality)
    return data[::factor, ::factor]


def build_mipmap_fast(data: np.ndarray, max_levels: int = 6) -> List[np.ndarray]:
    """Build mipmap levels quickly.
    
    Args:
        data: Base level data (height, width)
        max_levels: Number of levels to build
        
    Returns:
        List of mipmap levels [LOD0, LOD1, LOD2, ...]
    """
    levels = [data.astype(np.float32)]
    
    for lod in range(1, max_levels):
        factor = 2 ** lod
        downsampled = downsample_2d(data, factor)
        levels.append(downsampled.astype(np.float32))
        
        # Stop if too small
        if downsampled.shape[0] < 4 or downsampled.shape[1] < 4:
            break
    
    return levels

