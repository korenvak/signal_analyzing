"""
Tile Manager - Integration Layer
Coordinates between TileCache, TextureAtlas, and computation engines
"""

import threading
from typing import Tuple, Optional, Callable, List, Dict
import numpy as np
import logging

from .tile_cache import TileCache
from .mipmap_pyramid import MipmapPyramid
from ..rendering.texture_atlas import TextureAtlas

logger = logging.getLogger(__name__)


class TileRequest:
    """Represents a request for a tile."""
    
    def __init__(self, view_type: str, time_range: Tuple[float, float],
                 freq_range: Tuple[float, float], resolution_level: int = 0,
                 priority: int = 5, callback: Optional[Callable] = None):
        self.view_type = view_type
        self.time_range = time_range
        self.freq_range = freq_range
        self.resolution_level = resolution_level
        self.priority = priority
        self.callback = callback
        self.completed = False
        self.result = None
        self.error = None


class TileManager:
    """
    Central coordinator for tile-based rendering.
    
    Manages:
    - Tile cache (disk/memory)
    - Texture atlas (GPU)
    - Tile requests and priorities
    - Background computation
    """
    
    def __init__(self, tile_cache: TileCache, engines: Dict):
        """Initialize tile manager.
        
        Args:
            tile_cache: TileCache instance for persistence
            engines: Dict of computation engines (spectrogram, cepstrogram, fk)
        """
        self.tile_cache = tile_cache
        self.engines = engines
        
        # Multi-resolution mipmap pyramid
        self.mipmap_pyramid = MipmapPyramid(tile_cache, max_levels=6)
        
        # Texture atlases per view type
        self.atlases = {
            'spectrogram': TextureAtlas(atlas_size=4096, tile_size=(2048, 256)),
            'cepstrogram': TextureAtlas(atlas_size=4096, tile_size=(2048, 256)),
            'fk_transform': TextureAtlas(atlas_size=4096, tile_size=(512, 512))
        }
        
        # Request management
        self.pending_requests: List[TileRequest] = []
        self.active_requests: Dict[Tuple, TileRequest] = {}
        self.request_lock = threading.RLock()
        
        # Current visible region
        self.visible_region = {
            'spectrogram': {'time': (0, 10), 'freq': (0, 22050), 'zoom': 1.0},
            'cepstrogram': {'time': (0, 10), 'freq': (0, 4000), 'zoom': 1.0},
            'fk_transform': {'time': (0, 10), 'freq': (0, 22050), 'zoom': 1.0}
        }
        
        logger.info("TileManager initialized with atlases for 3 view types")
    
    def update_visible_region(self, view_type: str, time_range: Tuple[float, float],
                              freq_range: Tuple[float, float], zoom_level: float = 1.0):
        """Update the visible region for a view.
        
        This triggers loading of visible tiles and eviction of off-screen tiles.
        
        Args:
            view_type: Type of view
            time_range: (start, end) in seconds
            freq_range: (start, end) in Hz
            zoom_level: Current zoom level
        """
        # Update visible region
        self.visible_region[view_type] = {
            'time': time_range,
            'freq': freq_range,
            'zoom': zoom_level
        }
        
        # Get atlas for this view
        atlas = self.atlases.get(view_type)
        if not atlas:
            logger.error(f"No atlas for view type: {view_type}")
            return
        
        # Calculate which tiles should be visible
        visible_tiles = atlas.get_visible_tiles(time_range, freq_range, zoom_level)
        
        logger.debug(f"Visible region updated: {len(visible_tiles)} tiles needed for {view_type}")
        
        # Request visible tiles
        for tile_id in visible_tiles:
            self._request_tile(view_type, tile_id, priority=5)
    
    def _request_tile(self, view_type: str, tile_id: Tuple, priority: int = 5):
        """Request a tile to be loaded.
        
        Args:
            view_type: Type of view
            tile_id: (time_tile, freq_tile, lod)
            priority: Request priority (lower = higher priority)
        """
        # Check if already loaded in atlas
        atlas = self.atlases[view_type]
        if tile_id in atlas.tile_map:
            return  # Already loaded
        
        # Check if in cache
        # Convert tile_id to time/freq ranges
        time_tile, freq_tile, lod = tile_id
        tile_width = atlas.tile_width
        tile_height = atlas.tile_height
        
        time_range = (time_tile * tile_width, (time_tile + 1) * tile_width)
        freq_range = (freq_tile * tile_height, (freq_tile + 1) * tile_height)
        
        # Try to get from cache
        cached_data = self.tile_cache.get(view_type, time_range, freq_range, lod)
        
        if cached_data is not None:
            # Load into atlas immediately
            atlas.load_tile(tile_id, cached_data)
            logger.debug(f"Loaded tile {tile_id} from cache for {view_type}")
            return
        
        # Need to compute - add to request queue
        with self.request_lock:
            # Check if already requested
            request_key = (view_type, tile_id)
            if request_key in self.active_requests:
                return  # Already being computed
            
            # Create request
            request = TileRequest(
                view_type=view_type,
                time_range=time_range,
                freq_range=freq_range,
                resolution_level=lod,
                priority=priority,
                callback=lambda data, error: self._on_tile_computed(
                    view_type, tile_id, data, error)
            )
            
            self.pending_requests.append(request)
            self.active_requests[request_key] = request
            
            # Sort by priority
            self.pending_requests.sort(key=lambda r: r.priority)
        
        logger.debug(f"Queued tile {tile_id} for computation ({view_type})")
    
    def _on_tile_computed(self, view_type: str, tile_id: Tuple,
                         data: Optional[np.ndarray], error: Optional[str]):
        """Callback when a tile computation completes.
        
        Args:
            view_type: Type of view
            tile_id: Tile identifier
            data: Computed tile data
            error: Error message if failed
        """
        request_key = (view_type, tile_id)
        
        with self.request_lock:
            # Remove from active requests
            if request_key in self.active_requests:
                del self.active_requests[request_key]
        
        if error:
            logger.error(f"Tile computation failed: {error}")
            return
        
        if data is None:
            logger.error(f"Tile computation returned None")
            return
        
        # Store in cache and build mipmap pyramid
        time_tile, freq_tile, lod = tile_id
        atlas = self.atlases[view_type]
        time_range = (time_tile * atlas.tile_width, (time_tile + 1) * atlas.tile_width)
        freq_range = (freq_tile * atlas.tile_height, (freq_tile + 1) * atlas.tile_height)
        
        # If this is LOD 0 (full resolution), build complete pyramid
        if lod == 0:
            self.mipmap_pyramid.build_pyramid(view_type, time_range, freq_range, data)
            logger.debug(f"Built mipmap pyramid for tile at {time_range}")
        else:
            # Just store this LOD level
            self.tile_cache.put(view_type, time_range, freq_range, data, lod)
        
        # Load into atlas
        atlas.load_tile(tile_id, data)
        
        logger.debug(f"Tile {tile_id} computed and loaded for {view_type}")
    
    def process_pending_requests(self, audio_data: np.ndarray, max_tiles: int = 5):
        """Process pending tile requests.
        
        This should be called periodically to compute tiles.
        
        Args:
            audio_data: Audio data for computation
            max_tiles: Maximum tiles to process in this call
        """
        with self.request_lock:
            if not self.pending_requests:
                return
            
            # Process up to max_tiles requests
            to_process = self.pending_requests[:max_tiles]
            self.pending_requests = self.pending_requests[max_tiles:]
        
        for request in to_process:
            try:
                # Get appropriate engine
                engine = self.engines.get(request.view_type)
                if not engine:
                    logger.error(f"No engine for view type: {request.view_type}")
                    continue
                
                # Compute tile based on view type
                if request.view_type == 'spectrogram':
                    data = self._compute_spectrogram_tile(
                        engine, audio_data, request.time_range, request.freq_range)
                elif request.view_type == 'cepstrogram':
                    data = self._compute_cepstrogram_tile(
                        engine, audio_data, request.time_range, request.freq_range)
                elif request.view_type == 'fk_transform':
                    data = self._compute_fk_tile(
                        engine, audio_data, request.time_range, request.freq_range)
                else:
                    logger.error(f"Unknown view type: {request.view_type}")
                    continue
                
                # Call callback
                if request.callback:
                    request.callback(data, None)
                
            except Exception as e:
                logger.error(f"Error processing tile request: {e}")
                if request.callback:
                    request.callback(None, str(e))
    
    def _compute_spectrogram_tile(self, engine, audio_data: np.ndarray,
                                 time_range: Tuple[float, float],
                                 freq_range: Tuple[float, float]) -> np.ndarray:
        """Compute a spectrogram tile."""
        # Extract audio chunk for time range
        sample_rate = engine.sample_rate
        start_sample = int(time_range[0] * sample_rate)
        end_sample = int(time_range[1] * sample_rate)
        
        if start_sample >= len(audio_data):
            # Return empty tile if beyond data
            return np.zeros((257, 2048), dtype=np.float32)
        
        end_sample = min(end_sample, len(audio_data))
        audio_chunk = audio_data[start_sample:end_sample]
        
        # Check if chunk is too small
        if len(audio_chunk) < engine.fft_size:
            logger.warning(f"Audio chunk too small ({len(audio_chunk)} samples) for tile computation")
            return np.zeros((257, 2048), dtype=np.float32)
        
        # Compute STFT using batched engine
        try:
            magnitude_db, frequencies, times = engine.compute_stft_batched(audio_chunk)
        except Exception as e:
            logger.warning(f"Batched FFT failed for tile, using fallback: {e}")
            magnitude_db, frequencies, times = engine.compute_stft_cpu(audio_chunk)
        
        # Filter frequency range (keep data within requested range)
        if len(frequencies) > 0 and freq_range[1] < frequencies[-1]:
            freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
            if np.any(freq_mask):
                magnitude_db = magnitude_db[freq_mask, :]
                logger.debug(f"Filtered to {freq_range}: {magnitude_db.shape}")
            else:
                logger.warning(f"Frequency filter removed all data! Range: {freq_range}, freq: [{frequencies[0]}, {frequencies[-1]}]")
        
        # Ensure we return valid data
        if magnitude_db.size == 0:
            logger.warning(f"Empty magnitude_db after filtering, returning zeros")
            return np.zeros((256, 2048), dtype=np.float32)
        
        return magnitude_db.astype(np.float32)
    
    def _compute_cepstrogram_tile(self, engine, audio_data: np.ndarray,
                                 time_range: Tuple[float, float],
                                 freq_range: Tuple[float, float]) -> np.ndarray:
        """Compute a cepstrogram tile."""
        # Get spectrogram first
        spec_engine = self.engines.get('spectrogram')
        if not spec_engine:
            raise RuntimeError("Spectrogram engine required for cepstrogram")
        
        magnitude_db = self._compute_spectrogram_tile(
            spec_engine, audio_data, time_range, freq_range)
        
        # Compute cepstrogram from spectrogram
        frequencies = np.linspace(freq_range[0], freq_range[1], magnitude_db.shape[0])
        cepstral = engine.compute_cepstrogram_from_spectrogram(magnitude_db, frequencies)
        
        return cepstral.astype(np.float32)
    
    def _compute_fk_tile(self, engine, audio_data: np.ndarray,
                        time_range: Tuple[float, float],
                        freq_range: Tuple[float, float]) -> np.ndarray:
        """Compute an F-K transform tile."""
        # Similar to spectrogram but with F-K processing
        # TODO: Implement in Phase 5
        # For now, return placeholder
        return np.zeros((512, 512), dtype=np.float32)
    
    def get_atlas(self, view_type: str) -> Optional[TextureAtlas]:
        """Get texture atlas for a view type."""
        return self.atlases.get(view_type)
    
    def get_stats(self) -> Dict:
        """Get tile manager statistics."""
        stats = {
            'pending_requests': len(self.pending_requests),
            'active_requests': len(self.active_requests),
            'cache_stats': self.tile_cache.get_stats(),
            'atlases': {}
        }
        
        for view_type, atlas in self.atlases.items():
            stats['atlases'][view_type] = atlas.get_stats()
        
        return stats
    
    def clear(self, view_type: Optional[str] = None):
        """Clear tiles for a view type or all views."""
        if view_type:
            if view_type in self.atlases:
                self.atlases[view_type].clear()
            self.tile_cache.clear(view_type)
        else:
            for atlas in self.atlases.values():
                atlas.clear()
            self.tile_cache.clear()
        
        logger.info(f"Cleared tiles: {view_type or 'all'}")

