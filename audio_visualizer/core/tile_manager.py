"""
Tile Manager - Integration Layer
Coordinates between TileCache, TextureAtlas, and computation engines
"""

import math
import threading
from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable, List, Dict, Any
import numpy as np
import logging

from .tile_cache import TileCache
from .mipmap_pyramid import MipmapPyramid
from ..rendering.texture_atlas import TextureAtlas

logger = logging.getLogger(__name__)


@dataclass
class ViewConfig:
    """Configuration describing how world coordinates map to tiles."""
    tile_width_frames: int
    tile_height_bins: int
    time_per_frame: float
    freq_per_bin: float
    freq_range: Tuple[float, float]
    max_lod_levels: int = 6

    @property
    def tile_world_size(self) -> Tuple[float, float]:
        return (
            self.tile_width_frames * self.time_per_frame,
            self.tile_height_bins * self.freq_per_bin
        )

    def world_time_to_frame(self, time_value: float) -> float:
        return time_value / max(self.time_per_frame, 1e-12)

    def world_freq_to_bin(self, freq_value: float) -> float:
        return (freq_value - self.freq_range[0]) / max(self.freq_per_bin, 1e-12)


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
        self.audio_loader = None
        self.view_configs: Dict[str, ViewConfig] = {}
        
        # Multi-resolution mipmap pyramid
        self.mipmap_pyramid = MipmapPyramid(tile_cache, max_levels=6)
        
        # Texture atlases per view type
        self.atlases: Dict[str, Optional[TextureAtlas]] = {}
        
        # Request management
        self.pending_requests: List[TileRequest] = []
        self.active_requests: Dict[Tuple, TileRequest] = {}
        self.request_lock = threading.RLock()
        
        # Current visible region
        self.visible_region: Dict[str, Dict[str, Any]] = {}
        self.visible_tiles: Dict[str, Dict[str, Any]] = {}
        for view_type in engines.keys():
            self.refresh_view_config(view_type)
        
        logger.info("TileManager initialized with atlases for 3 view types")
    
    def _create_view_config(self, view_type: str) -> ViewConfig:
        """Create or update the configuration for a particular view type."""
        engine = self.engines.get(view_type)

        if view_type == 'spectrogram' and engine is not None:
            tile_width_frames, tile_height_bins = getattr(engine, 'tile_size', (2048, 512))
            time_per_frame = engine.hop_length / max(engine.sample_rate, 1)
            freq_per_bin = engine.sample_rate / max(engine.fft_size, 1)
            freq_range = (0.0, engine.sample_rate / 2.0)
            max_lod = 6
        elif view_type == 'cepstrogram' and engine is not None:
            # Cepstrogram shares time resolution with spectrogram but operates in quefrency domain
            spec_engine = getattr(engine, 'spectrogram_engine', None)
            if spec_engine is not None:
                time_per_frame = spec_engine.hop_length / max(spec_engine.sample_rate, 1)
            else:
                time_per_frame = 0.01
            tile_width_frames, tile_height_bins = (2048, 256)
            # Quefrency bins are roughly in milliseconds; keep 1 unit steps for now
            freq_per_bin = 1.0
            freq_range = (0.0, tile_height_bins * freq_per_bin)
            max_lod = 5
        else:
            tile_width_frames, tile_height_bins = (1024, 256)
            time_per_frame = 0.01
            freq_per_bin = 50.0
            freq_range = (0.0, tile_height_bins * freq_per_bin)
            max_lod = 4

        return ViewConfig(
            tile_width_frames=tile_width_frames,
            tile_height_bins=tile_height_bins,
            time_per_frame=time_per_frame,
            freq_per_bin=freq_per_bin,
            freq_range=freq_range,
            max_lod_levels=max_lod
        )

    def refresh_view_config(self, view_type: str):
        """Recompute configuration and rebuild atlas for a view."""
        config = self._create_view_config(view_type)
        self.view_configs[view_type] = config

        try:
            self.atlases[view_type] = TextureAtlas(
                atlas_size=4096,
                tile_size=(config.tile_width_frames, config.tile_height_bins),
                tile_world_size=config.tile_world_size
            )
        except RuntimeError:
            self.atlases[view_type] = None

        default_time = (0.0, config.tile_world_size[0])
        default_freq = (
            config.freq_range[0],
            min(config.freq_range[1], config.freq_range[0] + config.tile_world_size[1])
        )

        self.visible_region[view_type] = {
            'time': default_time,
            'freq': default_freq,
            'zoom': 1.0,
            'lod': 0,
            'viewport_width': 1920
        }
        self.visible_tiles[view_type] = {
            'tiles': [],
            'lod': 0,
            'time_range': default_time,
            'freq_range': default_freq
        }

    def set_audio_loader(self, loader) -> None:
        """Set audio loader used for streaming tile computations."""
        self.audio_loader = loader

    def assemble_visible_region(self, view_type: str) -> Tuple[np.ndarray, Tuple[float, float, float, float], bool]:
        """Assemble currently visible tiles into a single array for display.

        Returns:
            (data, extent, complete)
            where extent = (time_start, time_end, freq_start, freq_end) in world units
            and complete indicates whether all tiles were available without gaps.
        """
        state = self.visible_tiles.get(view_type)
        config = self.view_configs.get(view_type)
        if state is None or config is None:
            return np.zeros((1, 1), dtype=np.float32), (0.0, 0.0, 0.0, 0.0), False

        time_range = state['time_range']
        freq_range = state['freq_range']
        lod = state['lod']

        time_step = config.time_per_frame * (2 ** lod)
        freq_step = config.freq_per_bin * (2 ** lod)

        width = max(1, int(math.ceil((time_range[1] - time_range[0]) / max(time_step, 1e-9))))
        height = max(1, int(math.ceil((freq_range[1] - freq_range[0]) / max(freq_step, 1e-9))))

        assembled = np.zeros((height, width), dtype=np.float32)
        coverage = np.zeros((height, width), dtype=bool)
        complete = True

        for tile_id in state['tiles']:
            tile_time_range, tile_freq_range = self._tile_id_to_world_bounds(view_type, tile_id)

            overlap_time_start = max(time_range[0], tile_time_range[0])
            overlap_time_end = min(time_range[1], tile_time_range[1])
            overlap_freq_start = max(freq_range[0], tile_freq_range[0])
            overlap_freq_end = min(freq_range[1], tile_freq_range[1])

            if overlap_time_start >= overlap_time_end or overlap_freq_start >= overlap_freq_end:
                continue

            tile_data = self.tile_cache.get(view_type, tile_time_range, tile_freq_range, tile_id[2])
            if tile_data is None:
                complete = False
                continue

            tile_time_step = time_step
            tile_freq_step = freq_step

            tile_x_start = int(round((overlap_time_start - tile_time_range[0]) / max(tile_time_step, 1e-9)))
            tile_x_end = tile_x_start + int(math.ceil((overlap_time_end - overlap_time_start) / max(tile_time_step, 1e-9)))
            tile_y_start = int(round((overlap_freq_start - tile_freq_range[0]) / max(tile_freq_step, 1e-9)))
            tile_y_end = tile_y_start + int(math.ceil((overlap_freq_end - overlap_freq_start) / max(tile_freq_step, 1e-9)))

            result_x_start = int(round((overlap_time_start - time_range[0]) / max(time_step, 1e-9)))
            result_x_end = result_x_start + (tile_x_end - tile_x_start)
            result_y_start = int(round((overlap_freq_start - freq_range[0]) / max(freq_step, 1e-9)))
            result_y_end = result_y_start + (tile_y_end - tile_y_start)

            # Clamp indices to array bounds
            result_x_end = min(result_x_end, width)
            result_y_end = min(result_y_end, height)
            tile_x_end = tile_x_start + (result_x_end - result_x_start)
            tile_y_end = tile_y_start + (result_y_end - result_y_start)

            if result_x_start >= result_x_end or result_y_start >= result_y_end:
                continue

            sub_tile = tile_data[tile_y_start:tile_y_end, tile_x_start:tile_x_end]
            if sub_tile.size == 0:
                complete = False
                continue

            assembled[result_y_start:result_y_end, result_x_start:result_x_end] = sub_tile[:result_y_end - result_y_start, :result_x_end - result_x_start]
            coverage[result_y_start:result_y_end, result_x_start:result_x_end] = True

        if not np.all(coverage):
            complete = False

        extent = (time_range[0], time_range[1], freq_range[0], freq_range[1])
        return assembled, extent, complete

    def update_visible_region(self, view_type: str, time_range: Tuple[float, float],
                              freq_range: Tuple[float, float], zoom_level: float = 1.0,
                              viewport_width: int = 1920):
        """Update the visible region for a view.
        
        This triggers loading of visible tiles and eviction of off-screen tiles.
        
        Args:
            view_type: Type of view
            time_range: (start, end) in seconds
            freq_range: (start, end) in Hz
            zoom_level: Current zoom level
            viewport_width: Width of viewport in pixels (for LOD selection)
        """
        config = self.view_configs.get(view_type)
        if config is None:
            logger.error(f"No view configuration for {view_type}")
            return

        region_state = self.visible_region.setdefault(view_type, {})
        region_state.update({
            'time': time_range,
            'freq': freq_range,
            'zoom': zoom_level,
            'viewport_width': viewport_width
        })

        # Determine appropriate LOD based on requested span and viewport size
        time_span = max(time_range[1] - time_range[0], config.time_per_frame)
        data_width_frames = max(1, int(time_span / max(config.time_per_frame, 1e-9)))
        lod = self.mipmap_pyramid.select_lod(
            zoom_level=max(zoom_level, 1e-6),
            viewport_width=viewport_width,
            data_width=data_width_frames
        )
        lod = min(lod, config.max_lod_levels - 1)
        region_state['lod'] = lod

        lod_tile_world = (
            config.tile_world_size[0] * (2 ** lod),
            config.tile_world_size[1] * (2 ** lod)
        )

        atlas = self.atlases.get(view_type)
        if atlas is not None:
            atlas.set_tile_world_size(lod_tile_world)

        visible_tiles = self._compute_visible_tiles_for_region(
            time_range=time_range,
            freq_range=freq_range,
            lod=lod,
            tile_world_size=lod_tile_world
        )

        self.visible_tiles[view_type] = {
            'tiles': visible_tiles,
            'lod': lod,
            'time_range': time_range,
            'freq_range': freq_range
        }

        logger.debug(
            "Visible region updated: %d tiles (lod=%d) for %s",
            len(visible_tiles), lod, view_type
        )

        # Request visible tiles
        for tile_id in visible_tiles:
            self._request_tile(view_type, tile_id, priority=5)
    
    def _compute_visible_tiles_for_region(self, time_range: Tuple[float, float],
                                          freq_range: Tuple[float, float], lod: int,
                                          tile_world_size: Tuple[float, float]) -> List[Tuple[int, int, int]]:
        """Compute the set of tile ids needed to cover the requested region."""
        if tile_world_size[0] <= 0 or tile_world_size[1] <= 0:
            return []

        time_start_idx = int(math.floor(time_range[0] / tile_world_size[0]))
        time_end_idx = int(math.ceil(time_range[1] / tile_world_size[0])) - 1
        freq_start_idx = int(math.floor(freq_range[0] / tile_world_size[1]))
        freq_end_idx = int(math.ceil(freq_range[1] / tile_world_size[1])) - 1

        time_end_idx = max(time_start_idx, time_end_idx)
        freq_end_idx = max(freq_start_idx, freq_end_idx)

        visible_tiles: List[Tuple[int, int, int]] = []
        for t_idx in range(time_start_idx, time_end_idx + 1):
            if t_idx < 0:
                continue
            for f_idx in range(freq_start_idx, freq_end_idx + 1):
                if f_idx < 0:
                    continue
                visible_tiles.append((t_idx, f_idx, lod))

        return visible_tiles

    def _tile_id_to_world_bounds(self, view_type: str, tile_id: Tuple[int, int, int]) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Convert a tile id into world-space time and frequency ranges."""
        config = self.view_configs[view_type]
        time_tile, freq_tile, lod = tile_id
        lod_scale = 2 ** lod

        tile_time_world = config.tile_world_size[0] * lod_scale
        tile_freq_world = config.tile_world_size[1] * lod_scale

        time_start = time_tile * tile_time_world
        time_end = time_start + tile_time_world

        freq_start = config.freq_range[0] + freq_tile * tile_freq_world
        freq_end = freq_start + tile_freq_world
        freq_end = min(freq_end, config.freq_range[1])

        return (time_start, time_end), (freq_start, freq_end)

    def _request_tile(self, view_type: str, tile_id: Tuple, priority: int = 5):
        """Request a tile to be loaded.
        
        Args:
            view_type: Type of view
            tile_id: (time_tile, freq_tile, lod)
            priority: Request priority (lower = higher priority)
        """
        atlas = self.atlases.get(view_type)
        config = self.view_configs.get(view_type)
        if config is None:
            logger.error(f"Missing view config for {view_type}, cannot request tile")
            return

        # Check if already loaded in atlas
        if atlas is not None and tile_id in getattr(atlas, 'tile_map', {}):
            return  # Already loaded

        time_range, freq_range = self._tile_id_to_world_bounds(view_type, tile_id)

        # Try to get from cache first
        cached_data = self.tile_cache.get(view_type, time_range, freq_range, tile_id[2])

        if cached_data is not None:
            if atlas is not None:
                atlas.load_tile(
                    tile_id,
                    cached_data,
                    (time_range[0], time_range[1], freq_range[0], freq_range[1])
                )
            logger.debug(f"Loaded tile %s from cache for %s", tile_id, view_type)
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
                resolution_level=tile_id[2],
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
        
        time_range, freq_range = self._tile_id_to_world_bounds(view_type, tile_id)
        lod = tile_id[2]

        # Store in cache and build mipmap pyramid
        if lod == 0:
            self.mipmap_pyramid.build_pyramid(view_type, time_range, freq_range, data)
            logger.debug("Built mipmap pyramid for tile %s at %s", tile_id, time_range)
        else:
            self.tile_cache.put(view_type, time_range, freq_range, data, lod)

        atlas = self.atlases.get(view_type)
        if atlas is not None:
            atlas.load_tile(
                tile_id,
                data,
                (time_range[0], time_range[1], freq_range[0], freq_range[1])
            )
        
        logger.debug("Tile %s computed and loaded for %s", tile_id, view_type)
    
    def process_pending_requests(self, audio_data: Optional[np.ndarray] = None, max_tiles: int = 5):
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
                
                tile_audio, segment_start = self._get_audio_for_request(
                    request.view_type, request.time_range, audio_data
                )
                if tile_audio is None or tile_audio.size == 0:
                    logger.warning("No audio data available for tile %s", request.view_type)
                    continue

                # Compute tile based on view type
                if request.view_type == 'spectrogram':
                    data = self._compute_spectrogram_tile(
                        engine, tile_audio, segment_start,
                        request.time_range, request.freq_range,
                        request.resolution_level)
                elif request.view_type == 'cepstrogram':
                    data = self._compute_cepstrogram_tile(
                        engine, tile_audio, segment_start,
                        request.time_range, request.freq_range,
                        request.resolution_level)
                elif request.view_type == 'fk_transform':
                    data = self._compute_fk_tile(
                        engine, tile_audio, segment_start,
                        request.time_range, request.freq_range,
                        request.resolution_level)
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
    
    def _get_audio_for_request(self, view_type: str, time_range: Tuple[float, float],
                               global_audio: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], float]:
        """Fetch audio samples for the given tile request.

        Returns:
            (audio_segment, segment_start_time_seconds)
        """
        if global_audio is not None:
            return global_audio, 0.0

        if self.audio_loader is None:
            logger.error("Audio loader not set; cannot stream audio for tile requests")
            return None, 0.0

        engine = self.engines.get(view_type) or self.engines.get('spectrogram')
        if engine is None:
            logger.error("No engine available to infer sampling parameters for %s", view_type)
            return None, 0.0

        sample_rate = getattr(engine, 'sample_rate', None)
        if sample_rate is None or sample_rate <= 0:
            logger.error("Engine for %s missing sample_rate", view_type)
            return None, 0.0

        fft_size = getattr(engine, 'fft_size', 2048)
        hop_length = getattr(engine, 'hop_length', max(fft_size // 4, 1))
        overlap_samples = max(0, fft_size - hop_length)

        start_sample = max(0, int(math.floor(time_range[0] * sample_rate)) - overlap_samples)
        end_sample = int(math.ceil(time_range[1] * sample_rate)) + overlap_samples
        num_samples = max(fft_size, end_sample - start_sample)

        chunk = self.audio_loader.get_chunk(start_sample, num_samples)
        segment_start_time = start_sample / sample_rate
        return chunk, segment_start_time

    def _compute_spectrogram_tile(self, engine, audio_segment: np.ndarray,
                                   segment_start_time: float,
                                   time_range: Tuple[float, float],
                                   freq_range: Tuple[float, float],
                                   lod: int = 0) -> np.ndarray:
        """Compute a spectrogram tile."""
        config = self.view_configs.get('spectrogram')
        lod_factor = max(1, 2 ** lod)

        target_height = max(1, config.tile_height_bins // lod_factor) if config else 256
        target_width = max(1, config.tile_width_frames // lod_factor) if config else 2048
        target_shape = (target_height, target_width)

        if audio_segment is None or audio_segment.size < engine.fft_size:
            logger.warning("Audio segment too small for spectrogram tile computation")
            return np.zeros(target_shape, dtype=np.float32)

        try:
            magnitude_db, frequencies, times = engine.compute_stft_batched(audio_segment)
        except Exception as e:
            logger.warning(f"Batched FFT failed for tile, using fallback: {e}")
            magnitude_db, frequencies, times = engine.compute_stft_cpu(audio_segment)

        adjusted_times = times + segment_start_time
        time_mask = (adjusted_times >= time_range[0]) & (adjusted_times <= time_range[1])
        if not np.any(time_mask):
            logger.debug("No STFT frames fall within requested time range; using closest frames")
            closest_indices = np.argsort(np.abs(adjusted_times - np.mean(time_range)))[:target_width]
            time_mask = np.zeros_like(adjusted_times, dtype=bool)
            time_mask[closest_indices] = True

        magnitude_db = magnitude_db[:, time_mask]

        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        if not np.any(freq_mask):
            logger.debug("No frequency bins within requested range; selecting closest bins")
            closest_freq_idx = np.argsort(np.abs(frequencies - np.mean(freq_range)))[:target_height]
            freq_mask = np.zeros_like(frequencies, dtype=bool)
            freq_mask[closest_freq_idx] = True

        magnitude_db = magnitude_db[freq_mask, :]

        if lod_factor > 1:
            magnitude_db = magnitude_db[::lod_factor, ::lod_factor]

        if magnitude_db.size == 0:
            return np.zeros(target_shape, dtype=np.float32)

        result = np.zeros(target_shape, dtype=np.float32)
        height = min(target_shape[0], magnitude_db.shape[0])
        width = min(target_shape[1], magnitude_db.shape[1])
        result[:height, :width] = magnitude_db[:height, :width]

        return result
    
    def _compute_cepstrogram_tile(self, engine, audio_segment: np.ndarray,
                                   segment_start_time: float,
                                   time_range: Tuple[float, float],
                                   freq_range: Tuple[float, float],
                                   lod: int = 0) -> np.ndarray:
        """Compute a cepstrogram tile."""
        # Get spectrogram first
        spec_engine = self.engines.get('spectrogram')
        if not spec_engine:
            raise RuntimeError("Spectrogram engine required for cepstrogram")
        
        magnitude_db = self._compute_spectrogram_tile(
            spec_engine, audio_segment, segment_start_time,
            time_range, freq_range, lod)
        
        # Compute cepstrogram from spectrogram
        frequencies = np.linspace(freq_range[0], freq_range[1], magnitude_db.shape[0])
        cepstral = engine.compute_cepstrogram_from_spectrogram(magnitude_db, frequencies)
        
        return cepstral.astype(np.float32)
    
    def _compute_fk_tile(self, engine, audio_segment: np.ndarray,
                         segment_start_time: float,
                         time_range: Tuple[float, float],
                         freq_range: Tuple[float, float],
                         lod: int = 0) -> np.ndarray:
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
            if atlas is not None:
                stats['atlases'][view_type] = atlas.get_stats()
            else:
                stats['atlases'][view_type] = {}
        
        return stats
    
    def clear(self, view_type: Optional[str] = None):
        """Clear tiles for a view type or all views."""
        if view_type:
            if view_type in self.atlases:
                if self.atlases[view_type] is not None:
                    self.atlases[view_type].clear()
            self.tile_cache.clear(view_type)
        else:
            for atlas in self.atlases.values():
                if atlas is not None:
                    atlas.clear()
            self.tile_cache.clear()
        
        logger.info(f"Cleared tiles: {view_type or 'all'}")

