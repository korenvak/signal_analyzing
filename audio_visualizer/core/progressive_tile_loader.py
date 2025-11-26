"""
Progressive Tile Loader
Handles background tile computation and Level-of-Detail (LOD) management.
Provides instant zoom/pan response with progressive refinement.
"""

import threading
import queue
import time
import logging
from typing import Dict, List, Tuple, Optional, Callable, Set
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class TilePriority(Enum):
    """Tile loading priority levels."""
    IMMEDIATE = 0    # Currently visible tiles
    HIGH = 1        # Adjacent to visible area
    MEDIUM = 2      # Next zoom level
    LOW = 3         # Background prefetch
    BACKGROUND = 4  # Far prefetch


@dataclass
class TileRequest:
    """Progressive tile request with priority and LOD."""
    view_type: str
    time_range: Tuple[float, float]
    freq_range: Tuple[float, float]
    resolution_level: int
    priority: TilePriority
    callback: Optional[Callable] = None
    request_time: float = None
    
    def __post_init__(self):
        if self.request_time is None:
            self.request_time = time.time()
    
    def __hash__(self):
        return hash((
            self.view_type,
            round(self.time_range[0], 3),
            round(self.time_range[1], 3),
            round(self.freq_range[0], 1),
            round(self.freq_range[1], 1),
            self.resolution_level
        ))
    
    def __lt__(self, other):
        """Make TileRequest sortable for priority queue."""
        if not isinstance(other, TileRequest):
            return NotImplemented
        # Compare by priority first, then by request time
        if self.priority != other.priority:
            return self.priority.value < other.priority.value
        return self.request_time < other.request_time
    
    def __eq__(self, other):
        """Equality comparison for TileRequest."""
        if not isinstance(other, TileRequest):
            return NotImplemented
        return hash(self) == hash(other)


class ViewportTracker:
    """Tracks user viewport and predicts tile needs."""
    
    def __init__(self):
        self.current_viewport = None
        self.viewport_history = []
        self.max_history = 10
        self.prediction_enabled = True
    
    def update_viewport(self, time_range: Tuple[float, float], 
                       freq_range: Tuple[float, float], zoom_level: float):
        """Update current viewport and predict future needs."""
        viewport = {
            'time_range': time_range,
            'freq_range': freq_range,
            'zoom_level': zoom_level,
            'timestamp': time.time()
        }
        
        # Update current viewport
        self.current_viewport = viewport
        
        # Add to history
        self.viewport_history.append(viewport)
        if len(self.viewport_history) > self.max_history:
            self.viewport_history.pop(0)
    
    def predict_next_viewport(self) -> Optional[Dict]:
        """Predict where user is likely to navigate next."""
        if not self.prediction_enabled or len(self.viewport_history) < 3:
            return None
        
        # Simple prediction based on movement pattern
        recent = self.viewport_history[-3:]
        
        # Calculate movement trends
        time_trend = (recent[-1]['time_range'][0] - recent[0]['time_range'][0]) / 2
        zoom_trend = (recent[-1]['zoom_level'] - recent[0]['zoom_level']) / 2
        
        # Predict next position
        current = self.current_viewport
        if current:
            predicted_time = (
                current['time_range'][0] + time_trend,
                current['time_range'][1] + time_trend
            )
            predicted_zoom = current['zoom_level'] + zoom_trend
            
            return {
                'time_range': predicted_time,
                'freq_range': current['freq_range'],  # Usually stable
                'zoom_level': predicted_zoom
            }
        
        return None


class ProgressiveTileLoader:
    """Manages progressive tile loading with LOD and background computation."""
    
    def __init__(self, tile_cache, engines: Dict, max_workers: int = 3):
        """Initialize progressive tile loader.
        
        Args:
            tile_cache: TileCache instance
            engines: Dictionary of computation engines
            max_workers: Maximum background computation threads
        """
        self.tile_cache = tile_cache
        self.engines = engines
        self.max_workers = max_workers
        
        # Request management
        self.request_queue = queue.PriorityQueue()
        self.active_requests: Set[TileRequest] = set()
        self.completed_tiles: Dict[int, np.ndarray] = {}
        
        # Worker threads
        self.workers = []
        self.running = False
        self.worker_lock = threading.RLock()
        
        # Viewport tracking
        self.viewport_tracker = ViewportTracker()
        
        # Performance statistics
        self.stats = {
            'tiles_requested': 0,
            'tiles_computed': 0,
            'tiles_cached_hits': 0,
            'background_tiles': 0,
            'avg_computation_time': 0.0
        }
        
        logger.info(f"ProgressiveTileLoader initialized with {max_workers} workers")
    
    def start(self):
        """Start background worker threads."""
        if self.running:
            return
        
        self.running = True
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TileWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {len(self.workers)} tile loader worker threads")
    
    def stop(self):
        """Stop all worker threads."""
        self.running = False
        
        # Clear queue to unblock workers
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except queue.Empty:
                break
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2.0)
        
        self.workers.clear()
        logger.info("Progressive tile loader stopped")
    
    def request_tiles_for_viewport(self, view_type: str, time_range: Tuple[float, float],
                                  freq_range: Tuple[float, float], zoom_level: float,
                                  callback: Optional[Callable] = None) -> List[TileRequest]:
        """Request tiles for current viewport with progressive loading.
        
        Args:
            view_type: Type of visualization
            time_range: Time range to display
            freq_range: Frequency range to display
            zoom_level: Current zoom level
            callback: Callback for tile completion
            
        Returns:
            List of tile requests submitted
        """
        # Update viewport tracking
        self.viewport_tracker.update_viewport(time_range, freq_range, zoom_level)
        
        requests = []
        
        # 1. Request immediate tiles (current view)
        immediate_tiles = self._generate_tile_requests(
            view_type, time_range, freq_range, zoom_level, TilePriority.IMMEDIATE
        )
        
        for request in immediate_tiles:
            request.callback = callback
            requests.append(request)
            self._submit_request(request)
        
        # 2. Request adjacent tiles (high priority)
        adjacent_tiles = self._generate_adjacent_tile_requests(
            view_type, time_range, freq_range, zoom_level, TilePriority.HIGH
        )
        
        for request in adjacent_tiles:
            requests.append(request)
            self._submit_request(request)
        
        # 3. Request next LOD level (medium priority)
        if zoom_level > 0:  # If not at highest resolution
            next_lod_tiles = self._generate_tile_requests(
                view_type, time_range, freq_range, zoom_level - 1, TilePriority.MEDIUM
            )
            
            for request in next_lod_tiles:
                requests.append(request)
                self._submit_request(request)
        
        # 4. Predictive prefetch (low priority)
        predicted_viewport = self.viewport_tracker.predict_next_viewport()
        if predicted_viewport:
            predicted_tiles = self._generate_tile_requests(
                view_type, 
                predicted_viewport['time_range'],
                predicted_viewport['freq_range'],
                predicted_viewport['zoom_level'],
                TilePriority.LOW
            )
            
            for request in predicted_tiles:
                requests.append(request)
                self._submit_request(request)
        
        self.stats['tiles_requested'] += len(requests)
        logger.debug(f"Requested {len(requests)} tiles for viewport")
        
        return requests
    
    def _generate_tile_requests(self, view_type: str, time_range: Tuple[float, float],
                               freq_range: Tuple[float, float], resolution_level: int,
                               priority: TilePriority) -> List[TileRequest]:
        """Generate tile requests for a given area and resolution."""
        requests = []
        
        # Calculate tile size based on resolution level
        base_tile_duration = 60.0  # Base tile: 60 seconds
        base_tile_freq_range = 4000.0  # Base tile: 4kHz
        
        # Scale by resolution level (higher level = larger tiles, lower resolution)
        tile_duration = base_tile_duration * (2 ** resolution_level)
        tile_freq_span = base_tile_freq_range * (2 ** resolution_level)
        
        # Generate tile grid
        time_start = time_range[0]
        time_end = time_range[1]
        freq_start = freq_range[0]
        freq_end = freq_range[1]
        
        # Time tiles
        current_time = time_start
        while current_time < time_end:
            tile_time_end = min(current_time + tile_duration, time_end)
            
            # Frequency tiles
            current_freq = freq_start
            while current_freq < freq_end:
                tile_freq_end = min(current_freq + tile_freq_span, freq_end)
                
                request = TileRequest(
                    view_type=view_type,
                    time_range=(current_time, tile_time_end),
                    freq_range=(current_freq, tile_freq_end),
                    resolution_level=resolution_level,
                    priority=priority
                )
                
                requests.append(request)
                current_freq = tile_freq_end
            
            current_time = tile_time_end
        
        return requests
    
    def _generate_adjacent_tile_requests(self, view_type: str, time_range: Tuple[float, float],
                                        freq_range: Tuple[float, float], resolution_level: int,
                                        priority: TilePriority) -> List[TileRequest]:
        """Generate requests for tiles adjacent to current viewport."""
        requests = []
        
        # Expand viewport by 50% in each direction for prefetch
        time_span = time_range[1] - time_range[0]
        freq_span = freq_range[1] - freq_range[0]
        
        expanded_time_range = (
            time_range[0] - time_span * 0.5,
            time_range[1] + time_span * 0.5
        )
        
        expanded_freq_range = (
            max(0, freq_range[0] - freq_span * 0.5),
            freq_range[1] + freq_span * 0.5
        )
        
        # Generate tiles for expanded area (excluding current viewport)
        expanded_requests = self._generate_tile_requests(
            view_type, expanded_time_range, expanded_freq_range, resolution_level, priority
        )
        
        current_requests = self._generate_tile_requests(
            view_type, time_range, freq_range, resolution_level, TilePriority.IMMEDIATE
        )
        
        # Filter out current viewport tiles
        current_hashes = {hash(req) for req in current_requests}
        adjacent_requests = [req for req in expanded_requests 
                           if hash(req) not in current_hashes]
        
        return adjacent_requests
    
    def _submit_request(self, request: TileRequest):
        """Submit a tile request to the worker queue."""
        if request in self.active_requests:
            return  # Already requested
        
        # Check cache first
        cached_tile = self.tile_cache.get(
            request.view_type,
            request.time_range,
            request.freq_range,
            request.resolution_level
        )
        
        if cached_tile is not None:
            # Cache hit - notify immediately
            self.stats['tiles_cached_hits'] += 1
            if request.callback:
                request.callback(request, cached_tile, None)
            return
        
        # Add to active requests
        self.active_requests.add(request)
        
        # Submit to queue with priority
        priority_value = request.priority.value
        self.request_queue.put((priority_value, time.time(), request))
    
    def _worker_loop(self):
        """Worker thread loop for processing tile requests."""
        while self.running:
            try:
                # Get next request with timeout
                try:
                    priority, timestamp, request = self.request_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the request
                start_time = time.time()
                self._process_request(request)
                computation_time = time.time() - start_time
                
                # Update statistics
                self.stats['tiles_computed'] += 1
                if self.stats['tiles_computed'] == 1:
                    self.stats['avg_computation_time'] = computation_time
                else:
                    # Running average
                    self.stats['avg_computation_time'] = (
                        self.stats['avg_computation_time'] * 0.9 + 
                        computation_time * 0.1
                    )
                
                # Remove from active requests
                with self.worker_lock:
                    self.active_requests.discard(request)
                
                self.request_queue.task_done()
                
            except Exception as e:
                logger.error(f"Tile worker error: {e}")
    
    def _process_request(self, request: TileRequest):
        """Process a single tile request."""
        try:
            # Get appropriate engine
            engine = self.engines.get(request.view_type)
            if not engine:
                raise ValueError(f"No engine for view type: {request.view_type}")
            
            # Compute tile based on view type
            if request.view_type == 'spectrogram':
                tile_data = self._compute_spectrogram_tile(request)
            elif request.view_type == 'cepstrogram':
                tile_data = self._compute_cepstrogram_tile(request)
            elif request.view_type == 'fk_transform':
                tile_data = self._compute_fk_tile(request)
            else:
                raise ValueError(f"Unknown view type: {request.view_type}")
            
            # Store in cache
            self.tile_cache.put(
                request.view_type,
                request.time_range,
                request.freq_range,
                tile_data,
                request.resolution_level
            )
            
            # Notify callback
            if request.callback:
                request.callback(request, tile_data, None)
            
            logger.debug(f"Completed tile: {request.view_type} "
                        f"{request.time_range} LOD={request.resolution_level}")
            
        except Exception as e:
            logger.error(f"Failed to process tile request: {e}")
            if request.callback:
                request.callback(request, None, e)
    
    def _compute_spectrogram_tile(self, request: TileRequest) -> np.ndarray:
        """Compute spectrogram tile data."""
        engine = self.engines.get('spectrogram')
        if not engine:
            raise ValueError("No spectrogram engine available")
        
        # Get audio data from engine (assuming it's loaded)
        if not hasattr(engine, 'audio_data') or engine.audio_data is None:
            logger.warning("No audio data available for tile computation")
            return np.zeros((256, 512), dtype=np.float32)
        
        audio_data = engine.audio_data
        sample_rate = engine.sample_rate
        
        # Extract audio chunk for time range
        start_sample = int(request.time_range[0] * sample_rate)
        end_sample = int(request.time_range[1] * sample_rate)
        
        if start_sample >= len(audio_data):
            return np.zeros((256, 512), dtype=np.float32)
        
        end_sample = min(end_sample, len(audio_data))
        audio_chunk = audio_data[start_sample:end_sample]
        
        if len(audio_chunk) < engine.fft_size:
            logger.warning(f"Audio chunk too small ({len(audio_chunk)} samples) for tile computation")
            return np.zeros((256, 512), dtype=np.float32)
        
        # Compute STFT
        try:
            magnitude_db, frequencies, times = engine.compute_stft_batched(audio_chunk)
        except Exception as e:
            logger.warning(f"Batched FFT failed for tile, using fallback: {e}")
            magnitude_db, frequencies, times = engine.compute_stft_cpu(audio_chunk)
        
        # Filter frequency range
        if len(frequencies) > 0 and request.freq_range[1] < frequencies[-1]:
            freq_mask = (frequencies >= request.freq_range[0]) & (frequencies <= request.freq_range[1])
            if np.any(freq_mask):
                magnitude_db = magnitude_db[freq_mask, :]
        
        return magnitude_db.astype(np.float32) if magnitude_db.size > 0 else np.zeros((256, 512), dtype=np.float32)
    
    def _compute_cepstrogram_tile(self, request: TileRequest) -> np.ndarray:
        """Compute cepstrogram tile data."""
        # Get spectrogram first
        spec_engine = self.engines.get('spectrogram')
        cep_engine = self.engines.get('cepstrogram')
        
        if not spec_engine or not cep_engine:
            raise ValueError("Both spectrogram and cepstrogram engines required")
        
        # Compute spectrogram tile first
        magnitude_db = self._compute_spectrogram_tile(request)
        
        if magnitude_db.size == 0:
            return np.zeros((100, 512), dtype=np.float32)
        
        # Compute cepstrogram from spectrogram
        frequencies = np.linspace(request.freq_range[0], request.freq_range[1], magnitude_db.shape[0])
        cepstral = cep_engine.compute_cepstrogram_from_spectrogram(magnitude_db, frequencies)
        
        return cepstral.astype(np.float32)
    
    def _compute_fk_tile(self, request: TileRequest) -> np.ndarray:
        """Compute F-K transform tile data."""
        engine = self.engines.get('fk_transform')
        if not engine:
            logger.warning("No F-K transform engine available, returning placeholder")
            return np.zeros((256, 256), dtype=np.float32)
        
        # TODO: Implement F-K transform tile computation in Phase 5
        # For now, return placeholder data
        return np.zeros((256, 256), dtype=np.float32)
    
    def get_statistics(self) -> Dict:
        """Get loader performance statistics."""
        stats = self.stats.copy()
        stats['active_requests'] = len(self.active_requests)
        stats['queue_size'] = self.request_queue.qsize()
        stats['cache_hit_rate'] = (
            self.stats['tiles_cached_hits'] / 
            max(1, self.stats['tiles_requested']) * 100
        )
        return stats
    
    def clear_requests(self):
        """Clear all pending requests."""
        # Clear queue
        while not self.request_queue.empty():
            try:
                self.request_queue.get_nowait()
            except queue.Empty:
                break
        
        # Clear active requests
        with self.worker_lock:
            self.active_requests.clear()
        
        logger.debug("Cleared all pending tile requests")


# Global instance
_progressive_loader = None

def get_progressive_tile_loader(tile_cache=None, engines=None) -> ProgressiveTileLoader:
    """Get the global progressive tile loader instance."""
    global _progressive_loader
    if _progressive_loader is None and tile_cache and engines:
        _progressive_loader = ProgressiveTileLoader(tile_cache, engines)
        _progressive_loader.start()
    return _progressive_loader