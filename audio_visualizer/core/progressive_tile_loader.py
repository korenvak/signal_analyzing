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
import concurrent.futures

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
    
    def __init__(self, tile_cache, engines: Dict, max_workers: int = None):
        """Initialize progressive tile loader.
        
        Args:
            tile_cache: TileCache instance
            engines: Dictionary of computation engines
            max_workers: Maximum background computation threads (auto-detect if None)
        """
        self.tile_cache = tile_cache
        self.engines = engines
        
        # Auto-detect optimal worker count
        if max_workers is None:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            # Use more workers for better parallelization (up to 8)
            # Reserve 1-2 cores for UI and other tasks
            self.max_workers = min(max(2, cpu_count - 2), 8)
        else:
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
                                  callback: Optional[Callable] = None,
                                  lod_offset: int = 0) -> List[TileRequest]:
        """Request tiles for current viewport with progressive loading.
        
        Args:
            view_type: Type of visualization
            time_range: Time range to display
            freq_range: Frequency range to display
            zoom_level: Current zoom level
            callback: Callback for tile completion
            lod_offset: Additional LOD offset for adaptive quality (higher = lower quality)
            
        Returns:
            List of tile requests submitted
        """
        # Update viewport tracking
        self.viewport_tracker.update_viewport(time_range, freq_range, zoom_level)
        
        requests = []
        
        # Select appropriate LOD based on zoom level
        # When zoomed out, use lower LOD to reduce computation cost
        # Apply adaptive quality offset for fast movement
        base_lod = self._select_lod_for_zoom(zoom_level)
        effective_lod = min(base_lod + lod_offset, 5)  # Cap at LOD 5
        
        # 1. Request immediate tiles (current view) - use effective LOD (with adaptive quality)
        immediate_tiles = self._generate_tile_requests(
            view_type, time_range, freq_range, effective_lod, TilePriority.IMMEDIATE
        )
        
        for request in immediate_tiles:
            request.callback = callback
            requests.append(request)
            self._submit_request(request)
        
        # 2. Request adjacent tiles (high priority) - more aggressive prefetch
        # Expand by 100% instead of 50% for better prefetching
        # Use effective LOD for adjacent tiles too (faster during movement)
        adjacent_tiles = self._generate_adjacent_tile_requests(
            view_type, time_range, freq_range, effective_lod, TilePriority.HIGH, expand_factor=1.0
        )
        
        for request in adjacent_tiles:
            requests.append(request)
            self._submit_request(request)
        
        # 3. Request higher quality LOD if zoomed in (progressive refinement)
        if zoom_level > 1.0 and base_lod > 0:
            # Request one level higher quality for smooth zoom-in
            higher_lod = base_lod - 1
            higher_quality_tiles = self._generate_tile_requests(
                view_type, time_range, freq_range, higher_lod, TilePriority.MEDIUM
            )
            
            for request in higher_quality_tiles:
                requests.append(request)
                self._submit_request(request)
        
        # 4. Request lower quality LOD if zoomed out (for faster initial display)
        if zoom_level < 1.0 and base_lod < 3:
            # Request one level lower quality for faster display when zoomed out
            lower_lod = base_lod + 1
            lower_quality_tiles = self._generate_tile_requests(
                view_type, time_range, freq_range, lower_lod, TilePriority.MEDIUM
            )
            
            for request in lower_quality_tiles:
                requests.append(request)
                self._submit_request(request)
        
        # 5. Predictive prefetch (low priority) - more aggressive
        predicted_viewport = self.viewport_tracker.predict_next_viewport()
        if predicted_viewport:
            predicted_lod = self._select_lod_for_zoom(predicted_viewport['zoom_level'])
            predicted_tiles = self._generate_tile_requests(
                view_type, 
                predicted_viewport['time_range'],
                predicted_viewport['freq_range'],
                predicted_lod,
                TilePriority.LOW
            )
            
            for request in predicted_tiles:
                requests.append(request)
                self._submit_request(request)
        
        self.stats['tiles_requested'] += len(requests)
        if lod_offset > 0:
            logger.debug(f"Requested {len(requests)} tiles for viewport (zoom={zoom_level:.2f}, LOD={effective_lod}, adaptive quality active)")
        else:
            logger.debug(f"Requested {len(requests)} tiles for viewport (zoom={zoom_level:.2f}, LOD={effective_lod})")
        
        return requests
    
    def _select_lod_for_zoom(self, zoom_level: float) -> int:
        """Select appropriate LOD level based on zoom.
        
        Args:
            zoom_level: Current zoom level (1.0 = fit to screen, >1.0 = zoomed in, <1.0 = zoomed out)
        
        Returns:
            LOD level (0 = full resolution, higher = more downsampled)
        """
        # When zoomed out (zoom_level < 1.0), use lower resolution tiles to reduce computation
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
    
    def _generate_tile_requests(self, view_type: str, time_range: Tuple[float, float],
                               freq_range: Tuple[float, float], resolution_level: int,
                               priority: TilePriority) -> List[TileRequest]:
        """Generate tile requests for a given area and resolution.
        
        Uses adaptive tile sizing: larger tiles for lower LOD levels to reduce computation.
        """
        requests = []
        
        # Adaptive tile sizing based on LOD
        # Lower LOD (higher resolution_level) = larger tiles = fewer tiles to compute
        # This reduces computation cost when zoomed out
        base_tile_duration = 30.0  # Base tile: 30 seconds (smaller for better granularity)
        base_tile_freq_range = 2000.0  # Base tile: 2kHz
        
        # Scale by resolution level (higher level = larger tiles, lower resolution)
        # LOD 0: 30s x 2kHz tiles
        # LOD 1: 60s x 4kHz tiles
        # LOD 2: 120s x 8kHz tiles
        # etc.
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
                                        priority: TilePriority, expand_factor: float = 0.5) -> List[TileRequest]:
        """Generate requests for tiles adjacent to current viewport.
        
        Args:
            view_type: Type of view
            time_range: Current time range
            freq_range: Current frequency range
            resolution_level: LOD level
            priority: Request priority
            expand_factor: How much to expand viewport (0.5 = 50%, 1.0 = 100%)
        """
        requests = []
        
        # Expand viewport by expand_factor in each direction for prefetch
        time_span = time_range[1] - time_range[0]
        freq_span = freq_range[1] - freq_range[0]
        
        expanded_time_range = (
            time_range[0] - time_span * expand_factor,
            time_range[1] + time_span * expand_factor
        )
        
        expanded_freq_range = (
            max(0, freq_range[0] - freq_span * expand_factor),
            freq_range[1] + freq_span * expand_factor
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
        """Worker thread loop for processing tile requests with batching."""
        while self.running:
            try:
                # Batch processing: collect multiple requests of same priority
                batch = []
                batch_priority = None
                
                # Get first request
                try:
                    priority, timestamp, request = self.request_queue.get(timeout=0.1)
                    batch.append(request)
                    batch_priority = priority
                except queue.Empty:
                    continue
                
                # Collect additional requests of same priority (up to 4 tiles per batch)
                # This allows parallel computation of multiple tiles
                max_batch_size = 4
                try:
                    while len(batch) < max_batch_size:
                        priority, timestamp, request = self.request_queue.get_nowait()
                        if priority == batch_priority:
                            batch.append(request)
                        else:
                            # Different priority - put back and process current batch
                            self.request_queue.put((priority, timestamp, request))
                            break
                except queue.Empty:
                    pass
                
                # Process batch
                start_time = time.time()
                if len(batch) == 1:
                    # Single request - process normally
                    self._process_request(batch[0])
                    computation_time = time.time() - start_time
                else:
                    # Batch processing - process multiple tiles in parallel
                    self._process_batch(batch)
                    computation_time = time.time() - start_time
                    logger.debug(f"Processed batch of {len(batch)} tiles in {computation_time:.3f}s")
                
                # Update statistics
                self.stats['tiles_computed'] += len(batch)
                if self.stats['tiles_computed'] == len(batch):
                    self.stats['avg_computation_time'] = computation_time / len(batch)
                else:
                    # Running average
                    avg_time_per_tile = computation_time / len(batch)
                    self.stats['avg_computation_time'] = (
                        self.stats['avg_computation_time'] * 0.9 + 
                        avg_time_per_tile * 0.1
                    )
                
                # Remove from active requests
                with self.worker_lock:
                    for request in batch:
                        self.active_requests.discard(request)
                        self.request_queue.task_done()
                
            except Exception as e:
                logger.error(f"Tile worker error: {e}")
                # Mark any batch items as done
                for request in batch:
                    self.request_queue.task_done()
    
    def _process_batch(self, requests: List[TileRequest]):
        """Process multiple tile requests in batch for better performance.
        
        Args:
            requests: List of tile requests to process
        """
        # Group requests by view type for efficient batch processing
        by_view_type = {}
        for request in requests:
            if request.view_type not in by_view_type:
                by_view_type[request.view_type] = []
            by_view_type[request.view_type].append(request)
        
        # Process each view type's requests
        for view_type, view_requests in by_view_type.items():
            # Process in parallel using threads (for CPU-bound work)
            # For GPU work, batching is handled by the engine
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(view_requests), 4)) as executor:
                futures = [executor.submit(self._process_request, req) for req in view_requests]
                concurrent.futures.wait(futures)
    
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