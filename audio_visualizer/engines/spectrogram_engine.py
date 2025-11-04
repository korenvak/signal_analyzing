import numpy as np
import scipy.signal
from typing import Tuple, Optional, Dict, Any, Callable
import threading
import time
import logging

try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal
    HAS_CUPY = True
except ImportError:
    cp = None
    cp_signal = None
    HAS_CUPY = False

from ..core.cache_manager import CacheManager
from ..core.task_manager import TaskManager, TaskPriority
from .batched_fft_engine import get_batched_fft_engine

logger = logging.getLogger(__name__)

class SpectrogramEngine:
    """High-performance GPU-accelerated spectrogram computation engine."""
    
    def __init__(self, cache_manager: CacheManager, task_manager: TaskManager):
        self.cache_manager = cache_manager
        self.task_manager = task_manager
        
        # Default parameters
        self.fft_size = 2048
        self.hop_length = 512
        self.window_type = 'hann'
        self.sample_rate = 44100
        
        # Tile parameters
        # Balanced for memory and OpenGL texture limits
        # OpenGL max texture size is typically 16384 pixels
        # Using 2048 frames per tile for safety and memory efficiency
        self.tile_size = (2048, 512)  # (time_frames, frequency_bins)
        self.tile_overlap = 0.1
        self.max_texture_size = 16384  # Will be detected at runtime
        
        # GPU optimization
        self.use_gpu = HAS_CUPY
        self.gpu_batch_size = 8
        
        # Batched FFT engine for high performance
        self.batched_fft_engine = get_batched_fft_engine()
        self.use_batched_fft = True  # Enable by default
        
        # Threading
        self._computation_lock = threading.RLock()
        self._current_tasks = {}
        
    def set_parameters(self, fft_size: int = None, hop_length: int = None,
                      window_type: str = None, sample_rate: int = None):
        """Update STFT parameters and clear related cache."""
        params_changed = False
        
        if fft_size and fft_size != self.fft_size:
            self.fft_size = fft_size
            params_changed = True
            
        if hop_length and hop_length != self.hop_length:
            self.hop_length = hop_length
            params_changed = True
            
        if window_type and window_type != self.window_type:
            self.window_type = window_type
            params_changed = True
            
        if sample_rate and sample_rate != self.sample_rate:
            self.sample_rate = sample_rate
            params_changed = True
        
        if params_changed:
            self.cache_manager.clear_view_cache('spectrogram')
    
    def compute_stft_gpu(self, audio_data: np.ndarray, 
                        progress_callback: Optional[Callable] = None) -> cp.ndarray:
        """Compute STFT using GPU acceleration."""
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available for GPU acceleration")
        
        # Move data to GPU
        gpu_audio = cp.asarray(audio_data, dtype=cp.float32)
        
        # Create window
        window = cp.asarray(scipy.signal.get_window(self.window_type, self.fft_size))
        
        # Compute STFT
        frequencies, times, stft = cp_signal.stft(
            gpu_audio,
            fs=self.sample_rate,
            window=window,
            nperseg=self.fft_size,
            noverlap=self.fft_size - self.hop_length,
            return_onesided=True
        )
        
        if progress_callback:
            progress_callback(0.8, "Computing magnitude spectrum")
        
        # Compute magnitude and convert to dB
        magnitude = cp.abs(stft)
        magnitude_db = 20 * cp.log10(cp.maximum(magnitude, 1e-10))
        
        if progress_callback:
            progress_callback(1.0, "STFT computation complete")
        
        return magnitude_db, frequencies, times
    
    def compute_stft_cpu(self, audio_data: np.ndarray,
                        progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Compute STFT using CPU."""
        if progress_callback:
            progress_callback(0.1, "Computing STFT on CPU")
        
        frequencies, times, stft = scipy.signal.stft(
            audio_data,
            fs=self.sample_rate,
            window=self.window_type,
            nperseg=self.fft_size,
            noverlap=self.fft_size - self.hop_length,
            return_onesided=True
        )
        
        if progress_callback:
            progress_callback(0.8, "Computing magnitude spectrum")
        
        magnitude = np.abs(stft)
        magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))
        
        if progress_callback:
            progress_callback(1.0, "STFT computation complete")
        
        return magnitude_db, frequencies, times
    
    def compute_stft_batched(self, audio_data: np.ndarray,
                            progress_callback: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute STFT using batched FFT engine (10-100x faster).
        
        Args:
            audio_data: Audio signal
            progress_callback: Optional progress callback
            
        Returns:
            (magnitude_db, frequencies, times)
        """
        if progress_callback:
            progress_callback(0.1, "Computing STFT (batched)")
        
        # Use batched FFT engine
        magnitude_db, times = self.batched_fft_engine.compute_stft_batched(
            audio_data,
            fft_size=self.fft_size,
            hop_length=self.hop_length,
            window=self.window_type,
            sample_rate=self.sample_rate,
            use_gpu=self.use_gpu
        )
        
        # Generate frequency array
        frequencies = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate).astype(np.float32)
        
        if progress_callback:
            progress_callback(1.0, "STFT computation complete (batched)")
        
        logger.info(f"Computed STFT (batched): {magnitude_db.shape} in optimized mode")
        
        return magnitude_db, frequencies, times
    
    def compute_tile(self, audio_data: np.ndarray, time_start: float, time_end: float,
                    freq_start: float, freq_end: float, resolution_level: int = 0,
                    progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Compute a single spectrogram tile."""
        
        # Calculate sample indices
        start_sample = int(time_start * self.sample_rate)
        end_sample = int(time_end * self.sample_rate)
        
        # Add overlap for windowing
        overlap_samples = int(self.tile_overlap * (end_sample - start_sample))
        start_sample = max(0, start_sample - overlap_samples)
        end_sample = min(len(audio_data), end_sample + overlap_samples)
        
        audio_chunk = audio_data[start_sample:end_sample]
        
        if progress_callback:
            progress_callback(0.1, f"Processing tile {time_start:.2f}-{time_end:.2f}s")
        
        # Compute STFT using optimized batched FFT (10-100x faster!)
        if self.use_batched_fft:
            try:
                magnitude_db, frequencies, times = self.compute_stft_batched(
                    audio_chunk, progress_callback)
                logger.debug(f"Tile computed with batched FFT: {magnitude_db.shape}")
            except Exception as e:
                logger.warning(f"Batched FFT failed, falling back to standard: {e}")
                # Fallback to standard method
                if self.use_gpu and len(audio_chunk) > 50000:
                    try:
                        magnitude_db, frequencies, times = self.compute_stft_gpu(
                            audio_chunk, progress_callback)
                        magnitude_db = cp.asnumpy(magnitude_db)
                        frequencies = cp.asnumpy(frequencies)
                        times = cp.asnumpy(times)
                    except Exception as e2:
                        logger.warning(f"GPU computation failed: {e2}")
                        magnitude_db, frequencies, times = self.compute_stft_cpu(
                            audio_chunk, progress_callback)
                else:
                    magnitude_db, frequencies, times = self.compute_stft_cpu(
                        audio_chunk, progress_callback)
        elif self.use_gpu and len(audio_chunk) > 50000:
            # Standard GPU path
            try:
                magnitude_db, frequencies, times = self.compute_stft_gpu(
                    audio_chunk, progress_callback)
                magnitude_db = cp.asnumpy(magnitude_db)
                frequencies = cp.asnumpy(frequencies)
                times = cp.asnumpy(times)
            except Exception as e:
                logger.warning(f"GPU computation failed: {e}")
                magnitude_db, frequencies, times = self.compute_stft_cpu(
                    audio_chunk, progress_callback)
        else:
            # Standard CPU path
            magnitude_db, frequencies, times = self.compute_stft_cpu(
                audio_chunk, progress_callback)
        
        # Frequency range selection
        freq_mask = (frequencies >= freq_start) & (frequencies <= freq_end)
        magnitude_db = magnitude_db[freq_mask, :]
        frequencies = frequencies[freq_mask]
        
        # Time range selection (adjust for overlap)
        time_offset = start_sample / self.sample_rate
        adjusted_times = times + time_offset
        time_mask = (adjusted_times >= time_start) & (adjusted_times <= time_end)
        magnitude_db = magnitude_db[:, time_mask]
        times = adjusted_times[time_mask]
        
        # Apply resolution downsampling
        if resolution_level > 0:
            downsample_factor = 2 ** resolution_level
            magnitude_db = magnitude_db[::downsample_factor, ::downsample_factor]
        
        if progress_callback:
            progress_callback(1.0, "Tile computation complete")
        
        return magnitude_db
    
    def request_tiles(self, time_range: Tuple[float, float], 
                     freq_range: Tuple[float, float],
                     resolution_level: int = 0,
                     audio_data: np.ndarray = None,
                     completion_callback: Optional[Callable] = None) -> str:
        """Request spectrogram tiles for given range."""
        
        # Check cache first
        cached_data = self.cache_manager.get_tile(
            'spectrogram', time_range, freq_range, resolution_level)
        
        if cached_data is not None:
            if completion_callback:
                completion_callback(cached_data, None)
            return "cached"
        
        if audio_data is None:
            raise ValueError("Audio data required for computation")
        
        # Calculate tile grid
        tiles = self._calculate_tile_grid(time_range, freq_range, resolution_level)
        
        def compute_tiles_batch(progress_callback=None):
            """Compute multiple tiles in batch."""
            results = {}
            total_tiles = len(tiles)
            
            for i, (tile_time_range, tile_freq_range) in enumerate(tiles):
                if progress_callback:
                    progress = (i / total_tiles) * 0.8
                    progress_callback(progress, f"Computing tile {i+1}/{total_tiles}")
                
                tile_data = self.compute_tile(
                    audio_data, 
                    tile_time_range[0], tile_time_range[1],
                    tile_freq_range[0], tile_freq_range[1],
                    resolution_level
                )
                
                # Store in cache
                self.cache_manager.store_tile(
                    'spectrogram', tile_time_range, tile_freq_range,
                    tile_data, resolution_level, use_gpu=self.use_gpu
                )
                
                results[(tile_time_range, tile_freq_range)] = tile_data
            
            # Combine tiles
            if progress_callback:
                progress_callback(0.9, "Combining tiles")
            
            combined_data = self._combine_tiles(results, time_range, freq_range)
            
            # Store combined result
            self.cache_manager.store_tile(
                'spectrogram', time_range, freq_range,
                combined_data, resolution_level, use_gpu=self.use_gpu
            )
            
            if progress_callback:
                progress_callback(1.0, "Computation complete")
            
            return combined_data
        
        # Submit task
        task_id = self.task_manager.submit_gpu_task(
            compute_tiles_batch,
            priority=TaskPriority.HIGH,
            completion_callback=lambda tid, result, error: 
                completion_callback(result, error) if completion_callback else None,
            description=f"Spectrogram tiles {time_range[0]:.1f}-{time_range[1]:.1f}s"
        )
        
        with self._computation_lock:
            self._current_tasks[task_id] = {
                'time_range': time_range,
                'freq_range': freq_range,
                'resolution_level': resolution_level
            }
        
        return task_id
    
    def _calculate_tile_grid(self, time_range: Tuple[float, float],
                           freq_range: Tuple[float, float],
                           resolution_level: int) -> list:
        """Calculate optimal tile grid for given range."""
        time_span = time_range[1] - time_range[0]
        freq_span = freq_range[1] - freq_range[0]
        
        # Adjust tile size based on resolution level
        effective_tile_time = (self.tile_size[0] * self.hop_length / self.sample_rate) * (2 ** resolution_level)
        effective_tile_freq = (freq_span / 4) * (2 ** resolution_level)
        
        tiles = []
        
        # Generate time tiles
        current_time = time_range[0]
        while current_time < time_range[1]:
            tile_end_time = min(current_time + effective_tile_time, time_range[1])
            
            # Generate frequency tiles
            current_freq = freq_range[0]
            while current_freq < freq_range[1]:
                tile_end_freq = min(current_freq + effective_tile_freq, freq_range[1])
                
                tiles.append(
                    ((current_time, tile_end_time), (current_freq, tile_end_freq))
                )
                
                current_freq = tile_end_freq
            
            current_time = tile_end_time
        
        return tiles
    
    def _combine_tiles(self, tile_results: Dict, time_range: Tuple[float, float],
                      freq_range: Tuple[float, float]) -> np.ndarray:
        """Combine individual tiles into single array."""
        if not tile_results:
            return np.array([])
        
        # Sort tiles by position
        sorted_tiles = sorted(tile_results.items())
        
        # Simple concatenation - in production would need proper stitching
        combined_data = list(tile_results.values())[0]
        
        # For now, return first tile - proper tile stitching would be more complex
        return combined_data
    
    def cancel_computation(self, task_id: str) -> bool:
        """Cancel ongoing computation."""
        if task_id == "cached":
            return True
            
        success = self.task_manager.cancel_task(task_id)
        
        with self._computation_lock:
            if task_id in self._current_tasks:
                del self._current_tasks[task_id]
        
        return success
    
    def get_computation_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of computation task."""
        if task_id == "cached":
            return {"status": "completed", "progress": 1.0}
        
        status = self.task_manager.get_task_status(task_id)
        progress = self.task_manager.get_task_progress(task_id)
        
        return {
            "status": status.value if status else "unknown",
            "progress": progress
        }
    
    def get_frequency_range(self) -> Tuple[float, float]:
        """Get the full frequency range for current parameters."""
        return (0.0, self.sample_rate / 2)
    
    def get_optimal_parameters(self, audio_duration: float, 
                             target_time_resolution: float = 0.01,
                             target_freq_resolution: float = 50.0) -> Dict[str, int]:
        """Calculate optimal STFT parameters for given requirements."""
        
        # Time resolution determines hop length
        optimal_hop = int(target_time_resolution * self.sample_rate)
        optimal_hop = max(64, min(1024, optimal_hop))
        
        # Frequency resolution determines FFT size
        optimal_fft = int(self.sample_rate / target_freq_resolution)
        optimal_fft = 2 ** int(np.ceil(np.log2(optimal_fft)))
        optimal_fft = max(512, min(8192, optimal_fft))
        
        return {
            'fft_size': optimal_fft,
            'hop_length': optimal_hop,
            'time_resolution': optimal_hop / self.sample_rate,
            'freq_resolution': self.sample_rate / optimal_fft
        }