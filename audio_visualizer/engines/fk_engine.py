import numpy as np
import scipy.fft
from typing import Tuple, Optional, Dict, Any, Callable, List
import threading

try:
    import cupy as cp
    from cupyx.scipy import fft as cp_fft
    HAS_CUPY = True
except ImportError:
    cp = None
    cp_fft = None
    HAS_CUPY = False

from ..core.cache_manager import CacheManager
from ..core.task_manager import TaskManager, TaskPriority
from .spectrogram_engine import SpectrogramEngine

class FKEngine:
    """High-performance F-K (Frequency-Wavenumber) transform engine for beamforming analysis."""
    
    def __init__(self, cache_manager: CacheManager, task_manager: TaskManager,
                 spectrogram_engine: SpectrogramEngine):
        self.cache_manager = cache_manager
        self.task_manager = task_manager
        self.spectrogram_engine = spectrogram_engine
        
        # F-K Transform parameters
        self.array_geometry = None  # Array sensor positions
        self.sound_speed = 343.0    # Speed of sound in m/s
        self.max_wavenumber = None  # Maximum wavenumber for analysis
        self.wavenumber_resolution = 0.1  # Wavenumber resolution
        
        # Analysis parameters
        self.taper_function = 'hann'  # Spatial tapering
        self.zero_padding_factor = 2  # Zero padding for interpolation
        self.roi_enabled = False      # Region of interest analysis
        self.roi_bounds = None       # ROI bounds (freq_min, freq_max, k_min, k_max)
        
        # GPU optimization
        self.use_gpu = HAS_CUPY
        self.gpu_batch_size = 4
        
        # Threading
        self._computation_lock = threading.RLock()
        self._current_tasks = {}
        
        # Beamforming patterns cache
        self._steering_vectors = {}
        
    def set_array_geometry(self, sensor_positions: np.ndarray):
        """Set array geometry for F-K analysis.
        
        Args:
            sensor_positions: Array of shape (N, 2) or (N, 3) with sensor coordinates
        """
        self.array_geometry = sensor_positions.copy()
        self.max_wavenumber = self._calculate_max_wavenumber()
        
        # Clear cache when geometry changes
        self.cache_manager.clear_view_cache('fk_transform')
        self._steering_vectors.clear()
    
    def set_parameters(self, sound_speed: float = None, wavenumber_resolution: float = None,
                      taper_function: str = None, zero_padding_factor: int = None):
        """Update F-K analysis parameters."""
        params_changed = False
        
        if sound_speed and sound_speed != self.sound_speed:
            self.sound_speed = sound_speed
            params_changed = True
            
        if wavenumber_resolution and wavenumber_resolution != self.wavenumber_resolution:
            self.wavenumber_resolution = wavenumber_resolution
            params_changed = True
            
        if taper_function and taper_function != self.taper_function:
            self.taper_function = taper_function
            params_changed = True
            
        if zero_padding_factor and zero_padding_factor != self.zero_padding_factor:
            self.zero_padding_factor = zero_padding_factor
            params_changed = True
        
        if params_changed:
            self.cache_manager.clear_view_cache('fk_transform')
            self._steering_vectors.clear()
    
    def set_roi(self, enabled: bool, freq_range: Tuple[float, float] = None,
               wavenumber_range: Tuple[float, float] = None):
        """Set region of interest for focused F-K analysis."""
        self.roi_enabled = enabled
        
        if enabled and freq_range and wavenumber_range:
            self.roi_bounds = {
                'freq_min': freq_range[0],
                'freq_max': freq_range[1],
                'k_min': wavenumber_range[0],
                'k_max': wavenumber_range[1]
            }
        else:
            self.roi_bounds = None
    
    def _calculate_max_wavenumber(self) -> float:
        """Calculate maximum resolvable wavenumber based on array geometry."""
        if self.array_geometry is None:
            return 100.0  # Default value
        
        # Calculate maximum inter-sensor distance
        max_distance = 0.0
        n_sensors = len(self.array_geometry)
        
        for i in range(n_sensors):
            for j in range(i + 1, n_sensors):
                dist = np.linalg.norm(self.array_geometry[i] - self.array_geometry[j])
                max_distance = max(max_distance, dist)
        
        # Maximum wavenumber is limited by spatial sampling
        if max_distance > 0:
            return np.pi / (max_distance / (n_sensors - 1))
        else:
            return 100.0
    
    def _create_wavenumber_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create 2D wavenumber grid for F-K analysis."""
        if self.roi_enabled and self.roi_bounds:
            k_min = self.roi_bounds['k_min']
            k_max = self.roi_bounds['k_max']
        else:
            k_max = self.max_wavenumber or 100.0
            k_min = -k_max
        
        # Create wavenumber vectors
        k_points = int((k_max - k_min) / self.wavenumber_resolution)
        kx = np.linspace(k_min, k_max, k_points)
        ky = np.linspace(k_min, k_max, k_points)
        
        kx_grid, ky_grid = np.meshgrid(kx, ky)
        
        return kx_grid, ky_grid
    
    def _apply_spatial_taper(self, data: np.ndarray) -> np.ndarray:
        """Apply spatial tapering to reduce sidelobe levels."""
        if self.array_geometry is None:
            return data
        
        # For single-channel spectrogram data: (n_frequencies, n_time_frames)
        # We simulate array processing by applying frequency-domain tapering
        n_freqs = data.shape[0]
        
        if self.taper_function == 'hann':
            taper = np.hanning(n_freqs)
        elif self.taper_function == 'hamming':
            taper = np.hamming(n_freqs)
        elif self.taper_function == 'blackman':
            taper = np.blackman(n_freqs)
        else:
            taper = np.ones(n_freqs)  # No tapering
        
        # Apply taper across frequency bins
        tapered_data = data * taper[:, np.newaxis]
        
        return tapered_data
    
    def _compute_steering_vectors(self, frequencies: np.ndarray, 
                                 kx_grid: np.ndarray, ky_grid: np.ndarray) -> np.ndarray:
        """Compute steering vectors for beamforming."""
        if self.array_geometry is None:
            raise ValueError("Array geometry not set")
        
        n_sensors = len(self.array_geometry)
        n_freqs = len(frequencies)
        n_kx, n_ky = kx_grid.shape
        
        # Initialize steering vector array
        steering_vectors = np.zeros((n_freqs, n_kx, n_ky, n_sensors), dtype=complex)
        
        for f_idx, freq in enumerate(frequencies):
            if freq <= 0:
                continue
                
            omega = 2 * np.pi * freq
            
            for i in range(n_kx):
                for j in range(n_ky):
                    kx = kx_grid[i, j]
                    ky = ky_grid[i, j]
                    
                    # Compute steering vector for this wavenumber
                    for s_idx, sensor_pos in enumerate(self.array_geometry):
                        if len(sensor_pos) >= 2:
                            x, y = sensor_pos[0], sensor_pos[1]
                            phase = -(kx * x + ky * y)
                            steering_vectors[f_idx, i, j, s_idx] = np.exp(1j * phase)
        
        return steering_vectors
    
    def compute_fk_transform_gpu(self, spectrogram_data: cp.ndarray,
                                frequencies: np.ndarray,
                                progress_callback: Optional[Callable] = None) -> cp.ndarray:
        """Compute F-K transform using GPU acceleration."""
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available for GPU acceleration")
        
        if progress_callback:
            progress_callback(0.1, "Setting up wavenumber grid")
        
        # Create wavenumber grid
        kx_grid, ky_grid = self._create_wavenumber_grid()
        
        if progress_callback:
            progress_callback(0.2, "Computing steering vectors")
        
        # Compute steering vectors on CPU then move to GPU
        steering_vectors = self._compute_steering_vectors(frequencies, kx_grid, ky_grid)
        gpu_steering_vectors = cp.asarray(steering_vectors)
        
        if progress_callback:
            progress_callback(0.4, "Applying spatial tapering")
        
        # Apply spatial tapering
        tapered_data = self._apply_spatial_taper(cp.asnumpy(spectrogram_data))
        gpu_tapered_data = cp.asarray(tapered_data)
        
        if progress_callback:
            progress_callback(0.6, "Computing F-K spectrum")
        
        # Compute F-K spectrum using beamforming
        n_freqs, n_kx, n_ky, n_sensors = gpu_steering_vectors.shape
        n_time_frames = gpu_tapered_data.shape[1]
        
        fk_spectrum = cp.zeros((n_freqs, n_kx, n_ky, n_time_frames))
        
        for f_idx in range(n_freqs):
            if progress_callback and f_idx % 10 == 0:
                progress = 0.6 + 0.3 * (f_idx / n_freqs)
                progress_callback(progress, f"Processing frequency {f_idx+1}/{n_freqs}")
            
            freq_data = gpu_tapered_data[f_idx, :]  # Shape: (n_time_frames,)
            
            for i in range(n_kx):
                for j in range(n_ky):
                    # Get steering vector for this wavenumber
                    steering_vec = gpu_steering_vectors[f_idx, i, j, :]  # Shape: (n_sensors,)
                    
                    # Conventional beamforming: w^H * x
                    # Note: This is simplified - real array data would have multiple sensors
                    beamformed_output = cp.abs(cp.sum(steering_vec[:, cp.newaxis] * 
                                                    gpu_tapered_data[f_idx:f_idx+len(steering_vec), :], 
                                                    axis=0))**2
                    
                    fk_spectrum[f_idx, i, j, :] = beamformed_output
        
        if progress_callback:
            progress_callback(0.9, "Converting to dB scale")
        
        # Convert to dB scale
        fk_spectrum_db = 10 * cp.log10(cp.maximum(fk_spectrum, 1e-10))
        
        if progress_callback:
            progress_callback(1.0, "F-K transform complete")
        
        return fk_spectrum_db
    
    def compute_fk_transform_cpu(self, spectrogram_data: np.ndarray,
                                frequencies: np.ndarray,
                                progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Compute F-K transform using CPU."""
        if progress_callback:
            progress_callback(0.1, "Setting up wavenumber grid")
        
        kx_grid, ky_grid = self._create_wavenumber_grid()
        
        if progress_callback:
            progress_callback(0.2, "Computing steering vectors")
        
        steering_vectors = self._compute_steering_vectors(frequencies, kx_grid, ky_grid)
        
        if progress_callback:
            progress_callback(0.4, "Applying spatial tapering")
        
        tapered_data = self._apply_spatial_taper(spectrogram_data)
        
        if progress_callback:
            progress_callback(0.6, "Computing F-K spectrum")
        
        # Simplified F-K computation for single-channel data
        # In practice, this would use multi-channel array data
        n_freqs, n_time_frames = tapered_data.shape
        n_kx, n_ky = kx_grid.shape
        
        # For single channel, we'll compute a spatial spectrum estimate
        fk_spectrum = np.zeros((n_freqs, n_kx, n_ky, n_time_frames))
        
        for f_idx in range(n_freqs):
            if progress_callback and f_idx % 10 == 0:
                progress = 0.6 + 0.3 * (f_idx / n_freqs)
                progress_callback(progress, f"Processing frequency {f_idx+1}/{n_freqs}")
            
            freq_data = tapered_data[f_idx, :]
            
            # Simple spatial transform (this would be more sophisticated with array data)
            for i in range(n_kx):
                for j in range(n_ky):
                    # Simulate directional response based on wavenumber
                    kx, ky = kx_grid[i, j], ky_grid[i, j]
                    k_mag = np.sqrt(kx**2 + ky**2)
                    
                    # Simple frequency-wavenumber relationship
                    if frequencies[f_idx] > 0:
                        expected_k = 2 * np.pi * frequencies[f_idx] / self.sound_speed
                        response = np.exp(-0.5 * ((k_mag - expected_k) / (expected_k * 0.1))**2)
                    else:
                        response = 0.0
                    
                    fk_spectrum[f_idx, i, j, :] = response * np.abs(freq_data)**2
        
        if progress_callback:
            progress_callback(0.9, "Converting to dB scale")
        
        # Convert to dB scale
        fk_spectrum_db = 10 * np.log10(np.maximum(fk_spectrum, 1e-10))
        
        if progress_callback:
            progress_callback(1.0, "F-K transform complete")
        
        return fk_spectrum_db
    
    def compute_fk_from_spectrogram(self, magnitude_db: np.ndarray,
                                   frequencies: np.ndarray,
                                   progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Compute F-K transform from existing spectrogram data."""
        
        # Convert dB back to linear magnitude
        magnitude_linear = 10**(magnitude_db / 20)
        
        if progress_callback:
            progress_callback(0.05, "Converting magnitude spectrum")
        
        # Compute F-K transform
        if self.use_gpu and magnitude_linear.size > 25000:
            try:
                gpu_magnitude = cp.asarray(magnitude_linear)
                fk_spectrum = self.compute_fk_transform_gpu(
                    gpu_magnitude, frequencies, progress_callback)
                
                # Move result back to CPU
                fk_spectrum = cp.asnumpy(fk_spectrum)
                
            except Exception as e:
                print(f"GPU F-K computation failed, falling back to CPU: {e}")
                fk_spectrum = self.compute_fk_transform_cpu(
                    magnitude_linear, frequencies, progress_callback)
        else:
            fk_spectrum = self.compute_fk_transform_cpu(
                magnitude_linear, frequencies, progress_callback)
        
        return fk_spectrum
    
    def request_fk_transform(self, time_range: Tuple[float, float],
                            freq_range: Tuple[float, float],
                            resolution_level: int = 0,
                            audio_data: np.ndarray = None,
                            completion_callback: Optional[Callable] = None) -> str:
        """Request F-K transform computation with lazy loading."""
        
        # Check F-K cache first
        cached_fk = self.cache_manager.get_tile(
            'fk_transform', time_range, freq_range, resolution_level)
        
        if cached_fk is not None:
            if completion_callback:
                completion_callback(cached_fk, None)
            return "cached"
        
        def compute_fk_transform(progress_callback=None):
            """Compute F-K transform using cached or computed spectrogram."""
            try:
                if progress_callback:
                    progress_callback(0.1, "Loading spectrogram data")
                
                # Try to get spectrogram from cache first
                cached_spectrogram = self.cache_manager.get_tile(
                    'spectrogram', time_range, freq_range, resolution_level)
                
                if cached_spectrogram is not None:
                    # Use cached spectrogram data
                    magnitude_db = cached_spectrogram
                    
                    # Generate frequency array
                    freq_bins = magnitude_db.shape[0]
                    frequencies = np.linspace(freq_range[0], freq_range[1], freq_bins)
                    
                    if progress_callback:
                        progress_callback(0.2, "Computing F-K transform from cached spectrogram")
                    
                else:
                    # Need to compute spectrogram first
                    if audio_data is None:
                        raise ValueError("Audio data required for spectrogram computation")
                    
                    if progress_callback:
                        progress_callback(0.1, "Computing spectrogram for F-K analysis")
                    
                    # Compute spectrogram tile
                    magnitude_db = self.spectrogram_engine.compute_tile(
                        audio_data, time_range[0], time_range[1],
                        freq_range[0], freq_range[1], resolution_level
                    )
                    
                    # Cache the spectrogram
                    self.cache_manager.store_tile(
                        'spectrogram', time_range, freq_range,
                        magnitude_db, resolution_level, use_gpu=self.use_gpu
                    )
                    
                    freq_bins = magnitude_db.shape[0]
                    frequencies = np.linspace(freq_range[0], freq_range[1], freq_bins)
                    
                    if progress_callback:
                        progress_callback(0.3, "Computing F-K transform")
                
                # Compute F-K transform
                fk_spectrum = self.compute_fk_from_spectrogram(
                    magnitude_db, frequencies, progress_callback)
                
                # Cache the result
                self.cache_manager.store_tile(
                    'fk_transform', time_range, freq_range,
                    fk_spectrum, resolution_level, use_gpu=self.use_gpu
                )
                
                if progress_callback:
                    progress_callback(1.0, "F-K transform computation complete")
                
                return fk_spectrum
                
            except Exception as e:
                if progress_callback:
                    progress_callback(0.0, f"Error: {str(e)}")
                raise e
        
        # Submit task
        task_id = self.task_manager.submit_gpu_task(
            compute_fk_transform,
            priority=TaskPriority.NORMAL,
            completion_callback=lambda tid, result, error: 
                completion_callback(result, error) if completion_callback else None,
            description=f"F-K Transform {time_range[0]:.1f}-{time_range[1]:.1f}s"
        )
        
        with self._computation_lock:
            self._current_tasks[task_id] = {
                'time_range': time_range,
                'freq_range': freq_range,
                'resolution_level': resolution_level
            }
        
        return task_id
    
    def get_wavenumber_range(self) -> Tuple[float, float]:
        """Get the wavenumber range for current parameters."""
        if self.roi_enabled and self.roi_bounds:
            return (self.roi_bounds['k_min'], self.roi_bounds['k_max'])
        
        k_max = self.max_wavenumber or 100.0
        return (-k_max, k_max)
    
    def analyze_wave_propagation(self, fk_spectrum: np.ndarray,
                                frequencies: np.ndarray,
                                wavenumbers: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """Analyze wave propagation characteristics from F-K spectrum."""
        kx_grid, ky_grid = wavenumbers
        
        # Find dominant propagation directions
        max_indices = np.unravel_index(np.argmax(fk_spectrum, axis=(1, 2)), 
                                     fk_spectrum.shape[1:3])
        
        dominant_kx = kx_grid[max_indices]
        dominant_ky = ky_grid[max_indices]
        
        # Calculate propagation angles
        propagation_angles = np.arctan2(dominant_ky, dominant_kx) * 180 / np.pi
        
        # Calculate apparent velocities
        k_magnitudes = np.sqrt(dominant_kx**2 + dominant_ky**2)
        apparent_velocities = np.zeros_like(frequencies)
        
        valid_freq_mask = frequencies > 0
        apparent_velocities[valid_freq_mask] = (2 * np.pi * frequencies[valid_freq_mask] / 
                                              (k_magnitudes[valid_freq_mask] + 1e-10))
        
        return {
            'propagation_angles': propagation_angles,
            'apparent_velocities': apparent_velocities,
            'dominant_wavenumbers': (dominant_kx, dominant_ky),
            'k_magnitudes': k_magnitudes
        }
    
    def cancel_computation(self, task_id: str) -> bool:
        """Cancel ongoing F-K computation."""
        if task_id == "cached":
            return True
        
        success = self.task_manager.cancel_task(task_id)
        
        with self._computation_lock:
            if task_id in self._current_tasks:
                del self._current_tasks[task_id]
        
        return success
    
    def get_computation_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of F-K computation."""
        if task_id == "cached":
            return {"status": "completed", "progress": 1.0}
        
        status = self.task_manager.get_task_status(task_id)
        progress = self.task_manager.get_task_progress(task_id)
        
        return {
            "status": status.value if status else "unknown",
            "progress": progress
        }