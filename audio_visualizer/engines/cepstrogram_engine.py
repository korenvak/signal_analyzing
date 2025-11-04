import numpy as np
import scipy.fft
from typing import Tuple, Optional, Dict, Any, Callable
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

class CepstrogramEngine:
    """High-performance cepstrogram computation engine with lazy loading."""
    
    def __init__(self, cache_manager: CacheManager, task_manager: TaskManager,
                 spectrogram_engine: SpectrogramEngine):
        self.cache_manager = cache_manager
        self.task_manager = task_manager
        self.spectrogram_engine = spectrogram_engine
        
        # Cepstral analysis parameters
        self.lifter_cutoff = 13  # Lifter cutoff for formant enhancement
        self.mel_filters = 40    # Number of mel-frequency filters
        self.use_mel_scale = True
        self.cepstral_mean_norm = True
        
        # GPU optimization
        self.use_gpu = HAS_CUPY
        self.gpu_batch_size = 8
        
        # Threading
        self._computation_lock = threading.RLock()
        self._current_tasks = {}
        
        # Mel filter bank cache
        self._mel_filterbank = None
        self._mel_freqs = None
        
    def set_parameters(self, lifter_cutoff: int = None, mel_filters: int = None,
                      use_mel_scale: bool = None, cepstral_mean_norm: bool = None):
        """Update cepstral analysis parameters."""
        params_changed = False
        
        if lifter_cutoff and lifter_cutoff != self.lifter_cutoff:
            self.lifter_cutoff = lifter_cutoff
            params_changed = True
            
        if mel_filters and mel_filters != self.mel_filters:
            self.mel_filters = mel_filters
            params_changed = True
            self._mel_filterbank = None  # Force regeneration
            
        if use_mel_scale is not None and use_mel_scale != self.use_mel_scale:
            self.use_mel_scale = use_mel_scale
            params_changed = True
            self._mel_filterbank = None
            
        if cepstral_mean_norm is not None and cepstral_mean_norm != self.cepstral_mean_norm:
            self.cepstral_mean_norm = cepstral_mean_norm
            params_changed = True
        
        if params_changed:
            self.cache_manager.clear_view_cache('cepstrogram')
    
    def _hz_to_mel(self, hz: np.ndarray) -> np.ndarray:
        """Convert frequency in Hz to mel scale."""
        return 2595 * np.log10(1 + hz / 700)
    
    def _mel_to_hz(self, mel: np.ndarray) -> np.ndarray:
        """Convert mel scale to frequency in Hz."""
        return 700 * (10**(mel / 2595) - 1)
    
    def _create_mel_filterbank(self, freq_bins: np.ndarray, 
                              sample_rate: int) -> np.ndarray:
        """Create mel-frequency filterbank."""
        if self._mel_filterbank is not None:
            return self._mel_filterbank
        
        # Frequency range
        min_freq = 0
        max_freq = sample_rate / 2
        
        if self.use_mel_scale:
            # Convert to mel scale
            min_mel = self._hz_to_mel(np.array([min_freq]))[0]
            max_mel = self._hz_to_mel(np.array([max_freq]))[0]
            
            # Create mel-spaced filter centers
            mel_centers = np.linspace(min_mel, max_mel, self.mel_filters + 2)
            hz_centers = self._mel_to_hz(mel_centers)
        else:
            # Linear frequency spacing
            hz_centers = np.linspace(min_freq, max_freq, self.mel_filters + 2)
        
        # Create triangular filters
        filterbank = np.zeros((self.mel_filters, len(freq_bins)))
        
        for i in range(self.mel_filters):
            left = hz_centers[i]
            center = hz_centers[i + 1]
            right = hz_centers[i + 2]
            
            # Find frequency bin indices
            left_idx = np.argmin(np.abs(freq_bins - left))
            center_idx = np.argmin(np.abs(freq_bins - center))
            right_idx = np.argmin(np.abs(freq_bins - right))
            
            # Create triangular filter
            if left_idx < center_idx:
                # Rising edge
                for j in range(left_idx, center_idx + 1):
                    if center_idx > left_idx:
                        filterbank[i, j] = (j - left_idx) / (center_idx - left_idx)
            
            if center_idx < right_idx:
                # Falling edge
                for j in range(center_idx, right_idx + 1):
                    if right_idx > center_idx:
                        filterbank[i, j] = 1.0 - (j - center_idx) / (right_idx - center_idx)
        
        self._mel_filterbank = filterbank
        self._mel_freqs = hz_centers[1:-1]  # Remove endpoints
        
        return filterbank
    
    def compute_mfcc_gpu(self, magnitude_spectrum: cp.ndarray, 
                        frequencies: np.ndarray,
                        progress_callback: Optional[Callable] = None) -> cp.ndarray:
        """Compute MFCCs using GPU acceleration."""
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available for GPU acceleration")
        
        if progress_callback:
            progress_callback(0.1, "Creating mel filterbank")
        
        # Create mel filterbank on CPU then move to GPU
        mel_filterbank = self._create_mel_filterbank(frequencies, 
                                                   self.spectrogram_engine.sample_rate)
        gpu_filterbank = cp.asarray(mel_filterbank)
        
        if progress_callback:
            progress_callback(0.3, "Applying mel filterbank")
        
        # Apply mel filterbank to magnitude spectrum
        # magnitude_spectrum shape: (freq_bins, time_frames)
        # filterbank shape: (mel_filters, freq_bins)
        mel_spectrum = cp.dot(gpu_filterbank, magnitude_spectrum)
        
        # Convert to log scale (avoiding log(0))
        log_mel_spectrum = cp.log(cp.maximum(mel_spectrum, 1e-10))
        
        if progress_callback:
            progress_callback(0.6, "Computing DCT")
        
        # Apply DCT to get cepstral coefficients
        # Using DCT Type-II (most common for MFCC)
        cepstral_coeffs = cp_fft.dct(log_mel_spectrum, type=2, axis=0, norm='ortho')
        
        if progress_callback:
            progress_callback(0.8, "Applying liftering")
        
        # Apply liftering (cepstral filtering)
        if self.lifter_cutoff > 0:
            lifter = cp.ones(cepstral_coeffs.shape[0])
            lifter[self.lifter_cutoff:] = 0
            cepstral_coeffs = cepstral_coeffs * lifter[:, cp.newaxis]
        
        if progress_callback:
            progress_callback(0.9, "Applying cepstral mean normalization")
        
        # Cepstral mean normalization
        if self.cepstral_mean_norm:
            cepstral_mean = cp.mean(cepstral_coeffs, axis=1, keepdims=True)
            cepstral_coeffs = cepstral_coeffs - cepstral_mean
        
        if progress_callback:
            progress_callback(1.0, "Cepstrogram computation complete")
        
        return cepstral_coeffs
    
    def compute_mfcc_cpu(self, magnitude_spectrum: np.ndarray,
                        frequencies: np.ndarray,
                        progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Compute MFCCs using CPU."""
        if progress_callback:
            progress_callback(0.1, "Creating mel filterbank")
        
        mel_filterbank = self._create_mel_filterbank(frequencies, 
                                                   self.spectrogram_engine.sample_rate)
        
        if progress_callback:
            progress_callback(0.3, "Applying mel filterbank")
        
        # Apply mel filterbank
        mel_spectrum = np.dot(mel_filterbank, magnitude_spectrum)
        
        # Convert to log scale
        log_mel_spectrum = np.log(np.maximum(mel_spectrum, 1e-10))
        
        if progress_callback:
            progress_callback(0.6, "Computing DCT")
        
        # Apply DCT
        cepstral_coeffs = scipy.fft.dct(log_mel_spectrum, type=2, axis=0, norm='ortho')
        
        if progress_callback:
            progress_callback(0.8, "Applying liftering")
        
        # Apply liftering
        if self.lifter_cutoff > 0:
            lifter = np.ones(cepstral_coeffs.shape[0])
            lifter[self.lifter_cutoff:] = 0
            cepstral_coeffs = cepstral_coeffs * lifter[:, np.newaxis]
        
        if progress_callback:
            progress_callback(0.9, "Applying cepstral mean normalization")
        
        # Cepstral mean normalization
        if self.cepstral_mean_norm:
            cepstral_mean = np.mean(cepstral_coeffs, axis=1, keepdims=True)
            cepstral_coeffs = cepstral_coeffs - cepstral_mean
        
        if progress_callback:
            progress_callback(1.0, "Cepstrogram computation complete")
        
        return cepstral_coeffs
    
    def compute_cepstrogram_from_spectrogram(self, magnitude_db: np.ndarray,
                                           frequencies: np.ndarray,
                                           progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Compute cepstrogram from existing spectrogram data."""
        
        # Convert dB back to linear magnitude
        magnitude_linear = 10**(magnitude_db / 20)
        
        if progress_callback:
            progress_callback(0.05, "Converting magnitude spectrum")
        
        # Compute MFCCs
        if self.use_gpu and magnitude_linear.size > 50000:
            try:
                gpu_magnitude = cp.asarray(magnitude_linear)
                cepstral_coeffs = self.compute_mfcc_gpu(
                    gpu_magnitude, frequencies, progress_callback)
                
                # Move result back to CPU
                cepstral_coeffs = cp.asnumpy(cepstral_coeffs)
                
            except Exception as e:
                print(f"GPU cepstral computation failed, falling back to CPU: {e}")
                cepstral_coeffs = self.compute_mfcc_cpu(
                    magnitude_linear, frequencies, progress_callback)
        else:
            cepstral_coeffs = self.compute_mfcc_cpu(
                magnitude_linear, frequencies, progress_callback)
        
        return cepstral_coeffs
    
    def request_cepstrogram(self, time_range: Tuple[float, float],
                           freq_range: Tuple[float, float],
                           resolution_level: int = 0,
                           audio_data: np.ndarray = None,
                           completion_callback: Optional[Callable] = None) -> str:
        """Request cepstrogram computation with lazy loading from spectrogram cache."""
        
        # Check cepstrogram cache first
        cached_cepstrogram = self.cache_manager.get_tile(
            'cepstrogram', time_range, freq_range, resolution_level)
        
        if cached_cepstrogram is not None:
            if completion_callback:
                completion_callback(cached_cepstrogram, None)
            return "cached"
        
        def compute_cepstrogram(progress_callback=None):
            """Compute cepstrogram using cached or computed spectrogram."""
            try:
                if progress_callback:
                    progress_callback(0.1, "Loading spectrogram data")
                
                # Try to get spectrogram from cache first
                cached_spectrogram = self.cache_manager.get_tile(
                    'spectrogram', time_range, freq_range, resolution_level)
                
                if cached_spectrogram is not None:
                    # Use cached spectrogram data
                    magnitude_db = cached_spectrogram
                    
                    # Generate frequency array based on spectrogram engine parameters
                    freq_bins = magnitude_db.shape[0]
                    frequencies = np.linspace(freq_range[0], freq_range[1], freq_bins)
                    
                    if progress_callback:
                        progress_callback(0.2, "Computing cepstrogram from cached spectrogram")
                    
                else:
                    # Need to compute spectrogram first
                    if audio_data is None:
                        raise ValueError("Audio data required for spectrogram computation")
                    
                    if progress_callback:
                        progress_callback(0.1, "Computing spectrogram")
                    
                    # Compute spectrogram tile
                    magnitude_db = self.spectrogram_engine.compute_tile(
                        audio_data, time_range[0], time_range[1],
                        freq_range[0], freq_range[1], resolution_level
                    )
                    
                    # Cache the spectrogram for future use
                    self.cache_manager.store_tile(
                        'spectrogram', time_range, freq_range,
                        magnitude_db, resolution_level, use_gpu=self.use_gpu
                    )
                    
                    freq_bins = magnitude_db.shape[0]
                    frequencies = np.linspace(freq_range[0], freq_range[1], freq_bins)
                    
                    if progress_callback:
                        progress_callback(0.4, "Computing cepstrogram")
                
                # Compute cepstrogram
                cepstral_coeffs = self.compute_cepstrogram_from_spectrogram(
                    magnitude_db, frequencies, progress_callback)
                
                # Cache the result
                self.cache_manager.store_tile(
                    'cepstrogram', time_range, freq_range,
                    cepstral_coeffs, resolution_level, use_gpu=self.use_gpu
                )
                
                if progress_callback:
                    progress_callback(1.0, "Cepstrogram computation complete")
                
                return cepstral_coeffs
                
            except Exception as e:
                if progress_callback:
                    progress_callback(0.0, f"Error: {str(e)}")
                raise e
        
        # Submit task
        task_id = self.task_manager.submit_gpu_task(
            compute_cepstrogram,
            priority=TaskPriority.HIGH,
            completion_callback=lambda tid, result, error: 
                completion_callback(result, error) if completion_callback else None,
            description=f"Cepstrogram {time_range[0]:.1f}-{time_range[1]:.1f}s"
        )
        
        with self._computation_lock:
            self._current_tasks[task_id] = {
                'time_range': time_range,
                'freq_range': freq_range,
                'resolution_level': resolution_level
            }
        
        return task_id
    
    def get_quefrency_range(self) -> Tuple[float, float]:
        """Get the quefrency range for current parameters."""
        # Quefrency range depends on mel filters and sample rate
        max_quefrency = self.mel_filters / self.spectrogram_engine.sample_rate
        return (0.0, max_quefrency)
    
    def cancel_computation(self, task_id: str) -> bool:
        """Cancel ongoing cepstrogram computation."""
        if task_id == "cached":
            return True
        
        success = self.task_manager.cancel_task(task_id)
        
        with self._computation_lock:
            if task_id in self._current_tasks:
                del self._current_tasks[task_id]
        
        return success
    
    def get_computation_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of cepstrogram computation."""
        if task_id == "cached":
            return {"status": "completed", "progress": 1.0}
        
        status = self.task_manager.get_task_status(task_id)
        progress = self.task_manager.get_task_progress(task_id)
        
        return {
            "status": status.value if status else "unknown",
            "progress": progress
        }
    
    def get_mel_frequencies(self) -> Optional[np.ndarray]:
        """Get mel filter center frequencies."""
        if self._mel_freqs is not None:
            return self._mel_freqs.copy()
        return None
    
    def analyze_formants(self, cepstral_coeffs: np.ndarray,
                        num_formants: int = 4) -> Dict[str, np.ndarray]:
        """Analyze formant frequencies from cepstral coefficients."""
        # This is a simplified formant analysis
        # In practice, more sophisticated methods would be used
        
        # Use the first few cepstral coefficients for formant estimation
        formant_coeffs = cepstral_coeffs[:num_formants, :]
        
        # Estimate formant frequencies (this is a simplified approach)
        formant_freqs = np.zeros((num_formants, cepstral_coeffs.shape[1]))
        
        for i in range(num_formants):
            # Simple linear mapping from cepstral coefficients to frequencies
            # This would be much more sophisticated in a real implementation
            coeff_range = np.max(formant_coeffs[i, :]) - np.min(formant_coeffs[i, :])
            if coeff_range > 0:
                normalized_coeffs = (formant_coeffs[i, :] - np.min(formant_coeffs[i, :])) / coeff_range
                formant_freqs[i, :] = 200 + normalized_coeffs * (3000 * (i + 1))
        
        return {
            'formant_frequencies': formant_freqs,
            'formant_coefficients': formant_coeffs
        }