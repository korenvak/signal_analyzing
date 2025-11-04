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
from ..core.memory_pools import get_memory_optimizer
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
        
        # Memory optimizer for workspace management
        self.memory_optimizer = get_memory_optimizer()
        
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
        """Create mel-frequency filterbank (float32, cached)."""
        if self._mel_filterbank is not None:
            return self._mel_filterbank
        
        # Frequency range (float32 from start)
        min_freq = 0.0
        max_freq = float(sample_rate) / 2.0
        
        if self.use_mel_scale:
            # Convert to mel scale
            min_mel = self._hz_to_mel(np.array([min_freq], dtype=np.float32))[0]
            max_mel = self._hz_to_mel(np.array([max_freq], dtype=np.float32))[0]
            
            # Create mel-spaced filter centers (float32)
            mel_centers = np.linspace(min_mel, max_mel, self.mel_filters + 2, dtype=np.float32)
            hz_centers = self._mel_to_hz(mel_centers)
        else:
            # Linear frequency spacing (float32)
            hz_centers = np.linspace(min_freq, max_freq, self.mel_filters + 2, dtype=np.float32)
        
        # Create triangular filters using workspace pool (float32)
        filterbank = self.memory_optimizer.get_cpu_workspace(
            (self.mel_filters, len(freq_bins)), dtype=np.float32)
        filterbank.fill(0.0)
        
        for i in range(self.mel_filters):
            left = hz_centers[i]
            center = hz_centers[i + 1]
            right = hz_centers[i + 2]
            
            # Find frequency bin indices
            left_idx = np.argmin(np.abs(freq_bins - left))
            center_idx = np.argmin(np.abs(freq_bins - center))
            right_idx = np.argmin(np.abs(freq_bins - right))
            
            # Create triangular filter (vectorized for speed)
            if left_idx < center_idx:
                # Rising edge
                indices = np.arange(left_idx, center_idx + 1)
                filterbank[i, indices] = (indices - left_idx).astype(np.float32) / (center_idx - left_idx)
            
            if center_idx < right_idx:
                # Falling edge
                indices = np.arange(center_idx, right_idx + 1)
                filterbank[i, indices] = 1.0 - (indices - center_idx).astype(np.float32) / (right_idx - center_idx)
        
        # Cache the filterbank (copy from workspace)
        self._mel_filterbank = filterbank.copy()
        self._mel_freqs = hz_centers[1:-1]  # Remove endpoints
        
        return self._mel_filterbank
    
    def compute_real_cepstrum_gpu(self, log_magnitude: cp.ndarray, 
                                 progress_callback: Optional[Callable] = None) -> cp.ndarray:
        """Compute real cepstrum using GPU acceleration.
        
        Real cepstrum = IFFT(log(|FFT(signal)|))
        This reveals periodicities in the frequency domain.
        """
        if not HAS_CUPY:
            raise RuntimeError("CuPy not available for GPU acceleration")
        
        if progress_callback:
            progress_callback(0.2, "Computing IFFT of log spectrum")
        
        # Compute IFFT of log magnitude spectrum
        # log_magnitude shape: (freq_bins, time_frames)
        cepstrum = cp_fft.ifft(log_magnitude, axis=0)
        
        if progress_callback:
            progress_callback(0.6, "Taking real part and normalizing")
        
        # Take real part (cepstrum is real-valued)
        real_cepstrum = cp.real(cepstrum)
        
        # Keep only first half (symmetric property) and limit to meaningful quefrencies
        # For practical audio analysis, we keep first ~50-100 samples
        max_quefrency_samples = min(real_cepstrum.shape[0] // 2, 100)
        real_cepstrum = real_cepstrum[:max_quefrency_samples, :]
        
        if progress_callback:
            progress_callback(0.8, "Applying windowing")
        
        # Apply liftering (rectangular window to emphasize lower quefrencies)
        if self.lifter_cutoff > 0 and self.lifter_cutoff < real_cepstrum.shape[0]:
            lifter = cp.ones(real_cepstrum.shape[0])
            lifter[self.lifter_cutoff:] = 0.1  # Attenuate rather than zero
            real_cepstrum = real_cepstrum * lifter[:, cp.newaxis]
        
        if progress_callback:
            progress_callback(0.9, "Final normalization")
        
        # Normalize for better visualization
        if self.cepstral_mean_norm:
            cepstral_mean = cp.mean(real_cepstrum, axis=1, keepdims=True)
            real_cepstrum = real_cepstrum - cepstral_mean
        
        # Scale for better dynamic range in visualization
        real_cepstrum = real_cepstrum * 20.0  # Amplify for visibility
        
        return real_cepstrum
    
    def compute_real_cepstrum_cpu(self, log_magnitude: np.ndarray,
                                 progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Compute real cepstrum using CPU.
        
        Real cepstrum = IFFT(log(|FFT(signal)|))
        This reveals periodicities in the frequency domain.
        """
        if progress_callback:
            progress_callback(0.2, "Computing IFFT of log spectrum")
        
        # Compute IFFT of log magnitude spectrum
        cepstrum = scipy.fft.ifft(log_magnitude, axis=0)
        
        if progress_callback:
            progress_callback(0.6, "Taking real part and normalizing")
        
        # Take real part (cepstrum is real-valued)
        real_cepstrum = np.real(cepstrum)
        
        # Keep only first half and limit to meaningful quefrencies
        max_quefrency_samples = min(real_cepstrum.shape[0] // 2, 100)
        real_cepstrum = real_cepstrum[:max_quefrency_samples, :]
        
        if progress_callback:
            progress_callback(0.8, "Applying windowing")
        
        # Apply liftering
        if self.lifter_cutoff > 0 and self.lifter_cutoff < real_cepstrum.shape[0]:
            lifter = np.ones(real_cepstrum.shape[0])
            lifter[self.lifter_cutoff:] = 0.1  # Attenuate rather than zero
            real_cepstrum = real_cepstrum * lifter[:, np.newaxis]
        
        if progress_callback:
            progress_callback(0.9, "Final normalization")
        
        # Normalize for better visualization
        if self.cepstral_mean_norm:
            cepstral_mean = np.mean(real_cepstrum, axis=1, keepdims=True)
            real_cepstrum = real_cepstrum - cepstral_mean
        
        # Scale for better dynamic range in visualization
        real_cepstrum = real_cepstrum * 20.0  # Amplify for visibility
        
        return real_cepstrum
    
    def compute_cepstrogram_from_spectrogram(self, magnitude_db: np.ndarray,
                                           frequencies: np.ndarray,
                                           progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Compute cepstrogram from existing spectrogram data.
        
        This computes the real cepstrum (IFFT of log spectrum) which shows
        periodicity in the frequency domain. The result has:
        - X-axis: Time (same as input spectrogram)
        - Y-axis: Quefrency (τ) in seconds, representing period detection
        """
        
        if progress_callback:
            progress_callback(0.1, "Preparing log magnitude spectrum")
        
        # For true cepstrum, work with log magnitude (already in dB)
        # Convert dB to natural log scale for proper cepstral analysis
        log_magnitude = magnitude_db * np.log(10) / 20  # Convert dB to natural log
        
        if progress_callback:
            progress_callback(0.3, "Computing real cepstrum")
        
        # Compute real cepstrum using IFFT of log spectrum
        if self.use_gpu and log_magnitude.size > 50000:
            try:
                # Use pinned memory for 2-3x faster transfer
                gpu_log_mag = self.memory_optimizer.copy_to_gpu_pinned(log_magnitude)
                cepstral_coeffs = self.compute_real_cepstrum_gpu(
                    gpu_log_mag, progress_callback)
                
                # Move result back to CPU
                cepstral_coeffs = cp.asnumpy(cepstral_coeffs)
                
            except Exception as e:
                print(f"GPU cepstral computation failed, falling back to CPU: {e}")
                cepstral_coeffs = self.compute_real_cepstrum_cpu(
                    log_magnitude, progress_callback)
        else:
            cepstral_coeffs = self.compute_real_cepstrum_cpu(
                log_magnitude, progress_callback)
        
        if progress_callback:
            progress_callback(1.0, "Cepstrogram computation complete")
        
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
        """Get the quefrency range for current parameters.
        
        Quefrency is measured in seconds (time units) representing the 'time' 
        of periodicity in the frequency domain.
        """
        # For true cepstrum, quefrency range is related to fundamental period detection
        # Max quefrency should correspond to lowest detectable F0 (around 50-80 Hz)
        min_f0 = 50.0  # Hz - lowest fundamental frequency we care about
        max_quefrency = 1.0 / min_f0  # seconds - maximum quefrency
        
        # For MFCC-style analysis, we limit to first N coefficients
        # Each coefficient represents a different quefrency 'bin'
        sample_rate = self.spectrogram_engine.sample_rate
        fft_size = self.spectrogram_engine.fft_size
        
        # Quefrency resolution (time per bin)
        quefrency_resolution = 1.0 / sample_rate
        
        # Practical quefrency range for speech/audio analysis
        # From 0 to about 20ms (50 Hz fundamental period)
        max_practical_quefrency = min(max_quefrency, 0.02)  # 20ms max
        
        return (0.0, max_practical_quefrency)
    
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