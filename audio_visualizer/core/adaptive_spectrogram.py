"""
Adaptive Spectrogram Engine
Automatically adjusts FFT parameters based on zoom level for optimal quality.

The key insight: different zoom levels need different FFT parameters:
- Zoomed OUT (overview): larger hop_length is fine, focus on frequency resolution
- Zoomed IN (detail): smaller hop_length for better time resolution
- Narrow frequency band: larger FFT size for better frequency resolution
"""

import numpy as np
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import logging
import hashlib
from collections import OrderedDict

logger = logging.getLogger(__name__)


@dataclass
class FFTParameters:
    """FFT parameters for a specific zoom level."""
    fft_size: int
    hop_length: int
    window_type: str
    
    def __hash__(self):
        return hash((self.fft_size, self.hop_length, self.window_type))
    
    def __eq__(self, other):
        if not isinstance(other, FFTParameters):
            return False
        return (self.fft_size == other.fft_size and 
                self.hop_length == other.hop_length and 
                self.window_type == other.window_type)
    
    @property
    def time_resolution(self) -> float:
        """Time resolution in seconds (at 44100 Hz)."""
        return self.hop_length / 44100.0
    
    @property
    def freq_resolution(self) -> float:
        """Frequency resolution in Hz (at 44100 Hz)."""
        return 44100.0 / self.fft_size


@dataclass
class ViewRegion:
    """Represents a visible region in time-frequency space."""
    time_start: float
    time_end: float
    freq_start: float
    freq_end: float
    canvas_width: int
    canvas_height: int
    
    @property
    def time_span(self) -> float:
        return max(self.time_end - self.time_start, 1e-6)
    
    @property
    def freq_span(self) -> float:
        return max(self.freq_end - self.freq_start, 1e-6)
    
    @property
    def time_per_pixel(self) -> float:
        """Seconds per pixel on screen."""
        return self.time_span / max(self.canvas_width, 1)
    
    @property
    def freq_per_pixel(self) -> float:
        """Hz per pixel on screen."""
        return self.freq_span / max(self.canvas_height, 1)
    
    def cache_key(self) -> str:
        """Generate a cache key for this region."""
        return f"t{self.time_start:.3f}-{self.time_end:.3f}_f{self.freq_start:.0f}-{self.freq_end:.0f}"


# Available window functions with their properties
WINDOW_FUNCTIONS = {
    'hann': {
        'name': 'Hann',
        'description': 'Good general purpose, balanced time-frequency resolution',
        'main_lobe_width': 4,  # bins
        'side_lobe_level': -31,  # dB
        'best_for': ['general', 'music']
    },
    'hamming': {
        'name': 'Hamming', 
        'description': 'Similar to Hann, slightly better side-lobe suppression',
        'main_lobe_width': 4,
        'side_lobe_level': -43,
        'best_for': ['speech', 'voice']
    },
    'blackman': {
        'name': 'Blackman',
        'description': 'Excellent side-lobe suppression, wider main lobe',
        'main_lobe_width': 6,
        'side_lobe_level': -58,
        'best_for': ['pure_tones', 'narrow_band']
    },
    'blackmanharris': {
        'name': 'Blackman-Harris',
        'description': 'Best side-lobe suppression, widest main lobe',
        'main_lobe_width': 8,
        'side_lobe_level': -92,
        'best_for': ['pure_tones', 'precision']
    },
    'kaiser': {
        'name': 'Kaiser (β=9)',
        'description': 'Configurable trade-off, good for spectral analysis',
        'main_lobe_width': 5,
        'side_lobe_level': -70,
        'best_for': ['analysis', 'flexible']
    },
    'flattop': {
        'name': 'Flat-Top',
        'description': 'Accurate amplitude measurement, poor frequency resolution',
        'main_lobe_width': 9,
        'side_lobe_level': -93,
        'best_for': ['amplitude_accuracy', 'calibration']
    },
    'rectangular': {
        'name': 'Rectangular (None)',
        'description': 'Best frequency resolution, worst spectral leakage',
        'main_lobe_width': 2,
        'side_lobe_level': -13,
        'best_for': ['transients', 'impulses']
    }
}


class AdaptiveSpectrogramManager:
    """
    Manages adaptive spectrogram computation based on zoom level.
    
    Key features:
    - Automatically selects optimal FFT parameters for current view
    - Caches spectrograms at different parameter sets
    - Provides smooth transitions between zoom levels
    """
    
    def __init__(self, sample_rate: int = 44100, max_cache_entries: int = 10):
        self.sample_rate = sample_rate
        self.max_cache_entries = max_cache_entries
        
        # LRU cache for computed spectrograms: key -> (params, data, time_range, freq_range)
        self._cache: OrderedDict = OrderedDict()
        
        # Current FFT parameters
        self.current_params = FFTParameters(
            fft_size=2048,
            hop_length=512,
            window_type='hann'
        )
        
        # User-specified constraints
        self.min_fft_size = 256
        self.max_fft_size = 16384
        self.min_hop_ratio = 0.03125  # Allow 97% overlap (1/32)
        self.max_hop_ratio = 0.5    # hop = fft_size * ratio (maximum 50% = 50% overlap)
        
        # Quality presets
        self.quality_mode = 'balanced'  # 'fast', 'balanced', 'quality'
        
        logger.info(f"AdaptiveSpectrogramManager initialized (sample_rate={sample_rate})")
    
    def calculate_optimal_params(self, view: ViewRegion, 
                                  window_type: str = None) -> FFTParameters:
        """
        Calculate optimal FFT parameters for the given view region.
        
        The algorithm:
        1. Time resolution: We want ~2-4 pixels per spectrogram frame
        2. Frequency resolution: We want ~2-4 pixels per frequency bin
        3. Balance these with practical FFT size limits
        """
        if window_type is None:
            window_type = self.current_params.window_type
        
        # Target: 2-4 spectrogram data points per pixel for smooth appearance
        target_points_per_pixel = 2.0 if self.quality_mode == 'fast' else 4.0 if self.quality_mode == 'balanced' else 8.0
        
        # Calculate target hop length for good time resolution
        # We want: time_span / hop_length_sec = canvas_width * target_points_per_pixel
        # So: hop_length_sec = time_span / (canvas_width * target_points_per_pixel)
        target_hop_sec = view.time_span / (view.canvas_width * target_points_per_pixel)
        target_hop_samples = int(target_hop_sec * self.sample_rate)
        
        # Calculate target FFT size for good frequency resolution
        # We want: freq_span / freq_resolution = canvas_height * target_points_per_pixel
        # freq_resolution = sample_rate / fft_size
        # So: fft_size = sample_rate * canvas_height * target_points_per_pixel / freq_span
        target_freq_resolution = view.freq_span / (view.canvas_height * target_points_per_pixel)
        target_fft_size = int(self.sample_rate / target_freq_resolution)
        
        # Round FFT size to nearest power of 2 (required for efficient FFT)
        fft_size = self._nearest_power_of_2(target_fft_size)
        fft_size = max(self.min_fft_size, min(self.max_fft_size, fft_size))
        
        # Calculate hop length based on FFT size and constraints
        # ALLOW smaller hop for smoothness
        min_hop = int(fft_size * 0.03125) # Allow up to 97% overlap (1/32)
        max_hop = int(fft_size * self.max_hop_ratio)
        
        hop_length = max(min_hop, min(max_hop, target_hop_samples))
        
        # Ensure hop is at least 1
        hop_length = max(1, hop_length)
        
        # Apply quality mode adjustments
        if self.quality_mode == 'quality':
            # Higher overlap for smoother appearance
            hop_length = max(min_hop, hop_length // 2)
        elif self.quality_mode == 'fast':
            # Lower overlap for faster computation
            hop_length = min(max_hop, hop_length * 2)
        
        params = FFTParameters(
            fft_size=fft_size,
            hop_length=hop_length,
            window_type=window_type
        )
        
        logger.debug(
            f"Optimal params for view (t={view.time_span:.2f}s, f={view.freq_span:.0f}Hz, "
            f"canvas={view.canvas_width}x{view.canvas_height}): "
            f"fft={fft_size}, hop={hop_length}, window={window_type}"
        )
        
        return params
    
    def should_recompute(self, new_params: FFTParameters, 
                         tolerance: float = 0.3) -> bool:
        """
        Determine if we should recompute the spectrogram with new parameters.
        
        Returns True if parameters changed significantly.
        """
        current = self.current_params
        
        # Check FFT size change (more than 2x difference is significant)
        fft_ratio = new_params.fft_size / current.fft_size
        if fft_ratio > 2.0 or fft_ratio < 0.5:
            return True
        
        # Check hop length change
        hop_ratio = new_params.hop_length / current.hop_length
        if hop_ratio > (1 + tolerance) or hop_ratio < (1 - tolerance):
            return True
        
        # Window type change always triggers recompute
        if new_params.window_type != current.window_type:
            return True
        
        return False
    
    def get_cached_spectrogram(self, view: ViewRegion, 
                                params: FFTParameters) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get a cached spectrogram if available for the given view and parameters.
        
        Returns (magnitude_db, frequencies, times) or None if not cached.
        """
        cache_key = self._make_cache_key(view, params)
        
        if cache_key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(cache_key)
            cached = self._cache[cache_key]
            logger.debug(f"Cache hit for {cache_key}")
            return cached['magnitude_db'], cached['frequencies'], cached['times']
        
        return None
    
    def cache_spectrogram(self, view: ViewRegion, params: FFTParameters,
                          magnitude_db: np.ndarray, frequencies: np.ndarray, 
                          times: np.ndarray):
        """Cache a computed spectrogram."""
        cache_key = self._make_cache_key(view, params)
        
        # Evict oldest entries if cache is full
        while len(self._cache) >= self.max_cache_entries:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(f"Evicted cache entry: {oldest_key}")
        
        self._cache[cache_key] = {
            'magnitude_db': magnitude_db,
            'frequencies': frequencies,
            'times': times,
            'params': params,
            'view': view
        }
        
        logger.debug(f"Cached spectrogram: {cache_key} (size: {magnitude_db.shape})")
    
    def clear_cache(self):
        """Clear all cached spectrograms."""
        self._cache.clear()
        logger.info("Spectrogram cache cleared")
    
    def set_quality_mode(self, mode: str):
        """Set quality mode: 'fast', 'balanced', or 'quality'."""
        if mode in ('fast', 'balanced', 'quality'):
            self.quality_mode = mode
            logger.info(f"Quality mode set to: {mode}")
        else:
            logger.warning(f"Unknown quality mode: {mode}")
    
    def _make_cache_key(self, view: ViewRegion, params: FFTParameters) -> str:
        """Generate a unique cache key."""
        # Include relevant view and param info in key
        key_str = (
            f"{view.time_start:.3f}_{view.time_end:.3f}_"
            f"{view.freq_start:.0f}_{view.freq_end:.0f}_"
            f"{params.fft_size}_{params.hop_length}_{params.window_type}"
        )
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    @staticmethod
    def _nearest_power_of_2(x: int) -> int:
        """Round to nearest power of 2."""
        if x <= 0:
            return 256
        
        # Find powers of 2 above and below
        lower = 1 << (x - 1).bit_length() - 1
        upper = 1 << (x - 1).bit_length()
        
        # Return the closer one
        if x - lower < upper - x:
            return max(lower, 256)
        return upper
    
    def get_window_info(self, window_type: str = None) -> dict:
        """Get information about a window function."""
        if window_type is None:
            window_type = self.current_params.window_type
        return WINDOW_FUNCTIONS.get(window_type, WINDOW_FUNCTIONS['hann'])
    
    def get_available_windows(self) -> Dict[str, dict]:
        """Get all available window functions with their properties."""
        return WINDOW_FUNCTIONS.copy()


class ZoomLevelDetector:
    """
    Detects significant zoom level changes that warrant spectrogram recomputation.
    """
    
    def __init__(self, recompute_threshold: float = 2.0):
        """
        Args:
            recompute_threshold: Factor by which zoom must change to trigger recompute.
                                 2.0 means 2x zoom in or out.
        """
        self.recompute_threshold = recompute_threshold
        self.last_view: Optional[ViewRegion] = None
        self.last_recompute_zoom: float = 1.0
    
    def check_zoom_change(self, new_view: ViewRegion, 
                          reference_duration: float,
                          reference_freq_range: float) -> Tuple[bool, float]:
        """
        Check if zoom has changed enough to warrant recomputation.
        
        Args:
            new_view: New view region
            reference_duration: Total audio duration (for zoom calculation)
            reference_freq_range: Total frequency range (for zoom calculation)
            
        Returns:
            (should_recompute, current_zoom_level)
        """
        # Calculate zoom levels
        time_zoom = reference_duration / new_view.time_span
        freq_zoom = reference_freq_range / new_view.freq_span
        
        # Use geometric mean of both zoom factors
        current_zoom = np.sqrt(time_zoom * freq_zoom)
        
        if self.last_view is None:
            self.last_view = new_view
            self.last_recompute_zoom = current_zoom
            return True, current_zoom
        
        # Check if zoom changed significantly
        zoom_ratio = current_zoom / self.last_recompute_zoom
        
        if zoom_ratio >= self.recompute_threshold or zoom_ratio <= 1.0 / self.recompute_threshold:
            logger.info(f"Significant zoom change detected: {self.last_recompute_zoom:.2f}x -> {current_zoom:.2f}x")
            self.last_recompute_zoom = current_zoom
            self.last_view = new_view
            return True, current_zoom
        
        self.last_view = new_view
        return False, current_zoom
    
    def reset(self):
        """Reset zoom tracking (e.g., when loading new file)."""
        self.last_view = None
        self.last_recompute_zoom = 1.0


# Global instance
_adaptive_manager: Optional[AdaptiveSpectrogramManager] = None


def get_adaptive_spectrogram_manager(sample_rate: int = 44100) -> AdaptiveSpectrogramManager:
    """Get global adaptive spectrogram manager instance."""
    global _adaptive_manager
    if _adaptive_manager is None or _adaptive_manager.sample_rate != sample_rate:
        _adaptive_manager = AdaptiveSpectrogramManager(sample_rate)
    return _adaptive_manager

