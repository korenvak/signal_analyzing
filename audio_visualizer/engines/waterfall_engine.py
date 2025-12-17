"""
Waterfall Engine for DAS multi-channel visualization.

Processes DAS matrix data for waterfall display:
- Direct matrix visualization (no FFT needed)
- Normalization and scaling
- Optional filtering (time-domain)
- dB conversion (optional)
- Colormap application
"""

import numpy as np
import logging
import time
from typing import Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Try to import GPU libraries
try:
    import cupy as cp
    from cupyx.scipy import ndimage as ndi_gpu
    HAS_GPU = True
except ImportError:
    cp = None
    ndi_gpu = None
    HAS_GPU = False

from scipy import ndimage as ndi_cpu


class NormalizationMode(Enum):
    """Normalization modes for waterfall display."""
    NONE = "none"
    MINMAX = "minmax"           # Scale to [0, 1] using global min/max
    MINMAX_LOCAL = "minmax_local"  # Scale using visible region min/max
    STD = "std"                 # Scale using mean ± N*std
    PERCENTILE = "percentile"  # Scale using percentiles (robust to outliers)
    DB = "db"                   # Convert to dB scale


@dataclass
class WaterfallParams:
    """Parameters for waterfall processing."""
    normalization: NormalizationMode = NormalizationMode.STD
    std_scale: float = 2.5              # For STD mode: N standard deviations
    percentile_low: float = 2.0         # For percentile mode
    percentile_high: float = 98.0       # For percentile mode
    db_reference: float = 1.0           # Reference for dB conversion
    db_floor: float = -80.0             # Minimum dB value
    use_abs: bool = True                # Use absolute value (for phase derivatives)
    smooth_sigma: float = 0.0           # Gaussian smoothing (0 = disabled)
    decimate_time: int = 1              # Time decimation factor
    decimate_sensor: int = 1            # Sensor decimation factor


class WaterfallEngine:
    """
    Engine for processing DAS data into waterfall display format.

    Input: Matrix of shape (n_time_samples, n_sensors), float32
    Output: Normalized matrix ready for display, shape (n_time, n_sensors)

    The waterfall display shows:
    - X-axis: Sensors (columns)
    - Y-axis: Time (rows, top = earlier, bottom = later)
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize waterfall engine.

        Args:
            use_gpu: Whether to use GPU acceleration if available
        """
        self.use_gpu = use_gpu and HAS_GPU
        self._params = WaterfallParams()

        # Cache for reusing allocations
        self._cache: Dict[str, Any] = {}

        # Statistics
        self._last_process_time: float = 0.0

        logger.info(f"WaterfallEngine initialized (GPU: {self.use_gpu})")

    @property
    def params(self) -> WaterfallParams:
        return self._params

    @params.setter
    def params(self, value: WaterfallParams):
        self._params = value

    def process(
        self,
        data: np.ndarray,
        params: Optional[WaterfallParams] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Process DAS matrix data for waterfall display.

        Args:
            data: Input matrix of shape (n_time, n_sensors), float32
            params: Processing parameters (uses default if None)

        Returns:
            Tuple of:
            - Processed matrix (normalized, ready for display)
            - Stats dict with min, max, mean values
        """
        if params is None:
            params = self._params

        start_time = time.time()

        # Ensure float32
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        # Move to GPU if enabled
        if self.use_gpu:
            data = cp.asarray(data)
            xp = cp
            ndi = ndi_gpu
        else:
            xp = np
            ndi = ndi_cpu

        # Step 1: Take absolute value if requested (common for phase derivatives)
        if params.use_abs:
            data = xp.abs(data)

        # Step 2: Decimate if requested
        if params.decimate_time > 1 or params.decimate_sensor > 1:
            data = self._decimate(data, params.decimate_time, params.decimate_sensor, xp)

        # Step 3: Smooth if requested
        if params.smooth_sigma > 0:
            data = ndi.gaussian_filter(data, sigma=params.smooth_sigma)

        # Step 4: Convert to dB if requested
        if params.normalization == NormalizationMode.DB:
            data = self._to_db(data, params.db_reference, params.db_floor, xp)

        # Step 5: Normalize
        data, stats = self._normalize(data, params, xp)

        # Move back to CPU
        if self.use_gpu:
            data = cp.asnumpy(data)

        self._last_process_time = time.time() - start_time
        stats['process_time_ms'] = self._last_process_time * 1000

        return data, stats

    def _decimate(
        self,
        data: np.ndarray,
        time_factor: int,
        sensor_factor: int,
        xp
    ) -> np.ndarray:
        """Decimate data by averaging."""
        if time_factor > 1:
            # Average over time blocks
            n_time = data.shape[0]
            n_blocks = n_time // time_factor
            data = data[:n_blocks * time_factor].reshape(n_blocks, time_factor, -1)
            data = data.mean(axis=1)

        if sensor_factor > 1:
            # Average over sensor blocks
            n_sensors = data.shape[1]
            n_blocks = n_sensors // sensor_factor
            data = data[:, :n_blocks * sensor_factor].reshape(data.shape[0], n_blocks, sensor_factor)
            data = data.mean(axis=2)

        return data

    def _to_db(
        self,
        data: np.ndarray,
        reference: float,
        floor: float,
        xp
    ) -> np.ndarray:
        """Convert to dB scale."""
        # Avoid log(0)
        eps = xp.finfo(xp.float32).eps
        data = xp.maximum(data, eps)

        # Convert to dB
        data = 20.0 * xp.log10(data / reference)

        # Apply floor
        data = xp.maximum(data, floor)

        return data

    def _normalize(
        self,
        data: np.ndarray,
        params: WaterfallParams,
        xp
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Normalize data to [0, 1] range."""
        stats = {}

        # Compute statistics
        data_min = float(xp.min(data))
        data_max = float(xp.max(data))
        data_mean = float(xp.mean(data))
        data_std = float(xp.std(data))

        stats['raw_min'] = data_min
        stats['raw_max'] = data_max
        stats['raw_mean'] = data_mean
        stats['raw_std'] = data_std

        if params.normalization == NormalizationMode.NONE:
            # No normalization
            norm_min, norm_max = data_min, data_max

        elif params.normalization == NormalizationMode.MINMAX:
            # Global min-max
            norm_min, norm_max = data_min, data_max

        elif params.normalization == NormalizationMode.MINMAX_LOCAL:
            # Same as MINMAX but stats are already local to this chunk
            norm_min, norm_max = data_min, data_max

        elif params.normalization == NormalizationMode.STD:
            # Mean ± N*std
            norm_min = data_mean - params.std_scale * data_std
            norm_max = data_mean + params.std_scale * data_std

        elif params.normalization == NormalizationMode.PERCENTILE:
            # Percentile-based (robust to outliers)
            if self.use_gpu:
                flat_data = cp.asnumpy(data.ravel())
                norm_min = np.percentile(flat_data, params.percentile_low)
                norm_max = np.percentile(flat_data, params.percentile_high)
            else:
                norm_min = np.percentile(data, params.percentile_low)
                norm_max = np.percentile(data, params.percentile_high)

        elif params.normalization == NormalizationMode.DB:
            # For dB data, use floor as min
            norm_min = params.db_floor
            norm_max = data_max

        else:
            norm_min, norm_max = data_min, data_max

        # Apply normalization
        if norm_max - norm_min > 1e-10:
            data = (data - norm_min) / (norm_max - norm_min)
        else:
            data = xp.zeros_like(data)

        # Clip to [0, 1]
        data = xp.clip(data, 0.0, 1.0)

        stats['norm_min'] = norm_min
        stats['norm_max'] = norm_max

        return data, stats

    def process_for_spectrogram(
        self,
        sensor_data: np.ndarray,
        sample_rate: float,
        fft_size: int = 1024,
        hop_length: int = 256,
        window: str = 'hann'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process single-sensor data into spectrogram.

        This is used when switching to single-sensor spectrogram view.

        Args:
            sensor_data: 1D array of sensor values over time
            sample_rate: Sample rate in Hz
            fft_size: FFT window size
            hop_length: Hop length between frames
            window: Window function name

        Returns:
            Tuple of (spectrogram, frequencies, times)
        """
        from scipy import signal

        # Compute STFT
        f, t, Sxx = signal.spectrogram(
            sensor_data,
            fs=sample_rate,
            window=window,
            nperseg=fft_size,
            noverlap=fft_size - hop_length,
            scaling='spectrum'
        )

        # Convert to dB
        eps = np.finfo(np.float32).eps
        Sxx_db = 10 * np.log10(Sxx + eps)

        return Sxx_db.astype(np.float32), f.astype(np.float32), t.astype(np.float32)

    def apply_colormap(
        self,
        data: np.ndarray,
        colormap: str = 'viridis',
        vmin: float = 0.0,
        vmax: float = 1.0
    ) -> np.ndarray:
        """
        Apply colormap to normalized data.

        Args:
            data: Normalized data in [0, 1] range
            colormap: Matplotlib colormap name
            vmin: Minimum value for colormap
            vmax: Maximum value for colormap

        Returns:
            RGBA image array of shape (height, width, 4)
        """
        import matplotlib.pyplot as plt

        # Get colormap
        cmap = plt.get_cmap(colormap)

        # Normalize to colormap range
        if vmax - vmin > 1e-10:
            data_norm = (data - vmin) / (vmax - vmin)
        else:
            data_norm = np.zeros_like(data)

        data_norm = np.clip(data_norm, 0.0, 1.0)

        # Apply colormap (returns RGBA)
        colored = cmap(data_norm)

        # Convert to uint8
        return (colored * 255).astype(np.uint8)

    def clear_cache(self):
        """Clear internal caches."""
        self._cache.clear()
        if self.use_gpu:
            cp.get_default_memory_pool().free_all_blocks()


# Global engine instance
_waterfall_engine: Optional[WaterfallEngine] = None


def get_waterfall_engine() -> WaterfallEngine:
    """Get global waterfall engine."""
    global _waterfall_engine
    if _waterfall_engine is None:
        _waterfall_engine = WaterfallEngine()
    return _waterfall_engine
