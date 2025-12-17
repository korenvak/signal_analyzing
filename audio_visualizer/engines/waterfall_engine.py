"""
Waterfall Engine for DAS multi-channel visualization.

Processes DAS matrix data for waterfall display:
- Direct matrix visualization (no FFT needed)
- Normalization and scaling
- Optional filtering (time-domain)
- dB conversion (optional)
- Colormap application
- GPU memory management for large datasets
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

    # Get GPU memory info
    def get_gpu_memory_info() -> Tuple[int, int]:
        """Get GPU memory (free, total) in bytes."""
        mempool = cp.get_default_memory_pool()
        device = cp.cuda.Device()
        free, total = device.mem_info
        return free, total

except ImportError:
    cp = None
    ndi_gpu = None
    HAS_GPU = False

    def get_gpu_memory_info() -> Tuple[int, int]:
        return 0, 0

from scipy import ndimage as ndi_cpu


@dataclass
class GPUMemoryConfig:
    """GPU memory management configuration."""
    max_usage_fraction: float = 0.7   # Max fraction of GPU memory to use
    chunk_size_mb: float = 256.0      # Process in chunks of this size
    auto_fallback: bool = True        # Fallback to CPU if GPU OOM
    clear_cache_threshold: float = 0.9  # Clear cache when usage exceeds this


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

    Features:
    - Automatic GPU/CPU selection based on memory availability
    - Chunked processing for large datasets
    - Memory-efficient operations
    """

    def __init__(
        self,
        use_gpu: bool = True,
        gpu_config: Optional[GPUMemoryConfig] = None
    ):
        """
        Initialize waterfall engine.

        Args:
            use_gpu: Whether to use GPU acceleration if available
            gpu_config: GPU memory management configuration
        """
        self.use_gpu = use_gpu and HAS_GPU
        self._gpu_config = gpu_config or GPUMemoryConfig()
        self._params = WaterfallParams()

        # Cache for reusing allocations
        self._cache: Dict[str, Any] = {}

        # Statistics
        self._last_process_time: float = 0.0
        self._last_gpu_memory_mb: float = 0.0
        self._fallback_to_cpu: bool = False

        # Check GPU memory on init
        if self.use_gpu:
            free, total = get_gpu_memory_info()
            logger.info(f"WaterfallEngine initialized (GPU: {self.use_gpu}, "
                       f"VRAM: {free/1e9:.1f}/{total/1e9:.1f} GB free)")
        else:
            logger.info(f"WaterfallEngine initialized (CPU mode)")

    @property
    def params(self) -> WaterfallParams:
        return self._params

    @params.setter
    def params(self, value: WaterfallParams):
        self._params = value

    def process(
        self,
        data: np.ndarray,
        params: Optional[WaterfallParams] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Process DAS matrix data for waterfall display.

        Args:
            data: Input matrix of shape (n_time, n_sensors), float32
            params: Processing parameters (uses default if None)
            progress_callback: Optional callback for progress updates

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

        # Calculate memory requirements
        data_size_mb = data.nbytes / (1024 * 1024)
        use_gpu_for_this = self.use_gpu

        # Check if data fits in GPU memory
        if use_gpu_for_this:
            use_gpu_for_this = self._check_gpu_memory(data_size_mb * 3)  # 3x for processing overhead

        if use_gpu_for_this:
            try:
                result, stats = self._process_gpu(data, params, progress_callback)
            except (cp.cuda.memory.OutOfMemoryError, MemoryError) as e:
                logger.warning(f"GPU OOM, falling back to CPU: {e}")
                self._fallback_to_cpu = True
                result, stats = self._process_cpu(data, params, progress_callback)
        else:
            # Check if we should process in chunks for very large data
            if data_size_mb > self._gpu_config.chunk_size_mb:
                result, stats = self._process_chunked_cpu(data, params, progress_callback)
            else:
                result, stats = self._process_cpu(data, params, progress_callback)

        self._last_process_time = time.time() - start_time
        stats['process_time_ms'] = self._last_process_time * 1000
        stats['used_gpu'] = use_gpu_for_this and not self._fallback_to_cpu

        return result, stats

    def _check_gpu_memory(self, required_mb: float) -> bool:
        """Check if there's enough GPU memory for processing."""
        if not self.use_gpu:
            return False

        free, total = get_gpu_memory_info()
        free_mb = free / (1024 * 1024)

        # Check against threshold
        max_available = total * self._gpu_config.max_usage_fraction / (1024 * 1024)

        if required_mb > min(free_mb, max_available):
            logger.debug(f"Insufficient GPU memory: need {required_mb:.1f}MB, "
                        f"have {free_mb:.1f}MB free ({max_available:.1f}MB max)")
            return False

        # Clear cache if memory usage is high
        usage_fraction = 1 - (free / total)
        if usage_fraction > self._gpu_config.clear_cache_threshold:
            logger.debug("Clearing GPU cache due to high memory usage")
            self.clear_cache()

        return True

    def _process_gpu(
        self,
        data: np.ndarray,
        params: WaterfallParams,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Process data on GPU."""
        if progress_callback:
            progress_callback(0.1, "Transferring to GPU...")

        data = cp.asarray(data)
        xp = cp
        ndi = ndi_gpu

        if progress_callback:
            progress_callback(0.2, "Processing on GPU...")

        # Step 1: Take absolute value if requested
        if params.use_abs:
            data = xp.abs(data)

        # Step 2: Decimate if requested
        if params.decimate_time > 1 or params.decimate_sensor > 1:
            data = self._decimate(data, params.decimate_time, params.decimate_sensor, xp)

        if progress_callback:
            progress_callback(0.4, "Applying filters...")

        # Step 3: Smooth if requested
        if params.smooth_sigma > 0:
            data = ndi.gaussian_filter(data, sigma=params.smooth_sigma)

        # Step 4: Convert to dB if requested
        if params.normalization == NormalizationMode.DB:
            data = self._to_db(data, params.db_reference, params.db_floor, xp)

        if progress_callback:
            progress_callback(0.6, "Normalizing...")

        # Step 5: Normalize
        data, stats = self._normalize(data, params, xp)

        if progress_callback:
            progress_callback(0.8, "Transferring to CPU...")

        # Track GPU memory usage
        mempool = cp.get_default_memory_pool()
        self._last_gpu_memory_mb = mempool.used_bytes() / (1024 * 1024)
        stats['gpu_memory_mb'] = self._last_gpu_memory_mb

        # Move back to CPU
        result = cp.asnumpy(data)

        if progress_callback:
            progress_callback(1.0, "Done")

        return result, stats

    def _process_cpu(
        self,
        data: np.ndarray,
        params: WaterfallParams,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Process data on CPU."""
        if progress_callback:
            progress_callback(0.1, "Processing on CPU...")

        xp = np
        ndi = ndi_cpu

        # Make a copy to avoid modifying input
        data = data.copy()

        # Step 1: Take absolute value if requested
        if params.use_abs:
            data = xp.abs(data)

        # Step 2: Decimate if requested
        if params.decimate_time > 1 or params.decimate_sensor > 1:
            data = self._decimate(data, params.decimate_time, params.decimate_sensor, xp)

        if progress_callback:
            progress_callback(0.4, "Applying filters...")

        # Step 3: Smooth if requested
        if params.smooth_sigma > 0:
            data = ndi.gaussian_filter(data, sigma=params.smooth_sigma)

        # Step 4: Convert to dB if requested
        if params.normalization == NormalizationMode.DB:
            data = self._to_db(data, params.db_reference, params.db_floor, xp)

        if progress_callback:
            progress_callback(0.7, "Normalizing...")

        # Step 5: Normalize
        data, stats = self._normalize(data, params, xp)

        if progress_callback:
            progress_callback(1.0, "Done")

        return data, stats

    def _process_chunked_cpu(
        self,
        data: np.ndarray,
        params: WaterfallParams,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Process large data in chunks on CPU to manage memory."""
        if progress_callback:
            progress_callback(0.0, "Processing in chunks...")

        n_time, n_sensors = data.shape
        chunk_size_samples = int(self._gpu_config.chunk_size_mb * 1024 * 1024 / (n_sensors * 4))
        chunk_size_samples = max(1000, chunk_size_samples)  # At least 1000 samples

        n_chunks = (n_time + chunk_size_samples - 1) // chunk_size_samples
        logger.info(f"Processing {n_time}x{n_sensors} in {n_chunks} chunks")

        xp = np
        ndi = ndi_cpu

        # First pass: compute global statistics for normalization
        if progress_callback:
            progress_callback(0.1, "Computing statistics...")

        global_min = np.inf
        global_max = -np.inf
        global_sum = 0.0
        global_sq_sum = 0.0
        total_samples = 0

        for i in range(n_chunks):
            start = i * chunk_size_samples
            end = min(start + chunk_size_samples, n_time)
            chunk = data[start:end, :]

            if params.use_abs:
                chunk = np.abs(chunk)

            global_min = min(global_min, chunk.min())
            global_max = max(global_max, chunk.max())
            global_sum += chunk.sum()
            global_sq_sum += (chunk ** 2).sum()
            total_samples += chunk.size

        global_mean = global_sum / total_samples
        global_std = np.sqrt(global_sq_sum / total_samples - global_mean ** 2)

        # Calculate normalization bounds
        if params.normalization == NormalizationMode.STD:
            norm_min = global_mean - params.std_scale * global_std
            norm_max = global_mean + params.std_scale * global_std
        elif params.normalization == NormalizationMode.MINMAX:
            norm_min, norm_max = global_min, global_max
        else:
            norm_min, norm_max = global_min, global_max

        # Second pass: process and normalize chunks
        output = np.zeros_like(data)

        for i in range(n_chunks):
            start = i * chunk_size_samples
            end = min(start + chunk_size_samples, n_time)

            if progress_callback:
                progress_callback(0.2 + 0.7 * (i / n_chunks), f"Processing chunk {i+1}/{n_chunks}...")

            chunk = data[start:end, :].copy()

            # Apply processing
            if params.use_abs:
                chunk = np.abs(chunk)

            if params.decimate_time > 1 or params.decimate_sensor > 1:
                chunk = self._decimate(chunk, params.decimate_time, params.decimate_sensor, xp)

            if params.smooth_sigma > 0:
                chunk = ndi.gaussian_filter(chunk, sigma=params.smooth_sigma)

            if params.normalization == NormalizationMode.DB:
                chunk = self._to_db(chunk, params.db_reference, params.db_floor, xp)

            # Normalize using global bounds
            if norm_max - norm_min > 1e-10:
                chunk = (chunk - norm_min) / (norm_max - norm_min)
            else:
                chunk = np.zeros_like(chunk)

            chunk = np.clip(chunk, 0.0, 1.0)

            # Handle decimation affecting output size
            if params.decimate_time > 1:
                out_start = start // params.decimate_time
                out_end = out_start + len(chunk)
                if out_end <= output.shape[0]:
                    output[out_start:out_end, :chunk.shape[1]] = chunk
            else:
                output[start:end, :] = chunk

        if progress_callback:
            progress_callback(1.0, "Done")

        stats = {
            'raw_min': float(global_min),
            'raw_max': float(global_max),
            'raw_mean': float(global_mean),
            'raw_std': float(global_std),
            'norm_min': float(norm_min),
            'norm_max': float(norm_max),
            'processed_chunks': n_chunks
        }

        return output, stats

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
