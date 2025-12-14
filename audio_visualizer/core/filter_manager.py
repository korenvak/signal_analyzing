"""
Filter manager for applying image filters to spectrograms.
Supports GPU acceleration via CuPy.

Includes:
- Basic filters: Gaussian blur, median, contrast, threshold
- Advanced track removal: horizontal/vertical line removal, PCEN, spectral subtraction
- Morphological operations
- Adaptive noise gate
"""
import numpy as np
import logging
import time
from typing import Optional, List, Tuple, Union, Dict, Any

try:
    import cupy as cp
    from cupyx.scipy import ndimage as ndi_gpu
    HAS_GPU = True
except ImportError:
    cp = None
    ndi_gpu = None
    HAS_GPU = False

from scipy import ndimage as ndi_cpu
from skimage.filters import meijering as meijering_cpu

# Import Koren filter
try:
    from .koren_filter import KorenFilter
    KOREN_FILTER_AVAILABLE = True
except ImportError:
    KOREN_FILTER_AVAILABLE = False

logger = logging.getLogger(__name__)


def _ensure_cpu(data):
    """Ensure data is a numpy array on CPU."""
    if HAS_GPU and hasattr(data, 'get'):
        return data.get()
    return np.asarray(data)


def _ensure_gpu(data):
    """Ensure data is on GPU if available."""
    if HAS_GPU:
        if isinstance(data, cp.ndarray):
            return data
        return cp.array(data)
    return data

class FilterManager:
    """Manages application of image filters with Undo/Redo support."""
    
    def __init__(self):
        self.undo_stack: List[Union[np.ndarray, Any]] = []
        self.redo_stack: List[Union[np.ndarray, Any]] = []
        self.max_history = 10
        
    def push_state(self, data: np.ndarray):
        """Save current state to undo stack."""
        # If on GPU, move to CPU for storage to save VRAM
        if HAS_GPU and isinstance(data, cp.ndarray):
            data_cpu = data.get()
        else:
            data_cpu = data.copy()
            
        self.undo_stack.append(data_cpu)
        if len(self.undo_stack) > self.max_history:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        
    def undo(self, current_data: np.ndarray) -> Optional[np.ndarray]:
        """Restore previous state."""
        if not self.undo_stack:
            return None
            
        # Save current to redo
        if HAS_GPU and isinstance(current_data, cp.ndarray):
            self.redo_stack.append(current_data.get())
        else:
            self.redo_stack.append(current_data.copy())
            
        return self.undo_stack.pop()
        
    def redo(self, current_data: np.ndarray) -> Optional[np.ndarray]:
        """Re-apply undone state."""
        if not self.redo_stack:
            return None
            
        # Save current to undo
        if HAS_GPU and isinstance(current_data, cp.ndarray):
            self.undo_stack.append(current_data.get())
        else:
            self.undo_stack.append(current_data.copy())
            
        return self.redo_stack.pop()
        
    def can_undo(self) -> bool:
        return len(self.undo_stack) > 0
        
    def can_redo(self) -> bool:
        return len(self.redo_stack) > 0

    def apply_meijering(self, data: np.ndarray, sigmas: range = range(1, 4), 
                       black_ridges: bool = False, mode: str = 'reflect') -> Tuple[np.ndarray, str]:
        """
        Apply Meijering filter to detecting ridges.
        
        Args:
            data: Spectrogram data (2D)
            sigmas: Range of sigmas to check
            black_ridges: True if ridges are black lines on white background
            mode: Padding mode
            
        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        
        # Ensure we have valid data
        if data is None or data.size == 0:
            return data, "No data to filter"

        # Attempt GPU implementation
        if HAS_GPU:
            try:
                logger.info("Applying Meijering filter on GPU...")
                # Transfer to GPU if needed
                if not isinstance(data, cp.ndarray):
                    data_gpu = cp.array(data)
                else:
                    data_gpu = data
                
                # Run custom GPU Meijering implementation
                result_gpu = self._meijering_gpu(data_gpu, sigmas, black_ridges, mode)
                
                result = result_gpu.get() # Back to CPU
                dt = time.time() - start_time
                return result, f"Meijering Filter (GPU) applied in {dt:.3f}s"
                
            except Exception as e:
                logger.warning(f"GPU Meijering failed, falling back to CPU: {e}")
                if 'data_gpu' in locals():
                    del data_gpu
                cp.get_default_memory_pool().free_all_blocks()
        
        # CPU Fallback (scikit-image)
        logger.info("Applying Meijering filter on CPU...")
        if hasattr(data, 'get'): # If it's a cupy array
            data_cpu = data.get()
        else:
            data_cpu = data
            
        # Normalize data to 0-1 if needed, skimage expects float
        result = meijering_cpu(data_cpu, sigmas=sigmas, black_ridges=black_ridges, mode=mode)
        
        dt = time.time() - start_time
        return result, f"Meijering Filter (CPU) applied in {dt:.3f}s"

    def _meijering_gpu(self, image: Any, sigmas: range, black_ridges: bool, mode: str) -> Any:
        """
        Custom implementation of Meijering filter using CuPy/CuPyX.
        Meijering metric for 2D is typically based on eigenvalues of Hessian.
        """
        # Initialize output
        out = cp.zeros_like(image)
        
        for sigma in sigmas:
            # Compute Hessian
            # Hxx = d2/dx2, Hyy = d2/dy2, Hxy = d2/dxdy
            # Gaussian smoothing + derivative
            Hxx = ndi_gpu.gaussian_filter(image, sigma, order=(0, 2), mode=mode)
            Hxy = ndi_gpu.gaussian_filter(image, sigma, order=(1, 1), mode=mode)
            Hyy = ndi_gpu.gaussian_filter(image, sigma, order=(2, 0), mode=mode)
            
            # Compute eigenvalues of Hessian matrix [Hxx Hxy; Hxy Hyy]
            # Trace = l1 + l2 = Hxx + Hyy
            # Det = l1*l2 = Hxx*Hyy - Hxy^2
            # Discriminant D = sqrt(Trace^2 - 4*Det) = sqrt((Hxx-Hyy)^2 + 4Hxy^2)
            # l1 = (Trace + D) / 2
            # l2 = (Trace - D) / 2
            
            # We want to sort by absolute value |l1| < |l2| usually
            # But for Meijering specifically in 2D?
            # Usually it's max eigenvalue for ridges.
            
            # Let's verify standard definition. Often it's just the largest eigenvalue 
            # perpendicular to the ridge.
            
            # Calculate eigenvalues
            D = cp.sqrt((Hxx - Hyy)**2 + 4*Hxy**2)
            l1 = (Hxx + Hyy + D) / 2
            l2 = (Hxx + Hyy - D) / 2
            
            # Sort by magnitude: |l1| <= |l2|
            # Create mask where |l1| > |l2| and swap
            mask = cp.abs(l1) > cp.abs(l2)
            l1_sorted = cp.where(mask, l2, l1)
            l2_sorted = cp.where(mask, l1, l2)
            
            # Meijering neuriteness:
            # In 2D, often just looks for specific curvature. 
            # If black_ridges=False (white ridges): we want large NEGATIVE curvature (l2 << 0)
            # If black_ridges=True (black ridges): we want large POSITIVE curvature (l2 >> 0)
            
            # According to skimage documentation/source:
            # For 2D, Meijering is often just about the max eigenvalue (lambda2) if correct sign
            
            if black_ridges:
                # We want positive curvature
                # Filter out negative curvature (valleys)
                l2_sorted = cp.where(l2_sorted < 0, 0, l2_sorted)
                response = l2_sorted
            else:
                # We want negative curvature (bright ridges)
                # Filter out positive curvature (valleys)
                # Typically we take -l2, setting positive l2 to 0
                l2_sorted = cp.where(l2_sorted > 0, 0, l2_sorted)
                response = -l2_sorted
                
            # Maximize over scales
            out = cp.maximum(out, response)
            
        # Normalize to 0-1? Not strictly required but good for visualization
        # Standard skimage meijering usually returns values in range [0, 1] if input is [0, 1] 
        # but depends on sigma.
        # Let's just return the response map.
        
        # Skimage meijering implementation usually normalizes by dividing by something or
        # just returns the raw eigenvalue response. Let's inspect:
        # Skimage returns the maximum response over scales.

        return out

    # =========================================================================
    # Basic Filters
    # =========================================================================

    def apply_gaussian_blur(self, data: np.ndarray, sigma: float = 1.0) -> Tuple[np.ndarray, str]:
        """Apply Gaussian blur to spectrogram.

        Args:
            data: Input spectrogram
            sigma: Standard deviation of Gaussian kernel

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        if HAS_GPU:
            try:
                data_gpu = _ensure_gpu(data)
                result = ndi_gpu.gaussian_filter(data_gpu, sigma).get()
                dt = time.time() - start_time
                return result, f"Gaussian Blur (GPU, sigma={sigma:.1f}) applied in {dt:.3f}s"
            except Exception as e:
                logger.warning(f"GPU Gaussian failed: {e}")

        data_cpu = _ensure_cpu(data)
        result = ndi_cpu.gaussian_filter(data_cpu, sigma)
        dt = time.time() - start_time
        return result, f"Gaussian Blur (CPU, sigma={sigma:.1f}) applied in {dt:.3f}s"

    def apply_median_filter(self, data: np.ndarray, size: int = 3) -> Tuple[np.ndarray, str]:
        """Apply median filter for noise removal.

        Args:
            data: Input spectrogram
            size: Kernel size (should be odd)

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        # Ensure odd size
        size = size if size % 2 == 1 else size + 1

        if HAS_GPU:
            try:
                data_gpu = _ensure_gpu(data)
                result = ndi_gpu.median_filter(data_gpu, size=size).get()
                dt = time.time() - start_time
                return result, f"Median Filter (GPU, size={size}) applied in {dt:.3f}s"
            except Exception as e:
                logger.warning(f"GPU Median failed: {e}")

        data_cpu = _ensure_cpu(data)
        result = ndi_cpu.median_filter(data_cpu, size=size)
        dt = time.time() - start_time
        return result, f"Median Filter (CPU, size={size}) applied in {dt:.3f}s"

    def apply_contrast_enhancement(self, data: np.ndarray,
                                   percentile_low: float = 2.0,
                                   percentile_high: float = 98.0) -> Tuple[np.ndarray, str]:
        """Enhance contrast by percentile clipping.

        Args:
            data: Input spectrogram
            percentile_low: Clip values below this percentile
            percentile_high: Clip values above this percentile

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data)
        p_low = np.percentile(data_cpu, percentile_low)
        p_high = np.percentile(data_cpu, percentile_high)

        # Clip and rescale
        result = np.clip(data_cpu, p_low, p_high)
        if p_high > p_low:
            result = (result - p_low) / (p_high - p_low)

        dt = time.time() - start_time
        return result, f"Contrast Enhancement ({percentile_low:.0f}-{percentile_high:.0f}%) applied in {dt:.3f}s"

    def apply_threshold(self, data: np.ndarray, method: str = 'percentile',
                       value: float = 50.0, binary: bool = False) -> Tuple[np.ndarray, str]:
        """Apply threshold to spectrogram.

        Args:
            data: Input spectrogram
            method: 'manual', 'otsu', or 'percentile'
            value: Threshold value (0-1 for manual, percentile for percentile method)
            binary: If True, output is 0/1

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data)

        if method == 'otsu':
            from skimage.filters import threshold_otsu
            thresh = threshold_otsu(data_cpu)
        elif method == 'percentile':
            thresh = np.percentile(data_cpu, value)
        else:  # manual
            # Assume value is in 0-100 range, convert to data range
            data_min, data_max = data_cpu.min(), data_cpu.max()
            thresh = data_min + (value / 100.0) * (data_max - data_min)

        if binary:
            result = (data_cpu >= thresh).astype(np.float32)
        else:
            result = np.where(data_cpu >= thresh, data_cpu, 0)

        dt = time.time() - start_time
        return result, f"Threshold ({method}, value={value:.1f}, binary={binary}) applied in {dt:.3f}s"

    # =========================================================================
    # Advanced Track Removal Filters
    # =========================================================================

    def apply_horizontal_line_removal(self, data: np.ndarray,
                                      threshold_percentile: float = 95.0,
                                      min_width_ratio: float = 0.7,
                                      method: str = 'median') -> Tuple[np.ndarray, str]:
        """Remove horizontal lines (constant frequency interference).

        Detects rows with consistently high energy across time and removes them.

        Args:
            data: Input spectrogram (freq x time)
            threshold_percentile: Rows with energy above this percentile are candidates
            min_width_ratio: Minimum fraction of row that must be above threshold
            method: 'median', 'interpolate', or 'zero'

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data).copy()
        n_freq, n_time = data_cpu.shape

        # Calculate row-wise statistics
        row_means = np.mean(data_cpu, axis=1)
        row_threshold = np.percentile(row_means, threshold_percentile)

        # For each row, check if it's a "line"
        lines_removed = 0
        for i in range(n_freq):
            if row_means[i] > row_threshold:
                # Check width consistency
                row = data_cpu[i, :]
                above_local_thresh = row > np.percentile(row, 50)
                width_ratio = np.sum(above_local_thresh) / n_time

                if width_ratio >= min_width_ratio:
                    # This row is a horizontal line - remove it
                    if method == 'median':
                        # Replace with median of neighboring rows
                        neighbors = []
                        if i > 0:
                            neighbors.append(data_cpu[i-1, :])
                        if i < n_freq - 1:
                            neighbors.append(data_cpu[i+1, :])
                        if neighbors:
                            data_cpu[i, :] = np.median(neighbors, axis=0)
                        else:
                            data_cpu[i, :] = 0
                    elif method == 'interpolate':
                        # Linear interpolation from neighbors
                        if i > 0 and i < n_freq - 1:
                            data_cpu[i, :] = (data_cpu[i-1, :] + data_cpu[i+1, :]) / 2
                        elif i > 0:
                            data_cpu[i, :] = data_cpu[i-1, :]
                        else:
                            data_cpu[i, :] = data_cpu[i+1, :] if i+1 < n_freq else 0
                    else:  # zero
                        data_cpu[i, :] = 0
                    lines_removed += 1

        dt = time.time() - start_time
        return data_cpu, f"Horizontal Line Removal: {lines_removed} lines removed in {dt:.3f}s"

    def apply_vertical_line_removal(self, data: np.ndarray,
                                    threshold_percentile: float = 95.0,
                                    min_height_ratio: float = 0.5,
                                    method: str = 'interpolate') -> Tuple[np.ndarray, str]:
        """Remove vertical lines (clicks, impulses).

        Detects columns with consistently high energy across frequencies.

        Args:
            data: Input spectrogram (freq x time)
            threshold_percentile: Columns with energy above this percentile are candidates
            min_height_ratio: Minimum fraction of column that must be above threshold
            method: 'median', 'interpolate', or 'zero'

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data).copy()
        n_freq, n_time = data_cpu.shape

        # Calculate column-wise statistics
        col_means = np.mean(data_cpu, axis=0)
        col_threshold = np.percentile(col_means, threshold_percentile)

        lines_removed = 0
        for j in range(n_time):
            if col_means[j] > col_threshold:
                col = data_cpu[:, j]
                above_local_thresh = col > np.percentile(col, 50)
                height_ratio = np.sum(above_local_thresh) / n_freq

                if height_ratio >= min_height_ratio:
                    if method == 'median':
                        neighbors = []
                        if j > 0:
                            neighbors.append(data_cpu[:, j-1])
                        if j < n_time - 1:
                            neighbors.append(data_cpu[:, j+1])
                        if neighbors:
                            data_cpu[:, j] = np.median(neighbors, axis=0)
                        else:
                            data_cpu[:, j] = 0
                    elif method == 'interpolate':
                        if j > 0 and j < n_time - 1:
                            data_cpu[:, j] = (data_cpu[:, j-1] + data_cpu[:, j+1]) / 2
                        elif j > 0:
                            data_cpu[:, j] = data_cpu[:, j-1]
                        else:
                            data_cpu[:, j] = data_cpu[:, j+1] if j+1 < n_time else 0
                    else:
                        data_cpu[:, j] = 0
                    lines_removed += 1

        dt = time.time() - start_time
        return data_cpu, f"Vertical Line Removal: {lines_removed} lines removed in {dt:.3f}s"

    def apply_spectral_subtraction(self, data: np.ndarray,
                                   noise_percentile: float = 10.0,
                                   subtraction_factor: float = 1.0,
                                   floor: float = 0.0) -> Tuple[np.ndarray, str]:
        """Apply spectral subtraction for noise removal.

        Estimates noise floor per frequency bin and subtracts it.

        Args:
            data: Input spectrogram (freq x time)
            noise_percentile: Use this percentile of each freq bin as noise estimate
            subtraction_factor: Multiply noise estimate by this factor
            floor: Minimum value after subtraction

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data)

        # Estimate noise floor per frequency bin
        noise_estimate = np.percentile(data_cpu, noise_percentile, axis=1, keepdims=True)

        # Subtract noise
        result = data_cpu - (subtraction_factor * noise_estimate)

        # Apply floor
        result = np.maximum(result, floor)

        dt = time.time() - start_time
        return result, f"Spectral Subtraction (noise={noise_percentile:.0f}%, factor={subtraction_factor:.1f}) in {dt:.3f}s"

    def apply_pcen(self, data: np.ndarray,
                   time_constant: float = 0.06,
                   gain: float = 0.98,
                   power: float = 0.5,
                   bias: float = 2.0,
                   eps: float = 1e-6) -> Tuple[np.ndarray, str]:
        """Apply Per-Channel Energy Normalization (PCEN).

        PCEN is excellent for removing slowly varying interference and
        enhancing transient signals.

        Args:
            data: Input spectrogram (freq x time)
            time_constant: Smoothing time constant
            gain: AGC strength
            power: Compression exponent
            bias: Bias before compression
            eps: Small constant for numerical stability

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data)

        # Ensure positive values
        data_pos = np.maximum(data_cpu, eps)

        # IIR smoothing along time axis (axis=1)
        # M[n] = (1-s) * M[n-1] + s * E[n]
        # where s = time_constant
        smooth = np.zeros_like(data_pos)
        smooth[:, 0] = data_pos[:, 0]

        s = time_constant
        for t in range(1, data_pos.shape[1]):
            smooth[:, t] = (1 - s) * smooth[:, t-1] + s * data_pos[:, t]

        # PCEN formula: (E / (eps + M)^gain + bias)^power - bias^power
        # Simplified version often used:
        # output = (E / (eps + smooth)**gain) ** power
        normalized = (data_pos / (eps + smooth) ** gain + bias) ** power - bias ** power

        # Ensure non-negative
        result = np.maximum(normalized, 0)

        dt = time.time() - start_time
        return result, f"PCEN (tc={time_constant:.3f}, gain={gain:.2f}, power={power:.2f}) in {dt:.3f}s"

    def apply_morphological(self, data: np.ndarray,
                           operation: str = 'opening',
                           kernel_width: int = 3,
                           kernel_height: int = 3) -> Tuple[np.ndarray, str]:
        """Apply morphological operations.

        Args:
            data: Input spectrogram
            operation: 'erosion', 'dilation', 'opening', 'closing',
                      'gradient', 'tophat', 'blackhat'
            kernel_width: Structuring element width
            kernel_height: Structuring element height

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data)
        structure = np.ones((kernel_height, kernel_width))

        ops = {
            'erosion': ndi_cpu.grey_erosion,
            'dilation': ndi_cpu.grey_dilation,
            'opening': ndi_cpu.grey_opening,
            'closing': ndi_cpu.grey_closing,
        }

        if operation in ops:
            result = ops[operation](data_cpu, structure=structure)
        elif operation == 'gradient':
            dilated = ndi_cpu.grey_dilation(data_cpu, structure=structure)
            eroded = ndi_cpu.grey_erosion(data_cpu, structure=structure)
            result = dilated - eroded
        elif operation == 'tophat':
            opened = ndi_cpu.grey_opening(data_cpu, structure=structure)
            result = data_cpu - opened
        elif operation == 'blackhat':
            closed = ndi_cpu.grey_closing(data_cpu, structure=structure)
            result = closed - data_cpu
        else:
            return data_cpu, f"Unknown operation: {operation}"

        dt = time.time() - start_time
        return result, f"Morphological {operation} ({kernel_width}x{kernel_height}) in {dt:.3f}s"

    def apply_adaptive_noise_gate(self, data: np.ndarray,
                                  window_time: int = 50,
                                  window_freq: int = 10,
                                  threshold_db: float = 6.0,
                                  soft_knee: bool = True) -> Tuple[np.ndarray, str]:
        """Apply adaptive noise gate based on local statistics.

        Args:
            data: Input spectrogram
            window_time: Local window size (time frames)
            window_freq: Local window size (frequency bins)
            threshold_db: Signal must be this many dB above local noise
            soft_knee: Use soft transition

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data)

        # Compute local noise estimate using minimum filter (approximates noise floor)
        noise_estimate = ndi_cpu.minimum_filter(data_cpu, size=(window_freq, window_time))

        # Smooth the noise estimate
        noise_estimate = ndi_cpu.gaussian_filter(noise_estimate, sigma=(window_freq//2, window_time//2))

        # Convert threshold from dB to linear
        # If data is in dB, threshold is additive
        # If data is linear, we need to adjust
        threshold_linear = 10 ** (threshold_db / 20)

        if soft_knee:
            # Soft gate: smooth transition
            ratio = data_cpu / (noise_estimate * threshold_linear + 1e-10)
            gain = np.tanh(ratio - 1)  # Smooth transition around 1
            gain = np.clip((gain + 1) / 2, 0, 1)  # Normalize to 0-1
            result = data_cpu * gain
        else:
            # Hard gate
            mask = data_cpu > (noise_estimate * threshold_linear)
            result = data_cpu * mask

        dt = time.time() - start_time
        return result, f"Adaptive Noise Gate (thresh={threshold_db:.1f}dB, soft={soft_knee}) in {dt:.3f}s"

    def apply_track_suppression(self, data: np.ndarray,
                               method: str = 'ridge',
                               sigma: float = 2.0,
                               threshold: float = 0.1,
                               suppression_strength: float = 0.8,
                               inpaint: bool = True) -> Tuple[np.ndarray, str]:
        """Suppress track-like features (lines) in spectrogram.

        Args:
            data: Input spectrogram
            method: 'ridge', 'gradient', or 'hough'
            sigma: Scale for detection
            threshold: Detection threshold
            suppression_strength: 0-1, how much to suppress
            inpaint: Fill suppressed regions

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data)
        result = data_cpu.copy()

        # Detect tracks
        if method == 'ridge':
            # Use Meijering-like ridge detection
            from skimage.filters import meijering
            ridges = meijering(data_cpu, sigmas=[sigma], black_ridges=False)
            track_mask = ridges > threshold
        elif method == 'gradient':
            # Gradient magnitude
            gy, gx = np.gradient(data_cpu)
            grad_mag = np.sqrt(gx**2 + gy**2)
            grad_mag = ndi_cpu.gaussian_filter(grad_mag, sigma)
            track_mask = grad_mag > np.percentile(grad_mag, threshold * 100)
        else:  # hough or default
            # Simplified: use high-pass to find lines
            blurred = ndi_cpu.gaussian_filter(data_cpu, sigma * 2)
            high_pass = data_cpu - blurred
            track_mask = np.abs(high_pass) > np.percentile(np.abs(high_pass), threshold * 100)

        # Suppress tracks
        if inpaint:
            # Replace with local median
            median_bg = ndi_cpu.median_filter(data_cpu, size=int(sigma * 4) + 1)
            result = np.where(track_mask,
                            (1 - suppression_strength) * data_cpu + suppression_strength * median_bg,
                            data_cpu)
        else:
            result = np.where(track_mask,
                            data_cpu * (1 - suppression_strength),
                            data_cpu)

        dt = time.time() - start_time
        n_pixels = np.sum(track_mask)
        return result, f"Track Suppression ({method}, {n_pixels} pixels) in {dt:.3f}s"

    def apply_notch_filter(self, data: np.ndarray,
                          freq_bin: int,
                          width: int = 3,
                          method: str = 'interpolate') -> Tuple[np.ndarray, str]:
        """Apply notch filter at specific frequency bin.

        Args:
            data: Input spectrogram
            freq_bin: Center frequency bin to notch
            width: Width of notch in bins
            method: 'zero', 'interpolate', or 'median'

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data).copy()
        n_freq, n_time = data_cpu.shape

        # Calculate affected bins
        start_bin = max(0, freq_bin - width // 2)
        end_bin = min(n_freq, freq_bin + width // 2 + 1)

        if method == 'zero':
            data_cpu[start_bin:end_bin, :] = 0
        elif method == 'interpolate':
            if start_bin > 0 and end_bin < n_freq:
                for i in range(start_bin, end_bin):
                    alpha = (i - start_bin + 1) / (end_bin - start_bin + 1)
                    data_cpu[i, :] = (1 - alpha) * data_cpu[start_bin - 1, :] + alpha * data_cpu[end_bin, :]
        else:  # median
            neighbors = []
            if start_bin > 0:
                neighbors.append(data_cpu[start_bin - 1, :])
            if end_bin < n_freq:
                neighbors.append(data_cpu[end_bin, :])
            if neighbors:
                replacement = np.median(neighbors, axis=0)
                data_cpu[start_bin:end_bin, :] = replacement

        dt = time.time() - start_time
        return data_cpu, f"Notch Filter (bin={freq_bin}, width={width}) in {dt:.3f}s"

    # =========================================================================
    # Frequency Domain Filters (Low-pass, High-pass, Band-pass)
    # =========================================================================

    def apply_lowpass_filter(self, data: np.ndarray,
                            cutoff_bin: int,
                            rolloff: float = 10.0) -> Tuple[np.ndarray, str]:
        """Apply low-pass filter in frequency domain.

        Attenuates frequencies above the cutoff.

        Args:
            data: Input spectrogram (freq x time)
            cutoff_bin: Frequency bin above which to attenuate
            rolloff: Steepness of the filter rolloff (higher = steeper)

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data).copy()
        n_freq, n_time = data_cpu.shape

        # Create frequency mask with smooth rolloff
        freq_indices = np.arange(n_freq)
        # Sigmoid-like rolloff
        mask = 1.0 / (1.0 + np.exp((freq_indices - cutoff_bin) / max(rolloff, 0.1)))
        mask = mask.reshape(-1, 1)  # Column vector for broadcasting

        result = data_cpu * mask

        dt = time.time() - start_time
        return result, f"Low-pass Filter (cutoff={cutoff_bin}, rolloff={rolloff:.1f}) in {dt:.3f}s"

    def apply_highpass_filter(self, data: np.ndarray,
                             cutoff_bin: int,
                             rolloff: float = 10.0) -> Tuple[np.ndarray, str]:
        """Apply high-pass filter in frequency domain.

        Attenuates frequencies below the cutoff.

        Args:
            data: Input spectrogram (freq x time)
            cutoff_bin: Frequency bin below which to attenuate
            rolloff: Steepness of the filter rolloff

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data).copy()
        n_freq, n_time = data_cpu.shape

        freq_indices = np.arange(n_freq)
        # Inverted sigmoid for high-pass
        mask = 1.0 / (1.0 + np.exp(-(freq_indices - cutoff_bin) / max(rolloff, 0.1)))
        mask = mask.reshape(-1, 1)

        result = data_cpu * mask

        dt = time.time() - start_time
        return result, f"High-pass Filter (cutoff={cutoff_bin}, rolloff={rolloff:.1f}) in {dt:.3f}s"

    def apply_bandpass_filter(self, data: np.ndarray,
                             low_cutoff_bin: int,
                             high_cutoff_bin: int,
                             rolloff: float = 10.0) -> Tuple[np.ndarray, str]:
        """Apply band-pass filter in frequency domain.

        Passes frequencies between low and high cutoffs.

        Args:
            data: Input spectrogram (freq x time)
            low_cutoff_bin: Lower frequency bin cutoff
            high_cutoff_bin: Upper frequency bin cutoff
            rolloff: Steepness of the filter rolloff

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data).copy()
        n_freq, n_time = data_cpu.shape

        freq_indices = np.arange(n_freq)
        # Combine low-pass and high-pass
        high_pass = 1.0 / (1.0 + np.exp(-(freq_indices - low_cutoff_bin) / max(rolloff, 0.1)))
        low_pass = 1.0 / (1.0 + np.exp((freq_indices - high_cutoff_bin) / max(rolloff, 0.1)))
        mask = (high_pass * low_pass).reshape(-1, 1)

        result = data_cpu * mask

        dt = time.time() - start_time
        return result, f"Band-pass Filter ({low_cutoff_bin}-{high_cutoff_bin}, rolloff={rolloff:.1f}) in {dt:.3f}s"

    def apply_bandstop_filter(self, data: np.ndarray,
                             low_cutoff_bin: int,
                             high_cutoff_bin: int,
                             rolloff: float = 10.0) -> Tuple[np.ndarray, str]:
        """Apply band-stop (notch) filter in frequency domain.

        Attenuates frequencies between low and high cutoffs.

        Args:
            data: Input spectrogram (freq x time)
            low_cutoff_bin: Lower frequency bin cutoff
            high_cutoff_bin: Upper frequency bin cutoff
            rolloff: Steepness of the filter rolloff

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data).copy()
        n_freq, n_time = data_cpu.shape

        freq_indices = np.arange(n_freq)
        # Inverse of band-pass
        high_pass = 1.0 / (1.0 + np.exp(-(freq_indices - low_cutoff_bin) / max(rolloff, 0.1)))
        low_pass = 1.0 / (1.0 + np.exp((freq_indices - high_cutoff_bin) / max(rolloff, 0.1)))
        band_pass = high_pass * low_pass
        mask = (1.0 - band_pass).reshape(-1, 1)

        result = data_cpu * mask

        dt = time.time() - start_time
        return result, f"Band-stop Filter ({low_cutoff_bin}-{high_cutoff_bin}) in {dt:.3f}s"

    # =========================================================================
    # Advanced Filters
    # =========================================================================

    def apply_wiener_filter(self, data: np.ndarray,
                           noise_variance: Optional[float] = None,
                           window_size: int = 5) -> Tuple[np.ndarray, str]:
        """Apply Wiener filter for noise reduction.

        The Wiener filter minimizes mean square error between estimated
        and true signal.

        Args:
            data: Input spectrogram
            noise_variance: Estimated noise variance (None = auto-estimate)
            window_size: Local window size for variance estimation

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        from scipy.signal import wiener

        data_cpu = _ensure_cpu(data)

        if noise_variance is None:
            # Auto-estimate noise from lowest percentile
            result = wiener(data_cpu, mysize=window_size)
        else:
            result = wiener(data_cpu, mysize=window_size, noise=noise_variance)

        # Ensure non-negative
        result = np.maximum(result, 0)

        dt = time.time() - start_time
        return result, f"Wiener Filter (window={window_size}) in {dt:.3f}s"

    def apply_bilateral_filter(self, data: np.ndarray,
                              sigma_spatial: float = 3.0,
                              sigma_color: float = 0.1) -> Tuple[np.ndarray, str]:
        """Apply bilateral filter for edge-preserving smoothing.

        Smooths while preserving edges by considering both spatial
        distance and intensity similarity.

        Args:
            data: Input spectrogram
            sigma_spatial: Spatial smoothing strength
            sigma_color: Color/intensity similarity threshold

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data)

        # Normalize to 0-1 range for bilateral filter
        data_min, data_max = data_cpu.min(), data_cpu.max()
        if data_max > data_min:
            data_norm = (data_cpu - data_min) / (data_max - data_min)
        else:
            data_norm = data_cpu

        try:
            from skimage.restoration import denoise_bilateral
            result_norm = denoise_bilateral(
                data_norm.astype(np.float64),
                sigma_color=sigma_color,
                sigma_spatial=sigma_spatial,
                channel_axis=None
            )
            # Rescale back
            result = result_norm * (data_max - data_min) + data_min
        except ImportError:
            # Fallback to Gaussian if skimage not available
            result = ndi_cpu.gaussian_filter(data_cpu, sigma_spatial)

        dt = time.time() - start_time
        return result, f"Bilateral Filter (spatial={sigma_spatial:.1f}, color={sigma_color:.2f}) in {dt:.3f}s"

    def apply_harmonic_percussive_separation(self, data: np.ndarray,
                                             kernel_size_harmonic: int = 31,
                                             kernel_size_percussive: int = 31,
                                             output: str = 'harmonic') -> Tuple[np.ndarray, str]:
        """Separate harmonic and percussive components using median filtering.

        Based on Fitzgerald's method: median filter across time enhances
        harmonic content, median filter across frequency enhances percussive.

        Args:
            data: Input spectrogram (freq x time)
            kernel_size_harmonic: Median filter size for harmonic (time axis)
            kernel_size_percussive: Median filter size for percussive (freq axis)
            output: 'harmonic', 'percussive', or 'residual'

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data)

        # Ensure odd kernel sizes
        kh = kernel_size_harmonic if kernel_size_harmonic % 2 == 1 else kernel_size_harmonic + 1
        kp = kernel_size_percussive if kernel_size_percussive % 2 == 1 else kernel_size_percussive + 1

        # Median filter along time axis (horizontal) -> harmonic component
        harmonic = ndi_cpu.median_filter(data_cpu, size=(1, kh))

        # Median filter along frequency axis (vertical) -> percussive component
        percussive = ndi_cpu.median_filter(data_cpu, size=(kp, 1))

        # Create soft masks using Wiener-like masking
        eps = 1e-10
        harmonic_mask = (harmonic ** 2) / (harmonic ** 2 + percussive ** 2 + eps)
        percussive_mask = (percussive ** 2) / (harmonic ** 2 + percussive ** 2 + eps)

        if output == 'harmonic':
            result = data_cpu * harmonic_mask
            output_name = "Harmonic"
        elif output == 'percussive':
            result = data_cpu * percussive_mask
            output_name = "Percussive"
        else:  # residual
            result = data_cpu * (1 - harmonic_mask - percussive_mask)
            result = np.maximum(result, 0)
            output_name = "Residual"

        dt = time.time() - start_time
        return result, f"Harmonic-Percussive Separation ({output_name}) in {dt:.3f}s"

    def apply_spectral_gating(self, data: np.ndarray,
                             noise_percentile: float = 10.0,
                             threshold_db: float = -20.0,
                             smoothing_time: int = 5,
                             smoothing_freq: int = 3) -> Tuple[np.ndarray, str]:
        """Apply spectral gating for noise reduction.

        Gates (suppresses) spectral bins that fall below a frequency-varying
        threshold estimated from the noise floor.

        Args:
            data: Input spectrogram (freq x time)
            noise_percentile: Percentile for noise floor estimation
            threshold_db: Threshold in dB above noise floor
            smoothing_time: Smoothing window for mask (time)
            smoothing_freq: Smoothing window for mask (freq)

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data)

        # Estimate noise floor per frequency bin
        noise_floor = np.percentile(data_cpu, noise_percentile, axis=1, keepdims=True)

        # Convert threshold from dB to linear scale factor
        threshold_linear = 10 ** (threshold_db / 20)

        # Create gate threshold
        gate_threshold = noise_floor * threshold_linear

        # Create binary mask
        mask = (data_cpu > gate_threshold).astype(np.float32)

        # Smooth the mask to reduce musical noise
        if smoothing_time > 1 or smoothing_freq > 1:
            mask = ndi_cpu.uniform_filter(mask, size=(smoothing_freq, smoothing_time))

        # Apply soft gating
        result = data_cpu * mask

        dt = time.time() - start_time
        return result, f"Spectral Gating (thresh={threshold_db:.0f}dB) in {dt:.3f}s"

    def apply_denoise_tv(self, data: np.ndarray,
                        weight: float = 0.1) -> Tuple[np.ndarray, str]:
        """Apply Total Variation denoising.

        TV denoising preserves edges while smoothing flat regions.
        Good for removing noise while keeping sharp transitions.

        Args:
            data: Input spectrogram
            weight: Denoising strength (higher = more smoothing)

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data)

        try:
            from skimage.restoration import denoise_tv_chambolle
            # Normalize to 0-1
            data_min, data_max = data_cpu.min(), data_cpu.max()
            if data_max > data_min:
                data_norm = (data_cpu - data_min) / (data_max - data_min)
            else:
                data_norm = data_cpu

            result_norm = denoise_tv_chambolle(data_norm, weight=weight)
            result = result_norm * (data_max - data_min) + data_min
        except ImportError:
            # Fallback
            result = ndi_cpu.gaussian_filter(data_cpu, sigma=weight * 10)

        dt = time.time() - start_time
        return result, f"Total Variation Denoising (weight={weight:.2f}) in {dt:.3f}s"

    def apply_non_local_means(self, data: np.ndarray,
                             patch_size: int = 5,
                             patch_distance: int = 6,
                             h: float = 0.1) -> Tuple[np.ndarray, str]:
        """Apply Non-Local Means denoising.

        NLM averages similar patches across the image, excellent for
        preserving textures and repetitive patterns.

        Args:
            data: Input spectrogram
            patch_size: Size of patches to compare
            patch_distance: Maximum distance to search for patches
            h: Filter strength (higher = more smoothing)

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data)

        try:
            from skimage.restoration import denoise_nl_means, estimate_sigma
            # Normalize
            data_min, data_max = data_cpu.min(), data_cpu.max()
            if data_max > data_min:
                data_norm = (data_cpu - data_min) / (data_max - data_min)
            else:
                data_norm = data_cpu

            # Estimate sigma if not provided
            sigma_est = estimate_sigma(data_norm)

            result_norm = denoise_nl_means(
                data_norm,
                h=h * sigma_est,
                patch_size=patch_size,
                patch_distance=patch_distance,
                fast_mode=True
            )
            result = result_norm * (data_max - data_min) + data_min
        except ImportError:
            result = ndi_cpu.gaussian_filter(data_cpu, sigma=2)

        dt = time.time() - start_time
        return result, f"Non-Local Means (h={h:.2f}, patch={patch_size}) in {dt:.3f}s"

    def apply_local_contrast_normalization(self, data: np.ndarray,
                                           window_size: int = 51,
                                           epsilon: float = 1e-5) -> Tuple[np.ndarray, str]:
        """Apply local contrast normalization.

        Normalizes each pixel by local mean and standard deviation,
        enhancing local details while suppressing global variations.

        Args:
            data: Input spectrogram
            window_size: Size of local window
            epsilon: Small constant for numerical stability

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data)

        # Compute local mean
        local_mean = ndi_cpu.uniform_filter(data_cpu, size=window_size)

        # Compute local variance
        local_sq_mean = ndi_cpu.uniform_filter(data_cpu ** 2, size=window_size)
        local_var = local_sq_mean - local_mean ** 2
        local_std = np.sqrt(np.maximum(local_var, 0) + epsilon)

        # Normalize
        result = (data_cpu - local_mean) / local_std

        # Shift to positive range
        result = result - result.min()

        dt = time.time() - start_time
        return result, f"Local Contrast Normalization (window={window_size}) in {dt:.3f}s"

    def apply_clahe(self, data: np.ndarray,
                    clip_limit: float = 2.0,
                    tile_grid_size: int = 8) -> Tuple[np.ndarray, str]:
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        CLAHE improves local contrast while limiting noise amplification.
        It divides the image into tiles and applies histogram equalization
        to each tile, with contrast limiting to prevent noise amplification.

        Args:
            data: Input spectrogram
            clip_limit: Threshold for contrast limiting (higher = more contrast)
            tile_grid_size: Size of grid for histogram equalization

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()
        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data)

        # Normalize to 0-1 range
        data_min = data_cpu.min()
        data_max = data_cpu.max()
        if data_max - data_min > 0:
            normalized = (data_cpu - data_min) / (data_max - data_min)
        else:
            normalized = data_cpu.copy()

        # Convert to uint8 for CLAHE (OpenCV requirement)
        img_uint8 = (normalized * 255).astype(np.uint8)

        try:
            import cv2
            # Create CLAHE object
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=(tile_grid_size, tile_grid_size)
            )
            # Apply CLAHE
            result_uint8 = clahe.apply(img_uint8)
            # Convert back to float
            result = result_uint8.astype(np.float32) / 255.0
            # Scale back to original range
            result = result * (data_max - data_min) + data_min
        except ImportError:
            # Fallback: manual implementation without OpenCV
            result = self._clahe_fallback(normalized, clip_limit, tile_grid_size)
            result = result * (data_max - data_min) + data_min

        dt = time.time() - start_time
        return result, f"CLAHE (clip={clip_limit:.1f}, grid={tile_grid_size}) in {dt:.3f}s"

    def _clahe_fallback(self, data: np.ndarray, clip_limit: float,
                        tile_size: int) -> np.ndarray:
        """Fallback CLAHE implementation without OpenCV."""
        from skimage import exposure

        # Use skimage's adaptive histogram equalization
        # This is similar to CLAHE
        result = exposure.equalize_adapthist(
            data,
            kernel_size=tile_size,
            clip_limit=clip_limit / 100.0  # skimage uses 0-1 range
        )
        return result.astype(np.float32)

    # =========================================================================
    # Koren's Filter (Multi-Stage Enhancement Pipeline)
    # =========================================================================

    def apply_koren_filter(self, data: np.ndarray,
                          hpss_margin: float = 3.0,
                          pcen_gain: float = 0.8,
                          pcen_power: float = 0.5,
                          pcen_time_constant: float = 0.1,
                          pcen_bias: float = 10.0,
                          low_cut_bins: int = 5,
                          meijering_sigmas: tuple = (1, 2, 3),
                          smooth_sigma: float = 1.5,
                          sigmoid_std_factor: float = 0.5,
                          sigmoid_gain: float = 10.0,
                          tv_weight: float = 0.1,
                          contrast_power: float = 0.6,
                          sr: int = 44100,
                          hop_length: int = 512,
                          skip_hpss: bool = False,
                          skip_pcen: bool = False,
                          skip_meijering: bool = False,
                          skip_tv: bool = False) -> Tuple[np.ndarray, str]:
        """
        Apply Koren's multi-stage filter pipeline for Doppler track enhancement.
        Exactly matches V11.py implementation using librosa HPSS and PCEN.

        Pipeline stages:
        1. HPSS (librosa) - removes vertical/percussive noise
        2. PCEN (librosa) - flattens background
        3. Low frequency cut - removes DC offset / rumble
        4. Meijering ridge detection - enhances ridge-like structures
        5. Directional smoothing - smooths along time axis
        6. Sigmoid fusion - mask applied to harmonic base
        7. TV denoising - final polish
        8. Contrast boost - enhance visibility

        Args:
            data: Input spectrogram (freq x time)
            hpss_margin: HPSS margin parameter (default 3.0)
            pcen_gain: PCEN AGC gain (default 0.8)
            pcen_power: PCEN compression power (default 0.5)
            pcen_time_constant: PCEN time constant (default 0.1)
            pcen_bias: PCEN bias value (default 10.0)
            low_cut_bins: Number of low frequency bins to zero
            meijering_sigmas: Scales for Meijering filter
            smooth_sigma: Gaussian smoothing sigma
            sigmoid_std_factor: Factor for sigmoid threshold
            sigmoid_gain: Sigmoid steepness
            tv_weight: Total variation denoising weight
            contrast_power: Final contrast power boost
            sr: Sample rate for PCEN
            hop_length: Hop length for PCEN
            skip_hpss: Skip HPSS step
            skip_pcen: Skip PCEN step
            skip_meijering: Skip Meijering step
            skip_tv: Skip TV denoising step

        Returns:
            (filtered_data, log_message)
        """
        if not KOREN_FILTER_AVAILABLE:
            return data, "Koren filter not available (missing dependencies)"

        if data is None or data.size == 0:
            return data, "No data to filter"

        data_cpu = _ensure_cpu(data)

        # Create and apply Koren filter (exactly as V11.py)
        koren = KorenFilter(
            hpss_margin=hpss_margin,
            pcen_gain=pcen_gain,
            pcen_power=pcen_power,
            pcen_time_constant=pcen_time_constant,
            pcen_bias=pcen_bias,
            low_cut_bins=low_cut_bins,
            meijering_sigmas=meijering_sigmas,
            smooth_sigma=smooth_sigma,
            sigmoid_std_factor=sigmoid_std_factor,
            sigmoid_gain=sigmoid_gain,
            tv_weight=tv_weight,
            contrast_power=contrast_power,
            sr=sr,
            hop_length=hop_length
        )

        result, log_msg = koren.apply(
            data_cpu,
            skip_hpss=skip_hpss,
            skip_pcen=skip_pcen,
            skip_meijering=skip_meijering,
            skip_tv=skip_tv
        )

        return result, log_msg


