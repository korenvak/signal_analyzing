"""
Koren's Filter - Advanced spectrogram enhancement pipeline.

A multi-stage filter designed for enhancing Doppler tracks in spectrograms.
Exactly matches V11.py implementation.

IMPORTANT: This filter expects dB-scaled input (as used in the application)
and will convert to linear magnitude internally for processing.

Pipeline:
1. Convert dB to linear magnitude
2. HPSS (Harmonic-Percussive Source Separation) - removes vertical noise
3. PCEN (Per-Channel Energy Normalization) - flattens background
4. Low frequency cut - removes DC offset / rumble
5. Meijering ridge detection - enhances ridge-like structures
6. Directional smoothing - smooths along time axis
7. Sigmoid fusion - smart thresholding (mask applied to harmonic base)
8. TV denoising - final polish
9. Contrast boost - enhance visibility

Based on V11.py by Koren.
"""
import numpy as np
import logging
import time
from typing import Tuple

logger = logging.getLogger(__name__)

# Check for dependencies
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available - HPSS and PCEN will use fallback implementations")

try:
    from skimage.filters import meijering
    from skimage.restoration import denoise_tv_chambolle
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("skimage not available - using fallback implementations")

try:
    from scipy.ndimage import gaussian_filter1d, median_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class KorenFilter:
    """
    Koren's multi-stage spectrogram enhancement filter.
    Exactly matches V11.py implementation.

    NOTE: Input is expected to be in dB scale (as displayed in the application).
    The filter converts to linear magnitude internally for processing.
    """

    # Default parameters matching V11.py exactly
    DEFAULT_HPSS_MARGIN = 3.0
    DEFAULT_PCEN_GAIN = 0.8
    DEFAULT_PCEN_POWER = 0.5
    DEFAULT_PCEN_TIME_CONSTANT = 0.1
    DEFAULT_PCEN_BIAS = 10.0
    DEFAULT_LOW_CUT_BINS = 5
    DEFAULT_MEIJERING_SIGMAS = (1, 2, 3)
    DEFAULT_SMOOTH_SIGMA = 1.5
    DEFAULT_SIGMOID_STD_FACTOR = 0.5
    DEFAULT_SIGMOID_GAIN = 10.0
    DEFAULT_TV_WEIGHT = 0.1
    DEFAULT_CONTRAST_POWER = 0.6

    def __init__(self,
                 hpss_margin: float = DEFAULT_HPSS_MARGIN,
                 pcen_gain: float = DEFAULT_PCEN_GAIN,
                 pcen_power: float = DEFAULT_PCEN_POWER,
                 pcen_time_constant: float = DEFAULT_PCEN_TIME_CONSTANT,
                 pcen_bias: float = DEFAULT_PCEN_BIAS,
                 low_cut_bins: int = DEFAULT_LOW_CUT_BINS,
                 meijering_sigmas: tuple = DEFAULT_MEIJERING_SIGMAS,
                 smooth_sigma: float = DEFAULT_SMOOTH_SIGMA,
                 sigmoid_std_factor: float = DEFAULT_SIGMOID_STD_FACTOR,
                 sigmoid_gain: float = DEFAULT_SIGMOID_GAIN,
                 tv_weight: float = DEFAULT_TV_WEIGHT,
                 contrast_power: float = DEFAULT_CONTRAST_POWER,
                 sr: int = 44100,
                 hop_length: int = 512):
        """Initialize filter with V11.py default parameters."""
        self.hpss_margin = hpss_margin
        self.pcen_gain = pcen_gain
        self.pcen_power = pcen_power
        self.pcen_time_constant = pcen_time_constant
        self.pcen_bias = pcen_bias
        self.low_cut_bins = low_cut_bins
        self.meijering_sigmas = meijering_sigmas
        self.smooth_sigma = smooth_sigma
        self.sigmoid_std_factor = sigmoid_std_factor
        self.sigmoid_gain = sigmoid_gain
        self.tv_weight = tv_weight
        self.contrast_power = contrast_power
        self.sr = sr
        self.hop_length = hop_length

    def apply(self, data: np.ndarray,
              skip_hpss: bool = False,
              skip_pcen: bool = False,
              skip_meijering: bool = False,
              skip_tv: bool = False) -> Tuple[np.ndarray, str]:
        """
        Apply the full Koren filter pipeline (exactly as V11.py).

        Args:
            data: Input spectrogram (freq x time) in dB scale
            skip_hpss: Skip HPSS step
            skip_pcen: Skip PCEN step
            skip_meijering: Skip Meijering step
            skip_tv: Skip TV denoising step

        Returns:
            (filtered_data, log_message) - output is normalized [0, 1] range
        """
        start_time = time.time()

        if data is None or data.size == 0:
            return data, "No data to filter"

        steps = []
        logger.info(f"Koren Filter: Input shape {data.shape}, range [{data.min():.1f}, {data.max():.1f}]")

        # --- Convert dB to linear magnitude ---
        # Application stores data as: 20 * log10(magnitude)
        # So: magnitude = 10^(dB/20)
        t0 = time.time()
        S = np.power(10.0, data.astype(np.float64) / 20.0)
        steps.append(f"dB->Lin({time.time()-t0:.2f}s)")
        logger.info(f"After dB->linear: range [{S.min():.6f}, {S.max():.6f}]")

        # --- Stage 1: HPSS (Separation) ---
        t0 = time.time()
        if not skip_hpss and LIBROSA_AVAILABLE:
            try:
                S_harmonic, _ = librosa.decompose.hpss(S, margin=self.hpss_margin)
                steps.append(f"HPSS({time.time()-t0:.2f}s)")
            except Exception as e:
                logger.warning(f"HPSS failed: {e}")
                S_harmonic = S.copy()
        else:
            S_harmonic = S.copy()

        # --- Stage 2: PCEN (Normalization) ---
        t0 = time.time()
        if not skip_pcen and LIBROSA_AVAILABLE:
            try:
                # Scale input by 2^31 as in V11.py
                S_scaled = S_harmonic * (2 ** 31)
                S_harmonic = librosa.pcen(
                    S_scaled,
                    sr=self.sr,
                    hop_length=self.hop_length,
                    gain=self.pcen_gain,
                    power=self.pcen_power,
                    time_constant=self.pcen_time_constant,
                    bias=self.pcen_bias
                )
                steps.append(f"PCEN({time.time()-t0:.2f}s)")
            except Exception as e:
                logger.warning(f"PCEN failed: {e}")

        # --- Stage 3: Low frequency cut ---
        if self.low_cut_bins > 0:
            S_harmonic[:self.low_cut_bins, :] = 0.0
            steps.append(f"LowCut({self.low_cut_bins})")

        # Normalize to [0, 1]
        S_harmonic = self._normalize_01(S_harmonic)

        # --- Stage 4: Meijering (Ridge Detection) ---
        t0 = time.time()
        if not skip_meijering and SKIMAGE_AVAILABLE:
            try:
                ridge = meijering(S_harmonic, sigmas=self.meijering_sigmas,
                                black_ridges=False, mode='reflect')
                ridge = self._normalize_01(ridge)
                steps.append(f"Meijering({time.time()-t0:.2f}s)")
            except Exception as e:
                logger.warning(f"Meijering failed: {e}")
                ridge = S_harmonic.copy()
        else:
            ridge = S_harmonic.copy()

        # --- Stage 5: Smart Smoothing ---
        t0 = time.time()
        if self.smooth_sigma > 0 and SCIPY_AVAILABLE:
            ridge_smooth = gaussian_filter1d(ridge, sigma=self.smooth_sigma, axis=1)
            ridge_smooth = self._normalize_01(ridge_smooth)
            steps.append(f"Smooth({time.time()-t0:.2f}s)")
        else:
            ridge_smooth = ridge

        # --- Stage 6: Soft Sigmoid Fusion ---
        t0 = time.time()
        mean_ridge = np.mean(ridge_smooth)
        std_ridge = np.std(ridge_smooth)
        sigmoid_shift = mean_ridge + self.sigmoid_std_factor * std_ridge
        mask = 1.0 / (1.0 + np.exp(-self.sigmoid_gain * (ridge_smooth - sigmoid_shift)))

        # Fusion: Apply mask to HARMONIC BASE (not ridge!) - exactly as V11.py
        S_fused = S_harmonic * mask
        steps.append(f"Fusion({time.time()-t0:.2f}s)")

        # --- Stage 7: Total Variation Denoising ---
        t0 = time.time()
        if not skip_tv and SKIMAGE_AVAILABLE:
            try:
                S_final = denoise_tv_chambolle(S_fused, weight=self.tv_weight)
                steps.append(f"TV({time.time()-t0:.2f}s)")
            except Exception as e:
                logger.warning(f"TV denoise failed: {e}")
                S_final = S_fused
        else:
            S_final = S_fused

        # --- Stage 8: Contrast boost ---
        if self.contrast_power != 1.0:
            S_final = np.power(np.maximum(S_final, 0), self.contrast_power)
            steps.append(f"Contrast({self.contrast_power})")

        # Final normalization to [0, 1]
        S_final = self._normalize_01(S_final)

        dt = time.time() - start_time
        log_msg = f"Koren Filter: {' -> '.join(steps)} | Total: {dt:.2f}s"
        logger.info(log_msg)
        logger.info(f"Output range: [{S_final.min():.3f}, {S_final.max():.3f}]")

        return S_final.astype(np.float32), log_msg

    @staticmethod
    def _normalize_01(x: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range - exactly as V11.py."""
        x = x.astype(np.float64)
        xmin = np.min(x)
        xmax = np.max(x)
        if xmax <= xmin:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin + 1e-12)


def apply_koren_filter(data: np.ndarray, **kwargs) -> Tuple[np.ndarray, str]:
    """Convenience function to apply Koren filter with default V11.py parameters."""
    filter_obj = KorenFilter(**kwargs)
    return filter_obj.apply(data)
