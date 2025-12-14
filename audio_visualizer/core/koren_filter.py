"""
Koren's Filter - Advanced spectrogram enhancement pipeline.

A multi-stage filter designed for enhancing Doppler tracks in spectrograms.
Exactly matches V11.py implementation.

Pipeline:
1. HPSS (Harmonic-Percussive Source Separation) using librosa - removes vertical noise
2. PCEN (Per-Channel Energy Normalization) using librosa - flattens background
3. Low frequency cut - removes DC offset / rumble
4. Meijering ridge detection - enhances ridge-like structures
5. Directional smoothing - smooths along time axis
6. Sigmoid fusion - smart thresholding (mask applied to harmonic base)
7. TV denoising - final polish
8. Contrast boost - enhance visibility

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

try:
    from scipy.ndimage import gaussian_filter1d, median_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class KorenFilter:
    """
    Koren's multi-stage spectrogram enhancement filter.

    Exactly matches V11.py implementation.
    """

    def __init__(self,
                 # HPSS margin parameter (librosa)
                 hpss_margin: float = 3.0,
                 # PCEN parameters (librosa)
                 pcen_gain: float = 0.8,
                 pcen_power: float = 0.5,
                 pcen_time_constant: float = 0.1,
                 pcen_bias: float = 10.0,
                 # High-pass (low frequency cut)
                 low_cut_bins: int = 5,
                 # Meijering parameters
                 meijering_sigmas: tuple = (1, 2, 3),
                 # Smoothing (axis=1, horizontal/time direction)
                 smooth_sigma: float = 1.5,
                 # Sigmoid fusion
                 sigmoid_std_factor: float = 0.5,
                 sigmoid_gain: float = 10.0,
                 # TV denoising
                 tv_weight: float = 0.1,
                 # Final contrast boost
                 contrast_power: float = 0.6,
                 # Sample rate and hop length for PCEN
                 sr: int = 44100,
                 hop_length: int = 512):
        """
        Initialize filter parameters.

        Args:
            hpss_margin: Margin for HPSS separation (default 3.0 as in V11.py)
            pcen_gain: PCEN AGC gain (default 0.8)
            pcen_power: PCEN compression power (default 0.5)
            pcen_time_constant: PCEN time constant (default 0.1)
            pcen_bias: PCEN bias value (default 10.0)
            low_cut_bins: Number of low frequency bins to zero (default 5)
            meijering_sigmas: Scales for Meijering filter (default 1,2,3)
            smooth_sigma: Gaussian smoothing sigma along time axis (default 1.5)
            sigmoid_std_factor: Factor for sigmoid threshold (default 0.5)
            sigmoid_gain: Sigmoid steepness (default 10.0)
            tv_weight: Total variation denoising weight (default 0.1)
            contrast_power: Final contrast power boost (default 0.6)
            sr: Sample rate for PCEN
            hop_length: Hop length for PCEN
        """
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
            data: Input spectrogram (freq x time), should be linear magnitude (like np.abs(stft))
            skip_hpss: Skip HPSS step
            skip_pcen: Skip PCEN step
            skip_meijering: Skip Meijering step
            skip_tv: Skip TV denoising step

        Returns:
            (filtered_data, log_message)
        """
        start_time = time.time()

        if data is None or data.size == 0:
            return data, "No data to filter"

        # Ensure float64 for processing
        S = data.astype(np.float64)
        steps = []

        # --- Stage 1: HPSS (Separation) ---
        # "Applying HPSS (Removing Vertical Noise)..."
        if not skip_hpss:
            S_harmonic = self._apply_hpss(S)
            steps.append("HPSS")
        else:
            S_harmonic = S.copy()

        # --- Stage 2: PCEN (Normalization) ---
        # "Applying PCEN (Flattening Background)..."
        if not skip_pcen:
            S_harmonic = self._apply_pcen(S_harmonic)
            steps.append("PCEN")

        # --- Stage 3: Low frequency cut ---
        # Delete low frequencies (DC Offset / Rumble)
        if self.low_cut_bins > 0:
            S_harmonic[:self.low_cut_bins, :] = 0.0
            steps.append(f"LowCut({self.low_cut_bins})")

        # Normalize to [0, 1]
        S_harmonic = self._normalize_01(S_harmonic)

        # --- Stage 4: Meijering (Ridge Detection) ---
        # "Computing Meijering Ridge Response..."
        if not skip_meijering and SKIMAGE_AVAILABLE:
            ridge = meijering(S_harmonic, sigmas=self.meijering_sigmas,
                            black_ridges=False, mode='reflect')
            ridge = self._normalize_01(ridge)
            steps.append("Meijering")
        else:
            ridge = S_harmonic.copy()

        # --- Stage 5: Smart Smoothing ---
        # "Applying Directional Smoothing..."
        # Horizontal smoothing (less aggressive to not destroy steep lines)
        if SCIPY_AVAILABLE and self.smooth_sigma > 0:
            ridge_smooth = gaussian_filter1d(ridge, sigma=self.smooth_sigma, axis=1)
            ridge_smooth = self._normalize_01(ridge_smooth)
            steps.append(f"Smooth({self.smooth_sigma})")
        else:
            ridge_smooth = ridge

        # --- Stage 6: Soft Sigmoid Fusion ---
        # "Fusing Signals..."
        mean_ridge = np.mean(ridge_smooth)
        std_ridge = np.std(ridge_smooth)

        # Smart threshold
        sigmoid_shift = mean_ridge + self.sigmoid_std_factor * std_ridge

        # Sigmoid mask
        mask = 1.0 / (1.0 + np.exp(-self.sigmoid_gain * (ridge_smooth - sigmoid_shift)))

        # Fusion: Apply mask to HARMONIC BASE (not ridge!)
        S_fused = S_harmonic * mask
        steps.append("Fusion")

        # --- Stage 7: Total Variation Denoising (The Final Polish) ---
        # "Applying TV Denoising (Cleaning speckles while keeping edges)..."
        if not skip_tv and SKIMAGE_AVAILABLE:
            S_final = denoise_tv_chambolle(S_fused, weight=self.tv_weight)
            steps.append(f"TV({self.tv_weight})")
        else:
            S_final = S_fused

        # --- Stage 8: Contrast boost ---
        # Final contrast boost
        if self.contrast_power != 1.0:
            S_final = S_final ** self.contrast_power
            steps.append(f"Contrast({self.contrast_power})")

        # Final normalization
        S_final = self._normalize_01(S_final)

        dt = time.time() - start_time
        log_msg = f"Koren Filter ({' -> '.join(steps)}) applied in {dt:.3f}s"
        logger.info(log_msg)

        return S_final.astype(np.float32), log_msg

    def _apply_hpss(self, S: np.ndarray) -> np.ndarray:
        """
        Apply Harmonic-Percussive Source Separation using librosa.

        Exactly as V11.py: S_harmonic, S_percussive = librosa.decompose.hpss(S, margin=3.0)
        """
        if LIBROSA_AVAILABLE:
            try:
                S_harmonic, S_percussive = librosa.decompose.hpss(S, margin=self.hpss_margin)
                return S_harmonic
            except Exception as e:
                logger.warning(f"librosa HPSS failed: {e}, using fallback")

        # Fallback: median filter based HPSS
        return self._hpss_fallback(S)

    def _hpss_fallback(self, S: np.ndarray) -> np.ndarray:
        """Fallback HPSS using median filtering when librosa is not available."""
        if not SCIPY_AVAILABLE:
            return S

        # Median filter along time axis -> harmonic component
        harmonic = median_filter(S, size=(1, 31))

        # Median filter along frequency axis -> percussive component
        percussive = median_filter(S, size=(31, 1))

        # Soft mask for harmonic
        eps = 1e-10
        harmonic_mask = (harmonic ** 2) / (harmonic ** 2 + percussive ** 2 + eps)

        return S * harmonic_mask

    def _apply_pcen(self, S: np.ndarray) -> np.ndarray:
        """
        Apply Per-Channel Energy Normalization using librosa.

        Exactly as V11.py:
        librosa.pcen(S_harmonic * (2 ** 31), sr=self.sr, hop_length=self.hop_length,
                     bias=self.pcen_bias, gain=0.8, power=0.5, time_constant=0.1)
        """
        if LIBROSA_AVAILABLE:
            try:
                # Scale input by 2^31 as in V11.py
                S_scaled = S * (2 ** 31)

                result = librosa.pcen(
                    S_scaled,
                    sr=self.sr,
                    hop_length=self.hop_length,
                    gain=self.pcen_gain,
                    power=self.pcen_power,
                    time_constant=self.pcen_time_constant,
                    bias=self.pcen_bias
                )
                return result
            except Exception as e:
                logger.warning(f"librosa PCEN failed: {e}, using fallback")

        # Fallback: manual PCEN implementation
        return self._pcen_fallback(S)

    def _pcen_fallback(self, S: np.ndarray) -> np.ndarray:
        """Fallback PCEN implementation when librosa is not available."""
        eps = 1e-6

        # Ensure positive values
        S_pos = np.maximum(S, eps)

        # IIR smoothing along time axis
        smooth = np.zeros_like(S_pos)
        smooth[:, 0] = S_pos[:, 0]

        s = self.pcen_time_constant
        for t in range(1, S_pos.shape[1]):
            smooth[:, t] = (1 - s) * smooth[:, t-1] + s * S_pos[:, t]

        # PCEN formula
        normalized = (S_pos / (eps + smooth) ** self.pcen_gain + self.pcen_bias) ** self.pcen_power
        normalized = normalized - self.pcen_bias ** self.pcen_power

        return np.maximum(normalized, 0)

    @staticmethod
    def _normalize_01(x: np.ndarray) -> np.ndarray:
        """
        Normalize array to [0, 1] range.

        Exactly as V11.py:
        xmin = np.min(x)
        xmax = np.max(x)
        if xmax <= xmin: return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin + 1e-12)
        """
        x = x.astype(float)
        xmin = np.min(x)
        xmax = np.max(x)
        if xmax <= xmin:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin + 1e-12)


def apply_koren_filter(data: np.ndarray, **kwargs) -> Tuple[np.ndarray, str]:
    """
    Convenience function to apply Koren filter with default parameters.

    Args:
        data: Input spectrogram
        **kwargs: Override default filter parameters

    Returns:
        (filtered_data, log_message)
    """
    filter_obj = KorenFilter(**kwargs)
    return filter_obj.apply(data)
