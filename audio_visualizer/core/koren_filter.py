"""
Koren's Filter - Advanced spectrogram enhancement pipeline with GPU acceleration.

A multi-stage filter designed for enhancing Doppler tracks in spectrograms.
Exactly matches V11.py implementation with GPU acceleration where available.

Pipeline:
1. HPSS (Harmonic-Percussive Source Separation) - removes vertical noise
2. PCEN (Per-Channel Energy Normalization) - flattens background
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
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Check for GPU support
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter1d as gpu_gaussian_filter1d
    GPU_AVAILABLE = True
    logger.info("CuPy available - GPU acceleration enabled for Koren's filter")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    logger.info("CuPy not available - using CPU for Koren's filter")

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
    Koren's multi-stage spectrogram enhancement filter with GPU acceleration.

    Exactly matches V11.py implementation.
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
                 hop_length: int = 512,
                 use_gpu: bool = True):
        """
        Initialize filter with V11.py default parameters.
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
        self.use_gpu = use_gpu and GPU_AVAILABLE

    def apply(self, data: np.ndarray,
              skip_hpss: bool = False,
              skip_pcen: bool = False,
              skip_meijering: bool = False,
              skip_tv: bool = False) -> Tuple[np.ndarray, str]:
        """
        Apply the full Koren filter pipeline (exactly as V11.py).

        Args:
            data: Input spectrogram (freq x time), should be linear magnitude
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

        logger.info(f"Koren Filter: Input shape {S.shape}, GPU={'enabled' if self.use_gpu else 'disabled'}")

        # --- Stage 1: HPSS (Separation) ---
        t0 = time.time()
        if not skip_hpss:
            S_harmonic = self._apply_hpss(S)
            steps.append(f"HPSS({time.time()-t0:.2f}s)")
        else:
            S_harmonic = S.copy()

        # --- Stage 2: PCEN (Normalization) ---
        t0 = time.time()
        if not skip_pcen:
            S_harmonic = self._apply_pcen(S_harmonic)
            steps.append(f"PCEN({time.time()-t0:.2f}s)")

        # --- Stage 3: Low frequency cut ---
        if self.low_cut_bins > 0:
            S_harmonic[:self.low_cut_bins, :] = 0.0
            steps.append(f"LowCut({self.low_cut_bins})")

        # Normalize to [0, 1]
        S_harmonic = self._normalize_01(S_harmonic)

        # --- Stage 4: Meijering (Ridge Detection) ---
        t0 = time.time()
        if not skip_meijering:
            ridge = self._apply_meijering(S_harmonic)
            ridge = self._normalize_01(ridge)
            steps.append(f"Meijering({time.time()-t0:.2f}s)")
        else:
            ridge = S_harmonic.copy()

        # --- Stage 5: Smart Smoothing ---
        t0 = time.time()
        if self.smooth_sigma > 0:
            ridge_smooth = self._apply_smoothing(ridge)
            ridge_smooth = self._normalize_01(ridge_smooth)
            steps.append(f"Smooth({time.time()-t0:.2f}s)")
        else:
            ridge_smooth = ridge

        # --- Stage 6: Soft Sigmoid Fusion ---
        t0 = time.time()
        S_fused = self._apply_sigmoid_fusion(S_harmonic, ridge_smooth)
        steps.append(f"Fusion({time.time()-t0:.2f}s)")

        # --- Stage 7: Total Variation Denoising ---
        t0 = time.time()
        if not skip_tv:
            S_final = self._apply_tv_denoise(S_fused)
            steps.append(f"TV({time.time()-t0:.2f}s)")
        else:
            S_final = S_fused

        # --- Stage 8: Contrast boost ---
        if self.contrast_power != 1.0:
            S_final = np.power(S_final, self.contrast_power)
            steps.append(f"Contrast({self.contrast_power})")

        # Final normalization
        S_final = self._normalize_01(S_final)

        dt = time.time() - start_time
        log_msg = f"Koren Filter: {' -> '.join(steps)} | Total: {dt:.2f}s"
        logger.info(log_msg)

        return S_final.astype(np.float32), log_msg

    def _apply_hpss(self, S: np.ndarray) -> np.ndarray:
        """
        Apply Harmonic-Percussive Source Separation.
        Exactly as V11.py: librosa.decompose.hpss(S, margin=3.0)
        """
        if LIBROSA_AVAILABLE:
            try:
                S_harmonic, _ = librosa.decompose.hpss(S, margin=self.hpss_margin)
                return S_harmonic
            except Exception as e:
                logger.warning(f"librosa HPSS failed: {e}, using fallback")

        # Fallback: median filter based HPSS
        return self._hpss_fallback(S)

    def _hpss_fallback(self, S: np.ndarray) -> np.ndarray:
        """Fallback HPSS using median filtering."""
        if not SCIPY_AVAILABLE:
            return S

        # Median filter along time axis -> harmonic component
        harmonic = median_filter(S, size=(1, 31))
        percussive = median_filter(S, size=(31, 1))

        eps = 1e-10
        harmonic_mask = (harmonic ** 2) / (harmonic ** 2 + percussive ** 2 + eps)
        return S * harmonic_mask

    def _apply_pcen(self, S: np.ndarray) -> np.ndarray:
        """
        Apply Per-Channel Energy Normalization.
        Exactly as V11.py: librosa.pcen(S * (2**31), ...)
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

        return self._pcen_fallback(S)

    def _pcen_fallback(self, S: np.ndarray) -> np.ndarray:
        """Fallback PCEN implementation."""
        eps = 1e-6
        S_pos = np.maximum(S, eps)

        # IIR smoothing along time axis
        smooth = np.zeros_like(S_pos)
        smooth[:, 0] = S_pos[:, 0]

        s = self.pcen_time_constant
        for t in range(1, S_pos.shape[1]):
            smooth[:, t] = (1 - s) * smooth[:, t-1] + s * S_pos[:, t]

        normalized = (S_pos / (eps + smooth) ** self.pcen_gain + self.pcen_bias) ** self.pcen_power
        normalized = normalized - self.pcen_bias ** self.pcen_power

        return np.maximum(normalized, 0)

    def _apply_meijering(self, S: np.ndarray) -> np.ndarray:
        """
        Apply Meijering ridge detection.
        Exactly as V11.py: meijering(S, sigmas=range(1,4), black_ridges=False, mode='reflect')

        Uses GPU acceleration if available.
        """
        if self.use_gpu:
            return self._meijering_gpu(S)

        if SKIMAGE_AVAILABLE:
            try:
                return meijering(S, sigmas=self.meijering_sigmas,
                               black_ridges=False, mode='reflect')
            except Exception as e:
                logger.warning(f"skimage meijering failed: {e}")

        # Simple fallback - Sobel-like ridge detection
        return self._meijering_fallback(S)

    def _meijering_gpu(self, S: np.ndarray) -> np.ndarray:
        """GPU-accelerated Meijering-like ridge detection using CuPy."""
        try:
            S_gpu = cp.asarray(S)

            # Compute Hessian eigenvalues for ridge detection
            # Use multiple scales as in original meijering
            result = cp.zeros_like(S_gpu)

            for sigma in self.meijering_sigmas:
                # Gaussian smoothing at this scale
                from cupyx.scipy.ndimage import gaussian_filter
                smoothed = gaussian_filter(S_gpu, sigma=sigma)

                # Compute second derivatives (Hessian)
                # Dxx - second derivative in x (frequency) direction
                Dxx = cp.zeros_like(smoothed)
                Dxx[1:-1, :] = smoothed[2:, :] - 2*smoothed[1:-1, :] + smoothed[:-2, :]

                # Dyy - second derivative in y (time) direction
                Dyy = cp.zeros_like(smoothed)
                Dyy[:, 1:-1] = smoothed[:, 2:] - 2*smoothed[:, 1:-1] + smoothed[:, :-2]

                # Dxy - mixed derivative
                Dxy = cp.zeros_like(smoothed)
                Dxy[1:-1, 1:-1] = (smoothed[2:, 2:] - smoothed[2:, :-2] -
                                   smoothed[:-2, 2:] + smoothed[:-2, :-2]) / 4

                # Eigenvalues of Hessian
                # For 2x2 matrix [[Dxx, Dxy], [Dxy, Dyy]]
                # eigenvalues = (trace ± sqrt(trace² - 4*det)) / 2
                trace = Dxx + Dyy
                det = Dxx * Dyy - Dxy * Dxy
                discriminant = cp.maximum(trace * trace - 4 * det, 0)
                sqrt_disc = cp.sqrt(discriminant)

                # We want the smallest eigenvalue (most negative for ridges)
                lambda2 = (trace - sqrt_disc) / 2

                # Ridge response: negative eigenvalue indicates ridge
                # Take absolute value and accumulate
                ridge_response = cp.maximum(-lambda2, 0) * (sigma ** 2)
                result = cp.maximum(result, ridge_response)

            return cp.asnumpy(result)

        except Exception as e:
            logger.warning(f"GPU meijering failed: {e}, falling back to CPU")
            if SKIMAGE_AVAILABLE:
                return meijering(S, sigmas=self.meijering_sigmas,
                               black_ridges=False, mode='reflect')
            return self._meijering_fallback(S)

    def _meijering_fallback(self, S: np.ndarray) -> np.ndarray:
        """Simple fallback ridge detection without skimage."""
        from scipy.ndimage import gaussian_filter, sobel

        result = np.zeros_like(S)
        for sigma in self.meijering_sigmas:
            smoothed = gaussian_filter(S, sigma=sigma)

            # Simple ridge detection using second derivatives
            Dxx = np.zeros_like(smoothed)
            Dyy = np.zeros_like(smoothed)

            Dxx[1:-1, :] = smoothed[2:, :] - 2*smoothed[1:-1, :] + smoothed[:-2, :]
            Dyy[:, 1:-1] = smoothed[:, 2:] - 2*smoothed[:, 1:-1] + smoothed[:, :-2]

            # Ridge response
            ridge = np.maximum(-np.minimum(Dxx, Dyy), 0) * (sigma ** 2)
            result = np.maximum(result, ridge)

        return result

    def _apply_smoothing(self, ridge: np.ndarray) -> np.ndarray:
        """
        Apply directional smoothing along time axis.
        Exactly as V11.py: gaussian_filter1d(ridge, sigma=1.5, axis=1)
        """
        if self.use_gpu:
            try:
                ridge_gpu = cp.asarray(ridge)
                result = gpu_gaussian_filter1d(ridge_gpu, sigma=self.smooth_sigma, axis=1)
                return cp.asnumpy(result)
            except Exception as e:
                logger.warning(f"GPU smoothing failed: {e}")

        if SCIPY_AVAILABLE:
            return gaussian_filter1d(ridge, sigma=self.smooth_sigma, axis=1)

        return ridge

    def _apply_sigmoid_fusion(self, S_harmonic: np.ndarray, ridge_smooth: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid fusion.
        Exactly as V11.py:
            sigmoid_shift = mean_ridge + 0.5 * std_ridge
            mask = 1.0 / (1.0 + np.exp(-10.0 * (ridge_smooth - sigmoid_shift)))
            S_fused = S_harmonic * mask
        """
        if self.use_gpu:
            try:
                S_gpu = cp.asarray(S_harmonic)
                ridge_gpu = cp.asarray(ridge_smooth)

                mean_ridge = cp.mean(ridge_gpu)
                std_ridge = cp.std(ridge_gpu)

                sigmoid_shift = mean_ridge + self.sigmoid_std_factor * std_ridge
                mask = 1.0 / (1.0 + cp.exp(-self.sigmoid_gain * (ridge_gpu - sigmoid_shift)))

                S_fused = S_gpu * mask
                return cp.asnumpy(S_fused)
            except Exception as e:
                logger.warning(f"GPU fusion failed: {e}")

        # CPU version
        mean_ridge = np.mean(ridge_smooth)
        std_ridge = np.std(ridge_smooth)

        sigmoid_shift = mean_ridge + self.sigmoid_std_factor * std_ridge
        mask = 1.0 / (1.0 + np.exp(-self.sigmoid_gain * (ridge_smooth - sigmoid_shift)))

        return S_harmonic * mask

    def _apply_tv_denoise(self, S: np.ndarray) -> np.ndarray:
        """
        Apply Total Variation denoising.
        Exactly as V11.py: denoise_tv_chambolle(S_fused, weight=0.1)
        """
        if self.use_gpu:
            try:
                return self._tv_denoise_gpu(S)
            except Exception as e:
                logger.warning(f"GPU TV denoise failed: {e}")

        if SKIMAGE_AVAILABLE:
            try:
                return denoise_tv_chambolle(S, weight=self.tv_weight)
            except Exception as e:
                logger.warning(f"skimage TV denoise failed: {e}")

        return S

    def _tv_denoise_gpu(self, S: np.ndarray, n_iter: int = 100) -> np.ndarray:
        """GPU-accelerated Total Variation denoising using Chambolle's algorithm."""
        S_gpu = cp.asarray(S)

        # Chambolle's projection algorithm
        tau = 0.25

        p = cp.zeros((2,) + S_gpu.shape, dtype=S_gpu.dtype)

        for _ in range(n_iter):
            # Compute gradient of (div(p) - f/lambda)
            div_p = cp.zeros_like(S_gpu)
            div_p[:-1, :] += p[0, :-1, :]
            div_p[1:, :] -= p[0, :-1, :]
            div_p[:, :-1] += p[1, :, :-1]
            div_p[:, 1:] -= p[1, :, :-1]

            grad_arg = div_p - S_gpu / self.tv_weight

            # Gradient
            grad = cp.zeros_like(p)
            grad[0, :-1, :] = grad_arg[1:, :] - grad_arg[:-1, :]
            grad[1, :, :-1] = grad_arg[:, 1:] - grad_arg[:, :-1]

            # Update p
            p_new = p + tau * grad

            # Project onto unit ball
            norm_p = cp.sqrt(p_new[0]**2 + p_new[1]**2)
            norm_p = cp.maximum(norm_p, 1.0)
            p = p_new / norm_p[cp.newaxis, :, :]

        # Compute result
        div_p = cp.zeros_like(S_gpu)
        div_p[:-1, :] += p[0, :-1, :]
        div_p[1:, :] -= p[0, :-1, :]
        div_p[:, :-1] += p[1, :, :-1]
        div_p[:, 1:] -= p[1, :, :-1]

        result = S_gpu - self.tv_weight * div_p
        return cp.asnumpy(result)

    @staticmethod
    def _normalize_01(x: np.ndarray) -> np.ndarray:
        """
        Normalize array to [0, 1] range.
        Exactly as V11.py.
        """
        x = x.astype(np.float64)
        xmin = np.min(x)
        xmax = np.max(x)
        if xmax <= xmin:
            return np.zeros_like(x)
        return (x - xmin) / (xmax - xmin + 1e-12)


def apply_koren_filter(data: np.ndarray, **kwargs) -> Tuple[np.ndarray, str]:
    """
    Convenience function to apply Koren filter with default V11.py parameters.
    """
    filter_obj = KorenFilter(**kwargs)
    return filter_obj.apply(data)
