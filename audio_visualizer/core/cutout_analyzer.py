"""
Core logic for extracting, normalizing, and analyzing spectrogram cutouts.
"""
import logging
import numpy as np
import json
import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Union

# Try imports for image processing
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    cv2 = None
    HAS_OPENCV = False

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    cm = None
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


def extract_spectrogram_cutout(
    S: np.ndarray, 
    freqs: np.ndarray, 
    times: np.ndarray, 
    t_start: float, 
    t_end: float, 
    f_low: float, 
    f_high: float
) -> Dict[str, np.ndarray]:
    """
    Extract a rectangular region from the spectrogram.
    
    Args:
        S: Full spectrogram 2D array [n_freqs, n_times]
        freqs: Frequency axis 1D array [n_freqs]
        times: Time axis 1D array [n_times]
        t_start, t_end: Time range in seconds
        f_low, f_high: Frequency range in Hz
        
    Returns:
        Dictionary with 'S_crop', 'times_crop', 'freqs_crop'
    """
    # Ensure ordering
    t0, t1 = min(t_start, t_end), max(t_start, t_end)
    f0, f1 = min(f_low, f_high), max(f_low, f_high)
    
    # Find indices using robust nearest neighbor (works for unsorted axes)
    t_idx_0 = (np.abs(times - t0)).argmin()
    t_idx_1 = (np.abs(times - t1)).argmin()
    
    f_idx_0 = (np.abs(freqs - f0)).argmin()
    f_idx_1 = (np.abs(freqs - f1)).argmin()
    
    # Ensure start < end
    if t_idx_0 > t_idx_1:
        t_idx_0, t_idx_1 = t_idx_1, t_idx_0
    if f_idx_0 > f_idx_1:
        f_idx_0, f_idx_1 = f_idx_1, f_idx_0
        
    # Include the end index
    t_idx_1 += 1
    f_idx_1 += 1
    
    # Clamp
    t_idx_1 = min(t_idx_1, len(times))
    f_idx_1 = min(f_idx_1, len(freqs))
    
    logger.info(f"Extracting: time_idx=[{t_idx_0}:{t_idx_1}], freq_idx=[{f_idx_0}:{f_idx_1}]")
    
    # Slice
    # S is [freqs, times]
    S_crop = S[f_idx_0:f_idx_1, t_idx_0:t_idx_1].copy()
    times_crop = times[t_idx_0:t_idx_1].copy()
    freqs_crop = freqs[f_idx_0:f_idx_1].copy()
    
    return {
        "S_crop": S_crop,
        "times_crop": times_crop,
        "freqs_crop": freqs_crop
    }


def normalize_cutout(
    S_crop: np.ndarray,
    mode: str = "auto",
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
    noise_std_threshold: float = 10.0,  # Threshold in dB
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply smart normalization to a spectrogram cutout.
    
    Args:
        S_crop: Input spectrogram (linear or dB)
        mode: 'auto', 'db_center', 'percentile', 'noise_floor', 'clahe_only', 'db_then_clahe'
        percentile_low/high: For contrast stretching
        noise_std_threshold: For auto-detection of noisy regions
        clahe_clip_limit: Contrast limit for CLAHE
        clahe_tile_grid_size: Grid size for CLAHE
        
    Returns:
        Tuple of (S_norm [0..1], info_dict)
    """
    info = {"original_mode": mode, "applied_mode": mode}
    
    # 1. Ensure dB scale first (most processing works better in log domain)
    # Heuristic to detect if already in dB: min value < 0 or range implies it
    # Spectrogram magnitudes are usually >= 0. 
    # If max is very large (> 200), it might be linear magnitude or power.
    # If min is negative, it's likely already dB or log.
    
    is_db = (S_crop.min() < 0) or (S_crop.max() < 100 and S_crop.min() > -200)
    
    if not is_db:
        # Convert to dB
        S_db = 20 * np.log10(np.maximum(S_crop, 1e-9))
        info["converted_to_db"] = True
    else:
        S_db = S_crop.copy()
        info["converted_to_db"] = False
        
    # Calculate stats
    p_low = np.percentile(S_db, percentile_low)
    p_high = np.percentile(S_db, percentile_high)
    median = np.median(S_db)
    std_dev = np.std(S_db)
    dynamic_range = p_high - p_low
    
    info.update({
        "p_low": float(p_low),
        "p_high": float(p_high),
        "median": float(median),
        "std": float(std_dev),
        "dynamic_range": float(dynamic_range)
    })
    
    # AUTO MODE DECISION
    if mode == "auto":
        if dynamic_range > 80:
            # Huge dynamic range, needs compression
            mode = "db_then_clahe"
        elif std_dev > noise_std_threshold:
            # High variance, likely noisy
            mode = "noise_floor"
        elif dynamic_range < 20:
            # Low contrast, stretch it
            mode = "percentile"
        else:
            # Standard case
            mode = "db_center"
        info["applied_mode"] = mode

    # APPLY MODES
    S_out = S_db
    
    if mode == "db_center":
        # Center around median and scale roughly
        # Map [median - 20dB, median + 20dB] to [0, 1] roughly
        # Or just simple percentile scaling which is robust
        S_out = np.clip(S_db, p_low, p_high)
        if p_high > p_low:
            S_out = (S_out - p_low) / (p_high - p_low)
        else:
            S_out = np.zeros_like(S_out)
            
    elif mode == "percentile":
        # Strict percentile scaling
        S_out = np.clip(S_db, p_low, p_high)
        denom = p_high - p_low
        S_out = (S_out - p_low) / (denom if denom > 1e-6 else 1.0)
        
    elif mode == "noise_floor":
        # "Lift" the floor: Gamma correction on the lower end
        # First normalize 0-1
        S_temp = np.clip(S_db, p_low, p_high)
        denom = p_high - p_low
        S_norm = (S_temp - p_low) / (denom if denom > 1e-6 else 1.0)
        
        # Apply gamma < 1 to boost dark areas (noise) to gray
        # Or gamma > 1 to crush it?
        # Requirement: "raise the lower tail so background noise becomes mid-gray"
        # This implies we want to see the noise structure, so we boost low values.
        gamma = 0.5 
        S_out = np.power(S_norm, gamma)
        info["gamma"] = gamma
        
    elif mode == "clahe_only" or mode == "db_then_clahe":
        # For CLAHE, we need normalized input mapped to uint8
        if mode == "clahe_only":
            # Use raw linear crop if possible, but usually CLAHE on log is better
            # If we already converted to dB, use that
            work_data = S_db
        else:
            work_data = S_db
            
        # Normalize to 0-1 for conversion
        w_min, w_max = work_data.min(), work_data.max()
        if w_max > w_min:
            work_norm = (work_data - w_min) / (w_max - w_min)
        else:
            work_norm = np.zeros_like(work_data)
            
        # Convert to uint8
        img_uint8 = (work_norm * 255).astype(np.uint8)
        
        if HAS_OPENCV:
            # OpenCV CLAHE
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
            img_clahe = clahe.apply(img_uint8)
            # Convert back to float 0-1
            S_out = img_clahe.astype(np.float32) / 255.0
        else:
            logger.warning("OpenCV not found, falling back to percentile normalization instead of CLAHE")
            # Fallback
            S_out = np.clip(S_db, p_low, p_high)
            denom = p_high - p_low
            S_out = (S_out - p_low) / (denom if denom > 1e-6 else 1.0)
            
    # Final safety clip
    S_out = np.clip(S_out, 0.0, 1.0)
    
    return S_out, info


def save_cutout_image(
    S_norm: np.ndarray, 
    times: np.ndarray, 
    freqs: np.ndarray, 
    path_png: str,
    cmap_name: str = 'magma'
) -> bool:
    """
    Save normalized spectrogram cutout as PNG.
    
    Args:
        S_norm: Normalized spectrogram [0..1]
        times, freqs: Axes (for aspect ratio calculation if needed)
        path_png: Output path
        cmap_name: Colormap name
        
    Returns:
        True if successful
    """
    try:
        # Use matplotlib for colormapping if available
        if HAS_MATPLOTLIB:
            plt.imsave(path_png, S_norm, cmap=cmap_name, origin='lower')
            return True
        
        if HAS_OPENCV:
            # Fallback to OpenCV grayscale or heatmap
            img_uint8 = (S_norm * 255).astype(np.uint8)
            # Flip vertically because origin='lower' in spectrograms
            img_uint8 = np.flipud(img_uint8)
            # Apply colormap
            img_color = cv2.applyColorMap(img_uint8, cv2.COLORMAP_MAGMA)
            cv2.imwrite(path_png, img_color)
            return True
            
        return False
    except Exception as e:
        logger.error(f"Failed to save cutout image: {e}")
        return False


def save_cutout_numpy(
    S_norm: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    path_npy: str,
    raw_data: Optional[np.ndarray] = None
) -> bool:
    """
    Save cutout data to NPY file.
    
    Args:
        S_norm: The normalized data used for the image
        times, freqs: Axis vectors
        path_npy: Output path
        raw_data: Optional original raw data (before normalization)
        
    Returns:
        True if successful
    """
    try:
        save_dict = {
            'normalized': S_norm,
            'times': times,
            'freqs': freqs
        }
        if raw_data is not None:
            save_dict['raw'] = raw_data
            
        np.save(path_npy, save_dict)
        return True
    except Exception as e:
        logger.error(f"Failed to save numpy data: {e}")
        return False


def write_cutout_metadata(rect_metadata: Dict, manifest_path: str) -> bool:
    """
    Append metadata to a JSON Lines manifest file.
    
    Args:
        rect_metadata: Dictionary of metadata
        manifest_path: Path to the manifest file
        
    Returns:
        True if successful
    """
    try:
        # Ensure fields are serializable
        serializable_meta = {}
        for k, v in rect_metadata.items():
            if isinstance(v, (np.integer, np.int64, int)):
                serializable_meta[k] = int(v)
            elif isinstance(v, (np.floating, np.float64, float)):
                serializable_meta[k] = float(v)
            elif isinstance(v, (np.ndarray, list)):
                serializable_meta[k] = str(v) # Avoid dumping huge arrays
            else:
                serializable_meta[k] = v
                
        # Add timestamp if missing
        if 'created_utc' not in serializable_meta:
            serializable_meta['created_utc'] = datetime.datetime.utcnow().isoformat()
            
        with open(manifest_path, 'a', encoding='utf-8') as f:
            json.dump(serializable_meta, f)
            f.write('\n')
            
        return True
    except Exception as e:
        logger.error(f"Failed to write metadata: {e}")
        return False

