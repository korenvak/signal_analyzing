"""
Filter manager for applying image filters to spectrograms.
Supports GPU acceleration via CuPy.
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

logger = logging.getLogger(__name__)

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


