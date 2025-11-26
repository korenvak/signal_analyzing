"""
Memory Pool System
Provides preallocated workspace buffers for zero-copy operations and memory reuse
"""

import numpy as np
from typing import Dict, Tuple, Optional
import threading
import logging

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

logger = logging.getLogger(__name__)


class WorkspacePool:
    """
    Pool of preallocated workspace buffers.
    Eliminates malloc/free overhead by reusing buffers.
    """
    
    def __init__(self, use_gpu: bool = False):
        """Initialize workspace pool.
        
        Args:
            use_gpu: Use GPU memory if available
        """
        self.use_gpu = use_gpu and HAS_CUPY
        self.workspaces: Dict[Tuple, any] = {}  # (shape, dtype) -> array
        self.lock = threading.RLock()
        self.stats = {
            'allocations': 0,
            'reuses': 0,
            'total_bytes': 0
        }
        
        logger.info(f"WorkspacePool initialized (GPU: {self.use_gpu})")
    
    def get_workspace(self, shape: Tuple, dtype=np.float32, fill_value: Optional[float] = None):
        """Get a workspace buffer (reused if available).
        
        Args:
            shape: Required shape
            dtype: Required dtype (default: float32)
            fill_value: If provided, fill buffer with this value
            
        Returns:
            Workspace array (NumPy or CuPy)
        """
        key = (shape, np.dtype(dtype).str)
        
        with self.lock:
            if key in self.workspaces:
                # Reuse existing workspace
                workspace = self.workspaces[key]
                self.stats['reuses'] += 1
            else:
                # Allocate new workspace
                if self.use_gpu:
                    workspace = cp.empty(shape, dtype=dtype)
                else:
                    # Use C-contiguous allocation
                    workspace = np.empty(shape, dtype=dtype, order='C')
                
                self.workspaces[key] = workspace
                self.stats['allocations'] += 1
                self.stats['total_bytes'] += workspace.nbytes
                
                logger.debug(f"Allocated workspace: {shape} {dtype} ({workspace.nbytes / 1024**2:.1f} MB)")
            
            # Fill if requested
            if fill_value is not None:
                workspace.fill(fill_value)
            
            return workspace
    
    def get_output_workspace(self, shape: Tuple, dtype=np.float32):
        """Get workspace specifically for output (zero-initialized).
        
        Args:
            shape: Required shape
            dtype: Required dtype
            
        Returns:
            Workspace array filled with zeros
        """
        return self.get_workspace(shape, dtype, fill_value=0.0)
    
    def clear(self):
        """Clear all workspaces and free memory."""
        with self.lock:
            self.workspaces.clear()
            self.stats['allocations'] = 0
            self.stats['reuses'] = 0
            self.stats['total_bytes'] = 0
            
            if self.use_gpu and HAS_CUPY:
                try:
                    # Only free GPU memory if CUDA device is available
                    cp.cuda.Device().compute_capability  # Check if GPU is accessible
                    cp.get_default_memory_pool().free_all_blocks()
                except Exception:
                    # GPU not available or accessible, skip cleanup
                    pass
        
        logger.info("WorkspacePool cleared")
    
    def get_stats(self) -> Dict:
        """Get pool statistics."""
        with self.lock:
            return {
                **self.stats,
                'num_workspaces': len(self.workspaces),
                'reuse_rate': (self.stats['reuses'] / max(1, self.stats['allocations'] + self.stats['reuses']) * 100)
            }


class MemoryOptimizer:
    """
    Memory optimization utilities.
    Enforces float32, provides zero-copy operations, manages buffer reuse.
    """
    
    def __init__(self):
        """Initialize memory optimizer."""
        # Workspace pools
        self.cpu_pool = WorkspacePool(use_gpu=False)
        self.gpu_pool = WorkspacePool(use_gpu=True) if HAS_CUPY else None
        
        # Enforce float32 globally
        self.enforce_float32 = True
        
        logger.info("MemoryOptimizer initialized")
    
    def ensure_float32(self, data: np.ndarray, copy: bool = False) -> np.ndarray:
        """Ensure data is float32.
        
        Args:
            data: Input array
            copy: If True, always copy; if False, copy only if conversion needed
            
        Returns:
            Array as float32
        """
        if data.dtype == np.float32:
            return data.copy() if copy else data
        else:
            return data.astype(np.float32)
    
    def ensure_c_contiguous(self, data: np.ndarray) -> np.ndarray:
        """Ensure array is C-contiguous for optimal performance.
        
        Args:
            data: Input array
            
        Returns:
            C-contiguous array
        """
        if data.flags['C_CONTIGUOUS']:
            return data
        else:
            return np.ascontiguousarray(data, dtype=data.dtype)
    
    def get_cpu_workspace(self, shape: Tuple, dtype=np.float32) -> np.ndarray:
        """Get CPU workspace buffer.
        
        Args:
            shape: Required shape
            dtype: Required dtype
            
        Returns:
            Workspace array
        """
        return self.cpu_pool.get_workspace(shape, dtype)
    
    def get_gpu_workspace(self, shape: Tuple, dtype=np.float32):
        """Get GPU workspace buffer.
        
        Args:
            shape: Required shape
            dtype: Required dtype
            
        Returns:
            GPU workspace array (CuPy) or None if no GPU
        """
        if self.gpu_pool:
            return self.gpu_pool.get_workspace(shape, dtype)
        return None
    
    def copy_to_gpu_zerocopy(self, cpu_array: np.ndarray):
        """Copy to GPU using zero-copy when possible.
        
        Args:
            cpu_array: NumPy array
            
        Returns:
            CuPy array
        """
        if not HAS_CUPY:
            return None
        
        # Ensure C-contiguous for faster transfer
        if not cpu_array.flags['C_CONTIGUOUS']:
            cpu_array = np.ascontiguousarray(cpu_array)
        
        # Use CuPy's optimized transfer
        return cp.asarray(cpu_array)
    
    def copy_to_gpu_pinned(self, cpu_array: np.ndarray):
        """Copy to GPU using pinned (page-locked) memory for 2-3x faster transfers.
        
        Args:
            cpu_array: NumPy array
            
        Returns:
            CuPy array
        """
        if not HAS_CUPY:
            return None
        
        # Ensure C-contiguous
        if not cpu_array.flags['C_CONTIGUOUS']:
            cpu_array = np.ascontiguousarray(cpu_array)
        
        try:
            # Allocate pinned memory and copy data
            pinned_mem = cp.cuda.alloc_pinned_memory(cpu_array.nbytes)
            pinned_array = np.frombuffer(pinned_mem, dtype=cpu_array.dtype, count=cpu_array.size)
            pinned_array = pinned_array.reshape(cpu_array.shape)
            np.copyto(pinned_array, cpu_array)
            
            # Transfer from pinned memory (2-3x faster than regular memory)
            gpu_array = cp.asarray(pinned_array)
            
            return gpu_array
            
        except Exception as e:
            # Fallback to regular memory transfer if pinned memory fails
            logger.debug(f"Pinned memory transfer failed, using regular: {e}")
            return cp.asarray(cpu_array)
    
    def copy_to_cpu_zerocopy(self, gpu_array) -> np.ndarray:
        """Copy to CPU using zero-copy when possible.
        
        Args:
            gpu_array: CuPy array
            
        Returns:
            NumPy array
        """
        if not HAS_CUPY:
            return None
        
        return cp.asnumpy(gpu_array)
    
    def inplace_multiply(self, array, factor, out=None):
        """In-place multiplication to avoid temporary arrays.
        
        Args:
            array: Input array
            factor: Multiplication factor (scalar or array)
            out: Output buffer (if None, modifies array in-place)
            
        Returns:
            Result array
        """
        if out is None:
            if isinstance(array, np.ndarray):
                np.multiply(array, factor, out=array)
                return array
            elif HAS_CUPY and isinstance(array, cp.ndarray):
                cp.multiply(array, factor, out=array)
                return array
        else:
            if isinstance(array, np.ndarray):
                np.multiply(array, factor, out=out)
                return out
            elif HAS_CUPY and isinstance(array, cp.ndarray):
                cp.multiply(array, factor, out=out)
                return out
    
    def inplace_add(self, array, value, out=None):
        """In-place addition to avoid temporary arrays.
        
        Args:
            array: Input array
            value: Value to add
            out: Output buffer
            
        Returns:
            Result array
        """
        if out is None:
            if isinstance(array, np.ndarray):
                np.add(array, value, out=array)
                return array
            elif HAS_CUPY and isinstance(array, cp.ndarray):
                cp.add(array, value, out=array)
                return array
        else:
            if isinstance(array, np.ndarray):
                np.add(array, value, out=out)
                return out
            elif HAS_CUPY and isinstance(array, cp.ndarray):
                cp.add(array, value, out=out)
                return out
    
    def inplace_log10(self, array, out=None):
        """In-place log10 to avoid temporary arrays.
        
        Args:
            array: Input array
            out: Output buffer (if None, modifies array in-place)
            
        Returns:
            Result array
        """
        if out is None:
            if isinstance(array, np.ndarray):
                np.log10(array, out=array)
                return array
            elif HAS_CUPY and isinstance(array, cp.ndarray):
                cp.log10(array, out=array)
                return array
        else:
            if isinstance(array, np.ndarray):
                np.log10(array, out=out)
                return out
            elif HAS_CUPY and isinstance(array, cp.ndarray):
                cp.log10(array, out=out)
                return out
    
    def inplace_maximum(self, array, value, out=None):
        """In-place maximum to avoid temporary arrays.
        
        Args:
            array: Input array
            value: Comparison value
            out: Output buffer (if None, modifies array in-place)
            
        Returns:
            Result array
        """
        if out is None:
            if isinstance(array, np.ndarray):
                np.maximum(array, value, out=array)
                return array
            elif HAS_CUPY and isinstance(array, cp.ndarray):
                cp.maximum(array, value, out=array)
                return array
        else:
            if isinstance(array, np.ndarray):
                np.maximum(array, value, out=out)
                return out
            elif HAS_CUPY and isinstance(array, cp.ndarray):
                cp.maximum(array, value, out=out)
                return out
    
    def inplace_abs(self, array, out=None):
        """In-place absolute value to avoid temporary arrays.
        
        Args:
            array: Input array (complex or real)
            out: Output buffer (if None, creates new array for complex input)
            
        Returns:
            Result array
        """
        if isinstance(array, np.ndarray):
            if out is None and np.iscomplexobj(array):
                # Complex abs needs output buffer (can't modify in-place)
                out = self.get_cpu_workspace(array.shape, dtype=np.float32)
            np.abs(array, out=out if out is not None else array)
            return out if out is not None else array
        elif HAS_CUPY and isinstance(array, cp.ndarray):
            if out is None and cp.iscomplexobj(array):
                # Complex abs needs output buffer
                out = self.get_gpu_workspace(array.shape, dtype=cp.float32)
            cp.abs(array, out=out if out is not None else array)
            return out if out is not None else array
    
    def cleanup(self):
        """Cleanup all workspace pools."""
        self.cpu_pool.clear()
        if self.gpu_pool:
            self.gpu_pool.clear()
        logger.info("Memory pools cleared")
    
    def get_stats(self) -> Dict:
        """Get memory optimization statistics."""
        stats = {
            'cpu_pool': self.cpu_pool.get_stats(),
        }
        if self.gpu_pool:
            stats['gpu_pool'] = self.gpu_pool.get_stats()
        return stats


# Global instance
_memory_optimizer = None


def get_memory_optimizer() -> MemoryOptimizer:
    """Get global memory optimizer instance."""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer

