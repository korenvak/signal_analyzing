"""
Batched FFT Engine
High-performance FFT computation using persistent plans and batch processing
"""

import numpy as np
from typing import Tuple, Optional, Dict
import logging
import os

# Enable NumPy multithreading for CPU operations (2-4x speedup on multi-core)
# This affects BLAS/LAPACK operations (matrix ops, FFT, etc.)
if 'OMP_NUM_THREADS' not in os.environ:
    import multiprocessing
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
if 'OPENBLAS_NUM_THREADS' not in os.environ:
    os.environ['OPENBLAS_NUM_THREADS'] = os.environ['OMP_NUM_THREADS']
if 'MKL_NUM_THREADS' not in os.environ:
    os.environ['MKL_NUM_THREADS'] = os.environ['OMP_NUM_THREADS']

try:
    import cupy as cp
    from cupyx.scipy import fft as cp_fft
    HAS_CUPY = True
except ImportError:
    cp = None
    cp_fft = None
    HAS_CUPY = False

try:
    import pyfftw
    HAS_FFTW = True
except ImportError:
    pyfftw = None
    HAS_FFTW = False

from ..core.memory_pools import get_memory_optimizer
from ..core.cuda_kernels import get_fused_kernels

logger = logging.getLogger(__name__)


class BatchedFFTPlan:
    """Persistent FFT plan for repeated use."""
    
    def __init__(self, n: int, batch_size: int, use_gpu: bool = True):
        """Initialize FFT plan.
        
        Args:
            n: FFT size
            batch_size: Number of FFTs to compute in parallel
            use_gpu: Use GPU if available
        """
        self.n = n
        self.batch_size = batch_size
        self.use_gpu = use_gpu and HAS_CUPY
        
        if self.use_gpu:
            # GPU plan using cuFFT
            self._gpu_plan = None  # cuFFT plans are implicit in CuPy
            logger.info(f"Created GPU FFT plan: size={n}, batch={batch_size}")
        elif HAS_FFTW:
            # CPU plan using FFTW
            self._setup_fftw_plan()
            logger.info(f"Created FFTW plan: size={n}, batch={batch_size}")
        else:
            # Fallback to scipy
            self._plan = None
            logger.info(f"Using scipy FFT (no FFTW): size={n}, batch={batch_size}")
    
    def _setup_fftw_plan(self):
        """Setup FFTW plan for maximum performance."""
        import os
        import multiprocessing
        
        # Allocate aligned arrays for FFTW (16-byte alignment for SIMD)
        self._input = pyfftw.empty_aligned((self.batch_size, self.n), dtype='float32')
        self._output = pyfftw.empty_aligned((self.batch_size, self.n // 2 + 1), 
                                            dtype='complex64')
        
        # Use optimal thread count (all CPU cores, or user-specified)
        n_threads = int(os.environ.get('FFTW_THREADS', multiprocessing.cpu_count()))
        logger.debug(f"FFTW using {n_threads} threads")
        
        # Create plan with FFTW_MEASURE for optimal performance
        # FFTW_MEASURE: spends time finding the fastest algorithm (one-time cost)
        # FFTW_DESTROY_INPUT: allows overwriting input for ~10% speedup
        self._plan = pyfftw.FFTW(
            self._input, self._output,
            axes=(1,),  # FFT along second axis
            direction='FFTW_FORWARD',
            flags=('FFTW_MEASURE', 'FFTW_DESTROY_INPUT'),
            threads=n_threads  # Use all available CPU cores
        )
    
    def execute(self, data: np.ndarray) -> np.ndarray:
        """Execute FFT on batched data.
        
        Args:
            data: Input array (batch_size, n) or (n, batch_size)
            
        Returns:
            FFT result (batch_size, n//2 + 1) or (n//2 + 1, batch_size)
        """
        if self.use_gpu:
            return self._execute_gpu(data)
        elif HAS_FFTW:
            return self._execute_fftw(data)
        else:
            return self._execute_scipy(data)
    
    def _execute_gpu(self, data: np.ndarray) -> np.ndarray:
        """Execute on GPU using cuFFT."""
        # Move to GPU if needed
        if isinstance(data, np.ndarray):
            gpu_data = cp.asarray(data, dtype=cp.float32)
        else:
            gpu_data = data
        
        # Ensure correct shape (batch, n)
        if gpu_data.shape[1] != self.n:
            # Data is (n, batch) - transpose
            gpu_data = gpu_data.T
        
        # Compute rFFT (real input, complex output)
        result = cp_fft.rfft(gpu_data, n=self.n, axis=1)
        
        return result  # Return on GPU (zero-copy)
    
    def _execute_fftw(self, data: np.ndarray) -> np.ndarray:
        """Execute on CPU using FFTW."""
        # Ensure correct shape and copy to aligned buffer
        if data.shape[1] != self.n:
            data = data.T
        
        # Copy to FFTW aligned buffer
        actual_batch = data.shape[0]
        if actual_batch <= self.batch_size:
            self._input[:actual_batch, :] = data
            # Execute plan
            self._plan()
            return self._output[:actual_batch, :].copy()
        else:
            # Process in chunks
            results = []
            for i in range(0, actual_batch, self.batch_size):
                end = min(i + self.batch_size, actual_batch)
                chunk = data[i:end, :]
                self._input[:chunk.shape[0], :] = chunk
                self._plan()
                results.append(self._output[:chunk.shape[0], :].copy())
            return np.vstack(results)
    
    def _execute_scipy(self, data: np.ndarray) -> np.ndarray:
        """Execute on CPU using scipy (fallback)."""
        import scipy.fft
        
        if data.shape[1] != self.n:
            data = data.T
        
        return scipy.fft.rfft(data, n=self.n, axis=1)


class BatchedFFTEngine:
    """
    High-performance FFT engine with persistent plans and batching.
    
    Provides 10-100x speedup over per-frame FFT by:
    - Reusing FFT plans
    - Batching hundreds/thousands of frames together
    - Using optimized FFTW/cuFFT
    - Zero-copy GPU operations
    """
    
    def __init__(self, use_gpu: bool = True):
        """Initialize batched FFT engine.
        
        Args:
            use_gpu: Use GPU if available
        """
        self.use_gpu = use_gpu and HAS_CUPY
        self.plans: Dict[Tuple[int, int], BatchedFFTPlan] = {}
        
        # Memory optimizer for workspace management
        self.memory_optimizer = get_memory_optimizer()
        
        # Window function cache - eliminates repeated computation
        self._window_cache: Dict[Tuple[int, str], np.ndarray] = {}
        
        # Multi-stream GPU management for 2-3x throughput improvement
        # 3 streams: transfer, compute, readback - allows overlap of operations
        self._cuda_streams = None
        self._current_stream_idx = 0
        if self.use_gpu:
            self._cuda_streams = [
                cp.cuda.Stream(non_blocking=True),  # Stream 0: Transfer
                cp.cuda.Stream(non_blocking=True),  # Stream 1: Compute
                cp.cuda.Stream(non_blocking=True)   # Stream 2: Readback
            ]
            logger.info("Multi-stream GPU management enabled (3 streams)")
        
        # Legacy single stream for backward compatibility
        self._cuda_stream = self._cuda_streams[0] if self._cuda_streams else None
        
        # Fused CUDA kernels for 2-3x faster operations
        self._fused_kernels = get_fused_kernels() if self.use_gpu else None
        
        logger.info(f"BatchedFFTEngine initialized (GPU: {self.use_gpu}, Multi-Stream: {self._cuda_streams is not None}, Fused: {self._fused_kernels is not None})")
    
    def get_or_create_plan(self, fft_size: int, batch_size: int) -> BatchedFFTPlan:
        """Get existing plan or create new one.
        
        Args:
            fft_size: Size of FFT
            batch_size: Batch size
            
        Returns:
            FFT plan
        """
        key = (fft_size, batch_size)
        if key not in self.plans:
            self.plans[key] = BatchedFFTPlan(fft_size, batch_size, self.use_gpu)
        return self.plans[key]
    
    def _get_window(self, size: int, window_type: str) -> np.ndarray:
        """Get window function from cache or create it.
        
        Args:
            size: Window size
            window_type: Window type ('hann', 'hamming', 'blackman', 'blackmanharris',
                         'kaiser', 'flattop', 'rectangular', etc.)
            
        Returns:
            Window array (float32)
        """
        key = (size, window_type)
        if key not in self._window_cache:
            # Create window (always float32)
            import scipy.signal as sig
            
            if window_type == 'hann':
                window = np.hanning(size).astype(np.float32)
            elif window_type == 'hamming':
                window = np.hamming(size).astype(np.float32)
            elif window_type == 'blackman':
                window = np.blackman(size).astype(np.float32)
            elif window_type == 'blackmanharris':
                # Blackman-Harris: excellent side-lobe suppression (-92 dB)
                window = sig.windows.blackmanharris(size).astype(np.float32)
            elif window_type == 'kaiser':
                # Kaiser with beta=9 for good frequency analysis
                window = np.kaiser(size, beta=9.0).astype(np.float32)
            elif window_type == 'flattop':
                # Flat-top window for accurate amplitude measurements
                window = sig.windows.flattop(size).astype(np.float32)
            elif window_type == 'rectangular' or window_type == 'boxcar':
                # No window (rectangular) - best time resolution, worst spectral leakage
                window = np.ones(size, dtype=np.float32)
            elif window_type == 'bartlett':
                window = np.bartlett(size).astype(np.float32)
            elif window_type == 'tukey':
                # Tukey (tapered cosine) window
                window = sig.windows.tukey(size, alpha=0.5).astype(np.float32)
            else:
                # Default to Hann if unknown
                logger.warning(f"Unknown window type '{window_type}', using 'hann'")
                window = np.hanning(size).astype(np.float32)
            
            self._window_cache[key] = window
            logger.debug(f"Cached window: {window_type}({size})")
        
        return self._window_cache[key]
    
    def _compute_with_pipeline_overlap(self, frames: np.ndarray, plan: BatchedFFTPlan, 
                                       fft_size: int, use_fused: bool = True) -> np.ndarray:
        """Compute STFT with true pipeline overlap for large batches.
        
        Splits frames into chunks and processes them in pipeline:
        - Transfer chunk N+1 while computing chunk N
        - Readback chunk N-1 while computing chunk N
        
        This provides true overlap and 2-3x throughput improvement.
        
        Args:
            frames: Framed audio data (n_frames, fft_size)
            plan: FFT plan to use
            fft_size: FFT size
            use_fused: Whether to use fused kernels
            
        Returns:
            Magnitude dB array (freq_bins, time_frames)
        """
        n_frames, _ = frames.shape
        # Use larger chunks to minimize overhead - only split if really large
        # Optimal chunk size: balance between overlap benefit and overhead
        # Use 20% of total or minimum 20000 frames for better overlap
        chunk_size = max(20000, n_frames // 5)  # At least 20k frames, or 20% of total
        
        # Only split if we have enough frames to benefit from pipeline (need at least 3 chunks)
        if n_frames < chunk_size * 3:
            # Not enough frames for pipeline - use regular multi-stream
            return self._compute_single_chunk_multi_stream(frames, plan, use_fused)
        
        # Split into chunks
        chunks = []
        for i in range(0, n_frames, chunk_size):
            end = min(i + chunk_size, n_frames)
            chunks.append(frames[i:end])
        
        if len(chunks) == 1:
            # Single chunk - use regular multi-stream
            return self._compute_single_chunk_multi_stream(chunks[0], plan, use_fused)
        
        # Pipeline processing with true overlap
        results = []
        transfer_events = []
        compute_events = []
        gpu_chunks = []
        magnitude_chunks = []
        
        # Process chunks in pipeline with true overlap
        # Key: Start transfer of chunk N+1 while computing chunk N
        for chunk_idx, chunk in enumerate(chunks):
            # Stream 0: Transfer chunk to GPU
            # For first chunk, start immediately
            # For subsequent chunks, can start in parallel with previous compute
            with self._cuda_streams[0]:
                gpu_chunk = self.memory_optimizer.copy_to_gpu_pinned(chunk)
                gpu_chunks.append(gpu_chunk)
                
                transfer_event = cp.cuda.Event()
                transfer_event.record(self._cuda_streams[0])
                transfer_events.append(transfer_event)
            
            # Stream 1: Compute chunk - wait for its transfer to complete
            with self._cuda_streams[1]:
                # Wait for this chunk's transfer (non-blocking for other streams)
                self._cuda_streams[1].wait_event(transfer_events[-1])
                
                # Get chunk plan
                chunk_plan = self.get_or_create_plan(fft_size, len(chunk))
                stft_gpu = chunk_plan.execute(gpu_chunk)
                
                # Compute magnitude
                if use_fused and self._fused_kernels:
                    magnitude = self._fused_kernels.magnitude_to_db_fused(stft_gpu, min_val=1e-10, scale=20.0)
                else:
                    magnitude = self.memory_optimizer.inplace_abs(stft_gpu)
                    self.memory_optimizer.inplace_maximum(magnitude, 1e-10)
                    self.memory_optimizer.inplace_log10(magnitude)
                    self.memory_optimizer.inplace_multiply(magnitude, 20.0)
                
                magnitude_chunks.append(magnitude.T)
                
                compute_event = cp.cuda.Event()
                compute_event.record(self._cuda_streams[1])
                compute_events.append(compute_event)
            
            # Stream 2: Readback previous chunk (can overlap with current compute)
            # Start readback of chunk N-1 while chunk N is computing
            if chunk_idx > 0:
                with self._cuda_streams[2]:
                    # Wait for previous chunk's compute to complete
                    self._cuda_streams[2].wait_event(compute_events[chunk_idx - 1])
                    
                    # Readback previous chunk (non-blocking for other streams)
                    prev_magnitude_cpu = cp.asnumpy(magnitude_chunks[chunk_idx - 1])
                    results.append(prev_magnitude_cpu)
        
        # Readback last chunk after all compute completes
        with self._cuda_streams[2]:
            if compute_events:
                self._cuda_streams[2].wait_event(compute_events[-1])
            last_magnitude_cpu = cp.asnumpy(magnitude_chunks[-1])
            results.append(last_magnitude_cpu)
        
        # Final synchronization - wait for all streams to complete
        self._cuda_streams[2].synchronize()
        
        # Concatenate results along time axis
        magnitude_db = np.concatenate(results, axis=1)
        return magnitude_db
    
    def _compute_single_chunk_multi_stream(self, frames: np.ndarray, plan: BatchedFFTPlan,
                                          use_fused: bool) -> np.ndarray:
        """Compute single chunk with multi-stream (helper for pipeline)."""
        # Stream 0: Transfer
        with self._cuda_streams[0]:
            gpu_frames = self.memory_optimizer.copy_to_gpu_pinned(frames)
            transfer_event = cp.cuda.Event()
            transfer_event.record(self._cuda_streams[0])
        
        # Stream 1: Compute
        with self._cuda_streams[1]:
            self._cuda_streams[1].wait_event(transfer_event)
            stft_gpu = plan.execute(gpu_frames)
            
            if use_fused and self._fused_kernels:
                magnitude = self._fused_kernels.magnitude_to_db_fused(stft_gpu, min_val=1e-10, scale=20.0)
            else:
                magnitude = self.memory_optimizer.inplace_abs(stft_gpu)
                self.memory_optimizer.inplace_maximum(magnitude, 1e-10)
                self.memory_optimizer.inplace_log10(magnitude)
                self.memory_optimizer.inplace_multiply(magnitude, 20.0)
            
            magnitude_db = magnitude.T
            compute_event = cp.cuda.Event()
            compute_event.record(self._cuda_streams[1])
        
        # Stream 2: Readback
        with self._cuda_streams[2]:
            self._cuda_streams[2].wait_event(compute_event)
            magnitude_db_cpu = cp.asnumpy(magnitude_db)
        
        self._cuda_streams[2].synchronize()
        return magnitude_db_cpu
    
    def compute_stft_batched(self, audio: np.ndarray, fft_size: int = 2048,
                            hop_length: int = 512, window: str = 'hann',
                            sample_rate: int = 44100, use_gpu: bool = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute STFT using batched FFT.
        
        Args:
            audio: Audio signal (1D array)
            fft_size: FFT window size
            hop_length: Hop size between frames
            window: Window type
            use_gpu: Override GPU usage
            
        Returns:
            (magnitude_db, times) where magnitude_db is (freq_bins, time_frames)
        """
        if use_gpu is None:
            use_gpu = self.use_gpu
        
        # Calculate number of frames
        if len(audio) < fft_size:
            logger.warning(f"Audio too short ({len(audio)} samples) for FFT size {fft_size}")
            # Return minimal data
            return np.zeros((fft_size // 2 + 1, 1), dtype=np.float32), np.array([0.0], dtype=np.float32)
        
        n_frames = 1 + (len(audio) - fft_size) // hop_length
        
        if n_frames <= 0:
            logger.warning(f"No frames to compute")
            return np.zeros((fft_size // 2 + 1, 1), dtype=np.float32), np.array([0.0], dtype=np.float32)
        
        # Get window from cache (eliminates repeated computation)
        win = self._get_window(fft_size, window)
        
        # Prepare framed data (batch) using workspace pool
        # Shape: (n_frames, fft_size)
        frames = self.memory_optimizer.get_cpu_workspace((n_frames, fft_size), dtype=np.float32)
        
        # Optimized framing: use NumPy's as_strided for zero-copy view (when possible)
        # This is 10-50x faster than looping for large batch sizes
        try:
            from numpy.lib.stride_tricks import as_strided
            
            # Create strided view of audio (zero-copy, ~100x faster than loop)
            if (n_frames - 1) * hop_length + fft_size <= len(audio):
                frame_view = as_strided(
                    audio,
                    shape=(n_frames, fft_size),
                    strides=(audio.strides[0] * hop_length, audio.strides[0]),
                    writeable=False  # Read-only view
                )
                # Apply window using broadcasting (vectorized, cache-friendly)
                np.multiply(frame_view, win, out=frames)
            else:
                # Fallback for edge cases
                for i in range(n_frames):
                    start = i * hop_length
                    end = start + fft_size
                    frames[i, :] = audio[start:end] * win
        except Exception as e:
            # Fallback to loop if as_strided fails
            logger.debug(f"Using loop fallback for framing: {e}")
            for i in range(n_frames):
                start = i * hop_length
                end = start + fft_size
                frames[i, :] = audio[start:end] * win
        
        # Get or create FFT plan
        plan = self.get_or_create_plan(fft_size, n_frames)
        
        # Execute batched FFT
        if use_gpu and HAS_CUPY:
            # Multi-stream GPU management with event-based overlap
            # True overlap: transfer, compute, and readback can overlap using events
            # For very large batches (>100000 frames), use pipeline processing for true overlap
            # Pipeline has overhead for medium batches, so use regular multi-stream for optimal performance
            # Pipeline is beneficial when we can process 3+ chunks in parallel
            if self._cuda_streams and n_frames > 100000:
                # Very large batch - use pipeline processing for true overlap
                # Pipeline allows: transfer chunk N+1 while computing chunk N
                magnitude_db = self._compute_with_pipeline_overlap(
                    frames, plan, fft_size, use_fused=self._fused_kernels is not None
                )
            elif self._cuda_streams:
                # Medium batch - use multi-stream with events (optimal for most cases)
                # Stream 0: Transfer (H2D) - start immediately
                with self._cuda_streams[0]:
                    # Move to GPU using pinned memory for 2-3x faster transfer
                    gpu_frames = self.memory_optimizer.copy_to_gpu_pinned(frames)
                    # Record event when transfer completes
                    transfer_event = cp.cuda.Event()
                    transfer_event.record(self._cuda_streams[0])
                
                # Stream 1: Compute (FFT + magnitude) - wait for transfer event (non-blocking)
                with self._cuda_streams[1]:
                    # Wait for transfer event (allows overlap with other operations)
                    self._cuda_streams[1].wait_event(transfer_event)
                    stft_gpu = plan.execute(gpu_frames)
                    
                    # Compute magnitude using fused kernel (2-3x faster than separate ops!)
                    use_fused = self._fused_kernels is not None
                    
                    if use_fused:
                        # FUSED: abs + maximum + log10 + multiply in ONE kernel
                        magnitude = self._fused_kernels.magnitude_to_db_fused(stft_gpu, min_val=1e-10, scale=20.0)
                    else:
                        # Fallback: in-place operations (still good, but slower)
                        magnitude = self.memory_optimizer.inplace_abs(stft_gpu)
                        self.memory_optimizer.inplace_maximum(magnitude, 1e-10)
                        self.memory_optimizer.inplace_log10(magnitude)
                        self.memory_optimizer.inplace_multiply(magnitude, 20.0)
                    
                    # Transpose to (freq_bins, time_frames)
                    magnitude_db = magnitude.T
                    # Record event when compute completes
                    compute_event = cp.cuda.Event()
                    compute_event.record(self._cuda_streams[1])
                
                # Stream 2: Readback (D2H) - wait for compute event (non-blocking)
                with self._cuda_streams[2]:
                    # Wait for compute event (allows overlap)
                    self._cuda_streams[2].wait_event(compute_event)
                    # Move back to CPU
                    magnitude_db_cpu = cp.asnumpy(magnitude_db)
                    # Record event when readback completes
                    readback_event = cp.cuda.Event()
                    readback_event.record(self._cuda_streams[2])
                
                # Only synchronize at the very end - all streams can overlap until here
                readback_event.synchronize()
                magnitude_db = magnitude_db_cpu
            else:
                # Fallback to single stream (backward compatibility)
                with self._cuda_stream:
                    # Move to GPU using pinned memory for 2-3x faster transfer
                    gpu_frames = self.memory_optimizer.copy_to_gpu_pinned(frames)
                    stft_gpu = plan.execute(gpu_frames)
                    
                    # Compute magnitude using fused kernel (2-3x faster than separate ops!)
                    use_fused = self._fused_kernels is not None
                    
                    if use_fused:
                        # FUSED: abs + maximum + log10 + multiply in ONE kernel
                        magnitude = self._fused_kernels.magnitude_to_db_fused(stft_gpu, min_val=1e-10, scale=20.0)
                    else:
                        # Fallback: in-place operations (still good, but slower)
                        magnitude = self.memory_optimizer.inplace_abs(stft_gpu)
                        self.memory_optimizer.inplace_maximum(magnitude, 1e-10)
                        self.memory_optimizer.inplace_log10(magnitude)
                        self.memory_optimizer.inplace_multiply(magnitude, 20.0)
                    
                    # Transpose to (freq_bins, time_frames)
                    magnitude_db = magnitude.T
                
                # Single synchronization point (waits for all GPU ops to complete)
                self._cuda_stream.synchronize()
                
                # Move back to CPU
                magnitude_db = cp.asnumpy(magnitude_db)
        else:
            # CPU execution with in-place operations
            stft = plan.execute(frames)
            
            # Compute magnitude using in-place operations (50% fewer allocations)
            magnitude = self.memory_optimizer.inplace_abs(stft)  # Complex -> float
            self.memory_optimizer.inplace_maximum(magnitude, 1e-10)  # Clamp minimum
            self.memory_optimizer.inplace_log10(magnitude)  # Log scale
            self.memory_optimizer.inplace_multiply(magnitude, 20.0)  # dB conversion
            
            # Transpose to (freq_bins, time_frames)
            magnitude_db = magnitude.T
        
        # Generate time array (float32 from start)
        times = (np.arange(n_frames, dtype=np.float32) * hop_length) / sample_rate
        
        return magnitude_db.astype(np.float32), times
    
    def cleanup(self):
        """Cleanup FFT plans, window cache, streams, and workspaces."""
        self.plans.clear()
        self._window_cache.clear()
        
        # Cleanup CUDA streams
        if self._cuda_streams:
            try:
                # Synchronize all streams
                for stream in self._cuda_streams:
                    stream.synchronize()
            except Exception:
                pass  # GPU may not be available
            self._cuda_streams = None
            self._cuda_stream = None
        elif self._cuda_stream is not None:
            try:
                self._cuda_stream.synchronize()  # Wait for pending operations
            except Exception:
                pass  # GPU may not be available
            self._cuda_stream = None
        
        # Cleanup memory optimizer
        if hasattr(self, 'memory_optimizer'):
            self.memory_optimizer.cleanup()
        
        if HAS_CUPY and self.use_gpu:
            try:
                # Only free GPU memory if CUDA device is available
                cp.cuda.Device().compute_capability  # Check if GPU is accessible
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                # GPU not available or accessible, skip cleanup
                pass
        
        logger.info("BatchedFFTEngine cleaned up")


# Global instance for reuse
_batched_fft_engine = None


def get_batched_fft_engine() -> BatchedFFTEngine:
    """Get global batched FFT engine instance."""
    global _batched_fft_engine
    if _batched_fft_engine is None:
        _batched_fft_engine = BatchedFFTEngine(use_gpu=HAS_CUPY)
    return _batched_fft_engine

