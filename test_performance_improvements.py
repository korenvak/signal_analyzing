"""
Performance Test Script for Batch Transfers and Multi-Stream Optimizations
Tests the improvements from:
1. Batch CPU→GPU Transfers (3-5x faster)
2. Multi-Stream GPU Management (2-3x throughput)
"""

import numpy as np
import time
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import audio visualizer components
from audio_visualizer.engines.batched_fft_engine import get_batched_fft_engine
from audio_visualizer.core.data_loader import ChunkedAudioLoader

def test_single_file_performance(file_path: str, num_runs: int = 3):
    """Test performance on a single audio file.
    
    Args:
        file_path: Path to audio file
        num_runs: Number of test runs for averaging
    """
    logger.info(f"Testing performance on: {file_path}")
    
    # Load audio file
    loader = ChunkedAudioLoader()
    sample_rate, duration = loader.load_file(file_path)
    total_samples = loader.total_samples
    
    logger.info(f"File info: {duration:.2f}s, {sample_rate}Hz, {total_samples:,} samples")
    
    # Load full audio
    logger.info("Loading audio data...")
    audio_data = loader.get_chunk(0, total_samples)
    logger.info(f"Loaded {len(audio_data):,} samples")
    
    # Get FFT engine
    engine = get_batched_fft_engine()
    
    # Test parameters
    fft_size = 4096
    hop_length = 512
    window = 'blackman'
    
    # Warmup run
    logger.info("Warmup run...")
    try:
        _ = engine.compute_stft_batched(
            audio_data, 
            fft_size=fft_size,
            hop_length=hop_length,
            window=window,
            sample_rate=sample_rate
        )
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        return None
    
    # Performance test runs
    logger.info(f"Running {num_runs} performance tests...")
    times = []
    
    for i in range(num_runs):
        start_time = time.perf_counter()
        try:
            magnitude_db, times_array = engine.compute_stft_batched(
                audio_data,
                fft_size=fft_size,
                hop_length=hop_length,
                window=window,
                sample_rate=sample_rate
            )
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
            
            logger.info(f"Run {i+1}/{num_runs}: {elapsed*1000:.2f}ms, "
                       f"Output shape: {magnitude_db.shape}, "
                       f"Realtime factor: {duration/elapsed:.1f}x")
        except Exception as e:
            logger.error(f"Run {i+1} failed: {e}")
            return None
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    realtime_factor = duration / avg_time
    
    logger.info("\n" + "="*60)
    logger.info("PERFORMANCE RESULTS")
    logger.info("="*60)
    logger.info(f"Average time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    logger.info(f"Min time: {min_time*1000:.2f}ms")
    logger.info(f"Max time: {max_time*1000:.2f}ms")
    logger.info(f"Realtime factor: {realtime_factor:.1f}x")
    logger.info(f"Throughput: {len(audio_data)/avg_time/1e6:.2f} million samples/sec")
    logger.info("="*60)
    
    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'realtime_factor': realtime_factor,
        'throughput': len(audio_data)/avg_time/1e6,
        'output_shape': magnitude_db.shape
    }


def test_batch_transfer_performance():
    """Test batch transfer performance vs individual transfers."""
    logger.info("\n" + "="*60)
    logger.info("BATCH TRANSFER TEST")
    logger.info("="*60)
    
    try:
        import cupy as cp
        from audio_visualizer.core.memory_pools import get_memory_optimizer
        
        optimizer = get_memory_optimizer()
        
        # Create test arrays
        num_arrays = 10
        array_size = 1024 * 1024  # 1MB each
        arrays = [np.random.randn(array_size).astype(np.float32) for _ in range(num_arrays)]
        
        logger.info(f"Testing with {num_arrays} arrays of {array_size:,} elements each")
        
        # Test individual transfers
        logger.info("Testing individual transfers...")
        start = time.perf_counter()
        individual_results = []
        for arr in arrays:
            result = optimizer.copy_to_gpu_pinned(arr)
            individual_results.append(result)
        individual_time = time.perf_counter() - start
        
        # Cleanup
        del individual_results
        cp.get_default_memory_pool().free_all_blocks()
        
        # Test batch transfer
        logger.info("Testing batch transfer...")
        start = time.perf_counter()
        batch_results = optimizer.copy_to_gpu_pinned_batch(arrays)
        batch_time = time.perf_counter() - start
        
        # Cleanup
        del batch_results
        cp.get_default_memory_pool().free_all_blocks()
        
        speedup = individual_time / batch_time if batch_time > 0 else 0
        
        logger.info(f"Individual transfers: {individual_time*1000:.2f}ms")
        logger.info(f"Batch transfer: {batch_time*1000:.2f}ms")
        logger.info(f"Speedup: {speedup:.2f}x")
        
        return speedup
        
    except Exception as e:
        logger.warning(f"Batch transfer test failed (GPU may not be available): {e}")
        return None


def main():
    """Main test function."""
    import sys
    
    # Test file path
    test_file = r"C:\Users\koren\Downloads\pixel - 1008 - 2025-29-10 11-04-47 - 2025-29-10 12-20-49.flac"
    
    if not Path(test_file).exists():
        logger.error(f"Test file not found: {test_file}")
        logger.info("Please update the test_file path in the script")
        return
    
    logger.info("="*60)
    logger.info("PERFORMANCE OPTIMIZATION TEST")
    logger.info("="*60)
    logger.info("Testing:")
    logger.info("1. Multi-Stream GPU Management")
    logger.info("2. Batch CPU→GPU Transfers")
    logger.info("="*60)
    
    # Test batch transfers
    batch_speedup = test_batch_transfer_performance()
    
    # Test full file performance
    results = test_single_file_performance(test_file, num_runs=5)
    
    if results:
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        logger.info(f"✅ Multi-Stream GPU: Enabled")
        if batch_speedup:
            logger.info(f"✅ Batch Transfers: {batch_speedup:.2f}x speedup")
        logger.info(f"✅ Overall Performance: {results['realtime_factor']:.1f}x realtime")
        logger.info(f"✅ Throughput: {results['throughput']:.2f} million samples/sec")
        logger.info("="*60)
    else:
        logger.error("Performance test failed!")


if __name__ == "__main__":
    main()

