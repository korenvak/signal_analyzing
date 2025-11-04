"""
Performance profiling script to identify bottlenecks and optimization opportunities.
"""

import numpy as np
import time
import logging
from typing import Dict
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def profile_fft_engine():
    """Profile the batched FFT engine to find bottlenecks."""
    logger.info("\n" + "="*60)
    logger.info("Profiling BatchedFFTEngine")
    logger.info("="*60)
    
    from audio_visualizer.engines.batched_fft_engine import get_batched_fft_engine
    
    engine = get_batched_fft_engine()
    
    # Test different file sizes
    test_cases = [
        ("1 second", 44100),
        ("10 seconds", 441000),
        ("60 seconds", 2646000),
        ("5 minutes", 13230000),
    ]
    
    results = []
    
    for name, samples in test_cases:
        logger.info(f"\nTesting: {name} ({samples} samples)")
        
        # Generate test audio
        audio = np.random.randn(samples).astype(np.float32)
        
        # Warm-up run
        engine.compute_stft_batched(audio, fft_size=2048, hop_length=512, use_gpu=True)
        
        # Timed run
        times = []
        for i in range(3):
            start = time.perf_counter()
            magnitude_db, freq_times = engine.compute_stft_batched(
                audio, fft_size=2048, hop_length=512, use_gpu=True)
            end = time.perf_counter()
            elapsed = (end - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        # Calculate throughput
        frames = magnitude_db.shape[1]
        samples_per_sec = samples / (avg_time / 1000)
        realtime_factor = samples_per_sec / 44100
        
        logger.info(f"  Time: {avg_time:.1f}ms ± {std_time:.1f}ms")
        logger.info(f"  Frames: {frames}")
        logger.info(f"  Throughput: {samples_per_sec/1e6:.2f} million samples/sec")
        logger.info(f"  Realtime factor: {realtime_factor:.1f}x")
        
        results.append({
            'name': name,
            'samples': samples,
            'time_ms': avg_time,
            'frames': frames,
            'throughput': samples_per_sec,
            'realtime_factor': realtime_factor
        })
    
    engine.cleanup()
    
    return results


def profile_memory_operations():
    """Profile memory operations to find inefficiencies."""
    logger.info("\n" + "="*60)
    logger.info("Profiling Memory Operations")
    logger.info("="*60)
    
    from audio_visualizer.core.memory_pools import get_memory_optimizer
    
    optimizer = get_memory_optimizer()
    
    # Test different operation types
    size = (2000, 2048)
    
    # Test 1: Workspace allocation
    logger.info("\nTest 1: Workspace Allocation")
    times = []
    for _ in range(100):
        start = time.perf_counter()
        workspace = optimizer.get_cpu_workspace(size, dtype=np.float32)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    logger.info(f"  Average: {np.mean(times):.4f}ms (should be ~0ms with pools)")
    logger.info(f"  Std dev: {np.std(times):.4f}ms")
    
    # Test 2: In-place operations vs regular operations
    logger.info("\nTest 2: In-place vs Regular Operations")
    test_array = np.random.randn(*size).astype(np.float32)
    
    # Regular operations
    start = time.perf_counter()
    for _ in range(10):
        temp = test_array.copy()
        result = 20 * np.log10(np.maximum(np.abs(temp), 1e-10))
    end = time.perf_counter()
    regular_time = (end - start) * 100  # ms
    
    # In-place operations
    start = time.perf_counter()
    for _ in range(10):
        temp = test_array.copy()
        optimizer.inplace_maximum(temp, 1e-10)
        optimizer.inplace_log10(temp)
        optimizer.inplace_multiply(temp, 20.0)
    end = time.perf_counter()
    inplace_time = (end - start) * 100  # ms
    
    logger.info(f"  Regular ops: {regular_time:.2f}ms")
    logger.info(f"  In-place ops: {inplace_time:.2f}ms")
    logger.info(f"  Speedup: {regular_time/inplace_time:.2f}x")
    
    if HAS_CUPY:
        # Test 3: GPU transfer methods
        logger.info("\nTest 3: GPU Transfer Methods")
        test_data = np.random.randn(10000, 2048).astype(np.float32)
        
        # Regular transfer
        times = []
        for _ in range(10):
            start = time.perf_counter()
            gpu_data = cp.asarray(test_data)
            cp.cuda.Device().synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        regular_transfer = np.mean(times)
        
        # Pinned memory transfer
        times = []
        for _ in range(10):
            start = time.perf_counter()
            gpu_data = optimizer.copy_to_gpu_pinned(test_data)
            cp.cuda.Device().synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        pinned_transfer = np.mean(times)
        
        data_size_mb = test_data.nbytes / (1024 * 1024)
        
        logger.info(f"  Data size: {data_size_mb:.1f} MB")
        logger.info(f"  Regular transfer: {regular_transfer:.2f}ms ({data_size_mb/regular_transfer*1000:.1f} GB/s)")
        logger.info(f"  Pinned transfer: {pinned_transfer:.2f}ms ({data_size_mb/pinned_transfer*1000:.1f} GB/s)")
        logger.info(f"  Speedup: {regular_transfer/pinned_transfer:.2f}x")


def profile_cepstrogram_engine():
    """Profile cepstrogram computation."""
    logger.info("\n" + "="*60)
    logger.info("Profiling CepstrogramEngine")
    logger.info("="*60)
    
    from audio_visualizer.core.cache_manager import CacheManager
    from audio_visualizer.core.task_manager import TaskManager
    from audio_visualizer.engines.spectrogram_engine import SpectrogramEngine
    from audio_visualizer.engines.cepstrogram_engine import CepstrogramEngine
    
    cache_manager = CacheManager(max_memory_mb=512)
    task_manager = TaskManager()
    spec_engine = SpectrogramEngine(cache_manager, task_manager)
    ceps_engine = CepstrogramEngine(cache_manager, task_manager, spec_engine)
    
    # Generate test spectrogram
    logger.info("\nGenerating test spectrogram...")
    audio = np.random.randn(441000).astype(np.float32)  # 10 seconds
    magnitude_db, frequencies, times = spec_engine.compute_stft_batched(audio)
    
    logger.info(f"Spectrogram shape: {magnitude_db.shape}")
    
    # Profile cepstrogram computation
    logger.info("\nComputing cepstrogram...")
    times_list = []
    for i in range(3):
        start = time.perf_counter()
        cepstral = ceps_engine.compute_cepstrogram_from_spectrogram(magnitude_db, frequencies)
        end = time.perf_counter()
        elapsed = (end - start) * 1000
        times_list.append(elapsed)
    
    avg_time = np.mean(times_list)
    logger.info(f"  Time: {avg_time:.1f}ms ± {np.std(times_list):.1f}ms")
    logger.info(f"  Output shape: {cepstral.shape}")
    logger.info(f"  Throughput: {magnitude_db.size / (avg_time/1000) / 1e6:.1f} million elements/sec")


def identify_bottlenecks():
    """Analyze results and identify optimization opportunities."""
    logger.info("\n" + "="*60)
    logger.info("BOTTLENECK ANALYSIS & OPTIMIZATION OPPORTUNITIES")
    logger.info("="*60)
    
    opportunities = []
    
    # Opportunity 1: Batch transfers
    opportunities.append({
        'name': 'Batch CPU→GPU Transfers',
        'current': 'Each chunk transferred separately',
        'improvement': 'Transfer multiple chunks as single block',
        'expected_gain': '3-5x faster for multi-chunk',
        'effort': '2-3 hours',
        'priority': 'HIGH'
    })
    
    # Opportunity 2: Kernel fusion
    opportunities.append({
        'name': 'GPU Kernel Fusion',
        'current': 'Separate kernels for abs/max/log/mul',
        'improvement': 'Fuse into single custom CUDA kernel',
        'expected_gain': '2-3x faster magnitude computation',
        'effort': '4-6 hours',
        'priority': 'HIGH'
    })
    
    # Opportunity 3: Async CPU-GPU overlap
    opportunities.append({
        'name': 'CPU-GPU Overlap',
        'current': 'Sequential: compute frame → transfer → compute',
        'improvement': 'Overlap: transfer next while computing current',
        'expected_gain': '30-50% throughput boost',
        'effort': '3-4 hours',
        'priority': 'MEDIUM'
    })
    
    # Opportunity 4: FFT plan caching
    opportunities.append({
        'name': 'Enhanced FFT Plan Caching',
        'current': 'Plans cached by (fft_size, batch_size)',
        'improvement': 'Pre-generate common plans at startup',
        'expected_gain': '10-20% faster first computation',
        'effort': '1 hour',
        'priority': 'LOW'
    })
    
    # Opportunity 5: Memory-mapped I/O for large files
    opportunities.append({
        'name': 'Memory-Mapped Audio Loading',
        'current': 'Load entire audio file into RAM',
        'improvement': 'Memory-map large files, load chunks on-demand',
        'expected_gain': '5-10x faster file loading for large files',
        'effort': '2-3 hours',
        'priority': 'HIGH'
    })
    
    logger.info("\n🎯 Top Optimization Opportunities:\n")
    
    for i, opp in enumerate(opportunities, 1):
        priority_emoji = {'HIGH': '🔥', 'MEDIUM': '🟡', 'LOW': '🟢'}[opp['priority']]
        logger.info(f"{i}. {priority_emoji} {opp['name']} [{opp['priority']}]")
        logger.info(f"   Current: {opp['current']}")
        logger.info(f"   Improvement: {opp['improvement']}")
        logger.info(f"   Expected gain: {opp['expected_gain']}")
        logger.info(f"   Effort: {opp['effort']}")
        logger.info("")
    
    return opportunities


def main():
    """Run performance profiling suite."""
    logger.info("="*60)
    logger.info("PERFORMANCE PROFILING SUITE")
    logger.info("="*60)
    
    if not HAS_CUPY:
        logger.warning("⚠️  CuPy not available - some tests will be skipped")
    else:
        gpu_name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode()
        logger.info(f"✓ GPU available: {gpu_name}")
    
    try:
        # Profile components
        fft_results = profile_fft_engine()
        profile_memory_operations()
        profile_cepstrogram_engine()
        
        # Identify bottlenecks
        opportunities = identify_bottlenecks()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("SUMMARY")
        logger.info("="*60)
        
        logger.info("\n📊 Current Performance:")
        for result in fft_results:
            logger.info(f"  {result['name']}: {result['time_ms']:.1f}ms ({result['realtime_factor']:.1f}x realtime)")
        
        logger.info("\n🚀 Recommended Focus Areas:")
        high_priority = [o for o in opportunities if o['priority'] == 'HIGH']
        for i, opp in enumerate(high_priority, 1):
            logger.info(f"  {i}. {opp['name']} - {opp['expected_gain']}")
        
        logger.info("\n✅ Profiling complete!")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

