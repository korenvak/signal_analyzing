"""
Test script for DAS multi-channel visualization system.

Tests:
1. Synthetic data generation
2. File-based data loading with memory management
3. Waterfall processing with GPU/CPU fallback
4. Large dataset handling

Run with: python test_das_system.py
"""

import sys
import os
import time
import logging
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from audio_visualizer.core.synthetic_das_generator import (
    SyntheticDASGenerator, generate_test_dataset
)
from audio_visualizer.core.file_das_provider import FileDASProvider, load_das_folder
from audio_visualizer.core.das_data_provider import DASDataRequest
from audio_visualizer.engines.waterfall_engine import WaterfallEngine, WaterfallParams


def test_synthetic_generation():
    """Test synthetic data generation."""
    print("\n" + "="*60)
    print("TEST 1: Synthetic Data Generation")
    print("="*60)

    test_dir = Path("test_das_synthetic")

    # Clean up from previous runs
    if test_dir.exists():
        shutil.rmtree(test_dir)

    # Generate small dataset first
    print("\n1.1 Generating SMALL dataset (256 sensors, 10s)...")
    start = time.time()
    path = generate_test_dataset(
        output_dir=str(test_dir / "small"),
        size="tiny",
        pattern="realistic",
        seed=42
    )
    elapsed = time.time() - start
    print(f"    Generated in {elapsed:.2f}s at: {path}")

    # Check files exist
    metadata_file = path / "metadata.json"
    assert metadata_file.exists(), "Metadata file should exist"
    print(f"    Metadata file: {metadata_file.stat().st_size / 1024:.1f} KB")

    # Count data files
    data_files = list(path.glob("segment_*.npy"))
    print(f"    Data files: {len(data_files)}")
    for f in data_files:
        print(f"      - {f.name}: {f.stat().st_size / 1024 / 1024:.2f} MB")

    # Generate medium dataset
    print("\n1.2 Generating MEDIUM dataset (2048 sensors, 60s x 2 segments)...")
    start = time.time()
    path2 = generate_test_dataset(
        output_dir=str(test_dir / "medium"),
        size="medium",
        pattern="mixed",
        seed=123
    )
    elapsed = time.time() - start
    print(f"    Generated in {elapsed:.2f}s at: {path2}")

    data_files2 = list(path2.glob("segment_*.npy"))
    total_size = sum(f.stat().st_size for f in data_files2)
    print(f"    Total size: {total_size / 1024 / 1024:.1f} MB across {len(data_files2)} files")

    print("\n[PASS] Synthetic data generation working")
    return test_dir


def test_file_loading(test_dir: Path):
    """Test file-based data loading."""
    print("\n" + "="*60)
    print("TEST 2: File-Based Data Loading")
    print("="*60)

    small_path = test_dir / "small"

    # Load the small dataset
    print("\n2.1 Loading small dataset...")
    provider = FileDASProvider()
    metadata = provider.load_folder(str(small_path), "metadata.json")

    print(f"    Sensors: {metadata.n_sensors}")
    print(f"    Sample rate: {metadata.sample_rate} Hz")
    print(f"    Duration: {metadata.total_duration_seconds:.1f}s")
    print(f"    Files: {len(metadata.files)}")

    # Test data chunk loading
    print("\n2.2 Loading data chunk...")
    if metadata.time_ranges:
        time_start = metadata.time_ranges[0][0]
        time_end = metadata.time_ranges[0][1]

        # Request a small chunk
        request = DASDataRequest(
            sensor_start=0,
            sensor_end=min(100, metadata.n_sensors),
            time_start=time_start,
            time_end=time_start + (time_end - time_start) / 2
        )

        start = time.time()
        data = provider.get_data_chunk(request)
        elapsed = time.time() - start

        print(f"    Chunk shape: {data.shape}")
        print(f"    Data range: [{data.min():.4f}, {data.max():.4f}]")
        print(f"    Load time: {elapsed*1000:.1f}ms")

    # Test memory stats
    print("\n2.3 Memory usage...")
    stats = provider.get_memory_usage()
    print(f"    Cache items: {stats['cache_items']}")
    print(f"    Cache size: {stats['cache_mb']:.2f} MB")
    print(f"    Mmap files: {stats['mmap_files']}")

    provider.close()
    print("\n[PASS] File loading working")


def test_waterfall_processing(test_dir: Path):
    """Test waterfall engine processing."""
    print("\n" + "="*60)
    print("TEST 3: Waterfall Processing")
    print("="*60)

    # Create waterfall engine
    print("\n3.1 Creating waterfall engine...")
    engine = WaterfallEngine(use_gpu=True)
    print(f"    GPU available: {engine.use_gpu}")

    # Load data
    small_path = test_dir / "small"
    provider = FileDASProvider()
    metadata = provider.load_folder(str(small_path), "metadata.json")

    time_start = metadata.time_ranges[0][0]
    time_end = metadata.time_ranges[0][1]

    request = DASDataRequest(
        sensor_start=0,
        sensor_end=metadata.n_sensors,
        time_start=time_start,
        time_end=time_end
    )

    data = provider.get_data_chunk(request)
    print(f"    Input data: {data.shape}, {data.nbytes / 1024 / 1024:.1f} MB")

    # Process with different normalizations
    normalizations = ["minmax", "std", "percentile"]

    for norm in normalizations:
        print(f"\n3.2 Processing with {norm} normalization...")
        params = WaterfallParams()
        from audio_visualizer.engines.waterfall_engine import NormalizationMode
        params.normalization = NormalizationMode(norm)

        def progress_cb(p, msg):
            pass  # Silent for tests

        start = time.time()
        result, stats = engine.process(data, params, progress_callback=progress_cb)
        elapsed = time.time() - start

        print(f"    Output shape: {result.shape}")
        print(f"    Output range: [{result.min():.4f}, {result.max():.4f}]")
        print(f"    Process time: {stats.get('process_time_ms', elapsed*1000):.1f}ms")
        if 'used_gpu' in stats:
            print(f"    Used GPU: {stats['used_gpu']}")

    provider.close()
    print("\n[PASS] Waterfall processing working")


def test_large_data_handling(test_dir: Path):
    """Test handling of larger datasets."""
    print("\n" + "="*60)
    print("TEST 4: Large Data Handling")
    print("="*60)

    medium_path = test_dir / "medium"

    print("\n4.1 Loading medium dataset...")
    provider = FileDASProvider()
    metadata = provider.load_folder(str(medium_path), "metadata.json")

    print(f"    Sensors: {metadata.n_sensors}")
    print(f"    Duration: {metadata.total_duration_seconds:.1f}s")
    print(f"    Segments: {len(metadata.time_ranges)}")

    # Load first segment
    if metadata.time_ranges:
        time_start = metadata.time_ranges[0][0]
        time_end = metadata.time_ranges[0][1]

        print(f"\n4.2 Loading full first segment ({metadata.n_sensors} x {(time_end-time_start).total_seconds():.0f}s)...")

        request = DASDataRequest(
            sensor_start=0,
            sensor_end=metadata.n_sensors,
            time_start=time_start,
            time_end=time_end
        )

        start = time.time()
        data = provider.get_data_chunk(request)
        load_elapsed = time.time() - start

        data_mb = data.nbytes / 1024 / 1024
        print(f"    Loaded: {data.shape} ({data_mb:.1f} MB)")
        print(f"    Load time: {load_elapsed:.2f}s ({data_mb/load_elapsed:.1f} MB/s)")

        # Process with waterfall engine
        print(f"\n4.3 Processing {data_mb:.1f} MB with waterfall engine...")
        engine = WaterfallEngine(use_gpu=True)

        start = time.time()
        result, stats = engine.process(data)
        process_elapsed = time.time() - start

        print(f"    Process time: {process_elapsed:.2f}s")
        print(f"    Throughput: {data_mb/process_elapsed:.1f} MB/s")
        if 'used_gpu' in stats:
            print(f"    Used GPU: {stats['used_gpu']}")
        if 'gpu_memory_mb' in stats:
            print(f"    GPU memory used: {stats['gpu_memory_mb']:.1f} MB")

    provider.close()
    print("\n[PASS] Large data handling working")


def test_memory_efficiency():
    """Test memory-efficient processing of very large synthetic data."""
    print("\n" + "="*60)
    print("TEST 5: Memory Efficiency Test")
    print("="*60)

    # Create large in-memory data (simulating very large file)
    n_sensors = 4096
    n_samples = 100000  # ~1.6 GB
    data_mb = n_sensors * n_samples * 4 / 1024 / 1024

    print(f"\n5.1 Creating large test data: {n_samples} x {n_sensors} ({data_mb:.0f} MB)...")

    # Generate in chunks to avoid memory issues
    chunk_size = 10000
    n_chunks = n_samples // chunk_size

    data = np.zeros((n_samples, n_sensors), dtype=np.float32)
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        data[start:end, :] = np.random.randn(chunk_size, n_sensors).astype(np.float32)
        print(f"    Generated chunk {i+1}/{n_chunks}", end='\r')

    print(f"\n    Data created: {data.shape}")

    # Test chunked processing
    print("\n5.2 Testing chunked CPU processing...")
    from audio_visualizer.engines.waterfall_engine import GPUMemoryConfig

    # Force chunked processing by setting small chunk size
    gpu_config = GPUMemoryConfig(chunk_size_mb=100.0)  # 100 MB chunks
    engine = WaterfallEngine(use_gpu=False, gpu_config=gpu_config)

    def progress_cb(p, msg):
        print(f"    [{p*100:5.1f}%] {msg}", end='\r')

    start = time.time()
    result, stats = engine.process(data, progress_callback=progress_cb)
    elapsed = time.time() - start

    print(f"\n    Process time: {elapsed:.2f}s")
    print(f"    Chunks processed: {stats.get('processed_chunks', 'N/A')}")
    print(f"    Output range: [{result.min():.4f}, {result.max():.4f}]")

    # Cleanup
    del data
    del result

    print("\n[PASS] Memory efficiency test complete")


def run_all_tests():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# DAS MULTI-CHANNEL SYSTEM TEST SUITE")
    print("#"*60)

    try:
        # Test synthetic generation
        test_dir = test_synthetic_generation()

        # Test file loading
        test_file_loading(test_dir)

        # Test waterfall processing
        test_waterfall_processing(test_dir)

        # Test large data handling
        test_large_data_handling(test_dir)

        # Optional: Memory efficiency test (can be slow)
        # Uncomment to run:
        # test_memory_efficiency()

        print("\n" + "#"*60)
        print("# ALL TESTS PASSED!")
        print("#"*60)

        # Cleanup
        print("\nCleaning up test files...")
        if test_dir.exists():
            shutil.rmtree(test_dir)
        print("Done!")

    except AssertionError as e:
        print(f"\n[FAIL] Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
