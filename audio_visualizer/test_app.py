#!/usr/bin/env python3
"""
Test script for the GPU-Accelerated Audio Visualization Application.
Validates core functionality, performance, and error handling.
"""

import sys
import os
import time
import numpy as np
import tempfile
import unittest
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class TestAudioLoader(unittest.TestCase):
    """Test the chunked audio loader."""
    
    def setUp(self):
        from core.data_loader import ChunkedAudioLoader
        self.loader = ChunkedAudioLoader()
        
        # Create temporary test audio
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        self.create_test_audio(self.temp_file.name)
    
    def tearDown(self):
        os.unlink(self.temp_file.name)
    
    def create_test_audio(self, filename, duration=2.0, sample_rate=44100):
        """Create test audio file."""
        import soundfile as sf
        
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        # Multiple frequency components
        audio = (np.sin(2 * np.pi * 440 * t) + 
                0.5 * np.sin(2 * np.pi * 880 * t) + 
                0.1 * np.random.randn(len(t)))
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        sf.write(filename, audio, sample_rate)
    
    def test_file_loading(self):
        """Test basic file loading."""
        sample_rate, duration = self.loader.load_file(self.temp_file.name)
        
        self.assertGreater(sample_rate, 0)
        self.assertGreater(duration, 0)
        self.assertAlmostEqual(duration, 2.0, places=1)
    
    def test_chunk_reading(self):
        """Test chunk reading functionality."""
        self.loader.load_file(self.temp_file.name)
        
        # Read first chunk
        chunk = self.loader.get_chunk(0, 1024)
        self.assertEqual(len(chunk), 1024)
        self.assertTrue(np.all(np.isfinite(chunk)))

class TestCacheManager(unittest.TestCase):
    """Test the cache manager."""
    
    def setUp(self):
        from core.cache_manager import CacheManager
        self.cache = CacheManager(max_memory_mb=100, max_gpu_memory_mb=50)
    
    def test_tile_storage_retrieval(self):
        """Test storing and retrieving tiles."""
        # Create test data
        test_data = np.random.randn(256, 256).astype(np.float32)
        
        # Store tile
        success = self.cache.store_tile('test', (0.0, 1.0), (0.0, 1000.0), test_data)
        self.assertTrue(success)
        
        # Retrieve tile
        retrieved = self.cache.get_tile('test', (0.0, 1.0), (0.0, 1000.0))
        self.assertIsNotNone(retrieved)
        np.testing.assert_array_equal(test_data, retrieved)
    
    def test_memory_limits(self):
        """Test memory limit enforcement."""
        # Fill cache beyond limit
        for i in range(20):
            large_data = np.random.randn(512, 512).astype(np.float32)
            self.cache.store_tile('test', (i, i+1), (0.0, 1000.0), large_data)
        
        stats = self.cache.get_stats()
        self.assertLessEqual(stats['cpu_memory_mb'], 150)  # Allow some overhead

class TestSpectrogramEngine(unittest.TestCase):
    """Test the spectrogram engine."""
    
    def setUp(self):
        from core.cache_manager import CacheManager
        from core.task_manager import TaskManager
        from engines.spectrogram_engine import SpectrogramEngine
        
        self.cache = CacheManager(max_memory_mb=100, max_gpu_memory_mb=50)
        self.task_manager = TaskManager()
        self.engine = SpectrogramEngine(self.cache, self.task_manager)
        
        # Create test audio
        self.sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(duration * self.sample_rate), False)
        self.test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    def tearDown(self):
        self.task_manager.shutdown(wait=True)
    
    def test_stft_computation(self):
        """Test STFT computation."""
        self.engine.set_parameters(sample_rate=self.sample_rate)
        
        # Compute spectrogram tile
        result = self.engine.compute_tile(
            self.test_audio, 0.0, 1.0, 0.0, self.sample_rate/2
        )
        
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.ndim, 2)
        self.assertGreater(result.shape[0], 0)  # Frequency bins
        self.assertGreater(result.shape[1], 0)  # Time frames
    
    def test_parameter_updates(self):
        """Test parameter updates clear cache."""
        initial_fft_size = self.engine.fft_size
        
        # Change parameters
        self.engine.set_parameters(fft_size=4096)
        self.assertEqual(self.engine.fft_size, 4096)
        self.assertNotEqual(self.engine.fft_size, initial_fft_size)

class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""
    
    def setUp(self):
        from core.cache_manager import CacheManager
        from core.task_manager import TaskManager
        from engines.spectrogram_engine import SpectrogramEngine
        
        self.cache = CacheManager(max_memory_mb=200, max_gpu_memory_mb=100)
        self.task_manager = TaskManager()
        self.engine = SpectrogramEngine(self.cache, self.task_manager)
        
        # Create larger test audio (10 seconds)
        self.sample_rate = 44100
        duration = 10.0
        t = np.linspace(0, duration, int(duration * self.sample_rate), False)
        self.test_audio = (np.sin(2 * np.pi * 440 * t) + 
                          0.5 * np.sin(2 * np.pi * 880 * t) + 
                          0.1 * np.random.randn(len(t))).astype(np.float32)
    
    def tearDown(self):
        self.task_manager.shutdown(wait=True)
    
    def test_large_file_processing(self):
        """Test processing of large audio files."""
        self.engine.set_parameters(sample_rate=self.sample_rate)
        
        start_time = time.time()
        
        # Process large chunk
        result = self.engine.compute_tile(
            self.test_audio, 0.0, 10.0, 0.0, self.sample_rate/2
        )
        
        processing_time = time.time() - start_time
        
        self.assertIsInstance(result, np.ndarray)
        self.assertLess(processing_time, 5.0)  # Should complete within 5 seconds
    
    def test_memory_usage(self):
        """Test memory usage stays within bounds."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process multiple chunks
        for i in range(5):
            chunk_start = i * 2.0
            chunk_end = min((i + 1) * 2.0, 10.0)
            
            self.engine.compute_tile(
                self.test_audio, chunk_start, chunk_end, 0.0, self.sample_rate/2
            )
        
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (< 500MB)
        self.assertLess(memory_increase, 500)

class TestGPUAcceleration(unittest.TestCase):
    """Test GPU acceleration if available."""
    
    def setUp(self):
        try:
            import cupy as cp
            self.gpu_available = True
            self.gpu_count = cp.cuda.runtime.getDeviceCount()
        except ImportError:
            self.gpu_available = False
            self.gpu_count = 0
    
    @unittest.skipIf(not gpu_available, "GPU not available")
    def test_gpu_memory_management(self):
        """Test GPU memory management."""
        import cupy as cp
        
        # Check initial GPU memory
        initial_memory = cp.get_default_memory_pool().used_bytes()
        
        # Allocate and free GPU arrays
        for _ in range(10):
            data = cp.random.randn(1024, 1024, dtype=cp.float32)
            del data
        
        cp.get_default_memory_pool().free_all_blocks()
        final_memory = cp.get_default_memory_pool().used_bytes()
        
        # Memory should be freed
        self.assertEqual(final_memory, initial_memory)

def run_performance_benchmark():
    """Run performance benchmarks."""
    print("\n=== Performance Benchmark ===")
    
    try:
        from core.cache_manager import CacheManager
        from core.task_manager import TaskManager
        from engines.spectrogram_engine import SpectrogramEngine
        
        cache = CacheManager(max_memory_mb=500, max_gpu_memory_mb=250)
        task_manager = TaskManager()
        engine = SpectrogramEngine(cache, task_manager)
        
        # Create test audio (30 seconds)
        sample_rate = 44100
        duration = 30.0
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        
        engine.set_parameters(sample_rate=sample_rate)
        
        # Benchmark different FFT sizes
        fft_sizes = [1024, 2048, 4096, 8192]
        
        for fft_size in fft_sizes:
            engine.set_parameters(fft_size=fft_size)
            
            start_time = time.time()
            result = engine.compute_tile(test_audio, 0.0, 30.0, 0.0, sample_rate/2)
            processing_time = time.time() - start_time
            
            data_size = result.nbytes / 1024 / 1024  # MB
            throughput = data_size / processing_time
            
            print(f"FFT {fft_size}: {processing_time:.2f}s, {data_size:.1f}MB, {throughput:.1f}MB/s")
        
        task_manager.shutdown(wait=True)
        
    except Exception as e:
        print(f"Benchmark failed: {e}")

def check_dependencies():
    """Check all dependencies."""
    print("=== Dependency Check ===")
    
    required = ['numpy', 'scipy', 'librosa', 'soundfile', 'PySide6']
    optional = ['cupy', 'vispy', 'zarr', 'psutil']
    
    missing_required = []
    missing_optional = []
    
    for dep in required:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} (REQUIRED)")
            missing_required.append(dep)
    
    for dep in optional:
        try:
            __import__(dep)
            print(f"✓ {dep}")
        except ImportError:
            print(f"- {dep} (optional)")
            missing_optional.append(dep)
    
    if missing_required:
        print(f"\nMissing required dependencies: {', '.join(missing_required)}")
        return False
    
    if missing_optional:
        print(f"\nMissing optional dependencies: {', '.join(missing_optional)}")
    
    return True

def main():
    """Run all tests."""
    print("GPU-Accelerated Audio Visualizer - Test Suite")
    print("=" * 50)
    
    # Check dependencies first
    if not check_dependencies():
        print("Please install missing dependencies before running tests.")
        return 1
    
    # Run unit tests
    print("\n=== Running Unit Tests ===")
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAudioLoader,
        TestCacheManager, 
        TestSpectrogramEngine,
        TestPerformance
    ]
    
    # Add GPU tests if available
    try:
        import cupy
        test_classes.append(TestGPUAcceleration)
    except ImportError:
        print("Skipping GPU tests (CuPy not available)")
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Run performance benchmark
    run_performance_benchmark()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASS' if success else 'FAIL'}")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())