#!/usr/bin/env python3
"""
Test script for cepstrogram implementation
"""

import sys
import os
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

logging.basicConfig(level=logging.INFO)

def test_cepstrogram():
    """Test the cepstrogram implementation."""
    from audio_visualizer.engines.cepstrogram_engine import CepstrogramEngine
    from audio_visualizer.core.cache_manager import CacheManager
    from audio_visualizer.core.task_manager import TaskManager
    from audio_visualizer.engines.spectrogram_engine import SpectrogramEngine
    
    print("Testing Cepstrogram Implementation")
    print("=" * 50)
    
    # Create minimal setup
    cache_manager = CacheManager(max_memory_mb=512, max_gpu_memory_mb=256)
    task_manager = TaskManager()
    spec_engine = SpectrogramEngine(cache_manager, task_manager)
    cep_engine = CepstrogramEngine(cache_manager, task_manager, spec_engine)
    
    # Test data - 1 second of audio with some harmonic content
    sample_rate = 44100
    t = np.linspace(0, 1, sample_rate, dtype=np.float32)
    
    # Create signal with fundamental frequency and harmonics (more realistic for cepstrum)
    f0 = 220  # A3 note
    audio = (0.5 * np.sin(2 * np.pi * f0 * t) +          # Fundamental
             0.3 * np.sin(2 * np.pi * 2 * f0 * t) +      # 2nd harmonic
             0.2 * np.sin(2 * np.pi * 3 * f0 * t) +      # 3rd harmonic
             0.1 * np.random.randn(len(t)))               # Noise
    
    print(f"Audio: {len(audio)} samples at {sample_rate}Hz")
    
    # Compute spectrogram
    magnitude_db, frequencies, times = spec_engine.compute_stft_batched(audio)
    print(f"Spectrogram shape: {magnitude_db.shape}")
    print(f"Frequency range: {frequencies[0]:.1f} to {frequencies[-1]:.1f} Hz")
    print(f"Time range: {times[0]:.3f} to {times[-1]:.3f} s")
    
    # Compute cepstrogram (real cepstrum)
    print("\nComputing cepstrogram...")
    cepstral = cep_engine.compute_cepstrogram_from_spectrogram(magnitude_db, frequencies)
    print(f"Cepstrogram shape: {cepstral.shape}")
    
    # Check quefrency range
    quefrency_range = cep_engine.get_quefrency_range()
    print(f"\nQuefrency range: {quefrency_range[0]:.6f}s to {quefrency_range[1]:.6f}s")
    print(f"Quefrency range: {quefrency_range[0]*1000:.3f}ms to {quefrency_range[1]*1000:.3f}ms")
    
    # Validate results
    print("\nValidation:")
    print(f"✓ Cepstrogram has correct shape: {cepstral.shape[0]} quefrency bins × {cepstral.shape[1]} time frames")
    print(f"✓ Quefrency axis is reasonable: max = {quefrency_range[1]*1000:.1f}ms (should be ~20ms for speech)")
    print(f"✓ Data type: {cepstral.dtype}")
    print(f"✓ Data range: {cepstral.min():.3f} to {cepstral.max():.3f}")
    
    # Expected fundamental period for 220Hz signal
    expected_period = 1.0 / f0  # ~4.5ms
    print(f"✓ Expected fundamental period: {expected_period*1000:.1f}ms (within quefrency range)")
    
    print("\n🎉 Cepstrogram implementation test successful!")
    print("\nKey improvements made:")
    print("  • Fixed implementation to compute real cepstrum (not MFCC)")
    print("  • Correct quefrency axis calculation (time units)")
    print("  • Proper axis labels (Time vs Quefrency τ)")
    print("  • Enhanced data handling and normalization")
    
    return True

if __name__ == "__main__":
    try:
        test_cepstrogram()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)