"""
Comprehensive test to verify all functionality still works
"""

import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_multiple_scenarios():
    """Test multiple scenarios to ensure nothing broke."""
    logger.info("="*60)
    logger.info("COMPREHENSIVE FUNCTIONALITY TEST")
    logger.info("="*60)
    
    from audio_visualizer.engines.batched_fft_engine import get_batched_fft_engine
    
    engine = get_batched_fft_engine()
    results = []
    
    # Test 1: Short audio (1 second)
    logger.info("\nTest 1: Short audio (1 second)")
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    try:
        magnitude_db, times = engine.compute_stft_batched(
            audio, fft_size=2048, hop_length=512, window='hann', sample_rate=sample_rate
        )
        assert magnitude_db.shape[0] == 2048 // 2 + 1
        assert magnitude_db.shape[1] > 0
        logger.info(f"✅ Pass: Shape {magnitude_db.shape}")
        results.append(True)
    except Exception as e:
        logger.error(f"❌ Fail: {e}")
        results.append(False)
    
    # Test 2: Medium audio (10 seconds)
    logger.info("\nTest 2: Medium audio (10 seconds)")
    duration = 10.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    try:
        magnitude_db, times = engine.compute_stft_batched(
            audio, fft_size=4096, hop_length=512, window='blackman', sample_rate=sample_rate
        )
        assert magnitude_db.shape[0] == 4096 // 2 + 1
        assert magnitude_db.shape[1] > 0
        logger.info(f"✅ Pass: Shape {magnitude_db.shape}")
        results.append(True)
    except Exception as e:
        logger.error(f"❌ Fail: {e}")
        results.append(False)
    
    # Test 3: Different window types
    logger.info("\nTest 3: Different window types")
    duration = 2.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    windows = ['hann', 'hamming', 'blackman', 'blackmanharris', 'kaiser']
    for window in windows:
        try:
            magnitude_db, times = engine.compute_stft_batched(
                audio, fft_size=2048, hop_length=512, window=window, sample_rate=sample_rate
            )
            assert np.isfinite(magnitude_db).all()
            logger.info(f"✅ Pass: Window {window}")
            results.append(True)
        except Exception as e:
            logger.error(f"❌ Fail: Window {window} - {e}")
            results.append(False)
    
    # Test 4: Different FFT sizes
    logger.info("\nTest 4: Different FFT sizes")
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    fft_sizes = [1024, 2048, 4096, 8192]
    for fft_size in fft_sizes:
        try:
            magnitude_db, times = engine.compute_stft_batched(
                audio, fft_size=fft_size, hop_length=fft_size//4, window='hann', sample_rate=sample_rate
            )
            assert magnitude_db.shape[0] == fft_size // 2 + 1
            logger.info(f"✅ Pass: FFT size {fft_size}")
            results.append(True)
        except Exception as e:
            logger.error(f"❌ Fail: FFT size {fft_size} - {e}")
            results.append(False)
    
    # Test 5: Very short audio (edge case)
    logger.info("\nTest 5: Very short audio (edge case)")
    audio = np.random.randn(1000).astype(np.float32)
    
    try:
        magnitude_db, times = engine.compute_stft_batched(
            audio, fft_size=2048, hop_length=512, window='hann', sample_rate=sample_rate
        )
        assert magnitude_db.shape[0] == 2048 // 2 + 1
        logger.info(f"✅ Pass: Very short audio")
        results.append(True)
    except Exception as e:
        logger.error(f"❌ Fail: {e}")
        results.append(False)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    passed = sum(results)
    total = len(results)
    logger.info(f"Passed: {passed}/{total}")
    
    if passed == total:
        logger.info("✅ All tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} test(s) failed!")
        return False


if __name__ == "__main__":
    success = test_multiple_scenarios()
    exit(0 if success else 1)

