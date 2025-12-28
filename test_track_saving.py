"""
Quick test script to verify track saving functionality.

Run this to test the basic track manager functions.
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Setup logging to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test imports
print("Testing imports...")
try:
    from audio_visualizer.ui.track_manager import TrackManager
    from audio_visualizer.ui.track_data import PaintedTrack
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test track manager initialization
print("\nTesting TrackManager initialization...")
try:
    tm = TrackManager()
    print(f"✓ TrackManager created, session_active={tm.session_active}")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    sys.exit(1)

# Test session creation
print("\nTesting session creation...")
try:
    test_dir = Path("test_track_session")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)

    success = tm.create_new_session(test_dir)
    print(f"✓ Session created: {success}")
    print(f"  - CSV path: {tm.csv_path}")
    print(f"  - Image dir: {tm.image_dir}")
    print(f"  - Data dir: {tm.data_dir}")
except Exception as e:
    print(f"✗ Session creation failed: {e}")
    sys.exit(1)

# Test interpolation
print("\nTesting PCHIP interpolation...")
try:
    control_points = [
        (0.0, 1000.0),
        (1.0, 1500.0),
        (2.0, 2000.0),
        (3.0, 1800.0)
    ]

    t_interp, f_interp = tm.interpolate_track(control_points, num_samples=50)
    print(f"✓ Interpolation successful")
    print(f"  - Input: {len(control_points)} control points")
    print(f"  - Output: {len(t_interp)} interpolated points")
    print(f"  - Time range: {t_interp[0]:.3f} - {t_interp[-1]:.3f} s")
    print(f"  - Freq range: {f_interp.min():.1f} - {f_interp.max():.1f} Hz")
except Exception as e:
    print(f"✗ Interpolation failed: {e}")
    import traceback
    traceback.print_exc()

# Test track creation with mock data
print("\nTesting track creation with mock spectrogram...")
try:
    # Create mock spectrogram data
    S = np.random.randn(512, 1000) * 10 - 50  # Random dB values
    times = np.linspace(0, 10, 1000)  # 10 seconds
    freqs = np.linspace(0, 22050, 512)  # Up to 22050 Hz

    track = tm.create_track(
        control_points=control_points,
        audio_file="test_audio.wav",
        S=S,
        times_axis=times,
        freqs_axis=freqs,
        num_interp_samples=100,
        track_label="Test Track",
        notes="Testing track creation"
    )

    if track:
        print(f"✓ Track created successfully")
        print(f"  - Track ID: {track.id}")
        print(f"  - Control points: {track.n_control_points}")
        print(f"  - Interpolated points: {track.n_interpolated_points}")
        print(f"  - Duration: {track.duration:.3f} s")
        print(f"  - Freq range: {track.f_min:.1f} - {track.f_max:.1f} Hz")
        print(f"  - Has amplitudes: {len(track.interpolated_amplitudes) > 0}")
    else:
        print(f"✗ Track creation returned None")
except Exception as e:
    print(f"✗ Track creation failed: {e}")
    import traceback
    traceback.print_exc()

# Test CSV export
print("\nTesting CSV export...")
try:
    success = tm.save_track_summary_csv()
    print(f"✓ Summary CSV saved: {success}")

    if track:
        success = tm.export_track_detailed_csv(track)
        print(f"✓ Detailed CSV saved: {success}")

        # Check files exist
        summary_exists = tm.csv_path.exists()
        detailed_path = tm.data_dir / f"track_{track.id:04d}_data.csv"
        detailed_exists = detailed_path.exists()

        print(f"  - Summary CSV exists: {summary_exists}")
        print(f"  - Detailed CSV exists: {detailed_exists}")

        if summary_exists:
            with open(tm.csv_path, 'r') as f:
                lines = f.readlines()
                print(f"  - Summary CSV lines: {len(lines)}")

        if detailed_exists:
            with open(detailed_path, 'r') as f:
                lines = f.readlines()
                print(f"  - Detailed CSV lines: {len(lines)}")

except Exception as e:
    print(f"✗ CSV export failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("All tests completed!")
print(f"Test session directory: {test_dir.absolute()}")
print("="*60)
