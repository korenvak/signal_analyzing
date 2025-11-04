"""
Test tile system with actual large file
"""

import sys
from audio_visualizer.ui.main_window import MainWindow
from PySide6.QtWidgets import QApplication

# Calculate expected frames for the 21-minute file
duration = 21 * 60  # 1260 seconds
sample_rate = 44100
fft_size = 2048
hop_length = 512

n_samples = duration * sample_rate
expected_frames = 1 + (n_samples - fft_size) // hop_length

print("="*70)
print("TILE SYSTEM TEST")
print("="*70)
print(f"\nFile: 21-minute audio")
print(f"Duration: {duration}s ({duration/60:.1f} minutes)")
print(f"Sample rate: {sample_rate} Hz")
print(f"Total samples: {n_samples:,}")
print(f"FFT size: {fft_size}, Hop: {hop_length}")
print(f"\nExpected frames: {expected_frames:,}")
print(f"OpenGL limit: 16,384")
print(f"Exceeds limit: {expected_frames > 16384}")
print(f"Downsample factor needed: {expected_frames / 16384:.2f}x")

print("\n"+"="*70)
print("TILE SYSTEM LOGIC")
print("="*70)

if expected_frames > 16384:
    n_tiles = (duration + 59) // 60  # 60-second tiles
    print(f"\n[OK] File exceeds OpenGL limit")
    print(f"  -> Tile system SHOULD activate")
    print(f"  -> Need approximately {n_tiles} tiles (60s each)")
    print(f"  -> Each tile: ~{expected_frames//n_tiles} frames")
else:
    print(f"\n[Skip] File fits in single texture")
    print(f"  -> Direct rendering (no tiling needed)")

print("\n"+"="*70)
print("\nLoad the 21-minute file in the app to test tile system.")
print("Look for log message: 'File width (19687) exceeds limit'")
print("="*70)

