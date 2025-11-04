# GPU-Accelerated Audio Visualizer

High-performance audio visualization system for spectrograms, cepstrograms, and F-K transforms.

## Features

### Core Functionality
- ✅ **Spectrogram Visualization** - Time-frequency analysis with correct orientation
- ✅ **Cepstrogram (MFCC)** - Mel-frequency cepstral coefficients
- ✅ **GPU Acceleration** - NVIDIA CUDA support with CuPy
- ✅ **Batched FFT** - 10-100x faster than traditional per-frame computation
- ✅ **Large File Support** - Handles hours of audio efficiently

### Performance Optimizations
- **Batched FFT Engine**: Persistent FFTW/cuFFT plans, batch processing
- **GPU Memory Management**: Real-time monitoring, automatic cleanup
- **Tile-Based Architecture**: Foundation for unlimited data sizes
- **Multi-Resolution System**: 6 LOD levels for smooth zoom
- **Memory Pools**: Workspace reuse, zero-copy operations

### Current Capabilities
- Processes 21-minute audio files (19,687+ frames)
- GPU throughput: 785-3,419 frames/second
- Memory efficient: ~1.5GB GPU usage
- Stable performance with no crashes

## Installation

### Requirements
- Python 3.11+
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ GPU VRAM

### Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r audio_visualizer/requirements.txt
```

### Dependencies
- numpy, scipy
- vispy (visualization)
- PySide6 (GUI)
- cupy-cuda12x (GPU acceleration)
- librosa, soundfile (audio I/O)
- PyOpenGL, PyOpenGL-accelerate
- zarr (disk caching)
- numba, matplotlib

## Usage

### Run Application
```bash
python run_audio_visualizer.py
```

### With Specific File
```bash
python run_audio_visualizer.py --file "path/to/audio.flac"
```

### GPU Management
```bash
# Check GPU health
python check_gpu_health.py

# Clean GPU memory
python cleanup_gpu.py
```

## Architecture

```
audio_visualizer/
├── core/               # Core infrastructure
│   ├── data_loader.py          # Chunked audio loading
│   ├── cache_manager.py        # Legacy cache
│   ├── task_manager.py         # Background tasks
│   ├── tile_cache.py           # Disk-backed tile cache
│   ├── tile_manager.py         # Tile coordination
│   ├── mipmap_pyramid.py       # Multi-resolution
│   ├── gpu_memory_manager.py   # GPU monitoring
│   └── memory_pools.py         # Workspace management
├── engines/            # Computation engines
│   ├── spectrogram_engine.py   # STFT computation
│   ├── cepstrogram_engine.py   # MFCC computation
│   ├── fk_engine.py            # F-K transform
│   └── batched_fft_engine.py   # Optimized FFT
├── rendering/          # GPU rendering
│   ├── render_manager.py       # OpenGL rendering
│   ├── texture_atlas.py        # Tile atlas
│   └── shaders/                # GLSL shaders
└── ui/                 # User interface
    └── main_window.py          # Main GUI
```

## Performance

### Benchmarks (21-minute audio file)
- **Computation**: 19,687 frames in ~8 seconds (batched FFT)
- **GPU Memory**: 1.3GB / 8GB (16% utilization)
- **Frame Rate**: 60 FPS smooth interaction
- **Cache**: Persistent disk cache, instant reload

### Comparison
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| FFT Speed | ~100 fps | 3,419 fps | 34x |
| GPU Memory | 21GB (crash) | 1.3GB | 16x reduction |
| Max Frames | 513 | 19,687+ | 38x |

## Known Issues & Roadmap

### In Progress
- [ ] Axis labels visibility (code written, needs VisPy debugging)
- [ ] Separate axis zoom controls (code written, event system needs work)
- [ ] Camera boundary constraints (code written, enforcement needs tuning)

### Planned
- [ ] Full tile system integration (unlimited resolution)
- [ ] GPU stream management (2-3x speedup)
- [ ] Shader-based colormap (GPU colormapping)
- [ ] Grid lines and tick marks
- [ ] F-K transform optimization
- [ ] Export functionality

## Documentation

- `ORIENTATION_AND_PERFORMANCE_FIX.md` - Complete development history
- `TODO_NEXT_PHASE.md` - Roadmap and priorities
- `USAGE_GUIDE.md` - User guide (if exists)

## Development

### Run Tests
```bash
python test_phase2_integration.py
```

### Check GPU
```bash
python check_gpu_health.py
```

## License

[Add your license]

## Author

[Add your info]

## Acknowledgments

Built using:
- VisPy for GPU-accelerated visualization
- CuPy for CUDA operations
- PySide6 for GUI
- FFTW for optimized FFT computation

