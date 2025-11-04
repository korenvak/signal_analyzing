# GPU-Accelerated Audio Visualization Application

A high-performance, interactive Python application for visualizing large audio analysis data with GPU acceleration, featuring Spectrogram, Cepstrogram, and F-K Transform views.

## Features

### 🚀 Performance Optimizations
- **GPU Acceleration**: CUDA-accelerated FFT computations using CuPy
- **Memory Efficiency**: LRU tile caching with intelligent memory management
- **Lazy Loading**: Compute visualizations only when tabs are accessed
- **Multi-threading**: Background computation with responsive UI
- **High Frame Rates**: Optimized for 30-60 FPS real-time rendering

### 📊 Visualization Types
1. **Spectrogram**: Time-frequency analysis with configurable FFT parameters
2. **Cepstrogram**: Mel-frequency cepstral analysis for speech/audio processing
3. **F-K Transform**: Frequency-wavenumber analysis for beamforming applications

### 🎛️ Interactive Features
- **Real-time Controls**: FFT size, hop length, colormap, dB range adjustment
- **Zoom & Pan**: Smooth navigation through large datasets
- **Crosshair Display**: Precise time/frequency readouts
- **Multiple Colormaps**: viridis, plasma, jet, magma
- **Tab-based Interface**: Separate lazy-loaded analysis views

### 💾 Memory Management
- **Tile-based Caching**: Multi-resolution pyramid with LRU eviction
- **GPU Memory Pool**: Efficient VRAM usage with automatic cleanup
- **Disk Spillover**: Large datasets cached to disk using Zarr
- **Memory Monitoring**: Real-time memory usage statistics

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for acceleration)
- CUDA Toolkit 11.x+ (for GPU features)

### Required Dependencies
```bash
pip install numpy scipy librosa soundfile PySide6 vispy zarr pyopengl
```

### Optional GPU Dependencies
```bash
# For CUDA 11.x
pip install cupy-cuda11x

# For CUDA 12.x
pip install cupy-cuda12x
```

### Quick Install
```bash
# Clone or download the project
cd audio_visualizer

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Usage

### Basic Usage
```bash
# Start the application
python main.py

# Load a specific audio file
python main.py --file audio.wav

# Create sample audio for testing
python main.py --create-sample

# Disable GPU acceleration
python main.py --no-gpu
```

### Command Line Options
```
--file, -f FILE         Audio file to load on startup
--create-sample         Create a sample audio file for testing
--sample-file FILE      Output filename for sample audio (default: sample_audio.wav)
--no-gpu               Disable GPU acceleration
--cache-size SIZE      Cache size in MB (default: 2048)
--gpu-cache-size SIZE  GPU cache size in MB (default: 1024)
--debug                Enable debug logging
--version              Show version information
```

### Interface Controls

#### Main Controls
- **FFT Size**: 512, 1024, 2048, 4096, 8192 (affects frequency resolution)
- **Hop Length**: 128, 256, 512, 1024 (affects time resolution)
- **Colormap**: viridis, plasma, jet, magma
- **dB Range**: Adjustable min/max for dynamic range visualization
- **Refresh**: Recompute current view with new parameters

#### Keyboard Shortcuts
- `Ctrl+O`: Open audio file
- `Ctrl+0`: Zoom to fit
- `F5`: Refresh current view
- `Ctrl+Q`: Quit application

#### Mouse Controls
- **Zoom**: Mouse wheel or pinch gesture
- **Pan**: Click and drag
- **Crosshair**: Move mouse over visualization

### Analysis Tabs

#### 1. Spectrogram
- Real-time STFT computation
- GPU-accelerated for large files
- Logarithmic frequency scaling
- Dynamic range adjustment

#### 2. Cepstrogram
- Mel-frequency cepstral coefficients (MFCCs)
- Automatic formant analysis
- Liftering and cepstral mean normalization
- Reuses cached spectrogram data

#### 3. F-K Transform
- Frequency-wavenumber analysis
- Beamforming pattern visualization
- Region-of-interest (ROI) analysis
- Wave propagation direction estimation

## Architecture

### Core Components
```
audio_visualizer/
├── core/
│   ├── data_loader.py      # Chunked audio loading
│   ├── cache_manager.py    # LRU tile cache
│   └── task_manager.py     # Background workers
├── engines/
│   ├── spectrogram_engine.py  # GPU STFT computation
│   ├── cepstrogram_engine.py  # MFCC analysis
│   └── fk_engine.py          # F-K transforms
├── rendering/
│   ├── render_manager.py     # GPU rendering
│   └── shaders/              # OpenGL shaders
├── ui/
│   └── main_window.py        # Qt interface
└── main.py                   # Entry point
```

### Data Flow
```
Audio File → DataLoader → SpectrogramEngine → Cache → RenderManager → Display
                     ↓
              CepstrogramEngine (lazy)
                     ↓
                FKEngine (lazy)
```

### GPU Pipeline
1. **Audio Loading**: Chunked streaming from disk
2. **STFT Computation**: CuPy-accelerated FFT on GPU
3. **Tile Generation**: Split large data into manageable tiles
4. **Caching**: LRU cache with GPU/CPU memory management
5. **Rendering**: OpenGL textures with shader-based colormapping

## Performance Targets

- **Load Time**: < 3 seconds per tab
- **Memory Usage**: < 2GB RAM, < 1GB VRAM
- **Frame Rate**: 30-60 FPS for smooth interaction
- **File Support**: Up to 4 hours of audio data

## Supported Formats

- **WAV**: Uncompressed PCM audio
- **FLAC**: Lossless compression
- **MP3**: Lossy compression (via librosa)
- **OGG**: Ogg Vorbis (via librosa)

## GPU Requirements

### Minimum
- CUDA Compute Capability 3.5+
- 2GB VRAM
- CUDA Toolkit 11.0+

### Recommended
- CUDA Compute Capability 6.0+
- 4GB+ VRAM
- CUDA Toolkit 12.0+

## Configuration

### Environment Variables
```bash
export AUDIO_VISUALIZER_CACHE_SIZE=2048        # Cache size in MB
export AUDIO_VISUALIZER_GPU_CACHE_SIZE=1024    # GPU cache size in MB
export AUDIO_VISUALIZER_GPU_ENABLED=true       # Enable GPU acceleration
```

### Performance Tuning
- Adjust cache sizes based on available memory
- Use lower resolution levels for very large files
- Enable disk caching for datasets > 1GB
- Monitor memory usage via status bar

## Troubleshooting

### Common Issues

#### GPU Acceleration Not Working
```bash
# Check CUDA installation
nvidia-smi

# Check CuPy installation
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"

# Run without GPU
python main.py --no-gpu
```

#### Memory Issues
- Reduce cache sizes: `--cache-size 1024 --gpu-cache-size 512`
- Use lower resolution audio files
- Close unused applications

#### Performance Issues
- Enable GPU acceleration
- Reduce FFT size for real-time interaction
- Use SSD storage for large audio files

#### Import Errors
```bash
# Install missing dependencies
pip install -r requirements.txt

# Check specific imports
python -c "import librosa, soundfile, vispy, PySide6"
```

## Development

### Building from Source
```bash
git clone <repository>
cd audio_visualizer
pip install -e .
```

### Running Tests
```bash
python -m pytest tests/
```

### Code Structure
- Follow PEP 8 style guidelines
- Use type hints for all functions
- Document all public APIs
- Add unit tests for new features

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- **VisPy**: High-performance scientific visualization
- **CuPy**: GPU-accelerated computing
- **librosa**: Audio analysis library
- **Qt**: Cross-platform GUI framework

## Citation

If you use this software in research, please cite:
```
@software{audio_visualizer,
  title={GPU-Accelerated Audio Visualization Application},
  author={Audio Visualization Team},
  year={2024},
  url={https://github.com/your-repo/audio-visualizer}
}
```