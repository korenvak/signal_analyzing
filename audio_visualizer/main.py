#!/usr/bin/env python3
"""
GPU-Accelerated Audio Visualization Application
Main entry point for the interactive spectrogram, cepstrogram, and F-K transform visualizer.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('audio_visualizer.log')
    ]
)

logger = logging.getLogger(__name__)

def check_dependencies():
    """Check for required dependencies and provide helpful error messages."""
    missing_deps = []
    optional_missing = []
    
    # Required dependencies
    required_deps = {
        'numpy': 'numpy',
        'scipy': 'scipy', 
        'librosa': 'librosa',
        'soundfile': 'soundfile',
        'PySide6': 'PySide6'
    }
    
    # Optional but recommended dependencies
    optional_deps = {
        'cupy': 'cupy-cuda11x (for GPU acceleration)',
        'vispy': 'vispy (for advanced visualization)',
        'zarr': 'zarr (for efficient caching)'
    }
    
    for module, package in required_deps.items():
        try:
            __import__(module)
        except ImportError:
            missing_deps.append(package)
    
    for module, package in optional_deps.items():
        try:
            __import__(module)
        except ImportError:
            optional_missing.append(package)
    
    if missing_deps:
        logger.error("Missing required dependencies:")
        for dep in missing_deps:
            logger.error(f"  - {dep}")
        logger.error("Install with: pip install " + " ".join(missing_deps))
        return False
    
    if optional_missing:
        logger.warning("Missing optional dependencies (reduced functionality):")
        for dep in optional_missing:
            logger.warning(f"  - {dep}")
        logger.warning("Install with: pip install " + " ".join([d.split()[0] for d in optional_missing]))
    
    return True

def setup_gpu_acceleration():
    """Setup and verify GPU acceleration capabilities."""
    try:
        import cupy as cp
        
        # Test GPU availability
        gpu_count = cp.cuda.runtime.getDeviceCount()
        if gpu_count > 0:
            device_props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = device_props['name'].decode('utf-8')
            gpu_memory = device_props['totalGlobalMem'] / (1024**3)  # GB
            
            logger.info(f"GPU acceleration available: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            logger.warning("No CUDA-capable GPU found")
            return False
            
    except ImportError:
        logger.warning("CuPy not available - GPU acceleration disabled")
        return False
    except Exception as e:
        logger.warning(f"GPU initialization failed: {e}")
        return False

def create_sample_audio(output_path: str, duration: float = 10.0, sample_rate: int = 44100):
    """Create a sample audio file for testing."""
    import numpy as np
    import soundfile as sf
    
    logger.info(f"Creating sample audio file: {output_path}")
    
    # Generate test signal with multiple frequency components
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    
    # Fundamental frequency with harmonics
    f0 = 440.0  # A4
    signal = (np.sin(2 * np.pi * f0 * t) +
              0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
              0.25 * np.sin(2 * np.pi * 3 * f0 * t))
    
    # Add frequency modulation
    fm_signal = np.sin(2 * np.pi * 880 * t * (1 + 0.1 * np.sin(2 * np.pi * 2 * t)))
    
    # Add some noise
    noise = 0.1 * np.random.randn(len(t))
    
    # Combine signals
    audio = 0.3 * (signal + 0.5 * fm_signal + noise)
    
    # Apply envelope
    envelope = np.exp(-t / (duration * 0.3))
    audio = audio * envelope
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Save as WAV file
    sf.write(output_path, audio, sample_rate)
    logger.info(f"Sample audio created: {duration}s at {sample_rate}Hz")

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="GPU-Accelerated Audio Visualization Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Start with GUI
  python main.py --file audio.wav        # Load specific file
  python main.py --create-sample          # Create sample audio file
  python main.py --no-gpu                 # Disable GPU acceleration
        """
    )
    
    parser.add_argument('--file', '-f', type=str, 
                       help='Audio file to load on startup')
    parser.add_argument('--create-sample', action='store_true',
                       help='Create a sample audio file for testing')
    parser.add_argument('--sample-file', type=str, default='sample_audio.wav',
                       help='Output filename for sample audio')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Disable GPU acceleration')
    parser.add_argument('--cache-size', type=int, default=2048,
                       help='Cache size in MB (default: 2048)')
    parser.add_argument('--gpu-cache-size', type=int, default=1024,
                       help='GPU cache size in MB (default: 1024)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--version', action='version', version='Audio Visualizer 1.0.0')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    logger.info("Starting Audio Visualization Application")
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Create sample audio if requested
    if args.create_sample:
        create_sample_audio(args.sample_file)
        logger.info(f"Sample audio file created: {args.sample_file}")
        if not args.file:
            return 0
    
    # Setup GPU acceleration
    gpu_available = False if args.no_gpu else setup_gpu_acceleration()
    
    try:
        # Import and start the GUI application
        from audio_visualizer.ui.main_window import main as start_gui
        
        logger.info("Launching GUI application...")
        
        # Set initial file if provided
        if args.file:
            if not os.path.exists(args.file):
                logger.error(f"Audio file not found: {args.file}")
                return 1
            os.environ['AUDIO_VISUALIZER_INITIAL_FILE'] = args.file
        
        # Set cache configuration
        os.environ['AUDIO_VISUALIZER_CACHE_SIZE'] = str(args.cache_size)
        os.environ['AUDIO_VISUALIZER_GPU_CACHE_SIZE'] = str(args.gpu_cache_size)
        os.environ['AUDIO_VISUALIZER_GPU_ENABLED'] = str(not args.no_gpu and gpu_available)
        
        # Start the GUI
        exit_code = start_gui()
        
        logger.info("Application finished")
        return exit_code
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=args.debug)
        return 1

if __name__ == "__main__":
    sys.exit(main())