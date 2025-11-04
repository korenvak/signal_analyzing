"""
Configuration module for audio visualization application.
Optimized for performance with GPU acceleration and memory management.
"""

import os
import multiprocessing as mp
from typing import Dict, Any

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

class PerformanceConfig:
    """Performance optimization configuration."""
    
    def __init__(self):
        # CPU Configuration
        self.max_threads = min(8, mp.cpu_count())
        self.max_processes = max(1, mp.cpu_count() - 1)
        
        # Memory Configuration (MB)
        self.cache_size_mb = int(os.environ.get('AUDIO_VISUALIZER_CACHE_SIZE', 2048))
        self.gpu_cache_size_mb = int(os.environ.get('AUDIO_VISUALIZER_GPU_CACHE_SIZE', 1024))
        
        # GPU Configuration
        self.gpu_enabled = os.environ.get('AUDIO_VISUALIZER_GPU_ENABLED', 'true').lower() == 'true'
        self.gpu_batch_size = 8
        self.gpu_memory_fraction = 0.8  # Use 80% of available GPU memory
        
        # Audio Processing
        self.default_chunk_size = 1024 * 1024  # 1MB chunks
        self.default_overlap = 4096
        self.max_audio_duration = 4 * 3600  # 4 hours
        
        # Visualization
        self.target_fps = 60
        self.max_texture_size = 4096
        self.tile_size = (512, 512)
        self.max_tiles_memory = 100  # Maximum tiles in memory
        
        # File I/O
        self.use_memory_mapping = True
        self.compression_level = 3  # For zarr arrays
        
        # Performance monitoring
        self.enable_profiling = os.environ.get('AUDIO_VISUALIZER_PROFILE', 'false').lower() == 'true'
        self.performance_log_interval = 5.0  # seconds
        
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information and capabilities."""
        if not HAS_CUPY or not self.gpu_enabled:
            return {'available': False, 'reason': 'CuPy not available or GPU disabled'}
        
        try:
            gpu_count = cp.cuda.runtime.getDeviceCount()
            if gpu_count == 0:
                return {'available': False, 'reason': 'No CUDA devices found'}
            
            device_props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_memory = device_props['totalGlobalMem']
            
            return {
                'available': True,
                'device_count': gpu_count,
                'device_name': device_props['name'].decode('utf-8'),
                'memory_total_gb': gpu_memory / (1024**3),
                'memory_available_gb': (gpu_memory * self.gpu_memory_fraction) / (1024**3),
                'compute_capability': f"{device_props['major']}.{device_props['minor']}",
                'multiprocessor_count': device_props['multiProcessorCount']
            }
        except Exception as e:
            return {'available': False, 'reason': f'GPU initialization failed: {e}'}
    
    def optimize_for_system(self):
        """Automatically optimize configuration based on system capabilities."""
        gpu_info = self.get_gpu_info()
        
        if gpu_info['available']:
            # Adjust GPU cache based on available memory
            gpu_memory_gb = gpu_info.get('memory_available_gb', 1.0)
            
            if gpu_memory_gb >= 8.0:
                self.gpu_cache_size_mb = 2048
                self.gpu_batch_size = 16
            elif gpu_memory_gb >= 4.0:
                self.gpu_cache_size_mb = 1024
                self.gpu_batch_size = 8
            else:
                self.gpu_cache_size_mb = 512
                self.gpu_batch_size = 4
        else:
            # Disable GPU features
            self.gpu_enabled = False
            self.gpu_cache_size_mb = 0
        
        # Adjust CPU cache based on system RAM
        try:
            import psutil
            system_memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if system_memory_gb >= 16.0:
                self.cache_size_mb = 4096
            elif system_memory_gb >= 8.0:
                self.cache_size_mb = 2048
            else:
                self.cache_size_mb = 1024
        except ImportError:
            # Keep default if psutil not available
            pass
        
        # Adjust thread count for better performance
        cpu_count = mp.cpu_count()
        if cpu_count >= 8:
            self.max_threads = 8
        elif cpu_count >= 4:
            self.max_threads = 6
        else:
            self.max_threads = 4

class AudioConfig:
    """Audio processing configuration."""
    
    def __init__(self):
        # STFT Parameters
        self.default_fft_size = 2048
        self.default_hop_length = 512
        self.default_window = 'hann'
        
        # Supported sample rates
        self.supported_sample_rates = [8000, 16000, 22050, 44100, 48000, 96000]
        self.default_sample_rate = 44100
        
        # Frequency analysis
        self.log_frequency_scale = True
        self.min_frequency = 20.0  # Hz
        self.max_frequency_ratio = 0.5  # Fraction of Nyquist
        
        # Cepstral analysis
        self.default_mel_filters = 40
        self.default_lifter_cutoff = 13
        self.default_cepstral_coeffs = 13
        
        # F-K analysis  
        self.default_sound_speed = 343.0  # m/s
        self.default_wavenumber_resolution = 0.1

class VisualizationConfig:
    """Visualization settings configuration."""
    
    def __init__(self):
        # Colormaps
        self.available_colormaps = ['viridis', 'plasma', 'jet', 'magma', 'inferno', 'cividis']
        self.default_colormap = 'viridis'
        
        # Dynamic range
        self.default_db_min = -80.0
        self.default_db_max = 0.0
        self.db_range_limits = (-120.0, 20.0)
        
        # UI settings
        self.default_window_size = (1200, 800)
        self.min_window_size = (800, 600)
        self.status_update_interval = 1000  # ms
        
        # Rendering
        self.enable_antialiasing = True
        self.texture_filtering = 'linear'
        self.crosshair_color = (1.0, 1.0, 1.0, 0.8)
        self.crosshair_width = 2.0

# Global configuration instances
performance_config = PerformanceConfig()
audio_config = AudioConfig()
visualization_config = VisualizationConfig()

def initialize_config():
    """Initialize and optimize configuration for current system."""
    performance_config.optimize_for_system()
    
    # Log configuration
    gpu_info = performance_config.get_gpu_info()
    
    print("=== Audio Visualizer Configuration ===")
    print(f"CPU: {performance_config.max_threads} threads, {performance_config.max_processes} processes")
    print(f"Memory: {performance_config.cache_size_mb}MB cache")
    
    if gpu_info['available']:
        print(f"GPU: {gpu_info['device_name']} ({gpu_info['memory_available_gb']:.1f}GB available)")
        print(f"GPU Cache: {performance_config.gpu_cache_size_mb}MB, Batch size: {performance_config.gpu_batch_size}")
    else:
        print(f"GPU: Disabled ({gpu_info.get('reason', 'Unknown')})")
    
    print(f"Audio: {audio_config.default_sample_rate}Hz, FFT={audio_config.default_fft_size}")
    print("=" * 40)

def get_optimal_fft_params(sample_rate: int, target_time_res: float = 0.01, 
                          target_freq_res: float = 50.0) -> Dict[str, int]:
    """Calculate optimal FFT parameters for given requirements."""
    import numpy as np
    
    # Time resolution determines hop length
    optimal_hop = int(target_time_res * sample_rate)
    optimal_hop = max(64, min(1024, optimal_hop))
    
    # Frequency resolution determines FFT size
    optimal_fft = int(sample_rate / target_freq_res)
    optimal_fft = 2 ** int(np.ceil(np.log2(optimal_fft)))
    optimal_fft = max(512, min(8192, optimal_fft))
    
    return {
        'fft_size': optimal_fft,
        'hop_length': optimal_hop,
        'time_resolution': optimal_hop / sample_rate,
        'freq_resolution': sample_rate / optimal_fft
    }

if __name__ == "__main__":
    initialize_config()