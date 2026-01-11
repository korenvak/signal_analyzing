import numpy as np
import soundfile as sf
import librosa
from typing import Optional, Tuple, Generator
import threading
import queue
import gc
import mmap
import os
import logging

logger = logging.getLogger(__name__)


class MemoryMappedAudioLoader:
    """
    Memory-mapped audio loader for HUGE files (10+ GB).
    
    Uses memory mapping to:
    - Only load data on-demand (lazy loading)
    - Share memory with OS page cache (no double buffering)
    - Handle files larger than RAM
    - Near-instant file "loading" (just maps, doesn't read)
    """
    
    def __init__(self):
        self._file_path = None
        self._mmap = None
        self._file_handle = None
        self.sample_rate = None
        self.duration = 0.0
        self.total_samples = 0
        self.channels = 1
        self._header_size = 0
        self._dtype = np.float32
        self._bytes_per_sample = 4
    
    def load_file(self, file_path: str, target_sr: Optional[int] = None) -> Tuple[int, float]:
        """Memory-map an audio file for zero-copy access.
        
        For WAV files: Direct memory mapping (fastest)
        For other formats: Fall back to soundfile streaming
        """
        self.close()
        self._file_path = file_path
        
        # Get file info first
        info = sf.info(file_path)
        self.sample_rate = target_sr or info.samplerate
        self.duration = info.duration
        self.channels = info.channels
        
        # Calculate total samples
        if target_sr and target_sr != info.samplerate:
            self.total_samples = int(info.frames * target_sr / info.samplerate)
        else:
            self.total_samples = info.frames
        
        # Check if WAV file - can be memory-mapped directly
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.wav' and info.subtype in ('PCM_16', 'PCM_24', 'PCM_32', 'FLOAT', 'DOUBLE'):
            try:
                self._setup_wav_mmap(file_path, info)
                logger.info(f"Memory-mapped WAV file: {file_path} ({self.duration:.1f}s, {self.total_samples} samples)")
                return self.sample_rate, self.duration
            except Exception as e:
                logger.warning(f"WAV memory mapping failed, using streaming: {e}")
        
        # Fallback: File will be streamed on-demand (no mmap)
        self._mmap = None
        logger.info(f"Prepared for streaming: {file_path} ({self.duration:.1f}s)")
        return self.sample_rate, self.duration
    
    def _setup_wav_mmap(self, file_path: str, info):
        """Setup memory mapping for WAV file."""
        # WAV header is typically 44 bytes, but can vary
        # We'll find the 'data' chunk to get exact offset
        self._file_handle = open(file_path, 'rb')
        
        # Find data chunk offset
        self._file_handle.seek(0)
        header = self._file_handle.read(44)
        
        # Basic WAV header parsing
        if header[:4] != b'RIFF' or header[8:12] != b'WAVE':
            raise ValueError("Not a valid WAV file")
        
        # Find 'data' chunk
        offset = 12
        while offset < len(header):
            chunk_id = header[offset:offset+4]
            chunk_size = int.from_bytes(header[offset+4:offset+8], 'little')
            if chunk_id == b'data':
                self._header_size = offset + 8
                break
            offset += 8 + chunk_size
        
        if self._header_size == 0:
            # data chunk not in first 44 bytes, search further
            self._file_handle.seek(12)
            while True:
                chunk_header = self._file_handle.read(8)
                if len(chunk_header) < 8:
                    raise ValueError("Could not find data chunk")
                chunk_id = chunk_header[:4]
                chunk_size = int.from_bytes(chunk_header[4:8], 'little')
                if chunk_id == b'data':
                    self._header_size = self._file_handle.tell()
                    break
                self._file_handle.seek(chunk_size, 1)  # Skip chunk
        
        # Determine sample format
        if info.subtype == 'PCM_16':
            self._dtype = np.int16
            self._bytes_per_sample = 2
        elif info.subtype == 'PCM_24':
            # 24-bit needs special handling - not directly supported
            raise ValueError("24-bit WAV not supported for mmap")
        elif info.subtype == 'PCM_32':
            self._dtype = np.int32
            self._bytes_per_sample = 4
        elif info.subtype == 'FLOAT':
            self._dtype = np.float32
            self._bytes_per_sample = 4
        elif info.subtype == 'DOUBLE':
            self._dtype = np.float64
            self._bytes_per_sample = 8
        
        # Memory map the file
        file_size = os.path.getsize(file_path)
        self._mmap = mmap.mmap(
            self._file_handle.fileno(), 
            file_size,
            access=mmap.ACCESS_READ
        )
    
    def get_samples(self, start_sample: int, num_samples: int) -> np.ndarray:
        """Get audio samples from memory-mapped file.
        
        This is the key performance method - provides near-instant access
        to any portion of the file.
        
        Args:
            start_sample: Starting sample index
            num_samples: Number of samples to read
            
        Returns:
            Audio data as float32 numpy array
        """
        if self._mmap is not None:
            # Direct memory access - FAST
            return self._get_samples_mmap(start_sample, num_samples)
        else:
            # Streaming fallback
            return self._get_samples_stream(start_sample, num_samples)
    
    def _get_samples_mmap(self, start_sample: int, num_samples: int) -> np.ndarray:
        """Get samples from memory-mapped file."""
        # Calculate byte offset
        start_byte = self._header_size + (start_sample * self._bytes_per_sample * self.channels)
        end_byte = start_byte + (num_samples * self._bytes_per_sample * self.channels)
        
        # Read from memory map (OS handles page loading automatically)
        raw_data = self._mmap[start_byte:end_byte]
        
        # Convert to numpy array
        samples = np.frombuffer(raw_data, dtype=self._dtype)
        
        # Handle multi-channel: take first channel or average
        if self.channels > 1:
            samples = samples.reshape(-1, self.channels)
            samples = samples[:, 0]  # Take first channel
        
        # Convert to float32 normalized
        if self._dtype == np.int16:
            return (samples.astype(np.float32) / 32768.0)
        elif self._dtype == np.int32:
            return (samples.astype(np.float32) / 2147483648.0)
        elif self._dtype == np.float64:
            return samples.astype(np.float32)
        else:
            return samples.astype(np.float32)
    
    def _get_samples_stream(self, start_sample: int, num_samples: int) -> np.ndarray:
        """Get samples using soundfile streaming (fallback)."""
        try:
            audio, _ = sf.read(
                self._file_path,
                start=start_sample,
                stop=start_sample + num_samples,
                dtype='float32',
                always_2d=False
            )
            
            # Handle multi-channel
            if audio.ndim > 1:
                audio = audio[:, 0]  # Take first channel
            
            return audio.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Stream read error: {e}")
            return np.zeros(num_samples, dtype=np.float32)
    
    def close(self):
        """Close memory mapping and file handle."""
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
    
    def __del__(self):
        self.close()


class ChunkedAudioLoader:
    """High-performance chunked audio loader for large files with streaming support.
    
    PERFORMANCE FEATURES:
    - Memory-mapped loading for WAV files (instant access)
    - LRU chunk cache to minimize disk reads
    - Background preloading for smooth playback
    - Automatic format detection and optimal loading
    """
    
    def __init__(self, chunk_size: int = 1024*1024, overlap: int = 4096):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.audio_data = None
        self.sample_rate = None
        self.duration = 0.0
        self._file_path = None
        self._loading_thread = None
        self._chunk_cache = {}
        self._cache_order = []  # LRU tracking
        self._max_cache_chunks = 50
        
        # Memory-mapped loader for large files
        self._mmap_loader = MemoryMappedAudioLoader()
        self._use_mmap = False
        
    def load_file(self, file_path: str, target_sr: Optional[int] = None) -> Tuple[int, float]:
        """Load audio file metadata and prepare for chunked reading.
        
        PERFORMANCE: Automatically uses memory-mapping for large WAV files.
        
        Args:
            file_path: Path to audio file
            target_sr: Target sample rate (None = native rate)
            
        Returns:
            (sample_rate, duration) tuple
        """
        self._file_path = file_path
        self.clear_cache()
        
        # Get file info
        info = sf.info(file_path)
        self.sample_rate = target_sr or info.samplerate
        self.duration = info.duration
        
        if target_sr and target_sr != info.samplerate:
            self.total_samples = int(info.frames * target_sr / info.samplerate)
        else:
            self.total_samples = info.frames
        
        # Try memory mapping for large files (>100MB estimated)
        file_size = os.path.getsize(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        
        # Use mmap for WAV files > 100MB and no resampling needed
        if ext == '.wav' and file_size > 100_000_000 and (target_sr is None or target_sr == info.samplerate):
            try:
                self._mmap_loader.load_file(file_path, target_sr)
                self._use_mmap = True
                logger.info(f"Using memory-mapped loading for large file: {file_size / 1e6:.1f} MB")
            except Exception as e:
                logger.warning(f"Memory mapping failed, using standard loader: {e}")
                self._use_mmap = False
        else:
            self._use_mmap = False
            
        return self.sample_rate, self.duration
    
    def get_chunk(self, start_sample: int, num_samples: int) -> np.ndarray:
        """Get audio chunk with caching for overlapping requests.
        
        PERFORMANCE:
        - Uses memory mapping for instant access (large WAV files)
        - LRU cache to minimize disk reads
        - Returns view when possible to avoid copies
        
        Args:
            start_sample: Starting sample index
            num_samples: Number of samples to retrieve
            
        Returns:
            Audio data as float32 numpy array
        """
        # Use memory-mapped loader if available (FAST path)
        if self._use_mmap:
            return self._mmap_loader.get_samples(start_sample, num_samples)
        
        # Standard path with caching
        chunk_id = (start_sample, num_samples)
        
        # Check cache (LRU)
        if chunk_id in self._chunk_cache:
            # Move to end of LRU list
            if chunk_id in self._cache_order:
                self._cache_order.remove(chunk_id)
            self._cache_order.append(chunk_id)
            return self._chunk_cache[chunk_id].copy()
        
        if self._file_path is None:
            raise ValueError("No audio file loaded")
        
        try:
            # Use soundfile for faster loading than librosa when no resampling
            try:
                audio, sr = sf.read(
                    self._file_path,
                    start=start_sample,
                    stop=start_sample + num_samples,
                    dtype='float32',
                    always_2d=False
                )
                
                # Handle multi-channel
                if audio.ndim > 1:
                    audio = audio[:, 0]  # Take first channel
                    
            except Exception:
                # Fallback to librosa (handles resampling, more formats)
                audio, sr = librosa.load(
                    self._file_path,
                    sr=self.sample_rate,
                    offset=start_sample / self.sample_rate,
                    duration=num_samples / self.sample_rate,
                    mono=True
                )
            
            # Ensure correct length
            if len(audio) < num_samples:
                audio = np.pad(audio, (0, num_samples - len(audio)), mode='constant')
            elif len(audio) > num_samples:
                audio = audio[:num_samples]
            
            # Ensure float32
            audio = audio.astype(np.float32)
            
            # Add to LRU cache
            self._chunk_cache[chunk_id] = audio.copy()
            self._cache_order.append(chunk_id)
            
            # Evict oldest if cache is full
            while len(self._chunk_cache) > self._max_cache_chunks:
                oldest_key = self._cache_order.pop(0)
                if oldest_key in self._chunk_cache:
                    del self._chunk_cache[oldest_key]
                
            return audio
            
        except Exception as e:
            logger.error(f"Error loading chunk: {e}")
            return np.zeros(num_samples, dtype=np.float32)
    
    def stream_chunks(self, chunk_duration: float = 1.0) -> Generator[np.ndarray, None, None]:
        """Stream audio in chunks for real-time processing."""
        samples_per_chunk = int(chunk_duration * self.sample_rate)
        current_sample = 0
        
        while current_sample < self.total_samples:
            chunk_samples = min(samples_per_chunk, self.total_samples - current_sample)
            chunk = self.get_chunk(current_sample, chunk_samples)
            yield chunk
            current_sample += chunk_samples - self.overlap
    
    def preload_range(self, start_time: float, end_time: float, 
                     callback=None) -> None:
        """Preload audio range in background thread."""
        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        total_samples = end_sample - start_sample
        
        def load_worker():
            try:
                audio = self.get_chunk(start_sample, total_samples)
                if callback:
                    callback(audio, start_time, end_time)
            except Exception as e:
                if callback:
                    callback(None, start_time, end_time, str(e))
        
        self._loading_thread = threading.Thread(target=load_worker, daemon=True)
        self._loading_thread.start()
    
    def clear_cache(self):
        """Clear chunk cache to free memory."""
        self._chunk_cache.clear()
        self._cache_order.clear()
        # gc.collect() - Removed to improve file switching speed
    
    def close(self):
        """Close file handles and clear cache."""
        self.clear_cache()
        if self._mmap_loader:
            self._mmap_loader.close()
        self._use_mmap = False
    
    def get_file_info(self) -> dict:
        """Get comprehensive file information."""
        if self._file_path is None:
            return {}
        
        return {
            'file_path': self._file_path,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'total_samples': self.total_samples,
            'cache_size': len(self._chunk_cache),
            'using_mmap': self._use_mmap
        }
    
    def __del__(self):
        """Cleanup on destruction."""
        self.close()