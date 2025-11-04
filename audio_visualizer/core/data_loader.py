import numpy as np
import soundfile as sf
import librosa
from typing import Optional, Tuple, Generator
import threading
import queue
import gc

class ChunkedAudioLoader:
    """High-performance chunked audio loader for large files with streaming support."""
    
    def __init__(self, chunk_size: int = 1024*1024, overlap: int = 4096):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.audio_data = None
        self.sample_rate = None
        self.duration = 0.0
        self._file_path = None
        self._loading_thread = None
        self._chunk_cache = {}
        
    def load_file(self, file_path: str, target_sr: Optional[int] = None) -> Tuple[int, float]:
        """Load audio file metadata and prepare for chunked reading."""
        self._file_path = file_path
        
        info = sf.info(file_path)
        self.sample_rate = target_sr or info.samplerate
        self.duration = info.duration
        
        if target_sr and target_sr != info.samplerate:
            self.total_samples = int(info.frames * target_sr / info.samplerate)
        else:
            self.total_samples = info.frames
            
        return self.sample_rate, self.duration
    
    def get_chunk(self, start_sample: int, num_samples: int) -> np.ndarray:
        """Get audio chunk with caching for overlapping requests."""
        chunk_id = (start_sample, num_samples)
        
        if chunk_id in self._chunk_cache:
            return self._chunk_cache[chunk_id].copy()
        
        if self._file_path is None:
            raise ValueError("No audio file loaded")
        
        try:
            audio, sr = librosa.load(
                self._file_path,
                sr=self.sample_rate,
                offset=start_sample / self.sample_rate,
                duration=num_samples / self.sample_rate,
                mono=True
            )
            
            if len(audio) < num_samples:
                audio = np.pad(audio, (0, num_samples - len(audio)), mode='constant')
            
            self._chunk_cache[chunk_id] = audio.copy()
            
            if len(self._chunk_cache) > 50:
                oldest_key = next(iter(self._chunk_cache))
                del self._chunk_cache[oldest_key]
                
            return audio
            
        except Exception as e:
            print(f"Error loading chunk: {e}")
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
        gc.collect()
    
    def get_file_info(self) -> dict:
        """Get comprehensive file information."""
        if self._file_path is None:
            return {}
        
        return {
            'file_path': self._file_path,
            'sample_rate': self.sample_rate,
            'duration': self.duration,
            'total_samples': self.total_samples,
            'cache_size': len(self._chunk_cache)
        }