"""
Compressed Tile Storage System
Provides 3-4x more cache capacity by compressing tiles in memory and on disk.
Uses ultra-fast compression (LZ4/Blosc) with minimal CPU overhead.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple
import pickle
import threading

logger = logging.getLogger(__name__)

# Try to import compression libraries in order of preference
COMPRESSION_AVAILABLE = False
COMPRESSION_BACKEND = "none"

try:
    import blosc
    COMPRESSION_AVAILABLE = True
    COMPRESSION_BACKEND = "blosc"
    logger.info("Using Blosc compression for tile storage")
except ImportError:
    try:
        import lz4.frame
        COMPRESSION_AVAILABLE = True
        COMPRESSION_BACKEND = "lz4"
        logger.info("Using LZ4 compression for tile storage")
    except ImportError:
        try:
            import zlib
            COMPRESSION_AVAILABLE = True
            COMPRESSION_BACKEND = "zlib"
            logger.info("Using zlib compression for tile storage")
        except ImportError:
            logger.warning("No compression library available - tiles will not be compressed")


class CompressedTileStorage:
    """High-performance compressed storage for tiles."""
    
    def __init__(self, compression_level: int = 3, enable_compression: bool = True):
        """Initialize compressed tile storage.
        
        Args:
            compression_level: Compression level (1=fast, 9=best, 3=balanced)
            enable_compression: Whether to enable compression
        """
        self.compression_level = compression_level
        self.enable_compression = enable_compression and COMPRESSION_AVAILABLE
        self.backend = COMPRESSION_BACKEND if self.enable_compression else "none"
        
        # Statistics
        self.stats = {
            'tiles_stored': 0,
            'tiles_loaded': 0,
            'bytes_saved': 0,
            'compression_ratio': 0.0,
            'compression_time_ms': 0.0,
            'decompression_time_ms': 0.0
        }
        
        # Thread-local storage for compression contexts (if supported)
        self._local = threading.local()
        
        logger.info(f"Compressed tile storage initialized: backend={self.backend}, "
                   f"level={compression_level}, enabled={self.enable_compression}")
    
    def compress_tile(self, tile_data: np.ndarray) -> Tuple[bytes, Dict[str, Any]]:
        """Compress a tile for storage.
        
        Args:
            tile_data: Numpy array to compress
            
        Returns:
            Tuple of (compressed_bytes, metadata)
        """
        import time
        start_time = time.perf_counter()
        
        # Prepare metadata
        metadata = {
            'shape': tile_data.shape,
            'dtype': str(tile_data.dtype),
            'compression': self.backend,
            'level': self.compression_level,
            'original_size': tile_data.nbytes
        }
        
        if not self.enable_compression:
            # No compression - just serialize
            compressed_data = pickle.dumps(tile_data)
            metadata['compressed_size'] = len(compressed_data)
            metadata['ratio'] = 1.0
        else:
            # Compress the data
            try:
                if self.backend == "blosc":
                    compressed_data = self._compress_blosc(tile_data)
                elif self.backend == "lz4":
                    compressed_data = self._compress_lz4(tile_data)
                elif self.backend == "zlib":
                    compressed_data = self._compress_zlib(tile_data)
                else:
                    # Fallback to no compression
                    compressed_data = pickle.dumps(tile_data)
                
                metadata['compressed_size'] = len(compressed_data)
                metadata['ratio'] = metadata['original_size'] / len(compressed_data)
                
            except Exception as e:
                logger.warning(f"Compression failed, using uncompressed: {e}")
                compressed_data = pickle.dumps(tile_data)
                metadata['compression'] = 'none'
                metadata['compressed_size'] = len(compressed_data)
                metadata['ratio'] = 1.0
        
        # Update statistics
        compression_time = (time.perf_counter() - start_time) * 1000
        self.stats['tiles_stored'] += 1
        self.stats['bytes_saved'] += metadata['original_size'] - metadata['compressed_size']
        self.stats['compression_time_ms'] += compression_time
        self.stats['compression_ratio'] = (self.stats['compression_ratio'] * 
                                          (self.stats['tiles_stored'] - 1) + 
                                          metadata['ratio']) / self.stats['tiles_stored']
        
        return compressed_data, metadata
    
    def decompress_tile(self, compressed_data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """Decompress a tile from storage.
        
        Args:
            compressed_data: Compressed tile bytes
            metadata: Tile metadata including shape, dtype, compression info
            
        Returns:
            Decompressed numpy array
        """
        import time
        start_time = time.perf_counter()
        
        try:
            compression_type = metadata.get('compression', 'none')
            
            if compression_type == 'none':
                # No compression - just deserialize
                tile_data = pickle.loads(compressed_data)
            elif compression_type == 'blosc':
                tile_data = self._decompress_blosc(compressed_data, metadata)
            elif compression_type == 'lz4':
                tile_data = self._decompress_lz4(compressed_data, metadata)
            elif compression_type == 'zlib':
                tile_data = self._decompress_zlib(compressed_data, metadata)
            else:
                # Unknown compression, try pickle
                tile_data = pickle.loads(compressed_data)
            
            # Ensure correct shape and dtype
            expected_shape = tuple(metadata['shape'])
            expected_dtype = np.dtype(metadata['dtype'])
            
            if tile_data.shape != expected_shape:
                tile_data = tile_data.reshape(expected_shape)
            if tile_data.dtype != expected_dtype:
                tile_data = tile_data.astype(expected_dtype)
            
            # Update statistics
            decompression_time = (time.perf_counter() - start_time) * 1000
            self.stats['tiles_loaded'] += 1
            self.stats['decompression_time_ms'] += decompression_time
            
            return tile_data
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise
    
    def _compress_blosc(self, data: np.ndarray) -> bytes:
        """Compress using Blosc (fastest, best compression)."""
        # Blosc works best with contiguous arrays
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)
        
        # Use LZ4 codec with specified compression level
        compressed = blosc.compress(data.tobytes(), 
                                   typesize=data.dtype.itemsize,
                                   clevel=self.compression_level,
                                   cname='lz4',
                                   shuffle=blosc.SHUFFLE)
        
        # Add array metadata
        metadata_bytes = pickle.dumps({
            'shape': data.shape,
            'dtype': str(data.dtype)
        })
        
        # Combine metadata length + metadata + compressed data
        metadata_len = len(metadata_bytes)
        return (metadata_len.to_bytes(4, 'little') + 
                metadata_bytes + compressed)
    
    def _decompress_blosc(self, compressed_data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """Decompress using Blosc."""
        # Extract metadata length
        metadata_len = int.from_bytes(compressed_data[:4], 'little')
        
        # Skip metadata (we have it already) and decompress
        compressed_array_data = compressed_data[4 + metadata_len:]
        decompressed_bytes = blosc.decompress(compressed_array_data)
        
        # Reconstruct array
        shape = tuple(metadata['shape'])
        dtype = np.dtype(metadata['dtype'])
        return np.frombuffer(decompressed_bytes, dtype=dtype).reshape(shape)
    
    def _compress_lz4(self, data: np.ndarray) -> bytes:
        """Compress using LZ4."""
        # Serialize array with pickle first, then compress
        serialized = pickle.dumps(data)
        return lz4.frame.compress(serialized, compression_level=self.compression_level)
    
    def _decompress_lz4(self, compressed_data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """Decompress using LZ4."""
        decompressed = lz4.frame.decompress(compressed_data)
        return pickle.loads(decompressed)
    
    def _compress_zlib(self, data: np.ndarray) -> bytes:
        """Compress using zlib (fallback)."""
        serialized = pickle.dumps(data)
        return zlib.compress(serialized, level=self.compression_level)
    
    def _decompress_zlib(self, compressed_data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """Decompress using zlib."""
        decompressed = zlib.decompress(compressed_data)
        return pickle.loads(decompressed)
    
    def estimate_memory_savings(self, uncompressed_size_mb: float) -> Dict[str, float]:
        """Estimate memory savings from compression.
        
        Args:
            uncompressed_size_mb: Size in MB without compression
            
        Returns:
            Dictionary with compression estimates
        """
        if not self.enable_compression:
            return {
                'uncompressed_mb': uncompressed_size_mb,
                'compressed_mb': uncompressed_size_mb,
                'savings_mb': 0.0,
                'ratio': 1.0,
                'extra_capacity_percent': 0.0
            }
        
        # Use current compression ratio or estimate
        ratio = self.stats.get('compression_ratio', 3.5)  # Conservative estimate
        
        compressed_mb = uncompressed_size_mb / ratio
        savings_mb = uncompressed_size_mb - compressed_mb
        extra_capacity_percent = ((ratio - 1) / ratio) * 100
        
        return {
            'uncompressed_mb': uncompressed_size_mb,
            'compressed_mb': compressed_mb,
            'savings_mb': savings_mb,
            'ratio': ratio,
            'extra_capacity_percent': extra_capacity_percent
        }
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get detailed compression statistics."""
        stats = self.stats.copy()
        stats.update({
            'backend': self.backend,
            'compression_level': self.compression_level,
            'compression_enabled': self.enable_compression,
            'avg_compression_time_ms': (stats['compression_time_ms'] / 
                                       max(1, stats['tiles_stored'])),
            'avg_decompression_time_ms': (stats['decompression_time_ms'] / 
                                         max(1, stats['tiles_loaded']))
        })
        return stats
    
    def reset_stats(self):
        """Reset compression statistics."""
        self.stats = {
            'tiles_stored': 0,
            'tiles_loaded': 0,
            'bytes_saved': 0,
            'compression_ratio': 0.0,
            'compression_time_ms': 0.0,
            'decompression_time_ms': 0.0
        }


# Global instance
_compressed_storage = None

def get_compressed_tile_storage(compression_level: int = 3, 
                               enable_compression: bool = True) -> CompressedTileStorage:
    """Get the global compressed tile storage instance."""
    global _compressed_storage
    if _compressed_storage is None:
        _compressed_storage = CompressedTileStorage(compression_level, enable_compression)
    return _compressed_storage