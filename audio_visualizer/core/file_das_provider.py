"""
File-based DAS Data Provider.

Loads DAS data from files created by SyntheticDASGenerator or similar format.
Supports:
- Memory-mapped file access for large datasets
- Chunked loading to manage memory
- GPU acceleration for processing
- Progress callbacks for long operations
"""

import numpy as np
import json
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass
import threading
import time

from .das_data_provider import (
    DASDataProvider, DASFolderMetadata, DASDataRequest, DASFileInfo
)

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Memory management configuration."""
    max_chunk_mb: float = 500.0      # Max memory for a single chunk
    use_mmap: bool = True            # Use memory-mapped files
    use_gpu: bool = True             # Use GPU if available
    gpu_memory_limit_mb: float = 2000.0  # Max GPU memory to use
    prefetch_chunks: int = 2         # Number of chunks to prefetch
    cache_enabled: bool = True       # Enable LRU cache


class FileDASProvider(DASDataProvider):
    """
    File-based DAS data provider with memory management.

    Loads data from .npy or .bin files with JSON metadata.
    Optimized for very large datasets that don't fit in memory.
    """

    def __init__(self, memory_config: Optional[MemoryConfig] = None):
        """
        Initialize provider.

        Args:
            memory_config: Memory management settings
        """
        self._config = memory_config or MemoryConfig()
        self._metadata: Optional[DASFolderMetadata] = None
        self._folder_path: Optional[Path] = None
        self._raw_metadata: Optional[Dict[str, Any]] = None

        # File handles for memory-mapped access
        self._mmap_handles: Dict[str, np.memmap] = {}

        # Cache for recently accessed chunks
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_order: List[str] = []
        self._cache_max_items = 10

        # Progress callback
        self._progress_callback: Optional[Callable[[float, str], None]] = None

        # GPU availability
        self._has_gpu = False
        self._gpu_memory_used = 0
        try:
            import cupy as cp
            self._has_gpu = self._config.use_gpu
            self._cp = cp
        except ImportError:
            self._cp = None

        logger.info(f"FileDASProvider initialized (GPU: {self._has_gpu}, "
                   f"mmap: {self._config.use_mmap})")

    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set progress callback for long operations."""
        self._progress_callback = callback

    def _report_progress(self, progress: float, message: str):
        """Report progress if callback is set."""
        if self._progress_callback:
            self._progress_callback(progress, message)

    def load_folder(self, folder_path: str, metadata_path: str) -> DASFolderMetadata:
        """
        Load metadata and prepare for data access.

        Args:
            folder_path: Path to folder containing data files
            metadata_path: Path to JSON metadata file (absolute or relative to folder)

        Returns:
            DASFolderMetadata
        """
        self._folder_path = Path(folder_path)

        # Handle metadata path
        if os.path.isabs(metadata_path):
            meta_file = Path(metadata_path)
        else:
            meta_file = self._folder_path / metadata_path

        if not meta_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_file}")

        # Load JSON metadata
        logger.info(f"Loading metadata from {meta_file}")
        with open(meta_file, 'r') as f:
            self._raw_metadata = json.load(f)

        # Parse metadata
        self._metadata = self._parse_metadata(self._raw_metadata)

        # Verify files exist
        self._verify_files()

        # Open memory-mapped file handles if enabled
        if self._config.use_mmap:
            self._open_mmap_handles()

        logger.info(f"Loaded {self._metadata.n_sensors} sensors, "
                   f"{self._metadata.total_duration_seconds:.1f}s of data, "
                   f"{len(self._metadata.files)} files")

        return self._metadata

    def _parse_metadata(self, raw: Dict[str, Any]) -> DASFolderMetadata:
        """Parse raw JSON metadata into DASFolderMetadata."""

        # Parse time ranges
        time_ranges = []
        for tr in raw.get("time_ranges", []):
            start = datetime.fromisoformat(tr[0])
            end = datetime.fromisoformat(tr[1])
            time_ranges.append((start, end))

        # Parse file list
        files = []
        for fi in raw.get("files", []):
            files.append(DASFileInfo(
                filename=fi["filename"],
                time_start=datetime.fromisoformat(fi["time_start"]),
                time_end=datetime.fromisoformat(fi["time_end"]),
                sensor_start=fi.get("sensor_start", 0),
                sensor_end=fi.get("sensor_end", raw.get("n_sensors", 0)),
                n_samples=fi.get("n_samples", 0),
                file_size_bytes=fi.get("file_size_bytes", 0)
            ))

        return DASFolderMetadata(
            folder_path=str(self._folder_path),
            metadata_file=str(self._folder_path / "metadata.json"),
            sensor_range=(raw.get("sensor_range", [0, 0])[0],
                         raw.get("sensor_range", [0, 0])[1]),
            time_ranges=time_ranges,
            sample_rate=raw.get("sample_rate", 1000.0),
            total_samples=raw.get("total_samples", 0),
            sensor_spacing=raw.get("sensor_spacing_m"),
            units=raw.get("units", "unknown"),
            files=files,
            extra=raw
        )

    def _verify_files(self):
        """Verify all data files exist."""
        missing = []
        for file_info in self._metadata.files:
            filepath = self._folder_path / file_info.filename
            if not filepath.exists():
                missing.append(file_info.filename)

        if missing:
            raise FileNotFoundError(
                f"Missing data files: {', '.join(missing[:5])}"
                + (f"... and {len(missing)-5} more" if len(missing) > 5 else "")
            )

    def _open_mmap_handles(self):
        """Open memory-mapped file handles for all data files."""
        for file_info in self._metadata.files:
            filepath = self._folder_path / file_info.filename

            try:
                if filepath.suffix == '.npy':
                    # NumPy format
                    mmap = np.load(str(filepath), mmap_mode='r')
                else:
                    # Binary format - need shape from metadata
                    shape = self._get_file_shape(file_info)
                    mmap = np.memmap(
                        str(filepath), dtype=np.float32, mode='r', shape=shape
                    )

                self._mmap_handles[file_info.filename] = mmap
                logger.debug(f"Opened mmap for {file_info.filename}, shape={mmap.shape}")

            except Exception as e:
                logger.warning(f"Could not memory-map {file_info.filename}: {e}")

    def _get_file_shape(self, file_info: DASFileInfo) -> Tuple[int, int]:
        """Get shape of a data file."""
        # Try to get from raw metadata
        for fi in self._raw_metadata.get("files", []):
            if fi["filename"] == file_info.filename:
                if "shape" in fi:
                    return tuple(fi["shape"])

        # Calculate from file info
        n_sensors = file_info.sensor_end - file_info.sensor_start
        n_samples = file_info.n_samples
        return (n_samples, n_sensors)

    def get_metadata(self) -> DASFolderMetadata:
        """Get currently loaded metadata."""
        if self._metadata is None:
            raise ValueError("No data loaded. Call load_folder first.")
        return self._metadata

    def get_data_chunk(self, request: DASDataRequest) -> np.ndarray:
        """
        Get matrix chunk for specified range.

        Args:
            request: DASDataRequest with sensor/time ranges

        Returns:
            np.ndarray of shape (n_time_samples, n_sensors), dtype=float32
        """
        if self._metadata is None:
            raise ValueError("No data loaded. Call load_folder first.")

        # Find which files contain the requested range
        files_to_read = self._find_files_for_range(request)

        if not files_to_read:
            raise ValueError(f"No data available for requested time range: "
                           f"{request.time_start} to {request.time_end}")

        # Calculate output size
        total_duration = (request.time_end - request.time_start).total_seconds()
        n_samples = int(total_duration * self._metadata.sample_rate / request.downsample_factor)
        n_sensors = request.sensor_end - request.sensor_start

        # Check memory requirements
        required_mb = (n_samples * n_sensors * 4) / (1024 * 1024)
        if required_mb > self._config.max_chunk_mb:
            logger.warning(f"Requested chunk ({required_mb:.1f} MB) exceeds limit "
                          f"({self._config.max_chunk_mb:.1f} MB)")
            # Could implement chunked reading here

        # Allocate output array
        output = np.zeros((n_samples, n_sensors), dtype=np.float32)

        # Read from each file
        output_offset = 0
        for file_info, time_offset, samples_to_read in files_to_read:
            self._report_progress(
                output_offset / n_samples,
                f"Reading {file_info.filename}..."
            )

            chunk = self._read_file_chunk(
                file_info,
                sensor_start=request.sensor_start,
                sensor_end=request.sensor_end,
                sample_offset=time_offset,
                n_samples=samples_to_read,
                downsample=request.downsample_factor
            )

            # Copy to output
            chunk_output_samples = len(chunk)
            output[output_offset:output_offset + chunk_output_samples, :] = chunk
            output_offset += chunk_output_samples

        self._report_progress(1.0, "Data loaded")
        return output

    def _find_files_for_range(
        self, request: DASDataRequest
    ) -> List[Tuple[DASFileInfo, int, int]]:
        """
        Find which files contain data for the requested range.

        Returns:
            List of (file_info, sample_offset_in_file, n_samples_to_read)
        """
        files_to_read = []
        sample_rate = self._metadata.sample_rate

        for file_info in self._metadata.files:
            # Check if file overlaps with requested time range
            if file_info.time_end <= request.time_start:
                continue
            if file_info.time_start >= request.time_end:
                continue

            # Calculate the overlap
            read_start = max(file_info.time_start, request.time_start)
            read_end = min(file_info.time_end, request.time_end)

            # Convert to sample offsets within the file
            file_start_offset = (read_start - file_info.time_start).total_seconds()
            sample_offset = int(file_start_offset * sample_rate)

            duration = (read_end - read_start).total_seconds()
            n_samples = int(duration * sample_rate)

            if n_samples > 0:
                files_to_read.append((file_info, sample_offset, n_samples))

        return files_to_read

    def _read_file_chunk(
        self,
        file_info: DASFileInfo,
        sensor_start: int,
        sensor_end: int,
        sample_offset: int,
        n_samples: int,
        downsample: int = 1
    ) -> np.ndarray:
        """Read a chunk from a single file."""

        # Check cache first
        cache_key = f"{file_info.filename}:{sensor_start}:{sensor_end}:{sample_offset}:{n_samples}:{downsample}"
        if self._config.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]

        # Adjust for sensor offset in file
        local_sensor_start = sensor_start - file_info.sensor_start
        local_sensor_end = sensor_end - file_info.sensor_start

        # Use memory-mapped handle if available
        if file_info.filename in self._mmap_handles:
            mmap = self._mmap_handles[file_info.filename]

            # Slice the data
            end_sample = min(sample_offset + n_samples, mmap.shape[0])
            chunk = mmap[sample_offset:end_sample, local_sensor_start:local_sensor_end]

            # Convert to regular array (copies data out of mmap)
            chunk = np.array(chunk, dtype=np.float32)

        else:
            # Load from file directly
            filepath = self._folder_path / file_info.filename

            if filepath.suffix == '.npy':
                # NumPy format
                full_data = np.load(str(filepath))
                end_sample = min(sample_offset + n_samples, full_data.shape[0])
                chunk = full_data[sample_offset:end_sample, local_sensor_start:local_sensor_end]
                chunk = chunk.astype(np.float32)
            else:
                # Binary format
                shape = self._get_file_shape(file_info)
                full_data = np.fromfile(str(filepath), dtype=np.float32).reshape(shape)
                end_sample = min(sample_offset + n_samples, full_data.shape[0])
                chunk = full_data[sample_offset:end_sample, local_sensor_start:local_sensor_end]

        # Downsample if requested
        if downsample > 1:
            chunk = self._downsample(chunk, downsample)

        # Cache the result
        if self._config.cache_enabled:
            self._add_to_cache(cache_key, chunk)

        return chunk

    def _downsample(self, data: np.ndarray, factor: int) -> np.ndarray:
        """Downsample data by averaging."""
        if factor <= 1:
            return data

        n_samples = data.shape[0]
        n_out = n_samples // factor

        # Reshape and average
        truncated = data[:n_out * factor]
        reshaped = truncated.reshape(n_out, factor, -1)
        return reshaped.mean(axis=1).astype(np.float32)

    def _add_to_cache(self, key: str, data: np.ndarray):
        """Add item to cache with LRU eviction."""
        if key in self._cache:
            self._cache_order.remove(key)
            self._cache_order.append(key)
            return

        # Evict old items if cache is full
        while len(self._cache) >= self._cache_max_items:
            old_key = self._cache_order.pop(0)
            del self._cache[old_key]

        self._cache[key] = data
        self._cache_order.append(key)

    def is_range_available(
        self,
        sensor_start: int,
        sensor_end: int,
        time_start: datetime,
        time_end: datetime
    ) -> bool:
        """Check if requested range has continuous data."""
        if self._metadata is None:
            return False

        # Check sensor range
        if sensor_start < self._metadata.sensor_range[0]:
            return False
        if sensor_end > self._metadata.sensor_range[1]:
            return False

        # Check time range
        for seg_start, seg_end in self._metadata.time_ranges:
            if time_start >= seg_start and time_end <= seg_end:
                return True

        return False

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        stats = {
            "cache_items": len(self._cache),
            "cache_mb": sum(arr.nbytes for arr in self._cache.values()) / (1024 * 1024),
            "mmap_files": len(self._mmap_handles),
            "gpu_available": self._has_gpu,
            "gpu_memory_mb": self._gpu_memory_used / (1024 * 1024) if self._has_gpu else 0
        }
        return stats

    def clear_cache(self):
        """Clear the data cache."""
        self._cache.clear()
        self._cache_order.clear()
        logger.info("Cache cleared")

    def close(self):
        """Release resources."""
        # Close memory-mapped handles
        for name, mmap in self._mmap_handles.items():
            try:
                del mmap
            except Exception:
                pass
        self._mmap_handles.clear()

        # Clear cache
        self.clear_cache()

        logger.info("FileDASProvider closed")


def load_das_folder(folder_path: str, metadata_file: str = "metadata.json") -> FileDASProvider:
    """
    Convenience function to load a DAS data folder.

    Args:
        folder_path: Path to the data folder
        metadata_file: Name of the metadata JSON file

    Returns:
        Configured FileDASProvider
    """
    provider = FileDASProvider()
    provider.load_folder(folder_path, metadata_file)
    return provider
