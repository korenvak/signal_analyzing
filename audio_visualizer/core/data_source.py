"""
Abstract data source protocols for different input types.

This module defines the abstract interfaces that all data sources must implement,
allowing the visualization system to work with different data types:
- Audio files (current single-channel spectrogram)
- DAS matrix data (multi-channel waterfall)
- Future: real-time streams, remote data, etc.
"""

from typing import Protocol, Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime


class DataSourceType(Enum):
    """Type of data source."""
    AUDIO = "audio"           # Single-channel audio -> spectrogram
    DAS_MATRIX = "das_matrix" # Multi-channel DAS -> waterfall


class CoordinateSystem(Enum):
    """Coordinate system for visualization."""
    SPECTROGRAM = "spectrogram"  # X=time(s), Y=frequency(Hz)
    WATERFALL = "waterfall"      # X=sensor(int), Y=time(HH:MM:SS)


@dataclass
class DataSourceMetadata:
    """Base metadata for any data source."""
    source_type: DataSourceType
    sample_rate: float
    duration_seconds: float
    dtype: np.dtype = field(default_factory=lambda: np.dtype('float32'))
    units: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioMetadata(DataSourceMetadata):
    """Metadata for audio data sources."""
    n_channels: int = 1
    total_samples: int = 0
    file_path: str = ""

    def __post_init__(self):
        self.source_type = DataSourceType.AUDIO


@dataclass
class DASMetadata(DataSourceMetadata):
    """Metadata for DAS matrix data sources."""
    sensor_range: Tuple[int, int] = (0, 0)  # (min_id, max_id)
    n_sensors: int = 0
    time_ranges: List[Tuple[datetime, datetime]] = field(default_factory=list)  # Available ranges (gaps = missing)
    sensor_spacing: Optional[float] = None  # Meters between sensors
    base_time: Optional[datetime] = None    # Reference time for sample 0

    def __post_init__(self):
        self.source_type = DataSourceType.DAS_MATRIX
        if self.sensor_range[1] > self.sensor_range[0]:
            self.n_sensors = self.sensor_range[1] - self.sensor_range[0]


class DataSource(Protocol):
    """
    Abstract protocol for all data sources.

    This defines the minimum interface that any data source must implement
    to work with the visualization system.
    """

    @property
    def metadata(self) -> DataSourceMetadata:
        """Get source metadata."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Check if data source is loaded and ready."""
        ...

    def get_chunk(
        self,
        start_idx: int,
        end_idx: int,
        **kwargs
    ) -> np.ndarray:
        """
        Get a chunk of data by index.

        Args:
            start_idx: Start index
            end_idx: End index
            **kwargs: Additional source-specific parameters

        Returns:
            Data chunk as numpy array
        """
        ...

    def close(self) -> None:
        """Release resources."""
        ...


class AudioDataSource(Protocol):
    """
    Protocol for audio data sources (single-channel spectrogram).
    Extends base DataSource with audio-specific methods.
    """

    @property
    def metadata(self) -> AudioMetadata:
        """Get audio metadata."""
        ...

    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""
        ...

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        ...

    @property
    def total_samples(self) -> int:
        """Total number of samples."""
        ...

    def get_chunk(
        self,
        start_sample: int,
        num_samples: int,
        as_mono: bool = True
    ) -> np.ndarray:
        """
        Get audio chunk by sample indices.

        Args:
            start_sample: Start sample index
            num_samples: Number of samples to read
            as_mono: Convert to mono if multi-channel

        Returns:
            Audio data as float32 array
        """
        ...

    def get_time_range(
        self,
        start_time: float,
        end_time: float,
        as_mono: bool = True
    ) -> np.ndarray:
        """
        Get audio chunk by time range.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            as_mono: Convert to mono if multi-channel

        Returns:
            Audio data as float32 array
        """
        ...


class DASDataSource(Protocol):
    """
    Protocol for DAS matrix data sources (multi-channel waterfall).

    Data shape convention: (n_time_samples, n_sensors)
    - Rows = time samples
    - Columns = sensor channels
    """

    @property
    def metadata(self) -> DASMetadata:
        """Get DAS metadata."""
        ...

    @property
    def sensor_range(self) -> Tuple[int, int]:
        """Available sensor ID range (min, max)."""
        ...

    @property
    def n_sensors(self) -> int:
        """Number of sensors."""
        ...

    @property
    def sample_rate(self) -> float:
        """Sample rate in Hz."""
        ...

    @property
    def time_ranges(self) -> List[Tuple[datetime, datetime]]:
        """List of available time ranges (gaps = discontinuities)."""
        ...

    def get_chunk(
        self,
        sensor_start: int,
        sensor_end: int,
        time_start_idx: int,
        time_end_idx: int,
        downsample: int = 1
    ) -> np.ndarray:
        """
        Get matrix chunk for specified sensor and time range.

        Args:
            sensor_start: Start sensor ID (inclusive)
            sensor_end: End sensor ID (exclusive)
            time_start_idx: Start time sample index
            time_end_idx: End time sample index
            downsample: Downsampling factor (1 = full resolution)

        Returns:
            Matrix of shape (n_time_samples, n_sensors) as float32
        """
        ...

    def get_chunk_by_time(
        self,
        sensor_start: int,
        sensor_end: int,
        time_start: datetime,
        time_end: datetime,
        downsample: int = 1
    ) -> np.ndarray:
        """
        Get matrix chunk using datetime for time range.

        Args:
            sensor_start: Start sensor ID (inclusive)
            sensor_end: End sensor ID (exclusive)
            time_start: Start datetime
            time_end: End datetime
            downsample: Downsampling factor

        Returns:
            Matrix of shape (n_time_samples, n_sensors) as float32
        """
        ...

    def get_single_sensor(
        self,
        sensor_id: int,
        time_start_idx: int,
        time_end_idx: int
    ) -> np.ndarray:
        """
        Get data for a single sensor (for spectrogram view).

        Args:
            sensor_id: Sensor ID to extract
            time_start_idx: Start time sample index
            time_end_idx: End time sample index

        Returns:
            1D array of sensor data as float32
        """
        ...

    def is_range_available(
        self,
        sensor_start: int,
        sensor_end: int,
        time_start: datetime,
        time_end: datetime
    ) -> bool:
        """
        Check if requested range has continuous data (no gaps).

        Returns:
            True if data is available for entire range
        """
        ...

    def get_chunk_async(
        self,
        sensor_start: int,
        sensor_end: int,
        time_start_idx: int,
        time_end_idx: int,
        callback: callable,
        downsample: int = 1
    ) -> None:
        """
        Async version for background loading.

        Args:
            callback: Function(data: np.ndarray, error: Optional[str])
        """
        ...


# Type alias for any data source
AnyDataSource = DataSource | AudioDataSource | DASDataSource
