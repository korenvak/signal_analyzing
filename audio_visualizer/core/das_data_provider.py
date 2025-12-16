"""
DAS (Distributed Acoustic Sensing) Data Provider Interface.

This module defines the abstract interface for DAS data access.
Users implement this interface for their specific file format.

The visualization system uses this interface to:
1. Load metadata from a folder + JSON file
2. Query available sensor and time ranges
3. Fetch matrix chunks for display
4. Support async loading for large datasets
"""

from typing import Protocol, List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import numpy as np
import logging
import threading
from pathlib import Path

from .data_source import DASMetadata, DASDataSource

logger = logging.getLogger(__name__)


@dataclass
class DASDataRequest:
    """Request parameters for fetching a data chunk."""
    sensor_start: int
    sensor_end: int
    time_start: datetime
    time_end: datetime
    downsample_factor: int = 1  # 1 = full resolution, 2 = half, etc.

    @property
    def n_sensors(self) -> int:
        return self.sensor_end - self.sensor_start

    @property
    def duration_seconds(self) -> float:
        return (self.time_end - self.time_start).total_seconds()


@dataclass
class DASFileInfo:
    """Information about a single data file in the folder."""
    filename: str
    time_start: datetime
    time_end: datetime
    sensor_start: int
    sensor_end: int
    n_samples: int = 0
    file_size_bytes: int = 0


@dataclass
class DASFolderMetadata:
    """Complete metadata for a DAS data folder."""
    folder_path: str
    metadata_file: str
    sensor_range: Tuple[int, int]           # (min_sensor_id, max_sensor_id)
    time_ranges: List[Tuple[datetime, datetime]]  # Available ranges (gaps = missing)
    sample_rate: float                       # Samples per second
    total_samples: int = 0                   # Total time samples across all files
    sensor_spacing: Optional[float] = None   # Meters between sensors
    units: str = "phase_derivative"          # Data units description
    files: List[DASFileInfo] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_sensors(self) -> int:
        return self.sensor_range[1] - self.sensor_range[0]

    @property
    def total_duration_seconds(self) -> float:
        """Total duration of all available data (excluding gaps)."""
        total = 0.0
        for start, end in self.time_ranges:
            total += (end - start).total_seconds()
        return total

    def to_das_metadata(self) -> DASMetadata:
        """Convert to standard DASMetadata."""
        base_time = self.time_ranges[0][0] if self.time_ranges else None
        return DASMetadata(
            sample_rate=self.sample_rate,
            duration_seconds=self.total_duration_seconds,
            sensor_range=self.sensor_range,
            n_sensors=self.n_sensors,
            time_ranges=self.time_ranges,
            sensor_spacing=self.sensor_spacing,
            base_time=base_time,
            units=self.units,
            extra=self.extra
        )


class DASDataProvider(ABC):
    """
    Abstract base class for DAS data providers.

    Implement this class to load your specific DAS file format.
    The visualization system will use this interface to access data.

    Data shape convention: (n_time_samples, n_sensors)
    - Rows = time samples
    - Columns = sensor channels
    """

    @abstractmethod
    def load_folder(self, folder_path: str, metadata_path: str) -> DASFolderMetadata:
        """
        Load metadata and prepare for data access.

        This should:
        1. Parse the metadata JSON file
        2. Index available data files
        3. Build the time range list (identifying gaps)
        4. Return complete metadata

        Args:
            folder_path: Path to folder containing data files
            metadata_path: Path to JSON metadata file

        Returns:
            DASFolderMetadata with all available information

        Raises:
            FileNotFoundError: If folder or metadata file doesn't exist
            ValueError: If metadata is invalid
        """
        pass

    @abstractmethod
    def get_metadata(self) -> DASFolderMetadata:
        """Get currently loaded metadata."""
        pass

    @abstractmethod
    def get_data_chunk(self, request: DASDataRequest) -> np.ndarray:
        """
        Get matrix chunk for specified range.

        This is the main data access method. It should:
        1. Determine which files contain the requested range
        2. Read and concatenate data from those files
        3. Apply downsampling if requested
        4. Return the matrix

        Args:
            request: DASDataRequest with sensor/time ranges and downsample factor

        Returns:
            np.ndarray of shape (n_time_samples, n_sensors), dtype=float32

        Raises:
            ValueError: If requested range is invalid or has gaps
        """
        pass

    def get_data_chunk_async(
        self,
        request: DASDataRequest,
        callback: Callable[[Optional[np.ndarray], Optional[str]], None]
    ) -> None:
        """
        Async version for background loading.

        Default implementation runs get_data_chunk in a thread.
        Override for more efficient async implementation.

        Args:
            request: DASDataRequest
            callback: Function(data: Optional[np.ndarray], error: Optional[str])
        """
        def worker():
            try:
                data = self.get_data_chunk(request)
                callback(data, None)
            except Exception as e:
                logger.error(f"Async data load failed: {e}")
                callback(None, str(e))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    @abstractmethod
    def is_range_available(
        self,
        sensor_start: int,
        sensor_end: int,
        time_start: datetime,
        time_end: datetime
    ) -> bool:
        """
        Check if requested range has continuous data (no gaps).

        Args:
            sensor_start: Start sensor ID
            sensor_end: End sensor ID
            time_start: Start time
            time_end: End time

        Returns:
            True if data is available for the entire range without gaps
        """
        pass

    def get_single_sensor(
        self,
        sensor_id: int,
        time_start: datetime,
        time_end: datetime,
        downsample: int = 1
    ) -> np.ndarray:
        """
        Get data for a single sensor (for spectrogram view).

        Default implementation uses get_data_chunk.
        Override for more efficient single-sensor access.

        Args:
            sensor_id: Sensor ID to extract
            time_start: Start time
            time_end: End time
            downsample: Downsampling factor

        Returns:
            1D array of sensor data as float32
        """
        request = DASDataRequest(
            sensor_start=sensor_id,
            sensor_end=sensor_id + 1,
            time_start=time_start,
            time_end=time_end,
            downsample_factor=downsample
        )
        data = self.get_data_chunk(request)
        return data[:, 0]  # Extract single column

    def close(self) -> None:
        """Release resources. Override if cleanup is needed."""
        pass


class MockDASDataProvider(DASDataProvider):
    """
    Mock provider for UI testing without real data.

    Generates synthetic test patterns for development and testing.
    """

    def __init__(self):
        self._metadata: Optional[DASFolderMetadata] = None
        self._sample_rate = 1000.0  # 1 kHz

    def load_folder(self, folder_path: str, metadata_path: str) -> DASFolderMetadata:
        """Load mock metadata (ignores actual files)."""
        logger.info(f"MockDASDataProvider: Loading mock data for {folder_path}")

        # Create mock metadata with some gaps
        base_time = datetime(2024, 1, 15, 10, 0, 0)

        self._metadata = DASFolderMetadata(
            folder_path=folder_path,
            metadata_file=metadata_path,
            sensor_range=(0, 2048),
            time_ranges=[
                # First continuous segment: 30 minutes
                (base_time, base_time + timedelta(minutes=30)),
                # Gap of 15 minutes
                # Second segment: 45 minutes
                (base_time + timedelta(minutes=45), base_time + timedelta(minutes=90)),
            ],
            sample_rate=self._sample_rate,
            total_samples=int(75 * 60 * self._sample_rate),  # 75 minutes of data
            sensor_spacing=1.02,  # ~1 meter between sensors
            units="rad/s",
            files=[
                DASFileInfo(
                    filename="mock_segment_1.bin",
                    time_start=base_time,
                    time_end=base_time + timedelta(minutes=30),
                    sensor_start=0,
                    sensor_end=2048,
                    n_samples=int(30 * 60 * self._sample_rate)
                ),
                DASFileInfo(
                    filename="mock_segment_2.bin",
                    time_start=base_time + timedelta(minutes=45),
                    time_end=base_time + timedelta(minutes=90),
                    sensor_start=0,
                    sensor_end=2048,
                    n_samples=int(45 * 60 * self._sample_rate)
                ),
            ],
            extra={"mock": True, "description": "Mock data for testing"}
        )

        logger.info(f"MockDASDataProvider: Loaded {self._metadata.n_sensors} sensors, "
                   f"{self._metadata.total_duration_seconds:.0f}s of data")

        return self._metadata

    def get_metadata(self) -> DASFolderMetadata:
        """Get mock metadata."""
        if self._metadata is None:
            raise ValueError("No data loaded. Call load_folder first.")
        return self._metadata

    def get_data_chunk(self, request: DASDataRequest) -> np.ndarray:
        """Generate synthetic test data."""
        if self._metadata is None:
            raise ValueError("No data loaded. Call load_folder first.")

        n_sensors = request.sensor_end - request.sensor_start
        duration = request.duration_seconds
        n_samples = int(duration * self._sample_rate / request.downsample_factor)

        logger.debug(f"MockDASDataProvider: Generating {n_samples} x {n_sensors} chunk")

        # Generate test pattern
        data = np.zeros((n_samples, n_sensors), dtype=np.float32)

        # Time vector
        t = np.linspace(0, duration, n_samples, dtype=np.float32)

        # Create interesting patterns for visualization testing
        for i in range(n_sensors):
            sensor_id = request.sensor_start + i

            # Base: low-frequency oscillation varying with sensor position
            freq = 0.5 + 0.01 * sensor_id  # Slight frequency gradient
            data[:, i] = np.sin(2 * np.pi * freq * t + sensor_id * 0.1)

            # Add a "signal" moving across sensors (simulating a wave)
            wave_speed = 200  # sensors per second
            wave_time = sensor_id / wave_speed
            signal_mask = np.abs(t - wave_time - 1.0) < 0.5
            data[:, i] += 2.0 * signal_mask * np.sin(2 * np.pi * 10 * t)

            # Add noise
            data[:, i] += 0.1 * np.random.randn(n_samples).astype(np.float32)

        return data

    def is_range_available(
        self,
        sensor_start: int,
        sensor_end: int,
        time_start: datetime,
        time_end: datetime
    ) -> bool:
        """Check if range is available (mock always returns True for valid ranges)."""
        if self._metadata is None:
            return False

        # Check sensor range
        if sensor_start < self._metadata.sensor_range[0]:
            return False
        if sensor_end > self._metadata.sensor_range[1]:
            return False

        # Check time range against available segments
        for seg_start, seg_end in self._metadata.time_ranges:
            if time_start >= seg_start and time_end <= seg_end:
                return True

        return False


# Global provider instance (can be replaced with real implementation)
_das_provider: Optional[DASDataProvider] = None


def get_das_provider() -> Optional[DASDataProvider]:
    """Get the current DAS data provider."""
    return _das_provider


def set_das_provider(provider: DASDataProvider) -> None:
    """Set the DAS data provider."""
    global _das_provider
    _das_provider = provider
    logger.info(f"DAS provider set: {type(provider).__name__}")


def create_mock_provider() -> MockDASDataProvider:
    """Create and register a mock provider for testing."""
    provider = MockDASDataProvider()
    set_das_provider(provider)
    return provider
