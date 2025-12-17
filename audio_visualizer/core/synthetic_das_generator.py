"""
Synthetic DAS Data Generator.

Creates realistic DAS (Distributed Acoustic Sensing) test datasets
for validation and performance testing of the visualization system.

Generates:
1. Binary matrix files (.npy or .bin) with float32 data
2. JSON metadata files describing the dataset
3. Various test patterns (waves, events, noise)

Usage:
    from audio_visualizer.core.synthetic_das_generator import SyntheticDASGenerator

    gen = SyntheticDASGenerator(output_dir="test_das_data")
    gen.generate_dataset(
        n_sensors=4096,
        duration_seconds=300,  # 5 minutes
        sample_rate=1000,
        pattern="mixed"
    )
"""

import numpy as np
import json
import logging
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple, Callable, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class DataPattern(Enum):
    """Available test patterns for synthetic data."""
    NOISE = "noise"              # Pure random noise
    SINE_GRADIENT = "sine_gradient"  # Sinusoids with frequency varying by sensor
    TRAVELING_WAVE = "traveling_wave"  # Wave propagating across sensors
    EVENTS = "events"            # Discrete events (earthquakes, vehicles)
    MIXED = "mixed"              # Combination of patterns
    REALISTIC = "realistic"      # Most realistic DAS-like pattern


@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generation."""
    n_sensors: int = 2048
    sample_rate: float = 1000.0
    duration_seconds: float = 60.0
    chunk_duration_seconds: float = 10.0  # Write data in chunks to manage memory
    sensor_spacing_m: float = 1.0
    pattern: DataPattern = DataPattern.MIXED
    noise_level: float = 0.1
    seed: Optional[int] = None

    # Memory management
    max_memory_mb: float = 500.0  # Max memory per chunk
    use_mmap: bool = True         # Use memory-mapped files for large data


@dataclass
class SyntheticFileInfo:
    """Information about a generated file."""
    filename: str
    time_start: str  # ISO format
    time_end: str    # ISO format
    sensor_start: int
    sensor_end: int
    n_samples: int
    file_size_bytes: int
    shape: Tuple[int, int]


class SyntheticDASGenerator:
    """
    Generator for synthetic DAS test datasets.

    Creates realistic file structures with:
    - Multiple segment files (simulating continuous recording with breaks)
    - JSON metadata compatible with DASDataProvider interface
    - Various test patterns for validation
    """

    def __init__(self, output_dir: str = "synthetic_das_data"):
        """
        Initialize generator.

        Args:
            output_dir: Directory to write generated files
        """
        self.output_dir = Path(output_dir)
        self._rng: Optional[np.random.Generator] = None
        self._files_generated: List[SyntheticFileInfo] = []
        self._progress_callback: Optional[Callable[[float, str], None]] = None

    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """Set progress callback for long operations."""
        self._progress_callback = callback

    def _report_progress(self, progress: float, message: str):
        """Report progress if callback is set."""
        if self._progress_callback:
            self._progress_callback(progress, message)
        logger.info(f"[{progress*100:.1f}%] {message}")

    def generate_dataset(
        self,
        n_sensors: int = 2048,
        duration_seconds: float = 60.0,
        sample_rate: float = 1000.0,
        pattern: str = "mixed",
        n_segments: int = 1,
        gap_seconds: float = 0.0,
        noise_level: float = 0.1,
        seed: Optional[int] = None,
        chunk_seconds: float = 10.0,
        file_format: str = "npy"
    ) -> Path:
        """
        Generate a complete DAS test dataset.

        Args:
            n_sensors: Number of sensor channels
            duration_seconds: Total duration per segment
            sample_rate: Samples per second
            pattern: One of "noise", "sine_gradient", "traveling_wave", "events", "mixed", "realistic"
            n_segments: Number of data segments (with optional gaps between)
            gap_seconds: Gap duration between segments (0 = continuous)
            noise_level: Noise amplitude (0-1)
            seed: Random seed for reproducibility
            chunk_seconds: Write data in chunks of this duration (memory management)
            file_format: "npy" or "bin"

        Returns:
            Path to the output directory
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating DAS dataset in {self.output_dir}")

        # Initialize RNG
        self._rng = np.random.default_rng(seed)
        self._files_generated = []

        # Parse pattern
        try:
            data_pattern = DataPattern(pattern)
        except ValueError:
            logger.warning(f"Unknown pattern '{pattern}', using 'mixed'")
            data_pattern = DataPattern.MIXED

        # Calculate expected sizes
        samples_per_segment = int(duration_seconds * sample_rate)
        total_samples = samples_per_segment * n_segments
        expected_size_mb = (total_samples * n_sensors * 4) / (1024 * 1024)

        self._report_progress(0.0, f"Starting generation: {n_sensors} sensors, "
                             f"{duration_seconds}s x {n_segments} segments, "
                             f"~{expected_size_mb:.1f} MB total")

        # Generate each segment
        base_time = datetime(2024, 6, 15, 10, 0, 0)
        current_time = base_time

        for seg_idx in range(n_segments):
            seg_start = current_time
            seg_end = seg_start + timedelta(seconds=duration_seconds)

            self._report_progress(
                seg_idx / n_segments,
                f"Generating segment {seg_idx + 1}/{n_segments}"
            )

            # Generate segment data
            file_info = self._generate_segment(
                segment_idx=seg_idx,
                n_sensors=n_sensors,
                n_samples=samples_per_segment,
                sample_rate=sample_rate,
                pattern=data_pattern,
                noise_level=noise_level,
                time_start=seg_start,
                time_end=seg_end,
                chunk_samples=int(chunk_seconds * sample_rate),
                file_format=file_format
            )
            self._files_generated.append(file_info)

            # Advance time with optional gap
            current_time = seg_end + timedelta(seconds=gap_seconds)

        # Generate metadata JSON
        self._report_progress(0.95, "Writing metadata file...")
        metadata_path = self._write_metadata(
            n_sensors=n_sensors,
            sample_rate=sample_rate,
            sensor_spacing=1.0,
            base_time=base_time,
            gap_seconds=gap_seconds
        )

        self._report_progress(1.0, f"Dataset generation complete: {len(self._files_generated)} files")

        return self.output_dir

    def _generate_segment(
        self,
        segment_idx: int,
        n_sensors: int,
        n_samples: int,
        sample_rate: float,
        pattern: DataPattern,
        noise_level: float,
        time_start: datetime,
        time_end: datetime,
        chunk_samples: int,
        file_format: str
    ) -> SyntheticFileInfo:
        """Generate a single data segment file."""

        filename = f"segment_{segment_idx:03d}.{file_format}"
        filepath = self.output_dir / filename

        # For large files, write in chunks
        n_chunks = (n_samples + chunk_samples - 1) // chunk_samples

        if file_format == "npy":
            # For .npy format, we need to write as a single array
            # Use memory-mapped file for large datasets
            if n_samples * n_sensors * 4 > 100 * 1024 * 1024:  # > 100 MB
                data = self._generate_large_segment_mmap(
                    filepath, n_samples, n_sensors, sample_rate,
                    pattern, noise_level, chunk_samples
                )
            else:
                data = self._generate_segment_data(
                    n_samples, n_sensors, sample_rate, pattern, noise_level
                )
                np.save(filepath, data)
        else:
            # Binary format - write chunks sequentially
            with open(filepath, 'wb') as f:
                for chunk_idx in range(n_chunks):
                    start_sample = chunk_idx * chunk_samples
                    end_sample = min(start_sample + chunk_samples, n_samples)
                    actual_chunk_samples = end_sample - start_sample

                    chunk_data = self._generate_segment_data(
                        actual_chunk_samples, n_sensors, sample_rate,
                        pattern, noise_level,
                        time_offset=start_sample / sample_rate
                    )
                    chunk_data.tofile(f)

        # Get file size
        file_size = filepath.stat().st_size

        return SyntheticFileInfo(
            filename=filename,
            time_start=time_start.isoformat(),
            time_end=time_end.isoformat(),
            sensor_start=0,
            sensor_end=n_sensors,
            n_samples=n_samples,
            file_size_bytes=file_size,
            shape=(n_samples, n_sensors)
        )

    def _generate_large_segment_mmap(
        self,
        filepath: Path,
        n_samples: int,
        n_sensors: int,
        sample_rate: float,
        pattern: DataPattern,
        noise_level: float,
        chunk_samples: int
    ) -> None:
        """Generate large segment using memory-mapped file."""

        # Create empty memory-mapped file
        shape = (n_samples, n_sensors)

        # First save an empty array to get the header
        temp = np.zeros((1, n_sensors), dtype=np.float32)
        np.save(filepath, temp)

        # Now open for memory-mapped writing with the correct shape
        # We need to recreate with correct shape
        data = np.lib.format.open_memmap(
            str(filepath), mode='w+', dtype=np.float32, shape=shape
        )

        # Fill in chunks
        n_chunks = (n_samples + chunk_samples - 1) // chunk_samples
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_samples
            end = min(start + chunk_samples, n_samples)

            chunk = self._generate_segment_data(
                end - start, n_sensors, sample_rate,
                pattern, noise_level,
                time_offset=start / sample_rate
            )
            data[start:end, :] = chunk

            # Flush periodically
            if chunk_idx % 10 == 0:
                data.flush()

        data.flush()
        del data  # Close memory map

    def _generate_segment_data(
        self,
        n_samples: int,
        n_sensors: int,
        sample_rate: float,
        pattern: DataPattern,
        noise_level: float,
        time_offset: float = 0.0
    ) -> np.ndarray:
        """Generate synthetic data for a segment."""

        data = np.zeros((n_samples, n_sensors), dtype=np.float32)

        # Time vector
        t = np.arange(n_samples, dtype=np.float32) / sample_rate + time_offset

        # Sensor positions
        sensor_ids = np.arange(n_sensors, dtype=np.float32)

        if pattern == DataPattern.NOISE:
            data = self._generate_noise(n_samples, n_sensors, noise_level)

        elif pattern == DataPattern.SINE_GRADIENT:
            data = self._generate_sine_gradient(t, sensor_ids, noise_level)

        elif pattern == DataPattern.TRAVELING_WAVE:
            data = self._generate_traveling_wave(t, sensor_ids, noise_level)

        elif pattern == DataPattern.EVENTS:
            data = self._generate_events(t, sensor_ids, noise_level)

        elif pattern == DataPattern.MIXED:
            data = self._generate_mixed(t, sensor_ids, noise_level)

        elif pattern == DataPattern.REALISTIC:
            data = self._generate_realistic(t, sensor_ids, noise_level)

        return data.astype(np.float32)

    def _generate_noise(self, n_samples: int, n_sensors: int, level: float) -> np.ndarray:
        """Generate pure noise."""
        return level * self._rng.standard_normal((n_samples, n_sensors))

    def _generate_sine_gradient(
        self, t: np.ndarray, sensor_ids: np.ndarray, noise_level: float
    ) -> np.ndarray:
        """Generate sinusoids with frequency varying by sensor position."""
        n_samples = len(t)
        n_sensors = len(sensor_ids)
        data = np.zeros((n_samples, n_sensors), dtype=np.float32)

        # Base frequencies that vary with sensor position
        base_freqs = 0.5 + sensor_ids * 0.01

        for i, freq in enumerate(base_freqs):
            phase = sensor_ids[i] * 0.1
            data[:, i] = np.sin(2 * np.pi * freq * t + phase)

        # Add noise
        data += noise_level * self._rng.standard_normal(data.shape)
        return data

    def _generate_traveling_wave(
        self, t: np.ndarray, sensor_ids: np.ndarray, noise_level: float
    ) -> np.ndarray:
        """Generate waves traveling across sensors."""
        n_samples = len(t)
        n_sensors = len(sensor_ids)
        data = np.zeros((n_samples, n_sensors), dtype=np.float32)

        # Multiple traveling waves with different speeds
        wave_speeds = [100, 250, 500]  # sensors per second
        wave_freqs = [2.0, 5.0, 10.0]
        wave_amps = [1.0, 0.5, 0.3]

        for speed, freq, amp in zip(wave_speeds, wave_freqs, wave_amps):
            for i, sensor_id in enumerate(sensor_ids):
                # Wave arrives at different times for different sensors
                delay = sensor_id / speed
                phase = 2 * np.pi * freq * (t - delay)
                data[:, i] += amp * np.sin(phase)

        # Normalize and add noise
        data = data / np.abs(data).max() if data.max() != 0 else data
        data += noise_level * self._rng.standard_normal(data.shape)
        return data

    def _generate_events(
        self, t: np.ndarray, sensor_ids: np.ndarray, noise_level: float
    ) -> np.ndarray:
        """Generate discrete events (like seismic events or vehicles)."""
        n_samples = len(t)
        n_sensors = len(sensor_ids)
        data = noise_level * self._rng.standard_normal((n_samples, n_sensors))

        # Generate random events
        duration = t[-1] - t[0]
        n_events = max(1, int(duration / 5))  # ~1 event per 5 seconds

        for _ in range(n_events):
            # Event parameters
            event_time = t[0] + self._rng.uniform(0, duration)
            event_sensor = self._rng.integers(0, len(sensor_ids))
            event_width_time = self._rng.uniform(0.1, 0.5)
            event_width_space = self._rng.uniform(20, 100)
            event_amplitude = self._rng.uniform(1.0, 3.0)
            event_freq = self._rng.uniform(5, 20)

            # Create event signal (Gaussian envelope in space and time)
            for i, sensor_id in enumerate(sensor_ids):
                spatial_dist = abs(i - event_sensor)
                time_envelope = np.exp(-((t - event_time) ** 2) / (2 * event_width_time ** 2))
                space_envelope = np.exp(-(spatial_dist ** 2) / (2 * event_width_space ** 2))

                # Propagation delay
                delay = spatial_dist / 200  # Wave speed ~200 sensors/sec
                signal = np.sin(2 * np.pi * event_freq * (t - event_time - delay))

                data[:, i] += event_amplitude * time_envelope * space_envelope * signal

        return data

    def _generate_mixed(
        self, t: np.ndarray, sensor_ids: np.ndarray, noise_level: float
    ) -> np.ndarray:
        """Generate a mix of different patterns."""
        data = np.zeros((len(t), len(sensor_ids)), dtype=np.float32)

        # Background: low-frequency variations
        data += 0.3 * self._generate_sine_gradient(t, sensor_ids, 0)

        # Add traveling waves
        data += 0.4 * self._generate_traveling_wave(t, sensor_ids, 0)

        # Add events
        data += 0.3 * self._generate_events(t, sensor_ids, 0)

        # Add noise
        data += noise_level * self._rng.standard_normal(data.shape)

        return data

    def _generate_realistic(
        self, t: np.ndarray, sensor_ids: np.ndarray, noise_level: float
    ) -> np.ndarray:
        """Generate most realistic DAS-like pattern."""
        n_samples = len(t)
        n_sensors = len(sensor_ids)
        data = np.zeros((n_samples, n_sensors), dtype=np.float32)

        # 1. Coherent noise (correlated across nearby sensors)
        coherence_length = 50  # sensors
        for i in range(n_sensors):
            # Low-pass filtered noise
            noise = self._rng.standard_normal(n_samples)
            # Simple smoothing
            kernel_size = int(0.02 * len(t))  # 20ms smoothing
            if kernel_size > 1:
                kernel = np.ones(kernel_size) / kernel_size
                noise = np.convolve(noise, kernel, mode='same')
            data[:, i] = noise

        # Apply spatial correlation
        spatial_kernel = np.exp(-np.arange(coherence_length) / 10)
        spatial_kernel = spatial_kernel / spatial_kernel.sum()
        for t_idx in range(0, n_samples, 100):  # Sparse updates for speed
            row = data[t_idx, :]
            smoothed = np.convolve(row, spatial_kernel, mode='same')
            data[t_idx, :] = smoothed

        # 2. Add microseismic background
        for freq in [0.5, 1.0, 2.0, 5.0]:
            amp = 0.2 / freq
            for i in range(n_sensors):
                phase = self._rng.uniform(0, 2 * np.pi)
                data[:, i] += amp * np.sin(2 * np.pi * freq * t + phase)

        # 3. Add propagating disturbances
        n_disturbances = max(1, int((t[-1] - t[0]) / 10))
        for _ in range(n_disturbances):
            start_time = t[0] + self._rng.uniform(0, t[-1] - t[0])
            start_sensor = self._rng.integers(0, n_sensors)
            speed = self._rng.uniform(100, 1000)  # sensors/sec
            freq = self._rng.uniform(5, 50)
            duration = self._rng.uniform(0.1, 1.0)
            amplitude = self._rng.uniform(0.5, 2.0)

            for i in range(n_sensors):
                dist = abs(i - start_sensor)
                arrival = start_time + dist / speed

                # Gaussian envelope
                envelope = np.exp(-((t - arrival) ** 2) / (2 * (duration/3) ** 2))
                signal = np.sin(2 * np.pi * freq * (t - arrival))

                # Amplitude decay with distance
                decay = np.exp(-dist / 500)
                data[:, i] += amplitude * decay * envelope * signal

        # 4. Scale and add measurement noise
        data = data / (np.abs(data).max() + 1e-10)
        data += noise_level * self._rng.standard_normal(data.shape)

        return data.astype(np.float32)

    def _write_metadata(
        self,
        n_sensors: int,
        sample_rate: float,
        sensor_spacing: float,
        base_time: datetime,
        gap_seconds: float
    ) -> Path:
        """Write JSON metadata file."""

        # Build time ranges from files
        time_ranges = []
        for file_info in self._files_generated:
            time_ranges.append([file_info.time_start, file_info.time_end])

        # Build file list for metadata
        files_info = []
        for fi in self._files_generated:
            files_info.append({
                "filename": fi.filename,
                "time_start": fi.time_start,
                "time_end": fi.time_end,
                "sensor_start": fi.sensor_start,
                "sensor_end": fi.sensor_end,
                "n_samples": fi.n_samples,
                "file_size_bytes": fi.file_size_bytes,
                "shape": list(fi.shape)
            })

        metadata = {
            "version": "1.0",
            "generator": "SyntheticDASGenerator",
            "generated_at": datetime.now().isoformat(),

            "sensor_range": [0, n_sensors],
            "n_sensors": n_sensors,
            "sample_rate": sample_rate,
            "sensor_spacing_m": sensor_spacing,
            "units": "phase_derivative_rad_s",

            "time_ranges": time_ranges,
            "total_samples": sum(f.n_samples for f in self._files_generated),
            "total_duration_seconds": sum(
                (datetime.fromisoformat(f.time_end) -
                 datetime.fromisoformat(f.time_start)).total_seconds()
                for f in self._files_generated
            ),

            "files": files_info,

            "data_format": {
                "dtype": "float32",
                "shape_convention": "(n_time_samples, n_sensors)",
                "byte_order": "little_endian"
            }
        }

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Wrote metadata to {metadata_path}")
        return metadata_path

    def cleanup(self):
        """Remove generated files."""
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            logger.info(f"Cleaned up {self.output_dir}")


def generate_test_dataset(
    output_dir: str = "test_das_data",
    size: str = "small",
    pattern: str = "mixed",
    seed: int = 42
) -> Path:
    """
    Convenience function to generate standard test datasets.

    Args:
        output_dir: Where to save the dataset
        size: "tiny", "small", "medium", "large", "huge"
        pattern: Data pattern type
        seed: Random seed

    Returns:
        Path to the generated dataset
    """

    # Size presets
    sizes = {
        "tiny": {"n_sensors": 256, "duration": 10, "n_segments": 1},
        "small": {"n_sensors": 1024, "duration": 30, "n_segments": 1},
        "medium": {"n_sensors": 2048, "duration": 60, "n_segments": 2},
        "large": {"n_sensors": 4096, "duration": 300, "n_segments": 3},
        "huge": {"n_sensors": 8192, "duration": 600, "n_segments": 5},
    }

    config = sizes.get(size, sizes["small"])

    generator = SyntheticDASGenerator(output_dir)
    return generator.generate_dataset(
        n_sensors=config["n_sensors"],
        duration_seconds=config["duration"],
        n_segments=config["n_segments"],
        gap_seconds=30.0 if config["n_segments"] > 1 else 0.0,
        pattern=pattern,
        seed=seed,
        chunk_seconds=10.0
    )


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    path = generate_test_dataset(size="small", pattern="realistic")
    print(f"Generated dataset at: {path}")
