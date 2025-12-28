"""
Track data model for painted tracks over spectrograms.

Represents user-painted frequency tracks with interpolated values and metadata.
"""
import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PaintedTrack:
    """Represents a user-painted frequency track over a spectrogram.

    A track consists of:
    - User-clicked control points (time, frequency)
    - Interpolated curve using monotone cubic interpolation
    - Amplitude/dB values sampled from the spectrogram
    - Metadata: file info, timestamps, PNG export path
    """
    # Unique identifier
    id: int

    # File information
    audio_file: str                      # Full path or filename of audio file
    sensor_name: Optional[str] = None    # e.g., "pixel" (parsed from filename)
    sensor_id: Optional[int] = None      # e.g., 1008 (parsed from filename)

    # Control points (user-clicked points)
    control_points: List[Tuple[float, float]] = field(default_factory=list)  # [(time, freq), ...]

    # Interpolated track data (generated from control points)
    interpolated_times: List[float] = field(default_factory=list)      # Time values (seconds)
    interpolated_freqs: List[float] = field(default_factory=list)      # Frequency values (Hz)
    interpolated_amplitudes: List[float] = field(default_factory=list) # Amplitude/dB values

    # Track bounds
    t_start: float = 0.0                 # Track start time (first point)
    t_end: float = 0.0                   # Track end time (last point)
    f_min: float = 0.0                   # Minimum frequency
    f_max: float = 0.0                   # Maximum frequency

    # Absolute timestamps (computed from filename if parseable)
    file_start_time: Optional[datetime] = None    # Recording start time from filename
    file_end_time: Optional[datetime] = None      # Recording end time from filename
    track_start_absolute: Optional[datetime] = None  # Computed: file_start + t_start
    track_end_absolute: Optional[datetime] = None    # Computed: file_start + t_end

    # User-provided metadata
    notes: str = ""                      # Free-form notes
    track_label: str = ""                # User label for this track

    # Auto-generated paths
    image_path: str = ""                 # Path to PNG with track overlay

    # Timestamps
    created_at: Optional[datetime] = None     # When track was created

    def __post_init__(self):
        """Set creation timestamp if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now()

    @property
    def duration(self) -> float:
        """Duration of the track in seconds."""
        return abs(self.t_end - self.t_start)

    @property
    def frequency_range(self) -> float:
        """Frequency range in Hz."""
        return abs(self.f_max - self.f_min)

    @property
    def n_control_points(self) -> int:
        """Number of user-clicked control points."""
        return len(self.control_points)

    @property
    def n_interpolated_points(self) -> int:
        """Number of interpolated points."""
        return len(self.interpolated_times)

    def to_csv_row(self) -> Dict[str, Any]:
        """Convert track to dictionary for CSV export.

        Returns:
            Dictionary with all fields suitable for CSV writing
        """
        def format_datetime(dt: Optional[datetime]) -> str:
            if dt is None:
                return ""
            return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Milliseconds

        # Column 'pixel' contains sensor_id (the sensor_name is always 'pixel')
        pixel_value = self.sensor_id if self.sensor_id is not None else ""

        # Format control points as string: "t1,f1;t2,f2;..."
        control_points_str = ";".join(f"{t:.6f},{f:.2f}" for t, f in self.control_points)

        return {
            'id': self.id,
            'audio_file': Path(self.audio_file).name,  # Just filename, not full path
            'pixel': pixel_value,
            't_start': f"{self.t_start:.6f}",
            't_end': f"{self.t_end:.6f}",
            'f_min': f"{self.f_min:.2f}",
            'f_max': f"{self.f_max:.2f}",
            'n_control_points': self.n_control_points,
            'n_interpolated_points': self.n_interpolated_points,
            'control_points': control_points_str,
            'track_start_absolute': format_datetime(self.track_start_absolute),
            'track_end_absolute': format_datetime(self.track_end_absolute),
            'track_label': self.track_label,
            'notes': self.notes,
            'image_path': self.image_path,
            'created_at': format_datetime(self.created_at)
        }

    @classmethod
    def from_csv_row(cls, row: Dict[str, str]) -> 'PaintedTrack':
        """Create PaintedTrack from CSV row dictionary.

        Args:
            row: Dictionary from CSV DictReader

        Returns:
            PaintedTrack instance
        """
        def parse_datetime(s: str) -> Optional[datetime]:
            if not s or s.strip() == "":
                return None
            try:
                # Try with milliseconds first
                return datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f")
            except ValueError:
                try:
                    # Try without milliseconds
                    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    return None

        def parse_int(s: str) -> Optional[int]:
            if not s or s.strip() == "":
                return None
            try:
                return int(s)
            except ValueError:
                return None

        def parse_float(s: str) -> Optional[float]:
            if not s or s.strip() == "":
                return None
            try:
                return float(s)
            except ValueError:
                return None

        def parse_control_points(s: str) -> List[Tuple[float, float]]:
            """Parse control points string: 't1,f1;t2,f2;...'"""
            if not s or s.strip() == "":
                return []
            try:
                points = []
                for pair in s.split(';'):
                    if pair.strip():
                        t, f = pair.split(',')
                        points.append((float(t), float(f)))
                return points
            except Exception as e:
                logger.warning(f"Failed to parse control points: {e}")
                return []

        # Handle both old format (sensor_name, sensor_id) and new format (pixel)
        pixel_id = parse_int(row.get('pixel', ''))

        return cls(
            id=int(row.get('id', 0)),
            audio_file=row.get('audio_file', ''),
            sensor_name='pixel',  # Always 'pixel' for this format
            sensor_id=pixel_id,
            control_points=parse_control_points(row.get('control_points', '')),
            t_start=float(row.get('t_start', 0)),
            t_end=float(row.get('t_end', 0)),
            f_min=float(row.get('f_min', 0)),
            f_max=float(row.get('f_max', 22050)),
            track_start_absolute=parse_datetime(row.get('track_start_absolute', '')),
            track_end_absolute=parse_datetime(row.get('track_end_absolute', '')),
            track_label=row.get('track_label', ''),
            notes=row.get('notes', ''),
            image_path=row.get('image_path', ''),
            created_at=parse_datetime(row.get('created_at', ''))
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)."""
        return {
            'id': self.id,
            'audio_file': self.audio_file,
            'sensor_name': self.sensor_name,
            'sensor_id': self.sensor_id,
            'control_points': self.control_points,
            'interpolated_times': self.interpolated_times,
            'interpolated_freqs': self.interpolated_freqs,
            'interpolated_amplitudes': self.interpolated_amplitudes,
            't_start': self.t_start,
            't_end': self.t_end,
            'f_min': self.f_min,
            'f_max': self.f_max,
            'file_start_time': self.file_start_time.isoformat() if self.file_start_time else None,
            'file_end_time': self.file_end_time.isoformat() if self.file_end_time else None,
            'track_start_absolute': self.track_start_absolute.isoformat() if self.track_start_absolute else None,
            'track_end_absolute': self.track_end_absolute.isoformat() if self.track_end_absolute else None,
            'notes': self.notes,
            'track_label': self.track_label,
            'image_path': self.image_path,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaintedTrack':
        """Create from dictionary (for JSON deserialization)."""
        def parse_iso(s: Optional[str]) -> Optional[datetime]:
            if not s:
                return None
            try:
                return datetime.fromisoformat(s)
            except ValueError:
                return None

        track = cls(
            id=data['id'],
            audio_file=data.get('audio_file', ''),
            sensor_name=data.get('sensor_name'),
            sensor_id=data.get('sensor_id'),
            t_start=data.get('t_start', 0),
            t_end=data.get('t_end', 0),
            f_min=data.get('f_min', 0),
            f_max=data.get('f_max', 22050),
            file_start_time=parse_iso(data.get('file_start_time')),
            file_end_time=parse_iso(data.get('file_end_time')),
            track_start_absolute=parse_iso(data.get('track_start_absolute')),
            track_end_absolute=parse_iso(data.get('track_end_absolute')),
            notes=data.get('notes', ''),
            track_label=data.get('track_label', ''),
            image_path=data.get('image_path', ''),
            created_at=parse_iso(data.get('created_at'))
        )

        # Set mutable fields
        track.control_points = data.get('control_points', [])
        track.interpolated_times = data.get('interpolated_times', [])
        track.interpolated_freqs = data.get('interpolated_freqs', [])
        track.interpolated_amplitudes = data.get('interpolated_amplitudes', [])

        return track


# CSV column headers (in order)
CSV_COLUMNS = [
    'id',
    'audio_file',
    'pixel',
    't_start',
    't_end',
    'f_min',
    'f_max',
    'n_control_points',
    'n_interpolated_points',
    'control_points',
    'track_start_absolute',
    'track_end_absolute',
    'track_label',
    'notes',
    'image_path',
    'created_at'
]


# Detailed CSV columns for interpolated data export
INTERPOLATED_DATA_COLUMNS = [
    'track_id',
    'audio_file',
    'pixel',
    'time_seconds',
    'frequency_hz',
    'amplitude_db'
]
