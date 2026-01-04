"""
Data model for tagged acoustic events.

Events are marked by two vertical time lines on the spectrogram,
with associated metadata like harmonic number, SNR, and frequency range.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class TaggedEvent:
    """Represents a tagged acoustic event between two time markers.

    Events are defined by:
    - Time range (start/end) relative to the audio file
    - Frequency range (f_min/f_max) for the region of interest
    - Metadata: harmonic number, SNR estimate, notes
    - Auto-generated: image path, absolute timestamps (if filename parseable)
    """
    # Unique identifier
    id: int

    # File information
    audio_file: str                      # Full path or filename of audio file
    sensor_name: Optional[str] = None    # e.g., "pixel" (parsed from filename)
    sensor_id: Optional[int] = None      # e.g., 1008 (parsed from filename)

    # Time range (relative to audio file start, in seconds)
    t_start: float = 0.0                 # Event start time
    t_end: float = 0.0                   # Event end time

    # Frequency range (Hz)
    f_min: float = 0.0                   # Lower frequency bound
    f_max: float = 22050.0               # Upper frequency bound

    # Absolute timestamps (computed from filename if parseable)
    file_start_time: Optional[datetime] = None    # Recording start time from filename
    file_end_time: Optional[datetime] = None      # Recording end time from filename
    event_start_absolute: Optional[datetime] = None  # Computed: file_start + t_start
    event_end_absolute: Optional[datetime] = None    # Computed: file_start + t_end

    # User-provided metadata
    harmonic_number: Optional[int] = None     # Harmonic order (1=fundamental, 2,3,4...)
    snr_estimate_db: Optional[float] = None   # SNR in dB (user can override auto-estimate)
    notes: str = ""                           # Free-form notes

    # Auto-generated paths
    image_path: str = ""                 # Path to captured spectrogram image

    # Timestamps
    created_at: Optional[datetime] = None     # When event was tagged

    def __post_init__(self):
        """Set creation timestamp if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now()

    @property
    def duration(self) -> float:
        """Duration of the event in seconds."""
        return abs(self.t_end - self.t_start)

    @property
    def frequency_span(self) -> float:
        """Frequency span in Hz."""
        return abs(self.f_max - self.f_min)

    @property
    def center_frequency(self) -> float:
        """Center frequency in Hz."""
        return (self.f_min + self.f_max) / 2

    @property
    def center_time(self) -> float:
        """Center time in seconds (relative)."""
        return (self.t_start + self.t_end) / 2

    def to_csv_row(self) -> Dict[str, Any]:
        """Convert event to dictionary for CSV export.

        Returns:
            Dictionary with all fields suitable for CSV writing
        """
        def format_datetime(dt: Optional[datetime]) -> str:
            if dt is None:
                return ""
            return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Milliseconds

        # Column 'pixel' contains sensor_id (the sensor_name is always 'pixel')
        pixel_value = self.sensor_id if self.sensor_id is not None else ""

        return {
            'id': self.id,
            'audio_file': Path(self.audio_file).name,  # Just filename, not full path
            'pixel': pixel_value,
            't_start': f"{self.t_start:.6f}",
            't_end': f"{self.t_end:.6f}",
            'event_start_absolute': format_datetime(self.event_start_absolute),
            'event_end_absolute': format_datetime(self.event_end_absolute),
            'harmonic_number': self.harmonic_number if self.harmonic_number is not None else "",
            'SNR': self.notes,  # Signal quality: High/Medium/Low
            'image_path': self.image_path,
            'created_at': format_datetime(self.created_at)
        }

    @classmethod
    def from_csv_row(cls, row: Dict[str, str]) -> 'TaggedEvent':
        """Create TaggedEvent from CSV row dictionary.

        Args:
            row: Dictionary from CSV DictReader

        Returns:
            TaggedEvent instance
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

        # Handle both old format (sensor_name, sensor_id) and new format (pixel)
        pixel_id = parse_int(row.get('pixel', ''))
        if pixel_id is None:
            # Try old format
            pixel_id = parse_int(row.get('sensor_id', ''))

        # Handle both old format (notes) and new format (SNR)
        notes_value = row.get('SNR', '') or row.get('notes', '')

        return cls(
            id=int(row.get('id', 0)),
            audio_file=row.get('audio_file', ''),
            sensor_name='pixel',  # Always 'pixel' for this format
            sensor_id=pixel_id,
            t_start=float(row.get('t_start', 0)),
            t_end=float(row.get('t_end', 0)),
            f_min=float(row.get('f_min', 0)),
            f_max=float(row.get('f_max', 22050)),
            event_start_absolute=parse_datetime(row.get('event_start_absolute', '')),
            event_end_absolute=parse_datetime(row.get('event_end_absolute', '')),
            harmonic_number=parse_int(row.get('harmonic_number', '')),
            snr_estimate_db=parse_float(row.get('snr_estimate_db', '')),
            notes=notes_value,
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
            't_start': self.t_start,
            't_end': self.t_end,
            'f_min': self.f_min,
            'f_max': self.f_max,
            'file_start_time': self.file_start_time.isoformat() if self.file_start_time else None,
            'file_end_time': self.file_end_time.isoformat() if self.file_end_time else None,
            'event_start_absolute': self.event_start_absolute.isoformat() if self.event_start_absolute else None,
            'event_end_absolute': self.event_end_absolute.isoformat() if self.event_end_absolute else None,
            'harmonic_number': self.harmonic_number,
            'snr_estimate_db': self.snr_estimate_db,
            'notes': self.notes,
            'image_path': self.image_path,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaggedEvent':
        """Create from dictionary (for JSON deserialization)."""
        def parse_iso(s: Optional[str]) -> Optional[datetime]:
            if not s:
                return None
            try:
                return datetime.fromisoformat(s)
            except ValueError:
                return None

        return cls(
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
            event_start_absolute=parse_iso(data.get('event_start_absolute')),
            event_end_absolute=parse_iso(data.get('event_end_absolute')),
            harmonic_number=data.get('harmonic_number'),
            snr_estimate_db=data.get('snr_estimate_db'),
            notes=data.get('notes', ''),
            image_path=data.get('image_path', ''),
            created_at=parse_iso(data.get('created_at'))
        )


# CSV column headers (in order)
CSV_COLUMNS = [
    'id',
    'audio_file',
    'pixel',  # Sensor ID (column name is 'pixel', value is the numeric ID)
    't_start',
    't_end',
    'event_start_absolute',
    'event_end_absolute',
    'harmonic_number',
    'SNR',  # Signal quality: High/Medium/Low
    'image_path',
    'created_at'
]
