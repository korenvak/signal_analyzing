"""
Filename parser for extracting metadata from standardized audio filenames.

Handles format: 'sensor_name - sensor_id - start_time - end_time'
Example: 'pixel - 1008 - 2025-29-10 11-04-47 - 2025-29-10 12-20-49.flac'

Date format: yyyy-dd-mm HH-MM-SS (day before month)
"""
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedFilename:
    """Parsed components from a standardized audio filename."""
    sensor_name: str           # e.g., "pixel"
    sensor_id: int             # e.g., 1008
    start_time: datetime       # File recording start time
    end_time: datetime         # File recording end time
    original_filename: str     # Original filename for reference

    def compute_absolute_time(self, relative_seconds: float) -> datetime:
        """Compute absolute datetime from relative time offset.

        Args:
            relative_seconds: Time offset from start of file in seconds

        Returns:
            Absolute datetime
        """
        from datetime import timedelta
        return self.start_time + timedelta(seconds=relative_seconds)

    @property
    def duration_seconds(self) -> float:
        """Expected duration based on filename timestamps."""
        return (self.end_time - self.start_time).total_seconds()


def parse_datetime_component(date_str: str, time_str: str) -> Optional[datetime]:
    """Parse date and time strings in the format yyyy-dd-mm HH-MM-SS.

    Args:
        date_str: Date string like "2025-29-10" (yyyy-dd-mm)
        time_str: Time string like "11-04-47" (HH-MM-SS)

    Returns:
        datetime object or None if parsing fails
    """
    try:
        # Parse date: yyyy-dd-mm
        date_parts = date_str.split('-')
        if len(date_parts) != 3:
            return None
        year = int(date_parts[0])
        day = int(date_parts[1])
        month = int(date_parts[2])

        # Parse time: HH-MM-SS
        time_parts = time_str.split('-')
        if len(time_parts) != 3:
            return None
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        second = int(time_parts[2])

        return datetime(year, month, day, hour, minute, second)
    except (ValueError, IndexError) as e:
        logger.debug(f"Failed to parse datetime: {date_str} {time_str}: {e}")
        return None


def parse_pixel_filename(filename: str) -> Optional[ParsedFilename]:
    """Parse a standardized audio filename to extract metadata.

    Expected format: 'sensor_name - sensor_id - start_date start_time - end_date end_time'
    Example: 'pixel - 1008 - 2025-29-10 11-04-47 - 2025-29-10 12-20-49.flac'

    Args:
        filename: Full path or just filename

    Returns:
        ParsedFilename object or None if parsing fails
    """
    # Get just the filename without path and extension
    path = Path(filename)
    stem = path.stem  # Remove extension
    original = path.name

    # Pattern: sensor_name - sensor_id - date time - date time
    # Using flexible whitespace matching
    pattern = r'^(\w+)\s*-\s*(\d+)\s*-\s*(\d{4}-\d{1,2}-\d{1,2})\s+(\d{1,2}-\d{1,2}-\d{1,2})\s*-\s*(\d{4}-\d{1,2}-\d{1,2})\s+(\d{1,2}-\d{1,2}-\d{1,2})$'

    match = re.match(pattern, stem)
    if not match:
        logger.debug(f"Filename doesn't match expected pattern: {stem}")
        return None

    sensor_name = match.group(1)
    sensor_id_str = match.group(2)
    start_date = match.group(3)
    start_time = match.group(4)
    end_date = match.group(5)
    end_time = match.group(6)

    try:
        sensor_id = int(sensor_id_str)
    except ValueError:
        logger.warning(f"Invalid sensor ID: {sensor_id_str}")
        return None

    start_datetime = parse_datetime_component(start_date, start_time)
    end_datetime = parse_datetime_component(end_date, end_time)

    if start_datetime is None or end_datetime is None:
        logger.warning(f"Failed to parse timestamps from filename: {stem}")
        return None

    return ParsedFilename(
        sensor_name=sensor_name,
        sensor_id=sensor_id,
        start_time=start_datetime,
        end_time=end_datetime,
        original_filename=original
    )


def format_absolute_time(dt: datetime, include_milliseconds: bool = True) -> str:
    """Format datetime for display/CSV output.

    Args:
        dt: datetime object
        include_milliseconds: Whether to include milliseconds

    Returns:
        Formatted string like "2025-10-29 11:06:52.500" or "2025-10-29 11:06:52"
    """
    if include_milliseconds:
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # Trim to milliseconds
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def try_parse_filename(filename: str) -> Dict[str, Any]:
    """Try to parse filename and return a dict with results.

    This is a convenience function that returns a dict instead of dataclass,
    useful for direct integration with event creation.

    Args:
        filename: Full path or filename

    Returns:
        Dict with parsed info or empty dict with 'parsed': False
    """
    parsed = parse_pixel_filename(filename)
    if parsed is None:
        return {
            'parsed': False,
            'sensor_name': None,
            'sensor_id': None,
            'start_time': None,
            'end_time': None
        }

    return {
        'parsed': True,
        'sensor_name': parsed.sensor_name,
        'sensor_id': parsed.sensor_id,
        'start_time': parsed.start_time,
        'end_time': parsed.end_time,
        'duration_seconds': parsed.duration_seconds
    }
