"""
Time formatting utilities for DAS visualization.

Handles conversion between:
- Sample indices
- Datetime objects
- Display strings (HH:MM:SS format)
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple, Union
import re


class TimeFormatter:
    """
    Formats time values for DAS waterfall display.

    Converts between sample indices, datetime objects, and display strings.
    The Y-axis of the waterfall shows time in HH:MM:SS format.
    """

    # Time format patterns
    FORMAT_HMS = "%H:%M:%S"
    FORMAT_HMS_MS = "%H:%M:%S.%f"
    FORMAT_FULL = "%Y-%m-%d %H:%M:%S"
    FORMAT_FULL_MS = "%Y-%m-%d %H:%M:%S.%f"

    def __init__(
        self,
        base_time: Optional[datetime] = None,
        sample_rate: float = 1000.0
    ):
        """
        Initialize time formatter.

        Args:
            base_time: Reference datetime for sample index 0.
                       If None, uses relative time from 00:00:00.
            sample_rate: Samples per second
        """
        self.base_time = base_time or datetime(1970, 1, 1, 0, 0, 0)
        self.sample_rate = sample_rate
        self._use_absolute_time = base_time is not None

    @property
    def use_absolute_time(self) -> bool:
        """Whether using absolute datetime or relative time."""
        return self._use_absolute_time

    def sample_to_datetime(self, sample_idx: int) -> datetime:
        """
        Convert sample index to datetime.

        Args:
            sample_idx: Sample index (0-based)

        Returns:
            Corresponding datetime
        """
        seconds = sample_idx / self.sample_rate
        return self.base_time + timedelta(seconds=seconds)

    def datetime_to_sample(self, dt: datetime) -> int:
        """
        Convert datetime to sample index.

        Args:
            dt: Datetime to convert

        Returns:
            Corresponding sample index
        """
        delta = (dt - self.base_time).total_seconds()
        return int(delta * self.sample_rate)

    def sample_to_seconds(self, sample_idx: int) -> float:
        """
        Convert sample index to seconds from base time.

        Args:
            sample_idx: Sample index

        Returns:
            Seconds from base time
        """
        return sample_idx / self.sample_rate

    def seconds_to_sample(self, seconds: float) -> int:
        """
        Convert seconds to sample index.

        Args:
            seconds: Seconds from base time

        Returns:
            Sample index
        """
        return int(seconds * self.sample_rate)

    def format_sample(
        self,
        sample_idx: int,
        show_ms: bool = False,
        show_date: bool = False
    ) -> str:
        """
        Format sample index as display string.

        Args:
            sample_idx: Sample index to format
            show_ms: Include milliseconds
            show_date: Include date (only for absolute time)

        Returns:
            Formatted time string (e.g., "10:30:45" or "10:30:45.123")
        """
        dt = self.sample_to_datetime(sample_idx)
        return self.format_datetime(dt, show_ms=show_ms, show_date=show_date)

    def format_datetime(
        self,
        dt: datetime,
        show_ms: bool = False,
        show_date: bool = False
    ) -> str:
        """
        Format datetime as display string.

        Args:
            dt: Datetime to format
            show_ms: Include milliseconds
            show_date: Include date

        Returns:
            Formatted time string
        """
        if show_date and self._use_absolute_time:
            fmt = self.FORMAT_FULL_MS if show_ms else self.FORMAT_FULL
        else:
            fmt = self.FORMAT_HMS_MS if show_ms else self.FORMAT_HMS

        result = dt.strftime(fmt)

        # Trim microseconds to milliseconds if showing ms
        if show_ms:
            result = result[:-3]  # Remove last 3 digits (keep .XXX)

        return result

    def format_seconds(self, seconds: float, show_ms: bool = False) -> str:
        """
        Format seconds as HH:MM:SS string.

        Args:
            seconds: Seconds to format
            show_ms: Include milliseconds

        Returns:
            Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60

        if show_ms:
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
        else:
            return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"

    def format_duration(self, duration_seconds: float) -> str:
        """
        Format duration in human-readable form.

        Args:
            duration_seconds: Duration in seconds

        Returns:
            Human-readable duration (e.g., "1h 30m 45s")
        """
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 or not parts:
            parts.append(f"{seconds}s")

        return " ".join(parts)

    def parse_time_string(self, time_str: str) -> Tuple[int, Optional[str]]:
        """
        Parse user input time string to sample index.

        Supports formats:
        - "HH:MM:SS" (e.g., "10:30:45")
        - "HH:MM:SS.mmm" (e.g., "10:30:45.123")
        - "MM:SS" (e.g., "30:45" -> 00:30:45)
        - "SS" (e.g., "45" -> 00:00:45)

        Args:
            time_str: Time string to parse

        Returns:
            Tuple of (sample_index, error_message)
            error_message is None on success
        """
        time_str = time_str.strip()

        # Try different patterns
        patterns = [
            (r'^(\d{1,2}):(\d{2}):(\d{2})\.(\d{1,3})$', True),   # HH:MM:SS.mmm
            (r'^(\d{1,2}):(\d{2}):(\d{2})$', False),             # HH:MM:SS
            (r'^(\d{1,2}):(\d{2})$', False),                      # MM:SS
            (r'^(\d+)$', False),                                   # SS
        ]

        for pattern, has_ms in patterns:
            match = re.match(pattern, time_str)
            if match:
                groups = match.groups()

                try:
                    if len(groups) == 4:  # HH:MM:SS.mmm
                        hours = int(groups[0])
                        minutes = int(groups[1])
                        seconds = int(groups[2])
                        ms = int(groups[3].ljust(3, '0')[:3])  # Pad/trim to 3 digits
                    elif len(groups) == 3:  # HH:MM:SS
                        hours = int(groups[0])
                        minutes = int(groups[1])
                        seconds = int(groups[2])
                        ms = 0
                    elif len(groups) == 2:  # MM:SS
                        hours = 0
                        minutes = int(groups[0])
                        seconds = int(groups[1])
                        ms = 0
                    else:  # SS
                        hours = 0
                        minutes = 0
                        seconds = int(groups[0])
                        ms = 0

                    # Validate ranges
                    if minutes >= 60:
                        return 0, "Minutes must be < 60"
                    if seconds >= 60:
                        return 0, "Seconds must be < 60"

                    total_seconds = hours * 3600 + minutes * 60 + seconds + ms / 1000.0
                    sample_idx = self.seconds_to_sample(total_seconds)

                    return sample_idx, None

                except ValueError as e:
                    return 0, f"Invalid time format: {e}"

        return 0, f"Could not parse time string: {time_str}"

    def get_tick_values(
        self,
        start_sample: int,
        end_sample: int,
        max_ticks: int = 10
    ) -> Tuple[list, list]:
        """
        Generate tick values and labels for axis display.

        Args:
            start_sample: Start sample index
            end_sample: End sample index
            max_ticks: Maximum number of ticks

        Returns:
            Tuple of (tick_positions, tick_labels)
            tick_positions are sample indices
            tick_labels are formatted strings
        """
        duration_samples = end_sample - start_sample
        duration_seconds = duration_samples / self.sample_rate

        # Choose appropriate tick interval
        if duration_seconds <= 10:
            interval = 1  # 1 second
        elif duration_seconds <= 60:
            interval = 5  # 5 seconds
        elif duration_seconds <= 300:
            interval = 30  # 30 seconds
        elif duration_seconds <= 1800:
            interval = 60  # 1 minute
        elif duration_seconds <= 7200:
            interval = 300  # 5 minutes
        elif duration_seconds <= 14400:
            interval = 600  # 10 minutes
        else:
            interval = 1800  # 30 minutes

        # Generate ticks
        start_seconds = start_sample / self.sample_rate
        end_seconds = end_sample / self.sample_rate

        # Align to interval boundaries
        first_tick = ((int(start_seconds) // interval) + 1) * interval

        positions = []
        labels = []

        current = first_tick
        while current < end_seconds and len(positions) < max_ticks:
            sample_pos = self.seconds_to_sample(current)
            positions.append(sample_pos)
            labels.append(self.format_seconds(current))
            current += interval

        return positions, labels


class RelativeTimeFormatter(TimeFormatter):
    """
    Time formatter using relative time (starting from 00:00:00).

    Use this when there's no absolute timestamp for the data.
    """

    def __init__(self, sample_rate: float = 1000.0):
        super().__init__(base_time=None, sample_rate=sample_rate)


class AbsoluteTimeFormatter(TimeFormatter):
    """
    Time formatter using absolute datetime.

    Use this when data has actual timestamps.
    """

    def __init__(self, base_time: datetime, sample_rate: float = 1000.0):
        super().__init__(base_time=base_time, sample_rate=sample_rate)


def format_time_range(
    start: Union[int, datetime],
    end: Union[int, datetime],
    formatter: Optional[TimeFormatter] = None,
    sample_rate: float = 1000.0
) -> str:
    """
    Format a time range for display.

    Args:
        start: Start time (sample index or datetime)
        end: End time (sample index or datetime)
        formatter: Optional TimeFormatter (creates one if not provided)
        sample_rate: Sample rate (if no formatter provided)

    Returns:
        Formatted range string (e.g., "10:30:00 - 11:00:00")
    """
    if formatter is None:
        formatter = TimeFormatter(sample_rate=sample_rate)

    if isinstance(start, int):
        start_str = formatter.format_sample(start)
    else:
        start_str = formatter.format_datetime(start)

    if isinstance(end, int):
        end_str = formatter.format_sample(end)
    else:
        end_str = formatter.format_datetime(end)

    return f"{start_str} - {end_str}"
