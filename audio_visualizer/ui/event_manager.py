"""
Event manager for tagged acoustic events.

Handles:
- Event storage and retrieval
- CSV persistence (create new / continue existing)
- Image capture with matplotlib (includes axes)
- SNR estimation
"""
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from .event_data import TaggedEvent, CSV_COLUMNS
from ..core.filename_parser import parse_pixel_filename, ParsedFilename

logger = logging.getLogger(__name__)

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False
    logger.warning("Matplotlib not available - image capture will be disabled")


class EventManager:
    """Manages tagged events, CSV persistence, and image capture."""

    def __init__(self):
        """Initialize the event manager."""
        self.events: List[TaggedEvent] = []
        self.csv_path: Optional[Path] = None
        self.image_dir: Optional[Path] = None
        self.next_id: int = 1
        self._session_active: bool = False

    @property
    def session_active(self) -> bool:
        """Check if a session is currently active."""
        return self._session_active and self.csv_path is not None

    @property
    def event_count(self) -> int:
        """Number of events in the current session."""
        return len(self.events)

    # =========================================================================
    # Session Management
    # =========================================================================

    def create_new_session(self, session_dir: Path) -> bool:
        """Create a new event tagging session.

        Creates:
        - session_dir/events.csv (new CSV file with headers)
        - session_dir/images/ (directory for event images)

        Args:
            session_dir: Directory for the session

        Returns:
            True if session created successfully
        """
        try:
            session_dir = Path(session_dir)
            session_dir.mkdir(parents=True, exist_ok=True)

            # Setup paths
            self.csv_path = session_dir / "events.csv"
            self.image_dir = session_dir / "images"
            self.image_dir.mkdir(exist_ok=True)

            # Reset state
            self.events = []
            self.next_id = 1

            # Write CSV header
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

            self._session_active = True
            logger.info(f"Created new event session at: {session_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            self._session_active = False
            return False

    def load_existing_session(self, csv_path: Path) -> bool:
        """Load an existing event session to continue tagging.

        Args:
            csv_path: Path to existing events.csv file

        Returns:
            True if session loaded successfully
        """
        try:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                logger.error(f"CSV file not found: {csv_path}")
                return False

            self.csv_path = csv_path
            self.image_dir = csv_path.parent / "images"
            self.image_dir.mkdir(exist_ok=True)

            # Load existing events
            self.events = []
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        event = TaggedEvent.from_csv_row(row)
                        self.events.append(event)
                    except Exception as e:
                        logger.warning(f"Failed to parse row: {e}")

            # Set next ID
            if self.events:
                self.next_id = max(e.id for e in self.events) + 1
            else:
                self.next_id = 1

            self._session_active = True
            logger.info(f"Loaded session with {len(self.events)} events from: {csv_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load session: {e}")
            self._session_active = False
            return False

    def close_session(self):
        """Close the current session."""
        self._session_active = False
        self.csv_path = None
        self.image_dir = None
        self.events = []
        self.next_id = 1
        logger.info("Event session closed")

    # =========================================================================
    # Event Operations
    # =========================================================================

    def add_event(
        self,
        audio_file: str,
        t_start: float,
        t_end: float,
        f_min: float,
        f_max: float,
        harmonic_number: Optional[int] = None,
        snr_estimate_db: Optional[float] = None,
        notes: str = ""
    ) -> TaggedEvent:
        """Add a new event to the session.

        Args:
            audio_file: Path to the audio file
            t_start: Event start time (seconds, relative to file)
            t_end: Event end time (seconds, relative to file)
            f_min: Lower frequency bound (Hz)
            f_max: Upper frequency bound (Hz)
            harmonic_number: Harmonic order (optional)
            snr_estimate_db: SNR in dB (optional)
            notes: Free-form notes (optional)

        Returns:
            Created TaggedEvent
        """
        # Ensure ordering
        t_start, t_end = min(t_start, t_end), max(t_start, t_end)
        f_min, f_max = min(f_min, f_max), max(f_min, f_max)

        # Try to parse filename for sensor info and absolute times
        parsed = parse_pixel_filename(audio_file)

        sensor_name = None
        sensor_id = None
        file_start_time = None
        file_end_time = None
        event_start_absolute = None
        event_end_absolute = None

        if parsed:
            sensor_name = parsed.sensor_name
            sensor_id = parsed.sensor_id
            file_start_time = parsed.start_time
            file_end_time = parsed.end_time
            event_start_absolute = parsed.compute_absolute_time(t_start)
            event_end_absolute = parsed.compute_absolute_time(t_end)

        # Create event
        event = TaggedEvent(
            id=self.next_id,
            audio_file=audio_file,
            sensor_name=sensor_name,
            sensor_id=sensor_id,
            t_start=t_start,
            t_end=t_end,
            f_min=f_min,
            f_max=f_max,
            file_start_time=file_start_time,
            file_end_time=file_end_time,
            event_start_absolute=event_start_absolute,
            event_end_absolute=event_end_absolute,
            harmonic_number=harmonic_number,
            snr_estimate_db=snr_estimate_db,
            notes=notes
        )

        self.events.append(event)
        self.next_id += 1

        logger.info(f"Added event {event.id}: t=[{t_start:.3f}, {t_end:.3f}]s, "
                   f"f=[{f_min:.0f}, {f_max:.0f}]Hz")

        return event

    def delete_event(self, event_id: int) -> bool:
        """Delete an event by ID.

        Note: This also removes the event from the CSV by rewriting it.

        Args:
            event_id: ID of event to delete

        Returns:
            True if event was found and deleted
        """
        for i, event in enumerate(self.events):
            if event.id == event_id:
                # Remove image file if exists
                if event.image_path:
                    try:
                        img_path = Path(event.image_path)
                        if img_path.exists():
                            img_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete image: {e}")

                self.events.pop(i)
                self._save_all_to_csv()  # Rewrite CSV without deleted event
                logger.info(f"Deleted event {event_id}")
                return True

        logger.warning(f"Event {event_id} not found")
        return False

    def update_event(self, event_id: int, **kwargs) -> bool:
        """Update event fields.

        Args:
            event_id: ID of event to update
            **kwargs: Fields to update (harmonic_number, snr_estimate_db, notes, etc.)

        Returns:
            True if event was found and updated
        """
        for event in self.events:
            if event.id == event_id:
                for key, value in kwargs.items():
                    if hasattr(event, key):
                        setattr(event, key, value)
                self._save_all_to_csv()
                logger.info(f"Updated event {event_id}: {kwargs}")
                return True

        logger.warning(f"Event {event_id} not found")
        return False

    def get_event(self, event_id: int) -> Optional[TaggedEvent]:
        """Get event by ID."""
        for event in self.events:
            if event.id == event_id:
                return event
        return None

    def get_events_for_file(self, audio_file: str) -> List[TaggedEvent]:
        """Get all events for a specific audio file.

        Args:
            audio_file: Path or filename to match

        Returns:
            List of events for that file
        """
        filename = Path(audio_file).name
        return [e for e in self.events if Path(e.audio_file).name == filename]

    # =========================================================================
    # CSV Persistence
    # =========================================================================

    def append_event_to_csv(self, event: TaggedEvent) -> bool:
        """Append a single event to the CSV file (incremental save).

        Args:
            event: Event to append

        Returns:
            True if successful
        """
        if not self.csv_path:
            logger.error("No active session")
            return False

        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writerow(event.to_csv_row())
            return True
        except Exception as e:
            logger.error(f"Failed to append event to CSV: {e}")
            return False

    def _save_all_to_csv(self) -> bool:
        """Rewrite entire CSV file (used after delete/update).

        Returns:
            True if successful
        """
        if not self.csv_path:
            logger.error("No active session")
            return False

        try:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                for event in self.events:
                    writer.writerow(event.to_csv_row())
            return True
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            return False

    # =========================================================================
    # SNR Estimation
    # =========================================================================

    def estimate_snr(
        self,
        S: np.ndarray,
        times: np.ndarray,
        freqs: np.ndarray,
        t_start: float,
        t_end: float,
        f_min: float,
        f_max: float
    ) -> Optional[float]:
        """Estimate SNR in dB for a given spectrogram region.

        Method: SNR = max_value - median_value (in dB)
        This gives a rough estimate of signal strength above noise floor.

        Args:
            S: Spectrogram data in dB [n_freqs, n_times]
            times: Time axis array
            freqs: Frequency axis array
            t_start, t_end: Time range
            f_min, f_max: Frequency range

        Returns:
            Estimated SNR in dB, or None if estimation fails
        """
        try:
            # Find indices for the region
            t_mask = (times >= t_start) & (times <= t_end)
            f_mask = (freqs >= f_min) & (freqs <= f_max)

            if not np.any(t_mask) or not np.any(f_mask):
                logger.warning("No data in specified region")
                return None

            # Extract region
            region = S[np.ix_(f_mask, t_mask)]

            if region.size == 0:
                return None

            # Calculate SNR estimate
            max_val = np.max(region)
            median_val = np.median(region)

            snr = max_val - median_val

            logger.debug(f"SNR estimate: max={max_val:.1f}dB, median={median_val:.1f}dB, "
                        f"SNR={snr:.1f}dB")

            return float(snr)

        except Exception as e:
            logger.error(f"SNR estimation failed: {e}")
            return None

    # =========================================================================
    # Image Capture
    # =========================================================================

    def capture_event_image(
        self,
        event: TaggedEvent,
        S: np.ndarray,
        times: np.ndarray,
        freqs: np.ndarray,
        colormap: str = 'plasma',
        padding_fraction: float = 0.1,
        dpi: int = 150
    ) -> Optional[str]:
        """Capture spectrogram region as image with axes using matplotlib.

        Args:
            event: Event to capture
            S: Full spectrogram data in dB [n_freqs, n_times]
            times: Time axis array
            freqs: Frequency axis array
            colormap: Matplotlib colormap name
            padding_fraction: Extra time padding on each side (fraction of event duration)
            dpi: Image resolution

        Returns:
            Path to saved image, or None if capture fails
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available for image capture")
            return None

        if self.image_dir is None:
            logger.error("No image directory configured")
            return None

        try:
            # Calculate padded time range
            duration = event.t_end - event.t_start
            padding = duration * padding_fraction
            t_start_padded = max(event.t_start - padding, times[0])
            t_end_padded = min(event.t_end + padding, times[-1])

            # Find indices
            t_mask = (times >= t_start_padded) & (times <= t_end_padded)
            f_mask = (freqs >= event.f_min) & (freqs <= event.f_max)

            if not np.any(t_mask) or not np.any(f_mask):
                logger.warning("No data in specified region")
                return None

            # Extract region
            region = S[np.ix_(f_mask, t_mask)]
            region_times = times[t_mask]
            region_freqs = freqs[f_mask]

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot spectrogram
            extent = [region_times[0], region_times[-1],
                     region_freqs[0], region_freqs[-1]]

            im = ax.imshow(
                region,
                aspect='auto',
                origin='lower',
                extent=extent,
                cmap=colormap,
                interpolation='nearest'
            )

            # Draw vertical lines at event boundaries
            ax.axvline(x=event.t_start, color='lime', linewidth=2, linestyle='--',
                      label='Event Start')
            ax.axvline(x=event.t_end, color='lime', linewidth=2, linestyle='--',
                      label='Event End')

            # Labels and title
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Frequency (Hz)')

            title_parts = [f'Event {event.id}']
            if event.sensor_name and event.sensor_id:
                title_parts.append(f'{event.sensor_name} {event.sensor_id}')
            if event.harmonic_number:
                title_parts.append(f'H{event.harmonic_number}')
            if event.snr_estimate_db is not None:
                title_parts.append(f'SNR: {event.snr_estimate_db:.1f} dB')

            ax.set_title(' | '.join(title_parts))

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Power (dB)')

            # Add timestamp info if available (start time in top-left, end time in top-right)
            if event.event_start_absolute:
                start_time_str = event.event_start_absolute.strftime("%Y-%m-%d %H:%M:%S")
                ax.text(0.02, 0.98, start_time_str, transform=ax.transAxes,
                       fontsize=8, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            if event.event_end_absolute:
                end_time_str = event.event_end_absolute.strftime("%Y-%m-%d %H:%M:%S")
                ax.text(0.98, 0.98, end_time_str, transform=ax.transAxes,
                       fontsize=8, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Generate image filename: pixel_id - start_datetime - end_time
            # Format: 1001 - 2025-10-29 11_21_48 - 11_22_35.png
            if event.sensor_id and event.event_start_absolute and event.event_end_absolute:
                start_dt_str = event.event_start_absolute.strftime("%Y-%m-%d %H_%M_%S")
                end_time_str = event.event_end_absolute.strftime("%H_%M_%S")
                image_filename = f"{event.sensor_id} - {start_dt_str} - {end_time_str}.png"
            else:
                # Fallback to old format if timestamps not available
                image_filename = f"event_{event.id:04d}.png"
            image_path = self.image_dir / image_filename

            fig.savefig(image_path, dpi=dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close(fig)

            # Update event with image path
            event.image_path = str(image_path)

            logger.info(f"Captured event image: {image_path}")
            return str(image_path)

        except Exception as e:
            logger.error(f"Image capture failed: {e}", exc_info=True)
            return None

    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session.

        Returns:
            Dictionary with session info
        """
        return {
            'active': self._session_active,
            'csv_path': str(self.csv_path) if self.csv_path else None,
            'image_dir': str(self.image_dir) if self.image_dir else None,
            'event_count': len(self.events),
            'next_id': self.next_id
        }
