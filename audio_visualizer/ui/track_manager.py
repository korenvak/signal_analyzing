"""
Track manager for painted frequency tracks.

Handles:
- Track interpolation using monotone cubic splines (PCHIP)
- Amplitude/dB sampling from spectrograms
- CSV export (both summary and detailed data)
- PNG export with track overlay
- Track persistence and metadata
"""
import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from .track_data import PaintedTrack, CSV_COLUMNS, INTERPOLATED_DATA_COLUMNS
from ..core.filename_parser import parse_pixel_filename, ParsedFilename

logger = logging.getLogger(__name__)

# Optional imports
try:
    from scipy.interpolate import PchipInterpolator
    HAS_PCHIP = True
except ImportError:
    PchipInterpolator = None
    HAS_PCHIP = False
    logger.warning("scipy not available - will use linear interpolation instead of PCHIP")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False
    logger.warning("Matplotlib not available - image export will be disabled")


class TrackManager:
    """Manages painted frequency tracks, interpolation, and export."""

    def __init__(self):
        """Initialize the track manager."""
        self.tracks: List[PaintedTrack] = []
        self.csv_path: Optional[Path] = None
        self.image_dir: Optional[Path] = None
        self.next_id: int = 1
        self._session_active: bool = False

    @property
    def session_active(self) -> bool:
        """Check if a session is currently active."""
        return self._session_active and self.csv_path is not None

    @property
    def track_count(self) -> int:
        """Number of tracks in the current session."""
        return len(self.tracks)

    # =========================================================================
    # Session Management
    # =========================================================================

    def create_new_session(self, session_dir: Path) -> bool:
        """Create a new track tagging session.

        Creates:
        - session_dir/tracks.csv (new CSV file with headers)
        - session_dir/track_images/ (directory for track images)
        - session_dir/track_data/ (directory for detailed interpolated data CSVs)

        Args:
            session_dir: Directory for the session

        Returns:
            True if session created successfully
        """
        try:
            session_dir = Path(session_dir)
            session_dir.mkdir(parents=True, exist_ok=True)

            # Setup paths
            self.csv_path = session_dir / "tracks.csv"
            self.image_dir = session_dir / "track_images"
            self.data_dir = session_dir / "track_data"

            self.image_dir.mkdir(exist_ok=True)
            self.data_dir.mkdir(exist_ok=True)

            # Reset state
            self.tracks = []
            self.next_id = 1

            # Write CSV header
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
                writer.writeheader()

            self._session_active = True
            logger.info(f"Created new track session at: {session_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to create track session: {e}")
            self._session_active = False
            return False

    def load_existing_session(self, csv_path: Path) -> bool:
        """Load an existing track session.

        Args:
            csv_path: Path to existing tracks.csv file

        Returns:
            True if session loaded successfully
        """
        try:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                logger.error(f"CSV file not found: {csv_path}")
                return False

            self.csv_path = csv_path
            self.image_dir = csv_path.parent / "track_images"
            self.data_dir = csv_path.parent / "track_data"

            self.image_dir.mkdir(exist_ok=True)
            self.data_dir.mkdir(exist_ok=True)

            # Load existing tracks
            self.tracks = []
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        track = PaintedTrack.from_csv_row(row)
                        self.tracks.append(track)
                    except Exception as e:
                        logger.warning(f"Failed to parse track row: {e}")

            # Set next ID
            if self.tracks:
                self.next_id = max(t.id for t in self.tracks) + 1
            else:
                self.next_id = 1

            self._session_active = True
            logger.info(f"Loaded track session with {len(self.tracks)} tracks from: {csv_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load track session: {e}")
            self._session_active = False
            return False

    def close_session(self):
        """Close the current session."""
        self._session_active = False
        self.csv_path = None
        self.image_dir = None
        self.data_dir = None
        self.tracks = []
        self.next_id = 1
        logger.info("Track session closed")

    # =========================================================================
    # Track Interpolation (Monotone Cubic)
    # =========================================================================

    def interpolate_track(
        self,
        control_points: List[Tuple[float, float]],
        num_samples: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate track using monotone cubic interpolation (PCHIP).

        PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) preserves
        monotonicity and produces smooth curves without oscillations.

        Args:
            control_points: List of (time, frequency) tuples
            num_samples: Number of interpolated points to generate

        Returns:
            Tuple of (times, frequencies) arrays
        """
        if len(control_points) < 2:
            logger.warning("Need at least 2 control points for interpolation")
            return np.array([]), np.array([])

        # Sort by time
        sorted_points = sorted(control_points, key=lambda p: p[0])
        times = np.array([p[0] for p in sorted_points])
        freqs = np.array([p[1] for p in sorted_points])

        # Check for duplicate times (would break interpolation)
        if len(np.unique(times)) < len(times):
            logger.warning("Duplicate time values detected, removing duplicates")
            unique_indices = np.unique(times, return_index=True)[1]
            times = times[unique_indices]
            freqs = freqs[unique_indices]

        if len(times) < 2:
            logger.warning("Need at least 2 unique time points for interpolation")
            return np.array([]), np.array([])

        # Use PCHIP for monotone cubic interpolation if available
        if HAS_PCHIP:
            try:
                # PCHIP interpolator
                interpolator = PchipInterpolator(times, freqs)

                # Generate interpolated values
                t_interp = np.linspace(times[0], times[-1], num_samples)
                f_interp = interpolator(t_interp)

                logger.debug(f"PCHIP interpolation: {len(control_points)} control points -> {num_samples} samples")
                return t_interp, f_interp

            except Exception as e:
                logger.warning(f"PCHIP interpolation failed: {e}, falling back to linear")

        # Fallback to linear interpolation
        t_interp = np.linspace(times[0], times[-1], num_samples)
        f_interp = np.interp(t_interp, times, freqs)

        logger.debug(f"Linear interpolation: {len(control_points)} control points -> {num_samples} samples")
        return t_interp, f_interp

    def sample_amplitudes(
        self,
        S: np.ndarray,
        times_axis: np.ndarray,
        freqs_axis: np.ndarray,
        track_times: np.ndarray,
        track_freqs: np.ndarray
    ) -> np.ndarray:
        """Sample amplitude/dB values from spectrogram along the track.

        Args:
            S: Spectrogram data in dB [n_freqs, n_times]
            times_axis: Time axis array (seconds)
            freqs_axis: Frequency axis array (Hz)
            track_times: Interpolated track time values
            track_freqs: Interpolated track frequency values

        Returns:
            Array of amplitude/dB values along the track
        """
        amplitudes = []

        for t, f in zip(track_times, track_freqs):
            # Find nearest time index
            t_idx = np.argmin(np.abs(times_axis - t))

            # Find nearest frequency index
            f_idx = np.argmin(np.abs(freqs_axis - f))

            # Sample amplitude
            try:
                amp = S[f_idx, t_idx]
                amplitudes.append(amp)
            except IndexError:
                logger.warning(f"Index out of bounds: f_idx={f_idx}, t_idx={t_idx}")
                amplitudes.append(np.nan)

        return np.array(amplitudes)

    # =========================================================================
    # Track Creation
    # =========================================================================

    def create_track(
        self,
        control_points: List[Tuple[float, float]],
        audio_file: str,
        S: np.ndarray,
        times_axis: np.ndarray,
        freqs_axis: np.ndarray,
        num_interp_samples: int = 200,
        track_label: str = "",
        notes: str = ""
    ) -> Optional[PaintedTrack]:
        """Create a new track from control points.

        Args:
            control_points: User-clicked points [(time, freq), ...]
            audio_file: Path to audio file
            S: Spectrogram data in dB
            times_axis: Time axis from spectrogram
            freqs_axis: Frequency axis from spectrogram
            num_interp_samples: Number of interpolated samples
            track_label: User label for the track
            notes: User notes

        Returns:
            Created PaintedTrack or None if creation fails
        """
        if not self.session_active:
            logger.error("No active track session")
            return None

        if len(control_points) < 2:
            logger.error("Need at least 2 control points to create a track")
            return None

        try:
            # Interpolate track
            t_interp, f_interp = self.interpolate_track(control_points, num_interp_samples)

            if len(t_interp) == 0:
                logger.error("Track interpolation failed")
                return None

            # Sample amplitudes
            amplitudes = self.sample_amplitudes(S, times_axis, freqs_axis, t_interp, f_interp)

            # Calculate bounds
            times_all = [p[0] for p in control_points]
            freqs_all = [p[1] for p in control_points]
            t_start = min(times_all)
            t_end = max(times_all)
            f_min = min(freqs_all)
            f_max = max(freqs_all)

            # Parse filename for metadata
            parsed = parse_pixel_filename(Path(audio_file).name)
            sensor_id = parsed.sensor_id if parsed else None

            # Create track
            track = PaintedTrack(
                id=self.next_id,
                audio_file=audio_file,
                sensor_name='pixel' if sensor_id else None,
                sensor_id=sensor_id,
                control_points=control_points,
                interpolated_times=t_interp.tolist(),
                interpolated_freqs=f_interp.tolist(),
                interpolated_amplitudes=amplitudes.tolist(),
                t_start=t_start,
                t_end=t_end,
                f_min=f_min,
                f_max=f_max,
                track_label=track_label,
                notes=notes
            )

            # Set absolute timestamps if filename parseable
            if parsed and parsed.start_time:
                track.file_start_time = parsed.start_time
                track.file_end_time = parsed.end_time
                track.track_start_absolute = parsed.start_time + timedelta(seconds=t_start)
                track.track_end_absolute = parsed.start_time + timedelta(seconds=t_end)

            self.tracks.append(track)
            self.next_id += 1

            logger.info(f"Created track {track.id} with {len(control_points)} control points, "
                       f"{len(t_interp)} interpolated points")

            return track

        except Exception as e:
            logger.error(f"Failed to create track: {e}")
            return None

    # =========================================================================
    # Track Export
    # =========================================================================

    def save_track_summary_csv(self) -> bool:
        """Save track summary to CSV file.

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
                for track in self.tracks:
                    writer.writerow(track.to_csv_row())

            logger.info(f"Saved {len(self.tracks)} tracks to {self.csv_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save track summary CSV: {e}")
            return False

    def export_track_detailed_csv(self, track: PaintedTrack) -> bool:
        """Export detailed interpolated data for a single track to CSV.

        Creates a CSV with columns: track_id, audio_file, pixel, time_seconds, frequency_hz, amplitude_db

        Args:
            track: Track to export

        Returns:
            True if successful
        """
        if not self.data_dir:
            logger.error("No data directory configured")
            return False

        try:
            # Generate filename
            filename = f"track_{track.id:04d}_data.csv"
            csv_path = self.data_dir / filename

            # Write detailed data
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=INTERPOLATED_DATA_COLUMNS)
                writer.writeheader()

                for t, f, amp in zip(
                    track.interpolated_times,
                    track.interpolated_freqs,
                    track.interpolated_amplitudes
                ):
                    writer.writerow({
                        'track_id': track.id,
                        'audio_file': Path(track.audio_file).name,
                        'pixel': track.sensor_id if track.sensor_id else "",
                        'time_seconds': f"{t:.6f}",
                        'frequency_hz': f"{f:.2f}",
                        'amplitude_db': f"{amp:.2f}" if not np.isnan(amp) else ""
                    })

            logger.info(f"Exported detailed track data to {csv_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export detailed track CSV: {e}")
            return False

    def export_track_image(
        self,
        track: PaintedTrack,
        S: np.ndarray,
        times: np.ndarray,
        freqs: np.ndarray,
        colormap: str = 'plasma',
        padding_fraction: float = 0.1,
        dpi: int = 150
    ) -> Optional[str]:
        """Export spectrogram region with track overlay as PNG.

        Args:
            track: Track to export
            S: Full spectrogram data in dB [n_freqs, n_times]
            times: Time axis array
            freqs: Frequency axis array
            colormap: Matplotlib colormap name
            padding_fraction: Extra padding around track bounds
            dpi: Image resolution

        Returns:
            Path to saved image, or None if export fails
        """
        if not HAS_MATPLOTLIB:
            logger.error("Matplotlib not available for image export")
            return None

        if self.image_dir is None:
            logger.error("No image directory configured")
            return None

        try:
            # Calculate padded region
            duration = track.t_end - track.t_start
            f_range = track.f_max - track.f_min

            t_padding = duration * padding_fraction
            f_padding = f_range * padding_fraction

            t_start_padded = max(track.t_start - t_padding, times[0])
            t_end_padded = min(track.t_end + t_padding, times[-1])
            f_min_padded = max(track.f_min - f_padding, freqs[0])
            f_max_padded = min(track.f_max + f_padding, freqs[-1])

            # Find indices
            t_mask = (times >= t_start_padded) & (times <= t_end_padded)
            f_mask = (freqs >= f_min_padded) & (freqs <= f_max_padded)

            if not np.any(t_mask) or not np.any(f_mask):
                logger.warning("No data in specified region")
                return None

            # Extract region
            region = S[np.ix_(f_mask, t_mask)]
            region_times = times[t_mask]
            region_freqs = freqs[f_mask]

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))

            # Plot spectrogram
            extent = [region_times[0], region_times[-1],
                     region_freqs[0], region_freqs[-1]]

            im = ax.imshow(
                region,
                aspect='auto',
                origin='lower',
                extent=extent,
                cmap=colormap,
                interpolation='bilinear'
            )

            # Plot track (interpolated curve)
            if track.interpolated_times and track.interpolated_freqs:
                ax.plot(
                    track.interpolated_times,
                    track.interpolated_freqs,
                    color='cyan',
                    linewidth=2.5,
                    linestyle='-',
                    label='Interpolated Track',
                    zorder=10
                )

            # Plot control points
            if track.control_points:
                control_t = [p[0] for p in track.control_points]
                control_f = [p[1] for p in track.control_points]
                ax.scatter(
                    control_t,
                    control_f,
                    color='yellow',
                    s=100,
                    marker='o',
                    edgecolors='white',
                    linewidths=2,
                    label='Control Points',
                    zorder=11
                )

            # Labels and title
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Frequency (Hz)', fontsize=12)

            title_parts = [f'Track {track.id}']
            if track.track_label:
                title_parts.append(f'"{track.track_label}"')
            if track.sensor_id:
                title_parts.append(f'Pixel {track.sensor_id}')
            title_parts.append(f'{track.n_control_points} control points')

            ax.set_title(' | '.join(title_parts), fontsize=14, fontweight='bold')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, label='Amplitude (dB)')

            # Add legend
            ax.legend(loc='upper right', fontsize=10)

            # Add grid
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

            # Tight layout
            plt.tight_layout()

            # Generate filename
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"track_{track.id:04d}_{timestamp_str}.png"
            img_path = self.image_dir / filename

            # Save figure
            plt.savefig(img_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)

            # Update track with image path
            track.image_path = str(img_path)

            logger.info(f"Saved track image to {img_path}")
            return str(img_path)

        except Exception as e:
            logger.error(f"Failed to export track image: {e}")
            return None

    def save_track(
        self,
        track: PaintedTrack,
        S: np.ndarray,
        times: np.ndarray,
        freqs: np.ndarray,
        export_image: bool = True,
        export_detailed_csv: bool = True
    ) -> bool:
        """Save a track with all exports (summary CSV, detailed CSV, PNG).

        Args:
            track: Track to save
            S: Spectrogram data
            times: Time axis
            freqs: Frequency axis
            export_image: Whether to export PNG image
            export_detailed_csv: Whether to export detailed data CSV

        Returns:
            True if all operations successful
        """
        success = True

        # Export image
        if export_image:
            img_path = self.export_track_image(track, S, times, freqs)
            if img_path is None:
                logger.warning(f"Failed to export image for track {track.id}")
                success = False

        # Export detailed CSV
        if export_detailed_csv:
            if not self.export_track_detailed_csv(track):
                logger.warning(f"Failed to export detailed CSV for track {track.id}")
                success = False

        # Update summary CSV
        if not self.save_track_summary_csv():
            logger.warning(f"Failed to update summary CSV for track {track.id}")
            success = False

        return success

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_tracks_for_file(self, audio_file: str) -> List[PaintedTrack]:
        """Get all tracks for a specific audio file.

        Args:
            audio_file: Audio file path or name

        Returns:
            List of tracks for this file
        """
        filename = Path(audio_file).name
        return [t for t in self.tracks if Path(t.audio_file).name == filename]

    def delete_track(self, track_id: int) -> bool:
        """Delete a track by ID.

        Args:
            track_id: Track ID to delete

        Returns:
            True if deleted successfully
        """
        for i, track in enumerate(self.tracks):
            if track.id == track_id:
                # Delete associated files
                if track.image_path and Path(track.image_path).exists():
                    try:
                        Path(track.image_path).unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete image: {e}")

                # Delete detailed CSV
                if self.data_dir:
                    csv_path = self.data_dir / f"track_{track.id:04d}_data.csv"
                    if csv_path.exists():
                        try:
                            csv_path.unlink()
                        except Exception as e:
                            logger.warning(f"Failed to delete detailed CSV: {e}")

                self.tracks.pop(i)
                self.save_track_summary_csv()
                logger.info(f"Deleted track {track_id}")
                return True

        logger.warning(f"Track {track_id} not found")
        return False
