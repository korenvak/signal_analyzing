"""
Dialog for selecting detected tracks from automatic track detection.
"""
import numpy as np
from typing import List, Optional, Tuple

from .qt_compat import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter,
    QWidget, QFrame, QMessageBox, QProgressDialog,
    Qt, Signal, QColor, is_pyqt5
)

import matplotlib
# Use Qt5Agg for PyQt5, QtAgg for PySide6
if is_pyqt5():
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
else:
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from audio_visualizer.core.auto_detector import DetectedTrack, AutomaticTrackDetector

import logging
logger = logging.getLogger(__name__)


class TrackPreviewCanvas(FigureCanvas):
    """Canvas for displaying spectrogram with detected tracks overlay."""

    track_selected = Signal(int)  # Emits track index when clicked

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)

        self.spectrogram = None
        self.times = None
        self.freqs = None
        self.tracks: List[DetectedTrack] = []
        self.selected_track_idx: Optional[int] = None

        # Track colors
        self.track_colors = [
            '#00FF00',  # Green (selected)
            '#FF6B6B',  # Red
            '#4ECDC4',  # Teal
            '#FFE66D',  # Yellow
            '#95E1D3',  # Light teal
            '#F38181',  # Coral
            '#AA96DA',  # Purple
            '#FCBAD3',  # Pink
        ]

        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def set_data(self, spectrogram: np.ndarray, times: np.ndarray, freqs: np.ndarray,
                 tracks: List[DetectedTrack]):
        """Set spectrogram data and detected tracks."""
        self.spectrogram = spectrogram
        self.times = times
        self.freqs = freqs
        self.tracks = tracks
        self.selected_track_idx = 0 if tracks else None
        
        # Debug logging
        logger.info(f"TrackPreviewCanvas.set_data: spectrogram shape={spectrogram.shape if spectrogram is not None else None}, "
                   f"times len={len(times) if times is not None else 0}, "
                   f"freqs len={len(freqs) if freqs is not None else 0}, "
                   f"tracks count={len(tracks)}")
        
        if tracks:
            for i, track in enumerate(tracks):
                logger.info(f"  Track {i}: {len(track.points) if track.points else 0} points, "
                           f"score={track.score:.3f}")
                if track.points and len(track.points) > 0:
                    t_vals = [p[0] for p in track.points]
                    f_vals = [p[1] for p in track.points]
                    logger.debug(f"    Track {i} range: t=[{min(t_vals):.3f}, {max(t_vals):.3f}], "
                               f"f=[{min(f_vals):.0f}, {max(f_vals):.0f}]")
        
        self._update_plot()

    def select_track(self, idx: int):
        """Select a track by index."""
        if 0 <= idx < len(self.tracks):
            self.selected_track_idx = idx
            self._update_plot()

    def _update_plot(self):
        """Redraw the plot."""
        self.ax.clear()

        if self.spectrogram is None:
            self.draw()
            return

        # Plot spectrogram
        extent = [self.times[0], self.times[-1], self.freqs[0], self.freqs[-1]]
        im = self.ax.imshow(
            self.spectrogram,
            origin='lower',
            aspect='auto',
            extent=extent,
            cmap='magma',
            interpolation='nearest',
            zorder=0  # Spectrogram on bottom layer
        )

        # Plot tracks (on top of spectrogram)
        for i, track in enumerate(self.tracks):
            if not track.points or len(track.points) == 0:
                logger.debug(f"Track {i} has no points or empty points list")
                continue

            t_vals = [p[0] for p in track.points]
            f_vals = [p[1] for p in track.points]

            # Validate coordinates are within extent
            if len(t_vals) == 0 or len(f_vals) == 0:
                logger.warning(f"Track {i} has empty coordinate arrays")
                continue

            logger.info(f"Plotting track {i}: {len(t_vals)} points, "
                        f"t_range=[{min(t_vals):.3f}, {max(t_vals):.3f}], "
                        f"f_range=[{min(f_vals):.0f}, {max(f_vals):.0f}]")
            logger.info(f"  Plot extent: times=[{self.times[0]:.3f}, {self.times[-1]:.3f}], "
                       f"freqs=[{self.freqs[0]:.0f}, {self.freqs[-1]:.0f}]")

            if i == self.selected_track_idx:
                # Selected track - thick bright green line with markers
                self.ax.plot(t_vals, f_vals, color='#00FF00', linewidth=5,
                           marker='o', markersize=6, markevery=max(1, len(t_vals)//15),
                           zorder=10, label=f'Track {i+1} (selected)')
                # Also plot markers separately for better visibility
                self.ax.scatter(t_vals, f_vals, color='#00FF00', s=50,
                              edgecolors='white', linewidths=1.5, zorder=11, alpha=0.9)
            else:
                # Other tracks - colored lines with markers
                color = self.track_colors[(i % (len(self.track_colors) - 1)) + 1]
                self.ax.plot(t_vals, f_vals, color=color, linewidth=4,
                           marker='o', markersize=5, markevery=max(1, len(t_vals)//20),
                           alpha=0.95, zorder=8, label=f'Track {i+1}')
                # Markers for other tracks
                self.ax.scatter(t_vals, f_vals, color=color, s=40,
                              edgecolors='white', linewidths=1, zorder=9, alpha=0.85)

        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Frequency (Hz)')
        self.ax.set_title('Detected Tracks (click to select)')

        # Set axis limits to match extent
        self.ax.set_xlim(self.times[0], self.times[-1])
        self.ax.set_ylim(self.freqs[0], self.freqs[-1])

        if self.tracks:
            self.ax.legend(loc='upper right', fontsize=8)
            # Force display of all tracks by adding text labels at track positions
            for i, track in enumerate(self.tracks):
                if track.points and len(track.points) > 0:
                    t_mid = track.points[len(track.points)//2][0]
                    f_mid = track.points[len(track.points)//2][1]
                    self.ax.text(t_mid, f_mid, f'T{i+1}', color='white', fontsize=10,
                               fontweight='bold', ha='center', va='center',
                               bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
                               zorder=15)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.draw()

    def _on_click(self, event):
        """Handle mouse click to select nearest track."""
        if event.inaxes != self.ax or not self.tracks:
            return

        click_t = event.xdata
        click_f = event.ydata

        if click_t is None or click_f is None:
            return

        # Find nearest track
        min_dist = float('inf')
        nearest_idx = 0

        for i, track in enumerate(self.tracks):
            for t, f in track.points:
                # Normalize distance (time and freq have different scales)
                t_range = self.times[-1] - self.times[0]
                f_range = self.freqs[-1] - self.freqs[0]

                dt = (t - click_t) / t_range if t_range > 0 else 0
                df = (f - click_f) / f_range if f_range > 0 else 0
                dist = np.sqrt(dt**2 + df**2)

                if dist < min_dist:
                    min_dist = dist
                    nearest_idx = i

        if nearest_idx != self.selected_track_idx:
            self.selected_track_idx = nearest_idx
            self._update_plot()
            self.track_selected.emit(nearest_idx)


class AutoDetectDialog(QDialog):
    """Dialog for running auto-detection and selecting a track."""

    def __init__(self, parent=None,
                 spectrogram: np.ndarray = None,
                 times: np.ndarray = None,
                 freqs: np.ndarray = None,
                 t_start: float = 0,
                 t_end: float = 1,
                 f_min: float = 0,
                 f_max: float = 22050,
                 sample_rate: float = 44100,
                 hop_length: int = 512):
        super().__init__(parent)

        self.spectrogram = spectrogram
        self.times = times
        self.freqs = freqs
        self.t_start = t_start
        self.t_end = t_end
        self.f_min = f_min
        self.f_max = f_max
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        self.detected_tracks: List[DetectedTrack] = []
        self.selected_track: Optional[DetectedTrack] = None

        self._setup_ui()
        self._run_detection()

    def _setup_ui(self):
        """Setup dialog UI."""
        self.setWindowTitle("Auto Detect Track")
        self.setMinimumSize(900, 600)
        self.resize(1000, 700)

        layout = QVBoxLayout(self)

        # Info label
        info_label = QLabel(
            f"Region: t=[{self.t_start:.2f}, {self.t_end:.2f}]s, "
            f"f=[{self.f_min:.0f}, {self.f_max:.0f}]Hz"
        )
        layout.addWidget(info_label)

        # Splitter for canvas and table
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Preview canvas
        self.canvas = TrackPreviewCanvas(self)
        self.canvas.track_selected.connect(self._on_track_selected)
        splitter.addWidget(self.canvas)

        # Track table
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 0, 0)

        table_label = QLabel("Detected Tracks:")
        table_layout.addWidget(table_label)

        self.track_table = QTableWidget()
        self.track_table.setColumnCount(5)
        self.track_table.setHorizontalHeaderLabels([
            "Track", "Score", "Duration (s)", "Freq Range (Hz)", "Points"
        ])
        self.track_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.track_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.track_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.track_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        table_layout.addWidget(self.track_table)

        splitter.addWidget(table_container)
        splitter.setSizes([500, 200])

        layout.addWidget(splitter)

        # Status label
        self.status_label = QLabel("Detecting tracks...")
        layout.addWidget(self.status_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.ok_button = QPushButton("Use Selected Track")
        self.ok_button.setEnabled(False)
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

    def _run_detection(self):
        """Run the automatic track detection."""
        if self.spectrogram is None:
            self.status_label.setText("Error: No spectrogram data")
            return

        # Debug: log data info
        logger.info(f"AutoDetectDialog: spectrogram shape={self.spectrogram.shape}, "
                   f"times len={len(self.times) if self.times is not None else 0}, "
                   f"freqs len={len(self.freqs) if self.freqs is not None else 0}")
        logger.info(f"AutoDetectDialog: region bounds t=[{self.t_start:.2f}, {self.t_end:.2f}], "
                   f"f=[{self.f_min:.0f}, {self.f_max:.0f}]")
        if self.times is not None and len(self.times) > 0:
            logger.info(f"AutoDetectDialog: times range=[{self.times[0]:.2f}, {self.times[-1]:.2f}]")
        if self.freqs is not None and len(self.freqs) > 0:
            logger.info(f"AutoDetectDialog: freqs range=[{self.freqs[0]:.0f}, {self.freqs[-1]:.0f}]")

        detector = AutomaticTrackDetector()

        try:
            self.detected_tracks = detector.detect_in_region(
                spectrogram=self.spectrogram,
                times=self.times,
                freqs=self.freqs,
                t_start=self.t_start,
                t_end=self.t_end,
                f_min=self.f_min,
                f_max=self.f_max,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length
            )
        except Exception as e:
            logger.error(f"Detection failed: {e}", exc_info=True)
            self.status_label.setText(f"Detection failed: {e}")
            return

        if not self.detected_tracks:
            self.status_label.setText("No tracks detected in this region")
            # Still show the spectrogram preview even if no tracks found
            self._update_preview_no_tracks()
            return

        # Update UI
        self._populate_table()
        self._update_preview()

        self.status_label.setText(f"Found {len(self.detected_tracks)} track(s)")
        self.ok_button.setEnabled(True)

        # Select first track
        if self.detected_tracks:
            self.track_table.selectRow(0)
            self.selected_track = self.detected_tracks[0]

    def _populate_table(self):
        """Populate the track table."""
        self.track_table.setRowCount(len(self.detected_tracks))

        for i, track in enumerate(self.detected_tracks):
            self.track_table.setItem(i, 0, QTableWidgetItem(f"Track {i+1}"))
            self.track_table.setItem(i, 1, QTableWidgetItem(f"{track.score:.3f}"))
            self.track_table.setItem(i, 2, QTableWidgetItem(f"{track.duration:.2f}"))
            self.track_table.setItem(i, 3, QTableWidgetItem(
                f"{track.freq_range[0]:.0f} - {track.freq_range[1]:.0f}"
            ))
            self.track_table.setItem(i, 4, QTableWidgetItem(str(track.point_count)))

    def _update_preview(self):
        """Update the preview canvas."""
        # Extract region for display
        detector = AutomaticTrackDetector()
        region, region_times, region_freqs = detector._extract_region(
            self.spectrogram, self.times, self.freqs,
            self.t_start, self.t_end, self.f_min, self.f_max
        )

        # Debug: log region and track info
        logger.info(f"_update_preview: region shape={region.shape if region is not None else None}")
        if region_times is not None and len(region_times) > 0:
            logger.info(f"_update_preview: region_times=[{region_times[0]:.3f}, {region_times[-1]:.3f}]")
        if region_freqs is not None and len(region_freqs) > 0:
            logger.info(f"_update_preview: region_freqs=[{region_freqs[0]:.1f}, {region_freqs[-1]:.1f}]")

        for i, track in enumerate(self.detected_tracks):
            if track.points and len(track.points) > 0:
                t_vals = [p[0] for p in track.points]
                f_vals = [p[1] for p in track.points]
                logger.info(f"_update_preview: Track {i} points range: "
                           f"t=[{min(t_vals):.3f}, {max(t_vals):.3f}], "
                           f"f=[{min(f_vals):.1f}, {max(f_vals):.1f}]")

        self.canvas.set_data(region, region_times, region_freqs, self.detected_tracks)

    def _update_preview_no_tracks(self):
        """Update the preview canvas even when no tracks detected (for debugging)."""
        detector = AutomaticTrackDetector()
        region, region_times, region_freqs = detector._extract_region(
            self.spectrogram, self.times, self.freqs,
            self.t_start, self.t_end, self.f_min, self.f_max
        )

        if region is not None and region.size > 0:
            logger.info(f"Preview region: shape={region.shape}, "
                       f"value range=[{region.min():.2f}, {region.max():.2f}]")
            self.canvas.set_data(region, region_times, region_freqs, [])
        else:
            logger.warning("Could not extract region for preview")

    def _on_table_selection_changed(self):
        """Handle table selection change."""
        rows = self.track_table.selectionModel().selectedRows()
        if rows:
            idx = rows[0].row()
            if 0 <= idx < len(self.detected_tracks):
                self.selected_track = self.detected_tracks[idx]
                self.canvas.select_track(idx)

    def _on_track_selected(self, idx: int):
        """Handle track selection from canvas click."""
        if 0 <= idx < len(self.detected_tracks):
            self.selected_track = self.detected_tracks[idx]
            self.track_table.selectRow(idx)

    def get_selected_track(self) -> Optional[DetectedTrack]:
        """Get the selected track, or None if cancelled."""
        return self.selected_track


def run_auto_detect_dialog(parent,
                           spectrogram: np.ndarray,
                           times: np.ndarray,
                           freqs: np.ndarray,
                           t_start: float,
                           t_end: float,
                           f_min: float,
                           f_max: float,
                           sample_rate: float = 44100,
                           hop_length: int = 512) -> Optional[List[Tuple[float, float]]]:
    """
    Run the auto-detect dialog and return selected track points.

    Returns:
        List of (time, freq) tuples, or None if cancelled/no detection.
    """
    dialog = AutoDetectDialog(
        parent=parent,
        spectrogram=spectrogram,
        times=times,
        freqs=freqs,
        t_start=t_start,
        t_end=t_end,
        f_min=f_min,
        f_max=f_max,
        sample_rate=sample_rate,
        hop_length=hop_length
    )

    if dialog.exec() == QDialog.DialogCode.Accepted:
        track = dialog.get_selected_track()
        if track:
            return track.points

    return None
