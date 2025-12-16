"""
Range selector widget for DAS data.

Allows selecting:
- Sensor range (integer IDs)
- Time range (HH:MM:SS format)
- Frequency filter range (optional)
- Resolution/quality preset
"""

import logging
from typing import Optional, Tuple
from datetime import datetime, timedelta

from .qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QSpinBox, QLineEdit, QComboBox, QPushButton, QGroupBox,
    QDoubleSpinBox, QCheckBox, QSlider, QFrame,
    Qt, Signal
)

from ..core.das_data_provider import DASFolderMetadata, DASDataRequest
from ..core.time_formatter import TimeFormatter

logger = logging.getLogger(__name__)


class TimeRangeEdit(QWidget):
    """Widget for editing a time value in HH:MM:SS format."""

    value_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Hours
        self.hours = QSpinBox()
        self.hours.setRange(0, 99)
        self.hours.setFixedWidth(45)
        self.hours.valueChanged.connect(self._emit_changed)
        layout.addWidget(self.hours)

        layout.addWidget(QLabel(":"))

        # Minutes
        self.minutes = QSpinBox()
        self.minutes.setRange(0, 59)
        self.minutes.setFixedWidth(45)
        self.minutes.valueChanged.connect(self._emit_changed)
        layout.addWidget(self.minutes)

        layout.addWidget(QLabel(":"))

        # Seconds
        self.seconds = QSpinBox()
        self.seconds.setRange(0, 59)
        self.seconds.setFixedWidth(45)
        self.seconds.valueChanged.connect(self._emit_changed)
        layout.addWidget(self.seconds)

    def _emit_changed(self):
        self.value_changed.emit()

    def get_total_seconds(self) -> float:
        """Get time value as total seconds."""
        return (self.hours.value() * 3600 +
                self.minutes.value() * 60 +
                self.seconds.value())

    def set_total_seconds(self, seconds: float):
        """Set time from total seconds."""
        self.hours.setValue(int(seconds // 3600))
        self.minutes.setValue(int((seconds % 3600) // 60))
        self.seconds.setValue(int(seconds % 60))

    def get_timedelta(self) -> timedelta:
        """Get time value as timedelta."""
        return timedelta(seconds=self.get_total_seconds())

    def set_from_datetime(self, dt: datetime, base: datetime):
        """Set from datetime relative to base."""
        delta = (dt - base).total_seconds()
        self.set_total_seconds(max(0, delta))

    def get_as_datetime(self, base: datetime) -> datetime:
        """Get as datetime relative to base."""
        return base + self.get_timedelta()


class RangeSelectorWidget(QWidget):
    """
    Widget for selecting sensor and time ranges for waterfall computation.

    Emits compute_requested signal with DASDataRequest when user clicks compute.
    """

    # Signals
    compute_requested = Signal(object)  # DASDataRequest
    ranges_changed = Signal()            # Emitted when any range changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self._metadata: Optional[DASFolderMetadata] = None
        self._time_formatter: Optional[TimeFormatter] = None
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Sensor range group
        sensor_group = QGroupBox("Sensor Range")
        sensor_layout = QGridLayout(sensor_group)

        sensor_layout.addWidget(QLabel("Start:"), 0, 0)
        self.sensor_start = QSpinBox()
        self.sensor_start.setRange(0, 100000)
        self.sensor_start.valueChanged.connect(self._on_range_changed)
        sensor_layout.addWidget(self.sensor_start, 0, 1)

        sensor_layout.addWidget(QLabel("End:"), 0, 2)
        self.sensor_end = QSpinBox()
        self.sensor_end.setRange(0, 100000)
        self.sensor_end.valueChanged.connect(self._on_range_changed)
        sensor_layout.addWidget(self.sensor_end, 0, 3)

        # Quick select buttons
        quick_layout = QHBoxLayout()
        self.all_sensors_btn = QPushButton("All")
        self.all_sensors_btn.clicked.connect(self._select_all_sensors)
        quick_layout.addWidget(self.all_sensors_btn)

        self.first_100_btn = QPushButton("First 100")
        self.first_100_btn.clicked.connect(lambda: self._select_sensor_range(0, 100))
        quick_layout.addWidget(self.first_100_btn)

        self.first_500_btn = QPushButton("First 500")
        self.first_500_btn.clicked.connect(lambda: self._select_sensor_range(0, 500))
        quick_layout.addWidget(self.first_500_btn)

        sensor_layout.addLayout(quick_layout, 1, 0, 1, 4)

        layout.addWidget(sensor_group)

        # Time range group
        time_group = QGroupBox("Time Range")
        time_layout = QGridLayout(time_group)

        time_layout.addWidget(QLabel("Start:"), 0, 0)
        self.time_start = TimeRangeEdit()
        self.time_start.value_changed.connect(self._on_range_changed)
        time_layout.addWidget(self.time_start, 0, 1)

        time_layout.addWidget(QLabel("End:"), 1, 0)
        self.time_end = TimeRangeEdit()
        self.time_end.value_changed.connect(self._on_range_changed)
        time_layout.addWidget(self.time_end, 1, 1)

        # Time quick select
        time_quick_layout = QHBoxLayout()
        self.first_min_btn = QPushButton("First 1m")
        self.first_min_btn.clicked.connect(lambda: self._select_time_duration(60))
        time_quick_layout.addWidget(self.first_min_btn)

        self.first_5min_btn = QPushButton("First 5m")
        self.first_5min_btn.clicked.connect(lambda: self._select_time_duration(300))
        time_quick_layout.addWidget(self.first_5min_btn)

        self.all_time_btn = QPushButton("All")
        self.all_time_btn.clicked.connect(self._select_all_time)
        time_quick_layout.addWidget(self.all_time_btn)

        time_layout.addLayout(time_quick_layout, 2, 0, 1, 2)

        layout.addWidget(time_group)

        # Resolution/Quality group
        quality_group = QGroupBox("Resolution")
        quality_layout = QVBoxLayout(quality_group)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "Full Resolution (1x)",
            "Medium (2x downsample)",
            "Low (4x downsample)",
            "Overview (8x downsample)"
        ])
        self.resolution_combo.setCurrentIndex(0)
        quality_layout.addWidget(self.resolution_combo)

        # Estimated size label
        self.size_label = QLabel("Estimated: -- MB")
        self.size_label.setStyleSheet("color: #888; font-size: 11px;")
        quality_layout.addWidget(self.size_label)

        layout.addWidget(quality_group)

        # Frequency filter (optional, for single-sensor spectrogram)
        freq_group = QGroupBox("Frequency Filter (Optional)")
        freq_group.setCheckable(True)
        freq_group.setChecked(False)
        freq_layout = QGridLayout(freq_group)

        freq_layout.addWidget(QLabel("Min (Hz):"), 0, 0)
        self.freq_min = QDoubleSpinBox()
        self.freq_min.setRange(0, 100000)
        self.freq_min.setValue(0)
        freq_layout.addWidget(self.freq_min, 0, 1)

        freq_layout.addWidget(QLabel("Max (Hz):"), 0, 2)
        self.freq_max = QDoubleSpinBox()
        self.freq_max.setRange(0, 100000)
        self.freq_max.setValue(500)  # Default Nyquist for 1kHz
        freq_layout.addWidget(self.freq_max, 0, 3)

        self.freq_group = freq_group
        layout.addWidget(freq_group)

        # Compute button
        self.compute_btn = QPushButton("Compute Waterfall")
        self.compute_btn.clicked.connect(self._on_compute_clicked)
        self.compute_btn.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """)
        layout.addWidget(self.compute_btn)

        layout.addStretch()

    def set_metadata(self, metadata: DASFolderMetadata):
        """Set metadata and update widget limits."""
        self._metadata = metadata
        self._time_formatter = TimeFormatter(sample_rate=metadata.sample_rate)

        # Update sensor range limits
        self.sensor_start.setRange(metadata.sensor_range[0], metadata.sensor_range[1])
        self.sensor_end.setRange(metadata.sensor_range[0], metadata.sensor_range[1])
        self.sensor_start.setValue(metadata.sensor_range[0])
        self.sensor_end.setValue(min(metadata.sensor_range[1], metadata.sensor_range[0] + 500))

        # Update frequency max based on sample rate (Nyquist)
        nyquist = metadata.sample_rate / 2
        self.freq_max.setRange(0, nyquist)
        self.freq_max.setValue(nyquist)

        # Set default time range (first available segment, first 5 minutes)
        if metadata.time_ranges:
            first_start, first_end = metadata.time_ranges[0]
            max_duration = min(300, (first_end - first_start).total_seconds())  # Max 5 min

            self.time_start.set_total_seconds(0)
            self.time_end.set_total_seconds(max_duration)

        self._update_size_estimate()

    def _on_range_changed(self):
        """Handle range change."""
        self._update_size_estimate()
        self.ranges_changed.emit()

    def _update_size_estimate(self):
        """Update estimated data size."""
        if self._metadata is None:
            return

        n_sensors = self.sensor_end.value() - self.sensor_start.value()
        duration_sec = self.time_end.get_total_seconds() - self.time_start.get_total_seconds()
        n_samples = int(duration_sec * self._metadata.sample_rate)

        # Apply downsampling
        downsample = [1, 2, 4, 8][self.resolution_combo.currentIndex()]
        n_samples_ds = n_samples // downsample

        # Size in bytes (float32)
        size_bytes = n_sensors * n_samples_ds * 4
        size_mb = size_bytes / (1024 * 1024)

        if size_mb > 1000:
            self.size_label.setText(f"Estimated: {size_mb/1024:.1f} GB")
            if size_mb > 4000:
                self.size_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")
            else:
                self.size_label.setStyleSheet("color: #ffa500; font-size: 11px;")
        else:
            self.size_label.setText(f"Estimated: {size_mb:.1f} MB")
            self.size_label.setStyleSheet("color: #888; font-size: 11px;")

    def _select_all_sensors(self):
        """Select all sensors."""
        if self._metadata:
            self.sensor_start.setValue(self._metadata.sensor_range[0])
            self.sensor_end.setValue(self._metadata.sensor_range[1])

    def _select_sensor_range(self, start_offset: int, count: int):
        """Select a range of sensors from start."""
        if self._metadata:
            start = self._metadata.sensor_range[0] + start_offset
            end = min(start + count, self._metadata.sensor_range[1])
            self.sensor_start.setValue(start)
            self.sensor_end.setValue(end)

    def _select_all_time(self):
        """Select all available time."""
        if self._metadata and self._metadata.time_ranges:
            first_start, _ = self._metadata.time_ranges[0]
            _, last_end = self._metadata.time_ranges[-1]
            total_seconds = (last_end - first_start).total_seconds()
            self.time_start.set_total_seconds(0)
            self.time_end.set_total_seconds(total_seconds)

    def _select_time_duration(self, duration_seconds: float):
        """Select first N seconds of time."""
        self.time_start.set_total_seconds(0)
        self.time_end.set_total_seconds(duration_seconds)

    def _on_compute_clicked(self):
        """Handle compute button click."""
        if self._metadata is None:
            return

        # Build request
        downsample = [1, 2, 4, 8][self.resolution_combo.currentIndex()]

        # Get base time
        if self._metadata.time_ranges:
            base_time = self._metadata.time_ranges[0][0]
        else:
            base_time = datetime.now()

        time_start_dt = base_time + timedelta(seconds=self.time_start.get_total_seconds())
        time_end_dt = base_time + timedelta(seconds=self.time_end.get_total_seconds())

        request = DASDataRequest(
            sensor_start=self.sensor_start.value(),
            sensor_end=self.sensor_end.value(),
            time_start=time_start_dt,
            time_end=time_end_dt,
            downsample_factor=downsample
        )

        logger.info(f"Compute requested: sensors [{request.sensor_start}-{request.sensor_end}], "
                   f"time [{time_start_dt}-{time_end_dt}], downsample={downsample}")

        self.compute_requested.emit(request)

    def get_frequency_filter(self) -> Optional[Tuple[float, float]]:
        """Get frequency filter range if enabled."""
        if self.freq_group.isChecked():
            return (self.freq_min.value(), self.freq_max.value())
        return None
