"""
Detector parameters dialog for full spectrogram Doppler track detection.
"""
import logging
from typing import Optional

from .qt_compat import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QDialogButtonBox, QProgressBar, Qt, Signal
)

logger = logging.getLogger(__name__)


class DetectorParamsDialog(QDialog):
    """Dialog for configuring Doppler track detector parameters."""

    def __init__(self, parent=None, detector=None):
        """
        Initialize the detector parameters dialog.

        Args:
            parent: Parent widget
            detector: SpectrogramDetector instance (optional, will use defaults if None)
        """
        super().__init__(parent)
        self.setWindowTitle("Doppler Track Detection Parameters")
        self.setMinimumWidth(450)
        self.detector = detector

        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # =====================================================================
        # Frequency Range Group
        # =====================================================================
        freq_group = QGroupBox("Frequency Range")
        freq_layout = QFormLayout(freq_group)

        self.freq_min_spin = QSpinBox()
        self.freq_min_spin.setRange(0, 20000)
        self.freq_min_spin.setValue(self._get_param('freq_min', 50))
        self.freq_min_spin.setSuffix(" Hz")
        self.freq_min_spin.setToolTip("Minimum frequency to search for tracks")
        freq_layout.addRow("Min Frequency:", self.freq_min_spin)

        self.freq_max_spin = QSpinBox()
        self.freq_max_spin.setRange(0, 20000)
        self.freq_max_spin.setValue(self._get_param('freq_max', 1500))
        self.freq_max_spin.setSuffix(" Hz")
        self.freq_max_spin.setToolTip("Maximum frequency to search for tracks")
        freq_layout.addRow("Max Frequency:", self.freq_max_spin)

        layout.addWidget(freq_group)

        # =====================================================================
        # Threshold Group
        # =====================================================================
        thresh_group = QGroupBox("Peak Detection Thresholds")
        thresh_layout = QFormLayout(thresh_group)

        self.threshold_mode_combo = QComboBox()
        self.threshold_mode_combo.addItems(['adaptive', 'fixed'])
        current_mode = self._get_param('power_threshold_mode', 'adaptive')
        idx = self.threshold_mode_combo.findText(current_mode)
        if idx >= 0:
            self.threshold_mode_combo.setCurrentIndex(idx)
        self.threshold_mode_combo.setToolTip(
            "Adaptive: threshold based on percentile\n"
            "Fixed: use fixed threshold value"
        )
        self.threshold_mode_combo.currentTextChanged.connect(self._on_threshold_mode_changed)
        thresh_layout.addRow("Threshold Mode:", self.threshold_mode_combo)

        self.power_percentile_spin = QDoubleSpinBox()
        self.power_percentile_spin.setRange(50.0, 99.9)
        self.power_percentile_spin.setValue(self._get_param('power_threshold_percentile', 75.0))
        self.power_percentile_spin.setSingleStep(1.0)
        self.power_percentile_spin.setToolTip("Percentile for adaptive threshold (higher = fewer peaks)")
        thresh_layout.addRow("Percentile (adaptive):", self.power_percentile_spin)

        self.power_fixed_spin = QDoubleSpinBox()
        self.power_fixed_spin.setRange(0.01, 1.0)
        self.power_fixed_spin.setValue(self._get_param('power_threshold_fixed', 0.2))
        self.power_fixed_spin.setSingleStep(0.01)
        self.power_fixed_spin.setDecimals(2)
        self.power_fixed_spin.setToolTip("Fixed power threshold (0-1)")
        thresh_layout.addRow("Fixed Threshold:", self.power_fixed_spin)

        self.prominence_spin = QDoubleSpinBox()
        self.prominence_spin.setRange(0.001, 1.0)
        self.prominence_spin.setValue(self._get_param('peak_prominence', 0.05))
        self.prominence_spin.setSingleStep(0.005)
        self.prominence_spin.setDecimals(3)
        self.prominence_spin.setToolTip("Minimum peak prominence (higher = fewer peaks)")
        thresh_layout.addRow("Peak Prominence:", self.prominence_spin)

        layout.addWidget(thresh_group)

        # =====================================================================
        # Track Linking Group
        # =====================================================================
        link_group = QGroupBox("Track Linking")
        link_layout = QFormLayout(link_group)

        self.max_gap_spin = QSpinBox()
        self.max_gap_spin.setRange(0, 100)
        self.max_gap_spin.setValue(self._get_param('max_gap_frames', 4))
        self.max_gap_spin.setToolTip("Maximum frames to skip when linking peaks")
        link_layout.addRow("Max Gap Frames:", self.max_gap_spin)

        self.max_jump_spin = QDoubleSpinBox()
        self.max_jump_spin.setRange(1.0, 500.0)
        self.max_jump_spin.setValue(self._get_param('max_freq_jump_hz', 50.0))
        self.max_jump_spin.setSingleStep(5.0)
        self.max_jump_spin.setSuffix(" Hz")
        self.max_jump_spin.setToolTip("Maximum frequency jump between consecutive frames")
        link_layout.addRow("Max Freq Jump:", self.max_jump_spin)

        self.gap_power_spin = QDoubleSpinBox()
        self.gap_power_spin.setRange(0.1, 1.0)
        self.gap_power_spin.setValue(self._get_param('gap_power_factor', 0.8))
        self.gap_power_spin.setSingleStep(0.1)
        self.gap_power_spin.setToolTip("Relaxed power factor during gap bridging")
        link_layout.addRow("Gap Power Factor:", self.gap_power_spin)

        self.max_peaks_spin = QSpinBox()
        self.max_peaks_spin.setRange(1, 100)
        self.max_peaks_spin.setValue(self._get_param('max_peaks_per_frame', 20))
        self.max_peaks_spin.setToolTip("Maximum peaks to consider per frame")
        link_layout.addRow("Max Peaks/Frame:", self.max_peaks_spin)

        layout.addWidget(link_group)

        # =====================================================================
        # Track Filtering Group
        # =====================================================================
        filter_group = QGroupBox("Track Filtering")
        filter_layout = QFormLayout(filter_group)

        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(1, 500)
        self.min_length_spin.setValue(self._get_param('min_track_length_frames', 10))
        self.min_length_spin.setToolTip("Minimum number of points in a valid track")
        filter_layout.addRow("Min Track Length:", self.min_length_spin)

        self.min_power_spin = QDoubleSpinBox()
        self.min_power_spin.setRange(0.01, 1.0)
        self.min_power_spin.setValue(self._get_param('min_track_avg_power', 0.1))
        self.min_power_spin.setSingleStep(0.01)
        self.min_power_spin.setDecimals(2)
        self.min_power_spin.setToolTip("Minimum average power for valid track")
        filter_layout.addRow("Min Avg Power:", self.min_power_spin)

        self.max_std_spin = QDoubleSpinBox()
        self.max_std_spin.setRange(10.0, 1000.0)
        self.max_std_spin.setValue(self._get_param('max_track_freq_std_hz', 200.0))
        self.max_std_spin.setSingleStep(10.0)
        self.max_std_spin.setSuffix(" Hz")
        self.max_std_spin.setToolTip("Maximum frequency standard deviation (rejects horizontal lines)")
        filter_layout.addRow("Max Freq Std:", self.max_std_spin)

        layout.addWidget(filter_group)

        # =====================================================================
        # Merging Group
        # =====================================================================
        merge_group = QGroupBox("Track Merging")
        merge_layout = QFormLayout(merge_group)

        self.enable_merge_check = QCheckBox()
        self.enable_merge_check.setChecked(self._get_param('enable_post_merge', True))
        self.enable_merge_check.setToolTip("Merge fragmented tracks that appear continuous")
        merge_layout.addRow("Enable Merging:", self.enable_merge_check)

        self.merge_gap_spin = QSpinBox()
        self.merge_gap_spin.setRange(0, 500)
        self.merge_gap_spin.setValue(self._get_param('merge_gap_frames', 50))
        self.merge_gap_spin.setToolTip("Maximum gap between tracks for merging")
        merge_layout.addRow("Merge Gap Frames:", self.merge_gap_spin)

        self.merge_freq_spin = QDoubleSpinBox()
        self.merge_freq_spin.setRange(1.0, 500.0)
        self.merge_freq_spin.setValue(self._get_param('merge_max_freq_diff_hz', 50.0))
        self.merge_freq_spin.setSingleStep(5.0)
        self.merge_freq_spin.setSuffix(" Hz")
        self.merge_freq_spin.setToolTip("Maximum frequency difference for merging tracks")
        merge_layout.addRow("Merge Freq Diff:", self.merge_freq_spin)

        layout.addWidget(merge_group)

        # =====================================================================
        # Buttons
        # =====================================================================
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Connect restore defaults
        restore_btn = button_box.button(QDialogButtonBox.RestoreDefaults)
        if restore_btn:
            restore_btn.clicked.connect(self._restore_defaults)

        layout.addWidget(button_box)

        # Initial threshold mode state
        self._on_threshold_mode_changed(self.threshold_mode_combo.currentText())

    def _get_param(self, name: str, default):
        """Get parameter from detector or return default."""
        if self.detector is not None and hasattr(self.detector, name):
            return getattr(self.detector, name)
        return default

    def _on_threshold_mode_changed(self, mode: str):
        """Handle threshold mode change."""
        is_adaptive = mode == 'adaptive'
        self.power_percentile_spin.setEnabled(is_adaptive)
        self.power_fixed_spin.setEnabled(not is_adaptive)

    def _restore_defaults(self):
        """Restore all parameters to defaults."""
        self.freq_min_spin.setValue(50)
        self.freq_max_spin.setValue(1500)
        self.threshold_mode_combo.setCurrentText('adaptive')
        self.power_percentile_spin.setValue(75.0)
        self.power_fixed_spin.setValue(0.2)
        self.prominence_spin.setValue(0.05)
        self.max_gap_spin.setValue(4)
        self.max_jump_spin.setValue(50.0)
        self.gap_power_spin.setValue(0.8)
        self.max_peaks_spin.setValue(20)
        self.min_length_spin.setValue(10)
        self.min_power_spin.setValue(0.1)
        self.max_std_spin.setValue(200.0)
        self.enable_merge_check.setChecked(True)
        self.merge_gap_spin.setValue(50)
        self.merge_freq_spin.setValue(50.0)

    def get_parameters(self) -> dict:
        """Get all parameters as a dictionary."""
        return {
            'freq_min': self.freq_min_spin.value(),
            'freq_max': self.freq_max_spin.value(),
            'power_threshold_mode': self.threshold_mode_combo.currentText(),
            'power_threshold_percentile': self.power_percentile_spin.value(),
            'power_threshold_fixed': self.power_fixed_spin.value(),
            'peak_prominence': self.prominence_spin.value(),
            'max_gap_frames': self.max_gap_spin.value(),
            'max_freq_jump_hz': self.max_jump_spin.value(),
            'gap_power_factor': self.gap_power_spin.value(),
            'max_peaks_per_frame': self.max_peaks_spin.value(),
            'min_track_length_frames': self.min_length_spin.value(),
            'min_track_avg_power': self.min_power_spin.value(),
            'max_track_freq_std_hz': self.max_std_spin.value(),
            'enable_post_merge': self.enable_merge_check.isChecked(),
            'merge_gap_frames': self.merge_gap_spin.value(),
            'merge_max_freq_diff_hz': self.merge_freq_spin.value(),
        }

    def apply_to_detector(self, detector):
        """Apply current parameters to a detector instance."""
        params = self.get_parameters()
        for name, value in params.items():
            if hasattr(detector, name):
                setattr(detector, name, value)
