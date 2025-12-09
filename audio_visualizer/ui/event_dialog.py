"""
Dialog for entering event metadata after marking time region.

Collects:
- Harmonic number (optional)
- SNR estimate (with auto-suggested value, user can override)
- Frequency range (defaults to current view, adjustable)
- Notes (optional)
"""
import logging
from datetime import datetime
from typing import Optional, Tuple

from .qt_compat import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QLineEdit,
    QPushButton, QGroupBox, QFrame, QMessageBox,
    Qt, QFont
)

logger = logging.getLogger(__name__)


class EventInputDialog(QDialog):
    """Dialog for entering event metadata after marking time region."""

    def __init__(
        self,
        t_start: float,
        t_end: float,
        f_min: float,
        f_max: float,
        absolute_start: Optional[datetime] = None,
        absolute_end: Optional[datetime] = None,
        suggested_snr: Optional[float] = None,
        sensor_name: Optional[str] = None,
        sensor_id: Optional[int] = None,
        parent=None
    ):
        """Initialize the event input dialog.

        Args:
            t_start: Event start time (relative seconds)
            t_end: Event end time (relative seconds)
            f_min: Lower frequency bound (from current view)
            f_max: Upper frequency bound (from current view)
            absolute_start: Absolute start time (if filename parsed)
            absolute_end: Absolute end time (if filename parsed)
            suggested_snr: Auto-calculated SNR suggestion
            sensor_name: Sensor name (if filename parsed)
            sensor_id: Sensor ID (if filename parsed)
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle("Tag Event")
        self.setMinimumWidth(450)

        self.t_start = t_start
        self.t_end = t_end
        self.suggested_snr = suggested_snr

        self._setup_ui(
            t_start, t_end, f_min, f_max,
            absolute_start, absolute_end,
            suggested_snr, sensor_name, sensor_id
        )

    def _setup_ui(
        self,
        t_start: float,
        t_end: float,
        f_min: float,
        f_max: float,
        absolute_start: Optional[datetime],
        absolute_end: Optional[datetime],
        suggested_snr: Optional[float],
        sensor_name: Optional[str],
        sensor_id: Optional[int]
    ):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # =====================================================================
        # Time Information Group (read-only)
        # =====================================================================
        time_group = QGroupBox("Event Time")
        time_layout = QFormLayout(time_group)

        # Duration
        duration = t_end - t_start
        duration_label = QLabel(f"{duration:.3f} s")
        duration_label.setFont(QFont("", -1, QFont.Bold))
        time_layout.addRow("Duration:", duration_label)

        # Relative time
        relative_label = QLabel(f"{t_start:.3f} s  to  {t_end:.3f} s")
        time_layout.addRow("Relative Time:", relative_label)

        # Absolute time (if available)
        if absolute_start and absolute_end:
            abs_start_str = absolute_start.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            abs_end_str = absolute_end.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            abs_label = QLabel(f"{abs_start_str}\nto {abs_end_str}")
            abs_label.setWordWrap(True)
            time_layout.addRow("Absolute Time:", abs_label)

        # Sensor info (if available)
        if sensor_name and sensor_id:
            sensor_label = QLabel(f"{sensor_name} - {sensor_id}")
            sensor_label.setFont(QFont("", -1, QFont.Bold))
            time_layout.addRow("Sensor:", sensor_label)

        layout.addWidget(time_group)

        # =====================================================================
        # Frequency Range Group (editable)
        # =====================================================================
        freq_group = QGroupBox("Frequency Range")
        freq_layout = QFormLayout(freq_group)

        # F min
        self.f_min_spin = QDoubleSpinBox()
        self.f_min_spin.setRange(0, 100000)
        self.f_min_spin.setSuffix(" Hz")
        self.f_min_spin.setDecimals(1)
        self.f_min_spin.setValue(f_min)
        freq_layout.addRow("Min Frequency:", self.f_min_spin)

        # F max
        self.f_max_spin = QDoubleSpinBox()
        self.f_max_spin.setRange(0, 100000)
        self.f_max_spin.setSuffix(" Hz")
        self.f_max_spin.setDecimals(1)
        self.f_max_spin.setValue(f_max)
        freq_layout.addRow("Max Frequency:", self.f_max_spin)

        layout.addWidget(freq_group)

        # =====================================================================
        # Event Metadata Group
        # =====================================================================
        meta_group = QGroupBox("Event Metadata")
        meta_layout = QFormLayout(meta_group)

        # Harmonic number
        self.harmonic_spin = QSpinBox()
        self.harmonic_spin.setRange(0, 20)  # 0 = not set
        self.harmonic_spin.setSpecialValueText("Not set")
        self.harmonic_spin.setValue(0)
        self.harmonic_spin.setToolTip("Harmonic order: 1=fundamental, 2,3,4...=harmonics, 0=not set")
        meta_layout.addRow("Harmonic Number:", self.harmonic_spin)

        # SNR estimate
        snr_widget = QFrame()
        snr_layout = QHBoxLayout(snr_widget)
        snr_layout.setContentsMargins(0, 0, 0, 0)

        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(-50, 100)
        self.snr_spin.setSuffix(" dB")
        self.snr_spin.setDecimals(1)
        self.snr_spin.setSpecialValueText("Not set")
        self.snr_spin.setMinimum(-999)  # Allow "not set"
        self.snr_spin.setValue(-999)

        if suggested_snr is not None:
            self.snr_spin.setValue(suggested_snr)
            snr_layout.addWidget(self.snr_spin)
            suggestion_label = QLabel(f"(suggested: {suggested_snr:.1f} dB)")
            suggestion_label.setStyleSheet("color: #888;")
            snr_layout.addWidget(suggestion_label)
        else:
            snr_layout.addWidget(self.snr_spin)

        snr_layout.addStretch()
        meta_layout.addRow("SNR Estimate:", snr_widget)

        # Notes
        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("Optional notes about this event...")
        meta_layout.addRow("Notes:", self.notes_edit)

        layout.addWidget(meta_group)

        # =====================================================================
        # Buttons
        # =====================================================================
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        self.save_btn = QPushButton("Save Event")
        self.save_btn.setDefault(True)
        self.save_btn.clicked.connect(self._on_save)
        button_layout.addWidget(self.save_btn)

        layout.addLayout(button_layout)

    def _on_save(self):
        """Validate and accept the dialog."""
        # Validate frequency range
        if self.f_min_spin.value() >= self.f_max_spin.value():
            QMessageBox.warning(
                self, "Invalid Range",
                "Min frequency must be less than max frequency."
            )
            return

        self.accept()

    def get_values(self) -> dict:
        """Get the entered values.

        Returns:
            Dictionary with:
            - f_min: float
            - f_max: float
            - harmonic_number: Optional[int]
            - snr_estimate_db: Optional[float]
            - notes: str
        """
        harmonic = self.harmonic_spin.value()
        if harmonic == 0:
            harmonic = None

        snr = self.snr_spin.value()
        if snr <= -999:
            snr = None

        return {
            'f_min': self.f_min_spin.value(),
            'f_max': self.f_max_spin.value(),
            'harmonic_number': harmonic,
            'snr_estimate_db': snr,
            'notes': self.notes_edit.text().strip()
        }


class EventEditDialog(QDialog):
    """Dialog for editing an existing event's metadata."""

    def __init__(
        self,
        event_id: int,
        harmonic_number: Optional[int],
        snr_estimate_db: Optional[float],
        notes: str,
        parent=None
    ):
        """Initialize the edit dialog.

        Args:
            event_id: Event ID being edited
            harmonic_number: Current harmonic number
            snr_estimate_db: Current SNR estimate
            notes: Current notes
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle(f"Edit Event {event_id}")
        self.setMinimumWidth(350)

        self._setup_ui(harmonic_number, snr_estimate_db, notes)

    def _setup_ui(
        self,
        harmonic_number: Optional[int],
        snr_estimate_db: Optional[float],
        notes: str
    ):
        """Setup the edit dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        form_layout = QFormLayout()

        # Harmonic number
        self.harmonic_spin = QSpinBox()
        self.harmonic_spin.setRange(0, 20)
        self.harmonic_spin.setSpecialValueText("Not set")
        self.harmonic_spin.setValue(harmonic_number if harmonic_number else 0)
        form_layout.addRow("Harmonic Number:", self.harmonic_spin)

        # SNR estimate
        self.snr_spin = QDoubleSpinBox()
        self.snr_spin.setRange(-50, 100)
        self.snr_spin.setSuffix(" dB")
        self.snr_spin.setDecimals(1)
        self.snr_spin.setSpecialValueText("Not set")
        self.snr_spin.setMinimum(-999)
        if snr_estimate_db is not None:
            self.snr_spin.setValue(snr_estimate_db)
        else:
            self.snr_spin.setValue(-999)
        form_layout.addRow("SNR Estimate:", self.snr_spin)

        # Notes
        self.notes_edit = QLineEdit()
        self.notes_edit.setText(notes)
        self.notes_edit.setPlaceholderText("Optional notes...")
        form_layout.addRow("Notes:", self.notes_edit)

        layout.addLayout(form_layout)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save Changes")
        save_btn.setDefault(True)
        save_btn.clicked.connect(self.accept)
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)

    def get_values(self) -> dict:
        """Get the edited values."""
        harmonic = self.harmonic_spin.value()
        if harmonic == 0:
            harmonic = None

        snr = self.snr_spin.value()
        if snr <= -999:
            snr = None

        return {
            'harmonic_number': harmonic,
            'snr_estimate_db': snr,
            'notes': self.notes_edit.text().strip()
        }
