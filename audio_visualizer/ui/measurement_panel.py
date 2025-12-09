"""
Measurement Panel - floating window for storing and managing multiple measurements.
Supports continuous measurement sequences with summary statistics.
"""
import logging
from typing import List, Tuple, Optional
from .qt_compat import (QDialog, QVBoxLayout, QHBoxLayout, QTableWidget,
                        QTableWidgetItem, QPushButton, QLabel, QHeaderView,
                        QFileDialog, QMessageBox, QGroupBox, QGridLayout,
                        Qt, Signal)
import csv
import numpy as np

logger = logging.getLogger(__name__)

class MeasurementPanel(QDialog):
    """Floating panel that stores multiple measurements with sequence support."""
    
    # Signal to notify when new measurement is requested
    new_sequence_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Measurements")
        self.resize(700, 500)
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
        
        self.measurements: List[dict] = []
        self.current_sequence: int = 1
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Header with buttons
        header = QHBoxLayout()
        title = QLabel("📏 Measurement History")
        title.setStyleSheet("font-weight: bold; font-size: 14pt; color: #4FC3F7;")
        header.addWidget(title)
        header.addStretch()
        
        self.btn_new_sequence = QPushButton("🔄 New Sequence")
        self.btn_new_sequence.setToolTip("Start a new measurement sequence (clears current)")
        self.btn_new_sequence.clicked.connect(self.start_new_sequence)
        self.btn_new_sequence.setStyleSheet("background: #2196F3; color: white; padding: 5px 10px;")
        header.addWidget(self.btn_new_sequence)
        
        self.btn_clear = QPushButton("🗑 Clear All")
        self.btn_clear.clicked.connect(self.clear_all)
        header.addWidget(self.btn_clear)
        
        self.btn_export = QPushButton("💾 Export CSV")
        self.btn_export.clicked.connect(self.export_csv)
        header.addWidget(self.btn_export)
        
        layout.addLayout(header)
        
        # Summary box
        summary_group = QGroupBox("Current Sequence Summary")
        summary_group.setStyleSheet("QGroupBox { font-weight: bold; color: #81C784; }")
        summary_layout = QGridLayout(summary_group)
        
        self.lbl_total_time = QLabel("Total Δt: 0.000 s")
        self.lbl_total_freq = QLabel("Total Δf: 0.0 Hz")
        self.lbl_avg_slope = QLabel("Avg Slope: 0.0 Hz/s")
        self.lbl_count = QLabel("Points: 0")
        
        summary_layout.addWidget(self.lbl_total_time, 0, 0)
        summary_layout.addWidget(self.lbl_total_freq, 0, 1)
        summary_layout.addWidget(self.lbl_avg_slope, 1, 0)
        summary_layout.addWidget(self.lbl_count, 1, 1)
        
        layout.addWidget(summary_group)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            'Seq', 'ID', 'Time 1 (s)', 'Freq 1 (Hz)', 'Time 2 (s)', 'Freq 2 (Hz)', 'Δt / Δf / Slope'
        ])
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        self.table.setStyleSheet("""
            QTableWidget {
                background: rgba(30, 30, 40, 0.9);
                gridline-color: rgba(255, 255, 255, 0.1);
            }
            QTableWidget::item {
                padding: 4px;
            }
            QHeaderView::section {
                background: rgba(50, 50, 70, 0.9);
                padding: 5px;
                border: none;
            }
        """)
        
        layout.addWidget(self.table)
        
        # Status
        self.status_label = QLabel("Ready - Click on spectrogram to measure")
        self.status_label.setStyleSheet("color: #888;")
        layout.addWidget(self.status_label)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background: rgba(25, 25, 35, 0.95);
            }
            QLabel {
                color: #B0B0B0;
            }
            QPushButton {
                background: rgba(60, 60, 80, 0.9);
                color: #B0B0B0;
                border: 1px solid rgba(255, 255, 255, 0.1);
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background: rgba(80, 80, 100, 0.9);
                color: white;
            }
        """)
        
    def add_measurement(self, t1: float, f1: float, t2: float, f2: float):
        """Add a measurement to the history."""
        dt = t2 - t1  # Signed delta
        df = f2 - f1  # Signed delta
        slope = df / dt if abs(dt) > 1e-6 else 0
        
        measurement = {
            'sequence': self.current_sequence,
            'id': len([m for m in self.measurements if m['sequence'] == self.current_sequence]) + 1,
            't1': t1,
            'f1': f1,
            't2': t2,
            'f2': f2,
            'dt': dt,
            'df': df,
            'slope': slope
        }
        
        self.measurements.append(measurement)
        self._add_to_table(measurement)
        self.update_summary()
        self.update_status()
        
        logger.info(f"Added measurement: Δt={dt:.3f}s, Δf={df:.1f}Hz, slope={slope:.2f}Hz/s")
        
    def _add_to_table(self, m: dict):
        """Add measurement to table."""
        row = self.table.rowCount()
        self.table.insertRow(row)
        
        self.table.setItem(row, 0, QTableWidgetItem(str(m['sequence'])))
        self.table.setItem(row, 1, QTableWidgetItem(str(m['id'])))
        self.table.setItem(row, 2, QTableWidgetItem(f"{m['t1']:.3f}"))
        self.table.setItem(row, 3, QTableWidgetItem(f"{m['f1']:.1f}"))
        self.table.setItem(row, 4, QTableWidgetItem(f"{m['t2']:.3f}"))
        self.table.setItem(row, 5, QTableWidgetItem(f"{m['f2']:.1f}"))
        
        # Format delta with slope
        dt_str = f"{abs(m['dt']):.3f}s"
        df_str = f"{abs(m['df']):.1f}Hz"
        slope_str = f"{m['slope']:.2f}Hz/s"
        self.table.setItem(row, 6, QTableWidgetItem(f"{dt_str} / {df_str} / {slope_str}"))
        
        # Scroll to bottom
        self.table.scrollToBottom()
        
    def update_summary(self):
        """Update the summary statistics for current sequence."""
        seq_measurements = [m for m in self.measurements if m['sequence'] == self.current_sequence]
        
        if not seq_measurements:
            self.lbl_total_time.setText("Total Δt: 0.000 s")
            self.lbl_total_freq.setText("Total Δf: 0.0 Hz")
            self.lbl_avg_slope.setText("Avg Slope: 0.0 Hz/s")
            self.lbl_count.setText("Points: 0")
            return
        
        total_dt = sum(m['dt'] for m in seq_measurements)
        total_df = sum(m['df'] for m in seq_measurements)
        avg_slope = np.mean([m['slope'] for m in seq_measurements])
        
        # Format time
        if abs(total_dt) >= 60:
            time_str = f"{int(total_dt // 60)}m {abs(total_dt) % 60:.2f}s"
        else:
            time_str = f"{total_dt:.3f} s"
            
        # Format freq
        if abs(total_df) >= 1000:
            freq_str = f"{total_df / 1000:.2f} kHz"
        else:
            freq_str = f"{total_df:.1f} Hz"
        
        self.lbl_total_time.setText(f"Total Δt: {time_str}")
        self.lbl_total_freq.setText(f"Total Δf: {freq_str}")
        self.lbl_avg_slope.setText(f"Avg Slope: {avg_slope:.2f} Hz/s")
        self.lbl_count.setText(f"Points: {len(seq_measurements) + 1}")
        
    def start_new_sequence(self):
        """Start a new measurement sequence."""
        self.current_sequence += 1
        self.update_summary()
        self.status_label.setText(f"Started new sequence #{self.current_sequence}")
        self.new_sequence_requested.emit()
        
    def clear_all(self):
        """Clear all measurements."""
        self.measurements.clear()
        self.table.setRowCount(0)
        self.current_sequence = 1
        self.update_summary()
        self.update_status()
        
    def update_status(self):
        """Update status label."""
        total = len(self.measurements)
        seq_count = len([m for m in self.measurements if m['sequence'] == self.current_sequence])
        self.status_label.setText(f"Sequence #{self.current_sequence}: {seq_count} measurements | Total: {total}")
        
    def export_csv(self):
        """Export measurements to CSV."""
        if not self.measurements:
            QMessageBox.information(self, "No Data", "No measurements to export.")
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Measurements", "measurements.csv", "CSV Files (*.csv)"
        )
        
        if path:
            try:
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['sequence', 'id', 't1', 'f1', 't2', 'f2', 'dt', 'df', 'slope'])
                    writer.writeheader()
                    writer.writerows(self.measurements)
                    
                QMessageBox.information(self, "Export Success", f"Exported {len(self.measurements)} measurements to:\n{path}")
            except Exception as e:
                logger.error(f"Failed to export measurements: {e}")
                QMessageBox.critical(self, "Export Error", str(e))
                
    def closeEvent(self, event):
        """Handle close - hide instead of destroy."""
        self.hide()
        event.ignore()  # Don't actually close, just hide
