"""
Widget for Doppler effect analysis and controls.
"""
import logging
import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                              QPushButton, QGroupBox, QFormLayout, QDoubleSpinBox,
                              QRadioButton, QButtonGroup, QMessageBox, QFileDialog)
from PySide6.QtCore import Qt, Signal

from ..core.doppler_analysis import DopplerAnalyzer, DopplerResult

logger = logging.getLogger(__name__)

class DopplerWidget(QWidget):
    """
    Side panel for Doppler analysis controls.
    Allows user to switch between drawing modes and perform curve fitting.
    """
    
    # Signals
    mode_changed = Signal(str)  # "select", "draw"
    analysis_requested = Signal() # Request to run analysis on current data
    export_requested = Signal()   # Request to export all CSV
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.analyzer = DopplerAnalyzer()
        self.current_result = None
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # --- Mode Selection ---
        mode_group = QGroupBox("Interaction Mode")
        mode_layout = QVBoxLayout()
        
        self.btn_select = QRadioButton("Select Region (Rectangle)")
        self.btn_select.setChecked(True)
        self.btn_draw = QRadioButton("Draw Curve (Points)")
        
        self.mode_group_bg = QButtonGroup()
        self.mode_group_bg.addButton(self.btn_select, 0)
        self.mode_group_bg.addButton(self.btn_draw, 1)
        self.mode_group_bg.buttonClicked.connect(self.on_mode_changed)
        
        mode_layout.addWidget(self.btn_select)
        mode_layout.addWidget(self.btn_draw)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # --- Parameters ---
        param_group = QGroupBox("Parameters")
        param_layout = QFormLayout()
        
        self.spin_speed_sound = QDoubleSpinBox()
        self.spin_speed_sound.setRange(0.1, 5000.0)
        self.spin_speed_sound.setValue(343.0)
        self.spin_speed_sound.setSuffix(" m/s")
        self.spin_speed_sound.valueChanged.connect(self.update_analyzer_params)
        
        param_layout.addRow("Speed of Sound:", self.spin_speed_sound)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # --- Actions ---
        action_layout = QVBoxLayout()
        
        self.btn_fit = QPushButton("Fit Doppler Model")
        self.btn_fit.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 6px;")
        self.btn_fit.clicked.connect(self.analysis_requested.emit)
        
        self.btn_clear = QPushButton("Clear Curve")
        self.btn_clear.clicked.connect(self.on_clear_clicked)
        
        action_layout.addWidget(self.btn_fit)
        action_layout.addWidget(self.btn_clear)
        layout.addLayout(action_layout)
        
        # --- Results ---
        result_group = QGroupBox("Analysis Results")
        result_layout = QFormLayout()
        
        self.lbl_velocity = QLabel("-")
        self.lbl_velocity_kmh = QLabel("-")
        self.lbl_cpa_dist = QLabel("-")
        self.lbl_time_cpa = QLabel("-")
        self.lbl_f0 = QLabel("-")
        self.lbl_rmse = QLabel("-")
        
        result_layout.addRow("Velocity:", self.lbl_velocity)
        result_layout.addRow("Velocity (km/h):", self.lbl_velocity_kmh)
        result_layout.addRow("CPA Distance:", self.lbl_cpa_dist)
        result_layout.addRow("Time at CPA:", self.lbl_time_cpa)
        result_layout.addRow("Rest Freq (f0):", self.lbl_f0)
        result_layout.addRow("RMSE:", self.lbl_rmse)
        
        result_group.setLayout(result_layout)
        layout.addWidget(result_group)
        
        # --- Global Export ---
        self.btn_export = QPushButton("Export Project CSV")
        self.btn_export.setToolTip("Export all annotations from all files in the current folder")
        self.btn_export.clicked.connect(self.export_requested.emit)
        layout.addWidget(self.btn_export)
        
        layout.addStretch()
        
    def on_mode_changed(self, btn):
        if self.btn_draw.isChecked():
            self.mode_changed.emit("draw")
        else:
            self.mode_changed.emit("select")
            
    def update_analyzer_params(self):
        self.analyzer.c = self.spin_speed_sound.value()
        
    def on_clear_clicked(self):
        # Signal handled by main window to clear canvas curve
        pass 

    def set_results(self, result: DopplerResult):
        """Display results in the widget."""
        self.current_result = result
        if result:
            self.lbl_velocity.setText(f"{result.velocity:.2f} m/s")
            self.lbl_velocity_kmh.setText(f"<b>{result.velocity_kmh:.1f} km/h</b>")
            self.lbl_cpa_dist.setText(f"{result.cpa_distance:.1f} m")
            self.lbl_time_cpa.setText(f"{result.t_cpa:.3f} s")
            self.lbl_f0.setText(f"{result.f0:.1f} Hz")
            self.lbl_rmse.setText(f"{result.rmse:.2f}")
        else:
            self.clear_results()
            
    def clear_results(self):
        self.current_result = None
        for lbl in [self.lbl_velocity, self.lbl_velocity_kmh, self.lbl_cpa_dist,
                   self.lbl_time_cpa, self.lbl_f0, self.lbl_rmse]:
            lbl.setText("-")

