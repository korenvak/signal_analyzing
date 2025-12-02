"""
Dialog for previewing and saving spectrogram cutouts.
"""
import os
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                              QComboBox, QPushButton, QFileDialog, QFrame,
                              QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
                              QMessageBox)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap

from ..core.cutout_analyzer import (
    extract_spectrogram_cutout, 
    normalize_cutout, 
    save_cutout_image, 
    save_cutout_numpy, 
    write_cutout_metadata
)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)


class CutoutDialog(QDialog):
    """
    Popup dialog to display extracted cutout, adjust normalization, and save.
    """
    
    def __init__(self, parent=None, 
                 spectrogram_data=None, 
                 freqs=None, times=None, 
                 t_start=0, t_end=0, f_low=0, f_high=0,
                 audio_file_path=""):
        super().__init__(parent)
        self.setWindowTitle("Extract Cutout")
        self.resize(500, 650)
        
        # Store inputs
        self.S = spectrogram_data
        self.freqs = freqs
        self.times = times
        self.roi_coords = (t_start, t_end, f_low, f_high)
        self.audio_file_path = audio_file_path
        
        # Extract raw data immediately
        self.raw_cutout = extract_spectrogram_cutout(
            self.S, self.freqs, self.times, *self.roi_coords
        )
        
        # Processed data cache
        self.current_norm_data = None
        self.current_info = {}
        
        self.setup_ui()
        self.update_preview()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. Preview Image
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setStyleSheet("background-color: #111; border: 1px solid #444;")
        layout.addWidget(self.image_label, 1)
        
        # 2. Controls
        controls_group = QGroupBox("Normalization & Enhancement")
        controls_layout = QFormLayout(controls_group)
        
        # Mode selection
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "auto", 
            "db_center", 
            "percentile", 
            "noise_floor", 
            "clahe_only", 
            "db_then_clahe"
        ])
        self.mode_combo.currentTextChanged.connect(self.update_preview)
        controls_layout.addRow("Mode:", self.mode_combo)
        
        # CLAHE Parameters
        self.clip_limit_spin = QDoubleSpinBox()
        self.clip_limit_spin.setRange(0.1, 40.0)
        self.clip_limit_spin.setValue(2.0)
        self.clip_limit_spin.setSingleStep(0.5)
        self.clip_limit_spin.valueChanged.connect(self.update_preview)
        controls_layout.addRow("CLAHE Clip Limit:", self.clip_limit_spin)
        
        layout.addWidget(controls_group)
        
        # 3. Info Label
        self.info_label = QLabel("Statistics: -")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.info_label)
        
        # 4. Buttons
        btn_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save Cutout...")
        self.save_btn.setStyleSheet("""
            QPushButton { background-color: #2a82da; color: white; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #3a92ea; }
        """)
        self.save_btn.clicked.connect(self.save_cutout)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.close_btn)
        btn_layout.addWidget(self.save_btn)
        
        layout.addLayout(btn_layout)
        
    def update_preview(self):
        """Re-run normalization and update the displayed image."""
        if self.raw_cutout is None or self.raw_cutout['S_crop'].size == 0:
            self.image_label.setText("Empty selection")
            return
            
        mode = self.mode_combo.currentText()
        clip_limit = self.clip_limit_spin.value()
        
        try:
            S_norm, info = normalize_cutout(
                self.raw_cutout['S_crop'],
                mode=mode,
                clahe_clip_limit=clip_limit
            )
            
            self.current_norm_data = S_norm
            self.current_info = info
            
            # Convert to QPixmap for display
            # Use matplotlib colormap manually if available, else grayscale
            if HAS_MATPLOTLIB:
                # Apply magma colormap
                cmap = plt.get_cmap('magma')
                rgba_img = cmap(S_norm)
                # Convert to 0-255 uint8
                img_uint8 = (rgba_img * 255).astype(np.uint8)
                height, width, channel = img_uint8.shape
                bytes_per_line = 4 * width
                
                # Origin is usually bottom-left for spectrograms, flip for image display
                # (QImage starts top-left)
                img_uint8 = np.ascontiguousarray(np.flipud(img_uint8))
                
                q_img = QImage(img_uint8.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
            else:
                # Grayscale fallback
                img_uint8 = (S_norm * 255).astype(np.uint8)
                img_uint8 = np.ascontiguousarray(np.flipud(img_uint8))
                height, width = img_uint8.shape
                bytes_per_line = width
                q_img = QImage(img_uint8.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            
            # Scale to label - IgnoreAspectRatio to fill the view (like spectrogram stretch)
            pixmap = QPixmap.fromImage(q_img)
            w_label = self.image_label.width()
            h_label = self.image_label.height()
            self.image_label.setPixmap(pixmap.scaled(w_label, h_label, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
            
            # Update stats
            stats_text = (f"Applied: {info.get('applied_mode', mode)} | "
                          f"Range: {info.get('dynamic_range', 0):.1f}dB | "
                          f"Noise Floor Gamma: {info.get('gamma', 'N/A')}")
            self.info_label.setText(stats_text)
            
        except Exception as e:
            logger.error(f"Error previewing cutout: {e}")
            self.image_label.setText(f"Error: {str(e)}")

    def save_cutout(self):
        """Save the current cutout data."""
        if self.current_norm_data is None:
            return
            
        # Default filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"cutout_{timestamp}"
        
        # Default dir: same as audio file or user home
        start_dir = str(Path(self.audio_file_path).parent) if self.audio_file_path else os.path.expanduser("~")
        
        # Ask for location
        save_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory", start_dir)
        if not save_dir:
            return
            
        base_path = Path(save_dir) / default_name
        png_path = base_path.with_suffix(".png")
        npy_path = base_path.with_suffix(".npy")
        manifest_path = Path(save_dir) / "cutouts_manifest.jsonl"
        
        # Save Image
        success_img = save_cutout_image(
            self.current_norm_data,
            self.raw_cutout['times_crop'],
            self.raw_cutout['freqs_crop'],
            str(png_path)
        )
        
        # Save NPY
        success_npy = save_cutout_numpy(
            self.current_norm_data,
            self.raw_cutout['times_crop'],
            self.raw_cutout['freqs_crop'],
            str(npy_path),
            raw_data=self.raw_cutout['S_crop']
        )
        
        # Save Metadata
        t0, t1, f0, f1 = self.roi_coords
        meta = {
            "file_name": Path(self.audio_file_path).name,
            "time_start": t0,
            "time_end": t1,
            "freq_low": f0,
            "freq_high": f1,
            "cutout_png_path": str(png_path.name),
            "cutout_npy_path": str(npy_path.name),
            "normalization_used": self.mode_combo.currentText(),
            "created_utc": datetime.utcnow().isoformat(),
            "snr_estimate": self.current_info.get("dynamic_range", 0) # Rough proxy
        }
        success_meta = write_cutout_metadata(meta, str(manifest_path))
        
        if success_img and success_npy and success_meta:
            QMessageBox.information(self, "Success", f"Saved cutout to:\n{save_dir}")
            self.accept()
        else:
            QMessageBox.critical(self, "Error", "Failed to save one or more files.")

