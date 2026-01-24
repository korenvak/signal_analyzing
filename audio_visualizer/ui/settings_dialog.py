"""
Spectrogram settings dialog.
"""
from .qt_compat import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                        QSlider, QComboBox, QPushButton, Qt, QGroupBox)

class SpectrogramSettingsDialog(QDialog):
    """Dialog for configuring spectrogram parameters."""
    
    def __init__(self, parent=None, current_fft=4096, current_overlap=0.75, 
                 current_window='hann', current_interp='bicubic'):
        super().__init__(parent)
        self.setWindowTitle("Spectrogram Settings")
        self.setModal(True)
        self.resize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # === FFT Settings ===
        fft_group = QGroupBox("FFT Configuration")
        fft_layout = QVBoxLayout(fft_group)
        
        # FFT Size
        fft_row = QHBoxLayout()
        fft_row.addWidget(QLabel("FFT Size:"))
        self.fft_combo = QComboBox()
        self.fft_combo.addItems([str(x) for x in [1024, 2048, 4096, 8192, 16384, 32768]])
        if str(current_fft) in [str(x) for x in [1024, 2048, 4096, 8192, 16384, 32768]]:
            self.fft_combo.setCurrentText(str(current_fft))
        fft_row.addWidget(self.fft_combo)
        fft_layout.addLayout(fft_row)
        
        # Overlap
        overlap_row = QVBoxLayout()
        header = QHBoxLayout()
        header.addWidget(QLabel("Overlap:"))
        self.overlap_label = QLabel(f"{int(current_overlap * 100)}%")
        header.addWidget(self.overlap_label)
        overlap_row.addLayout(header)
        
        self.overlap_slider = QSlider(Qt.Orientation.Horizontal)
        self.overlap_slider.setRange(0, 95)  # Max 95% to avoid divide-by-zero or ultra-slow
        self.overlap_slider.setValue(int(current_overlap * 100))
        self.overlap_slider.valueChanged.connect(self._on_overlap_changed)
        overlap_row.addWidget(self.overlap_slider)
        fft_layout.addLayout(overlap_row)
        
        # Window Function
        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(['hann', 'hamming', 'blackman', 'bartlett', 'kaiser'])
        self.window_combo.setCurrentText(current_window)
        win_row.addWidget(self.window_combo)
        fft_layout.addLayout(win_row)
        
        layout.addWidget(fft_group)
        
        # === Rendering Settings ===
        render_group = QGroupBox("Rendering")
        render_layout = QVBoxLayout(render_group)
        
        # Interpolation
        interp_row = QHBoxLayout()
        interp_row.addWidget(QLabel("Interpolation:"))
        self.interp_combo = QComboBox()
        self.interp_combo.addItems(['nearest', 'linear', 'bicubic'])
        self.interp_combo.setCurrentText(current_interp)
        interp_row.addWidget(self.interp_combo)
        render_layout.addLayout(interp_row)
        
        layout.addWidget(render_group)
        
        layout.addStretch()
        
        # Buttons
        btn_layout = QHBoxLayout()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
    def _on_overlap_changed(self, value):
        self.overlap_label.setText(f"{value}%")
        
    def get_values(self):
        """Get the configured values."""
        return {
            'fft_size': int(self.fft_combo.currentText()),
            'overlap': self.overlap_slider.value() / 100.0,
            'window': self.window_combo.currentText(),
            'interpolation': self.interp_combo.currentText()
        }
