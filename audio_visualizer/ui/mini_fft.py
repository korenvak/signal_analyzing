"""
Mini FFT Widget for instant spectral analysis of selected regions.
"""
import logging
import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                              QCheckBox, QComboBox)
from PySide6.QtCore import Qt

try:
    from vispy import scene
    HAS_VISPY = True
except ImportError:
    HAS_VISPY = False

logger = logging.getLogger(__name__)

class MiniFFTWidget(QWidget):
    """
    Widget that displays a 1D FFT plot of the currently selected region or cursor position.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.audio_data = None
        self.sample_rate = 44100
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Controls
        controls_layout = QVBoxLayout()
        
        header = QLabel("Instant FFT")
        header.setStyleSheet("font-weight: bold; color: #8B9DC3;")
        controls_layout.addWidget(header)
        
        self.check_auto = QCheckBox("Auto-update on Select")
        self.check_auto.setChecked(True)
        controls_layout.addWidget(self.check_auto)
        
        self.combo_window = QComboBox()
        self.combo_window.addItems(["hann", "hamming", "blackman", "rectangular"])
        controls_layout.addWidget(self.combo_window)
        
        layout.addLayout(controls_layout)
        
        # VisPy Plot Canvas
        if HAS_VISPY:
            self.canvas = scene.SceneCanvas(keys=None, size=(300, 200), bgcolor='#1e1e1e')
            self.grid = self.canvas.central_widget.add_grid()
            
            # ViewBox
            self.view = self.grid.add_view(row=0, col=0, border_color='#555')
            self.view.camera = 'panzoom'
            self.view.camera.rect = (0, -100, 22050, 100)
            self.view.camera.aspect = None
            
            # Grid lines
            self.grid_lines = scene.visuals.GridLines(parent=self.view.scene, color=(1,1,1,0.1))
            
            # Line plot
            self.line = scene.visuals.Line(color='yellow', width=1.5, parent=self.view.scene)
            
            # Axis
            self.xaxis = scene.AxisWidget(orientation='bottom', axis_label='Freq (Hz)', text_color='white', font_size=6)
            self.xaxis.height_max = 30
            self.grid.add_widget(self.xaxis, row=1, col=0)
            
            self.yaxis = scene.AxisWidget(orientation='left', axis_label='dB', text_color='white', font_size=6)
            self.yaxis.width_max = 40
            self.grid.add_widget(self.yaxis, row=0, col=1)
            
            self.xaxis.link_view(self.view)
            self.yaxis.link_view(self.view)
            
            layout.addWidget(self.canvas.native)
        else:
            layout.addWidget(QLabel("VisPy not available"))
            
    def set_audio_data(self, data: np.ndarray, sample_rate: int):
        """Update reference to audio data."""
        self.audio_data = data
        self.sample_rate = sample_rate
        
    def update_plot(self, t_start: float, t_end: float):
        """Compute and plot FFT for the given time range."""
        if self.audio_data is None or not HAS_VISPY:
            return
            
        try:
            # Extract samples
            idx_start = int(t_start * self.sample_rate)
            idx_end = int(t_end * self.sample_rate)
            
            if idx_start >= idx_end:
                return
                
            # Limit size for performance
            max_samples = 131072 # ~3 seconds at 44k
            if idx_end - idx_start > max_samples:
                center = (idx_start + idx_end) // 2
                idx_start = center - max_samples // 2
                idx_end = center + max_samples // 2
                
            segment = self.audio_data[idx_start:idx_end]
            if len(segment) < 128:
                return
                
            # Apply window
            window_name = self.combo_window.currentText()
            if window_name == 'hann':
                win = np.hanning(len(segment))
            elif window_name == 'hamming':
                win = np.hamming(len(segment))
            elif window_name == 'blackman':
                win = np.blackman(len(segment))
            else:
                win = np.ones(len(segment))
                
            segment = segment * win
            
            # Compute FFT
            spectrum = np.fft.rfft(segment)
            freqs = np.fft.rfftfreq(len(segment), 1/self.sample_rate)
            
            # Convert to dB
            magnitude = np.abs(spectrum)
            # Normalize
            magnitude = magnitude / len(segment)
            with np.errstate(divide='ignore'):
                db = 20 * np.log10(magnitude + 1e-9)
                
            # Filter out extremely low frequencies (DC offset)
            mask = freqs > 20
            freqs = freqs[mask]
            db = db[mask]
            
            # Plot
            points = np.column_stack((freqs, db))
            self.line.set_data(pos=points)
            
            # Update camera
            db_max = np.max(db)
            db_min = max(db_max - 80, np.min(db)) # Show dynamic range of 80dB
            
            self.view.camera.rect = (0, db_min, self.sample_rate/2, db_max - db_min + 5)
            
        except Exception as e:
            logger.error(f"Error updating Mini FFT: {e}")


