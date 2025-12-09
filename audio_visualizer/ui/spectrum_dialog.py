"""
Pop-up dialog for displaying FFT spectrum of a selected region.
"""
import logging
import numpy as np
from .qt_compat import QDialog, QVBoxLayout, QLabel, QComboBox, QHBoxLayout, Qt

try:
    from vispy import scene
    HAS_VISPY = True
except ImportError:
    HAS_VISPY = False

logger = logging.getLogger(__name__)

class SpectrumDialog(QDialog):
    """Dialog that shows FFT spectrum of a time-frequency region."""
    
    def __init__(self, audio_data: np.ndarray, sample_rate: int, 
                 t_start: float, t_end: float, parent=None):
        super().__init__(parent)
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.t_start = t_start
        self.t_end = t_end
        
        self.setWindowTitle(f"FFT Spectrum: {t_start:.2f}s - {t_end:.2f}s")
        self.resize(700, 500)
        
        self.setup_ui()
        self.compute_and_plot()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Controls
        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Window:"))
        
        self.window_combo = QComboBox()
        self.window_combo.addItems(['hann', 'hamming', 'blackman', 'rectangular'])
        self.window_combo.currentTextChanged.connect(self.compute_and_plot)
        control_layout.addWidget(self.window_combo)
        control_layout.addStretch()
        
        layout.addLayout(control_layout)
        
        # VisPy Canvas
        if HAS_VISPY:
            self.canvas = scene.SceneCanvas(keys=None, size=(650, 400), bgcolor='#1e1e1e')
            self.grid = self.canvas.central_widget.add_grid()
            
            self.view = self.grid.add_view(row=0, col=0, border_color='white')
            self.view.camera = 'panzoom'
            self.view.camera.aspect = None
            
            # Line plot
            self.line = scene.visuals.Line(color='cyan', width=2.0, parent=self.view.scene)
            
            # Axes
            self.xaxis = scene.AxisWidget(orientation='bottom', axis_label='Frequency (Hz)', 
                                         text_color='white', axis_font_size=8)
            self.xaxis.height_max = 40
            self.grid.add_widget(self.xaxis, row=1, col=0)
            
            self.yaxis = scene.AxisWidget(orientation='left', axis_label='Magnitude (dB)', 
                                         text_color='white', axis_font_size=8)
            self.yaxis.width_max = 60
            self.grid.add_widget(self.yaxis, row=0, col=1)
            
            self.xaxis.link_view(self.view)
            self.yaxis.link_view(self.view)
            
            layout.addWidget(self.canvas.native)
        else:
            layout.addWidget(QLabel("VisPy not available"))
            
    def compute_and_plot(self):
        """Compute FFT and update plot."""
        if not HAS_VISPY or self.audio_data is None:
            return
            
        try:
            # Extract time segment
            idx_start = int(self.t_start * self.sample_rate)
            idx_end = int(self.t_end * self.sample_rate)
            idx_start = max(0, idx_start)
            idx_end = min(len(self.audio_data), idx_end)
            
            if idx_end <= idx_start:
                return
                
            segment = self.audio_data[idx_start:idx_end]
            
            # Limit size
            max_samples = 131072
            if len(segment) > max_samples:
                center = len(segment) // 2
                segment = segment[center - max_samples//2 : center + max_samples//2]
                
            if len(segment) < 128:
                return
                
            # Apply window
            window_name = self.window_combo.currentText()
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
            
            # Magnitude in dB
            magnitude = np.abs(spectrum) / len(segment)
            with np.errstate(divide='ignore'):
                db = 20 * np.log10(magnitude + 1e-12)
                
            # Filter low freqs
            mask = freqs > 20
            freqs = freqs[mask]
            db = db[mask]
            
            # Plot
            points = np.column_stack((freqs, db))
            self.line.set_data(pos=points)
            
            # Auto-scale
            db_max = np.max(db)
            db_min = max(db_max - 100, np.min(db))
            
            # Set camera rect to actual frequency range
            freq_min = np.min(freqs) if len(freqs) > 0 else 0
            freq_max = np.max(freqs) if len(freqs) > 0 else self.sample_rate/2
            self.view.camera.rect = (freq_min, db_min, freq_max - freq_min, db_max - db_min + 10)
            
        except Exception as e:
            logger.error(f"Error computing spectrum: {e}")
