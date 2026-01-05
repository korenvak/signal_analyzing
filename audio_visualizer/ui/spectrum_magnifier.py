"""
Spectrum Magnifier widget for displaying a 1D cross-section of the spectrogram.
"""
import logging
import numpy as np
from .qt_compat import QWidget, QVBoxLayout, QLabel, Signal, Qt

try:
    from vispy import scene
    HAS_VISPY = True
except ImportError:
    HAS_VISPY = False

logger = logging.getLogger(__name__)

class SpectrumMagnifier(QWidget):
    """Displays a 1D power spectrum slice at the current mouse position."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.data_buffer = None
        self.freq_axis = None
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        self.title_label = QLabel("Power Spectrum Slice")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(self.title_label)
        
        if HAS_VISPY:
            # Create a simple line plot using VisPy
            self.canvas = scene.SceneCanvas(keys='interactive', show=True, size=(300, 200))
            self.view = self.canvas.central_widget.add_view()
            
            # Use logarithmic x-axis if needed, but linear frequency is usually better for Doppler
            self.line = scene.visuals.Line(color='yellow', width=1.5, parent=self.view.scene)
            
            # Setup axes
            self.x_axis = scene.AxisWidget(orientation='bottom', axis_label='Frequency (Hz)')
            self.y_axis = scene.AxisWidget(orientation='left', axis_label='Power (dB)')
            
            grid = self.canvas.central_widget.add_grid()
            grid.add_widget(self.y_axis, row=0, col=0)
            grid.add_widget(self.view, row=0, col=1)
            grid.add_widget(self.x_axis, row=1, col=1)
            
            self.x_axis.link_view(self.view)
            self.y_axis.link_view(self.view)
            
            layout.addWidget(self.canvas.native)
        else:
            layout.addWidget(QLabel("VisPy not available for plot"))
            
        self.info_label = QLabel("Hover over spectrogram to see slice")
        self.info_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.info_label)
        
    def update_slice(self, power_db: np.ndarray, freqs: np.ndarray, time_val: float):
        """Update the displayed spectrum slice.
        
        Args:
            power_db: 1D array of power values in dB
            freqs: 1D array of frequency values in Hz
            time_val: Current time in seconds
        """
        if not HAS_VISPY or power_db is None or freqs is None:
            return
            
        try:
            # Ensure data is valid
            mask = np.isfinite(power_db)
            if not np.any(mask):
                return
                
            p_clean = power_db[mask]
            f_clean = freqs[mask]
            
            # Create (x, y) pairs for line
            pos = np.column_stack((f_clean, p_clean))
            self.line.set_data(pos=pos.astype(np.float32))
            
            # Auto-scale view
            f_min, f_max = np.min(f_clean), np.max(f_clean)
            p_min, p_max = np.min(p_clean), np.max(p_clean)
            
            # Add padding
            p_range = max(1.0, p_max - p_min)
            self.view.camera.rect = (f_min, p_min - 0.1 * p_range, f_max - f_min, 1.2 * p_range)
            
            self.info_label.setText(f"Time: {time_val:.3f}s | Peak: {f_clean[np.argmax(p_clean)]:.1f} Hz")
            
        except Exception as e:
            logger.debug(f"Error updating spectrum slice: {e}")

