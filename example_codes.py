"""
Optimized Spectrogram Canvas with smooth zoom and performance improvements
Uses caching, level-of-detail rendering, and GPU-accelerated drawing where possible
"""

import numpy as np
from datetime import timedelta
from functools import lru_cache
import pyqtgraph as pg
from PySide6.QtWidgets import QWidget, QVBoxLayout, QToolTip, QApplication
from PySide6.QtCore import Qt, Signal, QTimer, QRectF, QPointF
from PySide6.QtGui import QPainter, QColor, QBrush, QPen, QLinearGradient
import matplotlib.cm as cm
from numba import jit, prange


class OptimizedViewBox(pg.ViewBox):
    """
    Custom ViewBox with smooth zoom constrained to data boundaries
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseEnabled(x=True, y=True)
        self.enableAutoRange(axis='xy', enable=False)
        
        # Zoom constraints
        self.time_bounds = None
        self.freq_bounds = None
        
        # Smooth zoom parameters
        self.zoom_factor = 1.15  # Smoother zoom steps
        self.pan_speed = 1.0
        
    def set_data_bounds(self, time_min, time_max, freq_min, freq_max):
        """Set the data boundaries for zoom constraints"""
        self.time_bounds = (time_min, time_max)
        self.freq_bounds = (freq_min, freq_max)
        self.setLimits(xMin=time_min, xMax=time_max, 
                      yMin=freq_min, yMax=freq_max)
        
    def wheelEvent(self, ev, axis=None):
        """Smooth wheel zoom with boundary constraints"""
        if self.time_bounds is None or self.freq_bounds is None:
            super().wheelEvent(ev, axis)
            return
            
        # Get mouse position in scene coordinates
        pos = ev.pos()
        mask = np.array([1, 1], dtype=float)
        modifiers = ev.modifiers()
        
        # Determine zoom axis based on modifiers
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            mask = np.array([0, 1], dtype=float)  # Y-axis only
        elif modifiers & Qt.KeyboardModifier.ShiftModifier:
            mask = np.array([1, 0], dtype=float)  # X-axis only
            
        # Calculate smooth zoom
        s = self.zoom_factor ** (ev.delta() / 120.0)
        s = [(s - 1) * m + 1 for m in mask]
        
        # Get current view range
        vr = self.viewRect()
        center = self.mapSceneToView(pos)
        
        # Apply zoom with smooth interpolation
        new_width = vr.width() / s[0]
        new_height = vr.height() / s[1]
        
        # Constrain to data bounds
        if self.time_bounds:
            new_width = min(new_width, self.time_bounds[1] - self.time_bounds[0])
            new_width = max(new_width, (self.time_bounds[1] - self.time_bounds[0]) / 100)
            
        if self.freq_bounds:
            new_height = min(new_height, self.freq_bounds[1] - self.freq_bounds[0])
            new_height = max(new_height, (self.freq_bounds[1] - self.freq_bounds[0]) / 100)
            
        # Calculate new position
        new_x = center.x() - (center.x() - vr.x()) * new_width / vr.width()
        new_y = center.y() - (center.y() - vr.y()) * new_height / vr.height()
        
        # Constrain position to bounds
        if self.time_bounds:
            new_x = max(self.time_bounds[0], min(new_x, self.time_bounds[1] - new_width))
        if self.freq_bounds:
            new_y = max(self.freq_bounds[0], min(new_y, self.freq_bounds[1] - new_height))
            
        # Apply the new view range with animation
        self.setRange(xRange=(new_x, new_x + new_width),
                     yRange=(new_y, new_y + new_height),
                     padding=0, update=True)
        
        ev.accept()
        
    def mouseDragEvent(self, ev, axis=None):
        """Smooth panning with boundary constraints"""
        if ev.button() != Qt.MouseButton.LeftButton:
            super().mouseDragEvent(ev, axis)
            return
            
        # Calculate pan delta
        delta = ev.pos() - ev.lastPos()
        delta = self.mapToView(delta) - self.mapToView(QPointF(0, 0))
        
        # Apply pan speed factor
        delta = delta * self.pan_speed
        
        # Get current range
        vr = self.viewRect()
        
        # Calculate new position
        new_x = vr.x() - delta.x()
        new_y = vr.y() - delta.y()
        
        # Constrain to bounds
        if self.time_bounds:
            new_x = max(self.time_bounds[0], 
                       min(new_x, self.time_bounds[1] - vr.width()))
        if self.freq_bounds:
            new_y = max(self.freq_bounds[0], 
                       min(new_y, self.freq_bounds[1] - vr.height()))
            
        # Apply the pan
        self.setRange(xRange=(new_x, new_x + vr.width()),
                     yRange=(new_y, new_y + vr.height()),
                     padding=0, update=True)
        
        ev.accept()


class ModernTimeAxis(pg.AxisItem):
    """
    Modern time axis with clean formatting
    """
    def __init__(self, orientation="bottom", **kwargs):
        super().__init__(orientation, **kwargs)
        self.start_dt = None
        self.setStyle(tickTextOffset=10)
        
    def set_start_time(self, start_dt):
        """Set the start time for absolute time display"""
        self.start_dt = start_dt
        
    def tickStrings(self, values, scale, spacing):
        """Format tick strings as time"""
        if self.start_dt is None:
            # Relative time format
            strings = []
            for v in values:
                try:
                    total_seconds = float(v)
                    hours = int(total_seconds // 3600)
                    minutes = int((total_seconds % 3600) // 60)
                    seconds = total_seconds % 60
                    
                    if hours > 0:
                        strings.append(f"{hours:02d}:{minutes:02d}:{seconds:05.2f}")
                    else:
                        strings.append(f"{minutes:02d}:{seconds:05.2f}")
                except:
                    strings.append(f"{v:.2f}")
            return strings
        else:
            # Absolute time format
            strings = []
            for v in values:
                t = self.start_dt + timedelta(seconds=float(v))
                strings.append(t.strftime("%H:%M:%S"))
            return strings


@jit(nopython=True, parallel=True, cache=True)
def compute_spectrogram_fast(audio_data, nperseg, noverlap, window):
    """
    Fast spectrogram computation using Numba JIT compilation
    """
    hop_length = nperseg - noverlap
    n_frames = (len(audio_data) - nperseg) // hop_length + 1
    n_freqs = nperseg // 2 + 1
    
    spectrogram = np.zeros((n_freqs, n_frames), dtype=np.complex128)
    
    for i in prange(n_frames):
        start = i * hop_length
        frame = audio_data[start:start + nperseg] * window
        fft_result = np.fft.rfft(frame)
        spectrogram[:, i] = fft_result
        
    return np.abs(spectrogram) ** 2


class OptimizedSpectrogramCanvas(QWidget):
    """
    High-performance spectrogram canvas with modern design
    """
    click_callback = Signal(float, object)  # time, event
    hover_callback = Signal(str)  # hover text
    selection_callback = Signal(float, float)  # start, end time
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Data storage
        self.audio_data = None
        self.sample_rate = None
        self.freqs = None
        self.times = None
        self.Sxx = None
        self.Sxx_display = None
        
        # Performance optimizations
        self.cache_enabled = True
        self.lod_enabled = True  # Level of detail
        self.current_lod = 0
        self.display_cache = {}
        
        # Colormap
        self.colormap_name = "viridis"
        self.colormap_lut = None
        self.update_colormap()
        
        # Playback marker
        self.playback_line = None
        self.playback_position = 0
        
        # Selection
        self.selection_region = None
        
        # Auto-detection tracks
        self.auto_tracks_items = []
        
        # Update timer for smooth animations
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animations)
        self.animation_timer.start(16)  # 60 FPS
        
    def setup_ui(self):
        """Setup the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Create graphics view
        self.graphics_view = pg.GraphicsLayoutWidget()
        self.graphics_view.setBackground((15, 15, 30, 200))  # Semi-transparent dark
        layout.addWidget(self.graphics_view)
        
        # Create custom viewbox and axis
        self.time_axis = ModernTimeAxis(orientation="bottom")
        self.freq_axis = pg.AxisItem(orientation="left")
        self.viewbox = OptimizedViewBox()
        
        # Create plot with custom components
        self.plot = self.graphics_view.addPlot(
            viewBox=self.viewbox,
            axisItems={"bottom": self.time_axis, "left": self.freq_axis}
        )
        
        # Configure plot appearance
        self.plot.showGrid(x=True, y=True, alpha=0.1)
        self.plot.setLabel('left', 'Frequency', units='Hz', 
                          **{'color': '#9CA3AF', 'font-size': '10pt'})
        self.plot.setLabel('bottom', 'Time', units='s',
                          **{'color': '#9CA3AF', 'font-size': '10pt'})
        
        # Create image item for spectrogram
        self.img_item = pg.ImageItem()
        self.plot.addItem(self.img_item)
        
        # Setup mouse tracking for hover info
        self.img_item.scene().sigMouseMoved.connect(self.on_mouse_move)
        self.img_item.mouseClickEvent = self.on_mouse_click
        
        # Create crosshair for precise navigation
        self.crosshair_v = pg.InfiniteLine(angle=90, movable=False, 
                                          pen=pg.mkPen('#8B5CF6', width=1, style=Qt.PenStyle.DashLine))
        self.crosshair_h = pg.InfiniteLine(angle=0, movable=False,
                                          pen=pg.mkPen('#8B5CF6', width=1, style=Qt.PenStyle.DashLine))
        self.crosshair_v.setVisible(False)
        self.crosshair_h.setVisible(False)
        self.plot.addItem(self.crosshair_v)
        self.plot.addItem(self.crosshair_h)
        
    def update_colormap(self):
        """Update the colormap lookup table"""
        # Create high-quality gradient colormap
        cmap = cm.get_cmap(self.colormap_name)
        colors = cmap(np.linspace(0, 1, 256))
        self.colormap_lut = (colors * 255).astype(np.uint8)
        
        if self.img_item and self.Sxx_display is not None:
            self.img_item.setLookupTable(self.colormap_lut)
            
    def set_file_start_time(self, start_dt):
        """Set the file start time for proper time axis display"""
        if self.time_axis:
            self.time_axis.set_start_time(start_dt)
            
    def set_spectrogram_data(self, audio_data, sample_rate, freqs, times, Sxx):
        """
        Set spectrogram data with optimizations
        """
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.freqs = freqs
        self.times = times
        self.Sxx = Sxx
        
        # Clear cache
        self.display_cache.clear()
        
        # Set data bounds for zoom constraints
        if times is not None and freqs is not None:
            self.viewbox.set_data_bounds(
                times[0], times[-1],
                freqs[0], freqs[-1]
            )
            
        # Prepare display data
        self.prepare_display_data()
        
        # Update display
        self.update_display()
        
    def prepare_display_data(self):
        """
        Prepare data for display with level-of-detail optimization
        """
        if self.Sxx is None:
            return
            
        # Convert to dB scale with proper handling
        with np.errstate(divide='ignore', invalid='ignore'):
            Sxx_db = 10 * np.log10(self.Sxx + 1e-10)
            
        # Apply dynamic range compression for better visibility
        vmin, vmax = np.percentile(Sxx_db[np.isfinite(Sxx_db)], [1, 99])
        Sxx_db = np.clip(Sxx_db, vmin, vmax)
        
        # Normalize to 0-255 for display
        Sxx_norm = ((Sxx_db - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        
        # Transpose for correct display (time on X-axis, freq on Y-axis)
        self.Sxx_display = Sxx_norm.T
        
        # Generate LOD versions for large datasets
        if self.lod_enabled and self.Sxx_display.shape[0] > 2000:
            self.generate_lod_versions()
            
    def generate_lod_versions(self):
        """Generate level-of-detail versions for performance"""
        self.display_cache['lod_0'] = self.Sxx_display
        
        # Generate downsampled versions
        current = self.Sxx_display
        for lod in range(1, 4):
            # Downsample by factor of 2
            h, w = current.shape
            downsampled = current[::2, ::2]
            self.display_cache[f'lod_{lod}'] = downsampled
            current = downsampled
            
    def update_display(self):
        """Update the display with appropriate LOD"""
        if self.Sxx_display is None:
            return
            
        # Choose appropriate LOD based on zoom level
        view_range = self.viewbox.viewRange()
        if view_range and self.lod_enabled and self.display_cache:
            x_span = view_range[0][1] - view_range[0][0]
            total_span = self.times[-1] - self.times[0] if self.times is not None else 1
            zoom_ratio = x_span / total_span
            
            # Select LOD based on zoom
            if zoom_ratio > 0.5:
                lod = 2
            elif zoom_ratio > 0.25:
                lod = 1
            else:
                lod = 0
                
            display_data = self.display_cache.get(f'lod_{lod}', self.Sxx_display)
        else:
            display_data = self.Sxx_display
            
        # Update image
        self.img_item.setImage(display_data)
        self.img_item.setLookupTable(self.colormap_lut)
        
        # Set proper scaling
        if self.times is not None and self.freqs is not None:
            scale_x = (self.times[-1] - self.times[0]) / display_data.shape[0]
            scale_y = (self.freqs[-1] - self.freqs[0]) / display_data.shape[1]
            self.img_item.setRect(QRectF(self.times[0], self.freqs[0],
                                        self.times[-1] - self.times[0],
                                        self.freqs[-1] - self.freqs[0]))
                                        
    def on_mouse_move(self, pos):
        """Handle mouse movement for hover info"""
        if self.Sxx is None:
            return
            
        # Convert to view coordinates
        mouse_point = self.viewbox.mapSceneToView(pos)
        x, y = mouse_point.x(), mouse_point.y()
        
        # Check if in bounds
        if (self.times is not None and self.freqs is not None and
            self.times[0] <= x <= self.times[-1] and
            self.freqs[0] <= y <= self.freqs[-1]):
            
            # Find nearest indices
            time_idx = np.searchsorted(self.times, x)
            freq_idx = np.searchsorted(self.freqs, y)
            
            if 0 <= time_idx < len(self.times) and 0 <= freq_idx < len(self.freqs):
                # Get amplitude
                amp_db = 10 * np.log10(self.Sxx[freq_idx, time_idx] + 1e-10)
                
                # Format hover text
                hover_text = (f"Time: {x:.3f}s\n"
                            f"Freq: {y:.1f} Hz\n"
                            f"Amp: {amp_db:.1f} dB")
                
                self.hover_callback.emit(hover_text)
                
                # Update crosshair if Alt is pressed
                if QApplication.keyboardModifiers() & Qt.KeyboardModifier.AltModifier:
                    self.crosshair_v.setPos(x)
                    self.crosshair_h.setPos(y)
                    self.crosshair_v.setVisible(True)
                    self.crosshair_h.setVisible(True)
                else:
                    self.crosshair_v.setVisible(False)
                    self.crosshair_h.setVisible(False)
                    
    def on_mouse_click(self, event):
        """Handle mouse clicks"""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            mouse_point = self.viewbox.mapSceneToView(pos)
            x = mouse_point.x()
            
            if self.times is not None and self.times[0] <= x <= self.times[-1]:
                self.click_callback.emit(x, event)
                
        elif event.button() == Qt.MouseButton.RightButton:
            # Start region selection for right-click
            self.start_region_selection(event)
                
    def set_playback_position(self, position):
        """Update playback position marker"""
        self.playback_position = position
        
        if self.playback_line is None:
            self.playback_line = pg.InfiniteLine(
                angle=90, movable=False,
                pen=pg.mkPen('#EF4444', width=2)
            )
            self.plot.addItem(self.playback_line)
            
        self.playback_line.setPos(position)
        
    def update_animations(self):
        """Update any ongoing animations"""
        # Placeholder for future animation updates
        pass
        
    def set_selection(self, start, end):
        """Set selection region"""
        if self.selection_region is None:
            self.selection_region = pg.LinearRegionItem(
                values=(start, end),
                brush=pg.mkBrush(139, 92, 246, 50),
                pen=pg.mkPen('#8B5CF6', width=2)
            )
            self.selection_region.sigRegionChanged.connect(self.on_selection_changed)
            self.plot.addItem(self.selection_region)
        else:
            self.selection_region.setRegion((start, end))
            
    def on_selection_changed(self):
        """Handle selection region changes"""
        if self.selection_region:
            start, end = self.selection_region.getRegion()
            self.selection_callback.emit(start, end)
            
    def clear_selection(self):
        """Clear the selection region"""
        if self.selection_region:
            self.plot.removeItem(self.selection_region)
            self.selection_region = None
            
    def start_region_selection(self, event):
        """Start region selection with right-click drag"""
        pos = event.pos()
        mouse_point = self.viewbox.mapSceneToView(pos)
        self.selection_start = mouse_point.x()
        
        # Create selection region if not exists
        if self.selection_region is None:
            self.selection_region = pg.LinearRegionItem(
                values=(self.selection_start, self.selection_start),
                brush=pg.mkBrush(139, 92, 246, 50),
                pen=pg.mkPen('#8B5CF6', width=2)
            )
            self.selection_region.setZValue(10)
            self.plot.addItem(self.selection_region)
            
            # Connect region change to show context menu
            self.selection_region.sigRegionChangeFinished.connect(self.on_region_selected)
        else:
            self.selection_region.setRegion((self.selection_start, self.selection_start))
            
    def on_region_selected(self):
        """Handle region selection completion"""
        if self.selection_region:
            region = self.selection_region.getRegion()
            self.show_region_context_menu(region)
            
    def show_region_context_menu(self, region):
        """Show context menu for selected region"""
        from PySide6.QtWidgets import QMenu, QApplication
        from PySide6.QtGui import QCursor
        
        menu = QMenu()
        
        # Add menu actions
        fft_action = menu.addAction("Show FFT Analysis")
        fft_action.triggered.connect(lambda: self.analyze_region_fft(region))
        
        menu.addSeparator()
        
        apply_gain_action = menu.addAction("Apply Gain...")
        apply_gain_action.triggered.connect(lambda: self.apply_region_gain(region))
        
        apply_filter_action = menu.addAction("Apply Filter...")
        apply_filter_action.triggered.connect(lambda: self.apply_region_filter(region))
        
        menu.addSeparator()
        
        zoom_action = menu.addAction("Zoom to Selection")
        zoom_action.triggered.connect(lambda: self.zoom_to_region(region))
        
        clear_action = menu.addAction("Clear Selection")
        clear_action.triggered.connect(self.clear_selection)
        
        # Show menu at cursor position
        menu.exec(QCursor.pos())
        
    def analyze_region_fft(self, region):
        """Analyze FFT of selected region"""
        if self.audio_data is None or self.sample_rate is None:
            return
            
        # Get time bounds
        start_time, end_time = region
        
        # Convert to sample indices
        start_idx = int(start_time * self.sample_rate)
        end_idx = int(end_time * self.sample_rate)
        
        # Ensure valid indices
        start_idx = max(0, start_idx)
        end_idx = min(len(self.audio_data), end_idx)
        
        if end_idx <= start_idx:
            return
            
        # Extract audio segment
        segment = self.audio_data[start_idx:end_idx]
        
        # Show FFT dialog
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        
        dialog = QDialog()
        dialog.setWindowTitle(f"FFT Analysis ({start_time:.2f}s - {end_time:.2f}s)")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(dialog)
        
        # Create matplotlib figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Time domain plot
        time_axis = np.arange(len(segment)) / self.sample_rate
        ax1.plot(time_axis, segment)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Time Domain')
        ax1.grid(True, alpha=0.3)
        
        # FFT plot
        fft_result = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(len(segment), 1/self.sample_rate)
        magnitude = np.abs(fft_result)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        ax2.semilogx(freqs[1:], magnitude_db[1:])  # Skip DC component
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude (dB)')
        ax2.set_title('Frequency Domain (FFT)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([20, self.sample_rate/2])
        
        # Find peak frequency
        peak_idx = np.argmax(magnitude[1:]) + 1  # Skip DC
        peak_freq = freqs[peak_idx]
        peak_mag = magnitude_db[peak_idx]
        ax2.axvline(peak_freq, color='red', linestyle='--', alpha=0.5)
        ax2.text(peak_freq, peak_mag, f'{peak_freq:.1f} Hz', 
                color='red', fontsize=10)
        
        plt.tight_layout()
        
        # Add canvas to dialog
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        
        # Add info label
        info_label = QLabel(f"Duration: {end_time - start_time:.3f}s\n"
                           f"Samples: {len(segment)}\n"
                           f"Peak Frequency: {peak_freq:.1f} Hz")
        info_label.setStyleSheet("color: white; padding: 10px;")
        layout.addWidget(info_label)
        
        dialog.exec()
        
    def apply_region_gain(self, region):
        """Apply gain to selected region"""
        # This would need to be implemented with the main window
        self.selection_callback.emit(region[0], region[1])
        
    def apply_region_filter(self, region):
        """Apply filter to selected region"""
        # This would need to be implemented with the main window
        self.selection_callback.emit(region[0], region[1])
        
    def zoom_to_region(self, region):
        """Zoom view to selected region"""
        start_time, end_time = region
        if self.freqs is not None:
            self.viewbox.setRange(xRange=(start_time, end_time),
                                 yRange=(self.freqs[0], self.freqs[-1]),
                                 padding=0)
            
    def plot_auto_tracks(self, tracks):
        """
        Overlay automatic detection tracks on the spectrogram.
        tracks: list of (time_array, freq_array) tuples
        """
        # Clear existing tracks
        self.clear_auto_tracks()
        
        # Plot each track
        for t_arr, f_arr in tracks:
            xs = np.asarray(t_arr, dtype=float)
            ys = np.asarray(f_arr, dtype=float)
            
            # Create track curve with modern styling
            curve = pg.PlotDataItem(
                xs, ys,
                pen=pg.mkPen(width=2, color=(255, 255, 0, 200)),  # Yellow with transparency
                antialias=True,
                connect='finite'
            )
            self.plot.addItem(curve)
            self.auto_tracks_items.append(curve)
            
    def clear_auto_tracks(self):
        """Clear all auto-detection tracks from the display"""
        if not hasattr(self, 'auto_tracks_items'):
            self.auto_tracks_items = []
            
        for item in self.auto_tracks_items:
            try:
                self.plot.removeItem(item)
            except:
                pass
        self.auto_tracks_items.clear()
    
    def clear(self):
        """Clear all spectrogram data and display"""
        self.audio_data = None
        self.sample_rate = None
        self.freqs = None
        self.times = None
        self.Sxx = None
        self.Sxx_display = None
        self.display_cache.clear()
        self.clear_auto_tracks()
        if self.img_item:
            self.img_item.clear()
        if self.selection_region:
            self.plot.removeItem(self.selection_region)
            self.selection_region = None


            #from here is this is the main window code i have used in differnet projects

            """
Modern Spectrogram GUI with Glassmorphic Design
Built with PySide6 for better performance and modern UI
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
from functools import lru_cache

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFileDialog, QPushButton, QLabel, QListWidget, QListWidgetItem,
    QSplitter, QMessageBox, QMenu, QApplication, QFrame,
    QToolButton, QGraphicsDropShadowEffect, QProgressDialog,
    QSlider, QSpinBox, QGroupBox, QGridLayout, QScrollArea,
    QSizePolicy, QGraphicsBlurEffect, QGraphicsOpacityEffect
)
from PySide6.QtGui import (
    QKeySequence, QAction, QIcon, QPalette, QColor, 
    QLinearGradient, QBrush, QPainter, QFont, QFontDatabase,
    QShortcut, QPen, QPixmap
)
from PySide6.QtCore import (
    Qt, Signal, QSize, QEvent, QSettings, QTimer,
    QPropertyAnimation, QEasingCurve, QRect, QPoint,
    QParallelAnimationGroup, QSequentialAnimationGroup,
    Property, QRunnable, QThreadPool
)

import qtawesome as qta


class GlassPanel(QFrame):
    """
    Glass morphism panel with blur effect and subtle borders
    """
    def __init__(self, parent=None, blur_radius=16):
        super().__init__(parent)
        self.setObjectName("GlassPanel")
        self.blur_radius = blur_radius
        self.setup_glass_effect()
        
    def setup_glass_effect(self):
        """Apply glass morphism effect"""
        self.setStyleSheet("""
            QFrame#GlassPanel {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(255, 255, 255, 0.08),
                    stop: 1 rgba(255, 255, 255, 0.03)
                );
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
            }
        """)
        
        # Add drop shadow for depth
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)


class ModernButton(QPushButton):
    """
    Pill-shaped button with gradient and hover effects
    """
    def __init__(self, text="", parent=None, primary=False):
        super().__init__(text, parent)
        self.primary = primary
        self.setup_style()
        self.setup_animations()
        
    def setup_style(self):
        """Apply modern button styling"""
        if self.primary:
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(
                        x1: 0, y1: 0, x2: 1, y2: 0,
                        stop: 0 #6366F1,
                        stop: 1 #8B5CF6
                    );
                    color: white;
                    border: none;
                    border-radius: 20px;
                    padding: 10px 24px;
                    font-weight: 600;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background: qlineargradient(
                        x1: 0, y1: 0, x2: 1, y2: 0,
                        stop: 0 #7C7FFF,
                        stop: 1 #9F6FFF
                    );
                }
                QPushButton:pressed {
                    background: qlineargradient(
                        x1: 0, y1: 0, x2: 1, y2: 0,
                        stop: 0 #5558E3,
                        stop: 1 #7D4EE8
                    );
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background: rgba(255, 255, 255, 0.05);
                    color: rgba(255, 255, 255, 0.9);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 20px;
                    padding: 10px 24px;
                    font-weight: 500;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background: rgba(255, 255, 255, 0.1);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }
                QPushButton:pressed {
                    background: rgba(255, 255, 255, 0.03);
                }
            """)
            
    def setup_animations(self):
        """Setup hover animations"""
        self.installEventFilter(self)
        
    def eventFilter(self, obj, event):
        if obj == self:
            if event.type() == QEvent.Type.Enter:
                self.animate_hover(True)
            elif event.type() == QEvent.Type.Leave:
                self.animate_hover(False)
        return super().eventFilter(obj, event)
        
    def animate_hover(self, hover):
        """Animate button on hover"""
        # Add subtle scale animation
        pass  # Will implement with QPropertyAnimation later


class ModernFileList(QListWidget):
    """
    Modern file list with glass morphism and smooth animations
    """
    fileDeleteRequested = Signal()
    filesDropped = Signal(list)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setSpacing(4)
        
        # Modern styling
        self.setStyleSheet("""
            QListWidget {
                background: rgba(255, 255, 255, 0.02);
                border: none;
                border-radius: 12px;
                padding: 8px;
                outline: none;
            }
            QListWidget::item {
                background: rgba(255, 255, 255, 0.05);
                color: rgba(255, 255, 255, 0.9);
                border-radius: 8px;
                padding: 12px;
                margin: 2px 0;
            }
            QListWidget::item:selected {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 rgba(99, 102, 241, 0.3),
                    stop: 1 rgba(139, 92, 246, 0.3)
                );
                border: 1px solid rgba(139, 92, 246, 0.5);
            }
            QListWidget::item:hover {
                background: rgba(255, 255, 255, 0.08);
            }
        """)
        
        # Sorting state
        self.sort_key = "name"
        self.sort_ascending = True
        self.settings = QSettings("SpectrogramGUI", "ModernFileList")
        self.load_sort_settings()
        
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.fileDeleteRequested.emit()
        else:
            super().keyPressEvent(event)
            
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)
            
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)
            
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            files = []
            for url in event.mimeData().urls():
                local_path = url.toLocalFile()
                if os.path.isfile(local_path) and local_path.lower().endswith((".wav", ".flac")):
                    if not any(self.item(i).data(Qt.ItemDataRole.UserRole) == local_path
                              for i in range(self.count())):
                        files.append(local_path)
                        item = QListWidgetItem(os.path.basename(local_path))
                        item.setIcon(qta.icon('fa5s.music', color='#8B5CF6'))
                        item.setData(Qt.ItemDataRole.UserRole, local_path)
                        self.addItem(item)
            if files:
                self.filesDropped.emit(files)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
            
    def load_sort_settings(self):
        """Load sorting preferences"""
        self.sort_key = self.settings.value("sort_key", "name")
        self.sort_ascending = self.settings.value("sort_ascending", True, type=bool)
        
    def save_sort_settings(self):
        """Save sorting preferences"""
        self.settings.setValue("sort_key", self.sort_key)
        self.settings.setValue("sort_ascending", self.sort_ascending)


class ModernControlBar(QWidget):
    """
    Modern audio control bar with glass effect
    """
    playRequested = Signal()
    pauseRequested = Signal()
    stopRequested = Signal()
    seekRequested = Signal(float)
    volumeChanged = Signal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)
        
        # Play/Pause button
        self.play_btn = ModernButton("", primary=True)
        self.play_btn.setIcon(qta.icon('fa5s.play', color='white'))
        self.play_btn.setFixedSize(48, 48)
        self.play_btn.clicked.connect(self.toggle_play)
        layout.addWidget(self.play_btn)
        
        # Stop button
        self.stop_btn = ModernButton("")
        self.stop_btn.setIcon(qta.icon('fa5s.stop', color='#E5E7EB'))
        self.stop_btn.setFixedSize(48, 48)
        self.stop_btn.clicked.connect(self.stopRequested.emit)
        layout.addWidget(self.stop_btn)
        
        # Time display
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.8);
                font-family: 'SF Mono', 'Consolas', monospace;
                font-size: 14px;
                background: rgba(0, 0, 0, 0.2);
                border-radius: 8px;
                padding: 6px 12px;
            }
        """)
        layout.addWidget(self.time_label)
        
        # Progress slider
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                width: 16px;
                height: 16px;
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #8B5CF6,
                    stop: 1 #6366F1
                );
                border-radius: 8px;
                margin: -5px 0;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #6366F1,
                    stop: 1 #8B5CF6
                );
                border-radius: 3px;
            }
        """)
        self.progress_slider.valueChanged.connect(self.on_seek)
        layout.addWidget(self.progress_slider, 1)
        
        # Volume control
        volume_icon = QLabel()
        volume_icon.setPixmap(qta.icon('fa5s.volume-up', color='#9CA3AF').pixmap(20, 20))
        layout.addWidget(volume_icon)
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(70)
        self.volume_slider.setFixedWidth(100)
        self.volume_slider.setStyleSheet(self.progress_slider.styleSheet())
        self.volume_slider.valueChanged.connect(lambda v: self.volumeChanged.emit(v / 100))
        layout.addWidget(self.volume_slider)
        
        self.is_playing = False
        
    def toggle_play(self):
        if self.is_playing:
            self.play_btn.setIcon(qta.icon('fa5s.play', color='white'))
            self.pauseRequested.emit()
        else:
            self.play_btn.setIcon(qta.icon('fa5s.pause', color='white'))
            self.playRequested.emit()
        self.is_playing = not self.is_playing
        
    def on_seek(self, value):
        if self.progress_slider.isSliderDown():
            self.seekRequested.emit(value / 1000.0)
            
    def update_time(self, current, total):
        """Update time display"""
        current_str = f"{int(current//60):02d}:{int(current%60):02d}"
        total_str = f"{int(total//60):02d}:{int(total%60):02d}"
        self.time_label.setText(f"{current_str} / {total_str}")
        
        if total > 0:
            self.progress_slider.blockSignals(True)
            self.progress_slider.setMaximum(int(total * 1000))
            self.progress_slider.setValue(int(current * 1000))
            self.progress_slider.blockSignals(False)


class ModernMainWindow(QMainWindow):
    """
    Main window with modern glassmorphic design
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrogram Analyzer - Modern UI")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize components
        self.audio_data = None
        self.sample_rate = None
        self.spectrogram_data = None
        self.current_file = None
        
        # Thread pool for background tasks
        self.thread_pool = QThreadPool()
        
        self.setup_ui()
        self.apply_theme()
        self.setup_shortcuts()
        
    def setup_ui(self):
        """Setup the modern UI layout"""
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top toolbar
        self.setup_toolbar()
        
        # Content area with splitter
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setHandleWidth(1)
        
        # Left panel (file list)
        left_panel = GlassPanel()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(12, 12, 12, 12)
        
        # File list header
        header_layout = QHBoxLayout()
        header_label = QLabel("Audio Files")
        header_label.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.9);
                font-size: 16px;
                font-weight: 600;
            }
        """)
        header_layout.addWidget(header_label)
        
        add_btn = ModernButton("+")
        add_btn.setFixedSize(32, 32)
        add_btn.clicked.connect(self.add_files)
        header_layout.addWidget(add_btn)
        
        left_layout.addLayout(header_layout)
        
        # File list
        self.file_list = ModernFileList()
        self.file_list.itemDoubleClicked.connect(self.load_selected_file)
        self.file_list.fileDeleteRequested.connect(self.remove_selected_files)
        left_layout.addWidget(self.file_list)
        
        # Add file controls
        file_controls = QHBoxLayout()
        sort_btn = ModernButton("Sort")
        sort_btn.clicked.connect(self.show_sort_menu)
        clear_btn = ModernButton("Clear")
        clear_btn.clicked.connect(self.clear_files)
        file_controls.addWidget(sort_btn)
        file_controls.addWidget(clear_btn)
        left_layout.addLayout(file_controls)
        
        content_splitter.addWidget(left_panel)
        
        # Right panel (spectrogram and controls)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        
        # Spectrogram display area (placeholder for now)
        self.spectrogram_area = GlassPanel()
        spectrogram_layout = QVBoxLayout(self.spectrogram_area)
        
        # Placeholder for spectrogram canvas
        self.spectrogram_placeholder = QLabel("Drop audio files to begin")
        self.spectrogram_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spectrogram_placeholder.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.4);
                font-size: 18px;
                padding: 40px;
            }
        """)
        spectrogram_layout.addWidget(self.spectrogram_placeholder)
        
        right_layout.addWidget(self.spectrogram_area, 1)
        
        # Audio controls
        self.control_bar = ModernControlBar()
        self.control_bar.playRequested.connect(self.play_audio)
        self.control_bar.pauseRequested.connect(self.pause_audio)
        self.control_bar.stopRequested.connect(self.stop_audio)
        right_layout.addWidget(self.control_bar)
        
        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([300, 1100])
        
        main_layout.addWidget(content_splitter)
        
        # Status bar
        self.setup_status_bar()
        
    def setup_toolbar(self):
        """Setup modern toolbar"""
        toolbar = self.addToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setStyleSheet("""
            QToolBar {
                background: rgba(0, 0, 0, 0.3);
                border: none;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                padding: 8px;
                spacing: 8px;
            }
        """)
        
        # File operations
        open_action = QAction(qta.icon('fa5s.folder-open', color='#9CA3AF'), "Open", self)
        open_action.triggered.connect(self.add_files)
        toolbar.addAction(open_action)
        
        toolbar.addSeparator()
        
        # View operations
        zoom_in_action = QAction(qta.icon('fa5s.search-plus', color='#9CA3AF'), "Zoom In", self)
        zoom_out_action = QAction(qta.icon('fa5s.search-minus', color='#9CA3AF'), "Zoom Out", self)
        reset_action = QAction(qta.icon('fa5s.compress', color='#9CA3AF'), "Reset View", self)
        
        toolbar.addAction(zoom_in_action)
        toolbar.addAction(zoom_out_action)
        toolbar.addAction(reset_action)
        
        toolbar.addSeparator()
        
        # Processing operations
        filter_action = QAction(qta.icon('fa5s.filter', color='#9CA3AF'), "Filters", self)
        settings_action = QAction(qta.icon('fa5s.cog', color='#9CA3AF'), "Settings", self)
        
        toolbar.addAction(filter_action)
        toolbar.addAction(settings_action)
        
    def setup_status_bar(self):
        """Setup modern status bar"""
        status = self.statusBar()
        status.setStyleSheet("""
            QStatusBar {
                background: rgba(0, 0, 0, 0.3);
                color: rgba(255, 255, 255, 0.6);
                border-top: 1px solid rgba(255, 255, 255, 0.05);
                font-size: 12px;
            }
        """)
        status.showMessage("Ready")
        
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        # File operations
        QShortcut(QKeySequence("Ctrl+O"), self, self.add_files)
        QShortcut(QKeySequence("Delete"), self.file_list, self.remove_selected_files)
        
        # Playback
        QShortcut(QKeySequence("Space"), self, self.control_bar.toggle_play)
        QShortcut(QKeySequence("S"), self, self.stop_audio)
        
        # View
        QShortcut(QKeySequence("Ctrl++"), self, self.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self, self.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, self.reset_zoom)
        
    def apply_theme(self):
        """Apply the dark glassmorphic theme"""
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #0F0F1E,
                    stop: 0.5 #1A1A2E,
                    stop: 1 #16213E
                );
            }
            QSplitter::handle {
                background: rgba(255, 255, 255, 0.05);
            }
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 0.02);
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 0.15);
            }
            QMenu {
                background: rgba(20, 20, 30, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item {
                color: rgba(255, 255, 255, 0.9);
                padding: 8px 16px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: rgba(99, 102, 241, 0.3);
            }
        """)
        
    def add_files(self):
        """Add audio files to the list"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio Files",
            "",
            "Audio Files (*.wav *.flac);;All Files (*.*)"
        )
        
        for file_path in files:
            if not any(self.file_list.item(i).data(Qt.ItemDataRole.UserRole) == file_path
                      for i in range(self.file_list.count())):
                item = QListWidgetItem(os.path.basename(file_path))
                item.setIcon(qta.icon('fa5s.music', color='#8B5CF6'))
                item.setData(Qt.ItemDataRole.UserRole, file_path)
                self.file_list.addItem(item)
                
    def remove_selected_files(self):
        """Remove selected files from the list"""
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
            
    def clear_files(self):
        """Clear all files from the list"""
        self.file_list.clear()
        
    def show_sort_menu(self):
        """Show sorting options menu"""
        menu = QMenu(self)
        menu.setStyleSheet(self.styleSheet())
        
        # Sort options
        name_action = menu.addAction("Sort by Name")
        time_action = menu.addAction("Sort by Time")
        size_action = menu.addAction("Sort by Size")
        
        menu.addSeparator()
        
        asc_action = menu.addAction("Ascending")
        desc_action = menu.addAction("Descending")
        
        # Execute menu
        menu.exec(self.cursor().pos())
        
    def load_selected_file(self, item):
        """Load the selected audio file"""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path:
            self.current_file = file_path
            self.statusBar().showMessage(f"Loading: {os.path.basename(file_path)}")
            # Load audio and compute spectrogram (to be implemented)
            
    def play_audio(self):
        """Play the current audio"""
        if self.current_file:
            self.statusBar().showMessage("Playing audio...")
            
    def pause_audio(self):
        """Pause audio playback"""
        self.statusBar().showMessage("Paused")
        
    def stop_audio(self):
        """Stop audio playback"""
        self.control_bar.is_playing = False
        self.control_bar.play_btn.setIcon(qta.icon('fa5s.play', color='white'))
        self.statusBar().showMessage("Stopped")
        
    def zoom_in(self):
        """Zoom in on spectrogram"""
        pass
        
    def zoom_out(self):
        """Zoom out on spectrogram"""
        pass
        
    def reset_zoom(self):
        """Reset spectrogram zoom"""
        pass