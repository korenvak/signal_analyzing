import sys
import os
import logging
from typing import Optional, Dict, Any
import numpy as np

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                              QWidget, QTabWidget, QMenuBar, QStatusBar, QToolBar,
                              QSlider, QLabel, QComboBox, QPushButton, QProgressBar,
                              QFileDialog, QMessageBox, QSplitter, QFrame)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QAction, QKeySequence

try:
    import vispy
    from vispy import scene, app
    from vispy.scene import visuals
    HAS_VISPY = True
except ImportError:
    vispy = None
    HAS_VISPY = False

from ..core.data_loader import ChunkedAudioLoader
from ..core.cache_manager import CacheManager
from ..core.task_manager import TaskManager
from ..core.tile_cache import TileCache
from ..core.tile_manager import TileManager as TileMgr
from ..core.gpu_memory_manager import get_gpu_memory_manager
from ..engines.spectrogram_engine import SpectrogramEngine
from ..engines.cepstrogram_engine import CepstrogramEngine
from ..engines.fk_engine import FKEngine
from ..rendering.render_manager import RenderManager
from ..rendering.texture_atlas import TextureAtlas

logger = logging.getLogger(__name__)


class VisPyCanvas(scene.SceneCanvas):
    """Custom VisPy canvas for audio visualization."""
    
    def __init__(self, view_type: str, parent=None):
        if not HAS_VISPY:
            raise RuntimeError("VisPy not available")
        
        super().__init__(keys='interactive', parent=parent, size=(800, 600))
        
        self.unfreeze()
        self.view_type = view_type
        
        # Create main view with grid
        self.view = self.central_widget.add_view()
        
        # Use PanZoomCamera with independent axis control
        self.view.camera = scene.PanZoomCamera(aspect=None)  # aspect=None allows independent zoom
        self.view.camera.set_range(x=(-1, 1), y=(-1, 1))
        
        # Store data bounds for camera constraints
        self.data_bounds = None  # Will be set when data is loaded
        
        # Image visual for spectrogram data
        # Use 'nearest' interpolation to avoid blurring
        self.image_visual = scene.visuals.Image(parent=self.view.scene, interpolation='nearest')
        
        # OpenGL texture size limit (will be detected)
        self.max_texture_size = 16384  # Conservative default
        
        # Axis labels
        self.x_axis_label = scene.visuals.Text('', color='white', font_size=14,
                                               parent=self.view.scene, anchor_x='center', anchor_y='top')
        self.y_axis_label = scene.visuals.Text('', color='white', font_size=14,
                                               parent=self.view.scene, anchor_x='right', anchor_y='center')
        
        # Crosshair visuals
        self.crosshair_v = scene.visuals.Line(parent=self.view.scene, color='white', width=2)
        self.crosshair_h = scene.visuals.Line(parent=self.view.scene, color='white', width=2)
        
        # Text for readouts
        self.text_visual = scene.visuals.Text('', parent=self.view.scene, color='white', 
                                            font_size=12, anchor_x='left', anchor_y='top')
        
        # Mouse interaction
        self.mouse_pos = (0, 0)
        self.crosshair_enabled = False
        
        # Timer for updating camera constraints and labels
        self.update_timer = app.Timer(interval=0.1, connect=self.on_timer, start=True)
        
        # Connect events for zoom control
        self.events.mouse_wheel.connect(self.on_mouse_wheel)
        
        self.freeze()
    
    def on_timer(self, event):
        """Timer callback to update camera constraints and labels."""
        # Constrain camera to data bounds
        self.constrain_camera_to_bounds()
        # Update axis labels
        self.update_axis_labels()
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for axis-specific zoom.
        
        - Shift + Wheel: Zoom time axis (X) only
        - Ctrl + Wheel: Zoom frequency axis (Y) only
        - Wheel alone: Zoom both (default)
        """
        logger.debug(f"Mouse wheel: delta={event.delta}, modifiers={event.modifiers}")
        
        if event.modifiers:
            # Prevent default zoom
            event.handled = True
            
            # Determine zoom factor
            factor = 0.9 if event.delta[1] > 0 else 1.1
            
            logger.info(f"Zoom with modifiers: {event.modifiers}, factor={factor}")
            
            if 'Shift' in event.modifiers:
                # Zoom time axis only
                logger.info("Zooming TIME axis only")
                self.zoom_axis('x', factor)
            elif 'Control' in event.modifiers:
                # Zoom frequency axis only
                logger.info("Zooming FREQUENCY axis only")
                self.zoom_axis('y', factor)
    
    def constrain_camera_to_bounds(self):
        """Constrain camera to data bounds - prevent panning to empty areas."""
        if self.data_bounds is None:
            return
        
        # Get current camera range
        try:
            x_range, y_range = self.view.camera.get_range()
        except:
            return
        
        # Extract bounds
        time_min, freq_min, time_max, freq_max = self.data_bounds
        
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        # Check if we're outside bounds
        constrained = False
        
        # Constrain X (time)
        if x_min < time_min:
            x_max += (time_min - x_min)
            x_min = time_min
            constrained = True
        if x_max > time_max:
            x_min -= (x_max - time_max)
            x_max = time_max
            constrained = True
        
        # Constrain Y (frequency)
        if y_min < freq_min:
            y_max += (freq_min - y_min)
            y_min = freq_min
            constrained = True
        if y_max > freq_max:
            y_min -= (y_max - freq_max)
            y_max = freq_max
            constrained = True
        
        # Update camera if needed
        if constrained:
            logger.info(f"Constraining camera: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")
            self.view.camera.set_range(x=(x_min, x_max), y=(y_min, y_max), margin=0)
    
    def update_axis_labels(self):
        """Update axis labels with appropriate units based on zoom level."""
        if self.data_bounds is None:
            return
        
        # Get camera range instead of rect
        try:
            x_range, y_range = self.view.camera.get_range()
            if x_range is None or y_range is None:
                return
        except Exception as e:
            logger.debug(f"Cannot get camera range: {e}")
            return
        
        time_min, freq_min, time_max, freq_max = self.data_bounds
        x_min, x_max = x_range
        y_min, y_max = y_range
        w = x_max - x_min
        h = y_max - y_min
        
        # Format time axis label (X-axis)
        time_span = w
        if time_span < 1.0:
            time_unit = "TIME (ms)"
        elif time_span < 60.0:
            time_unit = "TIME (s)"
        elif time_span < 3600.0:
            time_unit = "TIME (min)"
        else:
            time_unit = "TIME (hr)"
        
        # Format frequency axis label (Y-axis)
        freq_span = h
        if freq_span < 1000.0:
            freq_unit = "FREQ (Hz)"
        else:
            freq_unit = "FREQ (kHz)"
        
        # Position labels at visible locations
        # Bottom right for time axis label
        self.x_axis_label.text = time_unit
        self.x_axis_label.pos = (x_max - w * 0.15, y_min + h * 0.05)
        self.x_axis_label.font_size = 18
        self.x_axis_label.color = (1, 1, 0, 1)  # Bright yellow
        
        # Top left for frequency axis label  
        self.y_axis_label.text = freq_unit
        self.y_axis_label.pos = (x_min + w * 0.1, y_max - h * 0.05)
        self.y_axis_label.font_size = 18
        self.y_axis_label.color = (1, 1, 0, 1)  # Bright yellow
        
        logger.debug(f"Labels updated: {time_unit} at {self.x_axis_label.pos}, {freq_unit} at {self.y_axis_label.pos}")
    
    def zoom_axis(self, axis: str, factor: float):
        """Zoom on a specific axis only.
        
        Args:
            axis: 'x' for time, 'y' for frequency
            factor: Zoom factor (<1 zoom in, >1 zoom out)
        """
        if self.data_bounds is None:
            return
        
        try:
            x_range, y_range = self.view.camera.get_range()
        except:
            return
        
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        if axis == 'x':
            # Zoom time axis (X) only
            w = x_max - x_min
            new_w = w * factor
            center_x = (x_min + x_max) / 2
            new_x_min = center_x - new_w / 2
            new_x_max = center_x + new_w / 2
            self.view.camera.set_range(x=(new_x_min, new_x_max), y=y_range, margin=0)
        elif axis == 'y':
            # Zoom frequency axis (Y) only
            h = y_max - y_min
            new_h = h * factor
            center_y = (y_min + y_max) / 2
            new_y_min = center_y - new_h / 2
            new_y_max = center_y + new_h / 2
            self.view.camera.set_range(x=x_range, y=(new_y_min, new_y_max), margin=0)
    
    def update_image(self, data: np.ndarray, extent: tuple = None):
        """Update the displayed image data."""
        if data is not None and data.size > 0:
            # Ensure data is in correct format for VisPy
            # VisPy Image expects: rows=Y-axis, columns=X-axis
            # For spectrogram: we want time=X-axis, frequency=Y-axis
            # Data comes as (frequency_bins, time_frames) which is already correct!
            if data.ndim == 2:
                # Data format (freq_bins, time_frames) is correct for VisPy
                # freq_bins = rows = Y-axis (frequency)
                # time_frames = columns = X-axis (time)
                display_data = data.astype(np.float32)
                
                # Check OpenGL texture size limits
                original_shape = display_data.shape
                downsample_factor_x = 1
                downsample_factor_y = 1
                
                # Check if we exceed texture limits
                if display_data.shape[1] > self.max_texture_size:
                    downsample_factor_x = int(np.ceil(display_data.shape[1] / self.max_texture_size))
                    logger.warning(f"Data width ({display_data.shape[1]}) exceeds OpenGL limit ({self.max_texture_size})")
                    logger.warning(f"Downsampling by {downsample_factor_x}x in time to fit texture")
                    display_data = display_data[:, ::downsample_factor_x]
                
                if display_data.shape[0] > self.max_texture_size:
                    downsample_factor_y = int(np.ceil(display_data.shape[0] / self.max_texture_size))
                    logger.warning(f"Data height ({display_data.shape[0]}) exceeds OpenGL limit ({self.max_texture_size})")
                    logger.warning(f"Downsampling by {downsample_factor_y}x in frequency to fit texture")
                    display_data = display_data[::downsample_factor_y, :]
                
                if downsample_factor_x > 1 or downsample_factor_y > 1:
                    logger.info(f"Downsampled: {original_shape} -> {display_data.shape}")
                
                # Normalize data for proper VisPy display
                # Spectrogram data is typically in dB (e.g., -120 to 0 dB)
                # Normalize to 0-1 range for proper colormap visualization
                data_min = np.min(display_data)
                data_max = np.max(display_data)
                
                if data_max > data_min:  # Avoid division by zero
                    display_data = (display_data - data_min) / (data_max - data_min)
                else:
                    display_data = np.zeros_like(display_data)
                
                logger.debug(f"Data normalized: {data_min:.1f} to {data_max:.1f}")
                logger.debug(f"Display shape: {display_data.shape}")
            else:
                display_data = data.astype(np.float32)
            
            if extent:
                # extent = (time_start, time_end, freq_start, freq_end)
                time_start, time_end, freq_start, freq_end = extent
                time_width = time_end - time_start
                freq_height = freq_end - freq_start
                
                logger.debug(f"Transform: time [{time_start}, {time_end}], freq [{freq_start}, {freq_end}]")
                logger.debug(f"Data pixels: {display_data.shape[1]} x {display_data.shape[0]}")
                
                # Set data with proper clim for color mapping
                self.image_visual.set_data(display_data)
                self.image_visual.clim = (0, 1)  # Data is normalized to 0-1
                
                # Calculate transform to map pixel coordinates to world coordinates
                # VisPy Image: pixel (j, i) where j=column (X), i=row (Y)
                # display_data[i, j] where i=row (Y/freq), j=column (X/time)
                
                # We want to map:
                # - pixel column 0 -> time_start
                # - pixel column (shape[1]-1) -> time_end
                # - pixel row 0 -> freq_start  
                # - pixel row (shape[0]-1) -> freq_end
                
                # Scale factor: world_units per pixel
                x_scale = time_width / display_data.shape[1]
                y_scale = freq_height / display_data.shape[0]
                
                # Translate: position of pixel (0, 0) in world coords
                # Pixel (0,0) is bottom-left, should map to (time_start, freq_start)
                x_translate = time_start
                y_translate = freq_start
                
                transform = scene.STTransform(
                    scale=(x_scale, y_scale),
                    translate=(x_translate, y_translate)
                )
                self.image_visual.transform = transform
                
                logger.debug(f"Transform scale: ({x_scale:.6f}, {y_scale:.2f})")
                
                # Store data bounds for camera constraints
                self.data_bounds = (time_start, time_end, freq_start, freq_end)
                
                # Update camera to show the full data range
                self.view.camera.set_range(
                    x=(time_start, time_end),
                    y=(freq_start, freq_end),
                    margin=0
                )
                
                # Update axis labels
                self.update_axis_labels()
                
                logger.debug(f"Camera range: X=[{time_start}, {time_end}], Y=[{freq_start}, {freq_end}]")
            else:
                self.image_visual.set_data(display_data)
                self.image_visual.clim = (0, 1)
    
    def set_crosshair(self, enabled: bool, pos: tuple = None):
        """Enable/disable crosshair display."""
        self.crosshair_enabled = enabled
        
        if enabled and pos:
            x, y = pos
            
            # Update crosshair lines
            view_bounds = self.view.camera.get_range()
            x_range = view_bounds[0]
            y_range = view_bounds[1]
            
            # Vertical line
            self.crosshair_v.set_data(np.array([[x, y_range[0]], [x, y_range[1]]]))
            
            # Horizontal line  
            self.crosshair_h.set_data(np.array([[x_range[0], y], [x_range[1], y]]))
            
            self.crosshair_v.visible = True
            self.crosshair_h.visible = True
        else:
            self.crosshair_v.visible = False
            self.crosshair_h.visible = False
    
    def update_text_readout(self, text: str, pos: tuple = None):
        """Update text readout display."""
        self.text_visual.text = text
        if pos:
            self.text_visual.pos = pos
    
    def on_mouse_move(self, event):
        """Handle mouse movement for crosshair updates."""
        if event.pos is not None:
            # Convert screen coordinates to world coordinates
            tr = self.scene.node_transform(self.view.scene)
            world_pos = tr.map(event.pos)
            
            self.mouse_pos = (world_pos[0], world_pos[1])
            
            if self.crosshair_enabled:
                self.set_crosshair(True, self.mouse_pos)
                
                # Update readout text
                readout = f"Time: {world_pos[0]:.3f}s, Freq: {world_pos[1]:.0f}Hz"
                self.update_text_readout(readout, (10, 30))

class StatusWidget(QWidget):
    """Status widget showing performance metrics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        
        self.fps_label = QLabel("FPS: --")
        self.memory_label = QLabel("Memory: --")
        self.gpu_label = QLabel("GPU: --")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        layout.addWidget(QLabel("Performance:"))
        layout.addWidget(self.fps_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.memory_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.gpu_label)
        layout.addStretch()
        layout.addWidget(self.progress_bar)
    
    def update_stats(self, fps: float = None, memory_mb: float = None, 
                    gpu_mb: float = None, progress: float = None):
        """Update status display."""
        if fps is not None:
            self.fps_label.setText(f"FPS: {fps:.1f}")
        
        if memory_mb is not None:
            self.memory_label.setText(f"Memory: {memory_mb:.0f}MB")
        
        if gpu_mb is not None:
            self.gpu_label.setText(f"GPU: {gpu_mb:.0f}MB")
        
        if progress is not None:
            if 0 <= progress < 1:
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(int(progress * 100))
            else:
                self.progress_bar.setVisible(False)

class ControlsWidget(QWidget):
    """Widget containing analysis controls."""
    
    # Signals
    parameters_changed = Signal(dict)
    colormap_changed = Signal(str)
    db_range_changed = Signal(float, float)
    refresh_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setup_ui()
        self.connect_signals()
    
    def setup_ui(self):
        """Setup the control interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # FFT Parameters
        fft_frame = QFrame()
        fft_frame.setFrameStyle(QFrame.StyledPanel)
        fft_layout = QHBoxLayout(fft_frame)
        
        fft_layout.addWidget(QLabel("FFT Size:"))
        self.fft_size_combo = QComboBox()
        self.fft_size_combo.addItems(['512', '1024', '2048', '4096', '8192'])
        self.fft_size_combo.setCurrentText('2048')
        fft_layout.addWidget(self.fft_size_combo)
        
        fft_layout.addWidget(QLabel("Hop:"))
        self.hop_combo = QComboBox()
        self.hop_combo.addItems(['128', '256', '512', '1024'])
        self.hop_combo.setCurrentText('512')
        fft_layout.addWidget(self.hop_combo)
        
        # Colormap
        colormap_frame = QFrame()
        colormap_frame.setFrameStyle(QFrame.StyledPanel)
        colormap_layout = QHBoxLayout(colormap_frame)
        
        colormap_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'jet', 'magma'])
        colormap_layout.addWidget(self.colormap_combo)
        
        # dB Range
        db_frame = QFrame()
        db_frame.setFrameStyle(QFrame.StyledPanel)
        db_layout = QHBoxLayout(db_frame)
        
        db_layout.addWidget(QLabel("dB Range:"))
        
        self.db_min_slider = QSlider(Qt.Horizontal)
        self.db_min_slider.setRange(-120, 0)
        self.db_min_slider.setValue(-80)
        self.db_min_label = QLabel("-80")
        
        self.db_max_slider = QSlider(Qt.Horizontal)
        self.db_max_slider.setRange(-80, 20)
        self.db_max_slider.setValue(0)
        self.db_max_label = QLabel("0")
        
        db_layout.addWidget(QLabel("Min:"))
        db_layout.addWidget(self.db_min_slider)
        db_layout.addWidget(self.db_min_label)
        db_layout.addWidget(QLabel("Max:"))
        db_layout.addWidget(self.db_max_slider)
        db_layout.addWidget(self.db_max_label)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        
        # Add all frames to main layout
        layout.addWidget(fft_frame)
        layout.addWidget(colormap_frame)
        layout.addWidget(db_frame)
        layout.addStretch()
        layout.addWidget(self.refresh_button)
    
    def connect_signals(self):
        """Connect widget signals."""
        self.fft_size_combo.currentTextChanged.connect(self.emit_parameters_changed)
        self.hop_combo.currentTextChanged.connect(self.emit_parameters_changed)
        self.colormap_combo.currentTextChanged.connect(self.colormap_changed.emit)
        
        self.db_min_slider.valueChanged.connect(self.update_db_range)
        self.db_max_slider.valueChanged.connect(self.update_db_range)
        
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
    
    def emit_parameters_changed(self):
        """Emit parameters changed signal."""
        params = {
            'fft_size': int(self.fft_size_combo.currentText()),
            'hop_length': int(self.hop_combo.currentText())
        }
        self.parameters_changed.emit(params)
    
    def update_db_range(self):
        """Update dB range display and emit signal."""
        db_min = self.db_min_slider.value()
        db_max = self.db_max_slider.value()
        
        # Ensure min < max
        if db_min >= db_max:
            if self.sender() == self.db_min_slider:
                db_max = db_min + 10
                self.db_max_slider.setValue(db_max)
            else:
                db_min = db_max - 10
                self.db_min_slider.setValue(db_min)
        
        self.db_min_label.setText(str(db_min))
        self.db_max_label.setText(str(db_max))
        
        self.db_range_changed.emit(db_min, db_max)

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("GPU-Accelerated Audio Visualizer")
        self.setMinimumSize(1200, 800)
        
        # Core components
        self.cache_manager = CacheManager(max_memory_mb=2048, max_gpu_memory_mb=1024)
        self.task_manager = TaskManager()
        self.audio_loader = ChunkedAudioLoader()
        
        # GPU Memory Management
        self.gpu_memory_manager = get_gpu_memory_manager()
        
        # Tile-based caching and rendering
        self.tile_cache = TileCache(max_memory_tiles=100, max_disk_gb=10.0)
        
        # Computation engines
        self.spectrogram_engine = SpectrogramEngine(self.cache_manager, self.task_manager)
        self.cepstrogram_engine = CepstrogramEngine(self.cache_manager, self.task_manager, self.spectrogram_engine)
        self.fk_engine = FKEngine(self.cache_manager, self.task_manager, self.spectrogram_engine)
        
        # Tile manager (coordinates tiles, cache, and atlases)
        self.tile_manager = TileMgr(
            tile_cache=self.tile_cache,
            engines={
                'spectrogram': self.spectrogram_engine,
                'cepstrogram': self.cepstrogram_engine,
                'fk_transform': self.fk_engine
            }
        )
        
        # Set up default array geometry for F-K analysis (simulated linear array)
        # For single-channel audio, we simulate a simple 8-element linear array
        array_spacing = 0.1  # 10 cm spacing
        n_sensors = 8
        sensor_positions = np.array([[i * array_spacing, 0.0] for i in range(n_sensors)])
        self.fk_engine.set_array_geometry(sensor_positions)
        
        # Current state
        self.current_file = None
        # Default view range (will be updated when file is loaded)
        self.current_view_range = ((0.0, 10.0), (0.0, 22050.0))  # (time, freq)
        
        # Setup UI
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_status_bar()
        
        # Performance timer
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self.update_performance_stats)
        self.perf_timer.start(1000)  # Update every second
        
        # Auto-load file if specified
        initial_file = os.environ.get('AUDIO_VISUALIZER_INITIAL_FILE')
        if initial_file and os.path.exists(initial_file):
            QTimer.singleShot(1000, lambda: self.load_audio_file(initial_file))
        
    def setup_ui(self):
        """Setup the main user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Controls
        self.controls_widget = ControlsWidget()
        main_layout.addWidget(self.controls_widget)
        
        # Tab widget for different views
        self.tab_widget = QTabWidget()
        
        # Create tabs
        if HAS_VISPY:
            self.spectrogram_canvas = VisPyCanvas('spectrogram')
            self.cepstrogram_canvas = VisPyCanvas('cepstrogram')  
            self.fk_canvas = VisPyCanvas('fk_transform')
            
            self.tab_widget.addTab(self.spectrogram_canvas.native, "Spectrogram")
            self.tab_widget.addTab(self.cepstrogram_canvas.native, "Cepstrogram")
            self.tab_widget.addTab(self.fk_canvas.native, "F-K Transform")
        else:
            # Fallback widgets when VisPy not available
            self.tab_widget.addTab(QLabel("VisPy not available"), "Spectrogram")
            self.tab_widget.addTab(QLabel("VisPy not available"), "Cepstrogram")
            self.tab_widget.addTab(QLabel("VisPy not available"), "F-K Transform")
        
        main_layout.addWidget(self.tab_widget)
        
        # Connect signals
        self.controls_widget.parameters_changed.connect(self.on_parameters_changed)
        self.controls_widget.colormap_changed.connect(self.on_colormap_changed)
        self.controls_widget.db_range_changed.connect(self.on_db_range_changed)
        self.controls_widget.refresh_requested.connect(self.refresh_current_view)
        
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
    
    def setup_menu_bar(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Audio File...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_audio_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        zoom_fit_action = QAction("Zoom to Fit", self)
        zoom_fit_action.setShortcut("Ctrl+0")
        zoom_fit_action.triggered.connect(self.zoom_to_fit)
        view_menu.addAction(zoom_fit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("Analysis")
        
        refresh_action = QAction("Refresh Current View", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.refresh_current_view)
        analysis_menu.addAction(refresh_action)
    
    def setup_status_bar(self):
        """Setup the status bar."""
        self.status_widget = StatusWidget()
        self.statusBar().addPermanentWidget(self.status_widget)
        self.statusBar().showMessage("Ready")
    
    def load_audio_file(self, file_path: str):
        """Load an audio file for analysis."""
        try:
            self.statusBar().showMessage("Loading audio file...")
            
            sample_rate, duration = self.audio_loader.load_file(file_path)
            
            self.current_file = file_path
            # Show full audio duration instead of just 10 seconds
            self.current_view_range = ((0.0, duration), 
                                     (0.0, sample_rate / 2))
            
            # Update spectrogram engine parameters
            self.spectrogram_engine.set_parameters(sample_rate=sample_rate)
            
            self.statusBar().showMessage(
                f"Loaded: {os.path.basename(file_path)} "
                f"({duration:.1f}s, {sample_rate}Hz)"
            )
            
            # Refresh current view
            self.refresh_current_view()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load audio file:\n{str(e)}")
            self.statusBar().showMessage("Ready")
    
    def open_audio_file(self):
        """Open an audio file dialog for analysis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File",
            "", "Audio Files (*.wav *.flac *.mp3 *.ogg);;All Files (*)"
        )
        
        if file_path:
            self.load_audio_file(file_path)
    
    def on_parameters_changed(self, params: dict):
        """Handle parameter changes."""
        self.spectrogram_engine.set_parameters(**params)
        
        # Clear cache for affected views
        self.cache_manager.clear_view_cache('spectrogram')
        self.cache_manager.clear_view_cache('cepstrogram')
        
        self.refresh_current_view()
    
    def on_colormap_changed(self, colormap: str):
        """Handle colormap changes."""
        # Update render manager when available
        pass
    
    def on_db_range_changed(self, db_min: float, db_max: float):
        """Handle dB range changes."""
        # Update render manager when available
        pass
    
    def on_tab_changed(self, index: int):
        """Handle tab changes - lazy loading trigger."""
        if not self.current_file:
            return
        
        tab_names = ['spectrogram', 'cepstrogram', 'fk_transform']
        if 0 <= index < len(tab_names):
            view_type = tab_names[index]
            self.load_view_data(view_type)
    
    def load_view_data(self, view_type: str):
        """Load data for specified view type."""
        if not self.current_file:
            return
        
        self.statusBar().showMessage(f"Computing {view_type}...")
        
        # Get audio data for current time range
        time_range = self.current_view_range[0]
        start_sample = int(time_range[0] * self.spectrogram_engine.sample_rate)
        end_sample = int(time_range[1] * self.spectrogram_engine.sample_rate)
        
        # Clamp to available samples
        total_samples = self.audio_loader.total_samples
        num_samples = min(end_sample - start_sample, total_samples - start_sample)
        
        try:
            audio_data = self.audio_loader.get_chunk(start_sample, num_samples)
            
            # Use direct computation for now (tile system integration ongoing)
            if view_type == 'spectrogram':
                self.load_spectrogram_data(audio_data)
            elif view_type == 'cepstrogram':
                self.load_cepstrogram_data(audio_data)
            elif view_type == 'fk_transform':
                self.load_fk_data(audio_data)
                
        except Exception as e:
            logger.error(f"Error loading {view_type}: {e}")
            self.statusBar().showMessage(f"Error: {str(e)}")
    
    def update_atlas_display(self, view_type: str):
        """Update the display with computed data.
        
        Temporary: Display computed spectrogram directly instead of atlas
        until full tile-based rendering is properly integrated.
        
        Args:
            view_type: Type of view to update
        """
        # For now, use the old direct display method
        # The full tile-atlas system needs more integration work
        # This ensures we show actual data instead of empty atlas
        pass  # Will be handled by old load_spectrogram_data methods
    
    def load_spectrogram_data(self, audio_data: np.ndarray):
        """Load spectrogram data using batched FFT."""
        try:
            # Compute STFT using optimized batched FFT
            magnitude_db, frequencies, times = self.spectrogram_engine.compute_stft_batched(audio_data)
            
            logger.info(f"Computed spectrogram: {magnitude_db.shape}")
            
            if HAS_VISPY and magnitude_db.size > 0:
                # Update display
                extent = (self.current_view_range[0][0], self.current_view_range[0][1],
                         self.current_view_range[1][0], self.current_view_range[1][1])
                self.spectrogram_canvas.update_image(magnitude_db, extent)
                frames = magnitude_db.shape[1]
                self.statusBar().showMessage(f"Spectrogram ready ({frames} frames)")
        except Exception as e:
            logger.error(f"Spectrogram computation error: {e}")
            self.statusBar().showMessage(f"Spectrogram error: {str(e)}")
    
    def load_cepstrogram_data(self, audio_data: np.ndarray):
        """Load cepstrogram data."""
        try:
            # First compute spectrogram
            magnitude_db, frequencies, times = self.spectrogram_engine.compute_stft_batched(audio_data)
            
            logger.info(f"Computing cepstrogram from spectrogram: {magnitude_db.shape}")
            
            # Clear mel filterbank cache to rebuild with correct dimensions
            self.cepstrogram_engine._mel_filterbank = None
            
            # Then compute cepstrogram from it
            cepstral = self.cepstrogram_engine.compute_cepstrogram_from_spectrogram(
                magnitude_db, frequencies)
            
            logger.info(f"Computed cepstrogram: {cepstral.shape}")
            
            if HAS_VISPY and cepstral.size > 0:
                extent = (self.current_view_range[0][0], self.current_view_range[0][1],
                         0, cepstral.shape[0])  # Quefrency range
                self.cepstrogram_canvas.update_image(cepstral, extent)
                frames = cepstral.shape[1]
                self.statusBar().showMessage(f"Cepstrogram ready ({frames} frames)")
        except Exception as e:
            logger.error(f"Cepstrogram computation error: {e}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"Cepstrogram error: {str(e)}")
    
    def load_fk_data(self, audio_data: np.ndarray):
        """Load F-K transform data."""
        # F-K transform is complex and memory-intensive
        # Disable for now - will be optimized in Phase 5
        logger.info("F-K transform disabled (Phase 5 optimization pending)")
        self.statusBar().showMessage("F-K transform - optimization in progress")
    
    def refresh_current_view(self):
        """Refresh the currently active view."""
        current_index = self.tab_widget.currentIndex()
        self.on_tab_changed(current_index)
    
    def zoom_to_fit(self):
        """Zoom to fit all data."""
        if not self.current_file:
            return
        
        # Reset view range to full audio duration
        duration = self.audio_loader.duration
        sample_rate = self.spectrogram_engine.sample_rate
        
        self.current_view_range = ((0.0, duration), (0.0, sample_rate / 2))
        self.refresh_current_view()
    
    def update_performance_stats(self):
        """Update performance statistics display."""
        # Get GPU memory info
        gpu_mem = self.gpu_memory_manager.get_memory_info()
        
        # Get tile manager stats
        tile_stats = self.tile_manager.get_stats()
        cache_stats = tile_stats.get('cache_stats', {})
        
        # Update status widget with GPU memory
        self.status_widget.update_stats(
            memory_mb=cache_stats.get('disk_usage_mb', 0),
            gpu_mb=gpu_mem.get('used_mb', 0)
        )
        
        # Show tile cache hit rate in status bar
        if cache_stats.get('hit_rate', 0) > 0:
            hit_rate = cache_stats['hit_rate']
            tiles_in_memory = cache_stats.get('memory_tiles', 0)
            tiles_on_disk = cache_stats.get('disk_tiles', 0)
            
            status_msg = (f"Cache: {hit_rate:.1f}% hit rate | "
                         f"{tiles_in_memory} tiles in RAM | "
                         f"{tiles_on_disk} tiles on disk")
            self.statusBar().showMessage(status_msg)
        
        # Show active task progress if any
        active_tasks = self.task_manager.get_active_tasks()
        if active_tasks:
            avg_progress = np.mean([task['progress'] for task in active_tasks])
            self.status_widget.update_stats(progress=avg_progress)
        else:
            self.status_widget.update_stats(progress=1.0)
    
    def closeEvent(self, event):
        """Handle application close."""
        logger.info("Application closing - cleaning up resources...")
        
        # Stop performance timer first
        if hasattr(self, 'perf_timer'):
            self.perf_timer.stop()
        
        # Shutdown task manager and wait for threads to finish
        if hasattr(self, 'task_manager'):
            self.task_manager.shutdown(wait=True)
        
        # Clear tile cache
        if hasattr(self, 'tile_manager'):
            self.tile_manager.clear()
        
        # Clear old cache manager
        if hasattr(self, 'cache_manager'):
            self.cache_manager.clear()
        
        # Cleanup GPU memory
        if hasattr(self, 'gpu_memory_manager'):
            self.gpu_memory_manager.cleanup(aggressive=True)
            logger.info("GPU memory cleaned up")
        
        # Cleanup batched FFT engine
        if hasattr(self, 'spectrogram_engine') and hasattr(self.spectrogram_engine, 'batched_fft_engine'):
            self.spectrogram_engine.batched_fft_engine.cleanup()
        
        logger.info("Application closed cleanly")
        event.accept()

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Audio Visualizer")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Audio Visualization Team")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    return app.exec()