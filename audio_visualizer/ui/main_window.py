"""
Main application window for the audio visualizer.
"""
import sys
import os
import logging
import numpy as np

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                              QWidget, QToolBar, QLabel, QPushButton, QFileDialog, 
                              QMessageBox, QSplitter, QFrame, QSizePolicy)
from PySide6.QtCore import QCoreApplication
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence

try:
    from vispy import scene
    HAS_VISPY = True
except ImportError:
    HAS_VISPY = False

from ..core.data_loader import ChunkedAudioLoader
from ..core.cache_manager import CacheManager
from ..core.task_manager import TaskManager
from ..core.tile_cache import TileCache
from ..core.tile_manager import TileManager as TileMgr
from ..core.gpu_memory_manager import get_gpu_memory_manager
from ..core.file_switch_manager import get_file_switch_manager
from ..core.smart_cache_invalidation import get_smart_cache_invalidator
from ..core.adaptive_spectrogram import (
    get_adaptive_spectrogram_manager, ViewRegion, FFTParameters
)
from ..engines.spectrogram_engine import SpectrogramEngine
from ..rendering.texture_atlas import AtlasRenderer
from ..rendering.shader_renderer import get_shader_renderer

# Import UI components from separate modules
from .vispy_canvas import VisPyCanvas
from .controls_widget import ControlsWidget
from .status_widget import StatusWidget
from .multi_file_manager import PlaylistWidget

logger = logging.getLogger(__name__)


# TiledImageRenderer, VisPyCanvas, StatusWidget, ControlsWidget
# have been moved to separate modules:
# - vispy_canvas.py
# - status_widget.py  
# - controls_widget.py



class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("GPU-Accelerated Audio Visualizer")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        
        # Apply modern glassmorphic theme
        self.apply_modern_theme()
        
        # Core components
        self.cache_manager = CacheManager(max_memory_mb=2048, max_gpu_memory_mb=1024)
        self.task_manager = TaskManager()
        self.audio_loader = ChunkedAudioLoader()
        
        # GPU Memory Management
        self.gpu_memory_manager = get_gpu_memory_manager()
        
        # File Switch Manager for optimized cleanup
        self.file_switch_manager = get_file_switch_manager()
        
        # Smart Cache Invalidation
        self.smart_invalidator = get_smart_cache_invalidator()
        
        # Shader-based Rendering for instant parameter updates
        self.shader_renderer = get_shader_renderer()
        
        # Tile-based caching and rendering
        self.tile_cache = TileCache(max_memory_tiles=100, max_disk_gb=10.0)
        
        # Computation engines (spectrogram only)
        self.spectrogram_engine = SpectrogramEngine(self.cache_manager, self.task_manager)
        
        # Engines dictionary for easy access
        self.engines = {
            'spectrogram': self.spectrogram_engine,
        }
        
        # Adaptive Spectrogram Manager for zoom-based quality optimization
        self.adaptive_manager = get_adaptive_spectrogram_manager(
            sample_rate=self.spectrogram_engine.sample_rate
        )
        
        # Settings state
        self.auto_db_range_enabled = True
        self.current_freq_scale = 'linear'
        
        # Tile manager (coordinates tiles, cache, and atlases)
        self.tile_manager = TileMgr(
            tile_cache=self.tile_cache,
            engines=self.engines
        )
        
        # Atlas renderer for spectrogram view
        self.atlas_renderers = {}
        atlas = self.tile_manager.get_atlas('spectrogram')
        if atlas:
            self.atlas_renderers['spectrogram'] = AtlasRenderer(atlas)
        
        logger.info(f"Initialized {len(self.atlas_renderers)} atlas renderers")
        
        
        # Multi-file state
        self.current_file = None  # Currently active file
        self.file_data = {}  # file_path -> {audio_data, sample_rate, duration, view_range}
        # Default view range (will be updated when file is loaded)
        self.current_view_range = ((0.0, 10.0), (0.0, 22050.0))  # (time, freq)
        
        # Spectrogram cache to avoid recomputation
        self.spectrogram_cache = {
            'time_range': None,
            'freq_range': None,
            'fft_size': None,
            'hop_length': None,
            'data': None,
            'extent': None
        }
        
        # Setup UI
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_zoom_toolbar()
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
        """Setup the main user interface with improved layout and containers."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with NO padding to maximize canvas space
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Controls at the top
        self.controls_widget = ControlsWidget()
        main_layout.addWidget(self.controls_widget)
        
        # Container for visualization - glassmorphic card style
        viz_container = QFrame()
        viz_container.setObjectName("viz_container")
        viz_container.setFrameShape(QFrame.NoFrame)
        viz_layout = QHBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)  # NO MARGINS!
        viz_layout.setSpacing(0)
        
        # Splitter for playlist and canvas
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        viz_layout.addWidget(self.main_splitter)
        
        # Simple playlist on the left (single panel, no tabs)
        self.playlist_widget = PlaylistWidget()
        self.playlist_widget.setMaximumWidth(220)
        self.playlist_widget.setMinimumWidth(180)
        self.connect_playlist_signals()
        self.main_splitter.addWidget(self.playlist_widget)
        
        # Center container for spectrogram canvas
        center_container = QWidget()
        center_layout = QVBoxLayout(center_container)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        self.main_splitter.addWidget(center_container)
        self.main_splitter.setSizes([260, 1200])
        
        if HAS_VISPY:
            spec_container = QFrame()
            spec_container.setFrameShape(QFrame.NoFrame)
            spec_layout = QVBoxLayout(spec_container)
            spec_layout.setContentsMargins(0, 0, 0, 0)
            spec_layout.setSpacing(0)
            
            self.spectrogram_canvas = VisPyCanvas('spectrogram')
            self.spectrogram_canvas.native.setMinimumSize(600, 400)
            spec_layout.addWidget(self.spectrogram_canvas.native)
            center_layout.addWidget(spec_container, 1)
            
            # Connect callbacks for status bar updates
            self.spectrogram_canvas.on_zoom_changed_callback = self.on_zoom_level_changed
            self.spectrogram_canvas.on_cursor_moved_callback = self.on_cursor_moved
            
            # Connect auto-recompute callback for high-quality zoom
            self.spectrogram_canvas.on_auto_recompute_callback = self.on_auto_recompute
            
            # Set measurement mode callback
            self.spectrogram_canvas.set_measurement_callback(self.on_measurement_mode_changed)
        else:
            center_layout.addWidget(QLabel("VisPy not available"))
        main_layout.addWidget(viz_container)
        
        # Connect signals
        self.controls_widget.parameters_changed.connect(self.on_parameters_changed)
        self.controls_widget.colormap_changed.connect(self.on_colormap_changed)
        self.controls_widget.db_range_changed.connect(self.on_db_range_changed)
        self.controls_widget.refresh_requested.connect(self.refresh_current_view)
        self.controls_widget.interpolation_changed.connect(self.on_interpolation_changed)
        
        # No tab changes; visualization is always spectrogram
    
    def connect_playlist_signals(self):
        """Connect signals from playlist widget."""
        self.playlist_widget.file_selected.connect(self.on_file_selected_from_playlist)
        self.playlist_widget.files_dropped.connect(self.on_files_dropped)
    
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
        
        # Export menu
        export_menu = file_menu.addMenu("Export")
        
        export_image_action = QAction("Export Current View as Image...", self)
        export_image_action.setShortcut("Ctrl+E")
        export_image_action.triggered.connect(self.export_current_view_as_image)
        export_menu.addAction(export_image_action)
        
        export_data_action = QAction("Export Data as NPY...", self)
        export_data_action.triggered.connect(self.export_data_as_npy)
        export_menu.addAction(export_data_action)
        
        export_csv_action = QAction("Export Data as CSV...", self)
        export_csv_action.triggered.connect(self.export_data_as_csv)
        export_menu.addAction(export_csv_action)
        
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
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.setShortcut("F1")
        shortcuts_action.triggered.connect(self.show_keyboard_shortcuts)
        help_menu.addAction(shortcuts_action)
    
    def setup_zoom_toolbar(self):
        """Setup compact zoom navigation toolbar."""
        zoom_toolbar = QToolBar("Navigation")
        zoom_toolbar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, zoom_toolbar)
        
        # Reset zoom
        reset_btn = QPushButton("Reset")
        reset_btn.setToolTip("Reset zoom to show all data (R)")
        reset_btn.setFixedWidth(55)
        reset_btn.clicked.connect(self.zoom_to_fit)
        zoom_toolbar.addWidget(reset_btn)
        
        zoom_toolbar.addSeparator()
        
        # Time axis zoom
        time_label = QLabel(" Time: ")
        time_label.setStyleSheet("color: rgba(255,255,255,0.7);")
        zoom_toolbar.addWidget(time_label)
        
        zoom_time_in = QPushButton("+")
        zoom_time_in.setToolTip("Zoom in time (Shift+Scroll)")
        zoom_time_in.setFixedWidth(28)
        zoom_time_in.clicked.connect(lambda: self.zoom_axis('time', 'in'))
        zoom_toolbar.addWidget(zoom_time_in)
        
        zoom_time_out = QPushButton("-")
        zoom_time_out.setToolTip("Zoom out time")
        zoom_time_out.setFixedWidth(28)
        zoom_time_out.clicked.connect(lambda: self.zoom_axis('time', 'out'))
        zoom_toolbar.addWidget(zoom_time_out)
        
        zoom_toolbar.addSeparator()
        
        # Frequency axis zoom
        freq_label = QLabel(" Freq: ")
        freq_label.setStyleSheet("color: rgba(255,255,255,0.7);")
        zoom_toolbar.addWidget(freq_label)
        
        zoom_freq_in = QPushButton("+")
        zoom_freq_in.setToolTip("Zoom in frequency (Ctrl+Scroll)")
        zoom_freq_in.setFixedWidth(28)
        zoom_freq_in.clicked.connect(lambda: self.zoom_axis('freq', 'in'))
        zoom_toolbar.addWidget(zoom_freq_in)
        
        zoom_freq_out = QPushButton("-")
        zoom_freq_out.setToolTip("Zoom out frequency")
        zoom_freq_out.setFixedWidth(28)
        zoom_freq_out.clicked.connect(lambda: self.zoom_axis('freq', 'out'))
        zoom_toolbar.addWidget(zoom_freq_out)
        
        zoom_toolbar.addSeparator()
        
        # Both axes zoom
        both_label = QLabel(" Both: ")
        both_label.setStyleSheet("color: rgba(255,255,255,0.7);")
        zoom_toolbar.addWidget(both_label)
        
        zoom_both_in = QPushButton("+ +")
        zoom_both_in.setToolTip("Zoom in both axes (Scroll)")
        zoom_both_in.setFixedWidth(35)
        zoom_both_in.clicked.connect(lambda: self.zoom_axis('both', 'in'))
        zoom_toolbar.addWidget(zoom_both_in)
        
        zoom_both_out = QPushButton("- -")
        zoom_both_out.setToolTip("Zoom out both axes")
        zoom_both_out.setFixedWidth(35)
        zoom_both_out.clicked.connect(lambda: self.zoom_axis('both', 'out'))
        zoom_toolbar.addWidget(zoom_both_out)
        
        # Add stretch spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        zoom_toolbar.addWidget(spacer)
        
        # Navigation hints
        hints = QLabel("Pan: drag | Zoom: scroll | Time zoom: Shift+scroll | Freq zoom: Ctrl+scroll")
        hints.setStyleSheet("color: rgba(255,255,255,0.4); font-size: 10px; padding-right: 10px;")
        zoom_toolbar.addWidget(hints)
    
    def zoom_axis(self, axis: str, direction: str):
        """Zoom on specified axis.
        
        Args:
            axis: 'time', 'freq', or 'both'
            direction: 'in' or 'out'
        """
        if not HAS_VISPY or not hasattr(self, 'spectrogram_canvas'):
            return
        
        factor = 1.5 if direction == 'in' else 2/3
        
        if axis == 'time':
            scale_factors = [factor, 1.0]
        elif axis == 'freq':
            scale_factors = [1.0, factor]
        else:  # both
            scale_factors = [factor, factor]
        
        self.spectrogram_canvas.zoom_with_center(scale_factors, None)
        logger.debug(f"Zoom {direction} on {axis} axis")
    
    def setup_status_bar(self):
        """Setup the status bar."""
        self.status_widget = StatusWidget()
        self.statusBar().addPermanentWidget(self.status_widget)
        self.statusBar().showMessage("Ready")
    
    def open_audio_file(self):
        """Open file dialog to select audio file."""
        from PySide6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*.*)"
        )
        
        if file_path:
            self.load_audio_file(file_path)
    
    def load_audio_file(self, file_path: str):
        """Load an audio file for analysis with optimized cleanup."""
        try:
            self.statusBar().showMessage("Loading audio file...")
            
            # PERFORMANCE: Aggressive cleanup before loading new file
            if hasattr(self, 'current_file') and self.current_file is not None:
                logger.info(f"Switching from {os.path.basename(self.current_file)} to {os.path.basename(file_path)}")
                self.file_switch_manager.cleanup_for_new_file(
                    cache_manager=self.cache_manager,
                    gpu_memory_manager=self.gpu_memory_manager,
                    memory_pools=getattr(self.spectrogram_engine, 'memory_optimizer', None),
                    tile_cache=self.tile_cache,
                    engines=self.engines
                )
            
            sample_rate, duration = self.audio_loader.load_file(file_path)
            
            self.current_file = file_path
            
            # IMPORTANT: Max frequency is Nyquist = sample_rate / 2
            nyquist_freq = sample_rate / 2
            self.current_view_range = ((0.0, duration), (0.0, nyquist_freq))
            
            # Update spectrogram engine parameters with correct sample rate
            self.spectrogram_engine.set_parameters(sample_rate=sample_rate)
            
            # Update adaptive manager with correct sample rate
            self.adaptive_manager.sample_rate = sample_rate
            self.adaptive_manager.clear_cache()
            
            # Reset zoom detector for new file
            if HAS_VISPY and hasattr(self, 'spectrogram_canvas'):
                self.spectrogram_canvas.zoom_detector.reset()
            
            logger.info(f"Loaded: {os.path.basename(file_path)} - {duration:.1f}s, {sample_rate}Hz, max freq {nyquist_freq:.0f}Hz")
            
            # Update playlist widget with file info
            if hasattr(self, 'playlist_widget'):
                self.playlist_widget.add_file_path(file_path)
                self.playlist_widget.update_file_info(
                    file_path, duration=duration, sample_rate=sample_rate, is_loaded=True
                )
            
            self.statusBar().showMessage(
                f"Loaded: {os.path.basename(file_path)} "
                f"({duration:.1f}s, {sample_rate}Hz, max {nyquist_freq:.0f}Hz)"
            )
            
            # Refresh current view
            self.refresh_current_view()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load audio file:\n{str(e)}")
            self.statusBar().showMessage("Ready")
    
    
    def on_parameters_changed(self, params: dict):
        """Handle parameter changes with smart cache invalidation."""
        
        # Clear spectrogram cache
        self.spectrogram_cache = {
            'time_range': None, 'freq_range': None,
            'fft_size': None, 'hop_length': None,
            'data': None, 'extent': None
        }
        
        # Clean up old data before computing new
        if HAS_VISPY and hasattr(self, 'spectrogram_canvas'):
            self.spectrogram_canvas.raw_display_data = None
            self.spectrogram_canvas.normalized_display_data = None
        
        # Release GPU memory
        self.cleanup_gpu_memory()
        
        # Get current parameters before updating
        old_params = {
            'fft_size': getattr(self.spectrogram_engine, 'fft_size', 4096),
            'hop_length': getattr(self.spectrogram_engine, 'hop_length', 512),
            'window': getattr(self.spectrogram_engine, 'window', 'blackman'),
            'sample_rate': getattr(self.spectrogram_engine, 'sample_rate', 44100)
        }
        
        # Update engine parameters
        self.spectrogram_engine.set_parameters(**params)
        
        # Get new parameters after updating  
        new_params = dict(old_params)
        new_params.update(params)
        
        # PERFORMANCE: Smart cache invalidation - only clear what's affected
        invalidation_actions = self.smart_invalidator.invalidate_affected_caches(
            old_params=old_params,
            new_params=new_params,
            cache_manager=self.cache_manager,
            tile_cache=self.tile_cache,
            engines=self.engines
        )
        
        # Clear tiled renderer visuals for affected views
        for view_type in invalidation_actions.keys():
            if view_type == 'spectrogram' and hasattr(self.spectrogram_canvas, 'tiled_renderer'):
                self.spectrogram_canvas.tiled_renderer.clear_tiles()
        
        # Log invalidation summary
        if invalidation_actions:
            affected_views = list(invalidation_actions.keys())
            logger.info(f"Smart invalidation completed: {len(affected_views)} views affected")
        else:
            logger.info("No cache invalidation needed - all caches preserved")
        
        self.refresh_current_view()
    
    def on_colormap_changed(self, colormap: str):
        """Handle colormap changes."""
        logger.info(f"Colormap changed to: {colormap}")
        
        try:
            # Update all canvas image visuals with new colormap
            if HAS_VISPY:
                if hasattr(self, 'spectrogram_canvas') and self.spectrogram_canvas.image_visual:
                    self.spectrogram_canvas.image_visual.cmap = colormap
                    logger.debug(f"Updated spectrogram colormap to {colormap}")
                
                # Trigger a visual update
                self.update_displays()
                
        except Exception as e:
            logger.error(f"Error updating colormap: {e}")
    
    def on_interpolation_changed(self, interpolation: str):
        """Handle interpolation changes."""
        logger.info(f"Interpolation changed to: {interpolation}")
        
        try:
            if HAS_VISPY and hasattr(self, 'spectrogram_canvas'):
                self.spectrogram_canvas.set_interpolation(interpolation)
        except Exception as e:
            logger.error(f"Error updating interpolation: {e}")
    
    def update_displays(self):
        """Force update of all displays."""
        try:
            if HAS_VISPY and hasattr(self, 'spectrogram_canvas'):
                self.spectrogram_canvas.update()
        except Exception as e:
            logger.debug(f"Error updating displays: {e}")
    
    def on_zoom_level_changed(self, time_zoom: float, freq_zoom: float):
        """Handle zoom level change - update status bar."""
        if hasattr(self, 'status_widget'):
            self.status_widget.update_zoom(time_zoom, freq_zoom)
    
    def on_cursor_moved(self, time_sec: float, freq_hz: float):
        """Handle cursor movement - update status bar."""
        if hasattr(self, 'status_widget'):
            self.status_widget.update_cursor(time_sec, freq_hz)
    
    def on_measurement_mode_changed(self, is_on: bool):
        """Handle measurement mode toggle - update status bar."""
        if hasattr(self, 'status_widget'):
            self.status_widget.update_measurement_mode(is_on)
    
    def on_auto_recompute(self, visible_region):
        """Auto-recompute spectrogram at current zoom level for high quality."""
        if not self.current_file:
            return
        
        # Check if zoom level is high enough to benefit from recompute
        time_zoom, freq_zoom = self.spectrogram_canvas.get_zoom_level()
        if max(time_zoom, freq_zoom) < 2.0:
            # Not zoomed in enough, skip recompute
            return
        
        # Store original bounds before recompute
        original_bounds = self.spectrogram_canvas.data_bounds
        
        # Update view range and recompute
        self.current_view_range = (
            (visible_region.time_start, visible_region.time_end),
            (visible_region.freq_start, visible_region.freq_end)
        )
        
        logger.info(f"Auto-recompute at zoom {max(time_zoom, freq_zoom):.1f}x")
        
        # Set flag to preserve view during this recompute
        self._preserve_view_on_update = True
        self.load_view_data('spectrogram')
        self._preserve_view_on_update = False
        
        # Restore original bounds so user can still navigate the full file
        if original_bounds:
            self.spectrogram_canvas.data_bounds = original_bounds
    
    def on_db_range_changed(self, db_min: float, db_max: float):
        """Handle dB range changes."""
        logger.info(f"dB range changed to: [{db_min:.1f}, {db_max:.1f}] dB")
        
        try:
            # Update all canvas image visuals with new dB range for color limits
            if HAS_VISPY:
                # Convert dB range to normalized range (spectrogram data is normalized 0-1)
                # This affects the color mapping range
                if hasattr(self, 'spectrogram_canvas') and self.spectrogram_canvas.image_visual:
                    # Store dB range for use during data normalization
                    self.spectrogram_canvas.db_range = (db_min, db_max)
                    self.spectrogram_canvas.update_dynamic_clim()
                    logger.debug(f"Updated spectrogram dB range to [{db_min:.1f}, {db_max:.1f}]")
                
                # Refresh current view to apply new dB range
                self.refresh_current_view()
                
        except Exception as e:
            logger.error(f"Error updating dB range: {e}")
    
    def update_shader_colormap(self, colormap: str):
        """Update shader colormap in real-time without recomputation."""
        try:
            if self.shader_renderer.is_enabled():
                changed = self.shader_renderer.set_colormap(colormap)
                if changed:
                    # Trigger redraw if shader is active
                    self.update_displays()
                    logger.debug(f"Shader colormap updated instantly: {colormap}")
            else:
                # Fallback to traditional colormap change
                logger.debug(f"Shader not available, using traditional colormap: {colormap}")
        except Exception as e:
            logger.error(f"Error updating shader colormap: {e}")
    
    def update_shader_db_range(self, db_min: float, db_max: float):
        """Update shader dB range in real-time without recomputation."""
        try:
            if self.shader_renderer.is_enabled():
                changed = self.shader_renderer.set_db_range(db_min, db_max)
                if changed:
                    # Trigger redraw if shader is active
                    self.update_displays()
                    logger.debug(f"Shader dB range updated instantly: [{db_min:.1f}, {db_max:.1f}]")
            else:
                # Fallback to traditional dB range change
                logger.debug(f"Shader not available, using traditional dB range: [{db_min:.1f}, {db_max:.1f}]")
        except Exception as e:
            logger.error(f"Error updating shader dB range: {e}")
    
    # ========== Adaptive Spectrogram Methods ==========
    
    def on_zoom_recompute_needed(self, view: ViewRegion):
        """Handle zoom-triggered spectrogram recomputation.
        
        Called when the zoom level changed significantly enough to warrant
        recomputing the spectrogram with optimized FFT parameters.
        """
        if not self.current_file:
            return
        
        # Check if adaptive FFT is enabled
        if not self.controls_widget.is_adaptive_fft_enabled():
            logger.debug("Adaptive FFT disabled, skipping zoom recompute")
            return
        
        try:
            # Get optimal parameters for this view
            window_type = self.controls_widget.get_current_window()
            optimal_params = self.adaptive_manager.calculate_optimal_params(view, window_type)
            
            # Check if we should actually recompute
            if not self.adaptive_manager.should_recompute(optimal_params):
                logger.debug("Parameters didn't change significantly, skipping recompute")
                return
            
            logger.info(f"Adaptive recompute: fft={optimal_params.fft_size}, hop={optimal_params.hop_length}")
            
            # Update current parameters
            self.adaptive_manager.current_params = optimal_params
            
            # Recompute spectrogram with new parameters
            self.compute_adaptive_spectrogram(view, optimal_params)
            
        except Exception as e:
            logger.error(f"Error in zoom recompute: {e}")
    
    def compute_adaptive_spectrogram(self, view: ViewRegion, params: FFTParameters):
        """Compute spectrogram with adaptive parameters for the visible region.
        
        Args:
            view: Current visible region
            params: Optimal FFT parameters
        """
        if not self.current_file:
            return
        
        try:
            self.statusBar().showMessage(f"Recomputing spectrogram (FFT={params.fft_size}, hop={params.hop_length})...")
            
            # Get audio data for the visible time range (with some padding)
            padding = (view.time_end - view.time_start) * 0.1  # 10% padding
            time_start = max(0, view.time_start - padding)
            time_end = min(self.audio_loader.duration, view.time_end + padding)
            
            sample_rate = self.spectrogram_engine.sample_rate
            start_sample = int(time_start * sample_rate)
            end_sample = int(time_end * sample_rate)
            
            num_samples = min(end_sample - start_sample, self.audio_loader.total_samples - start_sample)
            audio_data = self.audio_loader.get_chunk(start_sample, num_samples)
            
            if audio_data is None or len(audio_data) < params.fft_size:
                logger.warning("Insufficient audio data for adaptive computation")
                return
            
            # Compute STFT with adaptive parameters
            magnitude_db, times = self.spectrogram_engine.batched_fft_engine.compute_stft_batched(
                audio_data,
                fft_size=params.fft_size,
                hop_length=params.hop_length,
                window=params.window_type,
                sample_rate=sample_rate,
                use_gpu=self.spectrogram_engine.use_gpu
            )
            
            frequencies = np.fft.rfftfreq(params.fft_size, 1.0 / sample_rate).astype(np.float32)
            
            # Cache the result
            self.adaptive_manager.cache_spectrogram(view, params, magnitude_db, frequencies, times)
            
            # Update display
            if HAS_VISPY and magnitude_db.size > 0:
                # Calculate extent
                frame_duration = params.hop_length / sample_rate
                time_min = time_start
                time_max = time_start + len(audio_data) / sample_rate
                freq_min = 0.0
                freq_max = sample_rate / 2.0
                
                extent = (time_min, time_max, freq_min, freq_max)
                
                self.spectrogram_canvas.update_image(magnitude_db, extent)
                
                self.statusBar().showMessage(
                    f"Adaptive spectrogram: {magnitude_db.shape[1]} frames, "
                    f"FFT={params.fft_size}, hop={params.hop_length}"
                )
                
                logger.info(f"Adaptive spectrogram computed: {magnitude_db.shape}")
        
        except Exception as e:
            logger.error(f"Error computing adaptive spectrogram: {e}")
            self.statusBar().showMessage(f"Computation error: {str(e)}")
    
    def set_quality_mode(self, mode: str):
        """Set the quality mode for adaptive spectrogram computation.
        
        Args:
            mode: 'fast', 'balanced', or 'quality'
        """
        self.adaptive_manager.set_quality_mode(mode)
        logger.info(f"Quality mode set to: {mode}")
        
        # Trigger recompute if we have data
        if self.current_file:
            self.refresh_current_view()
    
    def set_interpolation(self, mode: str):
        """Set image interpolation mode.
        
        Args:
            mode: 'nearest', 'bilinear', or 'bicubic'
        """
        if HAS_VISPY and hasattr(self, 'spectrogram_canvas'):
            self.spectrogram_canvas.set_interpolation(mode)
            logger.info(f"Interpolation set to: {mode}")
    
    def set_freq_scale(self, scale: str):
        """Set frequency scale mode.
        
        Args:
            scale: 'linear', 'log', or 'mel'
        """
        self.current_freq_scale = scale
        logger.info(f"Frequency scale set to: {scale}")
        
        # TODO: Implement log and mel scale transformations
        # For now, just store the setting - full implementation requires
        # transforming the frequency axis display
        
        if self.current_file:
            self.refresh_current_view()
    
    def set_auto_db_range(self, auto: bool):
        """Set automatic dB range adjustment.
        
        Args:
            auto: If True, automatically adjust dB range based on visible data
        """
        self.auto_db_range_enabled = auto
        logger.info(f"Auto dB range: {'enabled' if auto else 'disabled'}")
        
        if HAS_VISPY and hasattr(self, 'spectrogram_canvas'):
            if auto:
                # Trigger dynamic normalization
                self.spectrogram_canvas.update_dynamic_clim()
            # else: manual mode uses the slider values directly
    
    def on_tab_changed(self, index: int):
        """Handle tab changes - lazy loading trigger."""
        if not self.current_file:
            return
        if index == 0:
            self.load_view_data('spectrogram')
    
    def load_view_data(self, view_type: str):
        """Load data for specified view type using tile system."""
        if not self.current_file:
            return
        
        self.statusBar().showMessage(f"Loading {view_type} tiles...")
        
        # Get current view range
        time_range = self.current_view_range[0]
        freq_range = self.current_view_range[1]
        
        logger.info(f"Loading {view_type} for time range {time_range}, freq range {freq_range}")
        
        try:
            # Try tile-based rendering first
            if self.load_view_data_tiled(view_type, time_range, freq_range):
                return
            
            # Fallback to direct computation if tile system fails
            logger.warning(f"Tile system failed for {view_type}, falling back to direct computation")
            self.load_view_data_direct(view_type, time_range)
                
        except Exception as e:
            logger.error(f"Error loading {view_type}: {e}")
            self.statusBar().showMessage(f"Error: {str(e)}")
    
    def load_view_data_tiled(self, view_type: str, time_range: tuple, freq_range: tuple) -> bool:
        """Load data using tile system with on-demand tile generation. Returns True if successful."""
        # DISABLED: Tile system has bugs - using direct computation with downsampling instead
        # TODO: Fix tile system properly in future
        logger.debug("Tile system disabled - using direct computation")
        return False
        
        try:
            # For large files, use tile-based rendering to avoid OpenGL texture limits
            # Check if we need tiling (file duration > 6 minutes or width > OpenGL limit)
            duration = time_range[1] - time_range[0]
            sample_rate = self.spectrogram_engine.sample_rate
            hop_length = self.spectrogram_engine.hop_length
            
            # Calculate expected width
            n_samples = int(duration * sample_rate)
            expected_width = 1 + (n_samples - self.spectrogram_engine.fft_size) // hop_length
            
            # OpenGL texture size limit
            max_texture_size = 16384
            
            if expected_width <= max_texture_size:
                # File is small enough for direct rendering
                logger.debug(f"File width ({expected_width}) fits in single texture, using direct computation")
                return False
            
            # File is too large - use tile-based rendering
            logger.info(f"File width ({expected_width}) exceeds limit ({max_texture_size}), using tile system")
            
            # Compute tiles on-demand for this region
            zoom_level = self.estimate_zoom_level(time_range, freq_range)
            
            # Request tiles to be computed if they don't exist
            if not self.ensure_tiles_exist(view_type, time_range, freq_range, zoom_level):
                logger.warning(f"Failed to generate tiles for {view_type}")
                return False
            
            # Update visible region
            self.tile_manager.update_visible_region(view_type, time_range, freq_range, zoom_level)
            
            # Get tiles for rendering
            visible_tiles = self.tile_manager.get_visible_tiles(view_type, time_range, freq_range, zoom_level)
            
            if not visible_tiles:
                logger.warning(f"No visible tiles after generation for {view_type}")
                return False
            
            # Render tiles to display
            if self.render_tiles_to_display(view_type, visible_tiles, time_range, freq_range):
                self.statusBar().showMessage(f"{view_type.title()} rendered from {len(visible_tiles)} tiles")
                return True
            
            # Fallback: if tile rendering failed, try direct rendering with the cached tiles
            logger.warning("Tile rendering failed, trying direct tile cache rendering")
            tile_data_list = []
            tile_time_duration = 60.0
            current_time = time_range[0]
            while current_time < time_range[1]:
                tile_end = min(current_time + tile_time_duration, time_range[1])
                cached = self.tile_cache.get(view_type, (current_time, tile_end), freq_range, 0)
                if cached is not None:
                    tile_data_list.append(cached)
                current_time = tile_end
            
            if tile_data_list and hasattr(self, 'spectrogram_canvas'):
                success = self.spectrogram_canvas.tiled_renderer.render_tiles(
                    tile_data_list, time_range, freq_range, tile_time_duration
                )
                if success:
                    self.spectrogram_canvas.set_data_bounds(time_range[0], time_range[1], freq_range[0], freq_range[1])
                    self.spectrogram_canvas.update()
                    self.statusBar().showMessage(f"Rendered {len(tile_data_list)} tiles directly")
                return True
            
            return False
            
        except Exception as e:
            logger.exception(f"Tile-based loading failed for {view_type}: {e}")
            return False
    
    def cleanup_gpu_memory(self):
        """Release GPU memory to prevent memory leaks."""
        try:
            import cupy as cp
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
            logger.debug("GPU memory released")
        except Exception:
            pass  # CuPy not available or error
    
    def load_view_data_direct(self, view_type: str, time_range: tuple):
        """Fallback to direct computation."""
        logger.info(f"Using direct computation for {view_type}")
        
        # Clean up GPU memory before new computation
        self.cleanup_gpu_memory()
        
        # Get audio data for current time range
        start_sample = int(time_range[0] * self.spectrogram_engine.sample_rate)
        end_sample = int(time_range[1] * self.spectrogram_engine.sample_rate)
        
        # Clamp to available samples
        total_samples = self.audio_loader.total_samples
        num_samples = min(end_sample - start_sample, total_samples - start_sample)
        
        audio_data = self.audio_loader.get_chunk(start_sample, num_samples)
        
        # Use original direct computation methods
        if view_type == 'spectrogram':
            self.load_spectrogram_data(audio_data, time_range)
        else:
            logger.warning(f"Unsupported view type requested: {view_type}")
    
    def ensure_tiles_exist(self, view_type: str, time_range: tuple, freq_range: tuple, zoom_level: float) -> bool:
        """Ensure tiles exist for the given region, computing them if necessary."""
        try:
            # Define tile size (in seconds for time, Hz for frequency)
            tile_time_duration = 60.0  # 60 seconds per tile
            
            # Calculate how many tiles we need to cover this region
            time_start, time_end = time_range
            freq_start, freq_end = freq_range
            
            # Generate tile grid
            # For now: each tile covers full frequency spectrum, divided only in time
            # TODO: Add frequency bands for very large spectra (>8000 Hz)
            tiles_needed = []
            current_time = time_start
            while current_time < time_end:
                tile_time_end = min(current_time + tile_time_duration, time_end)
                
                tiles_needed.append({
                    'time_range': (current_time, tile_time_end),
                    'freq_range': (freq_start, freq_end)  # Full frequency spectrum per tile
                })
                
                current_time = tile_time_end
            
            logger.info(f"Need {len(tiles_needed)} tiles to cover region {time_range} Ã— {freq_range}")
            
            # Get audio data for computation
            sample_rate = self.spectrogram_engine.sample_rate
            start_sample = int(time_start * sample_rate)
            end_sample = int(time_end * sample_rate)
            end_sample = min(end_sample, self.audio_loader.total_samples)
            
            audio_data = self.audio_loader.get_chunk(start_sample, end_sample - start_sample)
            
            if audio_data is None or len(audio_data) == 0:
                logger.error("Failed to load audio data for tile generation")
                return False
            
            # Compute each tile
            computed_tiles = []
            for tile_info in tiles_needed:
                tile_time_range = tile_info['time_range']
                tile_freq_range = tile_info['freq_range']
                
                # Check if tile already exists in cache
                cached_tile = self.tile_cache.get(
                    view_type, tile_time_range, tile_freq_range, resolution_level=0
                )
                
                if cached_tile is not None:
                    logger.debug(f"Tile already cached: {tile_time_range}")
                    computed_tiles.append(cached_tile)
                    continue
                
                # Compute tile
                logger.debug(f"Computing tile: time={tile_time_range}, freq={tile_freq_range}")
                
                # Extract audio chunk for this tile (relative to our audio_data)
                tile_start_sample = int((tile_time_range[0] - time_start) * sample_rate)
                tile_end_sample = int((tile_time_range[1] - time_start) * sample_rate)
                tile_end_sample = min(tile_end_sample, len(audio_data))
                
                if tile_start_sample >= len(audio_data):
                    logger.warning(f"Tile start beyond audio data, skipping")
                    continue
                
                audio_chunk = audio_data[tile_start_sample:tile_end_sample]
                
                # Compute based on view type
                engine = self.engines.get(view_type)
                if not engine:
                    logger.error(f"No engine for {view_type}")
                    return False
                
                if view_type != 'spectrogram':
                    logger.warning(f"Unsupported view type for tiling: {view_type}")
                    return False
                tile_data = self._compute_spectrogram_tile_data(audio_chunk, tile_freq_range)
                
                if tile_data is None or tile_data.size == 0:
                    logger.warning(f"Empty tile data for {tile_time_range}")
                    continue
                
                # Cache the computed tile
                self.tile_cache.put(
                    view_type, tile_time_range, tile_freq_range, tile_data, resolution_level=0
                )
                
                computed_tiles.append(tile_data)
            
            logger.info(f"Successfully ensured {len(computed_tiles)} tiles exist")
            return len(computed_tiles) > 0
            
        except Exception as e:
            logger.exception(f"Error ensuring tiles exist: {e}")
            return False
    
    def _compute_spectrogram_tile_data(self, audio_chunk: np.ndarray, freq_range: tuple) -> np.ndarray:
        """Compute spectrogram data for a tile."""
        try:
            # Use batched FFT engine
            magnitude_db, frequencies, times = self.spectrogram_engine.compute_stft_batched(audio_chunk)
            
            # Return full spectrum - let the camera/transform handle the visible frequency range
            # Filtering here causes display issues because the data shape doesn't match the extent
            return magnitude_db.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error computing spectrogram tile: {e}")
            return None
    
    
    def render_tiles_to_display(self, view_type: str, tiles: list, time_range: tuple, freq_range: tuple) -> bool:
        """Render tiles to the display using TiledImageRenderer."""
        try:
            if not tiles:
                return False
            
            # Get tile data from cache
            tile_data_list = []
            for tile in tiles:
                if isinstance(tile, np.ndarray):
                    tile_data_list.append(tile)
                elif hasattr(tile, 'data'):
                    tile_data_list.append(tile.data)
            
            if not tile_data_list:
                # Try to get from tile cache directly
                tile_time_duration = 60.0  # seconds per tile
                current_time = time_range[0]
                while current_time < time_range[1]:
                    tile_end = min(current_time + tile_time_duration, time_range[1])
                    cached = self.tile_cache.get(view_type, (current_time, tile_end), freq_range, 0)
                    if cached is not None:
                        tile_data_list.append(cached)
                    current_time = tile_end
            
            if not tile_data_list:
                logger.warning("No tile data available for rendering")
                return False
            
            # Use TiledImageRenderer on the canvas
            if view_type == 'spectrogram' and hasattr(self, 'spectrogram_canvas'):
                canvas = self.spectrogram_canvas
                tile_time_duration = 60.0
                
                success = canvas.tiled_renderer.render_tiles(
                    tile_data_list, time_range, freq_range, tile_time_duration
                )
                
                if success:
                    # Update canvas data bounds
                    canvas.set_data_bounds(time_range[0], time_range[1], freq_range[0], freq_range[1])
                    canvas.view.camera.rect = (time_range[0], freq_range[0],
                                               time_range[1] - time_range[0], freq_range[1] - freq_range[0])
                    canvas.update()
                    return True
            
            return False
        
        except Exception as e:
            logger.exception(f"Error rendering tiles to display: {e}")
            return False
    
    def render_atlas_to_display(self, view_type: str, time_range: tuple, freq_range: tuple) -> bool:
        """Render atlas data directly to the display."""
        try:
            # Get atlas and renderer for this view type
            atlas_renderer = self.atlas_renderers.get(view_type)
            if not atlas_renderer:
                logger.warning(f"No atlas renderer for view type: {view_type}")
                return False
            
            atlas = atlas_renderer.atlas
            
            # Get canvas for this view type
            canvas = getattr(self, f'{view_type}_canvas', None)
            if not canvas or not hasattr(canvas, 'image_visual'):
                logger.warning(f"No canvas or image visual for view type: {view_type}")
                return False
            
            # Check if atlas has any data
            atlas_stats = atlas.get_stats()
            if atlas_stats['slots_used'] == 0:
                logger.debug(f"No tiles loaded in {view_type} atlas, skipping render")
                return False
            
            # Update the display with atlas data
            atlas_renderer.update_display(canvas.image_visual)
            
            # Update world coordinates to match the atlas data
            # For now, use the provided time/freq range
            canvas.view.camera.rect = (
                time_range[0], freq_range[0],
                time_range[1] - time_range[0], freq_range[1] - freq_range[0]
            )
            
            logger.debug(f"Atlas rendering successful for {view_type}: {atlas_stats['slots_used']} tiles displayed")
            return True
            
        except Exception as e:
            logger.error(f"Error rendering atlas to display for {view_type}: {e}")
            return False
    
    def update_spectrogram_display(self, data: np.ndarray, time_range: tuple, freq_range: tuple):
        """Update spectrogram display with pre-computed data."""
        if not HAS_VISPY or not hasattr(self, 'spectrogram_canvas'):
            return
        
        # Normalize data for display
        if data.min() < data.max():
            display_data = (data - data.min()) / (data.max() - data.min())
        else:
            display_data = np.zeros_like(data)
        
        # Calculate proper extent based on actual data
        # Data shape is (freq_bins, time_frames)
        # Frequencies go from 0 to sample_rate/2
        sample_rate = self.spectrogram_engine.sample_rate
        extent = (time_range[0], time_range[1], 0, sample_rate / 2)
        
        # Update image with extent
        self.spectrogram_canvas.update_image(display_data.astype(np.float32), extent)
        
        # Reset camera to show full data
        self.spectrogram_canvas.view.camera.set_range(
            x=(time_range[0], time_range[1]),
            y=(0, sample_rate / 2)
        )
        self.spectrogram_canvas.update()
    
    def estimate_zoom_level(self, time_range: tuple, freq_range: tuple) -> float:
        """Estimate zoom level based on visible range."""
        if not self.current_file:
            return 1.0
        
        # Get total duration and sample rate
        total_duration = self.audio_loader.duration
        sample_rate = self.spectrogram_engine.sample_rate
        
        # Calculate what fraction of total range is visible
        time_span = time_range[1] - time_range[0]
        freq_span = freq_range[1] - freq_range[0]
        
        time_zoom = total_duration / time_span if time_span > 0 else 1.0
        freq_zoom = (sample_rate / 2) / freq_span if freq_span > 0 else 1.0
        
        # Use average zoom level
        zoom_level = (time_zoom + freq_zoom) / 2
        logger.debug(f"Estimated zoom level: {zoom_level:.2f} (time: {time_zoom:.2f}, freq: {freq_zoom:.2f})")
        
        return max(1.0, zoom_level)
    
    def request_tiles_for_region(self, view_type: str, time_range: tuple, freq_range: tuple, zoom_level: float):
        """Request tiles for the specified region."""
        try:
            # Submit tile requests to tile manager
            logger.info(f"Requesting tiles for {view_type}: time={time_range}, freq={freq_range}, zoom={zoom_level:.2f}")
            
            # The tile manager should handle this automatically when we call update_visible_region
            # This is a placeholder for more sophisticated tile request logic
            
        except Exception as e:
            logger.error(f"Error requesting tiles for {view_type}: {e}")
    
    def update_atlas_display_improved(self, view_type: str, time_range: tuple, freq_range: tuple) -> bool:
        """Update display with available atlas data."""
        try:
            atlas = self.tile_manager.atlases.get(view_type)
            if not atlas:
                return False
            
            # Get atlas statistics to see if we have useful data
            stats = atlas.stats
            if stats['slots_used'] == 0:
                logger.debug(f"No tiles loaded in {view_type} atlas")
                return False
            
            # For now, fall back to direct computation since the atlas rendering
            # integration needs more work. This ensures we show data instead of empty view.
            logger.info(f"Atlas has {stats['slots_used']} tiles, but falling back to direct computation")
            return False
            
        except Exception as e:
            logger.error(f"Error updating atlas display for {view_type}: {e}")
            return False
    
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
    
    def load_spectrogram_data(self, audio_data: np.ndarray, view_time_range: tuple = None):
        """Load spectrogram data using batched FFT - NO DOWNSAMPLING (use tile system instead)."""
        try:
            # Check if cached spectrogram covers this region
            # Skip cache during auto-recompute (we want higher resolution)
            preserve_view = getattr(self, '_preserve_view_on_update', False)
            if not preserve_view:
                cache = self.spectrogram_cache
                if (cache['data'] is not None and 
                    cache['time_range'] is not None and
                    view_time_range is not None):
                    
                    cached_start, cached_end = cache['time_range']
                    req_start, req_end = view_time_range
                    
                    # Check if cached region EXACTLY matches (not just contains)
                    time_match = abs(cached_start - req_start) < 0.1 and abs(cached_end - req_end) < 0.1
                    if (time_match and cache['fft_size'] == self.spectrogram_engine.fft_size):
                        
                        logger.info(f"Using cached spectrogram (exact match {req_start:.1f}-{req_end:.1f}s)")
                        self.spectrogram_canvas.update_image(cache['data'], cache['extent'])
                        return
            
            # Show progress bar
            self.status_widget.show_progress("Computing spectrogram")
            self.status_widget.update_progress(10)
            QApplication.processEvents()  # Update UI
            
            # Compute STFT using optimized batched FFT at FULL RESOLUTION
            base_fft = self.spectrogram_engine.fft_size
            base_hop = self.spectrogram_engine.hop_length
            sample_rate = float(getattr(self.spectrogram_engine, "sample_rate", 44100))
            
            # Adaptive FFT size based on frequency zoom level
            # When zoomed in on frequency, use larger FFT for better frequency resolution
            if view_time_range and hasattr(self, 'spectrogram_canvas'):
                time_zoom, freq_zoom = self.spectrogram_canvas.get_zoom_level()
                
                # Scale FFT size based on frequency zoom (up to 4x)
                if freq_zoom > 2.0:
                    fft_scale = min(4, int(freq_zoom / 2))
                    adaptive_fft = min(16384, base_fft * fft_scale)
                    if adaptive_fft != base_fft:
                        logger.info(f"Adaptive FFT: {base_fft} -> {adaptive_fft} (freq zoom {freq_zoom:.1f}x)")
                        base_fft = adaptive_fft
            
            # LOD: Calculate optimal hop length based on:
            # 1. Canvas width (no need for more frames than pixels)
            # 2. OpenGL texture limit (16384)
            # 3. User's base hop setting (minimum quality)
            
            n_samples = len(audio_data)
            max_texture_size = 16384
            
            if view_time_range:
                time_span = max(view_time_range[1] - view_time_range[0], 1e-6)
            else:
                time_span = n_samples / sample_rate if sample_rate > 0 else 1.0
            
            canvas_width = max(400, getattr(self.spectrogram_canvas.native, "width", lambda: 800)())
            if callable(canvas_width):
                canvas_width = max(400, canvas_width())
            
            # Calculate the minimum hop needed to exceed OpenGL limit
            min_hop_for_texture = max(1, n_samples // (max_texture_size - 100))
            
            # For quality: use user's base_hop as maximum (highest quality)
            # But also compute what hop would give us 4x canvas resolution
            target_frames = canvas_width * 4  # 4x oversampling for excellent quality
            hop_for_canvas = max(1, n_samples // target_frames)
            
            # Use the SMALLER hop (higher resolution) but respect texture limits
            # Priority: user quality setting > canvas-based > texture limit
            effective_hop = max(
                min_hop_for_texture,  # Must not exceed OpenGL limit
                min(base_hop, hop_for_canvas)  # Use best quality that makes sense
            )
            
            expected_frames = 1 + (n_samples - base_fft) // effective_hop
            logger.info(f"LOD: samples={n_samples}, time_span={time_span:.2f}s, "
                       f"canvas={canvas_width}, hop={effective_hop} -> {expected_frames} frames")
            
            # Update progress before FFT
            self.status_widget.update_progress(30)
            QApplication.processEvents()
            
            # Use batched FFT engine directly with adaptive hop
            magnitude_db, times = self.spectrogram_engine.batched_fft_engine.compute_stft_batched(
                audio_data,
                fft_size=base_fft,
                hop_length=effective_hop,
                window=self.spectrogram_engine.window_type,
                sample_rate=sample_rate,
                use_gpu=self.spectrogram_engine.use_gpu
            )
            
            # Update progress after FFT
            self.status_widget.update_progress(70)
            QApplication.processEvents()
            
            frequencies = np.fft.rfftfreq(base_fft, 1.0 / sample_rate).astype(np.float32)
            
            logger.info(f"Computed spectrogram: {magnitude_db.shape} (no downsampling)")
            
            if HAS_VISPY and magnitude_db.size > 0:
                frame_duration = effective_hop / sample_rate if sample_rate > 0 else 0.0
                
                # Determine frequency range from STFT output
                if frequencies is not None and len(frequencies) > 0:
                    freq_min = float(np.nanmin(frequencies))
                    freq_max = float(np.nanmax(frequencies))
                    if not np.isfinite(freq_min):
                        freq_min = 0.0
                    if not np.isfinite(freq_max):
                        freq_max = sample_rate / 2.0
                else:
                    freq_min = 0.0
                    freq_max = sample_rate / 2.0
                
                # Default time range before refinement
                if view_time_range:
                    time_offset = view_time_range[0]
                else:
                    time_offset = 0.0
                
                if sample_rate > 0:
                    time_min = time_offset
                    time_max = time_offset + len(audio_data) / sample_rate
                else:
                    time_min = time_offset
                    time_max = time_offset + float(magnitude_db.shape[1])
                
                # Check if result exceeds OpenGL texture limits
                max_texture_size = 16384
                
                if magnitude_db.shape[1] > max_texture_size:
                    # Data exceeds OpenGL limit - need to downsample for display
                    # (This should rarely happen since tile system should catch large files first)
                    logger.warning(f"Data width ({magnitude_db.shape[1]}) exceeds OpenGL limit ({max_texture_size})")
                    
                    downsample_factor = int(np.ceil(magnitude_db.shape[1] / max_texture_size))
                    logger.warning(f"Downsampling by {downsample_factor}x in time to fit texture")
                    
                    # Downsample in time axis
                    magnitude_db = magnitude_db[:, ::downsample_factor]
                    times = times[::downsample_factor]
                    
                    logger.info(f"Downsampled: {magnitude_db.shape[1]} frames")
                
                # Determine accurate time range using STFT times (center of frames)
                if times is not None and len(times) > 0:
                    time_min = float(np.nanmin(times)) + time_offset
                    time_max = float(np.nanmax(times)) + time_offset + frame_duration
                    if not np.isfinite(time_min):
                        time_min = time_offset
                    if not np.isfinite(time_max):
                        time_max = time_offset + (len(audio_data) / sample_rate if sample_rate > 0 else float(magnitude_db.shape[1]))
                # Ensure bounds are sane
                if time_max <= time_min:
                    time_max = time_min + max(frame_duration, 1e-6)
                if freq_max <= freq_min:
                    freq_max = freq_min + sample_rate / (2 * max(1, magnitude_db.shape[0])) if sample_rate > 0 else freq_min + 1.0
                
                # Update progress before display update
                self.status_widget.update_progress(90)
                QApplication.processEvents()
                
                # Update display
                extent = (time_min, time_max, freq_min, freq_max)
                preserve_view = getattr(self, '_preserve_view_on_update', False)
                self.spectrogram_canvas.update_image(magnitude_db, extent, preserve_view=preserve_view)
                frames = magnitude_db.shape[1]
                
                # Hide progress and show success
                self.status_widget.hide_progress()
                self.statusBar().showMessage(f"Spectrogram ready ({frames} frames)")
                
                # Track current view range using accurate physical axes
                self.current_view_range = ((time_min, time_max), (freq_min, freq_max))
                
                # Update cache
                self.spectrogram_cache = {
                    'time_range': (time_min, time_max),
                    'freq_range': (freq_min, freq_max),
                    'fft_size': base_fft,
                    'hop_length': effective_hop,
                    'data': magnitude_db.copy(),
                    'extent': extent
                }
        except Exception as e:
            logger.exception(f"Spectrogram computation error: {e}")
            self.status_widget.hide_progress()
            self.statusBar().showMessage(f"Spectrogram error: {str(e)}")
    
    def refresh_current_view(self):
        """Refresh the currently active view - always compute full file."""
        if not self.current_file:
            return
        
        # Always compute full file to avoid alignment issues
        # The LOD system will choose appropriate resolution
        duration = self.audio_loader.duration
        sample_rate = self.spectrogram_engine.sample_rate
        
        self.current_view_range = ((0.0, duration), (0.0, sample_rate / 2))
        logger.info(f"Recomputing full file: {duration:.1f}s")
        
        self.load_view_data('spectrogram')
    
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
    
    def export_current_view_as_image(self):
        """Export the current view as an image file."""
        if not self.current_file:
            QMessageBox.warning(self, "Export Warning", "No audio file loaded to export.")
            return
        
        try:
            view_type = 'spectrogram'
            
            # Get canvas
            canvas = getattr(self, 'spectrogram_canvas', None)
            
            if not canvas or not HAS_VISPY:
                QMessageBox.warning(self, "Export Warning", "No valid view to export.")
                return
            
            # File dialog
            filename, _ = QFileDialog.getSaveFileName(
                self, f"Export {view_type.title()} Image",
                f"{view_type}_{os.path.splitext(os.path.basename(self.current_file))[0]}.png",
                "PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)"
            )
            
            if filename:
                # Use VisPy's render method
                image = canvas.render()
                if hasattr(image, 'save'):
                    image.save(filename)
                    self.statusBar().showMessage(f"Exported {view_type} image to {filename}")
                    logger.info(f"Exported {view_type} image to {filename}")
                else:
                    # Fallback: save as numpy array converted to image
                    import imageio
                    imageio.imwrite(filename, image)
                    self.statusBar().showMessage(f"Exported {view_type} image to {filename}")
                    logger.info(f"Exported {view_type} image to {filename}")
                    
        except Exception as e:
            logger.error(f"Error exporting image: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export image:\n{str(e)}")
    
    def export_data_as_npy(self):
        """Export the current view data as a NumPy array."""
        if not self.current_file:
            QMessageBox.warning(self, "Export Warning", "No audio file loaded to export.")
            return
        
        try:
            view_type = 'spectrogram'
            
            # Get the last computed data for this view
            data = self.get_current_view_data(view_type)
            
            if data is None:
                QMessageBox.warning(self, "Export Warning", f"No {view_type} data available to export.")
                return
            
            # File dialog
            filename, _ = QFileDialog.getSaveFileName(
                self, f"Export {view_type.title()} Data",
                f"{view_type}_{os.path.splitext(os.path.basename(self.current_file))[0]}.npy",
                "NumPy Arrays (*.npy);;All Files (*)"
            )
            
            if filename:
                np.save(filename, data)
                self.statusBar().showMessage(f"Exported {view_type} data to {filename}")
                logger.info(f"Exported {view_type} data to {filename} (shape: {data.shape})")
                
        except Exception as e:
            logger.error(f"Error exporting NPY data: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{str(e)}")
    
    def export_data_as_csv(self):
        """Export the current view data as CSV."""
        if not self.current_file:
            QMessageBox.warning(self, "Export Warning", "No audio file loaded to export.")
            return
        
        try:
            view_type = 'spectrogram'
            
            data = self.get_current_view_data(view_type)
            
            if data is None:
                QMessageBox.warning(self, "Export Warning", f"No {view_type} data available to export.")
                return
            
            # File dialog
            filename, _ = QFileDialog.getSaveFileName(
                self, f"Export {view_type.title()} Data",
                f"{view_type}_{os.path.splitext(os.path.basename(self.current_file))[0]}.csv",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if filename:
                import pandas as pd
                
                # Convert 2D array to DataFrame
                if data.ndim == 2:
                    df = pd.DataFrame(data)
                    df.index.name = 'frequency_bin'
                    df.columns.name = 'time_frame'
                else:
                    df = pd.DataFrame(data.flatten(), columns=[view_type])
                
                df.to_csv(filename)
                self.statusBar().showMessage(f"Exported {view_type} data to {filename}")
                logger.info(f"Exported {view_type} data to {filename} (shape: {data.shape})")
                
        except ImportError:
            QMessageBox.critical(self, "Export Error", "pandas library not available for CSV export.")
        except Exception as e:
            logger.error(f"Error exporting CSV data: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{str(e)}")
    
    def get_current_view_data(self, view_type: str):
        """Get the current view's data for export."""
        try:
            # For now, recompute the data for the current view range
            # In a full implementation, this could cache the last computed result
            
            if not self.current_file:
                return None
            
            # Get audio data for current view range
            time_range = self.current_view_range[0]
            start_sample = int(time_range[0] * self.spectrogram_engine.sample_rate)
            end_sample = int(time_range[1] * self.spectrogram_engine.sample_rate)
            
            total_samples = self.audio_loader.total_samples
            num_samples = min(end_sample - start_sample, total_samples - start_sample)
            
            if num_samples <= 0:
                return None
            
            audio_data = self.audio_loader.get_chunk(start_sample, num_samples)
            
            if view_type == 'spectrogram':
                magnitude_db, frequencies, times = self.spectrogram_engine.compute_stft_batched(audio_data)
                return magnitude_db
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting view data for {view_type}: {e}")
            return None
    
    def apply_modern_theme(self):
        """Apply modern glassmorphic dark theme inspired by PyQtGraph example."""
        self.setStyleSheet("""
            /* Main Window - Darker gradient background */
            QMainWindow {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #0A0A15,
                    stop: 0.5 #12121F,
                    stop: 1 #0E1628
                );
            }
            
            /* Central Widget */
            QWidget {
                background: transparent;
                color: rgba(255, 255, 255, 0.9);
                font-size: 10pt;
            }
            
            /* Visualization Container - Distinct darker tone */
            QFrame#viz_container {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(20, 20, 35, 0.95),
                    stop: 1 rgba(15, 15, 30, 0.95)
                );
                border: 1px solid rgba(100, 100, 150, 0.2);
                border-radius: 0px;
            }
            
            /* Tab Widget - Modern style */
            QTabWidget::pane {
                border: none;
                background: transparent;
            }
            
            QTabBar::tab {
                background: rgba(255, 255, 255, 0.05);
                color: rgba(255, 255, 255, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 8px 20px;
                margin-right: 4px;
                font-weight: 500;
            }
            
            QTabBar::tab:selected {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 rgba(99, 102, 241, 0.3),
                    stop: 1 rgba(139, 92, 246, 0.3)
                );
                color: rgba(255, 255, 255, 0.95);
                border: 1px solid rgba(139, 92, 246, 0.5);
                border-bottom: none;
            }
            
            QTabBar::tab:hover {
                background: rgba(255, 255, 255, 0.1);
            }
            
            /* Control Frames */
            QFrame {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 8px;
                padding: 4px;
            }
            
            /* Labels */
            QLabel {
                color: rgba(255, 255, 255, 0.85);
                background: transparent;
                border: none;
                font-size: 10pt;
            }
            
            /* ComboBox - Dropdown */
            QComboBox {
                background: rgba(255, 255, 255, 0.05);
                color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                padding: 5px 10px;
                min-width: 80px;
            }
            
            QComboBox:hover {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid rgba(255, 255, 255, 0.7);
                margin-right: 5px;
            }
            
            QComboBox QAbstractItemView {
                background: rgba(20, 20, 30, 0.95);
                color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                selection-background-color: rgba(99, 102, 241, 0.3);
                padding: 4px;
            }
            
            /* Sliders - Modern style */
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
            
            QSlider::handle:horizontal:hover {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #9F6FFF,
                    stop: 1 #7C7FFF
                );
            }
            
            QSlider::sub-page:horizontal {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #6366F1,
                    stop: 1 #8B5CF6
                );
                border-radius: 3px;
            }
            
            /* Buttons - Modern pill style */
            QPushButton {
                background: rgba(255, 255, 255, 0.05);
                color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                padding: 6px 16px;
                font-weight: 500;
                font-size: 10pt;
            }
            
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.03);
            }
            
            /* Progress Bar */
            QProgressBar {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                text-align: center;
                color: rgba(255, 255, 255, 0.9);
                height: 20px;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #6366F1,
                    stop: 1 #8B5CF6
                );
                border-radius: 5px;
            }
            
            /* Status Bar */
            QStatusBar {
                background: rgba(0, 0, 0, 0.3);
                color: rgba(255, 255, 255, 0.7);
                border-top: 1px solid rgba(255, 255, 255, 0.05);
                font-size: 9pt;
            }
            
            /* Menu Bar */
            QMenuBar {
                background: rgba(0, 0, 0, 0.3);
                color: rgba(255, 255, 255, 0.9);
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                padding: 4px;
            }
            
            QMenuBar::item {
                background: transparent;
                padding: 6px 12px;
                border-radius: 4px;
            }
            
            QMenuBar::item:selected {
                background: rgba(255, 255, 255, 0.1);
            }
            
            QMenu {
                background: rgba(20, 20, 30, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 4px;
            }
            
            QMenu::item {
                color: rgba(255, 255, 255, 0.9);
                padding: 8px 20px;
                border-radius: 4px;
            }
            
            QMenu::item:selected {
                background: rgba(99, 102, 241, 0.3);
            }
            
            /* Scrollbars - Sleek modern style */
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
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            
            QScrollBar:horizontal {
                background: rgba(255, 255, 255, 0.02);
                height: 12px;
                border-radius: 6px;
            }
            
            QScrollBar::handle:horizontal {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                min-width: 30px;
            }
            
            QScrollBar::handle:horizontal:hover {
                background: rgba(255, 255, 255, 0.15);
            }
            
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
    
    def show_keyboard_shortcuts(self):
        """Show keyboard shortcuts help dialog."""
        shortcuts_text = """
<h2>Keyboard Shortcuts</h2>

<h3>Navigation</h3>
<table>
<tr><td><b>Scroll</b></td><td>Zoom both axes</td></tr>
<tr><td><b>Shift + Scroll</b></td><td>Zoom time axis only</td></tr>
<tr><td><b>Ctrl + Scroll</b></td><td>Zoom frequency axis only</td></tr>
<tr><td><b>Drag</b></td><td>Pan view</td></tr>
<tr><td><b>R</b></td><td>Reset zoom to show all</td></tr>
</table>

<h3>Keyboard Zoom</h3>
<table>
<tr><td><b>x</b></td><td>Zoom in time axis</td></tr>
<tr><td><b>X</b></td><td>Zoom out time axis</td></tr>
<tr><td><b>y</b></td><td>Zoom in frequency axis</td></tr>
<tr><td><b>Y</b></td><td>Zoom out frequency axis</td></tr>
</table>

<h3>Measurement</h3>
<table>
<tr><td><b>M</b></td><td>Toggle measurement mode</td></tr>
<tr><td><b>Click</b></td><td>Set start/end point (in measure mode)</td></tr>
<tr><td><b>ESC</b></td><td>Clear measurement</td></tr>
</table>

<h3>General</h3>
<table>
<tr><td><b>Ctrl+O</b></td><td>Open audio file</td></tr>
<tr><td><b>F5</b></td><td>Refresh / Recompute</td></tr>
<tr><td><b>Ctrl+0</b></td><td>Zoom to fit</td></tr>
<tr><td><b>Ctrl+E</b></td><td>Export as image</td></tr>
<tr><td><b>F1</b></td><td>Show this help</td></tr>
</table>
"""
        QMessageBox.information(self, "Keyboard Shortcuts", shortcuts_text)
    
    def closeEvent(self, event):
        """Handle application close."""
        logger.info("Application closing - cleaning up resources...")
        
        # Stop performance timer first
        if hasattr(self, 'perf_timer'):
            self.perf_timer.stop()
        
        # Shutdown task manager and wait for threads to finish
        if hasattr(self, 'task_manager'):
            self.task_manager.shutdown(wait=True)
        
        # Shutdown tile manager and progressive loader
        if hasattr(self, 'tile_manager'):
            self.tile_manager.shutdown()
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
    
    def on_file_selected_from_playlist(self, file_path: str):
        """Handle file selection from playlist."""
        if file_path != self.current_file:
            self.load_audio_file(file_path)
    
    def on_files_dropped(self, file_paths: list):
        """Handle files dropped onto playlist."""
        if file_paths:
            # Load the first dropped file
            self.load_audio_file(file_paths[0])
    
    def get_current_view_type(self) -> str:
        """Get the currently selected view type."""
        return 'spectrogram'
    
    def clear_all_displays(self):
        """Clear all visualization displays."""
        try:
            if hasattr(self, 'spectrogram_canvas') and hasattr(self.spectrogram_canvas, 'image_visual'):
                self.spectrogram_canvas.image_visual.set_data(None)
                self.spectrogram_canvas.update()
        except Exception as e:
            logger.debug(f"Error clearing displays: {e}")
    
    def get_current_file_data(self):
        """Get the current file's audio data."""
        if not self.current_file or self.current_file not in self.file_data:
            return None
        return self.file_data[self.current_file]

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
