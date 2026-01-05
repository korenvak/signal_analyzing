"""
Main application window for the audio visualizer.
"""
import sys
import os
import logging
import json
import numpy as np
from typing import Optional
from pathlib import Path

from .qt_compat import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
    QWidget, QToolBar, QLabel, QPushButton, QFileDialog,
    QMessageBox, QSplitter, QFrame, QSizePolicy, QDialog,
    Qt, QTimer, QAction, QKeySequence, QMenu, QCursor, QTabWidget,
    QInputDialog, QThread, Signal, QObject
)


class SpectrogramWorker(QObject):
    """Background worker for spectrogram computation - keeps UI responsive."""
    
    # Signals for thread-safe communication
    finished = Signal(object, object, object)  # magnitude_db, times, extent
    progress = Signal(int, str)  # progress percent, message
    error = Signal(str)  # error message
    
    def __init__(self, engine, audio_data, fft_params, view_time_range=None):
        super().__init__()
        self.engine = engine
        self.audio_data = audio_data
        self.fft_params = fft_params
        self.view_time_range = view_time_range
        self._cancelled = False
    
    def cancel(self):
        """Request cancellation of computation."""
        self._cancelled = True
    
    def run(self):
        """Compute spectrogram in background thread."""
        try:
            if self._cancelled:
                return
            
            self.progress.emit(10, "Starting FFT computation...")
            
            fft_size = self.fft_params.get('fft_size', 4096)
            hop_length = self.fft_params.get('hop_length', 512)
            window = self.fft_params.get('window', 'hamming')
            sample_rate = float(getattr(self.engine, 'sample_rate', 44100))
            use_gpu = self.fft_params.get('use_gpu', self.engine.use_gpu)
            
            if self._cancelled:
                return
            
            self.progress.emit(30, "Computing STFT...")
            
            # Use batched FFT engine
            magnitude_db, times = self.engine.batched_fft_engine.compute_stft_batched(
                self.audio_data,
                fft_size=fft_size,
                hop_length=hop_length,
                window=window,
                sample_rate=int(sample_rate),
                use_gpu=use_gpu
            )
            
            if self._cancelled:
                return
            
            self.progress.emit(70, "Generating frequency axis...")
            
            frequencies = np.fft.rfftfreq(fft_size, 1.0 / sample_rate).astype(np.float32)
            
            # Calculate extent
            frame_duration = hop_length / sample_rate if sample_rate > 0 else 0.0
            
            if frequencies is not None and len(frequencies) > 0:
                freq_min = float(np.nanmin(frequencies))
                freq_max = float(np.nanmax(frequencies))
            else:
                freq_min = 0.0
                freq_max = sample_rate / 2.0
            
            time_offset = self.view_time_range[0] if self.view_time_range else 0.0
            
            if times is not None and len(times) > 0:
                time_min = float(np.nanmin(times)) + time_offset
                time_max = float(np.nanmax(times)) + time_offset + frame_duration
            else:
                time_min = time_offset
                time_max = time_offset + (len(self.audio_data) / sample_rate if sample_rate > 0 else 1.0)
            
            extent = (time_min, time_max, freq_min, freq_max)
            
            self.progress.emit(90, "Finalizing...")
            
            # Emit result
            self.finished.emit(magnitude_db, times, extent)
            
        except Exception as e:
            self.error.emit(str(e))

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
from ..core.project_manager import ProjectManager
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
from .annotation_manager import AnnotationManager
from .annotation_table import AnnotationTableWidget
from .annotation_renderer import AnnotationRenderer
from .annotation_data import Annotation
from .interaction_manager import InteractionManager, CoordinateMapper
from .cutout_dialog import CutoutDialog
from .spectrum_dialog import SpectrumDialog
from .measurement_panel import MeasurementPanel
from .auto_detect_dialog import AutoDetectDialog, run_auto_detect_dialog
from .event_manager import EventManager
from .event_panel import EventPanel
from .event_dialog import EventInputDialog, EventEditDialog, QuickEventDialog
from .event_data import TaggedEvent
from .track_manager import TrackManager
from .track_data import PaintedTrack
from ..core.filter_manager import FilterManager
from ..core.filename_parser import parse_pixel_filename
from .filter_dialog import (
    GaussianBlurDialog, MedianFilterDialog, ContrastEnhanceDialog,
    ThresholdDialog, MeijeringDialog, MorphologicalDialog,
    HorizontalLineRemovalDialog, VerticalLineRemovalDialog,
    SpectralSubtractionDialog, PCENDialog, AdaptiveNoiseGateDialog,
    TrackSuppressionDialog, LowPassFilterDialog, HighPassFilterDialog,
    BandPassFilterDialog, BandStopFilterDialog, WienerFilterDialog,
    BilateralFilterDialog, HarmonicPercussiveDialog, SpectralGatingDialog,
    TotalVariationDialog, NonLocalMeansDialog, LocalContrastNormDialog,
    CLAHEDialog
)
from .detector_dialog import DetectorParamsDialog
from ..core.spectrogram_detector import SpectrogramDetector, DopplerTrack
from ..core.cutout_analyzer import (
    extract_spectrogram_cutout,
    normalize_cutout,
    save_cutout_image,
    save_cutout_numpy,
    write_cutout_metadata
)
from ..core.gpu_dsp_engine import get_dsp_engine, DetectedCurve, SNRResult, HarmonicResult

# DAS Multi-channel tab (lazy import to avoid circular deps)
DASTab = None

def _get_das_tab_class():
    """Lazy import of DASTab to avoid circular imports."""
    global DASTab
    if DASTab is None:
        from .das_tab import DASTab as _DASTab
        DASTab = _DASTab
    return DASTab

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
        
        # Project Manager for project-based data organization
        self.project_manager = ProjectManager()
        
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
        
        # Background worker management for non-blocking spectrogram computation
        self._spectrogram_thread = None
        self._spectrogram_worker = None
        self._pending_spectrogram_result = None
        
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

        # Annotation system
        self.annotation_manager = AnnotationManager()
        self.annotation_renderer = None  # Will be initialized after canvas creation
        self.selected_annotation_id = None

        # Event tagging system (separate from annotations)
        self.event_manager = EventManager()
        self.event_panel = None  # Will be initialized in setup_ui
        self._file_event_markers = {}  # Per-file storage: {file_path: (t1, t2)}

        # Track painting system (for frequency tracks over spectrograms)
        self.track_manager = TrackManager()

        # Measurement panel (floating window)
        self.measurement_panel = None

        # Filter manager
        self.filter_manager = FilterManager()

        # Doppler track detector
        self.spectrogram_detector = SpectrogramDetector()
        self.detected_tracks = []  # List of DopplerTrack objects
        self.detected_tracks_undo_stack = []  # For Ctrl+Z undo of tracks

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

        # Add compact toolbar at the top
        self.setup_compact_toolbar()
        main_layout.addWidget(self.toolbar)

        # Minimal controls at the top (hidden by default, toggle with Ctrl+P)
        self.controls_widget = ControlsWidget()
        self.controls_widget.setVisible(False)  # Hidden by default
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

        # Main tab widget for different visualization modes
        self.main_tabs = QTabWidget()
        self.main_tabs.setDocumentMode(True)
        self.main_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.main_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: transparent;
            }
            QTabBar::tab {
                padding: 8px 20px;
                margin-right: 2px;
                background: rgba(30, 30, 46, 0.8);
                border: 1px solid #333;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background: rgba(50, 50, 70, 0.95);
                border-bottom: 2px solid #4a9eff;
            }
            QTabBar::tab:hover:!selected {
                background: rgba(40, 40, 60, 0.9);
            }
        """)
        self.main_splitter.addWidget(self.main_tabs)
        self.main_splitter.setSizes([260, 1200])

        # ============ TAB 1: Single Channel Spectrogram ============
        single_channel_tab = QWidget()
        single_channel_layout = QVBoxLayout(single_channel_tab)
        single_channel_layout.setContentsMargins(0, 0, 0, 0)
        single_channel_layout.setSpacing(0)

        if HAS_VISPY:
            spec_container = QFrame()
            spec_container.setFrameShape(QFrame.NoFrame)
            spec_layout = QVBoxLayout(spec_container)
            spec_layout.setContentsMargins(0, 0, 0, 0)
            spec_layout.setSpacing(0)
            
            self.spectrogram_canvas = VisPyCanvas('spectrogram')
            self.spectrogram_canvas.native.setMinimumSize(600, 400)
            spec_layout.addWidget(self.spectrogram_canvas.native)
            
            # Set default normalization mode to STD (adaptive to zoom)
            self.spectrogram_canvas.set_normalization_mode('std', std_scale=2.5)
            
            # Initialize annotation renderer
            self.annotation_renderer = AnnotationRenderer(self.spectrogram_canvas.view)
            self.spectrogram_canvas.set_annotation_renderer(self.annotation_renderer)
            
            # Set annotation callbacks
            self.spectrogram_canvas.set_annotation_callbacks(
                on_created=self.on_annotation_created,
                on_clicked=self.on_annotation_clicked,
                on_context_menu=self.on_annotation_context_menu
            )
            
            # Note: spec_container will be added to splitter later with annotation_table

            # Connect callbacks for status bar updates
            self.spectrogram_canvas.on_zoom_changed_callback = self.on_zoom_level_changed
            self.spectrogram_canvas.on_cursor_moved_callback = self.on_cursor_moved
            
            # Connect auto-recompute callback for high-quality zoom
            self.spectrogram_canvas.on_auto_recompute_callback = self.on_auto_recompute
            
            # Set measurement mode callback
            self.spectrogram_canvas.set_measurement_callback(self.on_measurement_mode_changed)
            self.spectrogram_canvas.set_measurement_completed_callback(self.on_measurement_completed)
            
            # Set curve callback
            self.spectrogram_canvas.set_curve_callback(self.on_curve_completed)

            # Set event tagging callbacks
            self.spectrogram_canvas.set_event_created_callback(self.on_event_region_marked)
            self.spectrogram_canvas.set_event_mode_toggled_callback(self.on_event_mode_toggled_from_canvas)
            
            # Initialize InteractionManager for coordinate mapping
            self.interaction_manager = InteractionManager(
                canvas=self.spectrogram_canvas,
                sample_rate=44100,  # Will be updated when file loads
                hop_length=512,
                fft_size=4096
            )
            # Connect ROI selection signal
            self.interaction_manager.roi_selected.connect(self.on_roi_selected_wrapper)
        
        # Add annotation table below the spectrogram with resizable splitter
        self.annotation_table = AnnotationTableWidget()
        self.annotation_table.setMinimumHeight(80)  # Minimum when collapsed

        # Connect annotation table signals
        self.annotation_table.annotation_selected.connect(self.on_table_annotation_selected)
        self.annotation_table.annotation_deleted.connect(self.on_table_annotation_deleted)
        self.annotation_table.annotation_updated.connect(self.on_table_annotation_updated)
        self.annotation_table.annotation_visibility_changed.connect(self.on_annotation_visibility_changed)
        self.annotation_table.doppler_visibility_changed.connect(self.on_doppler_visibility_changed)

        # Create vertical splitter for spectrogram and annotation table
        if HAS_VISPY:
            self.spec_table_splitter = QSplitter(Qt.Orientation.Vertical)
            self.spec_table_splitter.addWidget(spec_container)
            self.spec_table_splitter.addWidget(self.annotation_table)
            # Set initial sizes (80% spectrogram, 20% table)
            self.spec_table_splitter.setSizes([600, 150])
            # Allow collapsing of annotation table
            self.spec_table_splitter.setCollapsible(0, False)  # Spectrogram not collapsible
            self.spec_table_splitter.setCollapsible(1, True)   # Table is collapsible
            single_channel_layout.addWidget(self.spec_table_splitter)
        else:
            single_channel_layout.addWidget(QLabel("VisPy not available"))

        # Add single channel tab to main tabs
        self.main_tabs.addTab(single_channel_tab, "Single Channel")

        # ============ TAB 2: DAS Multi-Channel ============
        try:
            DASTabClass = _get_das_tab_class()
            self.das_tab = DASTabClass(parent=self)
            self.das_tab.status_message.connect(self._on_das_status_message)
            self.main_tabs.addTab(self.das_tab, "DAS Multi-Channel")
            logger.info("DAS Multi-Channel tab created successfully")
        except Exception as e:
            logger.error(f"Failed to create DAS tab: {e}")
            das_placeholder = QLabel("DAS Multi-Channel tab failed to load.\nCheck logs for details.")
            das_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.main_tabs.addTab(das_placeholder, "DAS Multi-Channel")
            self.das_tab = None

        # Add Event Panel on the right side (hidden by default)
        self.event_panel = EventPanel(self.event_manager)
        self.event_panel.setMinimumWidth(280)
        self.event_panel.setMaximumWidth(400)

        # Connect event panel signals
        self.event_panel.event_mode_toggled.connect(self.on_event_mode_toggled)
        self.event_panel.goto_event_requested.connect(self.on_goto_event)
        self.event_panel.event_table.event_edit_requested.connect(self.on_event_edit_requested)

        self.main_splitter.addWidget(self.event_panel)
        self.main_splitter.setSizes([220, 1000, 300])  # Playlist, Center, Event Panel

        # Hide event panel by default - user toggles with E key or menu
        self.event_panel.hide()

        main_layout.addWidget(viz_container)
        
        # Connect signals
        self.controls_widget.parameters_changed.connect(self.on_parameters_changed)
        self.controls_widget.colormap_changed.connect(self.on_colormap_changed)
        self.controls_widget.db_range_changed.connect(self.on_db_range_changed)
        self.controls_widget.refresh_requested.connect(self.refresh_current_view)
        self.controls_widget.interpolation_changed.connect(self.on_interpolation_changed)
        self.controls_widget.normalization_mode_changed.connect(self.on_normalization_mode_changed)
        self.controls_widget.gamma_changed.connect(self.on_gamma_changed)

        # Connect tab change signal
        self.main_tabs.currentChanged.connect(self._on_main_tab_changed)
    
    def _on_main_tab_changed(self, index: int):
        """Handle main tab change."""
        tab_name = self.main_tabs.tabText(index)
        logger.info(f"Switched to tab: {tab_name}")

        # Update window title based on active tab
        if index == 0:  # Single Channel
            if self.current_file:
                self.setWindowTitle(f"Audio Visualizer - {Path(self.current_file).name}")
            else:
                self.setWindowTitle("GPU-Accelerated Audio Visualizer")
            # Show playlist sidebar for single channel mode
            if hasattr(self, 'playlist_widget'):
                self.playlist_widget.show()
        elif index == 1:  # DAS Multi-Channel
            self.setWindowTitle("DAS Multi-Channel Visualizer")
            # Hide playlist sidebar for DAS mode (DAS has its own sidebar)
            if hasattr(self, 'playlist_widget'):
                self.playlist_widget.hide()

    def _on_das_status_message(self, message: str):
        """Handle status messages from DAS tab."""
        self.statusBar().showMessage(message, 5000)

    def connect_playlist_signals(self):
        """Connect signals from playlist widget."""
        self.playlist_widget.file_selected.connect(self.on_file_selected_from_playlist)
        self.playlist_widget.files_dropped.connect(self.on_files_dropped)
        
    def on_export_project_csv(self):
        """Export summary CSV for the whole project folder."""
        if not self.current_file:
            QMessageBox.warning(self, "No File", "Please load a file first to determine the project folder.")
            return
            
        folder = os.path.dirname(self.current_file)
        default_name = os.path.join(folder, "project_doppler_summary.csv")
        
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Project Summary CSV", default_name, "CSV Files (*.csv)"
        )
        
        if path:
            success = self.annotation_manager.export_project_csv(folder, path)
            if success:
                QMessageBox.information(self, "Export Successful", f"Summary saved to:\n{path}")
            else:
                QMessageBox.critical(self, "Export Failed", "Could not generate the CSV report.")
    
    def on_table_annotation_selected(self, annotation_id: int):
        """Handle selection of annotation in table."""
        if not HAS_VISPY:
            return
            
        self.selected_annotation_id = annotation_id
        
        # Highlight in renderer
        if self.annotation_renderer:
            self.annotation_renderer.select_annotation(annotation_id)
        
        # Zoom to annotation? Optional.
        # ann = self.annotation_manager.get_annotation(annotation_id)
        # if ann:
        #    self.spectrogram_canvas.zoom_to_rect(...)

        # NEW: Load curve points if they exist
        ann = self.annotation_manager.get_annotation(annotation_id)
        if ann:
            # Load curve points to canvas
            if ann.points:
                self.spectrogram_canvas.set_curve_points(ann.points)
                # Also display results in widget
                if ann.doppler_result:
                     # Doppler results exist - just display in status or log
                     logger.info(f"Annotation {annotation_id} has Doppler data: "
                               f"v={ann.doppler_result.get('velocity', 0)*3.6:.1f} km/h")
            else:
                self.spectrogram_canvas.clear_curve()

    def setup_menu_bar(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # Project menu (before File menu)
        project_menu = menubar.addMenu("Project")
        
        new_project_action = QAction("New Project...", self)
        new_project_action.setShortcut("Ctrl+Shift+N")
        new_project_action.triggered.connect(self.new_project)
        project_menu.addAction(new_project_action)
        
        open_project_action = QAction("Open Project...", self)
        open_project_action.setShortcut("Ctrl+Shift+O")
        open_project_action.triggered.connect(self.open_project)
        project_menu.addAction(open_project_action)
        
        save_project_action = QAction("Save Project", self)
        save_project_action.setShortcut("Ctrl+Shift+S")
        save_project_action.triggered.connect(self.save_project)
        project_menu.addAction(save_project_action)
        
        project_menu.addSeparator()
        
        export_project_action = QAction("Export Project...", self)
        export_project_action.triggered.connect(self.export_project)
        project_menu.addAction(export_project_action)
        
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
        
        export_spectrogram_csv_action = QAction("Export Spectrogram as CSV...", self)
        export_spectrogram_csv_action.triggered.connect(self.export_spectrogram_as_csv)
        export_menu.addAction(export_spectrogram_csv_action)
        
        export_ann_csv_action = QAction("Export Annotations as CSV...", self)
        export_ann_csv_action.triggered.connect(self.export_annotations_as_csv)
        export_menu.addAction(export_ann_csv_action)
        
        export_menu.addSeparator()
        
        export_all_cutouts_action = QAction("Export All Cutouts...", self)
        export_all_cutouts_action.triggered.connect(self.export_all_cutouts)
        export_menu.addAction(export_all_cutouts_action)
        
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
        
        view_menu.addSeparator()
        
        self.show_annotations_action = QAction("Show Annotations", self)
        self.show_annotations_action.setCheckable(True)
        self.show_annotations_action.setChecked(True)
        self.show_annotations_action.triggered.connect(self.toggle_annotations_visibility)
        view_menu.addAction(self.show_annotations_action)
        
        # Toggle Annotation Table
        self.show_annotation_table_action = QAction("Show Annotation Table", self)
        self.show_annotation_table_action.setCheckable(True)
        self.show_annotation_table_action.setChecked(True)
        self.show_annotation_table_action.setShortcut("Ctrl+T")
        self.show_annotation_table_action.triggered.connect(self.toggle_annotation_table_visibility)
        view_menu.addAction(self.show_annotation_table_action)
        
        view_menu.addSeparator()
        
        # Measurement Panel
        measurement_panel_action = QAction("Show Measurement Panel", self)
        measurement_panel_action.setShortcut("Ctrl+M")
        measurement_panel_action.triggered.connect(self.open_measurement_panel)
        view_menu.addAction(measurement_panel_action)
        
        # Toggle Controls Panel
        toggle_controls_action = QAction("Toggle Controls Panel", self)
        toggle_controls_action.setShortcut("Ctrl+P")
        toggle_controls_action.triggered.connect(lambda: self.controls_widget.setVisible(not self.controls_widget.isVisible()))
        view_menu.addAction(toggle_controls_action)

        # Toggle Event Panel (E key toggles via canvas, this is for menu access)
        self.toggle_event_panel_action = QAction("Toggle Event Panel (E)", self)
        self.toggle_event_panel_action.setCheckable(True)
        self.toggle_event_panel_action.triggered.connect(self.toggle_event_panel)
        view_menu.addAction(self.toggle_event_panel_action)
        
        view_menu.addSeparator()
        
        # File Navigation shortcuts
        next_file_action = QAction("Next File", self)
        next_file_action.setShortcut("Ctrl+]")
        next_file_action.triggered.connect(self.goto_next_file)
        view_menu.addAction(next_file_action)
        
        prev_file_action = QAction("Previous File", self)
        prev_file_action.setShortcut("Ctrl+[")
        prev_file_action.triggered.connect(self.goto_previous_file)
        view_menu.addAction(prev_file_action)

        # Analysis menu
        analysis_menu = menubar.addMenu("Analysis")

        refresh_action = QAction("Refresh Current View", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.refresh_current_view)
        analysis_menu.addAction(refresh_action)

        analysis_menu.addSeparator()

        # === DSP Analysis Submenu ===
        dsp_menu = analysis_menu.addMenu("DSP Analysis")

        # Auto-detect tracks in visible area
        detect_tracks_action = QAction("Detect Curved Tracks in View...", self)
        detect_tracks_action.setShortcut("Ctrl+Shift+D")
        detect_tracks_action.triggered.connect(self.detect_tracks_in_view)
        dsp_menu.addAction(detect_tracks_action)

        dsp_menu.addSeparator()

        # SNR for selected annotation
        snr_action = QAction("Estimate SNR (Selected Annotation)", self)
        snr_action.triggered.connect(self.estimate_selected_annotation_snr)
        dsp_menu.addAction(snr_action)

        # Harmonics for selected annotation
        harmonics_action = QAction("Detect Harmonics (Selected Annotation)", self)
        harmonics_action.triggered.connect(self.detect_harmonics_selected_annotation)
        dsp_menu.addAction(harmonics_action)

        # Detect track for selected annotation
        detect_curve_action = QAction("Detect Curved Track (Selected Annotation)", self)
        detect_curve_action.triggered.connect(self.detect_curved_track_selected_annotation)
        dsp_menu.addAction(detect_curve_action)

        dsp_menu.addSeparator()

        # Suppress track
        suppress_action = QAction("Suppress Track (Selected Annotation)", self)
        suppress_action.triggered.connect(self.suppress_selected_annotation_track)
        dsp_menu.addAction(suppress_action)

        # Annotation menu
        annotation_menu = menubar.addMenu("Annotations")
        
        self.toggle_annotation_action = QAction("Enable Annotation Mode", self)
        self.toggle_annotation_action.setCheckable(True)
        self.toggle_annotation_action.setShortcut("A")
        self.toggle_annotation_action.triggered.connect(self.toggle_annotation_mode)
        annotation_menu.addAction(self.toggle_annotation_action)
        
        annotation_menu.addSeparator()
        
        save_annotations_action = QAction("Save Annotations...", self)
        save_annotations_action.setShortcut("Ctrl+S")
        save_annotations_action.triggered.connect(self.save_annotations)
        annotation_menu.addAction(save_annotations_action)
        
        load_annotations_action = QAction("Load Annotations...", self)
        load_annotations_action.triggered.connect(self.load_annotations)
        annotation_menu.addAction(load_annotations_action)
        
        annotation_menu.addSeparator()

        auto_detect_action = QAction("Auto Detect Track in Selected", self)
        auto_detect_action.setShortcut("Ctrl+D")
        auto_detect_action.triggered.connect(self.auto_detect_selected_annotation)
        annotation_menu.addAction(auto_detect_action)

        annotation_menu.addSeparator()

        delete_selected_action = QAction("Delete Selected Annotation", self)
        delete_selected_action.setShortcut("Delete")
        delete_selected_action.triggered.connect(self.delete_selected_annotation)
        annotation_menu.addAction(delete_selected_action)

        # Filters menu
        self.setup_filter_menu(menubar)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        shortcuts_action = QAction("Keyboard Shortcuts", self)
        shortcuts_action.setShortcut("F1")
        shortcuts_action.triggered.connect(self.show_keyboard_shortcuts)
        help_menu.addAction(shortcuts_action)
    
    def setup_filter_menu(self, menubar):
        """Setup the Filters menu with all available filters."""
        filter_menu = menubar.addMenu("Filters")

        # === Basic Filters Submenu ===
        basic_menu = filter_menu.addMenu("Basic Filters")

        gaussian_action = QAction("Gaussian Blur...", self)
        gaussian_action.triggered.connect(self.apply_gaussian_blur_filter)
        basic_menu.addAction(gaussian_action)

        median_action = QAction("Median Filter...", self)
        median_action.triggered.connect(self.apply_median_filter)
        basic_menu.addAction(median_action)

        contrast_action = QAction("Contrast Enhancement...", self)
        contrast_action.triggered.connect(self.apply_contrast_filter)
        basic_menu.addAction(contrast_action)

        threshold_action = QAction("Threshold...", self)
        threshold_action.triggered.connect(self.apply_threshold_filter)
        basic_menu.addAction(threshold_action)

        # === Frequency Domain Submenu ===
        freq_menu = filter_menu.addMenu("Frequency Domain")

        lowpass_action = QAction("Low-Pass Filter...", self)
        lowpass_action.triggered.connect(self.apply_lowpass_filter)
        freq_menu.addAction(lowpass_action)

        highpass_action = QAction("High-Pass Filter...", self)
        highpass_action.triggered.connect(self.apply_highpass_filter)
        freq_menu.addAction(highpass_action)

        bandpass_action = QAction("Band-Pass Filter...", self)
        bandpass_action.triggered.connect(self.apply_bandpass_filter)
        freq_menu.addAction(bandpass_action)

        bandstop_action = QAction("Band-Stop Filter...", self)
        bandstop_action.triggered.connect(self.apply_bandstop_filter)
        freq_menu.addAction(bandstop_action)

        # === Ridge/Edge Detection Submenu ===
        detection_menu = filter_menu.addMenu("Ridge/Edge Detection")

        meijering_action = QAction("Meijering Ridge Detection...", self)
        meijering_action.triggered.connect(self.apply_meijering_filter)
        detection_menu.addAction(meijering_action)

        morphological_action = QAction("Morphological Filter...", self)
        morphological_action.triggered.connect(self.apply_morphological_filter)
        detection_menu.addAction(morphological_action)

        # === Noise Removal Submenu ===
        noise_menu = filter_menu.addMenu("Noise Removal")

        spectral_sub_action = QAction("Spectral Subtraction...", self)
        spectral_sub_action.triggered.connect(self.apply_spectral_subtraction_filter)
        noise_menu.addAction(spectral_sub_action)

        pcen_action = QAction("PCEN (Per-Channel Energy Norm)...", self)
        pcen_action.triggered.connect(self.apply_pcen_filter)
        noise_menu.addAction(pcen_action)

        adaptive_gate_action = QAction("Adaptive Noise Gate...", self)
        adaptive_gate_action.triggered.connect(self.apply_adaptive_noise_gate_filter)
        noise_menu.addAction(adaptive_gate_action)

        spectral_gate_action = QAction("Spectral Gating...", self)
        spectral_gate_action.triggered.connect(self.apply_spectral_gating_filter)
        noise_menu.addAction(spectral_gate_action)

        wiener_action = QAction("Wiener Filter...", self)
        wiener_action.triggered.connect(self.apply_wiener_filter)
        noise_menu.addAction(wiener_action)

        # === Track Removal Submenu ===
        track_menu = filter_menu.addMenu("Track/Line Removal")

        h_line_action = QAction("Remove Horizontal Lines (Constant Freq)...", self)
        h_line_action.triggered.connect(self.apply_horizontal_line_removal_filter)
        track_menu.addAction(h_line_action)

        v_line_action = QAction("Remove Vertical Lines (Clicks)...", self)
        v_line_action.triggered.connect(self.apply_vertical_line_removal_filter)
        track_menu.addAction(v_line_action)

        track_suppress_action = QAction("Track Suppression...", self)
        track_suppress_action.triggered.connect(self.apply_track_suppression_filter)
        track_menu.addAction(track_suppress_action)

        hps_action = QAction("Harmonic-Percussive Separation...", self)
        hps_action.triggered.connect(self.apply_harmonic_percussive_filter)
        track_menu.addAction(hps_action)

        # === Advanced Denoising Submenu ===
        advanced_menu = filter_menu.addMenu("Advanced Denoising")

        bilateral_action = QAction("Bilateral Filter (Edge-Preserving)...", self)
        bilateral_action.triggered.connect(self.apply_bilateral_filter)
        advanced_menu.addAction(bilateral_action)

        tv_action = QAction("Total Variation Denoising...", self)
        tv_action.triggered.connect(self.apply_tv_denoise_filter)
        advanced_menu.addAction(tv_action)

        nlm_action = QAction("Non-Local Means...", self)
        nlm_action.triggered.connect(self.apply_nlm_filter)
        advanced_menu.addAction(nlm_action)

        lcn_action = QAction("Local Contrast Normalization...", self)
        lcn_action.triggered.connect(self.apply_lcn_filter)
        advanced_menu.addAction(lcn_action)

        clahe_action = QAction("CLAHE (Adaptive Histogram)...", self)
        clahe_action.triggered.connect(self.apply_clahe_filter)
        advanced_menu.addAction(clahe_action)

        # === Koren's Filter (Multi-Stage Enhancement) ===
        filter_menu.addSeparator()
        koren_action = QAction("Koren's Filter (Adaptive Enhancement)", self)
        koren_action.triggered.connect(self.apply_koren_filter)
        filter_menu.addAction(koren_action)

        filter_menu.addSeparator()

        # === Doppler Track Detection ===
        detect_menu = filter_menu.addMenu("Doppler Detection")

        detect_full_action = QAction("Detect Tracks in Full Spectrogram...", self)
        detect_full_action.triggered.connect(self.detect_doppler_tracks_full)
        detect_menu.addAction(detect_full_action)

        detect_clear_action = QAction("Clear Detected Tracks", self)
        detect_clear_action.triggered.connect(self.clear_detected_tracks)
        detect_menu.addAction(detect_clear_action)

        filter_menu.addSeparator()

        # === Undo/Redo ===
        # General undo action - handles annotations first, then filters
        self.undo_action = QAction("Undo", self)
        self.undo_action.setShortcut("Ctrl+Z")
        self.undo_action.triggered.connect(self.undo_last_action)
        filter_menu.addAction(self.undo_action)

        self.redo_action = QAction("Redo", self)
        self.redo_action.setShortcut("Ctrl+Y")
        self.redo_action.triggered.connect(self.redo_last_action)
        filter_menu.addAction(self.redo_action)

        filter_menu.addSeparator()

        # Keep filter-specific undo for menu (no shortcut)
        self.undo_filter_action = QAction("Undo Filter", self)
        self.undo_filter_action.triggered.connect(self.undo_filter)
        self.undo_filter_action.setEnabled(False)
        filter_menu.addAction(self.undo_filter_action)

        self.redo_filter_action = QAction("Redo Filter", self)
        self.redo_filter_action.triggered.connect(self.redo_filter)
        self.redo_filter_action.setEnabled(False)
        filter_menu.addAction(self.redo_filter_action)

    def setup_compact_toolbar(self):
        """Setup a compact toolbar with essential controls."""
        # QToolBar, QWidget, QLabel, QSizePolicy already imported from qt_compat

        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)
        self.toolbar.setMaximumHeight(32)
        self.toolbar.setStyleSheet("""
            QToolBar {
                background: rgba(30, 30, 40, 0.9);
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
                padding: 2px;
            }
            QToolButton {
                background: transparent;
                color: #B0B0B0;
                border: none;
                padding: 4px 8px;
                margin: 0 2px;
                border-radius: 4px;
            }
            QToolButton:hover {
                background: rgba(255, 255, 255, 0.1);
                color: white;
            }
            QToolButton:pressed {
                background: rgba(255, 255, 255, 0.2);
            }
        """)
        
        # Settings button - toggles settings panel visibility
        settings_action = QAction("⚙ Settings", self)
        settings_action.setCheckable(True)
        settings_action.triggered.connect(lambda checked: self.controls_widget.setVisible(checked))
        self.toolbar.addAction(settings_action)
        self.toggle_settings_action = settings_action
        
        self.toolbar.addSeparator()
        
        # Color palette button
        colors_action = QAction("🎨 Colors", self)
        colors_action.triggered.connect(self.show_colormap_menu)
        self.toolbar.addAction(colors_action)
        
        # View mode button
        view_action = QAction("👁 View", self)
        view_action.triggered.connect(self.show_view_menu)
        self.toolbar.addAction(view_action)
        
        self.toolbar.addSeparator()
        
        # Annotation tools
        annotation_action = QAction("📝 Annotations", self)
        annotation_action.setCheckable(True)
        annotation_action.triggered.connect(lambda checked: self.spectrogram_canvas.set_annotation_mode(checked))
        self.toolbar.addAction(annotation_action)
        
        # Measurement tool
        measure_action = QAction("📏 Measure", self)
        measure_action.setCheckable(True)
        measure_action.triggered.connect(lambda checked: self.spectrogram_canvas.toggle_measurement_mode())
        self.toolbar.addAction(measure_action)

        self.toolbar.addSeparator()

        # DSP Analysis button with dropdown menu
        dsp_action = QAction("🔬 DSP", self)
        dsp_action.triggered.connect(self.show_dsp_menu)
        self.toolbar.addAction(dsp_action)

        # Spacer to push info to the right
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer)
        
        # Info label
        self.toolbar_info = QLabel("Ready")
        self.toolbar_info.setStyleSheet("color: #888; padding: 0 10px;")
        self.toolbar.addWidget(self.toolbar_info)
        
    def show_settings_menu(self):
        """Show settings menu with FFT parameters."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: rgba(30, 30, 40, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item {
                color: #B0B0B0;
                padding: 6px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: rgba(100, 100, 255, 0.3);
                color: white;
            }
        """)
        
        # FFT Size submenu
        fft_menu = menu.addMenu("FFT Size")
        current_fft = getattr(self.spectrogram_engine, 'fft_size', 4096)
        for size in [1024, 2048, 4096, 8192, 16384]:
            action = fft_menu.addAction(f"{size}" + (" ✓" if size == current_fft else ""))
            action.triggered.connect(lambda checked, s=size: self.set_fft_size(s))
        
        # Overlap submenu - calculate current overlap from fft_size and hop_length
        overlap_menu = menu.addMenu("Overlap %")
        current_hop = getattr(self.spectrogram_engine, 'hop_length', 512)
        current_overlap = 1 - (current_hop / current_fft) if current_fft > 0 else 0.75
        for overlap in [0.5, 0.75, 0.875, 0.9375]:
            pct = int(overlap * 100)
            action = overlap_menu.addAction(f"{pct}%" + (" ✓" if abs(overlap - current_overlap) < 0.05 else ""))
            action.triggered.connect(lambda checked, o=overlap: self.set_overlap(o))
        
        # Window function submenu
        window_menu = menu.addMenu("Window Function")
        current_window = getattr(self.spectrogram_engine, 'window_type', 'blackman')
        for win in ['hann', 'hamming', 'blackman', 'bartlett', 'kaiser']:
            action = window_menu.addAction(win.capitalize() + (" ✓" if win == current_window else ""))
            action.triggered.connect(lambda checked, w=win: self.set_window(w))
        
        menu.addSeparator()
        
        # Toggle controls panel
        toggle_action = menu.addAction("Show/Hide Advanced Settings")
        toggle_action.triggered.connect(lambda: self.controls_widget.setVisible(not self.controls_widget.isVisible()))
        
        menu.exec(QCursor.pos())
        
    def set_fft_size(self, size: int):
        """Set FFT size and recompute spectrogram."""
        logger.info(f"Setting FFT size to {size}")
        # Calculate new hop length to maintain similar overlap ratio
        current_fft = getattr(self.spectrogram_engine, 'fft_size', 4096)
        current_hop = getattr(self.spectrogram_engine, 'hop_length', 512)
        overlap_ratio = 1 - (current_hop / current_fft)
        new_hop = int(size * (1 - overlap_ratio))
        
        self.on_parameters_changed({'fft_size': size, 'hop_length': new_hop})
        if hasattr(self, 'toolbar_info'):
            self.toolbar_info.setText(f"FFT: {size}")
        
    def set_overlap(self, overlap: float):
        """Set overlap and recompute spectrogram."""
        logger.info(f"Setting overlap to {overlap*100}%")
        fft_size = getattr(self.spectrogram_engine, 'fft_size', 4096)
        hop_length = int(fft_size * (1 - overlap))
        self.on_parameters_changed({'hop_length': hop_length})
        if hasattr(self, 'toolbar_info'):
            self.toolbar_info.setText(f"Overlap: {int(overlap*100)}%")
            
    def set_window(self, window: str):
        """Set window function and recompute spectrogram."""
        logger.info(f"Setting window to {window}")
        self.on_parameters_changed({'window_type': window})
        if hasattr(self, 'toolbar_info'):
            self.toolbar_info.setText(f"Window: {window}")
        
    def show_colormap_menu(self):
        """Show colormap selection menu."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: rgba(30, 30, 40, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item {
                color: #B0B0B0;
                padding: 6px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: rgba(100, 100, 255, 0.3);
                color: white;
            }
        """)
        
        colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'turbo', 'jet', 'hot', 'cool']
        for cmap in colormaps:
            action = menu.addAction(cmap.capitalize())
            action.triggered.connect(lambda checked, c=cmap: self.set_colormap(c))
        
        menu.exec(QCursor.pos())

    def show_dsp_menu(self):
        """Show DSP analysis menu from toolbar."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: rgba(30, 30, 40, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item {
                color: #B0B0B0;
                padding: 6px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: rgba(100, 100, 255, 0.3);
                color: white;
            }
            QMenu::separator {
                height: 1px;
                background: rgba(255, 255, 255, 0.1);
                margin: 4px 10px;
            }
        """)

        # Detect tracks in view (most commonly used)
        detect_view_action = menu.addAction("🔍 Detect Tracks in View...")
        detect_view_action.triggered.connect(self.detect_tracks_in_view)

        menu.addSeparator()

        # Analysis for selected annotation
        snr_action = menu.addAction("📊 Estimate SNR (Selected)")
        snr_action.triggered.connect(self.estimate_selected_annotation_snr)

        harmonics_action = menu.addAction("🎵 Detect Harmonics (Selected)")
        harmonics_action.triggered.connect(self.detect_harmonics_selected_annotation)

        detect_track_action = menu.addAction("📈 Detect Track (Selected)")
        detect_track_action.triggered.connect(self.detect_curved_track_selected_annotation)

        menu.addSeparator()

        suppress_action = menu.addAction("🚫 Suppress Track (Selected)")
        suppress_action.triggered.connect(self.suppress_selected_annotation_track)

        menu.exec(QCursor.pos())

    def show_view_menu(self):
        """Show view options menu."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: rgba(30, 30, 40, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.2);
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item {
                color: #B0B0B0;
                padding: 6px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: rgba(100, 100, 255, 0.3);
                color: white;
            }
        """)
        
        # Frequency scale
        freq_menu = menu.addMenu("Frequency Scale")
        linear_action = freq_menu.addAction("Linear")
        linear_action.triggered.connect(lambda: self.set_freq_scale('linear'))
        log_action = freq_menu.addAction("Logarithmic")
        log_action.triggered.connect(lambda: self.set_freq_scale('log'))
        mel_action = freq_menu.addAction("Mel")
        mel_action.triggered.connect(lambda: self.set_freq_scale('mel'))
        
        menu.addSeparator()
        
        # Zoom actions
        zoom_fit = menu.addAction("Zoom to Fit")
        zoom_fit.triggered.connect(self.zoom_to_fit)
        
        menu.exec(QCursor.pos())
        
    def setup_zoom_toolbar(self):
        """Setup minimal toolbar - removed for cleaner UI."""
        # All zoom/pan controls are now via keyboard/mouse
        # Toolbar removed per user request
        pass
    
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
        self.statusBar().showMessage("Ready | Shortcuts: A=Annotation C=Curve M=Measure | Right-click annotation for FFT/Doppler")
    
    def open_audio_file(self):
        """Open file dialog to select audio file."""
        # QFileDialog already imported from qt_compat

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Audio File",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*.*)"
        )
        
        if file_path:
            self.load_audio_file(file_path)
    
    def load_audio_file(self, file_path: str):
        """Load an audio file for analysis with optimized cleanup.
        
        PERFORMANCE OPTIMIZATIONS:
        - Cancel pending background computations immediately
        - Lightweight cleanup (preserve GPU memory, just clear caches)
        - Fast audio loader with memory mapping for large files
        """
        try:
            # IMMEDIATE: Cancel any pending background spectrogram computation
            self.cancel_background_spectrogram()
            
            # Show progress feedback
            self.status_widget.show_progress("Loading audio file")
            self.status_widget.update_progress(5)
            self.statusBar().showMessage(f"Loading {os.path.basename(file_path)}...")
            QApplication.processEvents()

            # PERFORMANCE: Lightweight cleanup for fast file switching
            if hasattr(self, 'current_file') and self.current_file is not None:
                logger.info(f"Fast switch: {os.path.basename(self.current_file)} -> {os.path.basename(file_path)}")

                # Clear event markers (but keep CSV data)
                self._clear_event_markers_on_file_switch()

                # LIGHTWEIGHT: Only clear essential caches, preserve GPU memory pool
                # GPU memory is expensive to reallocate - reuse it
                self.spectrogram_cache = {
                    'time_range': None, 'freq_range': None,
                    'fft_size': None, 'hop_length': None,
                    'data': None, 'extent': None
                }

                # Clear canvas display data (reuses buffer on next update)
                if HAS_VISPY and hasattr(self, 'spectrogram_canvas'):
                    self.spectrogram_canvas.raw_display_data = None
                    # Keep normalized_display_data buffer for reuse
                    self.spectrogram_canvas.reset_zoom_history()

                # Clear audio chunk cache only (preserves memory-mapped file)
                if hasattr(self, 'audio_loader'):
                    self.audio_loader.clear_cache()

            # Update progress: cleanup done
            self.status_widget.update_progress(15)
            QApplication.processEvents()

            sample_rate, duration = self.audio_loader.load_file(file_path)

            # Update progress: file loaded
            self.status_widget.update_progress(50)
            QApplication.processEvents()

            self.current_file = file_path
            
            # Add file to project if project is loaded
            if self.project_manager.is_project_loaded():
                analysis_params = {
                    'fft_size': getattr(self.spectrogram_engine, 'fft_size', 4096),
                    'hop_length': getattr(self.spectrogram_engine, 'hop_length', 512),
                    'window': getattr(self.spectrogram_engine, 'window_type', 'hamming'),
                    'sample_rate': sample_rate
                }
                audio_properties = {
                    'sample_rate': sample_rate,
                    'duration_seconds': duration,
                    'channels': 1,  # TODO: get from audio loader if available
                    'bit_depth': 16  # TODO: get from audio loader if available
                }
                self.project_manager.add_file(
                    Path(file_path), 
                    analysis_params=analysis_params,
                    audio_properties=audio_properties
                )
                self.project_manager.save_project(auto_export=True)
            
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
            
            # Update InteractionManager with new sample rate
            if hasattr(self, 'interaction_manager'):
                fft_size = getattr(self.spectrogram_engine, 'fft_size', 4096)
                hop_length = getattr(self.spectrogram_engine, 'hop_length', 512)
                self.interaction_manager.update_parameters(
                    sample_rate=sample_rate,
                    hop_length=hop_length,
                    fft_size=fft_size
                )
            
            logger.info(f"Loaded: {os.path.basename(file_path)} - {duration:.1f}s, {sample_rate}Hz, max freq {nyquist_freq:.0f}Hz")
            
            # Update playlist widget with file info
            if hasattr(self, 'playlist_widget'):
                # First, mark the previously loaded file as not loaded
                for path in self.playlist_widget.get_file_paths():
                    if path != file_path:
                        self.playlist_widget.update_file_info(path, is_loaded=False)

                # Add and mark the new file as loaded
                self.playlist_widget.add_file_path(file_path)
                self.playlist_widget.update_file_info(
                    file_path, duration=duration, sample_rate=sample_rate, is_loaded=True
                )
            
            self.statusBar().showMessage(
                f"Loaded: {os.path.basename(file_path)} "
                f"({duration:.1f}s, {sample_rate}Hz, max {nyquist_freq:.0f}Hz)"
            )
            
            # Save current annotations before switching (if any exist)
            if len(self.annotation_manager) > 0:
                self.save_annotations(silent=True)
                logger.info(f"Auto-saved {len(self.annotation_manager)} annotations before file switch")

            # Update annotation manager for new file (keep annotations persistent across files)
            self.annotation_manager.set_file_path(file_path, clear_annotations=False)
            logger.info(f"Keeping {len(self.annotation_manager)} annotations for file switch")
            
            # Set default normalization mode to STD
            if hasattr(self, 'spectrogram_canvas'):
                self.spectrogram_canvas.set_normalization_mode('std', std_scale=2.5)

            # Update progress: preparing view
            self.status_widget.update_progress(70)
            QApplication.processEvents()

            # Restore event markers for this file if they exist
            self._restore_event_markers_for_file(file_path)

            # Load and display saved tracks for this file
            self._load_tracks_for_file(file_path)

            # Refresh current view (this computes spectrogram - has its own progress)
            # IMPORTANT: This must happen BEFORE refreshing annotations so canvas has valid data
            self.refresh_current_view()

            # Now refresh annotation display AFTER spectrogram is computed
            # This ensures annotations are rendered on a valid canvas
            self.refresh_annotation_display()
            logger.info(f"Refreshed {len(self.annotation_manager)} annotation visuals after spectrogram load")

            # Hide progress bar when done
            self.status_widget.hide_progress()

        except Exception as e:
            self.status_widget.hide_progress()
            QMessageBox.critical(self, "Error", f"Failed to load audio file:\n{str(e)}")
            self.statusBar().showMessage("Ready")
    
    # ==================== Annotation Methods ====================
    
    def on_roi_selected_wrapper(self, t_start, t_end, f_min, f_max):
        """Wrapper to handle ROI selection."""
        self.on_annotation_created(t_start, t_end, f_min, f_max)
    
    def on_annotation_created(self, t_start: float, t_end: float, f_min: float, f_max: float):
        """Handle creation of a new annotation rectangle.
        
        Args:
            t_start: Start time in seconds
            t_end: End time in seconds
            f_min: Minimum frequency in Hz
            f_max: Maximum frequency in Hz
        """
        if not self.current_file:
            return
        
        # Get file name
        file_name = os.path.basename(self.current_file)
        
        # Create annotation
        annotation = Annotation(
            id=None,  # Will be auto-assigned
            file_name=file_name,
            t_start=min(t_start, t_end),
            t_end=max(t_start, t_end),
            f_min=min(f_min, f_max),
            f_max=max(f_min, f_max)
        )
        
        # Add to manager
        annotation = self.annotation_manager.add_annotation(annotation)

        # Add to renderer
        if self.annotation_renderer:
            self.annotation_renderer.add_annotation(annotation, is_selected=False)

        # Add to table
        self.annotation_table.add_annotation(annotation)

        # Auto-save to persist new annotation
        self.save_annotations(silent=True)

        logger.info(f"Created annotation {annotation.id}")
    
    def on_annotation_clicked(self, time: float, freq: float) -> Optional[Annotation]:
        """Handle click on canvas - check if clicking on existing annotation.
        
        Args:
            time: Time coordinate in seconds
            freq: Frequency coordinate in Hz
        
        Returns:
            Annotation if clicked on one, None otherwise
        """
        # Find annotation at this point
        annotation = self.annotation_manager.get_annotation_at_point(time, freq)
        if annotation:
            # Select this annotation
            self.select_annotation(annotation.id)
            return annotation
        return None
    
    def on_annotation_context_menu(self, time: float, freq: float):
        """Handle right-click context menu on annotations.

        Args:
            time: Time coordinate
            freq: Frequency coordinate
        """
        logger.info(f"Context menu requested at t={time:.3f}, f={freq:.0f}")
        annotation = self.annotation_manager.get_annotation_at_point(time, freq)
        if not annotation:
            logger.info(f"No annotation found at point ({time:.3f}, {freq:.0f}), {len(self.annotation_manager.annotations)} annotations exist")
            return
        logger.info(f"Found annotation {annotation.id} at point")
            
        # Select it first
        self.select_annotation(annotation.id)
        
        # Show menu
        menu = QMenu(self)
        
        # FFT Spectrum
        fft_action = menu.addAction("View FFT Spectrum")
        fft_action.triggered.connect(lambda: self.show_fft_dialog(annotation))
        
        menu.addSeparator()
        
        # Cutout extraction
        extract_action = menu.addAction("Extract Cutout...")
        extract_action.triggered.connect(lambda: self.extract_and_show_cutout(annotation))
        
        # Auto-detect track
        auto_detect_action = menu.addAction("Auto Detect Track (Ctrl+D)")
        auto_detect_action.triggered.connect(lambda: self.auto_detect_track_for_annotation(annotation))

        menu.addSeparator()

        # Draw Doppler Curve (if no curve yet)
        if not annotation.points or len(annotation.points) < 4:
            draw_curve_action = menu.addAction("Draw Doppler Curve (C)")
            draw_curve_action.triggered.connect(lambda: self.start_curve_for_annotation(annotation))
        else:
            # Already has curve - offer to calculate or redraw
            doppler_action = menu.addAction("Calculate Doppler Velocity")
            doppler_action.triggered.connect(lambda: self.calculate_doppler_for_annotation(annotation))

            redraw_action = menu.addAction("Redraw Curve")
            redraw_action.triggered.connect(lambda: self.start_curve_for_annotation(annotation))

        menu.addSeparator()

        # === Advanced DSP Analysis ===
        dsp_menu = menu.addMenu("Advanced Analysis")

        # SNR Estimation
        snr_action = dsp_menu.addAction("Estimate SNR")
        snr_action.triggered.connect(lambda: self.estimate_annotation_snr(annotation))

        # Harmonic Detection
        harmonic_action = dsp_menu.addAction("Detect Harmonics")
        harmonic_action.triggered.connect(lambda: self.detect_harmonics_in_annotation(annotation))

        # Curved Track Detection
        curve_detect_action = dsp_menu.addAction("Detect Curved Tracks")
        curve_detect_action.triggered.connect(lambda: self.detect_curved_tracks_in_annotation(annotation))

        dsp_menu.addSeparator()

        # Track Suppression (if has curve)
        if annotation.points and len(annotation.points) >= 4:
            suppress_action = dsp_menu.addAction("Suppress Track from Spectrogram")
            suppress_action.triggered.connect(lambda: self.suppress_annotation_track(annotation))

        menu.addSeparator()

        # Delete
        delete_action = menu.addAction("Delete Annotation")
        delete_action.triggered.connect(lambda: self.on_table_annotation_deleted(annotation.id))
        
        menu.exec(QCursor.pos())

    def show_fft_dialog(self, annotation: Annotation):
        """Show FFT spectrum dialog for the annotation region."""
        if not self.audio_loader or not hasattr(self.audio_loader, 'sample_rate'):
            QMessageBox.warning(self, "No Audio", "No audio data loaded.")
            return
            
        try:
            # Get audio data for the time range
            sample_rate = self.audio_loader.sample_rate
            
            # Calculate sample indices
            idx_start = int(annotation.t_start * sample_rate)
            idx_end = int(annotation.t_end * sample_rate)
            num_samples = idx_end - idx_start
            
            if num_samples <= 0:
                QMessageBox.warning(self, "Invalid Range", "Time range is invalid.")
                return
                
            # Get chunk of audio
            audio_data = self.audio_loader.get_chunk(idx_start, num_samples)
            
            # Create and show dialog
            # SpectrumDialog expects the full audio, not just a chunk
            # So we pass the loader and let it extract what it needs
            dialog = SpectrumDialog(
                audio_data=audio_data,
                sample_rate=sample_rate,
                t_start=0,  # Relative to the chunk we extracted
                t_end=num_samples / sample_rate,
                parent=self
            )
            dialog.setWindowTitle(f"FFT Spectrum - Annotation {annotation.id} | "
                                f"Time: {annotation.t_start:.2f}-{annotation.t_end:.2f}s | "
                                f"Freq: {annotation.f_min:.0f}-{annotation.f_max:.0f}Hz")
            dialog.exec()
            
        except Exception as e:
            logger.error(f"Error showing FFT dialog: {e}")
            QMessageBox.critical(self, "Error", f"Failed to show FFT: {e}")

    def start_curve_for_annotation(self, annotation: Annotation):
        """Start drawing a curve for the selected annotation."""
        # Select the annotation
        self.select_annotation(annotation.id)
        
        # Clear any existing curve
        if HAS_VISPY:
            self.spectrogram_canvas.clear_curve()
            
            # Enter curve mode
            self.spectrogram_canvas.set_curve_mode(True)
            self.statusBar().showMessage(f"Drawing curve for annotation #{annotation.id} | Click to add points | Enter when done")
    
    def calculate_doppler_for_annotation(self, annotation: Annotation):
        """Calculate Doppler velocity from annotation's curve points."""
        if not annotation.points or len(annotation.points) < 4:
            QMessageBox.warning(self, "Not enough points", 
                              "Please draw at least 4 points on the curve.")
            return
            
        try:
            from ..core.doppler_analysis import DopplerAnalyzer
            
            # Extract arrays
            times = np.array([p[0] for p in annotation.points])
            freqs = np.array([p[1] for p in annotation.points])
            
            # Create analyzer
            analyzer = DopplerAnalyzer(speed_of_sound=343.0)
            result = analyzer.fit_curve(times, freqs)
            
            if result:
                # Save to annotation
                annotation.doppler_result = {
                    'velocity': result.velocity,
                    'cpa_distance': result.cpa_distance,
                    't_cpa': result.t_cpa,
                    'f0': result.f0,
                    'rmse': result.rmse,
                    'direction': result.direction
                }
                
                # Update Doppler curve visual in renderer
                if self.annotation_renderer:
                    self.annotation_renderer.update_doppler_curve(annotation)
                
                # Update table to show the new values
                self.annotation_table.update_annotation(annotation)

                # Save to file (uses project manager if loaded)
                self.save_annotations(silent=True)

                QMessageBox.information(self, "Doppler Analysis", 
                                      f"Velocity: {result.velocity_kmh:.1f} km/h\n"
                                      f"Rest Frequency: {result.f0:.1f} Hz\n"
                                      f"CPA Distance: {result.cpa_distance:.1f} m\n"
                                      f"RMSE: {result.rmse:.2f}")
            else:
                QMessageBox.warning(self, "Fit Failed", 
                                  "Could not fit Doppler curve to the points.")
                
        except Exception as e:
            logger.error(f"Error calculating Doppler: {e}")
            QMessageBox.critical(self, "Error", f"Doppler calculation failed: {e}")

    def auto_detect_track_for_annotation(self, annotation: Annotation):
        """Run automatic track detection for an annotation.

        Opens a dialog showing detected tracks for user selection.
        The selected track is then applied to the annotation.
        """
        # Get spectrogram data
        full_data, freqs, times = self._get_spectrogram_axes()

        if full_data is None:
            QMessageBox.warning(self, "No Data", "No spectrogram data available.")
            return

        try:
            # Get engine parameters
            sr = getattr(self.spectrogram_engine, 'sample_rate', 44100)
            hop_length = self.spectrogram_cache.get('hop_length',
                getattr(self.spectrogram_engine, 'hop_length', 512))

            # Run dialog
            selected_points = run_auto_detect_dialog(
                parent=self,
                spectrogram=full_data,
                times=times,
                freqs=freqs,
                t_start=annotation.t_start,
                t_end=annotation.t_end,
                f_min=annotation.f_min,
                f_max=annotation.f_max,
                sample_rate=sr,
                hop_length=hop_length
            )

            if selected_points:
                # Set the track points on the annotation
                self.annotation_manager.set_annotation_track(annotation.id, selected_points)

                # Update the visual
                if self.annotation_renderer:
                    self.annotation_renderer.update_doppler_curve(annotation)

                # Update the table
                self.annotation_table.update_annotation(annotation)

                # Save to file (uses project manager if loaded)
                self.save_annotations(silent=True)

                self.statusBar().showMessage(
                    f"Auto-detected track with {len(selected_points)} points for annotation #{annotation.id}"
                )

                logger.info(f"Auto-detected track for annotation {annotation.id}: {len(selected_points)} points")
            else:
                self.statusBar().showMessage("Auto-detection cancelled or no track selected")

        except Exception as e:
            logger.error(f"Error in auto-detection: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Auto-detection failed: {e}")

    def extract_and_show_cutout(self, annotation: Annotation):
        """Extract spectrogram data for the annotation and show in popup."""
        # Get full spectrogram data
        # Priority: 1. Cache, 2. Canvas Raw Data (fallback)
        
        full_data = None
        if self.spectrogram_cache.get('data') is not None:
            full_data = self.spectrogram_cache['data']
        elif hasattr(self.spectrogram_canvas, 'raw_display_data'):
            full_data = self.spectrogram_canvas.raw_display_data
            logger.warning("Using display data for cutout (might be downsampled)")
            
        if full_data is None:
            QMessageBox.warning(self, "Error", "No spectrogram data available.")
            return
            
        # Reconstruct axes
        # We need times and freqs arrays corresponding to full_data
        # full_data shape is (n_freqs, n_times)
        
        n_freqs, n_times = full_data.shape
        
        # Get parameters - USE CACHED VALUES which reflect actual computation
        sr = getattr(self.spectrogram_engine, 'sample_rate', 44100)
        fft_size = self.spectrogram_cache.get('fft_size', getattr(self.spectrogram_engine, 'fft_size', 4096))
        # CRITICAL: Use the actual hop_length from cache, not engine default!
        hop_length = self.spectrogram_cache.get('hop_length', getattr(self.spectrogram_engine, 'hop_length', 512))
        
        logger.debug(f"Cutout extraction using: sr={sr}, fft_size={fft_size}, hop_length={hop_length}")
        
        # Frequency axis - use cached extent if available
        cache_freq_range = self.spectrogram_cache.get('freq_range')
        if cache_freq_range:
            freqs = np.linspace(cache_freq_range[0], cache_freq_range[1], n_freqs)
        else:
            expected_n_freqs = fft_size // 2 + 1
            if n_freqs == expected_n_freqs:
                freqs = np.fft.rfftfreq(fft_size, 1.0/sr)
            else:
                max_freq = sr / 2.0
                freqs = np.linspace(0, max_freq, n_freqs)
            
        # Time axis - use cached extent if available
        cache_time_range = self.spectrogram_cache.get('time_range')
        if cache_time_range:
            times = np.linspace(cache_time_range[0], cache_time_range[1], n_times)
        else:
            duration_per_frame = hop_length / sr
            times = np.arange(n_times) * duration_per_frame
        
        logger.debug(f"Cutout axes: times=[{times[0]:.2f}, {times[-1]:.2f}], freqs=[{freqs[0]:.1f}, {freqs[-1]:.1f}]")
        
        # Launch Dialog
        dialog = CutoutDialog(
            parent=self,
            spectrogram_data=full_data,
            freqs=freqs,
            times=times,
            t_start=annotation.t_start,
            t_end=annotation.t_end,
            f_low=annotation.f_min,
            f_high=annotation.f_max,
            audio_file_path=self.current_file
        )
        dialog.exec()

    def _get_spectrogram_axes(self):
        """Get the current spectrogram data and axes.
        
        Returns:
            Tuple of (full_data, freqs, times) or (None, None, None) if unavailable
        """
        full_data = None
        if self.spectrogram_cache.get('data') is not None:
            full_data = self.spectrogram_cache['data']
        elif hasattr(self.spectrogram_canvas, 'raw_display_data'):
            full_data = self.spectrogram_canvas.raw_display_data
            
        if full_data is None:
            return None, None, None
        
        n_freqs, n_times = full_data.shape
        sr = getattr(self.spectrogram_engine, 'sample_rate', 44100)
        
        # Frequency axis
        cache_freq_range = self.spectrogram_cache.get('freq_range')
        if cache_freq_range:
            freqs = np.linspace(cache_freq_range[0], cache_freq_range[1], n_freqs)
        else:
            fft_size = self.spectrogram_cache.get('fft_size', getattr(self.spectrogram_engine, 'fft_size', 4096))
            expected_n_freqs = fft_size // 2 + 1
            if n_freqs == expected_n_freqs:
                freqs = np.fft.rfftfreq(fft_size, 1.0/sr)
            else:
                max_freq = sr / 2.0
                freqs = np.linspace(0, max_freq, n_freqs)
        
        # Time axis
        cache_time_range = self.spectrogram_cache.get('time_range')
        if cache_time_range:
            times = np.linspace(cache_time_range[0], cache_time_range[1], n_times)
        else:
            hop_length = self.spectrogram_cache.get('hop_length', getattr(self.spectrogram_engine, 'hop_length', 512))
            duration_per_frame = hop_length / sr
            times = np.arange(n_times) * duration_per_frame
        
        return full_data, freqs, times


    def on_table_annotation_selected(self, annotation_id: int):
        """Handle annotation selection from table.
        
        Args:
            annotation_id: ID of selected annotation
        """
        self.select_annotation(annotation_id)
    
    def on_table_annotation_deleted(self, annotation_id: int):
        """Handle annotation deletion from table.
        
        Args:
            annotation_id: ID of annotation to delete
        """
        # Remove from manager
        if self.annotation_manager.remove_annotation(annotation_id):
            # Remove from renderer
            if self.annotation_renderer:
                self.annotation_renderer.remove_annotation(annotation_id)
                # Force canvas update to remove visual immediately
                if hasattr(self, 'spectrogram_canvas'):
                    self.spectrogram_canvas.update()
            
            # Remove from table (if called from elsewhere)
            # Note: if called from table itself, this might be redundant but safe
            self.annotation_table.remove_annotation(annotation_id)
            
            # Clear selection if this was selected
            if self.selected_annotation_id == annotation_id:
                self.selected_annotation_id = None
                if self.annotation_renderer:
                    self.annotation_renderer.set_selected(None)
            
            # Auto-save to persist deletion
            self.save_annotations(silent=True)
            
            logger.info(f"Deleted annotation {annotation_id} and updated file")
    
    def on_table_annotation_updated(self, annotation_id: int):
        """Handle annotation update from table.
        
        Args:
            annotation_id: ID of updated annotation
        """
        # Get annotation from manager
        annotation = self.annotation_manager.get_annotation(annotation_id)
        if not annotation:
            return
        
        # Get updated data from table
        row = None
        for r, ann_id in self.annotation_table.row_to_id.items():
            if ann_id == annotation_id:
                row = r
                break
        
        if row is None:
            return
        
        # Read updated values from table (View and Curve checkboxes)
        data = self.annotation_table.get_annotation_data_from_row(row)
        if data:
            annotation.is_visible = data.get('is_visible', True)
            annotation.show_doppler_curve = data.get('show_doppler_curve', True)

            # Auto-save to persist changes (uses project manager if loaded)
            self.save_annotations(silent=True)
            logger.debug(f"Updated annotation {annotation_id}")
    
    def on_annotation_visibility_changed(self, annotation_id: int, is_visible: bool):
        """Handle annotation visibility toggle from table.
        
        Args:
            annotation_id: ID of annotation
            is_visible: New visibility state
        """
        logger.info(f"on_annotation_visibility_changed: annotation_id={annotation_id}, is_visible={is_visible}")
        
        annotation = self.annotation_manager.get_annotation(annotation_id)
        if not annotation:
            logger.warning(f"Annotation {annotation_id} not found in manager")
            return
        
        annotation.is_visible = is_visible
        
        # Update visual in renderer
        if self.annotation_renderer:
            self.annotation_renderer.set_annotation_visible(annotation_id, is_visible)
        else:
            logger.warning("No annotation_renderer available")

        # Save changes (uses project manager if loaded)
        self.save_annotations(silent=True)
    
    def on_doppler_visibility_changed(self, annotation_id: int, show_curve: bool):
        """Handle Doppler curve visibility toggle from table.
        
        Args:
            annotation_id: ID of annotation
            show_curve: New visibility state for Doppler curve
        """
        logger.info(f"on_doppler_visibility_changed: annotation_id={annotation_id}, show_curve={show_curve}")
        
        annotation = self.annotation_manager.get_annotation(annotation_id)
        if not annotation:
            logger.warning(f"Annotation {annotation_id} not found in manager")
            return
        
        logger.info(f"Annotation {annotation_id} has {len(annotation.points) if annotation.points else 0} points")
        
        annotation.show_doppler_curve = show_curve
        
        # Update visual in renderer
        if self.annotation_renderer:
            self.annotation_renderer.set_doppler_curve_visible(annotation_id, show_curve)
        else:
            logger.warning("No annotation_renderer available")

        # Save changes (uses project manager if loaded)
        self.save_annotations(silent=True)
    
    def select_annotation(self, annotation_id: int):
        """Select an annotation (highlight it).
        
        Args:
            annotation_id: ID of annotation to select
        """
        self.selected_annotation_id = annotation_id
        
        # Update renderer
        if self.annotation_renderer:
            self.annotation_renderer.set_selected(annotation_id)
        
        # Select in table
        self.annotation_table.select_annotation(annotation_id)
    
    def refresh_annotation_display(self):
        """Refresh all annotation visuals from manager."""
        if not self.annotation_renderer:
            return
        
        # Clear existing visuals
        self.annotation_renderer.clear_all()
        self.annotation_table.clear_all()
        
        # Re-add all annotations
        for annotation in self.annotation_manager:
            self.annotation_renderer.add_annotation(
                annotation, 
                is_selected=(annotation.id == self.selected_annotation_id)
            )
            self.annotation_table.add_annotation(annotation)
    
    def toggle_annotation_mode(self):
        """Toggle annotation drawing mode on/off."""
        enabled = self.toggle_annotation_action.isChecked()
        
        if hasattr(self, 'spectrogram_canvas') and self.spectrogram_canvas:
            self.spectrogram_canvas.set_annotation_mode(enabled)
        
        if enabled:
            self.statusBar().showMessage("Annotation mode enabled - Click and drag to draw rectangles")
        else:
            self.statusBar().showMessage("Annotation mode disabled")
    
    def save_annotations(self, silent=False):
        """Save annotations to JSON file.
        
        Args:
            silent: If True, don't show success message (for auto-save)
        """
        if not self.current_file:
            if not silent:
                QMessageBox.warning(self, "Save Annotations", "No audio file loaded.")
            return
        
        # Get current spectrogram parameters
        spectrogram_params = {
            'fft_size': getattr(self.spectrogram_engine, 'fft_size', 4096),
            'hop_length': getattr(self.spectrogram_engine, 'hop_length', 512),
            'window_type': getattr(self.spectrogram_engine, 'window_type', 'hamming'),
            'sample_rate': getattr(self.spectrogram_engine, 'sample_rate', 44100)
        }
        
        # Calculate overlap percent
        if spectrogram_params['fft_size'] > 0:
            overlap = 1.0 - (spectrogram_params['hop_length'] / spectrogram_params['fft_size'])
            spectrogram_params['overlap_percent'] = overlap * 100.0
        
        # Save with project manager if project is loaded
        if self.annotation_manager.save_to_json(spectrogram_params=spectrogram_params,
                                                project_manager=self.project_manager):
            # Update project stats and auto-export
            if self.project_manager.is_project_loaded() and self.current_file:
                filename = Path(self.current_file).name
                self.project_manager.update_file_stats(filename, len(self.annotation_manager))
                # Auto-export cutouts (PNG/NPY) to project exports directory
                self._auto_export_cutouts_to_project()
                # Auto-export CSV/HTML on save
                self.project_manager.save_project(auto_export=True)

            if not silent:
                QMessageBox.information(self, "Save Annotations",
                                      f"Saved {len(self.annotation_manager)} annotations successfully.")
        else:
            if not silent:
                QMessageBox.critical(self, "Save Annotations", "Failed to save annotations.")
    
    def load_annotations(self):
        """Load annotations from a user-selected JSON file."""
        if not self.current_file:
            QMessageBox.warning(self, "Load Annotations", "No audio file loaded.")
            return
            
        # Start at current file dir or user home
        start_dir = str(Path(self.current_file).parent) if self.current_file else os.path.expanduser("~")
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Annotations File", start_dir, "Annotation Files (*.json *.jsonl);;All Files (*)"
        )
        
        if not file_path:
            return
        
        if self.annotation_manager.load_from_file(file_path):
            self.refresh_annotation_display()
            QMessageBox.information(self, "Load Annotations",
                                  f"Loaded {len(self.annotation_manager)} annotations successfully.")
        else:
            QMessageBox.critical(self, "Load Annotations", "Failed to load annotations.")
    
    def toggle_annotations_visibility(self):
        """Toggle visibility of annotation rectangles."""
        visible = self.show_annotations_action.isChecked()
        if self.annotation_renderer:
            self.annotation_renderer.set_visible(visible)
            # Force update
            if hasattr(self, 'spectrogram_canvas'):
                self.spectrogram_canvas.update()
    
    def toggle_annotation_table_visibility(self):
        """Toggle visibility of annotation table."""
        visible = self.show_annotation_table_action.isChecked()
        if hasattr(self, 'annotation_table'):
            self.annotation_table.setVisible(visible)
            logger.debug(f"Annotation table visibility: {visible}")
    
    def export_all_cutouts(self):
        """Export all annotations as cutouts."""
        if len(self.annotation_manager) == 0:
            QMessageBox.information(self, "Export Cutouts", "No annotations to export.")
            return
            
        # Ask for directory
        start_dir = str(Path(self.current_file).parent) if self.current_file else os.path.expanduser("~")
        save_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory", start_dir)
        if not save_dir:
            return
            
        # Get spectrogram data
        full_data = None
        if self.spectrogram_cache.get('data') is not None:
            full_data = self.spectrogram_cache['data']
        elif hasattr(self.spectrogram_canvas, 'raw_display_data'):
            full_data = self.spectrogram_canvas.raw_display_data
            
        if full_data is None:
            QMessageBox.warning(self, "Error", "No spectrogram data available.")
            return
            
        # Reconstruct axes - USE CACHED VALUES which reflect actual computation
        n_freqs, n_times = full_data.shape
        sr = getattr(self.spectrogram_engine, 'sample_rate', 44100)
        
        # Use cached extent for accurate axes
        cache_freq_range = self.spectrogram_cache.get('freq_range')
        cache_time_range = self.spectrogram_cache.get('time_range')
        
        if cache_freq_range:
            freqs = np.linspace(cache_freq_range[0], cache_freq_range[1], n_freqs)
        else:
            fft_size = self.spectrogram_cache.get('fft_size', getattr(self.spectrogram_engine, 'fft_size', 4096))
            expected_n_freqs = fft_size // 2 + 1
            if n_freqs == expected_n_freqs:
                freqs = np.fft.rfftfreq(fft_size, 1.0/sr)
            else:
                max_freq = sr / 2.0
                freqs = np.linspace(0, max_freq, n_freqs)
        
        if cache_time_range:
            times = np.linspace(cache_time_range[0], cache_time_range[1], n_times)
        else:
            hop_length = self.spectrogram_cache.get('hop_length', getattr(self.spectrogram_engine, 'hop_length', 512))
            duration_per_frame = hop_length / sr
            times = np.arange(n_times) * duration_per_frame
        
        # Process
        from datetime import datetime
        timestamp_batch = datetime.now().strftime("%Y%m%d_%H%M%S")
        manifest_path = Path(save_dir) / f"manifest_{timestamp_batch}.jsonl"
        
        count = 0
        for annotation in self.annotation_manager:
            try:
                # Extract
                cutout = extract_spectrogram_cutout(
                    full_data, freqs, times, 
                    annotation.t_start, annotation.t_end, 
                    annotation.f_min, annotation.f_max
                )
                
                if cutout['S_crop'].size == 0:
                    continue
                    
                # Normalize (Auto mode)
                S_norm, info = normalize_cutout(cutout['S_crop'], mode='auto')
                
                # Save
                base_name = f"cutout_{annotation.id}_{timestamp_batch}"
                png_path = Path(save_dir) / f"{base_name}.png"
                npy_path = Path(save_dir) / f"{base_name}.npy"
                
                save_cutout_image(S_norm, cutout['times_crop'], cutout['freqs_crop'], str(png_path))
                save_cutout_numpy(S_norm, cutout['times_crop'], cutout['freqs_crop'], str(npy_path), raw_data=cutout['S_crop'])
                
                meta = {
                    "annotation_id": annotation.id,
                    "file_name": Path(self.current_file).name,
                    "time_start": annotation.t_start,
                    "time_end": annotation.t_end,
                    "freq_low": annotation.f_min,
                    "freq_high": annotation.f_max,
                    "cutout_png_path": str(png_path.name),
                    "cutout_npy_path": str(npy_path.name),
                    "normalization_used": info.get('applied_mode', 'auto'),
                    "created_utc": datetime.utcnow().isoformat()
                }
                write_cutout_metadata(meta, str(manifest_path))
                count += 1
                
            except Exception as e:
                logger.error(f"Failed to export annotation {annotation.id}: {e}")
        
        QMessageBox.information(self, "Export Complete", f"Successfully exported {count} cutouts to:\n{save_dir}")

    def _auto_export_cutouts_to_project(self):
        """Auto-export all annotation cutouts to project exports directory."""
        if not self.project_manager.is_project_loaded():
            return

        if len(self.annotation_manager) == 0:
            return

        # Get spectrogram data
        full_data = None
        if self.spectrogram_cache.get('data') is not None:
            full_data = self.spectrogram_cache['data']
        elif hasattr(self.spectrogram_canvas, 'raw_display_data'):
            full_data = self.spectrogram_canvas.raw_display_data

        if full_data is None:
            logger.warning("No spectrogram data available for auto-export cutouts")
            return

        # Setup exports directory
        exports_dir = self.project_manager.exports_dir
        if not exports_dir:
            return

        cutouts_dir = exports_dir / "cutouts"
        cutouts_dir.mkdir(parents=True, exist_ok=True)

        # Reconstruct axes
        n_freqs, n_times = full_data.shape
        sr = getattr(self.spectrogram_engine, 'sample_rate', 44100)

        cache_freq_range = self.spectrogram_cache.get('freq_range')
        cache_time_range = self.spectrogram_cache.get('time_range')

        if cache_freq_range:
            freqs = np.linspace(cache_freq_range[0], cache_freq_range[1], n_freqs)
        else:
            fft_size = self.spectrogram_cache.get('fft_size', getattr(self.spectrogram_engine, 'fft_size', 4096))
            expected_n_freqs = fft_size // 2 + 1
            if n_freqs == expected_n_freqs:
                freqs = np.fft.rfftfreq(fft_size, 1.0/sr)
            else:
                max_freq = sr / 2.0
                freqs = np.linspace(0, max_freq, n_freqs)

        if cache_time_range:
            times = np.linspace(cache_time_range[0], cache_time_range[1], n_times)
        else:
            hop_length = self.spectrogram_cache.get('hop_length', getattr(self.spectrogram_engine, 'hop_length', 512))
            duration_per_frame = hop_length / sr
            times = np.arange(n_times) * duration_per_frame

        # Get current file stem for naming
        file_stem = Path(self.current_file).stem if self.current_file else "unknown"

        # Export each annotation
        count = 0
        for annotation in self.annotation_manager:
            try:
                cutout = extract_spectrogram_cutout(
                    full_data, freqs, times,
                    annotation.t_start, annotation.t_end,
                    annotation.f_min, annotation.f_max
                )

                if cutout['S_crop'].size == 0:
                    continue

                # Normalize
                S_norm, info = normalize_cutout(cutout['S_crop'], mode='auto')

                # Save with file-specific naming
                base_name = f"{file_stem}_ann{annotation.id}"
                png_path = cutouts_dir / f"{base_name}.png"
                npy_path = cutouts_dir / f"{base_name}.npz"

                save_cutout_image(S_norm, cutout['times_crop'], cutout['freqs_crop'], str(png_path))
                save_cutout_numpy(S_norm, cutout['times_crop'], cutout['freqs_crop'], str(npy_path), raw_data=cutout['S_crop'])
                count += 1

            except Exception as e:
                logger.warning(f"Failed to auto-export cutout for annotation {annotation.id}: {e}")

        if count > 0:
            logger.info(f"Auto-exported {count} cutouts to {cutouts_dir}")

    def delete_selected_annotation(self):
        """Delete the currently selected annotation."""
        if self.selected_annotation_id is None:
            QMessageBox.information(self, "Delete Annotation", "No annotation selected.")
            return

        # Remove via table (which will trigger the signal chain)
        self.annotation_table.remove_annotation(self.selected_annotation_id)

    def auto_detect_selected_annotation(self):
        """Run auto-detection on the currently selected annotation."""
        if self.selected_annotation_id is None:
            QMessageBox.information(self, "Auto Detect", "No annotation selected.")
            return

        annotation = self.annotation_manager.get_annotation(self.selected_annotation_id)
        if annotation:
            self.auto_detect_track_for_annotation(annotation)
    
    
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
            'window_type': getattr(self.spectrogram_engine, 'window_type', 'blackman'),
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
    
    def on_normalization_mode_changed(self, mode: str):
        """Handle normalization mode changes."""
        logger.info(f"Normalization mode changed to: {mode}")

        try:
            if HAS_VISPY and hasattr(self, 'spectrogram_canvas'):
                self.spectrogram_canvas.set_normalization_mode(mode, std_scale=2.5)
                # Refresh display to apply new normalization
                self.spectrogram_canvas.update_dynamic_clim()
        except Exception as e:
            logger.error(f"Error updating normalization mode: {e}")

    def on_gamma_changed(self, gamma: float):
        """Handle gamma correction changes."""
        logger.info(f"Gamma correction changed to: {gamma:.2f}")

        try:
            if HAS_VISPY and hasattr(self, 'spectrogram_canvas'):
                self.spectrogram_canvas.set_gamma_correction(gamma)
        except Exception as e:
            logger.error(f"Error updating gamma: {e}")

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
    
    def open_measurement_panel(self):
        """Open or show the measurement panel."""
        if self.measurement_panel is None:
            self.measurement_panel = MeasurementPanel(self)
            # Connect new sequence signal to clear visuals
            self.measurement_panel.new_sequence_requested.connect(self.on_new_measurement_sequence)
        
        # Always show and bring to front
        self.measurement_panel.show()
        self.measurement_panel.raise_()
        self.measurement_panel.activateWindow()
        logger.info("Measurement panel opened")
        
    def on_new_measurement_sequence(self):
        """Handle request for new measurement sequence."""
        if hasattr(self, 'spectrogram_canvas'):
            self.spectrogram_canvas.clear_measurement()
    
    def on_measurement_completed(self, t1: float, f1: float, t2: float, f2: float):
        """Handle completed measurement from canvas."""
        # Add to measurement panel
        if self.measurement_panel is None:
            self.open_measurement_panel()
        self.measurement_panel.add_measurement(t1, f1, t2, f2)
        
    def on_curve_completed(self, points: list):
        """Handle curve completion - save to selected annotation and calculate Doppler."""
        logger.info(f"on_curve_completed called with {len(points) if points else 0} points")

        if not points or len(points) < 2:
            logger.warning(f"Not enough points: {len(points) if points else 0}")
            self.statusBar().showMessage("Need at least 2 points for track")
            return

        logger.info("Showing save track dialog...")
        # Ask user if they want to save as a track
        try:
            reply = QMessageBox.question(
                self,
                "Save Track",
                f"You've painted a track with {len(points)} points.\n\n"
                "Do you want to save this track with interpolated data?\n\n"
                "This will create:\n"
                "- CSV file with track summary\n"
                "- CSV file with detailed time/frequency/amplitude data\n"
                "- PNG image with track overlay on spectrogram",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            logger.info(f"User replied: {'Yes' if reply == QMessageBox.StandardButton.Yes else 'No'}")

            if reply == QMessageBox.StandardButton.Yes:
                self.save_painted_track(points)
        except Exception as e:
            logger.error(f"Error showing dialog: {e}", exc_info=True)

        # Save curve to annotation for Doppler analysis
        ann_id = self.annotation_table.get_selected_annotation_id()
        
        # If no annotation selected, try to find one that overlaps with the curve
        if not ann_id and points:
            # Get curve bounds
            times = [p[0] for p in points]
            freqs = [p[1] for p in points]
            curve_t_min, curve_t_max = min(times), max(times)
            curve_f_min, curve_f_max = min(freqs), max(freqs)
            
            # Find overlapping annotation
            for ann in self.annotation_manager:
                if (ann.t_start <= curve_t_max and ann.t_end >= curve_t_min and
                    ann.f_min <= curve_f_max and ann.f_max >= curve_f_min):
                    ann_id = ann.id
                    logger.info(f"Auto-selected annotation {ann_id} based on curve overlap")
                    self.annotation_table.select_annotation(ann_id)
                    break
        
        if ann_id:
            annotation = self.annotation_manager.get_annotation(ann_id)
            if annotation:
                # Save curve points
                annotation.points = points
                annotation.show_doppler_curve = True  # Make sure curve is visible
                logger.info(f"Saved {len(points)} curve points to annotation {ann_id}")

                # Update the Doppler curve visual
                if self.annotation_renderer:
                    self.annotation_renderer.update_doppler_curve(annotation)

                # Automatically calculate Doppler if enough points
                if len(points) >= 4:
                    self.calculate_doppler_for_annotation(annotation)
                    self.statusBar().showMessage(f"Curve saved to annotation #{ann_id} - Doppler analysis complete")
                else:
                    self.statusBar().showMessage(f"Curve saved to annotation #{ann_id} ({len(points)} points)")

                # Update table to reflect curve state
                self.annotation_table.update_annotation(annotation)

                # Save to file (uses project manager if loaded)
                self.save_annotations(silent=True)
        else:
            # No annotation to associate with
            self.statusBar().showMessage(
                "Curve drawn but not linked to any annotation. "
                "Select an annotation first, or create one around the curve."
            )
            logger.info("Curve drawn but no annotation selected or overlapping to associate it with")
        
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
        """Handle dB range changes - DISPLAY ONLY, no FFT recomputation."""
        logger.debug(f"dB range changed to: [{db_min:.1f}, {db_max:.1f}] dB")
        
        try:
            # Update canvas display settings only - NO FFT recomputation needed
            if HAS_VISPY:
                if hasattr(self, 'spectrogram_canvas') and self.spectrogram_canvas.image_visual:
                    # Store dB range for use during data normalization
                    self.spectrogram_canvas.db_range = (db_min, db_max)
                    # Just update the color limits - this is instant
                    self.spectrogram_canvas.update_dynamic_clim()
                    # Force visual refresh without recomputing data
                    self.spectrogram_canvas.update()
                
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
        # NOTE: No recomputation needed - this is just a display transform
    
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
            print(f"\n===== LOAD_SPECTROGRAM_DATA =====")
            print(f"Audio data shape: {audio_data.shape}")
            print(f"View time range: {view_time_range}")
            
            # Check if cached spectrogram covers this region
            # Skip cache during auto-recompute (we want higher resolution)
            preserve_view = getattr(self, '_preserve_view_on_update', False)
            print(f"Preserve view: {preserve_view}")
            print(f"Cache data is None: {self.spectrogram_cache['data'] is None}")
            print(f"Cache time_range: {self.spectrogram_cache['time_range']}")
            
            # PERFORMANCE: Use cache when possible to avoid expensive FFT recomputation
            use_cache = True  # Enable caching for performance
            if use_cache and not preserve_view:
                cache = self.spectrogram_cache
                if (cache['data'] is not None and 
                    cache['time_range'] is not None and
                    view_time_range is not None):
                    
                    cached_start, cached_end = cache['time_range']
                    req_start, req_end = view_time_range
                    
                    # Check if cached region EXACTLY matches (not just contains)
                    time_match = abs(cached_start - req_start) < 0.1 and abs(cached_end - req_end) < 0.1
                    if (time_match and cache['fft_size'] == self.spectrogram_engine.fft_size):
                        
                        print(f"Using cached spectrogram (exact match {req_start:.1f}-{req_end:.1f}s)")
                        self.spectrogram_canvas.update_image(cache['data'], cache['extent'])
                        return
            
            print("Computing new spectrogram (cache bypassed)...")
            
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
    
    def refresh_current_view(self, force_full: bool = False):
        """Refresh the currently active view with smart lazy loading.
        
        PERFORMANCE: Only computes visible region for large files.
        
        Args:
            force_full: If True, always compute full file (for export/filters)
        """
        if not self.current_file:
            return
        
        duration = self.audio_loader.duration
        sample_rate = self.spectrogram_engine.sample_rate
        
        # LAZY LOADING THRESHOLD: Files longer than 5 minutes
        # For shorter files, compute full file (better UX for scrolling)
        lazy_threshold_seconds = 300  # 5 minutes
        
        if not force_full and duration > lazy_threshold_seconds:
            # Get current visible region from canvas
            if hasattr(self, 'spectrogram_canvas') and self.spectrogram_canvas.data_bounds:
                visible = self.spectrogram_canvas.get_visible_region()
                if visible:
                    # Add padding for smoother scrolling (20% on each side)
                    time_span = visible.time_end - visible.time_start
                    padding = time_span * 0.2
                    
                    time_start = max(0.0, visible.time_start - padding)
                    time_end = min(duration, visible.time_end + padding)
                    
                    self.current_view_range = (
                        (time_start, time_end),
                        (visible.freq_start, visible.freq_end)
                    )
                    logger.info(f"Lazy loading visible region: {time_start:.1f}s - {time_end:.1f}s (padded)")
                    self.load_view_data('spectrogram')
                    return
        
        # Full file computation (short files or forced)
        self.current_view_range = ((0.0, duration), (0.0, sample_rate / 2))
        logger.info(f"Computing full file: {duration:.1f}s")
        
        self.load_view_data('spectrogram')
    
    def zoom_to_fit(self):
        """Zoom to fit all data."""
        if not self.current_file:
            return
        
        # Reset view range to full audio duration
        duration = self.audio_loader.duration
        sample_rate = self.spectrogram_engine.sample_rate
        
        self.current_view_range = ((0.0, duration), (0.0, sample_rate / 2))
        # Force full file computation when fitting to view
        self.refresh_current_view(force_full=True)
    
    def goto_next_file(self):
        """Go to the next file in the playlist (Ctrl+])."""
        if hasattr(self, 'playlist_widget'):
            current_idx = self.playlist_widget.get_current_index()
            total = self.playlist_widget.get_total_files()
            
            if self.playlist_widget.select_next_file():
                new_idx = self.playlist_widget.get_current_index()
                self.statusBar().showMessage(f"File {new_idx + 1}/{total}")
            else:
                self.statusBar().showMessage(f"Already at last file ({total}/{total})")
    
    def goto_previous_file(self):
        """Go to the previous file in the playlist (Ctrl+[)."""
        if hasattr(self, 'playlist_widget'):
            current_idx = self.playlist_widget.get_current_index()
            total = self.playlist_widget.get_total_files()
            
            if self.playlist_widget.select_previous_file():
                new_idx = self.playlist_widget.get_current_index()
                self.statusBar().showMessage(f"File {new_idx + 1}/{total}")
            else:
                self.statusBar().showMessage(f"Already at first file (1/{total})")
    
    # =========================================================================
    # Background Spectrogram Computation (Performance Optimization)
    # =========================================================================
    
    def start_background_spectrogram(self, audio_data: np.ndarray, view_time_range: tuple = None):
        """Start spectrogram computation in background thread - keeps UI responsive.
        
        Use this for large audio files to prevent UI freezing.
        
        Args:
            audio_data: Audio samples to process
            view_time_range: Optional (start, end) time range
        """
        # Cancel any existing worker
        self.cancel_background_spectrogram()
        
        # Show progress
        self.status_widget.show_progress("Computing spectrogram...")
        self.status_widget.update_progress(5)
        
        # Calculate FFT parameters
        base_fft = self.spectrogram_engine.fft_size
        base_hop = self.spectrogram_engine.hop_length
        sample_rate = float(getattr(self.spectrogram_engine, "sample_rate", 44100))
        
        # Adaptive hop length for performance
        n_samples = len(audio_data)
        max_texture_size = 16384
        min_hop_for_texture = max(1, n_samples // (max_texture_size - 100))
        
        canvas_width = 800
        if hasattr(self, 'spectrogram_canvas') and self.spectrogram_canvas.native:
            try:
                canvas_width = max(400, self.spectrogram_canvas.native.width())
            except:
                pass
        
        target_frames = canvas_width * 4
        hop_for_canvas = max(1, n_samples // target_frames)
        effective_hop = max(min_hop_for_texture, min(base_hop, hop_for_canvas))
        
        fft_params = {
            'fft_size': base_fft,
            'hop_length': effective_hop,
            'window': self.spectrogram_engine.window_type,
            'sample_rate': int(sample_rate),
            'use_gpu': self.spectrogram_engine.use_gpu
        }
        
        # Create worker and thread
        self._spectrogram_thread = QThread()
        self._spectrogram_worker = SpectrogramWorker(
            self.spectrogram_engine, 
            audio_data, 
            fft_params, 
            view_time_range
        )
        
        # Move worker to thread
        self._spectrogram_worker.moveToThread(self._spectrogram_thread)
        
        # Connect signals
        self._spectrogram_thread.started.connect(self._spectrogram_worker.run)
        self._spectrogram_worker.progress.connect(self._on_spectrogram_progress)
        self._spectrogram_worker.finished.connect(self._on_spectrogram_finished)
        self._spectrogram_worker.error.connect(self._on_spectrogram_error)
        self._spectrogram_worker.finished.connect(self._spectrogram_thread.quit)
        self._spectrogram_worker.error.connect(self._spectrogram_thread.quit)
        self._spectrogram_thread.finished.connect(self._cleanup_spectrogram_worker)
        
        # Start
        self._spectrogram_thread.start()
        logger.info(f"Started background spectrogram computation: {n_samples} samples, hop={effective_hop}")
    
    def cancel_background_spectrogram(self):
        """Cancel any running background spectrogram computation."""
        if self._spectrogram_worker:
            self._spectrogram_worker.cancel()
        if self._spectrogram_thread and self._spectrogram_thread.isRunning():
            self._spectrogram_thread.quit()
            self._spectrogram_thread.wait(1000)  # Wait up to 1 second
    
    def _on_spectrogram_progress(self, percent: int, message: str):
        """Handle spectrogram computation progress updates."""
        self.status_widget.update_progress(percent)
        if message:
            self.statusBar().showMessage(message)
    
    def _on_spectrogram_finished(self, magnitude_db, times, extent):
        """Handle spectrogram computation completion."""
        try:
            if magnitude_db is None or magnitude_db.size == 0:
                self.status_widget.hide_progress()
                self.statusBar().showMessage("Spectrogram computation returned empty data")
                return
            
            self.status_widget.update_progress(95)
            
            # Update display
            preserve_view = getattr(self, '_preserve_view_on_update', False)
            self.spectrogram_canvas.update_image(magnitude_db, extent, preserve_view=preserve_view)
            
            # Update cache
            time_min, time_max, freq_min, freq_max = extent
            self.spectrogram_cache = {
                'time_range': (time_min, time_max),
                'freq_range': (freq_min, freq_max),
                'fft_size': self.spectrogram_engine.fft_size,
                'hop_length': self.spectrogram_engine.hop_length,
                'data': magnitude_db.copy(),
                'extent': extent
            }
            
            # Update view range
            self.current_view_range = ((time_min, time_max), (freq_min, freq_max))
            
            self.status_widget.hide_progress()
            frames = magnitude_db.shape[1]
            self.statusBar().showMessage(f"Spectrogram ready ({frames} frames)")
            logger.info(f"Background spectrogram completed: {magnitude_db.shape}")
            
        except Exception as e:
            logger.exception(f"Error handling spectrogram result: {e}")
            self.status_widget.hide_progress()
            self.statusBar().showMessage(f"Error: {str(e)}")
    
    def _on_spectrogram_error(self, error_msg: str):
        """Handle spectrogram computation error."""
        self.status_widget.hide_progress()
        self.statusBar().showMessage(f"Spectrogram error: {error_msg}")
        logger.error(f"Background spectrogram error: {error_msg}")
    
    def _cleanup_spectrogram_worker(self):
        """Clean up worker after thread finishes."""
        if self._spectrogram_worker:
            self._spectrogram_worker.deleteLater()
            self._spectrogram_worker = None
        if self._spectrogram_thread:
            self._spectrogram_thread.deleteLater()
            self._spectrogram_thread = None
    
    # =========================================================================
    # Filter Application Methods
    # =========================================================================

    def _apply_filter_common(self, filter_func, filter_name: str, **kwargs):
        """Common filter application logic with undo support.

        Args:
            filter_func: Filter function to call (from filter_manager)
            filter_name: Display name for status messages
            **kwargs: Parameters to pass to filter function
        """
        full_data, _, _ = self._get_spectrogram_axes()
        if full_data is None:
            QMessageBox.warning(self, "Filter Error", "No spectrogram data available.")
            return False

        # Push state for Undo
        self.filter_manager.push_state(full_data)
        self._update_undo_redo_state()

        self.statusBar().showMessage(f"Applying {filter_name}... Please wait.")
        QApplication.processEvents()

        try:
            filtered_data, log_msg = filter_func(full_data, **kwargs)

            # Update cache
            self.spectrogram_cache['data'] = filtered_data

            # Update display
            if hasattr(self.spectrogram_canvas, 'update_image'):
                self.spectrogram_canvas.update_image(filtered_data)

            self.statusBar().showMessage(f"Filter applied: {log_msg}")
            return True

        except Exception as e:
            logger.exception(f"{filter_name} failed")
            QMessageBox.critical(self, "Filter Error", f"Failed to apply {filter_name}: {e}")
            self.undo_filter()
            return False

    def _update_undo_redo_state(self):
        """Update undo/redo action enabled states."""
        self.undo_filter_action.setEnabled(self.filter_manager.can_undo())
        self.redo_filter_action.setEnabled(self.filter_manager.can_redo())

    # === Basic Filters ===

    def apply_gaussian_blur_filter(self):
        """Apply Gaussian blur filter with dialog."""
        dialog = GaussianBlurDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_gaussian_blur,
                "Gaussian Blur",
                sigma=params['sigma']
            )

    def apply_median_filter(self):
        """Apply median filter with dialog."""
        dialog = MedianFilterDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_median_filter,
                "Median Filter",
                size=params['size']
            )

    def apply_contrast_filter(self):
        """Apply contrast enhancement with dialog."""
        dialog = ContrastEnhanceDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_contrast_enhancement,
                "Contrast Enhancement",
                percentile_low=params['percentile_low'],
                percentile_high=params['percentile_high']
            )

    def apply_threshold_filter(self):
        """Apply threshold filter with dialog."""
        dialog = ThresholdDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_threshold,
                "Threshold",
                method=params['method'],
                value=params['value'],
                binary=params['binary']
            )

    # === Ridge/Edge Detection ===

    def apply_meijering_filter(self):
        """Apply Meijering ridge detection filter with dialog."""
        dialog = MeijeringDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            sigmas = range(params['sigma_min'], params['sigma_max'] + 1)
            self._apply_filter_common(
                self.filter_manager.apply_meijering,
                "Meijering Ridge Detection",
                sigmas=sigmas,
                black_ridges=params['black_ridges']
            )

    def apply_morphological_filter(self):
        """Apply morphological filter with dialog."""
        dialog = MorphologicalDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_morphological,
                "Morphological Filter",
                operation=params['operation'],
                kernel_width=params['kernel_width'],
                kernel_height=params['kernel_height']
            )

    # === Noise Removal ===

    def apply_spectral_subtraction_filter(self):
        """Apply spectral subtraction with dialog."""
        dialog = SpectralSubtractionDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_spectral_subtraction,
                "Spectral Subtraction",
                noise_percentile=params['noise_percentile'],
                subtraction_factor=params['subtraction_factor'],
                floor=params['floor']
            )

    def apply_pcen_filter(self):
        """Apply PCEN (Per-Channel Energy Normalization) with dialog."""
        dialog = PCENDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_pcen,
                "PCEN",
                time_constant=params['time_constant'],
                gain=params['gain'],
                power=params['power'],
                bias=params['bias'],
                eps=params['eps']
            )

    def apply_adaptive_noise_gate_filter(self):
        """Apply adaptive noise gate with dialog."""
        dialog = AdaptiveNoiseGateDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_adaptive_noise_gate,
                "Adaptive Noise Gate",
                window_time=params['window_time'],
                window_freq=params['window_freq'],
                threshold_db=params['threshold_db'],
                soft_knee=params['soft_knee']
            )

    # === Track/Line Removal ===

    def apply_horizontal_line_removal_filter(self):
        """Apply horizontal line removal with dialog."""
        dialog = HorizontalLineRemovalDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_horizontal_line_removal,
                "Horizontal Line Removal",
                threshold_percentile=params['threshold_percentile'],
                min_width_ratio=params['min_width_ratio'],
                method=params['method']
            )

    def apply_vertical_line_removal_filter(self):
        """Apply vertical line removal with dialog."""
        dialog = VerticalLineRemovalDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_vertical_line_removal,
                "Vertical Line Removal",
                threshold_percentile=params['threshold_percentile'],
                min_height_ratio=params['min_height_ratio'],
                method=params['method']
            )

    def apply_track_suppression_filter(self):
        """Apply track suppression with dialog."""
        dialog = TrackSuppressionDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_track_suppression,
                "Track Suppression",
                method=params['method'],
                sigma=params['sigma'],
                threshold=params['threshold'],
                suppression_strength=params['suppression_strength'],
                inpaint=params['inpaint']
            )

    # === Frequency Domain Filters ===

    def apply_lowpass_filter(self):
        """Apply low-pass filter with dialog."""
        dialog = LowPassFilterDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_lowpass_filter,
                "Low-Pass Filter",
                cutoff_bin=params['cutoff_bin'],
                rolloff=params['rolloff']
            )

    def apply_highpass_filter(self):
        """Apply high-pass filter with dialog."""
        dialog = HighPassFilterDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_highpass_filter,
                "High-Pass Filter",
                cutoff_bin=params['cutoff_bin'],
                rolloff=params['rolloff']
            )

    def apply_bandpass_filter(self):
        """Apply band-pass filter with dialog."""
        dialog = BandPassFilterDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_bandpass_filter,
                "Band-Pass Filter",
                low_cutoff_bin=params['low_cutoff_bin'],
                high_cutoff_bin=params['high_cutoff_bin'],
                rolloff=params['rolloff']
            )

    def apply_bandstop_filter(self):
        """Apply band-stop (notch) filter with dialog."""
        dialog = BandStopFilterDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_bandstop_filter,
                "Band-Stop Filter",
                low_cutoff_bin=params['low_cutoff_bin'],
                high_cutoff_bin=params['high_cutoff_bin'],
                rolloff=params['rolloff']
            )

    # === Advanced Denoising Filters ===

    def apply_wiener_filter(self):
        """Apply Wiener filter with dialog."""
        dialog = WienerFilterDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_wiener_filter,
                "Wiener Filter",
                noise_variance=params['noise_variance'],
                window_size=params['window_size']
            )

    def apply_bilateral_filter(self):
        """Apply bilateral filter with dialog."""
        dialog = BilateralFilterDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_bilateral_filter,
                "Bilateral Filter",
                sigma_spatial=params['sigma_spatial'],
                sigma_color=params['sigma_color']
            )

    def apply_harmonic_percussive_filter(self):
        """Apply harmonic-percussive separation with dialog."""
        dialog = HarmonicPercussiveDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_harmonic_percussive_separation,
                "Harmonic-Percussive Separation",
                kernel_size_harmonic=params['kernel_size_harmonic'],
                kernel_size_percussive=params['kernel_size_percussive'],
                output=params['output']
            )

    def apply_spectral_gating_filter(self):
        """Apply spectral gating with dialog."""
        dialog = SpectralGatingDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_spectral_gating,
                "Spectral Gating",
                noise_percentile=params['noise_percentile'],
                threshold_db=params['threshold_db'],
                smoothing_time=params['smoothing_time'],
                smoothing_freq=params['smoothing_freq']
            )

    def apply_tv_denoise_filter(self):
        """Apply Total Variation denoising with dialog."""
        dialog = TotalVariationDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_denoise_tv,
                "Total Variation Denoising",
                weight=params['weight']
            )

    def apply_nlm_filter(self):
        """Apply Non-Local Means denoising with dialog."""
        dialog = NonLocalMeansDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_non_local_means,
                "Non-Local Means Denoising",
                patch_size=params['patch_size'],
                patch_distance=params['patch_distance'],
                h=params['h']
            )

    def apply_lcn_filter(self):
        """Apply Local Contrast Normalization with dialog."""
        dialog = LocalContrastNormDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_local_contrast_normalization,
                "Local Contrast Normalization",
                window_size=params['window_size'],
                epsilon=params['epsilon']
            )

    def apply_clahe_filter(self):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) with dialog."""
        dialog = CLAHEDialog(self)
        if dialog.exec() == QDialog.Accepted:
            params = dialog.get_values()
            self._apply_filter_common(
                self.filter_manager.apply_clahe,
                "CLAHE",
                clip_limit=params['clip_limit'],
                tile_grid_size=params['tile_grid_size']
            )

    def apply_koren_filter(self):
        """Apply Koren's multi-stage filter pipeline (V11.py) - fully adaptive, no parameters needed."""
        # Get actual sample rate and hop length from spectrogram engine
        sr = getattr(self.spectrogram_engine, 'sample_rate', 44100)
        hop_length = self.spectrogram_cache.get('hop_length',
                     getattr(self.spectrogram_engine, 'hop_length', 512))

        # Koren's filter is fully adaptive - uses optimal parameters from V11.py
        # No dialog needed - just apply with default parameters
        self._apply_filter_common(
            self.filter_manager.apply_koren_filter,
            "Koren's Filter",
            sr=sr,
            hop_length=hop_length
        )

    # === Doppler Track Detection ===

    def detect_doppler_tracks_full(self):
        """Detect Doppler tracks in the full spectrogram."""
        # Get spectrogram data
        full_data, times, freqs = self._get_spectrogram_axes()
        if full_data is None:
            QMessageBox.warning(self, "Warning", "No spectrogram data available. Load a file first.")
            return

        # Show parameters dialog
        dialog = DetectorParamsDialog(self, self.spectrogram_detector)
        if dialog.exec() != QDialog.Accepted:
            return

        # Apply parameters to detector
        dialog.apply_to_detector(self.spectrogram_detector)

        # Save current tracks for undo
        if self.detected_tracks:
            self.detected_tracks_undo_stack.append(self.detected_tracks.copy())
            # Limit undo stack size
            if len(self.detected_tracks_undo_stack) > 10:
                self.detected_tracks_undo_stack.pop(0)

        # Run detection
        self.statusBar().showMessage("Detecting Doppler tracks...")
        QApplication.processEvents()

        try:
            def progress_callback(progress, message):
                self.statusBar().showMessage(message)
                QApplication.processEvents()

            tracks = self.spectrogram_detector.detect(
                full_data, freqs, times,
                progress_callback=progress_callback
            )

            self.detected_tracks = tracks
            logger.info(f"Detected {len(tracks)} Doppler tracks")

            # Draw tracks on spectrogram
            self._draw_detected_tracks()

            self.statusBar().showMessage(f"Detected {len(tracks)} Doppler tracks")
        except Exception as e:
            logger.error(f"Track detection failed: {e}")
            QMessageBox.critical(self, "Error", f"Track detection failed: {e}")
            self.statusBar().showMessage("Track detection failed")

    def clear_detected_tracks(self):
        """Clear all detected tracks from the display."""
        if not self.detected_tracks:
            self.statusBar().showMessage("No detected tracks to clear")
            return

        # Save for undo
        self.detected_tracks_undo_stack.append(self.detected_tracks.copy())
        if len(self.detected_tracks_undo_stack) > 10:
            self.detected_tracks_undo_stack.pop(0)

        self.detected_tracks = []
        self._draw_detected_tracks()
        self.statusBar().showMessage("Detected tracks cleared")

    def undo_detected_tracks(self):
        """Undo the last track detection or clear operation."""
        if not self.detected_tracks_undo_stack:
            self.statusBar().showMessage("No track changes to undo")
            return

        # Restore previous tracks
        self.detected_tracks = self.detected_tracks_undo_stack.pop()
        self._draw_detected_tracks()
        self.statusBar().showMessage(f"Restored {len(self.detected_tracks)} tracks")

    def _draw_detected_tracks(self):
        """Draw detected tracks on the spectrogram canvas."""
        if not hasattr(self, 'spectrogram_canvas'):
            return

        # Clear existing track visuals
        if hasattr(self.spectrogram_canvas, 'clear_detected_tracks'):
            self.spectrogram_canvas.clear_detected_tracks()

        if not self.detected_tracks:
            return

        # Draw each track
        for i, track in enumerate(self.detected_tracks):
            if hasattr(self.spectrogram_canvas, 'add_detected_track'):
                # Pass track data to canvas for rendering
                self.spectrogram_canvas.add_detected_track(
                    times=track.times,
                    freqs=track.freqs,
                    track_id=i,
                    color=(1.0, 0.3, 0.3, 0.8)  # Red-ish color
                )

        # Trigger redraw
        if hasattr(self.spectrogram_canvas, 'update'):
            self.spectrogram_canvas.update()

    # === Undo/Redo ===

    def undo_last_action(self):
        """Undo the last action (annotation or filter).

        Priority: Annotations first, then filters.
        """
        # Try annotation undo first
        if self.annotation_manager.can_undo():
            result = self.annotation_manager.undo()
            if result:
                action_type, annotation = result
                if action_type == 'remove':
                    # Annotation was removed (undid an add)
                    self.annotation_renderer.remove_annotation(annotation.id)
                    self.annotation_table.remove_annotation(annotation.id)
                    self.statusBar().showMessage(f"Undid: removed annotation {annotation.id}")
                elif action_type == 'add':
                    # Annotation was added back (undid a remove)
                    self.annotation_renderer.add_annotation(annotation, is_selected=False)
                    self.annotation_table.add_annotation(annotation)
                    self.statusBar().showMessage(f"Undid: restored annotation {annotation.id}")
                return

        # Fall back to filter undo
        if self.filter_manager.can_undo():
            self.undo_filter()
        else:
            self.statusBar().showMessage("Nothing to undo")

    def redo_last_action(self):
        """Redo the last undone action (annotation or filter).

        Priority: Annotations first, then filters.
        """
        # Try annotation redo first
        if self.annotation_manager.can_redo():
            result = self.annotation_manager.redo()
            if result:
                action_type, annotation = result
                if action_type == 'add':
                    # Annotation was added (redid an add)
                    self.annotation_renderer.add_annotation(annotation, is_selected=False)
                    self.annotation_table.add_annotation(annotation)
                    self.statusBar().showMessage(f"Redid: restored annotation {annotation.id}")
                elif action_type == 'remove':
                    # Annotation was removed (redid a remove)
                    self.annotation_renderer.remove_annotation(annotation.id)
                    self.annotation_table.remove_annotation(annotation.id)
                    self.statusBar().showMessage(f"Redid: removed annotation {annotation.id}")
                return

        # Fall back to filter redo
        if self.filter_manager.can_redo():
            self.redo_filter()
        else:
            self.statusBar().showMessage("Nothing to redo")

    def undo_filter(self):
        """Undo the last filter operation."""
        full_data, _, _ = self._get_spectrogram_axes()
        restored_data = self.filter_manager.undo(full_data)

        if restored_data is not None:
            self.spectrogram_cache['data'] = restored_data
            if hasattr(self.spectrogram_canvas, 'update_image'):
                self.spectrogram_canvas.update_image(restored_data)
            self.statusBar().showMessage("Filter undone.")

        self._update_undo_redo_state()

    def redo_filter(self):
        """Redo the last undone filter operation."""
        full_data, _, _ = self._get_spectrogram_axes()
        restored_data = self.filter_manager.redo(full_data)

        if restored_data is not None:
            self.spectrogram_cache['data'] = restored_data
            if hasattr(self.spectrogram_canvas, 'update_image'):
                self.spectrogram_canvas.update_image(restored_data)
            self.statusBar().showMessage("Filter redone.")

        self._update_undo_redo_state()

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
                # Try to use matplotlib for export with axes
                try:
                    from ..core.export_manager import export_annotation_image
                    
                    # Get raw data and extent
                    raw_data = getattr(canvas, 'raw_display_data', None)
                    extent = getattr(canvas, 'display_extent', None)
                    
                    if raw_data is not None and extent is not None:
                        time_start, time_end, freq_start, freq_end = extent
                        
                        # Create time and frequency arrays
                        num_time = raw_data.shape[1]
                        num_freq = raw_data.shape[0]
                        times = np.linspace(time_start, time_end, num_time)
                        freqs = np.linspace(freq_start, freq_end, num_freq)
                        
                        # Create annotation dict for export
                        annotation_dict = {
                            'id': 0,
                            't_start': time_start,
                            't_end': time_end,
                            'f_min': freq_start,
                            'f_max': freq_end,
                            'points': []
                        }
                        
                        # Export with matplotlib (includes axes)
                        if export_annotation_image(
                            annotation_dict,
                            raw_data,
                            times,
                            freqs,
                            Path(filename),
                            include_track=False,
                            dpi=150,
                            colormap='plasma'
                        ):
                            self.statusBar().showMessage(f"Exported {view_type} image to {filename}")
                            logger.info(f"Exported {view_type} image to {filename}")
                            return
                except Exception as e:
                    logger.warning(f"Matplotlib export failed, using fallback: {e}")
                
                # Fallback: Use VisPy's render method
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
    
    def export_spectrogram_as_csv(self):
        """Export the current spectrogram data as CSV."""
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
                self, f"Export Spectrogram Data",
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
                self.statusBar().showMessage(f"Exported spectrogram data to {filename}")
                logger.info(f"Exported spectrogram data to {filename} (shape: {data.shape})")
                
        except ImportError:
            QMessageBox.critical(self, "Export Error", "pandas library not available for CSV export.")
        except Exception as e:
            logger.error(f"Error exporting CSV data: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{str(e)}")

    def export_annotations_as_csv(self):
        """Export all annotations table as CSV."""
        if not self.current_file:
            QMessageBox.warning(self, "Export Warning", "No audio file loaded.")
            return
            
        if len(self.annotation_manager) == 0:
            QMessageBox.information(self, "Export Warning", "No annotations to export.")
            return
            
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Annotations Table",
                f"annotations_{os.path.splitext(os.path.basename(self.current_file))[0]}.csv",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if filename:
                import pandas as pd
                
                # Collect all annotations
                data = []
                for ann in self.annotation_manager:
                    row = ann.to_dict()
                    # Flatten or format if needed
                    data.append(row)
                
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)
                
                self.statusBar().showMessage(f"Exported {len(data)} annotations to {filename}")
                logger.info(f"Exported annotations to {filename}")
                
        except ImportError:
            QMessageBox.critical(self, "Export Error", "pandas library not available.")
        except Exception as e:
            logger.error(f"Error exporting annotations CSV: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export annotations:\n{str(e)}")
    
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

            /* Table Widget - Fix header visibility for PyQt5 */
            QTableWidget {
                background: rgba(20, 20, 35, 0.95);
                color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 4px;
                gridline-color: rgba(255, 255, 255, 0.1);
            }

            QTableWidget::item {
                padding: 4px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }

            QTableWidget::item:selected {
                background: rgba(99, 102, 241, 0.3);
                color: rgba(255, 255, 255, 0.95);
            }

            QHeaderView::section {
                background: rgba(40, 40, 60, 0.95);
                color: rgba(255, 255, 255, 0.9);
                padding: 6px;
                border: none;
                border-bottom: 1px solid rgba(255, 255, 255, 0.15);
                border-right: 1px solid rgba(255, 255, 255, 0.08);
                font-weight: bold;
            }

            QHeaderView::section:hover {
                background: rgba(60, 60, 80, 0.95);
            }

            /* List Widget - Similar styling */
            QListWidget {
                background: rgba(20, 20, 35, 0.95);
                color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 4px;
            }

            QListWidget::item {
                padding: 6px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            }

            QListWidget::item:selected {
                background: rgba(99, 102, 241, 0.3);
                color: rgba(255, 255, 255, 0.95);
            }

            QListWidget::item:hover {
                background: rgba(255, 255, 255, 0.05);
            }

            /* Tree Widget */
            QTreeWidget {
                background: rgba(20, 20, 35, 0.95);
                color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 4px;
            }

            QTreeWidget::item {
                padding: 4px;
            }

            QTreeWidget::item:selected {
                background: rgba(99, 102, 241, 0.3);
                color: rgba(255, 255, 255, 0.95);
            }

            /* Splitter handles */
            QSplitter::handle {
                background: rgba(255, 255, 255, 0.1);
            }

            QSplitter::handle:horizontal {
                width: 2px;
            }

            QSplitter::handle:vertical {
                height: 2px;
            }

            QSplitter::handle:hover {
                background: rgba(99, 102, 241, 0.5);
            }
        """)
    
    def show_keyboard_shortcuts(self):
        """Show keyboard shortcuts help dialog."""
        shortcuts_text = """
<h2>Keyboard Shortcuts</h2>

<h3>File Navigation</h3>
<table>
<tr><td><b>Ctrl+]</b></td><td>Next file in playlist</td></tr>
<tr><td><b>Ctrl+[</b></td><td>Previous file in playlist</td></tr>
<tr><td><b>Ctrl+O</b></td><td>Open audio file</td></tr>
</table>

<h3>View Navigation</h3>
<table>
<tr><td><b>Scroll</b></td><td>Zoom both axes</td></tr>
<tr><td><b>Shift + Scroll</b></td><td>Zoom time axis only</td></tr>
<tr><td><b>Ctrl + Scroll</b></td><td>Zoom frequency axis only</td></tr>
<tr><td><b>Drag</b></td><td>Pan view</td></tr>
<tr><td><b>R</b></td><td>Reset zoom to show all</td></tr>
<tr><td><b>Ctrl+0</b></td><td>Zoom to fit</td></tr>
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
<tr><td><b>F5</b></td><td>Refresh / Recompute</td></tr>
<tr><td><b>Ctrl+E</b></td><td>Export as image</td></tr>
<tr><td><b>Ctrl+S</b></td><td>Save annotations</td></tr>
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
    
    # ==================== Project Management Methods ====================
    
    def new_project(self):
        """Create a new project."""
        # QInputDialog, QFileDialog already imported from qt_compat

        # Get project name
        name, ok = QInputDialog.getText(
            self, "New Project", "Project Name:",
            text="My Audio Analysis Project"
        )
        if not ok or not name:
            return
        
        # Get project directory
        project_dir = QFileDialog.getExistingDirectory(
            self, "Select Project Directory", ""
        )
        if not project_dir:
            return
        
        # Create project
        project_path = Path(project_dir) / name.replace(" ", "_")
        if self.project_manager.create_project(project_path, name):
            QMessageBox.information(
                self, "Project Created",
                f"Project '{name}' created successfully at:\n{project_path}"
            )
            self.statusBar().showMessage(f"Project: {name}")
            logger.info(f"Created project: {name} at {project_path}")
        else:
            QMessageBox.critical(
                self, "Error",
                f"Failed to create project at:\n{project_path}"
            )
    
    def open_project(self):
        """Open an existing project."""
        # QFileDialog already imported from qt_compat

        project_dir = QFileDialog.getExistingDirectory(
            self, "Open Project", "",
            QFileDialog.ShowDirsOnly
        )
        if not project_dir:
            return

        project_path = Path(project_dir)
        if self.project_manager.load_project(project_path):
            # Get project info after counts are recalculated
            project_info = self.project_manager.get_project_info()
            logger.info(f"Project info: files={project_info['file_count']}, total_annotations={project_info['total_annotations']}")

            # If a file is currently loaded, try to load its annotations from project
            if self.current_file:
                filename = Path(self.current_file).name
                annotations_path = self.project_manager.get_file_annotations_path(filename)
                if annotations_path and annotations_path.exists():
                    self.annotation_manager.load_from_file(str(annotations_path))
                    # Redraw annotations on canvas
                    self.refresh_annotation_display()
                    # Update project stats with loaded annotations
                    self.project_manager.update_file_stats(filename, len(self.annotation_manager))
                    # Refresh project info after update
                    project_info = self.project_manager.get_project_info()
                    logger.info(f"Loaded {len(self.annotation_manager)} annotations for {filename}")

            QMessageBox.information(
                self, "Project Loaded",
                f"Project '{project_info['name']}' loaded successfully.\n"
                f"Files: {project_info['file_count']}\n"
                f"Total annotations: {project_info['total_annotations']}"
            )
            self.statusBar().showMessage(f"Project: {project_info['name']}")
            logger.info(f"Opened project: {project_info['name']}")
        else:
            QMessageBox.critical(
                self, "Error",
                f"Failed to load project from:\n{project_dir}"
            )
    
    def save_project(self):
        """Save current project."""
        if not self.project_manager.is_project_loaded():
            QMessageBox.warning(
                self, "No Project",
                "No project is currently loaded.\nUse 'New Project' or 'Open Project' first."
            )
            return
        
        # Save all annotations
        if self.current_file:
            self.save_annotations(silent=True)
        
        # Save project metadata
        if self.project_manager.save_project():
            self.statusBar().showMessage("Project saved successfully")
            logger.info("Project saved")
        else:
            QMessageBox.critical(self, "Error", "Failed to save project")
    
    def export_project(self):
        """Export project data (unified CSV, images, etc.) to custom location."""
        if not self.project_manager.is_project_loaded():
            QMessageBox.warning(
                self, "No Project",
                "No project is currently loaded."
            )
            return

        # QFileDialog already imported from qt_compat

        # Get export directory
        export_dir = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", ""
        )
        if not export_dir:
            return
        
        export_path = Path(export_dir)
        
        try:
            # Export unified CSV
            csv_path = export_path / "all_annotations.csv"
            self._export_unified_csv(csv_path)
            
            QMessageBox.information(
                self, "Export Complete",
                f"Project exported to:\n{export_path}\n\n"
                f"Files created:\n"
                f"- all_annotations.csv"
            )
            logger.info(f"Project exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export project:\n{str(e)}")
    
    def _export_unified_csv(self, output_path: Path):
        """Export all annotations from all files to unified CSV.
        
        Args:
            output_path: Path to output CSV file
        """
        import pandas as pd
        
        all_annotations = []
        
        # Collect annotations from all files in project
        for file_entry in self.project_manager.project_data.get("files", []):
            filename = file_entry.get("filename")
            annotations_path = self.project_manager.get_file_annotations_path(filename)
            
            if annotations_path and annotations_path.exists():
                try:
                    with open(annotations_path, 'r') as f:
                        data = json.load(f)
                    
                    for ann_dict in data.get("annotations", []):
                        # Add file name to each annotation
                        ann_dict["file_name"] = filename
                        all_annotations.append(ann_dict)
                        
                except Exception as e:
                    logger.warning(f"Failed to load annotations from {filename}: {e}")
        
        if not all_annotations:
            raise ValueError("No annotations found in project")
        
        # Create DataFrame and export
        df = pd.DataFrame(all_annotations)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(all_annotations)} annotations to {output_path}")
    
    def on_file_selected_from_playlist(self, file_path: str):
        """Handle file selection from playlist."""
        print(f"\n========== FILE SELECTED FROM PLAYLIST ==========")
        print(f"Selected file: {file_path}")
        print(f"Current file: {self.current_file}")
        if file_path != self.current_file:
            print(f"Loading new file...")
            self.load_audio_file(file_path)
        else:
            print(f"File already loaded, skipping")
    
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

    # ========== Event Tagging System Methods ==========

    def on_event_mode_toggled(self, enabled: bool):
        """Handle event mode toggle from panel."""
        if hasattr(self, 'spectrogram_canvas'):
            self.spectrogram_canvas.set_event_mode(enabled)
            if enabled:
                self.statusBar().showMessage("Event Mode: Click to place start line")
            else:
                self.statusBar().showMessage("Event Mode disabled")

    def on_event_mode_toggled_from_canvas(self, enabled: bool):
        """Handle event mode toggle from E key press on canvas.

        Shows/hides the event panel and syncs panel toggle button.
        """
        if hasattr(self, 'event_panel'):
            if enabled:
                self.event_panel.show()
                self.event_panel.mode_btn.setChecked(True)
            else:
                self.event_panel.hide()
                self.event_panel.mode_btn.setChecked(False)

            # Sync menu action checkbox
            if hasattr(self, 'toggle_event_panel_action'):
                self.toggle_event_panel_action.setChecked(enabled)

            if enabled:
                self.statusBar().showMessage("Event Mode: Click to place start line")
            else:
                self.statusBar().showMessage("Event Mode disabled")

    def toggle_event_panel(self):
        """Toggle the event panel visibility."""
        if hasattr(self, 'event_panel'):
            is_visible = self.event_panel.isVisible()
            if is_visible:
                self.event_panel.hide()
                if hasattr(self, 'spectrogram_canvas'):
                    self.spectrogram_canvas.set_event_mode(False)
                if hasattr(self, 'toggle_event_panel_action'):
                    self.toggle_event_panel_action.setChecked(False)
                self.statusBar().showMessage("Event Mode disabled")
            else:
                self.event_panel.show()
                if hasattr(self, 'spectrogram_canvas'):
                    self.spectrogram_canvas.set_event_mode(True)
                if hasattr(self, 'toggle_event_panel_action'):
                    self.toggle_event_panel_action.setChecked(True)
                self.statusBar().showMessage("Event Mode: Click to place start line")

    def on_event_region_marked(self, t_start: float, t_end: float):
        """Handle when user marks an event region with two vertical lines.

        This is called after both lines are placed on the spectrogram.
        Uses quick dialog for fast tagging with harmonics count and signal quality.
        """
        if not self.event_manager.session_active:
            QMessageBox.warning(
                self, "No Event Session",
                "Please create or load an event session first.\n\n"
                "Use the Event Panel on the right to start a new session."
            )
            self.spectrogram_canvas.clear_event_markers()
            return

        if not self.current_file:
            QMessageBox.warning(self, "No File", "No audio file is currently loaded.")
            self.spectrogram_canvas.clear_event_markers()
            return

        try:
            # Get current view's frequency range
            rect = self.spectrogram_canvas.view.camera.rect
            if rect:
                f_min = rect.bottom
                f_max = rect.bottom + rect.height
            else:
                f_min, f_max = 0, 22050

            # Show QUICK input dialog (just harmonics count and signal quality)
            dialog = QuickEventDialog(
                t_start=t_start,
                t_end=t_end,
                f_min=f_min,
                f_max=f_max,
                parent=self
            )

            if dialog.exec() == QDialog.Accepted:
                values = dialog.get_values()

                # Create event
                event = self.event_manager.add_event(
                    audio_file=self.current_file,
                    t_start=t_start,
                    t_end=t_end,
                    f_min=values['f_min'],
                    f_max=values['f_max'],
                    harmonic_number=values['harmonic_number'],
                    snr_estimate_db=values['snr_estimate_db'],
                    notes=values['notes']
                )

                # Capture image
                self._capture_event_image(event)

                # Save to CSV
                self.event_manager.append_event_to_csv(event)

                # Add to table
                self.event_panel.add_event(event)

                # DON'T clear markers - they stay visible until user clicks to start new event or exits mode
                self.statusBar().showMessage(f"Event {event.id} saved - click to mark new event")
            else:
                # User cancelled - keep markers visible, user can retry or click to start new
                pass

        except Exception as e:
            logger.error(f"Error creating event: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Failed to create event:\n{e}")
            # Keep markers visible - user can retry

    def _estimate_event_snr(self, t_start: float, t_end: float,
                           f_min: float, f_max: float) -> Optional[float]:
        """Estimate SNR for an event region from cached spectrogram data.

        Returns:
            Estimated SNR in dB, or None if estimation fails
        """
        try:
            S, freqs, times = self._get_spectrogram_axes()
            if S is None or freqs is None or times is None:
                return None

            return self.event_manager.estimate_snr(S, times, freqs, t_start, t_end, f_min, f_max)

        except Exception as e:
            logger.debug(f"SNR estimation failed: {e}")
            return None

    def _capture_event_image(self, event: TaggedEvent):
        """Capture spectrogram image for an event.

        Args:
            event: Event to capture image for
        """
        try:
            S, freqs, times = self._get_spectrogram_axes()
            if S is None or freqs is None or times is None:
                logger.warning("No spectrogram data available for image capture")
                return

            # Get current colormap from controls widget
            colormap = 'plasma'
            if hasattr(self, 'controls_widget') and hasattr(self.controls_widget, 'colormap_combo'):
                colormap = self.controls_widget.colormap_combo.currentText() or 'plasma'
                # Check if inverted
                if hasattr(self.controls_widget, 'invert_colormap') and self.controls_widget.invert_colormap.isChecked():
                    colormap = colormap + '_r' if not colormap.endswith('_r') else colormap[:-2]

            # Capture image
            image_path = self.event_manager.capture_event_image(
                event, S, times, freqs, colormap=colormap
            )

            if image_path:
                # Image path is already set on the event by capture_event_image
                # Don't call _save_all_to_csv here - the caller will save the event
                logger.info(f"Event {event.id} image captured: {image_path}")

        except Exception as e:
            logger.error(f"Failed to capture event image: {e}", exc_info=True)

    def on_event_edit_requested(self, event_id: int):
        """Handle request to edit an event."""
        event = self.event_manager.get_event(event_id)
        if not event:
            return

        dialog = EventEditDialog(
            event_id=event.id,
            harmonic_number=event.harmonic_number,
            snr_estimate_db=event.snr_estimate_db,
            notes=event.notes,
            parent=self
        )

        if dialog.exec() == QDialog.Accepted:
            values = dialog.get_values()
            self.event_manager.update_event(
                event_id,
                harmonic_number=values['harmonic_number'],
                snr_estimate_db=values['snr_estimate_db'],
                notes=values['notes']
            )
            # Update table
            updated_event = self.event_manager.get_event(event_id)
            if updated_event:
                self.event_panel.update_event(updated_event)
            self.statusBar().showMessage(f"Event {event_id} updated")

    def save_painted_track(self, points: list):
        """Save a painted frequency track with interpolation and metadata.

        Args:
            points: List of (time, frequency) tuples from user clicks
        """
        if not self.current_file:
            QMessageBox.warning(self, "Save Track", "No audio file loaded.")
            return

        # Create session directory for this specific audio file
        # Each audio file gets its own subfolder under a common _tracks directory
        audio_path = Path(self.current_file)
        base_tracks_dir = audio_path.parent / "tracks_session"
        file_session_dir = base_tracks_dir / audio_path.stem

        # Check if we need to create/switch to a new session for this file
        current_session_dir = self.track_manager.csv_path.parent if self.track_manager.csv_path else None
        if current_session_dir != file_session_dir:
            # Create or load session for this specific file
            tracks_csv = file_session_dir / "tracks.csv"
            if tracks_csv.exists():
                success = self.track_manager.load_existing_session(tracks_csv)
            else:
                success = self.track_manager.create_new_session(file_session_dir)

            if not success:
                QMessageBox.critical(self, "Save Track", "Failed to create/load track session.")
                return

            logger.info(f"Track session for {audio_path.name} at {file_session_dir}")

        # Get spectrogram data
        try:
            # Get current spectrogram data from cache/canvas
            S, freqs, times = self._get_spectrogram_axes()

            if S is None or times is None or freqs is None:
                QMessageBox.warning(self, "Save Track", "No spectrogram data available.")
                return

            # Ask for track label
            track_label, ok = QInputDialog.getText(
                self,
                "Track Label",
                "Enter a label for this track (optional):",
                text=""
            )
            if not ok:
                return  # User cancelled

            # Create track with interpolation and amplitude sampling
            track = self.track_manager.create_track(
                control_points=points,
                audio_file=self.current_file,
                S=S,
                times_axis=times,
                freqs_axis=freqs,
                num_interp_samples=200,  # Generate 200 interpolated points
                track_label=track_label,
                notes=""
            )

            if track is None:
                QMessageBox.critical(self, "Save Track", "Failed to create track.")
                return

            # Save track with all exports
            success = self.track_manager.save_track(
                track=track,
                S=S,
                times=times,
                freqs=freqs,
                export_image=True,
                export_detailed_csv=True
            )

            if success:
                # Add track to canvas visualization (keeps it visible after save)
                self._add_track_to_canvas(track)

                QMessageBox.information(
                    self,
                    "Track Saved",
                    f"Track #{track.id} saved successfully!\n\n"
                    f"Control points: {track.n_control_points}\n"
                    f"Interpolated points: {track.n_interpolated_points}\n"
                    f"Duration: {track.duration:.3f} s\n"
                    f"Frequency range: {track.f_min:.1f} - {track.f_max:.1f} Hz\n\n"
                    f"Files created:\n"
                    f"- Summary CSV: {self.track_manager.csv_path}\n"
                    f"- Detailed CSV: {self.track_manager.data_dir}\n"
                    f"- PNG image: {track.image_path}"
                )
                logger.info(f"Track {track.id} saved: {track.n_control_points} control points, "
                           f"{track.n_interpolated_points} interpolated points")
            else:
                QMessageBox.warning(self, "Save Track", "Track saved with some errors. Check logs.")

        except Exception as e:
            logger.error(f"Failed to save track: {e}")
            QMessageBox.critical(self, "Save Track", f"Error saving track: {e}")

    def _add_track_to_canvas(self, track):
        """Add a saved track to the canvas visualization.

        Args:
            track: PaintedTrack object with interpolated data
        """
        if not hasattr(self, 'spectrogram_canvas'):
            return

        try:
            # Prepare interpolated points as Nx2 array
            if track.interpolated_times and track.interpolated_freqs:
                interp_points = np.column_stack([
                    track.interpolated_times,
                    track.interpolated_freqs
                ])
            else:
                return

            # Prepare control points as Nx2 array
            control_pts = None
            if track.control_points:
                control_pts = np.array(track.control_points)

            # Add to canvas stored tracks
            self.spectrogram_canvas.add_stored_track(
                track_id=track.id,
                interpolated_points=interp_points,
                control_points=control_pts,
                visible=True,
                color='lime'  # Green for saved tracks
            )

            logger.info(f"Added track {track.id} to canvas visualization")

        except Exception as e:
            logger.error(f"Failed to add track to canvas: {e}")

    def _load_tracks_for_file(self, audio_file: str):
        """Load and display saved tracks for a specific audio file.

        Args:
            audio_file: Path to the audio file
        """
        if not hasattr(self, 'spectrogram_canvas'):
            return

        # Clear existing stored tracks from canvas
        self.spectrogram_canvas.clear_stored_tracks()

        # Try to load tracks for this file
        audio_path = Path(audio_file)
        base_tracks_dir = audio_path.parent / "tracks_session"
        file_session_dir = base_tracks_dir / audio_path.stem
        tracks_csv = file_session_dir / "tracks.csv"

        if tracks_csv.exists():
            # Load the session for this file
            if self.track_manager.load_existing_session(tracks_csv):
                # Get tracks for this file and display them
                tracks = self.track_manager.get_tracks_for_file(audio_file)
                for track in tracks:
                    self._add_track_to_canvas(track)
                logger.info(f"Loaded {len(tracks)} tracks for {audio_path.name}")

    def on_goto_event(self, t_center: float, f_center: float):
        """Navigate the spectrogram view to center on an event.

        Args:
            t_center: Time center of the event
            f_center: Frequency center of the event
        """
        if not hasattr(self, 'spectrogram_canvas'):
            return

        try:
            rect = self.spectrogram_canvas.view.camera.rect
            if rect is None:
                return

            # Keep current view size, just change center
            new_left = t_center - rect.width / 2
            new_bottom = f_center - rect.height / 2

            # Apply bounds
            if self.spectrogram_canvas.data_bounds:
                t_min, t_max, f_min, f_max = self.spectrogram_canvas.data_bounds
                new_left = max(t_min, min(new_left, t_max - rect.width))
                new_bottom = max(f_min, min(new_bottom, f_max - rect.height))

            self.spectrogram_canvas.view.camera.rect = (
                new_left, new_bottom, rect.width, rect.height
            )
            self.spectrogram_canvas.update()

        except Exception as e:
            logger.error(f"Failed to navigate to event: {e}")

    def _clear_event_markers_on_file_switch(self):
        """Save and clear event markers when switching files.

        Stores current markers for this file so they can be restored later.
        """
        if hasattr(self, 'spectrogram_canvas') and self.current_file:
            canvas = self.spectrogram_canvas
            # Save current markers for this file before clearing
            if canvas.event_t1 is not None or canvas.event_t2 is not None:
                self._file_event_markers[self.current_file] = (canvas.event_t1, canvas.event_t2)
            # Clear visual markers
            canvas.clear_event_markers()
        if hasattr(self, 'event_panel'):
            self.event_panel.on_file_switched()

    def _restore_event_markers_for_file(self, file_path: str):
        """Restore event markers when switching back to a file.

        Args:
            file_path: The file to restore markers for
        """
        if file_path in self._file_event_markers and hasattr(self, 'spectrogram_canvas'):
            t1, t2 = self._file_event_markers[file_path]
            canvas = self.spectrogram_canvas

            # Restore marker times
            canvas.event_t1 = t1
            canvas.event_t2 = t2

            # Recreate visual lines if times exist
            if t1 is not None:
                canvas.event_line_1 = canvas._create_event_vertical_line(t1)
                # If not in event mode, hide the line
                if not canvas.event_mode:
                    canvas.event_line_1.visible = False

            if t2 is not None:
                canvas.event_line_2 = canvas._create_event_vertical_line(t2)
                # If not in event mode, hide the line
                if not canvas.event_mode:
                    canvas.event_line_2.visible = False

            logger.debug(f"Restored event markers for {os.path.basename(file_path)}: t1={t1}, t2={t2}")

    # ========== Advanced DSP Analysis Methods ==========

    def _get_selected_annotation_or_warn(self) -> Optional[Annotation]:
        """Get the currently selected annotation, or show a warning if none selected."""
        ann_id = self.selected_annotation_id
        if ann_id is None:
            # Try from table
            ann_id = self.annotation_table.get_selected_annotation_id()

        if ann_id is None:
            QMessageBox.warning(
                self, "No Selection",
                "Please select an annotation first.\n\n"
                "You can:\n"
                "1. Click on an annotation in the spectrogram\n"
                "2. Select one from the annotation table\n"
                "3. Create a new annotation (press 'A' to enter annotation mode)"
            )
            return None

        annotation = self.annotation_manager.get_annotation(ann_id)
        if annotation is None:
            QMessageBox.warning(self, "Error", f"Annotation #{ann_id} not found.")
            return None

        return annotation

    def estimate_selected_annotation_snr(self):
        """Estimate SNR for the currently selected annotation."""
        annotation = self._get_selected_annotation_or_warn()
        if annotation:
            self.estimate_annotation_snr(annotation)

    def detect_harmonics_selected_annotation(self):
        """Detect harmonics in the currently selected annotation."""
        annotation = self._get_selected_annotation_or_warn()
        if annotation:
            self.detect_harmonics_in_annotation(annotation)

    def detect_curved_track_selected_annotation(self):
        """Detect curved track in the currently selected annotation."""
        annotation = self._get_selected_annotation_or_warn()
        if annotation:
            self.detect_curved_tracks_in_annotation(annotation)

    def suppress_selected_annotation_track(self):
        """Suppress track from the currently selected annotation."""
        annotation = self._get_selected_annotation_or_warn()
        if annotation:
            self.suppress_annotation_track(annotation)

    def detect_tracks_in_view(self):
        """Detect curved tracks in the visible spectrogram area.

        This scans the entire visible region for tracks and allows
        the user to create annotations from detected tracks.
        """
        full_data, freqs, times = self._get_spectrogram_axes()

        if full_data is None:
            QMessageBox.warning(self, "No Data", "No spectrogram data available.")
            return

        try:
            # Get visible region from canvas
            if HAS_VISPY and hasattr(self, 'spectrogram_canvas'):
                rect = self.spectrogram_canvas.view.camera.rect
                t_min_view, f_min_view = rect.left, rect.bottom
                t_max_view, f_max_view = rect.right, rect.top
            else:
                # Use full range
                t_min_view, t_max_view = times[0], times[-1]
                f_min_view, f_max_view = freqs[0], freqs[-1]

            # Clamp to data bounds
            t_min_view = max(t_min_view, times[0])
            t_max_view = min(t_max_view, times[-1])
            f_min_view = max(f_min_view, freqs[0])
            f_max_view = min(f_max_view, freqs[-1])

            # Find indices for visible region
            t_start_idx = np.searchsorted(times, t_min_view)
            t_end_idx = np.searchsorted(times, t_max_view)
            f_min_idx = np.searchsorted(freqs, f_min_view)
            f_max_idx = np.searchsorted(freqs, f_max_view)

            # Ensure valid bounds
            t_start_idx = max(0, t_start_idx)
            t_end_idx = min(full_data.shape[1], t_end_idx)
            f_min_idx = max(0, f_min_idx)
            f_max_idx = min(full_data.shape[0], f_max_idx)

            # Extract visible region
            region = full_data[f_min_idx:f_max_idx, t_start_idx:t_end_idx]
            region_times = times[t_start_idx:t_end_idx]
            region_freqs = freqs[f_min_idx:f_max_idx]

            if region.size == 0 or region.shape[0] < 10 or region.shape[1] < 10:
                QMessageBox.warning(self, "Region Too Small",
                    "The visible region is too small for track detection.\n"
                    "Please zoom out or pan to show more of the spectrogram.")
                return

            self.statusBar().showMessage("Detecting curved tracks in view...")
            QApplication.processEvents()

            # Get DSP engine
            dsp_engine = get_dsp_engine()

            # Detect tracks using SIMPLE peak-following algorithm
            # This follows actual intensity maxima at each time column
            # Much more reliable than DP for steep Doppler curves!
            curves = dsp_engine.detect_tracks_simple(
                region, region_times, region_freqs,
                num_tracks=3,
                min_length=max(10, region.shape[1] // 10)
            )

            if not curves:
                QMessageBox.information(
                    self, "No Tracks Found",
                    "No curved tracks were detected in the visible area.\n\n"
                    "Tips:\n"
                    "- Try zooming into a region with a visible signal\n"
                    "- Adjust the contrast/normalization\n"
                    "- The signal may be too weak or noisy"
                )
                self.statusBar().showMessage("No tracks detected")
                return

            # Show results dialog
            self._show_detected_tracks_dialog(curves, region_times, region_freqs)

        except Exception as e:
            logger.error(f"Error detecting tracks in view: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Track detection failed: {e}")

    def _show_detected_tracks_dialog(self, curves: list, times: np.ndarray, freqs: np.ndarray):
        """Show dialog with detected tracks and option to create annotations.

        Args:
            curves: List of DetectedCurve objects
            times: Time axis for the region
            freqs: Frequency axis for the region
        """
        from .qt_compat import QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QDialogButtonBox

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Detected Tracks ({len(curves)} found)")
        dialog.setMinimumSize(500, 400)

        layout = QVBoxLayout(dialog)

        # Info label
        info_label = QLabel(
            f"Found {len(curves)} curved track(s) in the visible area.\n"
            "Select tracks to create annotations from them."
        )
        layout.addWidget(info_label)

        # List of detected tracks
        track_list = QListWidget()
        track_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        for i, curve in enumerate(curves):
            item_text = (
                f"Track {i+1}: {curve.fit_type.capitalize()} | "
                f"Duration: {curve.duration:.2f}s | "
                f"Freq: {curve.freq_range[0]:.0f}-{curve.freq_range[1]:.0f} Hz | "
                f"SNR: {curve.snr_db:.1f} dB | "
                f"Score: {curve.score:.2f}"
            )
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, i)  # Store index
            track_list.addItem(item)

        # Select all by default
        track_list.selectAll()
        layout.addWidget(track_list)

        # Buttons
        button_box = QDialogButtonBox()
        create_btn = button_box.addButton("Create Annotations", QDialogButtonBox.ButtonRole.AcceptRole)
        preview_btn = button_box.addButton("Preview", QDialogButtonBox.ButtonRole.ActionRole)
        cancel_btn = button_box.addButton(QDialogButtonBox.StandardButton.Cancel)

        layout.addWidget(button_box)

        # Store curves for access in handlers
        dialog.curves = curves
        dialog.track_list = track_list

        def preview_tracks():
            """Show preview of detected tracks on spectrogram."""
            selected_indices = [
                item.data(Qt.ItemDataRole.UserRole)
                for item in track_list.selectedItems()
            ]
            if selected_indices and HAS_VISPY and hasattr(self, 'spectrogram_canvas'):
                # Draw preview curves
                for idx in selected_indices:
                    curve = curves[idx]
                    points = np.array(curve.points)
                    if len(points) > 1:
                        self.spectrogram_canvas.draw_preview_curve(points, color='yellow')
                self.statusBar().showMessage(f"Previewing {len(selected_indices)} track(s)")

        def create_annotations():
            """Create annotations from selected tracks."""
            selected_indices = [
                item.data(Qt.ItemDataRole.UserRole)
                for item in track_list.selectedItems()
            ]

            if not selected_indices:
                QMessageBox.warning(dialog, "No Selection", "Please select at least one track.")
                return

            created_count = 0
            for idx in selected_indices:
                curve = curves[idx]

                # Create annotation from curve
                t_start = min(p[0] for p in curve.points)
                t_end = max(p[0] for p in curve.points)
                f_min = min(p[1] for p in curve.points)
                f_max = max(p[1] for p in curve.points)

                # Add some padding
                t_padding = (t_end - t_start) * 0.05
                f_padding = (f_max - f_min) * 0.1

                annotation = self.annotation_manager.create_annotation(
                    t_start=t_start - t_padding,
                    t_end=t_end + t_padding,
                    f_min=f_min - f_padding,
                    f_max=f_max + f_padding,
                    label=f"Track_{curve.fit_type}",
                    color=None
                )

                if annotation:
                    # Set the curve points
                    annotation.points = curve.points
                    annotation.show_doppler_curve = True
                    annotation.snr_db = curve.snr_db
                    annotation.slope_hz_per_sec = curve.curvature * 1000

                    # Add to UI
                    self.annotation_table.add_annotation(annotation)
                    if self.annotation_renderer:
                        self.annotation_renderer.add_annotation(annotation)

                    created_count += 1

            # Save annotations
            self.save_annotations(silent=True)

            self.statusBar().showMessage(f"Created {created_count} annotation(s) from detected tracks")
            dialog.accept()

        preview_btn.clicked.connect(preview_tracks)
        create_btn.clicked.connect(create_annotations)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec()

        # Clear any preview curves
        if HAS_VISPY and hasattr(self, 'spectrogram_canvas'):
            self.spectrogram_canvas.clear_preview_curves()

    def estimate_annotation_snr(self, annotation: Annotation):
        """Estimate SNR for an annotation region using GPU DSP engine.

        Args:
            annotation: The annotation to analyze
        """
        full_data, freqs, times = self._get_spectrogram_axes()

        if full_data is None:
            QMessageBox.warning(self, "No Data", "No spectrogram data available.")
            return

        try:
            # Get DSP engine
            dsp_engine = get_dsp_engine()

            # Find indices for annotation bounds
            t_start_idx = np.searchsorted(times, annotation.t_start)
            t_end_idx = np.searchsorted(times, annotation.t_end)
            f_min_idx = np.searchsorted(freqs, annotation.f_min)
            f_max_idx = np.searchsorted(freqs, annotation.f_max)

            # Ensure valid bounds
            t_start_idx = max(0, t_start_idx)
            t_end_idx = min(full_data.shape[1], t_end_idx)
            f_min_idx = max(0, f_min_idx)
            f_max_idx = min(full_data.shape[0], f_max_idx)

            # Extract region
            region = full_data[f_min_idx:f_max_idx, t_start_idx:t_end_idx]

            if region.size == 0:
                QMessageBox.warning(self, "Invalid Region", "Annotation region is too small.")
                return

            self.statusBar().showMessage("Estimating SNR...")

            # Estimate SNR using DSP engine
            result = dsp_engine.snr_estimator.estimate_snr(region)

            # Store result in annotation
            annotation.snr_db = result.snr_db

            # Update table
            self.annotation_table.update_annotation(annotation)

            # Save annotations
            self.save_annotations(silent=True)

            # Show result dialog
            QMessageBox.information(
                self, "SNR Estimation Result",
                f"Annotation #{annotation.id}\n\n"
                f"SNR: {result.snr_db:.1f} dB\n"
                f"Peak SNR: {result.peak_snr_db:.1f} dB\n"
                f"Signal Power: {result.signal_power:.2e}\n"
                f"Noise Power: {result.noise_power:.2e}\n"
                f"Confidence: {result.confidence:.1%}\n"
                f"Method: {result.method}"
            )

            self.statusBar().showMessage(f"SNR: {result.snr_db:.1f} dB for annotation #{annotation.id}")
            logger.info(f"SNR estimation for annotation {annotation.id}: {result.snr_db:.1f} dB")

        except Exception as e:
            logger.error(f"Error estimating SNR: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"SNR estimation failed: {e}")

    def detect_harmonics_in_annotation(self, annotation: Annotation):
        """Detect harmonics in annotation region using GPU DSP engine.

        Args:
            annotation: The annotation to analyze
        """
        full_data, freqs, times = self._get_spectrogram_axes()

        if full_data is None:
            QMessageBox.warning(self, "No Data", "No spectrogram data available.")
            return

        try:
            # Get DSP engine
            dsp_engine = get_dsp_engine()

            # Find indices for annotation bounds
            t_start_idx = np.searchsorted(times, annotation.t_start)
            t_end_idx = np.searchsorted(times, annotation.t_end)
            f_min_idx = np.searchsorted(freqs, annotation.f_min)
            f_max_idx = np.searchsorted(freqs, annotation.f_max)

            # Ensure valid bounds
            t_start_idx = max(0, t_start_idx)
            t_end_idx = min(full_data.shape[1], t_end_idx)
            f_min_idx = max(0, f_min_idx)
            f_max_idx = min(full_data.shape[0], f_max_idx)

            # Extract region
            region = full_data[f_min_idx:f_max_idx, t_start_idx:t_end_idx]
            region_freqs = freqs[f_min_idx:f_max_idx]

            if region.size == 0:
                QMessageBox.warning(self, "Invalid Region", "Annotation region is too small.")
                return

            self.statusBar().showMessage("Detecting harmonics...")

            # Detect harmonics using DSP engine
            result = dsp_engine.harmonic_analyzer.find_harmonics(region, region_freqs)

            if result is None:
                QMessageBox.information(
                    self, "Harmonic Detection",
                    f"No clear harmonic structure detected in annotation #{annotation.id}."
                )
                return

            # Format harmonics list
            harmonics_text = "\n".join([
                f"  {i+1}. {freq:.1f} Hz (amplitude: {amp:.2f})"
                for i, (freq, amp) in enumerate(result.harmonics[:8])  # Show first 8
            ])

            # Show result dialog
            QMessageBox.information(
                self, "Harmonic Detection Result",
                f"Annotation #{annotation.id}\n\n"
                f"Fundamental Frequency: {result.fundamental_freq:.1f} Hz\n"
                f"Number of Harmonics: {result.num_harmonics}\n"
                f"Harmonic-to-Noise Ratio: {result.hnr_db:.1f} dB\n"
                f"Confidence: {result.confidence:.1%}\n\n"
                f"Detected Harmonics:\n{harmonics_text}"
            )

            self.statusBar().showMessage(
                f"Found {result.num_harmonics} harmonics (f0={result.fundamental_freq:.1f} Hz) "
                f"in annotation #{annotation.id}"
            )
            logger.info(f"Harmonic detection for annotation {annotation.id}: "
                       f"f0={result.fundamental_freq:.1f} Hz, {result.num_harmonics} harmonics")

        except Exception as e:
            logger.error(f"Error detecting harmonics: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Harmonic detection failed: {e}")

    def detect_curved_tracks_in_annotation(self, annotation: Annotation):
        """Detect curved tracks in annotation region using GPU DSP engine.

        Args:
            annotation: The annotation to analyze
        """
        full_data, freqs, times = self._get_spectrogram_axes()

        if full_data is None:
            QMessageBox.warning(self, "No Data", "No spectrogram data available.")
            return

        try:
            # Get DSP engine
            dsp_engine = get_dsp_engine()

            # Find indices for annotation bounds - NO PADDING!
            # We want the track to stay strictly within annotation bounds
            t_start_idx = np.searchsorted(times, annotation.t_start)
            t_end_idx = np.searchsorted(times, annotation.t_end)
            f_min_idx = np.searchsorted(freqs, annotation.f_min)
            f_max_idx = np.searchsorted(freqs, annotation.f_max)

            # Ensure valid bounds (no padding - stay within annotation)
            t_start_idx = max(0, t_start_idx)
            t_end_idx = min(full_data.shape[1], t_end_idx)
            f_min_idx = max(0, f_min_idx)
            f_max_idx = min(full_data.shape[0], f_max_idx)

            # Extract region - EXACTLY the annotation bounds
            region = full_data[f_min_idx:f_max_idx, t_start_idx:t_end_idx]
            region_times = times[t_start_idx:t_end_idx]
            region_freqs = freqs[f_min_idx:f_max_idx]

            if region.size == 0 or region.shape[0] < 10 or region.shape[1] < 10:
                QMessageBox.warning(self, "Invalid Region", "Annotation region is too small for track detection.")
                return

            self.statusBar().showMessage("Detecting curved tracks...")

            # Detect curved tracks using SIMPLE peak-following algorithm
            # Follows actual intensity maxima - works great for Doppler!
            curves = dsp_engine.detect_tracks_simple(
                region, region_times, region_freqs,
                num_tracks=1,  # Just find the best track for annotation
                min_length=max(10, region.shape[1] // 5)
            )

            if not curves:
                QMessageBox.information(
                    self, "Track Detection",
                    f"No curved tracks detected in annotation #{annotation.id}.\n\n"
                    "Try adjusting the annotation bounds or the signal may be too weak."
                )
                return

            # Take the best curve
            best_curve = curves[0]

            # CLIP track points to annotation bounds - ensure track stays inside
            clipped_points = [
                (t, f) for t, f in best_curve.points
                if annotation.t_start <= t <= annotation.t_end
                and annotation.f_min <= f <= annotation.f_max
            ]

            if len(clipped_points) < 4:
                QMessageBox.warning(
                    self, "Track Detection",
                    f"Track was detected but doesn't fit well within annotation bounds.\n"
                    "Try adjusting the annotation to better cover the signal."
                )
                return

            # Convert curve points to annotation format
            annotation.points = clipped_points
            annotation.show_doppler_curve = True

            # Store additional analysis info
            annotation.slope_hz_per_sec = best_curve.curvature * 1000  # Approximate slope
            if hasattr(annotation, 'snr_db') and annotation.snr_db is None:
                annotation.snr_db = best_curve.snr_db

            # Update visual
            if self.annotation_renderer:
                self.annotation_renderer.update_doppler_curve(annotation)

            # Update table
            self.annotation_table.update_annotation(annotation)

            # Save annotations
            self.save_annotations(silent=True)

            # Show result dialog
            QMessageBox.information(
                self, "Track Detection Result",
                f"Annotation #{annotation.id}\n\n"
                f"Track Type: {best_curve.fit_type.capitalize()}\n"
                f"Points Extracted: {len(best_curve.points)}\n"
                f"Duration: {best_curve.duration:.3f} s\n"
                f"Frequency Range: {best_curve.freq_range[0]:.1f} - {best_curve.freq_range[1]:.1f} Hz\n"
                f"Curvature: {best_curve.curvature:.4f}\n"
                f"SNR: {best_curve.snr_db:.1f} dB\n"
                f"Detection Score: {best_curve.score:.3f}\n"
                f"Inflection Points: {len(best_curve.inflection_points)}\n\n"
                f"Track has been applied to the annotation."
            )

            self.statusBar().showMessage(
                f"Detected {best_curve.fit_type} track with {len(best_curve.points)} points "
                f"for annotation #{annotation.id}"
            )
            logger.info(f"Curved track detection for annotation {annotation.id}: "
                       f"{best_curve.fit_type}, {len(best_curve.points)} points, score={best_curve.score:.3f}")

        except Exception as e:
            logger.error(f"Error detecting curved tracks: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Track detection failed: {e}")

    def suppress_annotation_track(self, annotation: Annotation):
        """Suppress/remove the annotation's track from the spectrogram.

        Args:
            annotation: The annotation with track points to suppress
        """
        if not annotation.points or len(annotation.points) < 4:
            QMessageBox.warning(
                self, "No Track",
                "This annotation doesn't have enough track points.\n"
                "Use 'Detect Curved Tracks' or draw a curve first."
            )
            return

        full_data, freqs, times = self._get_spectrogram_axes()

        if full_data is None:
            QMessageBox.warning(self, "No Data", "No spectrogram data available.")
            return

        try:
            # Get DSP engine
            dsp_engine = get_dsp_engine()

            # Convert annotation points to curve object
            from ..core.gpu_dsp_engine import DetectedCurve

            points_arr = np.array(annotation.points)
            curve = DetectedCurve(
                points=annotation.points,
                coefficients=np.polyfit(points_arr[:, 0], points_arr[:, 1], 3),
                fit_type='polynomial',
                degree=3,
                score=1.0,
                snr_db=getattr(annotation, 'snr_db', 0) or 0,
                duration=annotation.t_end - annotation.t_start,
                freq_range=(annotation.f_min, annotation.f_max),
                curvature=0.0,
                inflection_points=[]
            )

            self.statusBar().showMessage("Suppressing track from spectrogram...")

            # Suppress track using DSP engine
            suppressed_data = dsp_engine.track_suppressor.suppress_track(
                full_data.copy(), times, freqs, curve,
                width_hz=50.0,  # Width of suppression band
                method='interpolate'
            )

            # Update the spectrogram display
            if HAS_VISPY and hasattr(self, 'spectrogram_canvas'):
                # Store original data if not already stored
                if not hasattr(self, '_original_spectrogram_data'):
                    self._original_spectrogram_data = full_data.copy()

                # Update cache with suppressed data
                self.spectrogram_cache['data'] = suppressed_data

                # Update display
                self.spectrogram_canvas.update_spectrogram(
                    suppressed_data,
                    times=times,
                    freqs=freqs,
                    keep_view=True
                )

            self.statusBar().showMessage(f"Track suppressed for annotation #{annotation.id}")

            # Ask if user wants to keep the change
            reply = QMessageBox.question(
                self, "Track Suppressed",
                f"Track from annotation #{annotation.id} has been suppressed.\n\n"
                "Do you want to keep this change?\n\n"
                "Click 'Yes' to keep the suppressed spectrogram.\n"
                "Click 'No' to revert to the original.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )

            if reply == QMessageBox.StandardButton.No:
                # Revert to original
                if hasattr(self, '_original_spectrogram_data'):
                    self.spectrogram_cache['data'] = self._original_spectrogram_data
                    self.spectrogram_canvas.update_spectrogram(
                        self._original_spectrogram_data,
                        times=times,
                        freqs=freqs,
                        keep_view=True
                    )
                    self.statusBar().showMessage("Reverted to original spectrogram")
            else:
                # Clear original data reference to save memory
                if hasattr(self, '_original_spectrogram_data'):
                    del self._original_spectrogram_data
                self.statusBar().showMessage("Track suppression applied")

            logger.info(f"Track suppression for annotation {annotation.id} completed")

        except Exception as e:
            logger.error(f"Error suppressing track: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"Track suppression failed: {e}")


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
