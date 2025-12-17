"""
DAS (Distributed Acoustic Sensing) Multi-Channel Tab.

This tab provides:
1. Folder/metadata selection (sidebar)
2. Sensor/time range selection (sidebar)
3. Waterfall visualization (main area)
4. Single-sensor spectrogram view (switchable)
"""

import logging
from typing import Optional, Dict, Any, Callable
from pathlib import Path

from .qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QFrame,
    QTabWidget, QLabel, QPushButton, QMessageBox, QComboBox,
    Qt, QSizePolicy, Signal
)

from ..core.das_data_provider import (
    DASDataProvider, DASFolderMetadata, DASDataRequest,
    MockDASDataProvider, get_das_provider, set_das_provider
)
from ..core.time_formatter import TimeFormatter, AbsoluteTimeFormatter

logger = logging.getLogger(__name__)


class DASTab(QWidget):
    """
    Main container for the DAS multi-channel visualization system.

    Layout:
    ┌────────────────┬──────────────────────────────────────────┐
    │    SIDEBAR     │                                          │
    │   ┌──────────┐ │                                          │
    │   │ 1.Folder │ │           MAIN DISPLAY AREA              │
    │   │  Select  │ │                                          │
    │   ├──────────┤ │   (Waterfall or Single-Sensor Spec)      │
    │   │ 2.Range  │ │                                          │
    │   │  Select  │ │                                          │
    │   ├──────────┤ │                                          │
    │   │ 3.Display│ │                                          │
    │   │  Options │ │                                          │
    │   └──────────┘ │                                          │
    └────────────────┴──────────────────────────────────────────┘
    """

    # Signals
    waterfall_computed = Signal(object)  # Emits the waterfall data
    sensor_selected = Signal(int)        # Emits selected sensor ID for spectrogram
    status_message = Signal(str)         # Status updates

    def __init__(self, parent=None):
        super().__init__(parent)

        # Data state
        self._provider: Optional[DASDataProvider] = None
        self._metadata: Optional[DASFolderMetadata] = None
        self._time_formatter: Optional[TimeFormatter] = None
        self._current_waterfall_data = None

        # View mode
        self._view_mode = 'waterfall'  # 'waterfall' or 'spectrogram'
        self._selected_sensor: Optional[int] = None

        # Setup UI
        self._setup_ui()

        logger.info("DASTab initialized")

    def _setup_ui(self):
        """Setup the main UI layout."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create splitter for sidebar and main area
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)

        # Create sidebar
        self._setup_sidebar()

        # Create main display area
        self._setup_main_display()

        # Set splitter sizes
        self.splitter.setSizes([280, 1000])

    def _setup_sidebar(self):
        """Setup the sidebar with workflow tabs."""
        sidebar = QFrame()
        sidebar.setObjectName("das_sidebar")
        sidebar.setMinimumWidth(250)
        sidebar.setMaximumWidth(350)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(8)

        # Title
        title = QLabel("DAS Multi-Channel")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        sidebar_layout.addWidget(title)

        # Tab widget for workflow steps
        self.sidebar_tabs = QTabWidget()
        self.sidebar_tabs.setDocumentMode(True)
        sidebar_layout.addWidget(self.sidebar_tabs)

        # Tab 1: Folder Selection
        self._setup_folder_tab()

        # Tab 2: Range Selection
        self._setup_range_tab()

        # Tab 3: Display Options
        self._setup_display_tab()

        # Status area at bottom
        self.status_label = QLabel("No data loaded")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        self.status_label.setWordWrap(True)
        sidebar_layout.addWidget(self.status_label)

        self.splitter.addWidget(sidebar)

    def _setup_folder_tab(self):
        """Setup the folder selection tab."""
        from .folder_selector import FolderSelectorWidget

        self.folder_selector = FolderSelectorWidget()
        self.folder_selector.folder_loaded.connect(self._on_folder_loaded)
        self.folder_selector.load_error.connect(self._on_load_error)

        self.sidebar_tabs.addTab(self.folder_selector, "1. Data")

    def _setup_range_tab(self):
        """Setup the range selection tab."""
        from .range_selector import RangeSelectorWidget

        self.range_selector = RangeSelectorWidget()
        self.range_selector.compute_requested.connect(self._on_compute_requested)
        self.range_selector.setEnabled(False)  # Disabled until data loaded

        self.sidebar_tabs.addTab(self.range_selector, "2. Range")

    def _setup_display_tab(self):
        """Setup the display options tab."""
        display_tab = QWidget()
        layout = QVBoxLayout(display_tab)
        layout.setContentsMargins(8, 8, 8, 8)

        # View mode selector
        layout.addWidget(QLabel("View Mode:"))
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Waterfall", "Single Sensor Spectrogram"])
        self.view_mode_combo.currentIndexChanged.connect(self._on_view_mode_changed)
        layout.addWidget(self.view_mode_combo)

        # Sensor selector (for spectrogram mode)
        self.sensor_select_label = QLabel("Sensor ID:")
        layout.addWidget(self.sensor_select_label)

        self.sensor_combo = QComboBox()
        self.sensor_combo.setEnabled(False)
        self.sensor_combo.currentIndexChanged.connect(self._on_sensor_selected)
        layout.addWidget(self.sensor_combo)

        # Initially hide sensor selector
        self.sensor_select_label.hide()
        self.sensor_combo.hide()

        # Colormap selector
        layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis", "turbo"])
        self.colormap_combo.currentIndexChanged.connect(self._on_colormap_changed)
        layout.addWidget(self.colormap_combo)

        # Normalization
        layout.addWidget(QLabel("Normalization:"))
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["MinMax", "STD (Adaptive)", "Percentile"])
        self.norm_combo.setCurrentIndex(1)  # Default to STD (most robust for DAS data)
        self.norm_combo.currentIndexChanged.connect(self._on_normalization_changed)
        layout.addWidget(self.norm_combo)

        layout.addStretch()

        self.sidebar_tabs.addTab(display_tab, "3. Display")

    def _setup_main_display(self):
        """Setup the main display area."""
        from .qt_compat import QStackedWidget

        display_container = QFrame()
        display_container.setObjectName("das_display_container")
        display_layout = QVBoxLayout(display_container)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(0)

        # Stacked widget for switching between waterfall and spectrogram views
        self.display_stack = QStackedWidget()
        display_layout.addWidget(self.display_stack)

        # Page 0: Placeholder (initial state)
        self.display_placeholder = QLabel(
            "DAS Waterfall Display\n\n"
            "1. Select a data folder in the 'Data' tab\n"
            "2. Choose sensor and time ranges in the 'Range' tab\n"
            "3. Click 'Compute Waterfall' to visualize"
        )
        self.display_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display_placeholder.setStyleSheet("""
            QLabel {
                background-color: #1a1a2e;
                color: #666;
                font-size: 14px;
                border: 2px dashed #333;
                border-radius: 8px;
                padding: 40px;
            }
        """)
        self.display_stack.addWidget(self.display_placeholder)

        # Page 1: Waterfall canvas
        self._waterfall_canvas = None
        self._waterfall_container = QFrame()
        waterfall_layout = QVBoxLayout(self._waterfall_container)
        waterfall_layout.setContentsMargins(0, 0, 0, 0)

        try:
            from .waterfall_canvas import WaterfallCanvas
            self._waterfall_canvas = WaterfallCanvas()
            self._waterfall_canvas.on_cursor_moved_callback = self._on_waterfall_cursor_moved
            waterfall_layout.addWidget(self._waterfall_canvas.native)
            logger.info("WaterfallCanvas created successfully")
        except ImportError as e:
            logger.warning(f"Could not create WaterfallCanvas: {e}")
            fallback = QLabel("Waterfall visualization requires VisPy")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            waterfall_layout.addWidget(fallback)

        self.display_stack.addWidget(self._waterfall_container)

        # Page 2: Single-sensor spectrogram (reuse existing VisPyCanvas)
        self._spectrogram_container = QFrame()
        spec_layout = QVBoxLayout(self._spectrogram_container)
        spec_layout.setContentsMargins(0, 0, 0, 0)

        try:
            from .vispy_canvas import VisPyCanvas
            self._sensor_spectrogram_canvas = VisPyCanvas('spectrogram')
            self._sensor_spectrogram_canvas.native.setMinimumSize(400, 300)
            spec_layout.addWidget(self._sensor_spectrogram_canvas.native)
            logger.info("Single-sensor spectrogram canvas created")
        except ImportError as e:
            logger.warning(f"Could not create spectrogram canvas: {e}")
            fallback = QLabel("Spectrogram visualization requires VisPy")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            spec_layout.addWidget(fallback)
            self._sensor_spectrogram_canvas = None

        self.display_stack.addWidget(self._spectrogram_container)

        # Info bar at bottom
        self.info_bar = QLabel("")
        self.info_bar.setStyleSheet("""
            QLabel {
                background-color: rgba(30, 30, 46, 0.9);
                color: #aaa;
                padding: 4px 8px;
                font-size: 11px;
                font-family: monospace;
            }
        """)
        display_layout.addWidget(self.info_bar)

        self.splitter.addWidget(display_container)

    # ==================== Data Loading ====================

    def _on_folder_loaded(self, metadata: DASFolderMetadata, provider: DASDataProvider):
        """Handle successful folder load."""
        self._metadata = metadata
        self._provider = provider
        set_das_provider(provider)

        # Setup time formatter
        if metadata.time_ranges:
            base_time = metadata.time_ranges[0][0]
            self._time_formatter = AbsoluteTimeFormatter(base_time, metadata.sample_rate)
        else:
            self._time_formatter = TimeFormatter(sample_rate=metadata.sample_rate)

        # Update range selector with available ranges
        self.range_selector.set_metadata(metadata)
        self.range_selector.setEnabled(True)

        # Update sensor combo for spectrogram mode
        self.sensor_combo.clear()
        for i in range(metadata.sensor_range[0], min(metadata.sensor_range[1], metadata.sensor_range[0] + 100)):
            self.sensor_combo.addItem(f"Sensor {i}", i)
        self.sensor_combo.setEnabled(True)

        # Update status
        self.status_label.setText(
            f"Loaded: {metadata.n_sensors} sensors\n"
            f"Duration: {metadata.total_duration_seconds:.1f}s\n"
            f"Sample rate: {metadata.sample_rate:.0f} Hz"
        )

        # Switch to range selection tab
        self.sidebar_tabs.setCurrentIndex(1)

        self.status_message.emit(f"Loaded DAS data: {metadata.n_sensors} sensors")
        logger.info(f"DAS folder loaded: {metadata.folder_path}")

    def _on_load_error(self, error_msg: str):
        """Handle folder load error."""
        self.status_label.setText(f"Error: {error_msg}")
        self.status_message.emit(f"DAS load error: {error_msg}")
        logger.error(f"DAS load error: {error_msg}")

    # ==================== Computation ====================

    def _on_compute_requested(self, request: DASDataRequest):
        """Handle compute waterfall request."""
        if self._provider is None:
            QMessageBox.warning(self, "No Data", "Please load a data folder first.")
            return

        self.status_label.setText("Computing waterfall...")
        self.status_message.emit("Computing waterfall...")

        try:
            # Get data chunk from provider
            data = self._provider.get_data_chunk(request)
            self._current_waterfall_data = data
            self._current_request = request

            # Store extent information
            self._current_sensor_start = request.sensor_start
            self._current_sensor_end = request.sensor_end
            self._current_time_start = request.time_start
            self._current_time_end = request.time_end

            # Update waterfall display
            self._update_waterfall_display()

            self.status_label.setText(
                f"Computed: {data.shape[0]} × {data.shape[1]} "
                f"({data.shape[0] * data.shape[1] * 4 / 1024 / 1024:.1f} MB)"
            )
            self.waterfall_computed.emit(data)

            # Switch to display tab
            self.sidebar_tabs.setCurrentIndex(2)

        except Exception as e:
            logger.error(f"Waterfall computation failed: {e}")
            QMessageBox.critical(self, "Computation Error", str(e))
            self.status_label.setText(f"Error: {e}")

    def _update_waterfall_display(self):
        """Update waterfall canvas with current data."""
        if self._current_waterfall_data is None:
            return

        if self._waterfall_canvas is not None:
            # Convert datetime to sample indices for display
            # time_start/time_end in set_data are sample indices, not datetime
            n_time_samples = self._current_waterfall_data.shape[0]
            sensor_start = getattr(self, '_current_sensor_start', 0)
            sensor_end = getattr(self, '_current_sensor_end', None)
            if sensor_end is None:
                sensor_end = sensor_start + self._current_waterfall_data.shape[1]

            # Set data on canvas with sample indices (0 to n_samples)
            self._waterfall_canvas.set_data(
                data=self._current_waterfall_data,
                sensor_start=sensor_start,
                sensor_end=sensor_end,
                time_start=0,
                time_end=n_time_samples,
                time_formatter=self._time_formatter
            )

            # Apply current colormap
            colormap = self.colormap_combo.currentText()
            self._waterfall_canvas.set_colormap(colormap)

            # Switch to waterfall view
            self.display_stack.setCurrentIndex(1)
            logger.info("Waterfall display updated")
        else:
            # Fallback to placeholder
            data = self._current_waterfall_data
            self.display_placeholder.setText(
                f"Waterfall Data Ready\n\n"
                f"Shape: {data.shape[0]} time samples × {data.shape[1]} sensors\n"
                f"Data range: [{data.min():.3f}, {data.max():.3f}]\n\n"
                f"(Waterfall canvas not available)"
            )
            self.display_stack.setCurrentIndex(0)

    # ==================== View Mode ====================

    def _on_view_mode_changed(self, index: int):
        """Handle view mode change."""
        if index == 0:
            self._view_mode = 'waterfall'
            self.sensor_select_label.hide()
            self.sensor_combo.hide()

            # Switch to waterfall view if data exists
            if self._current_waterfall_data is not None and self._waterfall_canvas is not None:
                self.display_stack.setCurrentIndex(1)  # Waterfall canvas
            else:
                self.display_stack.setCurrentIndex(0)  # Placeholder
        else:
            self._view_mode = 'spectrogram'
            self.sensor_select_label.show()
            self.sensor_combo.show()

            # Trigger sensor selection if available
            if self.sensor_combo.currentIndex() >= 0:
                self._on_sensor_selected(self.sensor_combo.currentIndex())

        logger.info(f"View mode changed to: {self._view_mode}")

    def _on_sensor_selected(self, index: int):
        """Handle sensor selection for spectrogram view."""
        if index < 0:
            return

        sensor_id = self.sensor_combo.itemData(index)
        self._selected_sensor = sensor_id

        if self._view_mode == 'spectrogram':
            self.sensor_selected.emit(sensor_id)
            self.status_message.emit(f"Selected sensor {sensor_id} for spectrogram")

            # Compute and display single-sensor spectrogram
            self._compute_sensor_spectrogram(sensor_id)

    def _compute_sensor_spectrogram(self, sensor_id: int):
        """
        Compute and display spectrogram for a single sensor.

        Args:
            sensor_id: The sensor ID to display spectrogram for
        """
        if self._current_waterfall_data is None:
            self.display_placeholder.setText(
                f"Single Sensor Spectrogram\n\n"
                f"Sensor ID: {sensor_id}\n\n"
                f"(No waterfall data loaded - compute waterfall first)"
            )
            self.display_stack.setCurrentIndex(0)
            return

        try:
            # Get sensor column from current data
            sensor_start = getattr(self, '_current_sensor_start', 0)
            local_sensor_idx = sensor_id - sensor_start

            if local_sensor_idx < 0 or local_sensor_idx >= self._current_waterfall_data.shape[1]:
                self.display_placeholder.setText(
                    f"Sensor {sensor_id} not in current range\n\n"
                    f"Current range: {sensor_start} - {sensor_start + self._current_waterfall_data.shape[1]}"
                )
                self.display_stack.setCurrentIndex(0)
                return

            # Extract single sensor time series
            sensor_data = self._current_waterfall_data[:, local_sensor_idx]

            # Compute spectrogram using waterfall engine
            from ..engines.waterfall_engine import get_waterfall_engine
            engine = get_waterfall_engine()

            sample_rate = self._metadata.sample_rate if self._metadata else 1000.0
            spectrogram, freqs, times = engine.process_for_spectrogram(
                sensor_data,
                sample_rate=sample_rate,
                fft_size=1024,
                hop_length=256
            )

            # Display in spectrogram canvas
            if self._sensor_spectrogram_canvas is not None:
                # Spectrogram is (freq, time) - need to display with time on x-axis, freq on y-axis
                # update_image expects extent as (time_start, time_end, freq_start, freq_end)
                time_max = times[-1] if len(times) > 0 else 1.0
                extent = (0, time_max, freqs[0], freqs[-1])

                self._sensor_spectrogram_canvas.update_image(
                    spectrogram.T,  # Transpose: (freq, time) -> (time, freq) for display
                    extent=extent,
                    preserve_view=False
                )
                # Set data bounds for proper zoom behavior
                self._sensor_spectrogram_canvas.set_data_bounds(0, time_max, freqs[0], freqs[-1])
                self._sensor_spectrogram_canvas.reset_camera_to_data_bounds()

                self.display_stack.setCurrentIndex(2)  # Spectrogram canvas
                self.info_bar.setText(f"Sensor {sensor_id} spectrogram | {len(freqs)} freq bins | {len(times)} time frames")
                logger.info(f"Displayed spectrogram for sensor {sensor_id}")
            else:
                # Fallback
                self.display_placeholder.setText(
                    f"Single Sensor Spectrogram\n\n"
                    f"Sensor ID: {sensor_id}\n"
                    f"Shape: {spectrogram.shape}\n"
                    f"Freq range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz\n\n"
                    f"(Spectrogram canvas not available)"
                )
                self.display_stack.setCurrentIndex(0)

        except Exception as e:
            logger.error(f"Failed to compute sensor spectrogram: {e}")
            self.display_placeholder.setText(
                f"Error computing spectrogram\n\n"
                f"Sensor ID: {sensor_id}\n"
                f"Error: {str(e)}"
            )
            self.display_stack.setCurrentIndex(0)

    # ==================== Public API ====================

    def set_provider(self, provider: DASDataProvider):
        """Set the data provider programmatically."""
        self._provider = provider
        set_das_provider(provider)

    def get_current_data(self):
        """Get currently computed waterfall data."""
        return self._current_waterfall_data

    def get_metadata(self) -> Optional[DASFolderMetadata]:
        """Get current metadata."""
        return self._metadata

    def get_time_formatter(self) -> Optional[TimeFormatter]:
        """Get current time formatter."""
        return self._time_formatter

    def load_mock_data(self):
        """Load mock data for testing."""
        mock_provider = MockDASDataProvider()
        metadata = mock_provider.load_folder("mock_folder", "mock_metadata.json")
        self._on_folder_loaded(metadata, mock_provider)

    # ==================== Waterfall Callbacks ====================

    def _on_waterfall_cursor_moved(self, info: Dict[str, Any]):
        """
        Handle cursor movement over the waterfall canvas.

        Updates the info bar with sensor ID, time, and value.

        Args:
            info: Dictionary with 'sensor', 'time_idx', 'time_str', 'value'
        """
        sensor = info.get('sensor', 0)
        time_str = info.get('time_str', '')
        value = info.get('value')

        if value is not None:
            self.info_bar.setText(
                f"Sensor: {sensor}  |  Time: {time_str}  |  Value: {value:.4f}"
            )
        else:
            self.info_bar.setText(f"Sensor: {sensor}  |  Time: {time_str}")

    def _on_colormap_changed(self, index: int):
        """Handle colormap selection change."""
        colormap = self.colormap_combo.currentText()
        if self._waterfall_canvas is not None:
            self._waterfall_canvas.set_colormap(colormap)
        logger.debug(f"Colormap changed to: {colormap}")

    def _on_normalization_changed(self, index: int):
        """Handle normalization mode change."""
        from ..engines.waterfall_engine import NormalizationMode, WaterfallParams

        mode_map = {
            0: NormalizationMode.MINMAX,
            1: NormalizationMode.STD,
            2: NormalizationMode.PERCENTILE,
        }
        mode = mode_map.get(index, NormalizationMode.STD)

        # If we have current data, reprocess it
        if self._current_waterfall_data is not None and self._waterfall_canvas is not None:
            from ..engines.waterfall_engine import get_waterfall_engine
            engine = get_waterfall_engine()
            engine.params.normalization = mode

            # Reprocess and update canvas
            self._update_waterfall_display()

        logger.debug(f"Normalization changed to: {mode.value}")
