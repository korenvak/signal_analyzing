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

        # Colormap selector (placeholder)
        layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(["viridis", "plasma", "inferno", "magma", "cividis", "turbo"])
        layout.addWidget(self.colormap_combo)

        # Normalization (placeholder)
        layout.addWidget(QLabel("Normalization:"))
        self.norm_combo = QComboBox()
        self.norm_combo.addItems(["MinMax", "STD (Adaptive)", "Percentile"])
        layout.addWidget(self.norm_combo)

        layout.addStretch()

        self.sidebar_tabs.addTab(display_tab, "3. Display")

    def _setup_main_display(self):
        """Setup the main display area."""
        display_container = QFrame()
        display_container.setObjectName("das_display_container")
        display_layout = QVBoxLayout(display_container)
        display_layout.setContentsMargins(0, 0, 0, 0)
        display_layout.setSpacing(0)

        # Placeholder for waterfall canvas
        # Will be replaced with actual WaterfallCanvas once implemented
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
        display_layout.addWidget(self.display_placeholder)

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

        # TODO: Implement actual waterfall computation
        # For now, show a message
        try:
            # Get data chunk from provider
            data = self._provider.get_data_chunk(request)
            self._current_waterfall_data = data

            # Update display placeholder with info
            self.display_placeholder.setText(
                f"Waterfall Data Ready\n\n"
                f"Shape: {data.shape[0]} time samples × {data.shape[1]} sensors\n"
                f"Data range: [{data.min():.3f}, {data.max():.3f}]\n\n"
                f"(Waterfall canvas implementation pending)"
            )

            self.status_label.setText(f"Computed: {data.shape}")
            self.waterfall_computed.emit(data)

        except Exception as e:
            logger.error(f"Waterfall computation failed: {e}")
            QMessageBox.critical(self, "Computation Error", str(e))
            self.status_label.setText(f"Error: {e}")

    # ==================== View Mode ====================

    def _on_view_mode_changed(self, index: int):
        """Handle view mode change."""
        if index == 0:
            self._view_mode = 'waterfall'
            self.sensor_select_label.hide()
            self.sensor_combo.hide()
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

            # Update display placeholder
            self.display_placeholder.setText(
                f"Single Sensor Spectrogram\n\n"
                f"Sensor ID: {sensor_id}\n\n"
                f"(Spectrogram computation pending)"
            )

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
