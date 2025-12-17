"""
Folder selector widget for DAS data.

Allows selecting a folder and metadata JSON file,
then displays available sensor and time ranges.
"""

import json
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

from .qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QGroupBox, QTextEdit, QFrame,
    Signal
)

from ..core.das_data_provider import (
    DASDataProvider, DASFolderMetadata, MockDASDataProvider
)
from ..core.file_das_provider import FileDASProvider
from ..core.time_formatter import TimeFormatter

logger = logging.getLogger(__name__)


class FolderSelectorWidget(QWidget):
    """
    Widget for selecting DAS data folder and metadata file.

    Displays:
    - Folder path input with browse button
    - Metadata file input with browse button
    - Available data information (sensors, time ranges)
    - Load button
    """

    # Signals
    folder_loaded = Signal(object, object)  # (DASFolderMetadata, DASDataProvider)
    load_error = Signal(str)  # Error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._provider: Optional[DASDataProvider] = None
        self._setup_ui()

    def _setup_ui(self):
        """Setup the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # Folder selection group
        folder_group = QGroupBox("Data Folder")
        folder_layout = QVBoxLayout(folder_group)

        # Folder path
        folder_path_layout = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Select folder containing DAS data...")
        folder_path_layout.addWidget(self.folder_input)

        self.browse_folder_btn = QPushButton("Browse")
        self.browse_folder_btn.clicked.connect(self._browse_folder)
        folder_path_layout.addWidget(self.browse_folder_btn)

        folder_layout.addLayout(folder_path_layout)

        # Metadata file
        meta_layout = QHBoxLayout()
        meta_layout.addWidget(QLabel("Metadata:"))
        self.meta_input = QLineEdit()
        self.meta_input.setPlaceholderText("metadata.json")
        meta_layout.addWidget(self.meta_input)

        self.browse_meta_btn = QPushButton("...")
        self.browse_meta_btn.setMaximumWidth(30)
        self.browse_meta_btn.clicked.connect(self._browse_metadata)
        meta_layout.addWidget(self.browse_meta_btn)

        folder_layout.addLayout(meta_layout)

        layout.addWidget(folder_group)

        # Information display
        info_group = QGroupBox("Available Data")
        info_layout = QVBoxLayout(info_group)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        self.info_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e2e;
                color: #ccc;
                font-family: monospace;
                font-size: 11px;
                border: 1px solid #333;
                border-radius: 4px;
            }
        """)
        self.info_text.setPlainText("No data loaded.\n\nSelect a folder and click 'Load Metadata'.")
        info_layout.addWidget(self.info_text)

        layout.addWidget(info_group)

        # Load button
        btn_layout = QHBoxLayout()

        self.load_btn = QPushButton("Load Metadata")
        self.load_btn.clicked.connect(self._load_metadata)
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a9eff;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a8eef;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """)
        btn_layout.addWidget(self.load_btn)

        # Mock data button (for testing)
        self.mock_btn = QPushButton("Load Mock Data")
        self.mock_btn.clicked.connect(self._load_mock_data)
        self.mock_btn.setStyleSheet("""
            QPushButton {
                background-color: #666;
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #777;
            }
        """)
        btn_layout.addWidget(self.mock_btn)

        layout.addLayout(btn_layout)

        layout.addStretch()

    def _browse_folder(self):
        """Browse for data folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select DAS Data Folder",
            str(Path.home())
        )
        if folder:
            self.folder_input.setText(folder)

            # Try to find metadata.json automatically
            meta_path = Path(folder) / "metadata.json"
            if meta_path.exists():
                self.meta_input.setText("metadata.json")

    def _browse_metadata(self):
        """Browse for metadata file."""
        folder = self.folder_input.text() or str(Path.home())
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Metadata File",
            folder,
            "JSON Files (*.json);;All Files (*)"
        )
        if filepath:
            # Store relative path if in same folder
            folder_path = Path(self.folder_input.text())
            file_path = Path(filepath)
            try:
                rel_path = file_path.relative_to(folder_path)
                self.meta_input.setText(str(rel_path))
            except ValueError:
                self.meta_input.setText(filepath)

    def _load_metadata(self):
        """Load metadata from selected folder using FileDASProvider."""
        folder = self.folder_input.text().strip()
        meta_file = self.meta_input.text().strip() or "metadata.json"

        if not folder:
            self.load_error.emit("Please select a data folder.")
            return

        folder_path = Path(folder)
        if not folder_path.exists():
            self.load_error.emit(f"Folder not found: {folder}")
            return

        # Resolve metadata path
        meta_path = folder_path / meta_file
        if not meta_path.exists():
            self.load_error.emit(f"Metadata file not found: {meta_path}")
            return

        try:
            self.info_text.setPlainText("Loading data folder...")

            # Use FileDASProvider to load the data
            provider = FileDASProvider()
            metadata = provider.load_folder(str(folder_path), meta_file)
            self._provider = provider

            # Display metadata info
            info_lines = [
                "DATA LOADED SUCCESSFULLY",
                "=" * 30,
                f"",
                f"Sensors: {metadata.sensor_range[0]} - {metadata.sensor_range[1]}",
                f"Total sensors: {metadata.n_sensors}",
                f"Sample rate: {metadata.sample_rate:.0f} Hz",
                f"Duration: {metadata.total_duration_seconds:.1f}s",
                f"Files: {len(metadata.files)}",
            ]

            if metadata.sensor_spacing:
                info_lines.append(f"Sensor spacing: {metadata.sensor_spacing} m")

            info_lines.append(f"Units: {metadata.units}")
            info_lines.append(f"")
            info_lines.append("Time ranges:")

            formatter = TimeFormatter(sample_rate=metadata.sample_rate)
            for i, (start, end) in enumerate(metadata.time_ranges):
                duration = (end - start).total_seconds()
                info_lines.append(
                    f"  {i+1}. {start.strftime('%H:%M:%S')} - {end.strftime('%H:%M:%S')} "
                    f"({formatter.format_duration(duration)})"
                )

            if len(metadata.time_ranges) > 1:
                info_lines.append("")
                info_lines.append("Note: Gap between ranges indicates missing data")

            # Show file sizes
            info_lines.append("")
            info_lines.append("Data files:")
            for fi in metadata.files:
                size_mb = fi.file_size_bytes / 1024 / 1024
                info_lines.append(f"  - {fi.filename}: {size_mb:.1f} MB")

            self.info_text.setPlainText("\n".join(info_lines))

            # Emit success signal
            self.folder_loaded.emit(metadata, provider)

            logger.info(f"DAS data loaded from {folder_path}")

        except json.JSONDecodeError as e:
            self.load_error.emit(f"Invalid JSON: {e}")
        except FileNotFoundError as e:
            self.load_error.emit(f"File not found: {e}")
        except Exception as e:
            self.load_error.emit(f"Error loading data: {e}")
            logger.error(f"Data load error: {e}")

    def _load_mock_data(self):
        """Load mock data for testing the UI."""
        try:
            mock_provider = MockDASDataProvider()
            metadata = mock_provider.load_folder("mock_folder", "mock_metadata.json")
            self._provider = mock_provider

            # Display metadata info
            info_lines = [
                "MOCK DATA LOADED",
                "=" * 30,
                f"",
                f"Sensors: {metadata.sensor_range[0]} - {metadata.sensor_range[1]}",
                f"Total sensors: {metadata.n_sensors}",
                f"Sample rate: {metadata.sample_rate:.0f} Hz",
                f"Sensor spacing: {metadata.sensor_spacing} m",
                f"Units: {metadata.units}",
                f"",
                "Time ranges:"
            ]

            formatter = TimeFormatter(sample_rate=metadata.sample_rate)
            for i, (start, end) in enumerate(metadata.time_ranges):
                duration = (end - start).total_seconds()
                info_lines.append(
                    f"  {i+1}. {start.strftime('%H:%M:%S')} - {end.strftime('%H:%M:%S')} "
                    f"({formatter.format_duration(duration)})"
                )

            if len(metadata.time_ranges) > 1:
                info_lines.append("")
                info_lines.append("Note: Gap between ranges indicates missing data")

            self.info_text.setPlainText("\n".join(info_lines))

            # Emit success signal
            self.folder_loaded.emit(metadata, mock_provider)

            logger.info("Mock DAS data loaded successfully")

        except Exception as e:
            self.load_error.emit(f"Error loading mock data: {e}")
            logger.error(f"Mock data load error: {e}")

    def set_provider_factory(self, factory):
        """
        Set a factory function for creating DASDataProvider instances.

        The factory should accept (folder_path, metadata_path) and return
        a DASDataProvider instance.

        This allows users to plug in their own data providers.
        """
        self._provider_factory = factory
