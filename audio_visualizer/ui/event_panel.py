"""
Event panel widget containing event controls and table.

Provides:
- Session controls (New, Continue)
- Event mode toggle
- Event table
- Status display
"""
import logging
from pathlib import Path
from typing import Optional

from .qt_compat import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QFileDialog, QMessageBox,
    QSplitter, QFrame, Qt, Signal
)

from .event_table import EventTableWidget
from .event_manager import EventManager
from .event_data import TaggedEvent

logger = logging.getLogger(__name__)


class EventPanel(QWidget):
    """Panel containing event tagging controls and event table."""

    # Signals
    event_mode_toggled = Signal(bool)       # Emitted when event mode is toggled
    session_changed = Signal(bool)           # Emitted when session state changes (active/inactive)
    goto_event_requested = Signal(float, float)  # Emitted to navigate to (t_center, f_center)

    def __init__(self, event_manager: EventManager, parent=None):
        """Initialize the event panel.

        Args:
            event_manager: EventManager instance to use
            parent: Parent widget
        """
        super().__init__(parent)
        self.event_manager = event_manager
        self._event_mode_active = False

        self._setup_ui()
        self._connect_signals()
        self._update_session_status()

    def _setup_ui(self):
        """Setup the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # =====================================================================
        # Session Controls
        # =====================================================================
        session_group = QGroupBox("Event Session")
        session_layout = QVBoxLayout(session_group)
        session_layout.setSpacing(6)

        # Session status
        self.session_label = QLabel("No session active")
        self.session_label.setStyleSheet("font-weight: bold;")
        session_layout.addWidget(self.session_label)

        # Session buttons
        btn_layout = QHBoxLayout()

        self.new_session_btn = QPushButton("New Session...")
        self.new_session_btn.setToolTip("Create a new event tagging session")
        btn_layout.addWidget(self.new_session_btn)

        self.load_session_btn = QPushButton("Continue Session...")
        self.load_session_btn.setToolTip("Continue an existing session from CSV file")
        btn_layout.addWidget(self.load_session_btn)

        session_layout.addLayout(btn_layout)

        layout.addWidget(session_group)

        # =====================================================================
        # Event Mode Toggle
        # =====================================================================
        mode_layout = QHBoxLayout()

        self.mode_btn = QPushButton("Event Mode (E)")
        self.mode_btn.setCheckable(True)
        self.mode_btn.setToolTip("Toggle event tagging mode.\n"
                                 "Click twice on spectrogram to mark event start/end.\n"
                                 "Shortcut: E")
        self.mode_btn.setMinimumHeight(40)
        self.mode_btn.setStyleSheet("""
            QPushButton {
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:checked {
                background-color: #4CAF50;
                color: white;
            }
        """)
        mode_layout.addWidget(self.mode_btn)

        layout.addLayout(mode_layout)

        # =====================================================================
        # Event Counter
        # =====================================================================
        counter_layout = QHBoxLayout()

        self.event_count_label = QLabel("Events: 0")
        self.event_count_label.setStyleSheet("font-size: 11px; color: #666;")
        counter_layout.addWidget(self.event_count_label)

        counter_layout.addStretch()

        layout.addLayout(counter_layout)

        # =====================================================================
        # Event Table
        # =====================================================================
        self.event_table = EventTableWidget()
        layout.addWidget(self.event_table, stretch=1)

    def _connect_signals(self):
        """Connect internal signals."""
        self.new_session_btn.clicked.connect(self._on_new_session)
        self.load_session_btn.clicked.connect(self._on_load_session)
        self.mode_btn.clicked.connect(self._on_mode_toggle)

        # Table signals
        self.event_table.event_deleted.connect(self._on_event_deleted)
        self.event_table.event_goto_requested.connect(self._on_goto_event)

    def _update_session_status(self):
        """Update the session status display."""
        if self.event_manager.session_active:
            csv_path = self.event_manager.csv_path
            if csv_path:
                self.session_label.setText(f"Session: {csv_path.name}")
                self.session_label.setToolTip(str(csv_path))
            self.mode_btn.setEnabled(True)
        else:
            self.session_label.setText("No session active")
            self.session_label.setToolTip("")
            self.mode_btn.setEnabled(False)
            self.mode_btn.setChecked(False)
            self._event_mode_active = False

        self._update_event_count()

    def _update_event_count(self):
        """Update the event count label."""
        count = self.event_manager.event_count
        self.event_count_label.setText(f"Events: {count}")

    # =========================================================================
    # Session Management
    # =========================================================================

    def _on_new_session(self):
        """Handle new session button click."""
        # Ask user for session directory
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Session Directory",
            "",
            QFileDialog.ShowDirsOnly
        )

        if not dir_path:
            return

        session_dir = Path(dir_path)

        # Check if CSV already exists
        csv_path = session_dir / "events.csv"
        if csv_path.exists():
            result = QMessageBox.question(
                self,
                "Session Exists",
                f"A session already exists in this directory:\n{csv_path}\n\n"
                "Do you want to continue this existing session instead?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )

            if result == QMessageBox.Cancel:
                return
            elif result == QMessageBox.Yes:
                # Load existing
                if self.event_manager.load_existing_session(csv_path):
                    self.event_table.load_events(self.event_manager.events)
                    self._update_session_status()
                    self.session_changed.emit(True)
                    QMessageBox.information(self, "Session Loaded",
                                          f"Loaded {self.event_manager.event_count} events.")
                return

        # Create new session
        if self.event_manager.create_new_session(session_dir):
            self.event_table.clear_all()
            self._update_session_status()
            self.session_changed.emit(True)
            QMessageBox.information(self, "Session Created",
                                  f"New session created in:\n{session_dir}")
        else:
            QMessageBox.critical(self, "Error",
                               "Failed to create session. Check the logs.")

    def _on_load_session(self):
        """Handle load session button click."""
        csv_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Event CSV File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if not csv_path:
            return

        if self.event_manager.load_existing_session(Path(csv_path)):
            self.event_table.load_events(self.event_manager.events)
            self._update_session_status()
            self.session_changed.emit(True)
            QMessageBox.information(self, "Session Loaded",
                                  f"Loaded {self.event_manager.event_count} events.")
        else:
            QMessageBox.critical(self, "Error",
                               "Failed to load session. Check the file format.")

    # =========================================================================
    # Event Mode
    # =========================================================================

    def _on_mode_toggle(self):
        """Handle mode toggle button click."""
        if not self.event_manager.session_active:
            self.mode_btn.setChecked(False)
            QMessageBox.warning(self, "No Session",
                              "Please create or load an event session first.")
            return

        self._event_mode_active = self.mode_btn.isChecked()
        self.event_mode_toggled.emit(self._event_mode_active)

    def set_event_mode(self, enabled: bool):
        """Set event mode state (called from main window).

        Args:
            enabled: Whether event mode should be enabled
        """
        if enabled and not self.event_manager.session_active:
            return

        self._event_mode_active = enabled
        self.mode_btn.setChecked(enabled)

    @property
    def event_mode_active(self) -> bool:
        """Check if event mode is currently active."""
        return self._event_mode_active

    # =========================================================================
    # Event Operations
    # =========================================================================

    def add_event(self, event: TaggedEvent):
        """Add a new event to the table.

        Args:
            event: Event to add
        """
        self.event_table.add_event(event)
        self._update_event_count()

    def update_event(self, event: TaggedEvent):
        """Update an event in the table.

        Args:
            event: Updated event
        """
        self.event_table.update_event(event)

    def _on_event_deleted(self, event_id: int):
        """Handle event deletion request from table."""
        if self.event_manager.delete_event(event_id):
            self.event_table.remove_event(event_id)
            self._update_event_count()

    def _on_goto_event(self, event_id: int):
        """Handle go to event request."""
        event = self.event_manager.get_event(event_id)
        if event:
            t_center = event.center_time
            f_center = event.center_frequency
            self.goto_event_requested.emit(t_center, f_center)

    def get_selected_event(self) -> Optional[TaggedEvent]:
        """Get the currently selected event."""
        event_id = self.event_table.get_selected_event_id()
        if event_id is not None:
            return self.event_manager.get_event(event_id)
        return None

    # =========================================================================
    # File Switch
    # =========================================================================

    def on_file_switched(self):
        """Called when the audio file is switched.

        Clears visual markers but keeps CSV data.
        """
        # The table still shows all events from all files
        # The canvas markers are cleared separately by main_window
        pass

    def refresh_table(self):
        """Refresh the table from the event manager."""
        self.event_table.load_events(self.event_manager.events)
        self._update_event_count()
