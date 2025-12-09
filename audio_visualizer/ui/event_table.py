"""
Event table widget for displaying and editing tagged events.

Features:
- Display all events in current session
- Context menu: Edit, Delete, View Image, Go to Event
- Double-click to edit
- Selection signals for navigation
"""
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, List

from .qt_compat import (
    QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QMenu, QMessageBox,
    Qt, Signal, QAction, QColor
)

from .event_data import TaggedEvent

logger = logging.getLogger(__name__)


class EventTableWidget(QTableWidget):
    """Table widget for displaying and editing tagged events."""

    # Signals
    event_selected = Signal(int)       # event_id - emitted on single click
    event_deleted = Signal(int)        # event_id - emitted when delete requested
    event_edit_requested = Signal(int) # event_id - emitted when edit requested
    event_goto_requested = Signal(int) # event_id - emitted when "go to" requested

    # Column configuration
    COLUMNS = [
        ('ID', 50),
        ('File', 150),
        ('Sensor', 60),
        ('ID#', 50),
        ('t_start', 80),
        ('t_end', 80),
        ('f_min', 70),
        ('f_max', 70),
        ('Absolute Start', 150),
        ('Harmonic', 70),
        ('SNR (dB)', 70),
        ('Notes', 150),
    ]

    def __init__(self, parent=None):
        """Initialize the event table."""
        super().__init__(parent)

        # Setup columns
        self.setColumnCount(len(self.COLUMNS))
        self.setHorizontalHeaderLabels([col[0] for col in self.COLUMNS])

        # Configure table behavior
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)  # Read-only, edit via dialog
        self.setAlternatingRowColors(True)

        # Setup header
        header = self.horizontalHeader()
        for i, (_, width) in enumerate(self.COLUMNS):
            if i == len(self.COLUMNS) - 1:  # Last column (Notes) stretches
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
            else:
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
                self.setColumnWidth(i, width)

        # Enable context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        # Connect signals
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.itemDoubleClicked.connect(self._on_double_click)

        # Mapping from row to event ID
        self._row_to_id: dict = {}
        self._id_to_row: dict = {}

        # Store events for reference
        self._events: dict = {}  # id -> TaggedEvent

    def add_event(self, event: TaggedEvent) -> int:
        """Add an event row to the table.

        Args:
            event: The event to add

        Returns:
            The row index where the event was added
        """
        row = self.rowCount()
        self.insertRow(row)

        # Store mappings
        self._row_to_id[row] = event.id
        self._id_to_row[event.id] = row
        self._events[event.id] = event

        # Fill cells
        self._set_row_data(row, event)

        logger.debug(f"Added event {event.id} to table at row {row}")
        return row

    def _set_row_data(self, row: int, event: TaggedEvent):
        """Set data for a row from an event."""
        # ID
        self._set_item(row, 0, str(event.id))

        # File (just filename, not full path)
        filename = Path(event.audio_file).name if event.audio_file else ""
        self._set_item(row, 1, filename)

        # Sensor name
        self._set_item(row, 2, event.sensor_name or "")

        # Sensor ID
        self._set_item(row, 3, str(event.sensor_id) if event.sensor_id else "")

        # Time range
        self._set_item(row, 4, f"{event.t_start:.3f}")
        self._set_item(row, 5, f"{event.t_end:.3f}")

        # Frequency range
        self._set_item(row, 6, f"{event.f_min:.0f}")
        self._set_item(row, 7, f"{event.f_max:.0f}")

        # Absolute start time
        if event.event_start_absolute:
            abs_str = event.event_start_absolute.strftime("%Y-%m-%d %H:%M:%S")
            self._set_item(row, 8, abs_str)
        else:
            self._set_item(row, 8, "")

        # Harmonic
        self._set_item(row, 9, str(event.harmonic_number) if event.harmonic_number else "")

        # SNR
        if event.snr_estimate_db is not None:
            self._set_item(row, 10, f"{event.snr_estimate_db:.1f}")
        else:
            self._set_item(row, 10, "")

        # Notes
        self._set_item(row, 11, event.notes or "")

    def _set_item(self, row: int, col: int, text: str):
        """Set a table item with read-only flags."""
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, col, item)

    def update_event(self, event: TaggedEvent):
        """Update an existing event in the table.

        Args:
            event: The updated event
        """
        if event.id not in self._id_to_row:
            logger.warning(f"Event {event.id} not in table")
            return

        row = self._id_to_row[event.id]
        self._events[event.id] = event
        self._set_row_data(row, event)

    def remove_event(self, event_id: int):
        """Remove an event from the table.

        Args:
            event_id: ID of the event to remove
        """
        if event_id not in self._id_to_row:
            logger.warning(f"Event {event_id} not in table")
            return

        row = self._id_to_row[event_id]

        # Remove row
        self.removeRow(row)

        # Update mappings
        del self._id_to_row[event_id]
        del self._row_to_id[row]
        if event_id in self._events:
            del self._events[event_id]

        # Update row indices for rows below
        new_row_to_id = {}
        new_id_to_row = {}
        for r, eid in self._row_to_id.items():
            new_row = r if r < row else r - 1
            new_row_to_id[new_row] = eid
            new_id_to_row[eid] = new_row
        self._row_to_id = new_row_to_id
        self._id_to_row = new_id_to_row

        logger.debug(f"Removed event {event_id} from table")

    def clear_all(self):
        """Clear all events from the table."""
        self.setRowCount(0)
        self._row_to_id.clear()
        self._id_to_row.clear()
        self._events.clear()

    def load_events(self, events: List[TaggedEvent]):
        """Load multiple events into the table.

        Args:
            events: List of events to load
        """
        self.clear_all()
        for event in events:
            self.add_event(event)

    def get_selected_event_id(self) -> Optional[int]:
        """Get the ID of the currently selected event."""
        rows = self.selectionModel().selectedRows()
        if not rows:
            return None

        row = rows[0].row()
        return self._row_to_id.get(row)

    def select_event(self, event_id: int):
        """Select a specific event by ID.

        Args:
            event_id: ID of the event to select
        """
        if event_id not in self._id_to_row:
            return

        row = self._id_to_row[event_id]
        self.selectRow(row)
        self.scrollTo(self.model().index(row, 0))

    # =========================================================================
    # Event Handlers
    # =========================================================================

    def _on_selection_changed(self):
        """Handle selection change."""
        event_id = self.get_selected_event_id()
        if event_id is not None:
            self.event_selected.emit(event_id)

    def _on_double_click(self, item: QTableWidgetItem):
        """Handle double-click (edit event)."""
        row = item.row()
        event_id = self._row_to_id.get(row)
        if event_id is not None:
            self.event_edit_requested.emit(event_id)

    def _show_context_menu(self, pos):
        """Show context menu for event operations."""
        item = self.itemAt(pos)
        if item is None:
            return

        row = item.row()
        event_id = self._row_to_id.get(row)
        if event_id is None:
            return

        event = self._events.get(event_id)
        if event is None:
            return

        menu = QMenu(self)

        # Go to event
        goto_action = QAction("Go to Event", self)
        goto_action.triggered.connect(lambda: self.event_goto_requested.emit(event_id))
        menu.addAction(goto_action)

        menu.addSeparator()

        # Edit
        edit_action = QAction("Edit...", self)
        edit_action.triggered.connect(lambda: self.event_edit_requested.emit(event_id))
        menu.addAction(edit_action)

        # View image (if exists)
        if event.image_path and Path(event.image_path).exists():
            view_img_action = QAction("View Image", self)
            view_img_action.triggered.connect(lambda: self._open_image(event.image_path))
            menu.addAction(view_img_action)

        menu.addSeparator()

        # Delete
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self._confirm_delete(event_id))
        menu.addAction(delete_action)

        menu.exec(self.viewport().mapToGlobal(pos))

    def _confirm_delete(self, event_id: int):
        """Show confirmation dialog before deleting."""
        result = QMessageBox.question(
            self,
            "Delete Event",
            f"Are you sure you want to delete event {event_id}?\n\n"
            "This will also delete the associated image.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if result == QMessageBox.Yes:
            self.event_deleted.emit(event_id)

    def _open_image(self, image_path: str):
        """Open image with default system viewer."""
        try:
            path = Path(image_path)
            if not path.exists():
                QMessageBox.warning(self, "Image Not Found",
                                   f"Image file not found:\n{image_path}")
                return

            if sys.platform == 'win32':
                os.startfile(str(path))
            elif sys.platform == 'darwin':
                subprocess.run(['open', str(path)])
            else:
                subprocess.run(['xdg-open', str(path)])

        except Exception as e:
            logger.error(f"Failed to open image: {e}")
            QMessageBox.warning(self, "Error",
                               f"Failed to open image:\n{e}")

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Delete:
            event_id = self.get_selected_event_id()
            if event_id is not None:
                self._confirm_delete(event_id)
                event.accept()
                return

        super().keyPressEvent(event)
