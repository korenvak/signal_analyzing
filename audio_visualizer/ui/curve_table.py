"""
Annotation table widget for displaying and editing annotations.
"""
import logging
from typing import Optional, List

from .qt_compat import (QTableWidget, QTableWidgetItem, QHeaderView,
                        QAbstractItemView, QMenu, QMessageBox, Qt, Signal, QKeyEvent)

from .annotation_data import Annotation

logger = logging.getLogger(__name__)


class CurveTableWidget(QTableWidget):
    """Table widget for displaying and editing curves."""
    
    # Signals
    annotation_selected = Signal(int)  # annotation_id
    annotation_deleted = Signal(int)  # annotation_id
    annotation_updated = Signal(int)  # annotation_id
    annotation_visibility_changed = Signal(int, bool)  # annotation_id, is_visible
    curve_visibility_changed = Signal(int, bool)  # annotation_id, show_curve
    
    # Column indices (for easy reference)
    # Column indices (for easy reference)
    COL_ID = 0
    COL_FILE = 1
    COL_POINTS = 2
    COL_VIEW = 3
    
    def __init__(self, parent=None):
        """Initialize the annotation table."""
        super().__init__(parent)
        
        # Style checkboxes to have visible checkmarks
        self.setStyleSheet("""
            QTableWidget::indicator {
                width: 16px;
                height: 16px;
            }
            QTableWidget::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #2E7D32;
                border-radius: 3px;
            }
            QTableWidget::indicator:unchecked {
                background-color: #424242;
                border: 2px solid #616161;
                border-radius: 3px;
            }
            QTableWidget::indicator:checked:hover {
                background-color: #66BB6A;
            }
            QTableWidget::indicator:unchecked:hover {
                background-color: #555555;
            }
        """)
        
        # Setup columns
        columns = [
            'ID', 'File', 'Points', 'View'
        ]
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        
        # Configure table behavior
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.DoubleClicked | 
                            QAbstractItemView.SelectedClicked)
        
        # Setup header - allow user resizing with sensible defaults
        header = self.horizontalHeader()
        # Default to interactive (user can resize all columns)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(False)

        # Set initial column widths (pixels)
        # Set initial column widths (pixels)
        self.setColumnWidth(self.COL_ID, 40)
        self.setColumnWidth(self.COL_FILE, 120)
        self.setColumnWidth(self.COL_POINTS, 60)
        self.setColumnWidth(self.COL_VIEW, 50)

        # File column should stretch to fill available space
        header.setSectionResizeMode(self.COL_FILE, QHeaderView.ResizeMode.Stretch)
        
        # Enable context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        # Connect signals
        self.itemChanged.connect(self.on_item_changed)
        self.itemSelectionChanged.connect(self.on_selection_changed)
        
        # Track which columns are editable (View checkbox)
        self.editable_columns = {self.COL_VIEW}
        
        # Mapping from row to annotation ID
        self.row_to_id: dict = {}
        
        # Block signals during updates
        self._updating = False
    
    def add_annotation(self, annotation: Annotation) -> int:
        """Add an annotation row to the table.
        
        Args:
            annotation: The annotation to add
        
        Returns:
            The row index where the annotation was added
        """
        self._updating = True
        
        row = self.rowCount()
        self.insertRow(row)
        
        # Store mapping
        self.row_to_id[row] = annotation.id
        
        # Fill cells
        self._set_row_data(row, annotation)
        
        self._updating = False
        
        logger.debug(f"Added annotation {annotation.id} to table at row {row}")
        return row
    
    def _set_row_data(self, row: int, annotation: Annotation):
        """Set data for a row from an annotation."""
        # ID (read-only)
        id_item = QTableWidgetItem(str(annotation.id))
        id_item.setFlags(id_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, self.COL_ID, id_item)
        
        # File name (read-only)
        name_item = QTableWidgetItem(annotation.file_name)
        name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, self.COL_FILE, name_item)
        
        # Points count (read-only)
        points_item = QTableWidgetItem(str(len(annotation.points)))
        points_item.setFlags(points_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, self.COL_POINTS, points_item)
        
        # View checkbox (show/hide curve)
        view_item = QTableWidgetItem()
        view_item.setCheckState(Qt.CheckState.Checked if annotation.show_doppler_curve else Qt.CheckState.Unchecked)
        view_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
        self.setItem(row, self.COL_VIEW, view_item)
    
    def remove_annotation(self, annotation_id: int) -> bool:
        """Remove an annotation row from the table.
        
        Args:
            annotation_id: ID of the annotation to remove
        
        Returns:
            True if removed, False if not found
        """
        # Find row
        row = None
        for r, ann_id in self.row_to_id.items():
            if ann_id == annotation_id:
                row = r
                break
        
        if row is None:
            return False
        
        self._updating = True
        
        # Remove row
        self.removeRow(row)
        
        # Update row mapping
        new_mapping = {}
        for r, ann_id in self.row_to_id.items():
            if r < row:
                new_mapping[r] = ann_id
            elif r > row:
                new_mapping[r - 1] = ann_id
        self.row_to_id = new_mapping
        
        self._updating = False
        
        logger.debug(f"Removed annotation {annotation_id} from table")
        return True
    
    def update_annotation(self, annotation: Annotation):
        """Update an existing annotation row.
        
        Args:
            annotation: The annotation with updated data
        """
        # Find row
        row = None
        for r, ann_id in self.row_to_id.items():
            if ann_id == annotation.id:
                row = r
                break
        
        if row is None:
            logger.warning(f"Cannot update annotation {annotation.id}: not found in table")
            return
        
        self._updating = True
        self._set_row_data(row, annotation)
        self._updating = False
    
    def clear_all(self):
        """Clear all rows from the table."""
        self._updating = True
        self.setRowCount(0)
        self.row_to_id.clear()
        self._updating = False
    
    def select_annotation(self, annotation_id: int):
        """Select an annotation row by ID.
        
        Args:
            annotation_id: ID of the annotation to select
        """
        # Find row
        row = None
        for r, ann_id in self.row_to_id.items():
            if ann_id == annotation_id:
                row = r
                break
        
        if row is not None:
            self.selectRow(row)
            self.scrollToItem(self.item(row, 0))
    
    def get_selected_annotation_id(self) -> Optional[int]:
        """Get the ID of the currently selected annotation.
        
        Returns:
            Annotation ID, or None if no selection
        """
        selected_rows = self.selectionModel().selectedRows()
        if not selected_rows:
            return None
        
        row = selected_rows[0].row()
        return self.row_to_id.get(row)
    
    def on_item_changed(self, item: QTableWidgetItem):
        """Handle item edit."""
        if self._updating:
            return
        
        row = item.row()
        annotation_id = self.row_to_id.get(row)
        if annotation_id is None:
            return
        
        col = item.column()
        
        # Handle visibility toggle (View column)
        if col == self.COL_VIEW:
            show_curve = item.checkState() == Qt.CheckState.Checked
            logger.debug(f"Curve view checkbox changed for annotation {annotation_id}: {show_curve}")
            self.curve_visibility_changed.emit(annotation_id, show_curve)
        
        # Emit general update signal for other changes
        self.annotation_updated.emit(annotation_id)
    
    def on_selection_changed(self):
        """Handle selection change."""
        if self._updating:
            return
        
        annotation_id = self.get_selected_annotation_id()
        if annotation_id is not None:
            self.annotation_selected.emit(annotation_id)
    
    def show_context_menu(self, position):
        """Show context menu for the table."""
        item = self.itemAt(position)
        if item is None:
            return
        
        row = item.row()
        annotation_id = self.row_to_id.get(row)
        if annotation_id is None:
            return
        
        menu = QMenu(self)
        delete_action = menu.addAction("Delete Annotation")
        
        action = menu.exec(self.viewport().mapToGlobal(position))
        if action == delete_action:
            self.annotation_deleted.emit(annotation_id)
    
    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Delete:
            annotation_id = self.get_selected_annotation_id()
            if annotation_id is not None:
                self.annotation_deleted.emit(annotation_id)
                event.accept()
                return
        
        super().keyPressEvent(event)
    
    def get_annotation_data_from_row(self, row: int) -> Optional[dict]:
        """Get annotation data from a table row.
        
        Args:
            row: Row index
        
        Returns:
            Dictionary with annotation data, or None if row not found
        """
        annotation_id = self.row_to_id.get(row)
        if annotation_id is None:
            return None
        
        try:
            show_doppler_curve = self.item(row, self.COL_VIEW).checkState() == Qt.CheckState.Checked
            
            return {
                'show_doppler_curve': show_doppler_curve
            }
        except (ValueError, AttributeError) as e:
            logger.error(f"Error reading annotation data from row {row}: {e}")
            return None
