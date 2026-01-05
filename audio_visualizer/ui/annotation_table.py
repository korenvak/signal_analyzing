"""
Annotation table widget for displaying and editing annotations.
"""
import logging
from typing import Optional, List

from .qt_compat import (QTableWidget, QTableWidgetItem, QHeaderView,
                        QAbstractItemView, QMenu, QMessageBox, Qt, Signal, QKeyEvent)

from .annotation_data import Annotation

logger = logging.getLogger(__name__)


class AnnotationTableWidget(QTableWidget):
    """Table widget for displaying and editing annotations."""
    
    # Signals
    annotation_selected = Signal(int)  # annotation_id
    annotation_deleted = Signal(int)  # annotation_id
    annotation_updated = Signal(int)  # annotation_id
    annotation_visibility_changed = Signal(int, bool)  # annotation_id, is_visible
    doppler_visibility_changed = Signal(int, bool)  # annotation_id, show_doppler
    
    # Column indices (for easy reference)
    COL_ID = 0
    COL_FILE = 1
    COL_T_START = 2
    COL_T_END = 3
    COL_F_MIN = 4
    COL_F_MAX = 5
    COL_SNR = 6
    COL_SLOPE = 7
    COL_VIEW = 8
    COL_CURVE = 9
    
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
        
        # Setup columns - removed Label column
        columns = [
            'ID', 'File', 't_start', 't_end', 'f_min', 'f_max',
            'SNR (dB)', 'Slope (Hz/s)', 'View', 'Curve'
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
        self.setColumnWidth(self.COL_ID, 40)      # ID
        self.setColumnWidth(self.COL_FILE, 120)   # File
        self.setColumnWidth(self.COL_T_START, 70) # t_start
        self.setColumnWidth(self.COL_T_END, 70)   # t_end
        self.setColumnWidth(self.COL_F_MIN, 65)   # f_min
        self.setColumnWidth(self.COL_F_MAX, 65)   # f_max
        self.setColumnWidth(self.COL_SNR, 70)     # SNR
        self.setColumnWidth(self.COL_SLOPE, 80)   # Slope
        self.setColumnWidth(self.COL_VIEW, 50)    # View
        self.setColumnWidth(self.COL_CURVE, 55)   # Curve

        # File column should stretch to fill available space
        header.setSectionResizeMode(self.COL_FILE, QHeaderView.ResizeMode.Stretch)
        
        # Enable context menu
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        # Connect signals
        self.itemChanged.connect(self.on_item_changed)
        self.itemSelectionChanged.connect(self.on_selection_changed)
        
        # Track which columns are editable (View and Curve checkboxes)
        self.editable_columns = {self.COL_VIEW, self.COL_CURVE}
        
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
        
        # Time range (read-only)
        t_start_item = QTableWidgetItem(f"{annotation.t_start:.3f}")
        t_start_item.setFlags(t_start_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, self.COL_T_START, t_start_item)
        
        t_end_item = QTableWidgetItem(f"{annotation.t_end:.3f}")
        t_end_item.setFlags(t_end_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, self.COL_T_END, t_end_item)
        
        # Frequency range (read-only)
        f_min_item = QTableWidgetItem(f"{annotation.f_min:.1f}")
        f_min_item.setFlags(f_min_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, self.COL_F_MIN, f_min_item)
        
        f_max_item = QTableWidgetItem(f"{annotation.f_max:.1f}")
        f_max_item.setFlags(f_max_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, self.COL_F_MAX, f_max_item)
        
        # SNR (dB) (read-only, from track analysis)
        snr_text = ""
        if hasattr(annotation, 'snr_db') and annotation.snr_db is not None:
            snr_text = f"{annotation.snr_db:.1f}"
        snr_item = QTableWidgetItem(snr_text)
        snr_item.setFlags(snr_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, self.COL_SNR, snr_item)
        
        # Slope (Hz/s) (read-only, from track analysis)
        slope_text = ""
        if hasattr(annotation, 'slope_hz_per_sec') and annotation.slope_hz_per_sec is not None:
            slope_text = f"{annotation.slope_hz_per_sec:.1f}"
        slope_item = QTableWidgetItem(slope_text)
        slope_item.setFlags(slope_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.setItem(row, self.COL_SLOPE, slope_item)
        
        # View checkbox (show/hide annotation rectangle) - always enabled
        view_item = QTableWidgetItem()
        view_item.setCheckState(Qt.CheckState.Checked if annotation.is_visible else Qt.CheckState.Unchecked)
        view_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
        self.setItem(row, self.COL_VIEW, view_item)
        
        # Curve checkbox (show/hide Doppler curve)
        # Only enabled if annotation has curve points
        curve_item = QTableWidgetItem()
        has_curve = bool(annotation.points)
        curve_item.setCheckState(Qt.CheckState.Checked if annotation.show_doppler_curve and has_curve else Qt.CheckState.Unchecked)
        if has_curve:
            curve_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
            curve_item.setToolTip(f"Toggle curve visibility ({len(annotation.points)} points)")
        else:
            # Disabled - no curve points
            curve_item.setFlags(Qt.ItemFlag.ItemIsSelectable)
            curve_item.setToolTip("No curve drawn. Select this annotation, then draw a curve to enable.")
        self.setItem(row, self.COL_CURVE, curve_item)
    
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
            is_visible = item.checkState() == Qt.CheckState.Checked
            logger.debug(f"View checkbox changed for annotation {annotation_id}: {is_visible}")
            self.annotation_visibility_changed.emit(annotation_id, is_visible)
        
        # Handle Doppler curve visibility toggle (Curve column)
        elif col == self.COL_CURVE:
            show_curve = item.checkState() == Qt.CheckState.Checked
            logger.debug(f"Curve checkbox changed for annotation {annotation_id}: {show_curve}")
            self.doppler_visibility_changed.emit(annotation_id, show_curve)
        
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
            is_visible = self.item(row, self.COL_VIEW).checkState() == Qt.CheckState.Checked
            show_doppler_curve = self.item(row, self.COL_CURVE).checkState() == Qt.CheckState.Checked
            
            return {
                'is_visible': is_visible,
                'show_doppler_curve': show_doppler_curve
            }
        except (ValueError, AttributeError) as e:
            logger.error(f"Error reading annotation data from row {row}: {e}")
            return None
