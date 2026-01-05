"""
Global Annotation Panel for searching and jumping to annotations across all files.
"""
import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Optional

from .qt_compat import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                        QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
                        QAbstractItemView, Signal, Qt)

logger = logging.getLogger(__name__)

class GlobalAnnotationPanel(QWidget):
    """Panel listing all annotations across all files in the project."""
    
    # Signals
    jump_requested = Signal(str, float, float, float, float)  # file_path, t_start, t_end, f_min, f_max
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.all_annotations = []
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Header
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("Global Annotations"))
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setFixedWidth(80)
        self.refresh_btn.clicked.connect(self.refresh_list)
        header_layout.addWidget(self.refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["File", "ID", "Time (s)", "Freq (Hz)", "Label"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setColumnWidth(1, 40)
        self.table.setColumnWidth(2, 80)
        self.table.setColumnWidth(3, 80)
        
        self.table.itemDoubleClicked.connect(self._on_item_double_clicked)
        
        layout.addWidget(self.table)
        
        # Status
        self.status_label = QLabel("0 annotations found")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)
        
    def refresh_list(self, root_dir: Optional[str] = None):
        """Scan directory for all annotation files and update list."""
        if not root_dir:
            # Try to get from current environment or project
            return
            
        logger.info(f"Scanning for global annotations in: {root_dir}")
        self.all_annotations = []
        root = Path(root_dir)
        
        # Find all _annotations.json files
        json_files = list(root.rglob("*_annotations.json"))
        
        # Also check project structure if it exists (files/*/annotations.json)
        project_ann_files = list(root.rglob("files/*/annotations.json"))
        json_files.extend([f for f in project_ann_files if f not in json_files])
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Determine the absolute audio file path
                # Try metadata in JSON first
                file_name = data.get('file_name')
                audio_path = None
                
                if file_name:
                    # Look for audio file in same directory as JSON or in root_dir
                    potential_paths = [
                        json_file.parent / file_name,
                        root / file_name,
                        root / "audio" / file_name  # common subfolder
                    ]
                    for p in potential_paths:
                        if p.exists():
                            audio_path = str(p)
                            break
                
                # Fallback: use JSON filename to guess audio filename
                if not audio_path:
                    audio_stem = json_file.stem.replace('_annotations', '')
                    # Search for any audio extension
                    for ext in ['.wav', '.flac', '.mp3', '.ogg']:
                        p = json_file.parent / (audio_stem + ext)
                        if p.exists():
                            audio_path = str(p)
                            break
                
                if not audio_path:
                    # Still no audio path? Just use the filename if we have it
                    audio_path = file_name if file_name else json_file.stem
                
                for ann in data.get('annotations', []):
                    ann['full_audio_path'] = audio_path
                    self.all_annotations.append(ann)
                    
            except Exception as e:
                logger.warning(f"Failed to load {json_file}: {e}")
        
        self._update_table()
        
    def _update_table(self):
        """Update the UI table with loaded annotations."""
        self.table.setRowCount(0)
        for ann in self.all_annotations:
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            # File
            file_name = Path(ann.get('full_audio_path', 'unknown')).name
            file_item = QTableWidgetItem(file_name)
            file_item.setData(Qt.ItemDataRole.UserRole, ann.get('full_audio_path'))
            self.table.setItem(row, 0, file_item)
            
            # ID
            self.table.setItem(row, 1, QTableWidgetItem(str(ann.get('id', ''))))
            
            # Time
            t_start = ann.get('t_start', 0)
            t_end = ann.get('t_end', 0)
            self.table.setItem(row, 2, QTableWidgetItem(f"{t_start:.2f}-{t_end:.2f}"))
            
            # Freq
            f_min = ann.get('f_min', 0)
            f_max = ann.get('f_max', 0)
            self.table.setItem(row, 3, QTableWidgetItem(f"{f_min:.0f}-{f_max:.0f}"))
            
            # Label
            self.table.setItem(row, 4, QTableWidgetItem(ann.get('track_label', '')))
            
            # Store full data in first item
            file_item.setData(Qt.ItemDataRole.UserRole + 1, ann)
            
        self.status_label.setText(f"{len(self.all_annotations)} annotations found")
        
    def _on_item_double_clicked(self, item):
        """Handle double-click to jump."""
        row = item.row()
        file_item = self.table.item(row, 0)
        file_path = file_item.data(Qt.ItemDataRole.UserRole)
        ann = file_item.data(Qt.ItemDataRole.UserRole + 1)
        
        if file_path and ann:
            self.jump_requested.emit(
                file_path, 
                ann.get('t_start', 0), 
                ann.get('t_end', 0),
                ann.get('f_min', 0), 
                ann.get('f_max', 0)
            )

