"""
Multi-File Manager
Handles multiple audio files with tabs and playlist functionality.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from .qt_compat import (QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
                        QListWidget, QListWidgetItem, QPushButton, QLabel,
                        QSplitter, QFrame, QFileDialog, QMessageBox, QMenu,
                        Qt, Signal, QTimer, QDragEnterEvent, QDropEvent)
from ..core.filename_parser import parse_pixel_filename

logger = logging.getLogger(__name__)


class SmartFileSorter:
    """Smart file sorting - extracts time from filenames when possible."""
    
    @staticmethod
    def get_sort_key(file_path: str) -> Tuple:
        """Get sort key for a file - (has_time, datetime or name, name).
        
        Files with parseable timestamps sort by time.
        Files without timestamps sort alphabetically after them.
        """
        path = Path(file_path)
        parsed = parse_pixel_filename(file_path)
        
        if parsed is not None:
            # Has timestamp - sort by start_time, then filename
            return (0, parsed.start_time, path.name.lower())
        else:
            # No timestamp - sort alphabetically (after timestamped files)
            return (1, datetime.min, path.name.lower())
    
    @staticmethod
    def sort_files(file_paths: List[str], reverse: bool = False) -> List[str]:
        """Sort file paths smartly - by time if parseable, else alphabetically.
        
        Args:
            file_paths: List of file paths to sort
            reverse: If True, sort newest first / Z-A
            
        Returns:
            Sorted list of file paths
        """
        return sorted(file_paths, key=SmartFileSorter.get_sort_key, reverse=reverse)

@dataclass
class AudioFileInfo:
    """Information about an audio file."""
    path: str
    name: str
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    size_mb: float = 0.0
    is_loaded: bool = False

class PlaylistWidget(QWidget):
    """Playlist widget for managing multiple audio files with smart sorting."""
    
    # Signals
    file_selected = Signal(str)  # path
    file_remove_requested = Signal(str)  # path
    files_dropped = Signal(list)  # list of paths
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.audio_files: Dict[str, AudioFileInfo] = {}
        self._current_sort = 'smart'  # 'smart', 'name', 'size', 'none'
        self._sort_reverse = False
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the playlist UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Header
        header_layout = QHBoxLayout()
        self.title_label = QLabel("Playlist")
        self.title_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(self.title_label)
        
        # Sort button
        self.sort_button = QPushButton("Sort ▼")
        self.sort_button.setMaximumWidth(55)
        self.sort_button.setToolTip("Sort playlist")
        self.sort_button.clicked.connect(self.show_sort_menu)
        header_layout.addWidget(self.sort_button)
        
        # Add files button
        self.add_button = QPushButton("Add...")
        self.add_button.clicked.connect(self.add_files)
        self.add_button.setMaximumWidth(50)
        header_layout.addWidget(self.add_button)
        
        layout.addLayout(header_layout)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.file_list.itemClicked.connect(self.on_file_selected)
        self.file_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.file_list)
        
        # Status
        self.status_label = QLabel("0 files")
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)
        
    def show_sort_menu(self):
        """Show sorting options menu."""
        menu = QMenu(self)
        
        # Smart sort (by time from filename, then alphabetical)
        smart_action = menu.addAction("📅 Smart Sort (by time)")
        smart_action.setCheckable(True)
        smart_action.setChecked(self._current_sort == 'smart' and not self._sort_reverse)
        smart_action.triggered.connect(lambda: self.sort_playlist('smart', False))
        
        smart_rev_action = menu.addAction("📅 Smart Sort (newest first)")
        smart_rev_action.setCheckable(True)
        smart_rev_action.setChecked(self._current_sort == 'smart' and self._sort_reverse)
        smart_rev_action.triggered.connect(lambda: self.sort_playlist('smart', True))
        
        menu.addSeparator()
        
        # Alphabetical
        name_action = menu.addAction("🔤 Name (A-Z)")
        name_action.setCheckable(True)
        name_action.setChecked(self._current_sort == 'name' and not self._sort_reverse)
        name_action.triggered.connect(lambda: self.sort_playlist('name', False))
        
        name_rev_action = menu.addAction("🔤 Name (Z-A)")
        name_rev_action.setCheckable(True)
        name_rev_action.setChecked(self._current_sort == 'name' and self._sort_reverse)
        name_rev_action.triggered.connect(lambda: self.sort_playlist('name', True))
        
        menu.addSeparator()
        
        # Size
        size_action = menu.addAction("📊 Size (smallest first)")
        size_action.setCheckable(True)
        size_action.setChecked(self._current_sort == 'size' and not self._sort_reverse)
        size_action.triggered.connect(lambda: self.sort_playlist('size', False))
        
        size_rev_action = menu.addAction("📊 Size (largest first)")
        size_rev_action.setCheckable(True)
        size_rev_action.setChecked(self._current_sort == 'size' and self._sort_reverse)
        size_rev_action.triggered.connect(lambda: self.sort_playlist('size', True))
        
        # Show menu below button
        menu.exec(self.sort_button.mapToGlobal(self.sort_button.rect().bottomLeft()))
    
    def sort_playlist(self, sort_type: str = 'smart', reverse: bool = False):
        """Sort the playlist by specified criteria.
        
        Args:
            sort_type: 'smart' (by time from filename), 'name', 'size'
            reverse: If True, reverse the sort order
        """
        if not self.audio_files:
            return
        
        self._current_sort = sort_type
        self._sort_reverse = reverse
        
        file_paths = list(self.audio_files.keys())
        
        if sort_type == 'smart':
            # Smart sort - by time from filename, then alphabetical
            sorted_paths = SmartFileSorter.sort_files(file_paths, reverse=reverse)
        elif sort_type == 'name':
            # Alphabetical sort
            sorted_paths = sorted(file_paths, key=lambda p: Path(p).name.lower(), reverse=reverse)
        elif sort_type == 'size':
            # Sort by file size
            sorted_paths = sorted(file_paths, key=lambda p: self.audio_files[p].size_mb, reverse=reverse)
        else:
            sorted_paths = file_paths
        
        # Rebuild the list widget
        self._rebuild_list_from_paths(sorted_paths)
        
        sort_desc = {
            'smart': 'time' if not reverse else 'time (newest first)',
            'name': 'A-Z' if not reverse else 'Z-A',
            'size': 'smallest' if not reverse else 'largest'
        }
        logger.info(f"Playlist sorted by {sort_desc.get(sort_type, sort_type)}")
    
    def _rebuild_list_from_paths(self, sorted_paths: List[str]):
        """Rebuild the list widget with sorted paths."""
        # Save current selection
        current_item = self.file_list.currentItem()
        current_path = current_item.data(Qt.ItemDataRole.UserRole) if current_item else None
        
        # Clear and rebuild
        self.file_list.clear()
        
        for file_path in sorted_paths:
            file_info = self.audio_files.get(file_path)
            if file_info:
                item = self._create_list_item(file_path, file_info)
                self.file_list.addItem(item)
        
        # Restore selection
        if current_path:
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == current_path:
                    self.file_list.setCurrentItem(item)
                    break
    
    def _create_list_item(self, file_path: str, file_info: AudioFileInfo) -> QListWidgetItem:
        """Create a list widget item for a file."""
        # Check if file has parseable timestamp
        parsed = parse_pixel_filename(file_path)
        
        # Build display text
        status = "●" if file_info.is_loaded else "○"
        
        info_parts = [f"{file_info.size_mb:.1f} MB"]
        if file_info.duration > 0:
            info_parts.append(f"{file_info.duration:.1f}s")
        if file_info.sample_rate > 0:
            info_parts.append(f"{file_info.sample_rate}Hz")
        
        # Add time info if parseable
        if parsed:
            time_str = parsed.start_time.strftime("%H:%M:%S")
            display_text = f"{status} {file_info.name}\n    📅 {time_str} ({', '.join(info_parts)})"
        else:
            display_text = f"{status} {file_info.name} ({', '.join(info_parts)})"
        
        item = QListWidgetItem(display_text)
        item.setData(Qt.ItemDataRole.UserRole, file_path)
        
        # Add tooltip with full info
        tooltip = f"Path: {file_path}\nSize: {file_info.size_mb:.2f} MB"
        if parsed:
            tooltip += f"\nStart: {parsed.start_time}"
            tooltip += f"\nEnd: {parsed.end_time}"
            tooltip += f"\nSensor: {parsed.sensor_name} ({parsed.sensor_id})"
        item.setToolTip(tooltip)
        
        return item
    
    def add_files(self):
        """Open file dialog to add audio files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Add Audio Files",
            "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a *.aac);;All Files (*)"
        )
        
        if files:
            self.add_file_paths(files)
    
    def add_file_paths(self, file_paths: List[str], auto_sort: bool = True):
        """Add multiple file paths to the playlist.
        
        Args:
            file_paths: List of file paths to add
            auto_sort: If True, automatically sort after adding (default: True)
        """
        added_count = 0
        
        for file_path in file_paths:
            if self.add_file_path(file_path, auto_sort=False):
                added_count += 1
        
        if added_count > 0:
            self.update_status()
            logger.info(f"Added {added_count} files to playlist")
            
            # Auto-sort after batch add
            if auto_sort and self._current_sort != 'none':
                self.sort_playlist(self._current_sort, self._sort_reverse)
    
    def add_file_path(self, file_path: str, auto_sort: bool = False) -> bool:
        """Add a single file path to the playlist.
        
        Args:
            file_path: Path to add
            auto_sort: If True, sort playlist after adding
            
        Returns:
            True if file was added successfully
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"File does not exist: {file_path}")
                return False
            
            if not self.is_audio_file(file_path):
                logger.warning(f"Not an audio file: {file_path}")
                return False
            
            # Check if already in playlist
            if file_path in self.audio_files:
                logger.debug(f"File already in playlist: {file_path}")
                return False
            
            # Create file info
            file_info = AudioFileInfo(
                path=file_path,
                name=path.name,
                size_mb=path.stat().st_size / (1024 * 1024)
            )
            
            # Add to playlist data
            self.audio_files[file_path] = file_info
            
            # Add to UI list using helper
            item = self._create_list_item(file_path, file_info)
            self.file_list.addItem(item)
            
            # Optional auto-sort
            if auto_sort and self._current_sort != 'none':
                self.sort_playlist(self._current_sort, self._sort_reverse)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding file {file_path}: {e}")
            return False
    
    def remove_file(self, file_path: str):
        """Remove a file from the playlist."""
        if file_path not in self.audio_files:
            return
        
        # Remove from data
        del self.audio_files[file_path]
        
        # Remove from UI
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == file_path:
                self.file_list.takeItem(i)
                break
        
        self.update_status()
        logger.info(f"Removed file from playlist: {Path(file_path).name}")
    
    def clear_playlist(self):
        """Clear all files from the playlist."""
        self.audio_files.clear()
        self.file_list.clear()
        self.update_status()
        logger.info("Playlist cleared")
    
    def get_file_paths(self) -> List[str]:
        """Get all file paths in the playlist."""
        return list(self.audio_files.keys())
    
    def get_file_info(self, file_path: str) -> Optional[AudioFileInfo]:
        """Get file info for a specific path."""
        return self.audio_files.get(file_path)
    
    def update_file_info(self, file_path: str, duration: float = None, 
                        sample_rate: int = None, channels: int = None, is_loaded: bool = None):
        """Update file information."""
        if file_path not in self.audio_files:
            return
        
        file_info = self.audio_files[file_path]
        if duration is not None:
            file_info.duration = duration
        if sample_rate is not None:
            file_info.sample_rate = sample_rate
        if channels is not None:
            file_info.channels = channels
        if is_loaded is not None:
            file_info.is_loaded = is_loaded
        
        # Update UI item
        self.update_ui_item(file_path, file_info)
    
    def update_ui_item(self, file_path: str, file_info: AudioFileInfo):
        """Update the UI item for a file."""
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == file_path:
                # Create updated item text using same format as _create_list_item
                parsed = parse_pixel_filename(file_path)
                status = "●" if file_info.is_loaded else "○"
                
                info_parts = [f"{file_info.size_mb:.1f} MB"]
                if file_info.duration > 0:
                    info_parts.append(f"{file_info.duration:.1f}s")
                if file_info.sample_rate > 0:
                    info_parts.append(f"{file_info.sample_rate}Hz")
                
                if parsed:
                    time_str = parsed.start_time.strftime("%H:%M:%S")
                    display_text = f"{status} {file_info.name}\n    📅 {time_str} ({', '.join(info_parts)})"
                else:
                    display_text = f"{status} {file_info.name} ({', '.join(info_parts)})"
                
                item.setText(display_text)
                break
    
    def update_status(self):
        """Update the status label."""
        count = len(self.audio_files)
        total_size = sum(info.size_mb for info in self.audio_files.values())
        loaded_count = sum(1 for info in self.audio_files.values() if info.is_loaded)
        
        self.status_label.setText(f"{count} files, {total_size:.1f} MB ({loaded_count} loaded)")
    
    def on_file_selected(self, item: QListWidgetItem):
        """Handle file selection."""
        file_path = item.data(Qt.ItemDataRole.UserRole)
        logger.info(f"PlaylistWidget: Item clicked, file_path={file_path}")
        if file_path:
            logger.info(f"PlaylistWidget: Emitting file_selected signal for {file_path}")
            self.file_selected.emit(file_path)
        else:
            logger.warning("PlaylistWidget: No file_path data in clicked item")
    
    def select_next_file(self) -> bool:
        """Select the next file in the playlist.
        
        Returns:
            True if a next file was selected, False if at end or empty
        """
        if self.file_list.count() == 0:
            return False
        
        current_row = self.file_list.currentRow()
        next_row = current_row + 1
        
        if next_row >= self.file_list.count():
            return False  # Already at end
        
        self.file_list.setCurrentRow(next_row)
        item = self.file_list.item(next_row)
        if item:
            file_path = item.data(Qt.ItemDataRole.UserRole)
            if file_path:
                self.file_selected.emit(file_path)
                return True
        return False
    
    def select_previous_file(self) -> bool:
        """Select the previous file in the playlist.
        
        Returns:
            True if a previous file was selected, False if at start or empty
        """
        if self.file_list.count() == 0:
            return False
        
        current_row = self.file_list.currentRow()
        prev_row = current_row - 1
        
        if prev_row < 0:
            return False  # Already at start
        
        self.file_list.setCurrentRow(prev_row)
        item = self.file_list.item(prev_row)
        if item:
            file_path = item.data(Qt.ItemDataRole.UserRole)
            if file_path:
                self.file_selected.emit(file_path)
                return True
        return False
    
    def get_current_index(self) -> int:
        """Get current file index (0-based)."""
        return self.file_list.currentRow()
    
    def get_total_files(self) -> int:
        """Get total number of files in playlist."""
        return self.file_list.count()
    
    def show_context_menu(self, position):
        """Show context menu for file operations."""
        item = self.file_list.itemAt(position)
        if not item:
            return
        
        # For now, just emit remove signal
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path:
            self.file_remove_requested.emit(file_path)
    
    def is_audio_file(self, file_path: str) -> bool:
        """Check if file is a supported audio format."""
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.mp4'}
        return Path(file_path).suffix.lower() in audio_extensions
    
    # Drag and drop support
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event."""
        urls = event.mimeData().urls()
        file_paths = []
        
        for url in urls:
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if self.is_audio_file(file_path):
                    file_paths.append(file_path)
        
        if file_paths:
            self.files_dropped.emit(file_paths)
            event.acceptProposedAction()

class FileTabWidget(QTabWidget):
    """Tab widget for managing multiple file views."""
    
    # Signals
    current_file_changed = Signal(str)  # file_path
    file_close_requested = Signal(str)  # file_path
    
    def __init__(self):
        super().__init__()
        self.setTabsClosable(True)
        self.setMovable(True)
        self.file_tabs: Dict[str, QWidget] = {}  # file_path -> tab_widget
        
        # Connect signals
        self.currentChanged.connect(self.on_current_changed)
        self.tabCloseRequested.connect(self.on_tab_close_requested)
    
    def add_file_tab(self, file_path: str, tab_widget: QWidget) -> bool:
        """Add a tab for a file."""
        if file_path in self.file_tabs:
            # Switch to existing tab
            existing_widget = self.file_tabs[file_path]
            index = self.indexOf(existing_widget)
            if index >= 0:
                self.setCurrentIndex(index)
            return False
        
        # Add new tab
        file_name = Path(file_path).name
        index = self.addTab(tab_widget, file_name)
        self.file_tabs[file_path] = tab_widget
        self.setCurrentIndex(index)
        
        logger.info(f"Added file tab: {file_name}")
        return True
    
    def remove_file_tab(self, file_path: str):
        """Remove a file tab."""
        if file_path not in self.file_tabs:
            return
        
        tab_widget = self.file_tabs[file_path]
        index = self.indexOf(tab_widget)
        
        if index >= 0:
            self.removeTab(index)
        
        del self.file_tabs[file_path]
        logger.info(f"Removed file tab: {Path(file_path).name}")
    
    def get_current_file_path(self) -> Optional[str]:
        """Get the file path of the current tab."""
        current_widget = self.currentWidget()
        if not current_widget:
            return None
        
        for file_path, widget in self.file_tabs.items():
            if widget == current_widget:
                return file_path
        
        return None
    
    def has_file_tab(self, file_path: str) -> bool:
        """Check if a file tab exists."""
        return file_path in self.file_tabs
    
    def get_file_paths(self) -> List[str]:
        """Get all open file paths."""
        return list(self.file_tabs.keys())
    
    def on_current_changed(self, index: int):
        """Handle current tab change."""
        if index < 0:
            return
        
        file_path = self.get_current_file_path()
        if file_path:
            self.current_file_changed.emit(file_path)
    
    def on_tab_close_requested(self, index: int):
        """Handle tab close request."""
        widget = self.widget(index)
        if not widget:
            return
        
        # Find file path for this widget
        file_path = None
        for path, w in self.file_tabs.items():
            if w == widget:
                file_path = path
                break
        
        if file_path:
            self.file_close_requested.emit(file_path)

class MultiFileManager(QWidget):
    """Main multi-file manager widget."""
    
    # Signals
    file_opened = Signal(str)  # file_path
    file_closed = Signal(str)  # file_path
    current_file_changed = Signal(str)  # file_path
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.connect_signals()
        
    def setup_ui(self):
        """Set up the UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Playlist widget (left side)
        self.playlist = PlaylistWidget()
        self.playlist.setMaximumWidth(300)
        self.playlist.setMinimumWidth(200)
        splitter.addWidget(self.playlist)
        
        # File tabs (right side)
        self.file_tabs = FileTabWidget()
        splitter.addWidget(self.file_tabs)
        
        # Set splitter proportions
        splitter.setSizes([200, 800])
        
    def connect_signals(self):
        """Connect internal signals."""
        # Playlist signals
        self.playlist.file_selected.connect(self.open_file)
        self.playlist.file_remove_requested.connect(self.remove_file)
        self.playlist.files_dropped.connect(self.add_files)
        
        # Tab signals
        self.file_tabs.current_file_changed.connect(self.current_file_changed.emit)
        self.file_tabs.file_close_requested.connect(self.close_file)
    
    def add_files(self, file_paths: List[str]):
        """Add files to the playlist."""
        self.playlist.add_file_paths(file_paths)
    
    def open_file(self, file_path: str):
        """Open a file (create tab if needed)."""
        if not self.file_tabs.has_file_tab(file_path):
            # Create placeholder tab widget
            # The actual visualization will be set up by the main window
            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)
            tab_layout.addWidget(QLabel(f"Loading {Path(file_path).name}..."))
            
            self.file_tabs.add_file_tab(file_path, tab_widget)
            self.file_opened.emit(file_path)
        else:
            # Switch to existing tab
            widget = self.file_tabs.file_tabs[file_path]
            index = self.file_tabs.indexOf(widget)
            self.file_tabs.setCurrentIndex(index)
    
    def close_file(self, file_path: str):
        """Close a file tab."""
        self.file_tabs.remove_file_tab(file_path)
        self.file_closed.emit(file_path)
    
    def remove_file(self, file_path: str):
        """Remove file from playlist and close tab."""
        self.close_file(file_path)
        self.playlist.remove_file(file_path)
    
    def get_current_file_path(self) -> Optional[str]:
        """Get the current active file path."""
        return self.file_tabs.get_current_file_path()
    
    def update_file_info(self, file_path: str, **kwargs):
        """Update file information in the playlist."""
        self.playlist.update_file_info(file_path, **kwargs)
    
    def set_file_tab_widget(self, file_path: str, widget: QWidget):
        """Replace the placeholder tab widget with the actual visualization."""
        if file_path in self.file_tabs.file_tabs:
            old_widget = self.file_tabs.file_tabs[file_path]
            index = self.file_tabs.indexOf(old_widget)
            
            if index >= 0:
                # Remove old tab
                self.file_tabs.removeTab(index)
                
                # Insert new tab at same position
                file_name = Path(file_path).name
                self.file_tabs.insertTab(index, widget, file_name)
                self.file_tabs.file_tabs[file_path] = widget
                self.file_tabs.setCurrentIndex(index)