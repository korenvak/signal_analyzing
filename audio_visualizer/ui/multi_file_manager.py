"""
Multi-File Manager
Handles multiple audio files with tabs and playlist functionality.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from .qt_compat import (QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
                        QListWidget, QListWidgetItem, QPushButton, QLabel,
                        QSplitter, QFrame, QFileDialog, QMessageBox,
                        Qt, Signal, QTimer, QDragEnterEvent, QDropEvent)

logger = logging.getLogger(__name__)

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
    """Playlist widget for managing multiple audio files."""
    
    # Signals
    file_selected = Signal(str)  # path
    file_remove_requested = Signal(str)  # path
    files_dropped = Signal(list)  # list of paths
    
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.audio_files: Dict[str, AudioFileInfo] = {}
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
        
        # Add files button
        self.add_button = QPushButton("Add Files...")
        self.add_button.clicked.connect(self.add_files)
        self.add_button.setMaximumWidth(80)
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
    
    def add_file_paths(self, file_paths: List[str]):
        """Add multiple file paths to the playlist."""
        added_count = 0
        
        for file_path in file_paths:
            if self.add_file_path(file_path):
                added_count += 1
        
        if added_count > 0:
            self.update_status()
            logger.info(f"Added {added_count} files to playlist")
    
    def add_file_path(self, file_path: str) -> bool:
        """Add a single file path to the playlist."""
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
            
            # Add to playlist
            self.audio_files[file_path] = file_info
            
            # Add to UI list
            item = QListWidgetItem(f"{file_info.name} ({file_info.size_mb:.1f} MB)")
            item.setData(Qt.ItemDataRole.UserRole, file_path)
            self.file_list.addItem(item)
            
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
                # Update text with more info
                status = "●" if file_info.is_loaded else "○"
                duration_str = f"{file_info.duration:.1f}s" if file_info.duration > 0 else ""
                sr_str = f"{file_info.sample_rate}Hz" if file_info.sample_rate > 0 else ""
                
                info_parts = [f"{file_info.size_mb:.1f} MB"]
                if duration_str:
                    info_parts.append(duration_str)
                if sr_str:
                    info_parts.append(sr_str)
                
                item.setText(f"{status} {file_info.name} ({', '.join(info_parts)})")
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