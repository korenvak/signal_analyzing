"""
Status widget for displaying performance metrics.
"""
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QProgressBar


class StatusWidget(QWidget):
    """Status widget showing performance metrics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        
        self.fps_label = QLabel("FPS: --")
        self.memory_label = QLabel("Memory: --")
        self.gpu_label = QLabel("GPU: --")
        self.zoom_label = QLabel("Zoom: 1.0x")
        self.cursor_label = QLabel("Time: -- | Freq: --")
        self.measure_label = QLabel("")
        self.measure_label.setStyleSheet("color: yellow; font-weight: bold;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumWidth(150)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Computing... %p%")
        
        layout.addWidget(QLabel("Performance:"))
        layout.addWidget(self.fps_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.memory_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.gpu_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.zoom_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.cursor_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.measure_label)
        layout.addStretch()
        layout.addWidget(self.progress_bar)
    
    def update_stats(self, fps: float = None, memory_mb: float = None, 
                    gpu_mb: float = None, progress: float = None):
        """Update status display."""
        if fps is not None:
            self.fps_label.setText(f"FPS: {fps:.1f}")
        
        if memory_mb is not None:
            self.memory_label.setText(f"Memory: {memory_mb:.0f}MB")
        
        if gpu_mb is not None:
            self.gpu_label.setText(f"GPU: {gpu_mb:.0f}MB")
        
        if progress is not None:
            if 0 <= progress < 1:
                self.progress_bar.setVisible(True)
                self.progress_bar.setValue(int(progress * 100))
            else:
                self.progress_bar.setVisible(False)
    
    def show_progress(self, message: str = "Computing..."):
        """Show the progress bar with a message."""
        self.progress_bar.setFormat(f"{message} %p%")
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
    
    def update_progress(self, value: int):
        """Update progress value (0-100)."""
        self.progress_bar.setValue(value)
    
    def hide_progress(self):
        """Hide the progress bar."""
        self.progress_bar.setVisible(False)
    
    def update_zoom(self, time_zoom: float, freq_zoom: float):
        """Update zoom level display."""
        # Show the larger zoom factor
        zoom = max(time_zoom, freq_zoom)
        if zoom >= 10:
            self.zoom_label.setText(f"Zoom: {zoom:.0f}x")
        elif zoom >= 1:
            self.zoom_label.setText(f"Zoom: {zoom:.1f}x")
        else:
            self.zoom_label.setText(f"Zoom: {zoom:.2f}x")
    
    def update_cursor(self, time_sec: float, freq_hz: float):
        """Update cursor position display."""
        # Format time as mm:ss.ms
        minutes = int(time_sec // 60)
        seconds = time_sec % 60
        time_str = f"{minutes:02d}:{seconds:05.2f}"
        
        # Format frequency
        if freq_hz >= 1000:
            freq_str = f"{freq_hz/1000:.2f} kHz"
        else:
            freq_str = f"{freq_hz:.1f} Hz"
        
        self.cursor_label.setText(f"Time: {time_str} | Freq: {freq_str}")
    
    def update_measurement_mode(self, is_on: bool):
        """Update measurement mode indicator."""
        if is_on:
            self.measure_label.setText("[M] MEASURE MODE (click 2 points, ESC to clear)")
        else:
            self.measure_label.setText("")

