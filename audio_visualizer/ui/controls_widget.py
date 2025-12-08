"""
Controls widget for spectrogram settings.
"""
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QLabel, 
                               QComboBox, QPushButton, QFrame, QToolButton,
                               QMenu, QWidgetAction, QSlider, QSizePolicy)
from PySide6.QtCore import Qt, Signal


class ControlsWidget(QWidget):
    """Widget containing analysis controls."""
    
    # Signals
    parameters_changed = Signal(dict)
    colormap_changed = Signal(str)
    db_range_changed = Signal(float, float)
    refresh_requested = Signal()
    window_changed = Signal(str)
    interpolation_changed = Signal(str)
    freq_scale_changed = Signal(str)
    normalization_mode_changed = Signal(str)  # 'minmax' or 'std'
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.connect_signals()
    
    def setup_ui(self):
        """Setup the control interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(12)
        
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setMaximumHeight(80)
        
        # Colormap
        layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems([
            'plasma', 'viridis', 'magma', 'inferno', 'cividis',
            'jet', 'turbo', 'hot', 'cool', 'grays', 'bone'
        ])
        self.colormap_combo.setCurrentText('plasma')
        self.colormap_combo.setFixedWidth(100)
        self.colormap_combo.setToolTip("Color scheme for spectrogram")
        layout.addWidget(self.colormap_combo)
        
        # Invert colormap
        self.invert_colormap = QPushButton("Inv")
        self.invert_colormap.setCheckable(True)
        self.invert_colormap.setFixedWidth(40)
        self.invert_colormap.setToolTip("Invert colormap colors")
        layout.addWidget(self.invert_colormap)
        
        self._add_separator(layout)
        
        # Window function
        layout.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems([
            'blackman', 'blackmanharris', 'hann', 'hamming',
            'kaiser', 'flattop', 'rectangular'
        ])
        self.window_combo.setCurrentText('hamming')
        self.window_combo.setFixedWidth(110)
        self.window_combo.setToolTip("Window function for FFT")
        layout.addWidget(self.window_combo)
        
        self._add_separator(layout)
        
        # Advanced settings button
        self.advanced_button = QToolButton()
        self.advanced_button.setText("Settings")
        self.advanced_button.setPopupMode(QToolButton.InstantPopup)
        self.advanced_button.setToolTip("Advanced FFT and display settings")
        layout.addWidget(self.advanced_button)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setFixedWidth(90)
        self.refresh_button.setToolTip("Recompute spectrogram (F5)")
        layout.addWidget(self.refresh_button)
        
        self._add_separator(layout)
        
        # Normalization mode selector (prominent location)
        layout.addWidget(QLabel("Normalize:"))
        self.normalization_combo = QComboBox()
        self.normalization_combo.addItems(['Min-Max', 'STD'])
        self.normalization_combo.setCurrentText('STD')
        self.normalization_combo.setFixedWidth(90)
        self.normalization_combo.setToolTip("Normalization method: Min-Max (range) or STD (statistical)")
        layout.addWidget(self.normalization_combo)
        
        layout.addStretch()
        
        self.setup_advanced_menu()
    
    def _add_separator(self, layout):
        """Add a vertical separator."""
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: rgba(255,255,255,0.2);")
        layout.addWidget(sep)
    
    def setup_advanced_menu(self):
        """Create advanced settings dropdown."""
        self.advanced_menu = QMenu(self)
        self.advanced_menu.setMinimumWidth(380)
        self.advanced_button.setMenu(self.advanced_menu)
        
        advanced_widget = QWidget()
        advanced_layout = QVBoxLayout(advanced_widget)
        advanced_layout.setContentsMargins(12, 10, 12, 10)
        advanced_layout.setSpacing(12)
        
        # FFT Parameters section
        fft_group = self._create_group("FFT Parameters")
        fft_layout = fft_group.layout()
        
        fft_row = QHBoxLayout()
        fft_row.setSpacing(10)
        
        fft_row.addWidget(QLabel("FFT Size:"))
        self.fft_size_combo = QComboBox()
        self.fft_size_combo.addItems(['512', '1024', '2048', '4096', '8192', '16384'])
        self.fft_size_combo.setCurrentText('4096')
        self.fft_size_combo.setFixedWidth(90)
        fft_row.addWidget(self.fft_size_combo)
        
        fft_row.addWidget(QLabel("Overlap:"))
        self.overlap_combo = QComboBox()
        self.overlap_combo.addItems(['50%', '75%', '87.5%', '93.75%', '96.875%'])
        self.overlap_combo.setCurrentText('87.5%')
        self.overlap_combo.setFixedWidth(80)
        self.overlap_combo.setToolTip("Higher overlap = smoother but slower")
        fft_row.addWidget(self.overlap_combo)
        
        self.adaptive_fft_check = QPushButton("Auto-adapt")
        self.adaptive_fft_check.setCheckable(True)
        self.adaptive_fft_check.setChecked(True)
        self.adaptive_fft_check.setFixedWidth(90)
        fft_row.addWidget(self.adaptive_fft_check)
        
        fft_row.addStretch()
        fft_layout.addLayout(fft_row)
        advanced_layout.addWidget(fft_group)
        
        # Display Settings section
        display_group = self._create_group("Display Settings")
        display_layout = display_group.layout()
        
        display_row = QHBoxLayout()
        display_row.setSpacing(10)
        
        display_row.addWidget(QLabel("Interpolation:"))
        self.interpolation_combo = QComboBox()
        self.interpolation_combo.addItems(['nearest', 'bilinear', 'bicubic'])
        self.interpolation_combo.setCurrentText('bilinear')
        self.interpolation_combo.setFixedWidth(90)
        display_row.addWidget(self.interpolation_combo)
        
        display_row.addWidget(QLabel("Freq Scale:"))
        self.freq_scale_combo = QComboBox()
        self.freq_scale_combo.addItems(['linear', 'log', 'mel'])
        self.freq_scale_combo.setCurrentText('linear')
        self.freq_scale_combo.setFixedWidth(80)
        display_row.addWidget(self.freq_scale_combo)
        
        display_row.addStretch()
        display_layout.addLayout(display_row)
        advanced_layout.addWidget(display_group)
        
        # dB Range section
        db_group = self._create_group("Dynamic Range (dB)")
        db_layout = db_group.layout()
        
        # Auto button in title
        db_title_row = QHBoxLayout()
        self.auto_db_range = QPushButton("Auto")
        self.auto_db_range.setCheckable(True)
        self.auto_db_range.setChecked(True)
        self.auto_db_range.setFixedWidth(50)
        db_title_row.addWidget(self.auto_db_range)
        db_title_row.addStretch()
        db_layout.addLayout(db_title_row)
        
        slider_row = QHBoxLayout()
        slider_row.setSpacing(8)
        
        slider_row.addWidget(QLabel("Min:"))
        self.db_min_slider = QSlider(Qt.Horizontal)
        self.db_min_slider.setRange(-120, 0)
        self.db_min_slider.setValue(-80)
        self.db_min_slider.setFixedWidth(120)
        slider_row.addWidget(self.db_min_slider)
        self.db_min_label = QLabel("-80 dB")
        self.db_min_label.setFixedWidth(50)
        slider_row.addWidget(self.db_min_label)
        
        slider_row.addWidget(QLabel("Max:"))
        self.db_max_slider = QSlider(Qt.Horizontal)
        self.db_max_slider.setRange(-60, 20)
        self.db_max_slider.setValue(0)
        self.db_max_slider.setFixedWidth(120)
        slider_row.addWidget(self.db_max_slider)
        self.db_max_label = QLabel("0 dB")
        self.db_max_label.setFixedWidth(50)
        slider_row.addWidget(self.db_max_label)
        
        slider_row.addStretch()
        db_layout.addLayout(slider_row)
        advanced_layout.addWidget(db_group)
        
        # Normalization Mode section
        norm_group = self._create_group("Normalization Mode")
        norm_layout = norm_group.layout()
        
        norm_row = QHBoxLayout()
        norm_row.setSpacing(10)
        
        norm_row.addWidget(QLabel("Method:"))
        self.advanced_norm_combo = QComboBox()
        self.advanced_norm_combo.addItems(['Min-Max', 'STD'])
        self.advanced_norm_combo.setCurrentText('STD')
        self.advanced_norm_combo.setFixedWidth(100)
        self.advanced_norm_combo.setToolTip("Min-Max: Use dB range. STD: Use statistical normalization (mean ± std)")
        norm_row.addWidget(self.advanced_norm_combo)
        
        # Sync with main combo
        self.advanced_norm_combo.currentTextChanged.connect(
            lambda text: self.normalization_combo.setCurrentText(text) if self.normalization_combo.currentText() != text else None
        )
        self.normalization_combo.currentTextChanged.connect(
            lambda text: self.advanced_norm_combo.setCurrentText(text) if self.advanced_norm_combo.currentText() != text else None
        )
        
        norm_row.addStretch()
        norm_layout.addLayout(norm_row)
        advanced_layout.addWidget(norm_group)
        
        # Style
        advanced_widget.setStyleSheet("""
            QFrame#settings_group {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 6px;
            }
        """)
        
        advanced_action = QWidgetAction(self.advanced_menu)
        advanced_action.setDefaultWidget(advanced_widget)
        self.advanced_menu.addAction(advanced_action)
    
    def _create_group(self, title: str) -> QFrame:
        """Create a styled group frame."""
        group = QFrame()
        group.setObjectName("settings_group")
        layout = QVBoxLayout(group)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("font-weight: bold; color: #8B9DC3;")
        layout.addWidget(title_label)
        
        return group
    
    def connect_signals(self):
        """Connect widget signals."""
        self.fft_size_combo.currentTextChanged.connect(self.emit_parameters_changed)
        self.overlap_combo.currentTextChanged.connect(self.emit_parameters_changed)
        self.window_combo.currentTextChanged.connect(self.on_window_changed)
        self.colormap_combo.currentTextChanged.connect(self.on_colormap_changed_realtime)
        self.invert_colormap.toggled.connect(self.on_colormap_invert_toggled)
        self.db_min_slider.valueChanged.connect(self.update_db_range_realtime)
        self.db_max_slider.valueChanged.connect(self.update_db_range_realtime)
        self.auto_db_range.toggled.connect(self.on_auto_db_toggled)
        self.interpolation_combo.currentTextChanged.connect(self.on_interpolation_changed)
        self.freq_scale_combo.currentTextChanged.connect(self.on_freq_scale_changed)
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
        self.normalization_combo.currentTextChanged.connect(self.on_normalization_mode_changed)
    
    def emit_parameters_changed(self):
        """Emit parameters changed signal."""
        fft_size = int(self.fft_size_combo.currentText())
        
        overlap_str = self.overlap_combo.currentText().replace('%', '')
        overlap_pct = float(overlap_str) / 100.0
        hop_length = max(1, int(fft_size * (1.0 - overlap_pct)))
        
        params = {
            'fft_size': fft_size,
            'hop_length': hop_length,
            'window_type': self.window_combo.currentText()
        }
        self.parameters_changed.emit(params)
    
    def on_window_changed(self, window_type: str):
        """Handle window function change."""
        self.window_changed.emit(window_type)
        self.emit_parameters_changed()
    
    def on_colormap_changed_realtime(self, colormap: str):
        """Handle colormap change."""
        if self.invert_colormap.isChecked():
            colormap = colormap + '_r' if not colormap.endswith('_r') else colormap[:-2]
        self.colormap_changed.emit(colormap)
    
    def on_colormap_invert_toggled(self, checked: bool):
        """Handle colormap inversion."""
        self.on_colormap_changed_realtime(self.colormap_combo.currentText())
    
    def on_interpolation_changed(self, interpolation: str):
        """Handle interpolation change."""
        self.interpolation_changed.emit(interpolation)
    
    def on_freq_scale_changed(self, scale: str):
        """Handle frequency scale change."""
        self.freq_scale_changed.emit(scale)
    
    def on_auto_db_toggled(self, auto: bool):
        """Handle auto dB toggle."""
        self.db_min_slider.setEnabled(not auto)
        self.db_max_slider.setEnabled(not auto)
    
    def update_db_range_realtime(self):
        """Handle dB range changes."""
        db_min = self.db_min_slider.value()
        db_max = self.db_max_slider.value()
        
        if db_min >= db_max:
            if self.sender() == self.db_min_slider:
                db_max = db_min + 10
                self.db_max_slider.setValue(db_max)
            else:
                db_min = db_max - 10
                self.db_min_slider.setValue(db_min)
        
        self.db_min_label.setText(f"{db_min} dB")
        self.db_max_label.setText(f"{db_max} dB")
        self.db_range_changed.emit(db_min, db_max)
    
    def is_adaptive_fft_enabled(self) -> bool:
        """Check if adaptive FFT is enabled."""
        return self.adaptive_fft_check.isChecked()
    
    def get_current_window(self) -> str:
        """Get current window function."""
        return self.window_combo.currentText()
    
    def get_current_interpolation(self) -> str:
        """Get current interpolation mode."""
        return self.interpolation_combo.currentText()
    
    def get_current_freq_scale(self) -> str:
        """Get current frequency scale."""
        return self.freq_scale_combo.currentText()
    
    def on_normalization_mode_changed(self, mode_text: str):
        """Handle normalization mode change."""
        # Convert display text to internal mode
        mode = 'std' if mode_text == 'STD' else 'minmax'
        self.normalization_mode_changed.emit(mode)
    
    def get_normalization_mode(self) -> str:
        """Get current normalization mode ('minmax' or 'std')."""
        mode_text = self.normalization_combo.currentText()
        return 'std' if mode_text == 'STD' else 'minmax'

