"""
Filter parameter dialogs for spectrogram processing.

Provides user-configurable dialogs for various filters with real-time preview.
"""
import logging
from typing import Dict, Any, Optional

from .qt_compat import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QSlider, QDialogButtonBox, Qt, Signal
)

logger = logging.getLogger(__name__)


class FilterParameterDialog(QDialog):
    """Generic dialog for filter parameters."""

    def __init__(self, title: str, parameters: Dict[str, Dict], parent=None):
        """Initialize filter dialog.

        Args:
            title: Dialog title
            parameters: Dict of parameter definitions:
                {
                    'param_name': {
                        'type': 'float' | 'int' | 'bool' | 'choice',
                        'label': 'Display Label',
                        'default': value,
                        'min': min_value,  # for numeric
                        'max': max_value,  # for numeric
                        'step': step_value,  # for numeric
                        'choices': ['a', 'b'],  # for choice type
                        'tooltip': 'Description'
                    }
                }
            parent: Parent widget
        """
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(350)
        self.parameters = parameters
        self.widgets = {}

        self._setup_ui()

    def _setup_ui(self):
        """Setup the dialog UI."""
        layout = QVBoxLayout(self)

        # Parameters group
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(10)

        for param_name, config in self.parameters.items():
            widget = self._create_widget(config)
            self.widgets[param_name] = widget
            label = config.get('label', param_name)
            params_layout.addRow(f"{label}:", widget)

            if 'tooltip' in config:
                widget.setToolTip(config['tooltip'])

        layout.addWidget(params_group)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _create_widget(self, config: Dict) -> Any:
        """Create appropriate widget for parameter type."""
        param_type = config.get('type', 'float')

        if param_type == 'float':
            widget = QDoubleSpinBox()
            widget.setRange(config.get('min', 0.0), config.get('max', 100.0))
            widget.setValue(config.get('default', 1.0))
            widget.setSingleStep(config.get('step', 0.1))
            widget.setDecimals(config.get('decimals', 2))

        elif param_type == 'int':
            widget = QSpinBox()
            widget.setRange(config.get('min', 0), config.get('max', 100))
            widget.setValue(config.get('default', 1))
            widget.setSingleStep(config.get('step', 1))

        elif param_type == 'bool':
            widget = QCheckBox()
            widget.setChecked(config.get('default', False))

        elif param_type == 'choice':
            widget = QComboBox()
            choices = config.get('choices', [])
            widget.addItems(choices)
            default = config.get('default', choices[0] if choices else '')
            idx = widget.findText(default)
            if idx >= 0:
                widget.setCurrentIndex(idx)

        else:
            widget = QLabel(f"Unknown type: {param_type}")

        return widget

    def get_values(self) -> Dict[str, Any]:
        """Get current parameter values."""
        values = {}
        for param_name, widget in self.widgets.items():
            config = self.parameters[param_name]
            param_type = config.get('type', 'float')

            if param_type in ('float', 'int'):
                values[param_name] = widget.value()
            elif param_type == 'bool':
                values[param_name] = widget.isChecked()
            elif param_type == 'choice':
                values[param_name] = widget.currentText()

        return values


# ============================================================================
# Specific Filter Dialogs with Presets
# ============================================================================

class GaussianBlurDialog(FilterParameterDialog):
    """Dialog for Gaussian blur parameters."""

    def __init__(self, parent=None):
        super().__init__(
            "Gaussian Blur",
            {
                'sigma': {
                    'type': 'float',
                    'label': 'Sigma (blur strength)',
                    'default': 1.0,
                    'min': 0.1,
                    'max': 20.0,
                    'step': 0.5,
                    'tooltip': 'Standard deviation of Gaussian kernel.\nHigher = more blur.'
                }
            },
            parent
        )


class MedianFilterDialog(FilterParameterDialog):
    """Dialog for median filter parameters."""

    def __init__(self, parent=None):
        super().__init__(
            "Median Filter",
            {
                'size': {
                    'type': 'int',
                    'label': 'Kernel size',
                    'default': 3,
                    'min': 3,
                    'max': 15,
                    'step': 2,
                    'tooltip': 'Size of median filter kernel (must be odd).\nLarger = stronger noise removal.'
                }
            },
            parent
        )


class ContrastEnhanceDialog(FilterParameterDialog):
    """Dialog for contrast enhancement parameters."""

    def __init__(self, parent=None):
        super().__init__(
            "Contrast Enhancement",
            {
                'percentile_low': {
                    'type': 'float',
                    'label': 'Low percentile',
                    'default': 2.0,
                    'min': 0.0,
                    'max': 49.0,
                    'step': 1.0,
                    'tooltip': 'Clip values below this percentile'
                },
                'percentile_high': {
                    'type': 'float',
                    'label': 'High percentile',
                    'default': 98.0,
                    'min': 51.0,
                    'max': 100.0,
                    'step': 1.0,
                    'tooltip': 'Clip values above this percentile'
                }
            },
            parent
        )


class ThresholdDialog(FilterParameterDialog):
    """Dialog for threshold filter parameters."""

    def __init__(self, parent=None):
        super().__init__(
            "Threshold",
            {
                'method': {
                    'type': 'choice',
                    'label': 'Method',
                    'choices': ['manual', 'otsu', 'percentile'],
                    'default': 'percentile',
                    'tooltip': 'Thresholding method'
                },
                'value': {
                    'type': 'float',
                    'label': 'Threshold value / percentile',
                    'default': 50.0,
                    'min': 0.0,
                    'max': 100.0,
                    'step': 1.0,
                    'tooltip': 'Manual threshold (0-1 range) or percentile'
                },
                'binary': {
                    'type': 'bool',
                    'label': 'Binary output',
                    'default': False,
                    'tooltip': 'If checked, output is 0/1. Otherwise, zeros below threshold.'
                }
            },
            parent
        )





# ============================================================================
# Frequency Domain Filter Dialogs
# ============================================================================

class LowPassFilterDialog(FilterParameterDialog):
    """Dialog for low-pass filter parameters."""

    def __init__(self, parent=None):
        super().__init__(
            "Low-Pass Filter",
            {
                'cutoff_bin': {
                    'type': 'int',
                    'label': 'Cutoff frequency (bin)',
                    'default': 500,
                    'min': 1,
                    'max': 10000,
                    'step': 10,
                    'tooltip': 'Frequency bin above which to attenuate'
                },
                'rolloff': {
                    'type': 'float',
                    'label': 'Rolloff steepness',
                    'default': 10.0,
                    'min': 0.5,
                    'max': 100.0,
                    'step': 1.0,
                    'tooltip': 'Higher = steeper cutoff transition'
                }
            },
            parent
        )


class HighPassFilterDialog(FilterParameterDialog):
    """Dialog for high-pass filter parameters."""

    def __init__(self, parent=None):
        super().__init__(
            "High-Pass Filter",
            {
                'cutoff_bin': {
                    'type': 'int',
                    'label': 'Cutoff frequency (bin)',
                    'default': 50,
                    'min': 1,
                    'max': 10000,
                    'step': 10,
                    'tooltip': 'Frequency bin below which to attenuate'
                },
                'rolloff': {
                    'type': 'float',
                    'label': 'Rolloff steepness',
                    'default': 10.0,
                    'min': 0.5,
                    'max': 100.0,
                    'step': 1.0,
                    'tooltip': 'Higher = steeper cutoff transition'
                }
            },
            parent
        )


class BandPassFilterDialog(FilterParameterDialog):
    """Dialog for band-pass filter parameters."""

    def __init__(self, parent=None):
        super().__init__(
            "Band-Pass Filter",
            {
                'low_cutoff_bin': {
                    'type': 'int',
                    'label': 'Low cutoff (bin)',
                    'default': 50,
                    'min': 1,
                    'max': 10000,
                    'step': 10,
                    'tooltip': 'Lower frequency cutoff bin'
                },
                'high_cutoff_bin': {
                    'type': 'int',
                    'label': 'High cutoff (bin)',
                    'default': 500,
                    'min': 1,
                    'max': 10000,
                    'step': 10,
                    'tooltip': 'Upper frequency cutoff bin'
                },
                'rolloff': {
                    'type': 'float',
                    'label': 'Rolloff steepness',
                    'default': 10.0,
                    'min': 0.5,
                    'max': 100.0,
                    'step': 1.0,
                    'tooltip': 'Higher = steeper cutoff transition'
                }
            },
            parent
        )


class BandStopFilterDialog(FilterParameterDialog):
    """Dialog for band-stop (notch) filter parameters."""

    def __init__(self, parent=None):
        super().__init__(
            "Band-Stop Filter (Notch)",
            {
                'low_cutoff_bin': {
                    'type': 'int',
                    'label': 'Low cutoff (bin)',
                    'default': 100,
                    'min': 1,
                    'max': 10000,
                    'step': 10,
                    'tooltip': 'Lower frequency cutoff bin'
                },
                'high_cutoff_bin': {
                    'type': 'int',
                    'label': 'High cutoff (bin)',
                    'default': 200,
                    'min': 1,
                    'max': 10000,
                    'step': 10,
                    'tooltip': 'Upper frequency cutoff bin'
                },
                'rolloff': {
                    'type': 'float',
                    'label': 'Rolloff steepness',
                    'default': 10.0,
                    'min': 0.5,
                    'max': 100.0,
                    'step': 1.0,
                    'tooltip': 'Higher = steeper cutoff transition'
                }
            },
            parent
        )



