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


class MeijeringDialog(FilterParameterDialog):
    """Dialog for Meijering ridge detection parameters."""

    def __init__(self, parent=None):
        super().__init__(
            "Meijering Ridge Detection",
            {
                'sigma_min': {
                    'type': 'int',
                    'label': 'Min sigma',
                    'default': 1,
                    'min': 1,
                    'max': 10,
                    'tooltip': 'Minimum scale for ridge detection'
                },
                'sigma_max': {
                    'type': 'int',
                    'label': 'Max sigma',
                    'default': 4,
                    'min': 1,
                    'max': 20,
                    'tooltip': 'Maximum scale for ridge detection'
                },
                'black_ridges': {
                    'type': 'bool',
                    'label': 'Black ridges',
                    'default': False,
                    'tooltip': 'Check for dark ridges on bright background'
                }
            },
            parent
        )


# ============================================================================
# Advanced Track Removal Dialogs
# ============================================================================

class HorizontalLineRemovalDialog(FilterParameterDialog):
    """Dialog for horizontal line (constant frequency) removal."""

    def __init__(self, parent=None):
        super().__init__(
            "Remove Horizontal Lines (Constant Frequencies)",
            {
                'threshold_percentile': {
                    'type': 'float',
                    'label': 'Detection threshold (percentile)',
                    'default': 95.0,
                    'min': 50.0,
                    'max': 99.9,
                    'step': 1.0,
                    'tooltip': 'Lines above this percentile of row energy are removed'
                },
                'min_width_ratio': {
                    'type': 'float',
                    'label': 'Min width ratio',
                    'default': 0.7,
                    'min': 0.1,
                    'max': 1.0,
                    'step': 0.05,
                    'tooltip': 'Minimum fraction of row that must be "on" to be considered a line'
                },
                'method': {
                    'type': 'choice',
                    'label': 'Removal method',
                    'choices': ['median', 'interpolate', 'zero'],
                    'default': 'median',
                    'tooltip': 'How to fill removed lines'
                }
            },
            parent
        )


class VerticalLineRemovalDialog(FilterParameterDialog):
    """Dialog for vertical line (impulse/click) removal."""

    def __init__(self, parent=None):
        super().__init__(
            "Remove Vertical Lines (Clicks/Impulses)",
            {
                'threshold_percentile': {
                    'type': 'float',
                    'label': 'Detection threshold (percentile)',
                    'default': 95.0,
                    'min': 50.0,
                    'max': 99.9,
                    'step': 1.0,
                    'tooltip': 'Columns above this percentile of energy are removed'
                },
                'min_height_ratio': {
                    'type': 'float',
                    'label': 'Min height ratio',
                    'default': 0.5,
                    'min': 0.1,
                    'max': 1.0,
                    'step': 0.05,
                    'tooltip': 'Minimum fraction of column that must be "on"'
                },
                'method': {
                    'type': 'choice',
                    'label': 'Removal method',
                    'choices': ['median', 'interpolate', 'zero'],
                    'default': 'interpolate',
                    'tooltip': 'How to fill removed lines'
                }
            },
            parent
        )


class SpectralSubtractionDialog(FilterParameterDialog):
    """Dialog for spectral subtraction (noise floor removal)."""

    def __init__(self, parent=None):
        super().__init__(
            "Spectral Subtraction (Noise Removal)",
            {
                'noise_percentile': {
                    'type': 'float',
                    'label': 'Noise estimation percentile',
                    'default': 10.0,
                    'min': 1.0,
                    'max': 50.0,
                    'step': 1.0,
                    'tooltip': 'Use this percentile of each frequency bin as noise estimate'
                },
                'subtraction_factor': {
                    'type': 'float',
                    'label': 'Subtraction factor',
                    'default': 1.0,
                    'min': 0.1,
                    'max': 5.0,
                    'step': 0.1,
                    'tooltip': 'Multiply noise estimate by this factor before subtracting'
                },
                'floor': {
                    'type': 'float',
                    'label': 'Floor value',
                    'default': 0.0,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'decimals': 3,
                    'tooltip': 'Minimum value after subtraction (prevents negative values)'
                }
            },
            parent
        )


class PCENDialog(FilterParameterDialog):
    """Dialog for Per-Channel Energy Normalization (PCEN)."""

    def __init__(self, parent=None):
        super().__init__(
            "PCEN (Per-Channel Energy Normalization)",
            {
                'time_constant': {
                    'type': 'float',
                    'label': 'Time constant (frames)',
                    'default': 0.06,
                    'min': 0.001,
                    'max': 1.0,
                    'step': 0.01,
                    'decimals': 3,
                    'tooltip': 'Smoothing time constant (smaller = faster adaptation)'
                },
                'gain': {
                    'type': 'float',
                    'label': 'Gain',
                    'default': 0.98,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': 'AGC strength (higher = more normalization)'
                },
                'power': {
                    'type': 'float',
                    'label': 'Power (compression)',
                    'default': 0.5,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.05,
                    'tooltip': 'Compression exponent (0.5 = square root compression)'
                },
                'bias': {
                    'type': 'float',
                    'label': 'Bias',
                    'default': 2.0,
                    'min': 0.1,
                    'max': 20.0,
                    'step': 0.5,
                    'tooltip': 'Bias added before compression'
                },
                'eps': {
                    'type': 'float',
                    'label': 'Epsilon',
                    'default': 1e-6,
                    'min': 1e-10,
                    'max': 1e-3,
                    'step': 1e-7,
                    'decimals': 10,
                    'tooltip': 'Small constant to prevent division by zero'
                }
            },
            parent
        )


class MorphologicalDialog(FilterParameterDialog):
    """Dialog for morphological operations."""

    def __init__(self, parent=None):
        super().__init__(
            "Morphological Filter",
            {
                'operation': {
                    'type': 'choice',
                    'label': 'Operation',
                    'choices': ['erosion', 'dilation', 'opening', 'closing',
                               'gradient', 'tophat', 'blackhat'],
                    'default': 'opening',
                    'tooltip': 'Morphological operation type'
                },
                'kernel_width': {
                    'type': 'int',
                    'label': 'Kernel width (time)',
                    'default': 3,
                    'min': 1,
                    'max': 21,
                    'step': 2,
                    'tooltip': 'Structuring element width'
                },
                'kernel_height': {
                    'type': 'int',
                    'label': 'Kernel height (freq)',
                    'default': 3,
                    'min': 1,
                    'max': 21,
                    'step': 2,
                    'tooltip': 'Structuring element height'
                }
            },
            parent
        )


class AdaptiveNoiseGateDialog(FilterParameterDialog):
    """Dialog for adaptive noise gate."""

    def __init__(self, parent=None):
        super().__init__(
            "Adaptive Noise Gate",
            {
                'window_time': {
                    'type': 'int',
                    'label': 'Time window (frames)',
                    'default': 50,
                    'min': 5,
                    'max': 500,
                    'step': 5,
                    'tooltip': 'Local window size for noise estimation (time axis)'
                },
                'window_freq': {
                    'type': 'int',
                    'label': 'Frequency window (bins)',
                    'default': 10,
                    'min': 1,
                    'max': 100,
                    'step': 1,
                    'tooltip': 'Local window size for noise estimation (freq axis)'
                },
                'threshold_db': {
                    'type': 'float',
                    'label': 'Threshold above noise (dB)',
                    'default': 6.0,
                    'min': 0.0,
                    'max': 30.0,
                    'step': 1.0,
                    'tooltip': 'Signal must be this many dB above local noise to pass'
                },
                'soft_knee': {
                    'type': 'bool',
                    'label': 'Soft knee',
                    'default': True,
                    'tooltip': 'Use soft transition instead of hard gate'
                }
            },
            parent
        )


class HarmonicEnhanceDialog(FilterParameterDialog):
    """Dialog for harmonic track enhancement."""

    def __init__(self, parent=None):
        super().__init__(
            "Harmonic Track Enhancement",
            {
                'fundamental_range_low': {
                    'type': 'float',
                    'label': 'Fundamental freq min (Hz)',
                    'default': 50.0,
                    'min': 10.0,
                    'max': 1000.0,
                    'step': 10.0,
                    'tooltip': 'Minimum expected fundamental frequency'
                },
                'fundamental_range_high': {
                    'type': 'float',
                    'label': 'Fundamental freq max (Hz)',
                    'default': 500.0,
                    'min': 50.0,
                    'max': 5000.0,
                    'step': 50.0,
                    'tooltip': 'Maximum expected fundamental frequency'
                },
                'n_harmonics': {
                    'type': 'int',
                    'label': 'Number of harmonics',
                    'default': 5,
                    'min': 1,
                    'max': 20,
                    'step': 1,
                    'tooltip': 'Number of harmonic partials to enhance'
                },
                'enhancement_factor': {
                    'type': 'float',
                    'label': 'Enhancement factor',
                    'default': 2.0,
                    'min': 1.0,
                    'max': 10.0,
                    'step': 0.5,
                    'tooltip': 'How much to boost harmonic content'
                }
            },
            parent
        )


class TrackSuppressionDialog(FilterParameterDialog):
    """Dialog for general track/line suppression using Hough or ridge detection."""

    def __init__(self, parent=None):
        super().__init__(
            "Track Suppression (Line Removal)",
            {
                'method': {
                    'type': 'choice',
                    'label': 'Detection method',
                    'choices': ['ridge', 'gradient', 'hough'],
                    'default': 'ridge',
                    'tooltip': 'Method to detect tracks'
                },
                'sigma': {
                    'type': 'float',
                    'label': 'Detection scale (sigma)',
                    'default': 2.0,
                    'min': 0.5,
                    'max': 10.0,
                    'step': 0.5,
                    'tooltip': 'Scale of features to detect'
                },
                'threshold': {
                    'type': 'float',
                    'label': 'Detection threshold',
                    'default': 0.1,
                    'min': 0.01,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': 'Threshold for track detection (0-1)'
                },
                'suppression_strength': {
                    'type': 'float',
                    'label': 'Suppression strength',
                    'default': 0.8,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.1,
                    'tooltip': '0 = no suppression, 1 = full removal'
                },
                'inpaint': {
                    'type': 'bool',
                    'label': 'Inpaint removed regions',
                    'default': True,
                    'tooltip': 'Fill removed regions with interpolated values'
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


# ============================================================================
# Advanced Filter Dialogs
# ============================================================================

class WienerFilterDialog(FilterParameterDialog):
    """Dialog for Wiener filter parameters."""

    def __init__(self, parent=None):
        super().__init__(
            "Wiener Filter",
            {
                'window_size': {
                    'type': 'int',
                    'label': 'Window size',
                    'default': 5,
                    'min': 3,
                    'max': 31,
                    'step': 2,
                    'tooltip': 'Local window for variance estimation (odd)'
                },
                'noise_variance': {
                    'type': 'float',
                    'label': 'Noise variance (0=auto)',
                    'default': 0.0,
                    'min': 0.0,
                    'max': 1.0,
                    'step': 0.001,
                    'decimals': 4,
                    'tooltip': 'Estimated noise variance (0 = auto-estimate)'
                }
            },
            parent
        )


class BilateralFilterDialog(FilterParameterDialog):
    """Dialog for bilateral filter parameters."""

    def __init__(self, parent=None):
        super().__init__(
            "Bilateral Filter (Edge-Preserving)",
            {
                'sigma_spatial': {
                    'type': 'float',
                    'label': 'Spatial sigma',
                    'default': 3.0,
                    'min': 0.5,
                    'max': 20.0,
                    'step': 0.5,
                    'tooltip': 'Spatial smoothing strength'
                },
                'sigma_color': {
                    'type': 'float',
                    'label': 'Color/intensity sigma',
                    'default': 0.1,
                    'min': 0.01,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': 'Intensity similarity threshold (lower = more edge preservation)'
                }
            },
            parent
        )


class HarmonicPercussiveDialog(FilterParameterDialog):
    """Dialog for harmonic-percussive separation."""

    def __init__(self, parent=None):
        super().__init__(
            "Harmonic-Percussive Separation",
            {
                'kernel_size_harmonic': {
                    'type': 'int',
                    'label': 'Harmonic kernel size',
                    'default': 31,
                    'min': 3,
                    'max': 101,
                    'step': 2,
                    'tooltip': 'Median filter size for harmonic extraction (time axis)'
                },
                'kernel_size_percussive': {
                    'type': 'int',
                    'label': 'Percussive kernel size',
                    'default': 31,
                    'min': 3,
                    'max': 101,
                    'step': 2,
                    'tooltip': 'Median filter size for percussive extraction (freq axis)'
                },
                'output': {
                    'type': 'choice',
                    'label': 'Output component',
                    'choices': ['harmonic', 'percussive', 'residual'],
                    'default': 'harmonic',
                    'tooltip': 'Which component to extract'
                }
            },
            parent
        )


class SpectralGatingDialog(FilterParameterDialog):
    """Dialog for spectral gating parameters."""

    def __init__(self, parent=None):
        super().__init__(
            "Spectral Gating",
            {
                'noise_percentile': {
                    'type': 'float',
                    'label': 'Noise percentile',
                    'default': 10.0,
                    'min': 1.0,
                    'max': 50.0,
                    'step': 1.0,
                    'tooltip': 'Percentile for noise floor estimation'
                },
                'threshold_db': {
                    'type': 'float',
                    'label': 'Threshold (dB)',
                    'default': -20.0,
                    'min': -60.0,
                    'max': 0.0,
                    'step': 1.0,
                    'tooltip': 'Threshold in dB above noise floor'
                },
                'smoothing_time': {
                    'type': 'int',
                    'label': 'Time smoothing',
                    'default': 5,
                    'min': 1,
                    'max': 31,
                    'step': 2,
                    'tooltip': 'Smoothing window for mask (time frames)'
                },
                'smoothing_freq': {
                    'type': 'int',
                    'label': 'Freq smoothing',
                    'default': 3,
                    'min': 1,
                    'max': 31,
                    'step': 2,
                    'tooltip': 'Smoothing window for mask (freq bins)'
                }
            },
            parent
        )


class TotalVariationDialog(FilterParameterDialog):
    """Dialog for Total Variation denoising parameters."""

    def __init__(self, parent=None):
        super().__init__(
            "Total Variation Denoising",
            {
                'weight': {
                    'type': 'float',
                    'label': 'Denoising weight',
                    'default': 0.1,
                    'min': 0.01,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': 'Higher = more smoothing, lower = more detail preserved'
                }
            },
            parent
        )


class NonLocalMeansDialog(FilterParameterDialog):
    """Dialog for Non-Local Means denoising parameters."""

    def __init__(self, parent=None):
        super().__init__(
            "Non-Local Means Denoising",
            {
                'patch_size': {
                    'type': 'int',
                    'label': 'Patch size',
                    'default': 5,
                    'min': 3,
                    'max': 15,
                    'step': 2,
                    'tooltip': 'Size of patches to compare'
                },
                'patch_distance': {
                    'type': 'int',
                    'label': 'Search distance',
                    'default': 6,
                    'min': 1,
                    'max': 20,
                    'step': 1,
                    'tooltip': 'Maximum distance to search for similar patches'
                },
                'h': {
                    'type': 'float',
                    'label': 'Filter strength (h)',
                    'default': 0.1,
                    'min': 0.01,
                    'max': 1.0,
                    'step': 0.01,
                    'tooltip': 'Higher = more smoothing'
                }
            },
            parent
        )


class LocalContrastNormDialog(FilterParameterDialog):
    """Dialog for local contrast normalization parameters."""

    def __init__(self, parent=None):
        super().__init__(
            "Local Contrast Normalization",
            {
                'window_size': {
                    'type': 'int',
                    'label': 'Window size',
                    'default': 51,
                    'min': 3,
                    'max': 201,
                    'step': 2,
                    'tooltip': 'Local window size for normalization (odd)'
                }
            },
            parent
        )


class CLAHEDialog(FilterParameterDialog):
    """Dialog for CLAHE (Contrast Limited Adaptive Histogram Equalization) parameters."""

    def __init__(self, parent=None):
        super().__init__(
            "CLAHE - Adaptive Histogram Equalization",
            {
                'clip_limit': {
                    'type': 'float',
                    'label': 'Clip Limit',
                    'default': 2.0,
                    'min': 0.5,
                    'max': 10.0,
                    'step': 0.5,
                    'tooltip': 'Contrast limit threshold (higher = more contrast, but may amplify noise)'
                },
                'tile_grid_size': {
                    'type': 'int',
                    'label': 'Tile Grid Size',
                    'default': 8,
                    'min': 2,
                    'max': 32,
                    'step': 1,
                    'tooltip': 'Size of the grid for local histogram equalization'
                }
            },
            parent
        )
