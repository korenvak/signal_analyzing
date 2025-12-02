"""
InteractionManager - handles coordinate mapping, axis formatting, and ROI selection.

This module provides:
1. Screen -> Data -> Physics coordinate transformation
2. Custom axis tick formatting for Time (mm:ss) and Frequency (kHz)
3. Interactive ROI selection with visual feedback
"""
import logging
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np

from PySide6.QtCore import QObject, Signal

try:
    from vispy import scene
    from vispy.scene import visuals
    HAS_VISPY = True
except ImportError:
    scene = None
    visuals = None
    HAS_VISPY = False

logger = logging.getLogger(__name__)


@dataclass
class AudioCoordinates:
    """Represents a point in audio coordinate space."""
    time_sec: float      # Time in seconds
    freq_hz: float       # Frequency in Hz
    frame_idx: int       # Spectrogram frame index
    bin_idx: int         # Frequency bin index
    
    @property
    def time_formatted(self) -> str:
        """Return time as mm:ss.ms"""
        minutes = int(self.time_sec // 60)
        seconds = self.time_sec % 60
        return f"{minutes:02d}:{seconds:05.2f}"
    
    @property
    def freq_formatted(self) -> str:
        """Return frequency as kHz or Hz"""
        if self.freq_hz >= 1000:
            return f"{self.freq_hz / 1000:.2f} kHz"
        return f"{self.freq_hz:.1f} Hz"


@dataclass
class ROISelection:
    """Represents a selected Region of Interest."""
    t_start: float   # Start time in seconds
    t_end: float     # End time in seconds
    f_start: float   # Start frequency in Hz  
    f_end: float     # End frequency in Hz
    
    @property
    def time_range(self) -> Tuple[float, float]:
        return (min(self.t_start, self.t_end), max(self.t_start, self.t_end))
    
    @property
    def freq_range(self) -> Tuple[float, float]:
        return (min(self.f_start, self.f_end), max(self.f_start, self.f_end))
    
    @property
    def duration(self) -> float:
        return abs(self.t_end - self.t_start)
    
    @property
    def bandwidth(self) -> float:
        return abs(self.f_end - self.f_start)


class CoordinateMapper:
    """
    Handles coordinate transformations between screen, data, and physics spaces.
    
    Coordinate Spaces:
    - Screen: pixel coordinates from mouse events (origin top-left)
    - Visual: VisPy normalized device coordinates within ViewBox
    - World/Data: actual time/frequency coordinates
    
    This correctly handles PanZoomCamera transformations.
    """
    
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512, fft_size: int = 4096):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fft_size = fft_size
        
        # ViewBox reference (set via set_view)
        self._view = None
        self._canvas = None
    
    def set_view(self, view, canvas):
        """Set the VisPy ViewBox and canvas for coordinate transforms."""
        self._view = view
        self._canvas = canvas
    
    def update_parameters(self, sample_rate: int = None, hop_length: int = None, fft_size: int = None):
        """Update audio parameters for physics coordinate conversion."""
        if sample_rate is not None:
            self.sample_rate = sample_rate
        if hop_length is not None:
            self.hop_length = hop_length
        if fft_size is not None:
            self.fft_size = fft_size
    
    def screen_to_world(self, screen_pos: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        Convert screen (pixel) coordinates to world (time, freq) coordinates.
        
        This is the CRITICAL method that handles PanZoomCamera correctly.
        
        Args:
            screen_pos: (x, y) pixel position from mouse event
        
        Returns:
            (time_seconds, frequency_hz) or None if conversion fails
        """
        if self._view is None or self._canvas is None:
            logger.warning("CoordinateMapper: view/canvas not set")
            return None
        
        try:
            screen_x, screen_y = float(screen_pos[0]), float(screen_pos[1])
            
            # Method 1: Use the scene.node_transform chain
            # This maps from canvas pixels to the scene coordinate system
            try:
                # Get the full transform from canvas to scene
                tr = self._view.scene.transform
                
                # Get the ViewBox's position in canvas coordinates
                # We need to account for the grid layout (axes around the ViewBox)
                view_to_canvas = self._view.node_transform(self._canvas)
                
                # Get ViewBox size in pixels
                view_size = self._view.size
                if view_size[0] <= 0 or view_size[1] <= 0:
                    return None
                
                # Map ViewBox corners to canvas to find its pixel bounds
                corner_00 = view_to_canvas.map((0, 0, 0, 1))[:2]
                corner_11 = view_to_canvas.map((view_size[0], view_size[1], 0, 1))[:2]
                
                vb_left = min(corner_00[0], corner_11[0])
                vb_right = max(corner_00[0], corner_11[0])
                vb_bottom = min(corner_00[1], corner_11[1])  # In screen coords, Y increases downward
                vb_top = max(corner_00[1], corner_11[1])
                
                vb_width = max(vb_right - vb_left, 1)
                vb_height = max(vb_top - vb_bottom, 1)
                
                # Check if click is within ViewBox pixel bounds
                if screen_x < vb_left or screen_x > vb_right:
                    return None
                if screen_y < vb_bottom or screen_y > vb_top:
                    return None
                
                # Normalize position within ViewBox (0 to 1)
                norm_x = (screen_x - vb_left) / vb_width
                norm_y = (screen_y - vb_bottom) / vb_height
                
                # In screen coordinates, Y=0 is at top
                # In world coordinates (spectrogram), Y=0 is at bottom (low freq)
                # So we need to flip Y
                norm_y = 1.0 - norm_y
                
                # Get camera rect (visible world coordinates)
                rect = self._view.camera.rect
                if rect is None:
                    return None
                
                # Map normalized to world coordinates
                time = float(rect.left) + norm_x * float(rect.width)
                freq = float(rect.bottom) + norm_y * float(rect.height)
                
                return (time, freq)
                
            except Exception as e:
                logger.debug(f"Transform method 1 failed: {e}")
            
            # Method 2: Direct inverse transform from scene
            try:
                # Use the camera's transform to map from canvas to scene
                # This is the recommended VisPy approach
                tr = self._view.scene.transform
                inv_tr = tr.inverse
                
                if inv_tr is not None:
                    # Map screen position to scene coordinates
                    world = inv_tr.map((screen_x, screen_y, 0, 1))
                    return (float(world[0]), float(world[1]))
            except Exception as e:
                logger.debug(f"Transform method 2 failed: {e}")
            
            return None
            
        except Exception as e:
            logger.error(f"screen_to_world failed: {e}")
            return None
    
    def world_to_audio(self, time: float, freq: float) -> AudioCoordinates:
        """
        Convert world coordinates to audio physics coordinates.
        
        Args:
            time: Time in seconds
            freq: Frequency in Hz
        
        Returns:
            AudioCoordinates with frame index and bin index
        """
        # Calculate frame index from time
        # Formula: frame_idx = time * sample_rate / hop_length
        frame_idx = int(time * self.sample_rate / self.hop_length)
        
        # Calculate frequency bin index
        # Formula: bin_idx = freq * fft_size / sample_rate
        bin_idx = int(freq * self.fft_size / self.sample_rate)
        
        return AudioCoordinates(
            time_sec=time,
            freq_hz=freq,
            frame_idx=frame_idx,
            bin_idx=bin_idx
        )
    
    def frame_to_time(self, frame_idx: int) -> float:
        """Convert frame index to time in seconds."""
        return frame_idx * self.hop_length / self.sample_rate
    
    def bin_to_freq(self, bin_idx: int) -> float:
        """Convert frequency bin index to Hz."""
        return bin_idx * self.sample_rate / self.fft_size
    
    def time_to_frame(self, time: float) -> int:
        """Convert time in seconds to frame index."""
        return int(time * self.sample_rate / self.hop_length)
    
    def freq_to_bin(self, freq: float) -> int:
        """Convert frequency in Hz to bin index."""
        return int(freq * self.fft_size / self.sample_rate)


class AxisFormatter:
    """
    Provides custom tick formatting for VisPy AxisWidget.
    
    Time axis: displays as mm:ss or ss.ms depending on zoom level
    Frequency axis: displays as kHz or Hz depending on range
    """
    
    @staticmethod
    def format_time_tick(value: float, scale: float = 1.0) -> str:
        """
        Format time value for axis tick label.
        
        Args:
            value: Time in seconds
            scale: Current axis scale (for determining precision)
        
        Returns:
            Formatted string
        """
        if value < 0:
            return ""
        
        # Determine precision based on scale
        if scale < 0.1:  # Very zoomed in - show milliseconds
            return f"{value:.3f}s"
        elif scale < 1.0:  # Moderately zoomed
            return f"{value:.2f}s"
        elif value < 60:  # Less than a minute
            return f"{value:.1f}s"
        elif value < 3600:  # Less than an hour
            minutes = int(value // 60)
            seconds = value % 60
            return f"{minutes}:{seconds:04.1f}"
        else:  # Hour or more
            hours = int(value // 3600)
            minutes = int((value % 3600) // 60)
            seconds = value % 60
            return f"{hours}:{minutes:02d}:{seconds:04.1f}"
    
    @staticmethod
    def format_freq_tick(value: float, scale: float = 1.0) -> str:
        """
        Format frequency value for axis tick label.
        
        Args:
            value: Frequency in Hz
            scale: Current axis scale
        
        Returns:
            Formatted string (kHz for large values)
        """
        if value < 0:
            return ""
        
        if value >= 10000:  # 10kHz or more
            return f"{value/1000:.1f}k"
        elif value >= 1000:  # 1-10 kHz
            return f"{value/1000:.2f}k"
        else:  # Less than 1kHz
            return f"{value:.0f}"
    
    @staticmethod
    def get_nice_time_ticks(t_min: float, t_max: float, max_ticks: int = 10) -> list:
        """
        Generate 'nice' tick values for time axis.
        
        Args:
            t_min: Minimum time (seconds)
            t_max: Maximum time (seconds)
            max_ticks: Maximum number of ticks
        
        Returns:
            List of tick values
        """
        span = t_max - t_min
        if span <= 0:
            return [t_min]
        
        # Choose nice intervals based on span
        if span < 0.1:
            interval = 0.01
        elif span < 1:
            interval = 0.1
        elif span < 10:
            interval = 1
        elif span < 60:
            interval = 5
        elif span < 300:
            interval = 30
        elif span < 600:
            interval = 60
        else:
            interval = 120
        
        # Generate ticks
        start = np.ceil(t_min / interval) * interval
        ticks = []
        current = start
        while current <= t_max and len(ticks) < max_ticks:
            ticks.append(current)
            current += interval
        
        return ticks
    
    @staticmethod
    def get_nice_freq_ticks(f_min: float, f_max: float, max_ticks: int = 10) -> list:
        """
        Generate 'nice' tick values for frequency axis.
        
        Args:
            f_min: Minimum frequency (Hz)
            f_max: Maximum frequency (Hz)
            max_ticks: Maximum number of ticks
        
        Returns:
            List of tick values
        """
        span = f_max - f_min
        if span <= 0:
            return [f_min]
        
        # Choose nice intervals based on span
        if span < 100:
            interval = 10
        elif span < 1000:
            interval = 100
        elif span < 5000:
            interval = 500
        elif span < 10000:
            interval = 1000
        else:
            interval = 2000
        
        # Generate ticks
        start = np.ceil(f_min / interval) * interval
        ticks = []
        current = start
        while current <= f_max and len(ticks) < max_ticks:
            ticks.append(current)
            current += interval
        
        return ticks


class ROIVisual:
    """
    Visual representation of an ROI selection rectangle.
    
    Uses VisPy visuals for rendering:
    - Semi-transparent fill (tints but doesn't hide spectrogram)
    - Clear colored border
    - Raven Pro style appearance
    """
    
    # Raven Pro style colors - red border with semi-transparent red fill
    DEFAULT_FILL_COLOR = (1.0, 0.0, 0.0, 0.25)     # Red, 25% opacity - visible but transparent
    DEFAULT_BORDER_COLOR = (1.0, 0.0, 0.0, 1.0)    # Solid red border
    ACTIVE_BORDER_COLOR = (1.0, 0.0, 0.0, 1.0)     # Red for active drawing
    ACTIVE_FILL_COLOR = (1.0, 0.0, 0.0, 0.35)      # Red fill while drawing (more visible)
    SELECTED_BORDER_COLOR = (1.0, 0.3, 0.0, 1.0)   # Orange for selected
    
    def __init__(self, parent_scene, border_color=None, fill_color=None):
        """
        Initialize ROI visual.
        
        Args:
            parent_scene: VisPy scene to add visuals to
            border_color: RGBA tuple for border
            fill_color: RGBA tuple for fill
        """
        if not HAS_VISPY:
            raise RuntimeError("VisPy not available")
        
        self.parent = parent_scene
        self.border_color = border_color or self.DEFAULT_BORDER_COLOR
        self.fill_color = fill_color or self.DEFAULT_FILL_COLOR
        
        # Create filled rectangle using Rectangle visual
        # This gives us a proper filled rectangle with border
        self._roi_rect = scene.visuals.Rectangle(
            center=(0.5, 0.5),
            width=1.0,
            height=1.0,
            color=self.fill_color,
            border_color=self.border_color,
            border_width=2.5,
            parent=parent_scene
        )
        self._roi_rect.visible = False
        self._roi_rect.order = 100  # Render on top of spectrogram
        self._roi_rect.set_gl_state('translucent', depth_test=False)
        
        self._bounds = None
    
    def update(self, t_start: float, t_end: float, f_start: float, f_end: float):
        """
        Update ROI visual position and size.
        
        Args:
            t_start: Start time
            t_end: End time  
            f_start: Start frequency
            f_end: End frequency
        """
        # Ensure correct ordering
        t_min, t_max = min(t_start, t_end), max(t_start, t_end)
        f_min, f_max = min(f_start, f_end), max(f_start, f_end)
        
        self._bounds = (t_min, t_max, f_min, f_max)
        
        width = t_max - t_min
        height = f_max - f_min
        center_x = (t_min + t_max) / 2
        center_y = (f_min + f_max) / 2
        
        self._roi_rect.center = (center_x, center_y)
        self._roi_rect.width = width
        self._roi_rect.height = height
        self._roi_rect.visible = True
    
    def show(self):
        """Show the ROI visual."""
        self._roi_rect.visible = True
    
    def hide(self):
        """Hide the ROI visual."""
        self._roi_rect.visible = False
    
    def set_border_color(self, color: tuple):
        """Set border color."""
        self.border_color = color
        self._roi_rect.border_color = color
    
    def set_fill_color(self, color: tuple):
        """Set fill color."""
        self.fill_color = color
        self._roi_rect.color = color
    
    def destroy(self):
        """Remove visuals from scene."""
        if self._roi_rect and self._roi_rect.parent:
            self._roi_rect.parent = None
    
    @property
    def bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Get current bounds as (t_min, t_max, f_min, f_max)."""
        return self._bounds


class InteractionManager(QObject):
    """
    Main class for handling user interaction with the spectrogram.
    
    Provides:
    - Coordinate mapping (screen -> world -> physics)
    - ROI selection with visual feedback
    - Signals for ROI completion
    
    Usage:
        interaction_mgr = InteractionManager(canvas)
        interaction_mgr.roi_selected.connect(self.on_roi_selected)
        interaction_mgr.set_roi_mode(True)
    """
    
    # Signals
    roi_selected = Signal(float, float, float, float)  # t_start, t_end, f_start, f_end
    coordinate_changed = Signal(float, float)  # time, freq (for status bar)
    
    def __init__(self, canvas=None, sample_rate: int = 44100, 
                 hop_length: int = 512, fft_size: int = 4096):
        """
        Initialize interaction manager.
        
        Args:
            canvas: VisPyCanvas instance
            sample_rate: Audio sample rate
            hop_length: Spectrogram hop length
            fft_size: FFT size
        """
        super().__init__()
        
        self._canvas = canvas
        self._view = canvas.view if canvas else None
        
        # Coordinate mapper
        self.coord_mapper = CoordinateMapper(sample_rate, hop_length, fft_size)
        if canvas:
            self.coord_mapper.set_view(canvas.view, canvas)
        
        # ROI state
        self._roi_mode = False
        self._roi_drawing = False
        self._roi_start = None  # (time, freq)
        self._roi_visual = None
        
        # Active ROI visual for drawing - Raven Pro style (red border + semi-transparent red fill)
        if HAS_VISPY and canvas:
            self._roi_visual = ROIVisual(
                canvas.view.scene,
                border_color=ROIVisual.ACTIVE_BORDER_COLOR,
                fill_color=ROIVisual.ACTIVE_FILL_COLOR  # Red tint while drawing
            )
    
    def set_canvas(self, canvas):
        """Set or update the canvas reference."""
        self._canvas = canvas
        self._view = canvas.view if canvas else None
        if canvas:
            self.coord_mapper.set_view(canvas.view, canvas)
            if HAS_VISPY and not self._roi_visual:
                self._roi_visual = ROIVisual(
                    canvas.view.scene,
                    border_color=ROIVisual.ACTIVE_BORDER_COLOR
                )
    
    def update_parameters(self, sample_rate: int = None, 
                         hop_length: int = None, fft_size: int = None):
        """Update audio parameters."""
        self.coord_mapper.update_parameters(sample_rate, hop_length, fft_size)
    
    def set_roi_mode(self, enabled: bool):
        """
        Enable or disable ROI selection mode.
        
        Args:
            enabled: True to enable, False to disable
        """
        self._roi_mode = enabled
        if not enabled:
            self.cancel_roi_drawing()
        logger.info(f"ROI mode: {'ON' if enabled else 'OFF'}")
    
    @property
    def roi_mode(self) -> bool:
        """Check if ROI mode is enabled."""
        return self._roi_mode
    
    def screen_to_world(self, screen_pos: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Convert screen coordinates to world (time, freq) coordinates."""
        return self.coord_mapper.screen_to_world(screen_pos)
    
    def screen_to_audio(self, screen_pos: Tuple[float, float]) -> Optional[AudioCoordinates]:
        """Convert screen coordinates to full audio coordinates."""
        world = self.coord_mapper.screen_to_world(screen_pos)
        if world:
            return self.coord_mapper.world_to_audio(world[0], world[1])
        return None
    
    def on_mouse_press(self, screen_pos: Tuple[float, float]) -> bool:
        """
        Handle mouse press event.
        
        Args:
            screen_pos: (x, y) screen position
        
        Returns:
            True if event was handled (ROI drawing started)
        """
        if not self._roi_mode:
            return False
        
        world = self.screen_to_world(screen_pos)
        if world is None:
            return False
        
        # Start ROI drawing
        self._roi_drawing = True
        self._roi_start = world
        
        if self._roi_visual:
            self._roi_visual.update(world[0], world[0], world[1], world[1])
            self._roi_visual.show()
        
        logger.debug(f"ROI drawing started at t={world[0]:.3f}s, f={world[1]:.0f}Hz")
        return True
    
    def on_mouse_move(self, screen_pos: Tuple[float, float]) -> bool:
        """
        Handle mouse move event.
        
        Args:
            screen_pos: (x, y) screen position
        
        Returns:
            True if event was handled (ROI being drawn)
        """
        world = self.screen_to_world(screen_pos)
        if world:
            # Always emit coordinate for status bar
            self.coordinate_changed.emit(world[0], world[1])
        
        if not self._roi_drawing or not self._roi_start:
            return False
        
        if world is None:
            return False
        
        # Update ROI visual
        if self._roi_visual:
            self._roi_visual.update(
                self._roi_start[0], world[0],
                self._roi_start[1], world[1]
            )
            # Force canvas update
            if self._canvas:
                self._canvas.update()
        
        return True
    
    def on_mouse_release(self, screen_pos: Tuple[float, float]) -> Optional[ROISelection]:
        """
        Handle mouse release event.
        
        Args:
            screen_pos: (x, y) screen position
        
        Returns:
            ROISelection if a valid ROI was created, None otherwise
        """
        if not self._roi_drawing or not self._roi_start:
            return None
        
        world = self.screen_to_world(screen_pos)
        if world is None:
            self.cancel_roi_drawing()
            return None
        
        t_start, f_start = self._roi_start
        t_end, f_end = world
        
        # Validate ROI has minimum size
        min_time = 0.001  # 1ms minimum
        min_freq = 1.0    # 1Hz minimum
        
        if abs(t_end - t_start) < min_time or abs(f_end - f_start) < min_freq:
            logger.debug("ROI too small, cancelled")
            self.cancel_roi_drawing()
            return None
        
        # Create ROI selection
        roi = ROISelection(
            t_start=min(t_start, t_end),
            t_end=max(t_start, t_end),
            f_start=min(f_start, f_end),
            f_end=max(f_start, f_end)
        )
        
        # Emit signal
        self.roi_selected.emit(roi.t_start, roi.t_end, roi.f_start, roi.f_end)
        
        logger.info(f"ROI selected: time=[{roi.t_start:.3f}, {roi.t_end:.3f}]s, "
                   f"freq=[{roi.f_start:.0f}, {roi.f_end:.0f}]Hz")
        
        # Reset state (but keep visual visible as "completed" ROI)
        self._roi_drawing = False
        self._roi_start = None
        
        # Keep the same Raven Pro style colors for completed ROI
        if self._roi_visual:
            self._roi_visual.set_border_color(ROIVisual.DEFAULT_BORDER_COLOR)
            self._roi_visual.set_fill_color(ROIVisual.DEFAULT_FILL_COLOR)
        
        return roi
    
    def cancel_roi_drawing(self):
        """Cancel current ROI drawing operation."""
        self._roi_drawing = False
        self._roi_start = None
        if self._roi_visual:
            self._roi_visual.hide()
        if self._canvas:
            self._canvas.update()
    
    def clear_roi_visual(self):
        """Clear any displayed ROI visual."""
        if self._roi_visual:
            self._roi_visual.hide()
        if self._canvas:
            self._canvas.update()
    
    def destroy(self):
        """Clean up resources."""
        if self._roi_visual:
            self._roi_visual.destroy()
            self._roi_visual = None

