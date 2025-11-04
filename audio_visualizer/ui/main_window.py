import sys
import os
import logging
from typing import Optional, Dict, Any
import numpy as np

from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                              QWidget, QTabWidget, QMenuBar, QStatusBar, QToolBar,
                              QSlider, QLabel, QComboBox, QPushButton, QProgressBar,
                              QFileDialog, QMessageBox, QSplitter, QFrame)
from PySide6.QtCore import Qt, QTimer, Signal, QThread
from PySide6.QtGui import QAction, QKeySequence

try:
    import vispy
    from vispy import scene, app
    from vispy.scene import visuals
    HAS_VISPY = True
except ImportError:
    vispy = None
    HAS_VISPY = False

from ..core.data_loader import ChunkedAudioLoader
from ..core.cache_manager import CacheManager
from ..core.task_manager import TaskManager
from ..core.tile_cache import TileCache
from ..core.tile_manager import TileManager as TileMgr
from ..core.gpu_memory_manager import get_gpu_memory_manager
from ..engines.spectrogram_engine import SpectrogramEngine
from ..engines.cepstrogram_engine import CepstrogramEngine
from ..engines.fk_engine import FKEngine
from ..rendering.render_manager import RenderManager
from ..rendering.texture_atlas import TextureAtlas

logger = logging.getLogger(__name__)


class VisPyCanvas(scene.SceneCanvas):
    """Custom VisPy canvas for audio visualization with proper axes."""
    
    def __init__(self, view_type: str, parent=None):
        if not HAS_VISPY:
            raise RuntimeError("VisPy not available")
        
        super().__init__(keys='interactive', parent=parent, size=(800, 600))
        
        self.unfreeze()
        self.view_type = view_type
        
        # Minimal grid layout to get axes at window borders - NO MARGINS!
        self.grid = self.central_widget.add_grid(margin=0)
        self.grid.spacing = 0
        
        # Create minimal axis widgets with proper tick configuration
        self.y_axis = scene.AxisWidget(orientation='left', axis_label='Frequency (Hz)', 
                                     axis_font_size=6, axis_label_margin=30,
                                     tick_label_margin=5)
        self.y_axis.width_max = 60  # Wider to accommodate labels properly
        self.y_axis.width_min = 55
        
        self.x_axis = scene.AxisWidget(orientation='bottom', axis_label='Time (s)',
                                     axis_font_size=6, axis_label_margin=20,
                                     tick_label_margin=5) 
        self.x_axis.height_max = 40  # Taller for proper label spacing
        self.x_axis.height_min = 35
        
        # Create ViewBox for the main plot area - NO PADDING!
        self.view = scene.ViewBox(camera='panzoom', parent=None)
        
        # Simple 2x2 grid layout like the working example:
        # [ y_axis ] [ plot_area ]
        # [ empty  ] [ x_axis   ]
        self.grid.add_widget(self.y_axis, row=0, col=0)
        self.grid.add_widget(self.view, row=0, col=1)  
        self.grid.add_widget(self.x_axis, row=1, col=1)
        
        # Link axes to the ViewBox so they update together
        self.y_axis.link_view(self.view)
        self.x_axis.link_view(self.view)
        
        # Configure axis ticks for proper label spacing and alignment
        # Reduce number of ticks to prevent overlap
        self.y_axis.axis.tick_font_size = 6
        self.x_axis.axis.tick_font_size = 6
        
        # Set reasonable tick density (fewer ticks = less overlap)
        try:
            # These help control tick density in VisPy
            self.y_axis.axis.tick_label_format = '%.0f'  # No decimals for frequency
            self.x_axis.axis.tick_label_format = '%.1f'  # One decimal for time
        except AttributeError:
            pass  # Some VisPy versions may not support this
        
        # Configure camera for custom zoom control
        self.view.camera.aspect = None  # Allows independent zoom on X and Y axes
        self.view.camera.rect = (-1, -1, 2, 2)  # Initial range: x=(-1,1), y=(-1,1)
        
        # Disable interactive controls to prevent conflicts with our custom zoom
        self.view.camera.interactive = False
        
        # Set proper axis orientation for spectrograms
        # (False, False, False) = no flipping, frequency increases upward naturally
        self.view.camera.flip = (False, False, False)
        
        # Set zoom limits to prevent zooming too far out or in
        self.view.camera.zoom_factor = 2.0  # Ultra sensitive zoom for instant navigation
        
        # Store data bounds for camera constraints
        self.data_bounds = None  # Will be set when data is loaded
        
        # Image visual for spectrogram data in the ViewBox
        self.image_visual = scene.visuals.Image(parent=self.view.scene, interpolation='nearest')
        
        # Initialize mouse tracking and crosshair variables
        self.mouse_pos = (0, 0)
        self.crosshair_enabled = False
        
        # Create crosshair visuals (but keep them invisible initially)
        self.crosshair_v = scene.visuals.Line(color=(0.5, 0.3, 0.9, 0.8), width=1.5, parent=self.view.scene)
        self.crosshair_h = scene.visuals.Line(color=(0.5, 0.3, 0.9, 0.8), width=1.5, parent=self.view.scene)
        self.crosshair_v.visible = False
        self.crosshair_h.visible = False
        
        # Create text visual for readouts
        self.text_visual = scene.visuals.Text('', color='white', font_size=11, 
                                            pos=(10, 30), parent=self.view.scene)
        
        # OpenGL texture size limit
        self.max_texture_size = 16384
        
        # Mouse navigation state
        self.is_panning = False
        self.last_mouse_pos = None
        self.pan_speed = 1.0  # Pan speed factor (matching PyQtGraph)
        
        # Connect events for independent axis zoom and mouse tracking
        self.view.events.mouse_wheel.connect(self.on_mouse_wheel)
        self.events.key_press.connect(self.on_key_press)
        self.events.mouse_move.connect(self.on_mouse_move)
        self.events.mouse_press.connect(self.on_mouse_press)
        self.events.mouse_release.connect(self.on_mouse_release)
        
        # Timer for camera constraints (disabled to prevent interference with zoom)
        # self.update_timer = app.Timer(interval=0.1, connect=self.on_timer, start=True)
        
        self.freeze()
    
    def on_timer(self, event):
        """Timer callback to update camera constraints - DISABLED."""
        # Disabled to prevent interference with zoom
        # self.constrain_camera_to_bounds()
        # self.update_axis_labels()
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for axis-specific zoom - ALWAYS handle to prevent crashes.
        
        - Shift + Wheel: Zoom time axis (X) only  
        - Ctrl + Wheel: Zoom frequency axis (Y) only
        - Wheel alone: Zoom both axes
        """
        # ALWAYS handle the event to prevent VisPy's problematic default zoom
        event.handled = True
        
        # Get modifier state with more robust detection
        modifiers = []
        try:
            # Try multiple ways to get modifiers
            if hasattr(event, 'modifiers') and event.modifiers:
                modifiers = event.modifiers
            elif hasattr(event, 'mouse_event') and hasattr(event.mouse_event, 'modifiers'):
                modifiers = event.mouse_event.modifiers or []
        except AttributeError:
            modifiers = []
        
        # Convert modifiers to strings for easier detection
        mod_strings = [str(mod).lower() for mod in modifiers]
        shift_pressed = any('shift' in s for s in mod_strings)
        ctrl_pressed = any('ctrl' in s or 'control' in s for s in mod_strings)
        
        logger.info(f"Mouse wheel: delta={event.delta}, mods={mod_strings}, shift={shift_pressed}, ctrl={ctrl_pressed}")
        
        # Ultra sensitive zoom factor for instant navigation
        zoom_base = 2.5  # Extremely high sensitivity for fastest zoom response
        factor = zoom_base ** (event.delta[1] / 120.0)
        
        # Determine zoom behavior
        if shift_pressed:
            # X-axis only (time) - more sensitive
            scale_factors = [factor, 1.0]
            logger.info(f"TIME axis zoom: factor={factor:.3f}")
        elif ctrl_pressed:
            # Y-axis only (frequency)
            scale_factors = [1.0, factor]
            logger.info(f"FREQUENCY axis zoom: factor={factor:.3f}")
        else:
            # Both axes
            scale_factors = [factor, factor]
            logger.debug(f"Both axes zoom: factor={factor:.3f}")
        
        # Perform the zoom
        self.zoom_with_center(scale_factors, event.pos)
        
    def zoom_with_center(self, scale_factors, mouse_pos):
        """Zoom with center-based scaling around mouse position."""
        try:
            # Get current camera rect
            rect = self.view.camera.rect
            if rect is None:
                logger.warning("Camera rect is None, skipping zoom")
                return
                
            # Current view dimensions
            current_width = float(rect.width)
            current_height = float(rect.height)
            current_x = float(rect.left)
            current_y = float(rect.bottom)
            
            # Validate current state
            if current_width <= 0 or current_height <= 0:
                logger.warning("Invalid camera dimensions, skipping zoom")
                return
            
            # Calculate new dimensions
            new_width = current_width / scale_factors[0]
            new_height = current_height / scale_factors[1]
            
            # Apply boundary constraints early to prevent invalid states
            if self.data_bounds:
                time_min, time_max, freq_min, freq_max = self.data_bounds
                
                # Constrain width (time axis)
                max_width = time_max - time_min
                min_width = max_width / 100  # Allow 100x zoom in
                new_width = max(min_width, min(new_width, max_width * 1.1))
                
                # Constrain height (frequency axis)  
                max_height = freq_max - freq_min
                min_height = max_height / 100  # Allow 100x zoom in
                new_height = max(min_height, min(new_height, max_height * 1.1))
            
            # Use view center for zoom (simpler and more reliable)
            center_x = current_x + current_width / 2
            center_y = current_y + current_height / 2
            
            # Calculate new position (zoom around center)
            new_x = center_x - new_width / 2
            new_y = center_y - new_height / 2
            
            # Apply boundary constraints for position
            if self.data_bounds:
                time_min, time_max, freq_min, freq_max = self.data_bounds
                
                # Add small margins
                margin_x = (time_max - time_min) * 0.02
                margin_y = (freq_max - freq_min) * 0.02
                
                # Constrain X position
                min_x = time_min - margin_x
                max_x = time_max + margin_x - new_width
                new_x = max(min_x, min(new_x, max_x))
                    
                # Constrain Y position
                min_y = freq_min - margin_y  
                max_y = freq_max + margin_y - new_height
                new_y = max(min_y, min(new_y, max_y))
            
            # Validate final values before applying
            if new_width > 0 and new_height > 0:
                self.view.camera.rect = (new_x, new_y, new_width, new_height)
                logger.debug(f"Zoom applied: rect=({new_x:.2f}, {new_y:.2f}, {new_width:.2f}, {new_height:.2f})")
            else:
                logger.warning("Invalid zoom dimensions calculated, skipping")
            
        except Exception as e:
            logger.error(f"Error in zoom_with_center: {e}")
            # Silently ignore zoom errors to prevent crashes
    
    def on_key_press(self, event):
        """Handle key press events for additional zoom controls."""
        logger.info(f"Key pressed: {event.key}")
        
        if event.key == 'X':
            # Zoom out on time axis
            logger.info("Key X: Zooming out TIME axis")
            self.zoom_with_center([0.8, 1.0], None)  # X-axis zoom out
        elif event.key == 'x':
            # Zoom in on time axis  
            logger.info("Key x: Zooming in TIME axis")
            self.zoom_with_center([1.2, 1.0], None)  # X-axis zoom in
        elif event.key == 'Y':
            # Zoom out on frequency axis
            logger.info("Key Y: Zooming out FREQUENCY axis")
            self.zoom_with_center([1.0, 0.8], None)  # Y-axis zoom out
        elif event.key == 'y':
            # Zoom in on frequency axis
            logger.info("Key y: Zooming in FREQUENCY axis")
            self.zoom_with_center([1.0, 1.2], None)  # Y-axis zoom in
        elif event.key == 'R' or event.key == 'r':
            # Reset zoom to show all data
            logger.info("Key R: Resetting zoom")
            self.reset_camera_to_data_bounds()
    
    def constrain_camera_to_bounds(self):
        """Constrain camera to data bounds - prevent invalid states."""
        if self.data_bounds is None:
            return
        
        try:
            rect = self.view.camera.rect
            if rect is None:
                return
            
            # Get current rect properties safely
            current_x = float(rect.left)
            current_y = float(rect.bottom)
            current_width = float(rect.width)
            current_height = float(rect.height)
            
            # Validate current dimensions
            if current_width <= 0 or current_height <= 0:
                logger.warning("Invalid camera dimensions - resetting to data bounds")
                self.reset_camera_to_data_bounds()
                return
                
            # Extract bounds
            time_min, time_max, freq_min, freq_max = self.data_bounds
            data_width = time_max - time_min
            data_height = freq_max - freq_min
            
            # Prevent extreme zoom levels
            min_width = data_width / 1000  # Max 1000x zoom
            max_width = data_width * 2     # Allow slight overzoom
            min_height = data_height / 1000
            max_height = data_height * 2
            
            # Constrain dimensions
            new_width = max(min_width, min(current_width, max_width))
            new_height = max(min_height, min(current_height, max_height))
            
            # Constrain position with small margin
            margin_x = data_width * 0.05
            margin_y = data_height * 0.05
            
            new_x = max(time_min - margin_x, 
                       min(current_x, time_max + margin_x - new_width))
            new_y = max(freq_min - margin_y,
                       min(current_y, freq_max + margin_y - new_height))
            
            # Only update if values changed significantly
            if (abs(new_x - current_x) > 1e-6 or abs(new_y - current_y) > 1e-6 or
                abs(new_width - current_width) > 1e-6 or abs(new_height - current_height) > 1e-6):
                
                # Validate the new rect before applying
                if new_width > 0 and new_height > 0:
                    self.view.camera.rect = (new_x, new_y, new_width, new_height)
                    
        except Exception as e:
            logger.debug(f"Error in camera constraints: {e}")
            # Don't crash - camera constraints are not critical
            
    def reset_camera_to_data_bounds(self):
        """Reset camera to show all data."""
        if self.data_bounds:
            time_min, time_max, freq_min, freq_max = self.data_bounds
            self.view.camera.rect = (time_min, freq_min, 
                                   time_max - time_min, freq_max - freq_min)
    
    def update_axis_labels(self):
        """Update axis labels with appropriate units based on zoom level.
        
        With AxisWidget, the labels are automatically managed, but we can
        update the axis label text to show appropriate units.
        """
        if self.data_bounds is None:
            return
        
        try:
            rect = self.view.camera.rect
            if rect is None:
                return
            
            x_min = rect.left
            x_max = rect.right  
            y_min = rect.bottom
            y_max = rect.top
        except Exception as e:
            logger.debug(f"Cannot get camera rect: {e}")
            return
        
        time_min, freq_min, time_max, freq_max = self.data_bounds
        w = x_max - x_min
        h = y_max - y_min
        
        # Format time axis label (X-axis)
        time_span = w
        if time_span < 1.0:
            time_unit = "Time (ms)"
        elif time_span < 60.0:
            time_unit = "Time (s)"
        elif time_span < 3600.0:
            time_unit = "Time (min)"
        else:
            time_unit = "Time (hr)"
        
        # Format frequency axis label (Y-axis)
        freq_span = h
        if freq_span < 1000.0:
            freq_unit = "Frequency (Hz)"
        else:
            freq_unit = "Frequency (kHz)"
        
        # Update AxisWidget labels (need to unfreeze first)
        self.x_axis.unfreeze()
        self.x_axis.axis_label = time_unit
        self.x_axis.freeze()
        
        self.y_axis.unfreeze()
        self.y_axis.axis_label = freq_unit
        self.y_axis.freeze()
        
        logger.debug(f"Axis labels updated: {time_unit}, {freq_unit}")
    
    def set_data_bounds(self, time_min, time_max, freq_min, freq_max):
        """Set data bounds for zoom and pan constraints."""
        self.data_bounds = (time_min, time_max, freq_min, freq_max)
        logger.info(f"Data bounds set: time=[{time_min:.3f}, {time_max:.3f}], freq=[{freq_min:.1f}, {freq_max:.1f}]")
    
    def update_image(self, data: np.ndarray, extent: tuple = None):
        """Update the displayed image data."""
        if data is not None and data.size > 0:
            # Ensure data is in correct format for VisPy
            # VisPy Image expects: rows=Y-axis, columns=X-axis
            # For spectrogram: we want time=X-axis, frequency=Y-axis
            # Data comes as (frequency_bins, time_frames) which is already correct!
            if data.ndim == 2:
                # Data format (freq_bins, time_frames) is correct for VisPy
                # freq_bins = rows = Y-axis (frequency)
                # time_frames = columns = X-axis (time)
                display_data = data.astype(np.float32)
                
                # Check OpenGL texture size limits
                original_shape = display_data.shape
                downsample_factor_x = 1
                downsample_factor_y = 1
                
                # Check if we exceed texture limits
                if display_data.shape[1] > self.max_texture_size:
                    downsample_factor_x = int(np.ceil(display_data.shape[1] / self.max_texture_size))
                    logger.warning(f"Data width ({display_data.shape[1]}) exceeds OpenGL limit ({self.max_texture_size})")
                    logger.warning(f"Downsampling by {downsample_factor_x}x in time to fit texture")
                    display_data = display_data[:, ::downsample_factor_x]
                
                if display_data.shape[0] > self.max_texture_size:
                    downsample_factor_y = int(np.ceil(display_data.shape[0] / self.max_texture_size))
                    logger.warning(f"Data height ({display_data.shape[0]}) exceeds OpenGL limit ({self.max_texture_size})")
                    logger.warning(f"Downsampling by {downsample_factor_y}x in frequency to fit texture")
                    display_data = display_data[::downsample_factor_y, :]
                
                if downsample_factor_x > 1 or downsample_factor_y > 1:
                    logger.info(f"Downsampled: {original_shape} -> {display_data.shape}")
                
                # Normalize data for proper VisPy display
                # Spectrogram data is typically in dB (e.g., -120 to 0 dB)
                # Normalize to 0-1 range for proper colormap visualization
                data_min = np.min(display_data)
                data_max = np.max(display_data)
                
                if data_max > data_min:  # Avoid division by zero
                    display_data = (display_data - data_min) / (data_max - data_min)
                else:
                    display_data = np.zeros_like(display_data)
                
                logger.debug(f"Data normalized: {data_min:.1f} to {data_max:.1f}")
                logger.debug(f"Display shape: {display_data.shape}")
            else:
                display_data = data.astype(np.float32)
            
            if extent:
                # extent = (time_start, time_end, freq_start, freq_end)
                time_start, time_end, freq_start, freq_end = extent
                time_width = time_end - time_start
                freq_height = freq_end - freq_start
                
                logger.debug(f"Transform: time [{time_start}, {time_end}], freq [{freq_start}, {freq_end}]")
                logger.debug(f"Data pixels: {display_data.shape[1]} x {display_data.shape[0]}")
                
                # Set data with proper clim for color mapping
                self.image_visual.set_data(display_data)
                self.image_visual.clim = (0, 1)  # Data is normalized to 0-1
                
                # Calculate transform to map pixel coordinates to world coordinates
                # VisPy Image: pixel (j, i) where j=column (X), i=row (Y)
                # display_data[i, j] where i=row (Y/freq), j=column (X/time)
                
                # We want to map:
                # - pixel column 0 -> time_start
                # - pixel column (shape[1]-1) -> time_end
                # - pixel row 0 -> freq_start  
                # - pixel row (shape[0]-1) -> freq_end
                
                # Scale factor: world_units per pixel
                x_scale = time_width / display_data.shape[1]
                y_scale = freq_height / display_data.shape[0]
                
                # Translate: position of pixel (0, 0) in world coords
                # Pixel (0,0) is bottom-left, should map to (time_start, freq_start)
                x_translate = time_start
                y_translate = freq_start
                
                transform = scene.STTransform(
                    scale=(x_scale, y_scale),
                    translate=(x_translate, y_translate)
                )
                self.image_visual.transform = transform
                
                logger.debug(f"Transform scale: ({x_scale:.6f}, {y_scale:.2f})")
                
                # Set data bounds for proper zoom/pan constraints
                self.set_data_bounds(time_start, time_end, freq_start, freq_end)
                
                # Update camera to show the full data range
                self.view.camera.rect = (
                    time_start, freq_start, 
                    time_end - time_start, freq_end - freq_start
                )
                
                # Update axis labels
                self.update_axis_labels()
                
                logger.debug(f"Camera range: X=[{time_start}, {time_end}], Y=[{freq_start}, {freq_end}]")
            else:
                self.image_visual.set_data(display_data)
                self.image_visual.clim = (0, 1)
    
    def set_crosshair(self, enabled: bool, pos: tuple = None):
        """Enable/disable crosshair display."""
        self.crosshair_enabled = enabled
        
        if enabled and pos:
            x, y = pos
            
            # Update crosshair lines
            rect = self.view.camera.rect
            x_range = (rect.left, rect.right)
            y_range = (rect.bottom, rect.top)
            
            # Vertical line
            self.crosshair_v.set_data(np.array([[x, y_range[0]], [x, y_range[1]]]))
            
            # Horizontal line  
            self.crosshair_h.set_data(np.array([[x_range[0], y], [x_range[1], y]]))
            
            self.crosshair_v.visible = True
            self.crosshair_h.visible = True
        else:
            self.crosshair_v.visible = False
            self.crosshair_h.visible = False
    
    def update_text_readout(self, text: str, pos: tuple = None):
        """Update text readout display."""
        self.text_visual.text = text
        if pos:
            self.text_visual.pos = pos
    
    def on_mouse_press(self, event):
        """Handle mouse press for starting pan navigation."""
        if event.button == 1:  # Left button
            self.is_panning = True
            self.last_mouse_pos = event.pos
            event.handled = True
    
    def on_mouse_release(self, event):
        """Handle mouse release for ending pan navigation."""
        if event.button == 1:  # Left button
            self.is_panning = False
            self.last_mouse_pos = None
            event.handled = True
    
    def on_mouse_move(self, event):
        """Handle mouse movement for pan navigation and crosshair updates."""
        if event.pos is not None:
            # Handle panning if left mouse button is pressed
            if self.is_panning and self.last_mouse_pos is not None:
                try:
                    # Calculate delta in screen space
                    delta_screen = event.pos - self.last_mouse_pos
                    
                    # Get current camera rect
                    rect = self.view.camera.rect
                    if rect is None:
                        return
                    
                    # Get canvas size to convert screen delta to world delta
                    canvas_size = self.size
                    if canvas_size[0] <= 0 or canvas_size[1] <= 0:
                        return
                    
                    # Convert screen delta to world delta (with pan speed)
                    delta_x = -(delta_screen[0] / canvas_size[0]) * rect.width * self.pan_speed
                    delta_y = (delta_screen[1] / canvas_size[1]) * rect.height * self.pan_speed
                    
                    # Calculate new position
                    new_left = rect.left + delta_x
                    new_bottom = rect.bottom + delta_y
                    
                    # Apply boundary constraints
                    if self.data_bounds:
                        time_min, time_max, freq_min, freq_max = self.data_bounds
                        
                        # Constrain X (time)
                        if new_left < time_min:
                            new_left = time_min
                        if new_left + rect.width > time_max:
                            new_left = time_max - rect.width
                        
                        # Constrain Y (frequency)
                        if new_bottom < freq_min:
                            new_bottom = freq_min
                        if new_bottom + rect.height > freq_max:
                            new_bottom = freq_max - rect.height
                    
                    # Apply the pan
                    self.view.camera.rect = (new_left, new_bottom, rect.width, rect.height)
                    
                    # Update last position
                    self.last_mouse_pos = event.pos
                    
                    logger.debug(f"Panning: new rect=({new_left:.1f}, {new_bottom:.1f}, {rect.width:.1f}, {rect.height:.1f})")
                    
                except Exception as e:
                    logger.debug(f"Pan error: {e}")
            
            # Handle crosshair updates (only when not panning)
            elif not self.is_panning:
                try:
                    # Convert screen coordinates to world coordinates using ViewBox transform
                    tr = self.view.scene.node_transform(self.view.scene)
                    if tr is not None:
                        world_pos = tr.map(event.pos)
                        self.mouse_pos = (world_pos[0], world_pos[1])
                        
                        if self.crosshair_enabled:
                            self.set_crosshair(True, self.mouse_pos)
                            
                            # Update readout text
                            readout = f"Time: {world_pos[0]:.3f}s, Freq: {world_pos[1]:.0f}Hz"
                            self.update_text_readout(readout, (10, 30))
                except Exception as e:
                    # Silently handle coordinate transform errors during mouse movement
                    pass

class StatusWidget(QWidget):
    """Status widget showing performance metrics."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        
        self.fps_label = QLabel("FPS: --")
        self.memory_label = QLabel("Memory: --")
        self.gpu_label = QLabel("GPU: --")
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        layout.addWidget(QLabel("Performance:"))
        layout.addWidget(self.fps_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.memory_label)
        layout.addWidget(QLabel("|"))
        layout.addWidget(self.gpu_label)
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

class ControlsWidget(QWidget):
    """Widget containing analysis controls."""
    
    # Signals
    parameters_changed = Signal(dict)
    colormap_changed = Signal(str)
    db_range_changed = Signal(float, float)
    refresh_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setup_ui()
        self.connect_signals()
    
    def setup_ui(self):
        """Setup the control interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # FFT Parameters
        fft_frame = QFrame()
        fft_frame.setFrameStyle(QFrame.StyledPanel)
        fft_layout = QHBoxLayout(fft_frame)
        
        fft_layout.addWidget(QLabel("FFT Size:"))
        self.fft_size_combo = QComboBox()
        self.fft_size_combo.addItems(['512', '1024', '2048', '4096', '8192'])
        self.fft_size_combo.setCurrentText('2048')
        fft_layout.addWidget(self.fft_size_combo)
        
        fft_layout.addWidget(QLabel("Hop:"))
        self.hop_combo = QComboBox()
        self.hop_combo.addItems(['128', '256', '512', '1024'])
        self.hop_combo.setCurrentText('512')
        fft_layout.addWidget(self.hop_combo)
        
        # Colormap
        colormap_frame = QFrame()
        colormap_frame.setFrameStyle(QFrame.StyledPanel)
        colormap_layout = QHBoxLayout(colormap_frame)
        
        colormap_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'plasma', 'jet', 'magma'])
        colormap_layout.addWidget(self.colormap_combo)
        
        # dB Range
        db_frame = QFrame()
        db_frame.setFrameStyle(QFrame.StyledPanel)
        db_layout = QHBoxLayout(db_frame)
        
        db_layout.addWidget(QLabel("dB Range:"))
        
        self.db_min_slider = QSlider(Qt.Horizontal)
        self.db_min_slider.setRange(-120, 0)
        self.db_min_slider.setValue(-80)
        self.db_min_label = QLabel("-80")
        
        self.db_max_slider = QSlider(Qt.Horizontal)
        self.db_max_slider.setRange(-80, 20)
        self.db_max_slider.setValue(0)
        self.db_max_label = QLabel("0")
        
        db_layout.addWidget(QLabel("Min:"))
        db_layout.addWidget(self.db_min_slider)
        db_layout.addWidget(self.db_min_label)
        db_layout.addWidget(QLabel("Max:"))
        db_layout.addWidget(self.db_max_slider)
        db_layout.addWidget(self.db_max_label)
        
        # Refresh button
        self.refresh_button = QPushButton("Refresh")
        
        # Add all frames to main layout
        layout.addWidget(fft_frame)
        layout.addWidget(colormap_frame)
        layout.addWidget(db_frame)
        layout.addStretch()
        layout.addWidget(self.refresh_button)
    
    def connect_signals(self):
        """Connect widget signals."""
        self.fft_size_combo.currentTextChanged.connect(self.emit_parameters_changed)
        self.hop_combo.currentTextChanged.connect(self.emit_parameters_changed)
        self.colormap_combo.currentTextChanged.connect(self.colormap_changed.emit)
        
        self.db_min_slider.valueChanged.connect(self.update_db_range)
        self.db_max_slider.valueChanged.connect(self.update_db_range)
        
        self.refresh_button.clicked.connect(self.refresh_requested.emit)
    
    def emit_parameters_changed(self):
        """Emit parameters changed signal."""
        params = {
            'fft_size': int(self.fft_size_combo.currentText()),
            'hop_length': int(self.hop_combo.currentText())
        }
        self.parameters_changed.emit(params)
    
    def update_db_range(self):
        """Update dB range display and emit signal."""
        db_min = self.db_min_slider.value()
        db_max = self.db_max_slider.value()
        
        # Ensure min < max
        if db_min >= db_max:
            if self.sender() == self.db_min_slider:
                db_max = db_min + 10
                self.db_max_slider.setValue(db_max)
            else:
                db_min = db_max - 10
                self.db_min_slider.setValue(db_min)
        
        self.db_min_label.setText(str(db_min))
        self.db_max_label.setText(str(db_max))
        
        self.db_range_changed.emit(db_min, db_max)

class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("GPU-Accelerated Audio Visualizer")
        self.setMinimumSize(1200, 800)
        
        # Apply modern glassmorphic theme
        self.apply_modern_theme()
        
        # Core components
        self.cache_manager = CacheManager(max_memory_mb=2048, max_gpu_memory_mb=1024)
        self.task_manager = TaskManager()
        self.audio_loader = ChunkedAudioLoader()
        
        # GPU Memory Management
        self.gpu_memory_manager = get_gpu_memory_manager()
        
        # Tile-based caching and rendering
        self.tile_cache = TileCache(max_memory_tiles=100, max_disk_gb=10.0)
        
        # Computation engines
        self.spectrogram_engine = SpectrogramEngine(self.cache_manager, self.task_manager)
        self.cepstrogram_engine = CepstrogramEngine(self.cache_manager, self.task_manager, self.spectrogram_engine)
        self.fk_engine = FKEngine(self.cache_manager, self.task_manager, self.spectrogram_engine)
        
        # Tile manager (coordinates tiles, cache, and atlases)
        self.tile_manager = TileMgr(
            tile_cache=self.tile_cache,
            engines={
                'spectrogram': self.spectrogram_engine,
                'cepstrogram': self.cepstrogram_engine,
                'fk_transform': self.fk_engine
            }
        )
        
        # Set up default array geometry for F-K analysis (simulated linear array)
        # For single-channel audio, we simulate a simple 8-element linear array
        array_spacing = 0.1  # 10 cm spacing
        n_sensors = 8
        sensor_positions = np.array([[i * array_spacing, 0.0] for i in range(n_sensors)])
        self.fk_engine.set_array_geometry(sensor_positions)
        
        # Current state
        self.current_file = None
        # Default view range (will be updated when file is loaded)
        self.current_view_range = ((0.0, 10.0), (0.0, 22050.0))  # (time, freq)
        
        # Setup UI
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_status_bar()
        
        # Performance timer
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self.update_performance_stats)
        self.perf_timer.start(1000)  # Update every second
        
        # Auto-load file if specified
        initial_file = os.environ.get('AUDIO_VISUALIZER_INITIAL_FILE')
        if initial_file and os.path.exists(initial_file):
            QTimer.singleShot(1000, lambda: self.load_audio_file(initial_file))
        
    def setup_ui(self):
        """Setup the main user interface with improved layout and containers."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with NO padding to maximize canvas space
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Controls at the top
        self.controls_widget = ControlsWidget()
        main_layout.addWidget(self.controls_widget)
        
        # Container for visualization - glassmorphic card style
        viz_container = QFrame()
        viz_container.setObjectName("viz_container")
        viz_container.setFrameShape(QFrame.NoFrame)
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)  # NO MARGINS!
        viz_layout.setSpacing(0)
        
        # Tab widget for different views - NO PADDING
        self.tab_widget = QTabWidget()
        self.tab_widget.setContentsMargins(0, 0, 0, 0)
        
        # Create tabs with proper containers
        if HAS_VISPY:
            # Spectrogram tab
            spec_container = QFrame()
            spec_container.setFrameShape(QFrame.NoFrame)
            spec_layout = QVBoxLayout(spec_container)
            spec_layout.setContentsMargins(0, 0, 0, 0)  # NO MARGINS!
            spec_layout.setSpacing(0)
            
            self.spectrogram_canvas = VisPyCanvas('spectrogram')
            spec_layout.addWidget(self.spectrogram_canvas.native)
            
            # Cepstrogram tab
            cepstro_container = QFrame()
            cepstro_container.setFrameShape(QFrame.NoFrame)
            cepstro_layout = QVBoxLayout(cepstro_container)
            cepstro_layout.setContentsMargins(0, 0, 0, 0)  # NO MARGINS!
            cepstro_layout.setSpacing(0)
            
            self.cepstrogram_canvas = VisPyCanvas('cepstrogram')
            cepstro_layout.addWidget(self.cepstrogram_canvas.native)
            
            # F-K tab
            fk_container = QFrame()
            fk_container.setFrameShape(QFrame.NoFrame)
            fk_layout = QVBoxLayout(fk_container)
            fk_layout.setContentsMargins(0, 0, 0, 0)  # NO MARGINS!
            fk_layout.setSpacing(0)
            
            self.fk_canvas = VisPyCanvas('fk_transform')
            fk_layout.addWidget(self.fk_canvas.native)
            
            # Add tabs
            self.tab_widget.addTab(spec_container, "Spectrogram")
            self.tab_widget.addTab(cepstro_container, "Cepstrogram")
            self.tab_widget.addTab(fk_container, "F-K Transform")
        else:
            # Fallback widgets when VisPy not available
            self.tab_widget.addTab(QLabel("VisPy not available"), "Spectrogram")
            self.tab_widget.addTab(QLabel("VisPy not available"), "Cepstrogram")
            self.tab_widget.addTab(QLabel("VisPy not available"), "F-K Transform")
        
        viz_layout.addWidget(self.tab_widget)
        main_layout.addWidget(viz_container)
        
        # Connect signals
        self.controls_widget.parameters_changed.connect(self.on_parameters_changed)
        self.controls_widget.colormap_changed.connect(self.on_colormap_changed)
        self.controls_widget.db_range_changed.connect(self.on_db_range_changed)
        self.controls_widget.refresh_requested.connect(self.refresh_current_view)
        
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
    
    def setup_menu_bar(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Audio File...", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self.open_audio_file)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        # Export menu
        export_menu = file_menu.addMenu("Export")
        
        export_image_action = QAction("Export Current View as Image...", self)
        export_image_action.setShortcut("Ctrl+E")
        export_image_action.triggered.connect(self.export_current_view_as_image)
        export_menu.addAction(export_image_action)
        
        export_data_action = QAction("Export Data as NPY...", self)
        export_data_action.triggered.connect(self.export_data_as_npy)
        export_menu.addAction(export_data_action)
        
        export_csv_action = QAction("Export Data as CSV...", self)
        export_csv_action.triggered.connect(self.export_data_as_csv)
        export_menu.addAction(export_csv_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        zoom_fit_action = QAction("Zoom to Fit", self)
        zoom_fit_action.setShortcut("Ctrl+0")
        zoom_fit_action.triggered.connect(self.zoom_to_fit)
        view_menu.addAction(zoom_fit_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu("Analysis")
        
        refresh_action = QAction("Refresh Current View", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.refresh_current_view)
        analysis_menu.addAction(refresh_action)
    
    def setup_status_bar(self):
        """Setup the status bar."""
        self.status_widget = StatusWidget()
        self.statusBar().addPermanentWidget(self.status_widget)
        self.statusBar().showMessage("Ready")
    
    def load_audio_file(self, file_path: str):
        """Load an audio file for analysis."""
        try:
            self.statusBar().showMessage("Loading audio file...")
            
            sample_rate, duration = self.audio_loader.load_file(file_path)
            
            self.current_file = file_path
            # Show full audio duration instead of just 10 seconds
            self.current_view_range = ((0.0, duration), 
                                     (0.0, sample_rate / 2))
            
            # Update spectrogram engine parameters
            self.spectrogram_engine.set_parameters(sample_rate=sample_rate)
            
            self.statusBar().showMessage(
                f"Loaded: {os.path.basename(file_path)} "
                f"({duration:.1f}s, {sample_rate}Hz)"
            )
            
            # Refresh current view
            self.refresh_current_view()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load audio file:\n{str(e)}")
            self.statusBar().showMessage("Ready")
    
    def open_audio_file(self):
        """Open an audio file dialog for analysis."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File",
            "", "Audio Files (*.wav *.flac *.mp3 *.ogg);;All Files (*)"
        )
        
        if file_path:
            self.load_audio_file(file_path)
    
    def on_parameters_changed(self, params: dict):
        """Handle parameter changes."""
        self.spectrogram_engine.set_parameters(**params)
        
        # Clear cache for affected views
        self.cache_manager.clear_view_cache('spectrogram')
        self.cache_manager.clear_view_cache('cepstrogram')
        
        self.refresh_current_view()
    
    def on_colormap_changed(self, colormap: str):
        """Handle colormap changes."""
        logger.info(f"Colormap changed to: {colormap}")
        
        try:
            # Update all canvas image visuals with new colormap
            if HAS_VISPY:
                if hasattr(self, 'spectrogram_canvas') and self.spectrogram_canvas.image_visual:
                    self.spectrogram_canvas.image_visual.cmap = colormap
                    logger.debug(f"Updated spectrogram colormap to {colormap}")
                
                if hasattr(self, 'cepstrogram_canvas') and self.cepstrogram_canvas.image_visual:
                    self.cepstrogram_canvas.image_visual.cmap = colormap
                    logger.debug(f"Updated cepstrogram colormap to {colormap}")
                
                if hasattr(self, 'fk_canvas') and self.fk_canvas.image_visual:
                    self.fk_canvas.image_visual.cmap = colormap
                    logger.debug(f"Updated F-K transform colormap to {colormap}")
                
                # Trigger a visual update
                self.update_displays()
                
        except Exception as e:
            logger.error(f"Error updating colormap: {e}")
    
    def update_displays(self):
        """Force update of all displays."""
        try:
            if HAS_VISPY:
                if hasattr(self, 'spectrogram_canvas'):
                    self.spectrogram_canvas.update()
                if hasattr(self, 'cepstrogram_canvas'):
                    self.cepstrogram_canvas.update()
                if hasattr(self, 'fk_canvas'):
                    self.fk_canvas.update()
        except Exception as e:
            logger.debug(f"Error updating displays: {e}")
    
    def on_db_range_changed(self, db_min: float, db_max: float):
        """Handle dB range changes."""
        logger.info(f"dB range changed to: [{db_min:.1f}, {db_max:.1f}] dB")
        
        try:
            # Update all canvas image visuals with new dB range for color limits
            if HAS_VISPY:
                # Convert dB range to normalized range (spectrogram data is normalized 0-1)
                # This affects the color mapping range
                if hasattr(self, 'spectrogram_canvas') and self.spectrogram_canvas.image_visual:
                    # Store dB range for use during data normalization
                    self.spectrogram_canvas.db_range = (db_min, db_max)
                    logger.debug(f"Updated spectrogram dB range to [{db_min:.1f}, {db_max:.1f}]")
                
                if hasattr(self, 'cepstrogram_canvas') and self.cepstrogram_canvas.image_visual:
                    self.cepstrogram_canvas.db_range = (db_min, db_max)
                    logger.debug(f"Updated cepstrogram dB range to [{db_min:.1f}, {db_max:.1f}]")
                
                # Refresh current view to apply new dB range
                self.refresh_current_view()
                
        except Exception as e:
            logger.error(f"Error updating dB range: {e}")
    
    def on_tab_changed(self, index: int):
        """Handle tab changes - lazy loading trigger."""
        if not self.current_file:
            return
        
        tab_names = ['spectrogram', 'cepstrogram', 'fk_transform']
        if 0 <= index < len(tab_names):
            view_type = tab_names[index]
            self.load_view_data(view_type)
    
    def load_view_data(self, view_type: str):
        """Load data for specified view type using tile system."""
        if not self.current_file:
            return
        
        self.statusBar().showMessage(f"Loading {view_type} tiles...")
        
        # Get current view range
        time_range = self.current_view_range[0]
        freq_range = self.current_view_range[1]
        
        logger.info(f"Loading {view_type} for time range {time_range}, freq range {freq_range}")
        
        try:
            # Try tile-based rendering first
            if self.load_view_data_tiled(view_type, time_range, freq_range):
                return
            
            # Fallback to direct computation if tile system fails
            logger.warning(f"Tile system failed for {view_type}, falling back to direct computation")
            self.load_view_data_direct(view_type, time_range)
                
        except Exception as e:
            logger.error(f"Error loading {view_type}: {e}")
            self.statusBar().showMessage(f"Error: {str(e)}")
    
    def load_view_data_tiled(self, view_type: str, time_range: tuple, freq_range: tuple) -> bool:
        """Load data using tile system. Returns True if successful."""
        try:
            # Update tile manager with current visible region
            zoom_level = self.estimate_zoom_level(time_range, freq_range)
            self.tile_manager.update_visible_region(view_type, time_range, freq_range, zoom_level)
            
            # Get visible tiles for this region
            visible_tiles = self.tile_manager.get_visible_tiles(view_type, time_range, freq_range, zoom_level)
            
            if not visible_tiles:
                logger.warning(f"No visible tiles found for {view_type}")
                return False
            
            # Check if we have cached tiles for this region
            atlas = self.tile_manager.atlases.get(view_type)
            if not atlas:
                logger.warning(f"No atlas found for {view_type}")
                return False
            
            # For now, use a simplified approach: request the most important tiles
            # and display what we have while computing missing ones
            self.request_tiles_for_region(view_type, time_range, freq_range, zoom_level)
            
            # Try to display available atlas data
            if self.update_atlas_display_improved(view_type, time_range, freq_range):
                self.statusBar().showMessage(f"{view_type.title()} tiles loaded")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Tile-based loading failed for {view_type}: {e}")
            return False
    
    def load_view_data_direct(self, view_type: str, time_range: tuple):
        """Fallback to direct computation."""
        logger.info(f"Using direct computation for {view_type}")
        
        # Get audio data for current time range
        start_sample = int(time_range[0] * self.spectrogram_engine.sample_rate)
        end_sample = int(time_range[1] * self.spectrogram_engine.sample_rate)
        
        # Clamp to available samples
        total_samples = self.audio_loader.total_samples
        num_samples = min(end_sample - start_sample, total_samples - start_sample)
        
        audio_data = self.audio_loader.get_chunk(start_sample, num_samples)
        
        # Use original direct computation methods
        if view_type == 'spectrogram':
            self.load_spectrogram_data(audio_data)
        elif view_type == 'cepstrogram':
            self.load_cepstrogram_data(audio_data)
        elif view_type == 'fk_transform':
            self.load_fk_data(audio_data)
    
    def estimate_zoom_level(self, time_range: tuple, freq_range: tuple) -> float:
        """Estimate zoom level based on visible range."""
        if not self.current_file:
            return 1.0
        
        # Get total duration and sample rate
        total_duration = self.audio_loader.duration
        sample_rate = self.spectrogram_engine.sample_rate
        
        # Calculate what fraction of total range is visible
        time_span = time_range[1] - time_range[0]
        freq_span = freq_range[1] - freq_range[0]
        
        time_zoom = total_duration / time_span if time_span > 0 else 1.0
        freq_zoom = (sample_rate / 2) / freq_span if freq_span > 0 else 1.0
        
        # Use average zoom level
        zoom_level = (time_zoom + freq_zoom) / 2
        logger.debug(f"Estimated zoom level: {zoom_level:.2f} (time: {time_zoom:.2f}, freq: {freq_zoom:.2f})")
        
        return max(1.0, zoom_level)
    
    def request_tiles_for_region(self, view_type: str, time_range: tuple, freq_range: tuple, zoom_level: float):
        """Request tiles for the specified region."""
        try:
            # Submit tile requests to tile manager
            logger.info(f"Requesting tiles for {view_type}: time={time_range}, freq={freq_range}, zoom={zoom_level:.2f}")
            
            # The tile manager should handle this automatically when we call update_visible_region
            # This is a placeholder for more sophisticated tile request logic
            
        except Exception as e:
            logger.error(f"Error requesting tiles for {view_type}: {e}")
    
    def update_atlas_display_improved(self, view_type: str, time_range: tuple, freq_range: tuple) -> bool:
        """Update display with available atlas data."""
        try:
            atlas = self.tile_manager.atlases.get(view_type)
            if not atlas:
                return False
            
            # Get atlas statistics to see if we have useful data
            stats = atlas.stats
            if stats['slots_used'] == 0:
                logger.debug(f"No tiles loaded in {view_type} atlas")
                return False
            
            # For now, fall back to direct computation since the atlas rendering
            # integration needs more work. This ensures we show data instead of empty view.
            logger.info(f"Atlas has {stats['slots_used']} tiles, but falling back to direct computation")
            return False
            
        except Exception as e:
            logger.error(f"Error updating atlas display for {view_type}: {e}")
            return False
    
    def update_atlas_display(self, view_type: str):
        """Update the display with computed data.
        
        Temporary: Display computed spectrogram directly instead of atlas
        until full tile-based rendering is properly integrated.
        
        Args:
            view_type: Type of view to update
        """
        # For now, use the old direct display method
        # The full tile-atlas system needs more integration work
        # This ensures we show actual data instead of empty atlas
        pass  # Will be handled by old load_spectrogram_data methods
    
    def load_spectrogram_data(self, audio_data: np.ndarray):
        """Load spectrogram data using batched FFT with optional downsampling for large files."""
        try:
            # Check if audio is very large and needs downsampling for display
            max_samples_for_full_resolution = 10 * 60 * 44100  # 10 minutes at 44.1kHz
            
            if len(audio_data) > max_samples_for_full_resolution:
                # For very large files, downsample for initial display
                # Full resolution will be computed tile-by-tile on demand (Phase 3)
                logger.info(f"Large file detected ({len(audio_data)} samples), computing downsampled preview...")
                
                # Downsample factor to bring under limit
                downsample_factor = max(1, int(np.ceil(len(audio_data) / max_samples_for_full_resolution)))
                audio_preview = audio_data[::downsample_factor]
                
                logger.info(f"Downsampled by {downsample_factor}x: {len(audio_preview)} samples")
            else:
                audio_preview = audio_data
                downsample_factor = 1
            
            # Compute STFT using optimized batched FFT
            magnitude_db, frequencies, times = self.spectrogram_engine.compute_stft_batched(audio_preview)
            
            # Adjust times if downsampled
            if downsample_factor > 1:
                times = times * downsample_factor
            
            logger.info(f"Computed spectrogram: {magnitude_db.shape} (downsample: {downsample_factor}x)")
            
            if HAS_VISPY and magnitude_db.size > 0:
                # Update display
                extent = (self.current_view_range[0][0], self.current_view_range[0][1],
                         self.current_view_range[1][0], self.current_view_range[1][1])
                self.spectrogram_canvas.update_image(magnitude_db, extent)
                frames = magnitude_db.shape[1]
                
                if downsample_factor > 1:
                    self.statusBar().showMessage(
                        f"Spectrogram preview ready ({frames} frames, {downsample_factor}x downsampled)")
                else:
                    self.statusBar().showMessage(f"Spectrogram ready ({frames} frames)")
        except Exception as e:
            logger.error(f"Spectrogram computation error: {e}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"Spectrogram error: {str(e)}")
    
    def load_cepstrogram_data(self, audio_data: np.ndarray):
        """Load cepstrogram data."""
        try:
            # First compute spectrogram
            magnitude_db, frequencies, times = self.spectrogram_engine.compute_stft_batched(audio_data)
            
            logger.info(f"Computing cepstrogram from spectrogram: {magnitude_db.shape}")
            
            # Clear mel filterbank cache to rebuild with correct dimensions
            self.cepstrogram_engine._mel_filterbank = None
            
            # Then compute cepstrogram from it
            cepstral = self.cepstrogram_engine.compute_cepstrogram_from_spectrogram(
                magnitude_db, frequencies)
            
            logger.info(f"Computed cepstrogram: {cepstral.shape}")
            
            if HAS_VISPY and cepstral.size > 0:
                extent = (self.current_view_range[0][0], self.current_view_range[0][1],
                         0, cepstral.shape[0])  # Quefrency range
                self.cepstrogram_canvas.update_image(cepstral, extent)
                frames = cepstral.shape[1]
                self.statusBar().showMessage(f"Cepstrogram ready ({frames} frames)")
        except Exception as e:
            logger.error(f"Cepstrogram computation error: {e}")
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"Cepstrogram error: {str(e)}")
    
    def load_fk_data(self, audio_data: np.ndarray):
        """Load F-K transform data."""
        # F-K transform is complex and memory-intensive
        # Disable for now - will be optimized in Phase 5
        logger.info("F-K transform disabled (Phase 5 optimization pending)")
        self.statusBar().showMessage("F-K transform - optimization in progress")
    
    def refresh_current_view(self):
        """Refresh the currently active view."""
        current_index = self.tab_widget.currentIndex()
        self.on_tab_changed(current_index)
    
    def zoom_to_fit(self):
        """Zoom to fit all data."""
        if not self.current_file:
            return
        
        # Reset view range to full audio duration
        duration = self.audio_loader.duration
        sample_rate = self.spectrogram_engine.sample_rate
        
        self.current_view_range = ((0.0, duration), (0.0, sample_rate / 2))
        self.refresh_current_view()
    
    def update_performance_stats(self):
        """Update performance statistics display."""
        # Get GPU memory info
        gpu_mem = self.gpu_memory_manager.get_memory_info()
        
        # Get tile manager stats
        tile_stats = self.tile_manager.get_stats()
        cache_stats = tile_stats.get('cache_stats', {})
        
        # Update status widget with GPU memory
        self.status_widget.update_stats(
            memory_mb=cache_stats.get('disk_usage_mb', 0),
            gpu_mb=gpu_mem.get('used_mb', 0)
        )
        
        # Show tile cache hit rate in status bar
        if cache_stats.get('hit_rate', 0) > 0:
            hit_rate = cache_stats['hit_rate']
            tiles_in_memory = cache_stats.get('memory_tiles', 0)
            tiles_on_disk = cache_stats.get('disk_tiles', 0)
            
            status_msg = (f"Cache: {hit_rate:.1f}% hit rate | "
                         f"{tiles_in_memory} tiles in RAM | "
                         f"{tiles_on_disk} tiles on disk")
            self.statusBar().showMessage(status_msg)
        
        # Show active task progress if any
        active_tasks = self.task_manager.get_active_tasks()
        if active_tasks:
            avg_progress = np.mean([task['progress'] for task in active_tasks])
            self.status_widget.update_stats(progress=avg_progress)
        else:
            self.status_widget.update_stats(progress=1.0)
    
    def export_current_view_as_image(self):
        """Export the current view as an image file."""
        if not self.current_file:
            QMessageBox.warning(self, "Export Warning", "No audio file loaded to export.")
            return
        
        try:
            # Get current tab
            current_index = self.tab_widget.currentIndex()
            tab_names = ['spectrogram', 'cepstrogram', 'fk_transform']
            view_type = tab_names[current_index] if 0 <= current_index < len(tab_names) else 'spectrogram'
            
            # Get canvas
            canvas = None
            if view_type == 'spectrogram' and hasattr(self, 'spectrogram_canvas'):
                canvas = self.spectrogram_canvas
            elif view_type == 'cepstrogram' and hasattr(self, 'cepstrogram_canvas'):
                canvas = self.cepstrogram_canvas
            elif view_type == 'fk_transform' and hasattr(self, 'fk_canvas'):
                canvas = self.fk_canvas
            
            if not canvas or not HAS_VISPY:
                QMessageBox.warning(self, "Export Warning", "No valid view to export.")
                return
            
            # File dialog
            filename, _ = QFileDialog.getSaveFileName(
                self, f"Export {view_type.title()} Image",
                f"{view_type}_{os.path.splitext(os.path.basename(self.current_file))[0]}.png",
                "PNG Images (*.png);;JPEG Images (*.jpg);;All Files (*)"
            )
            
            if filename:
                # Use VisPy's render method
                image = canvas.render()
                if hasattr(image, 'save'):
                    image.save(filename)
                    self.statusBar().showMessage(f"Exported {view_type} image to {filename}")
                    logger.info(f"Exported {view_type} image to {filename}")
                else:
                    # Fallback: save as numpy array converted to image
                    import imageio
                    imageio.imwrite(filename, image)
                    self.statusBar().showMessage(f"Exported {view_type} image to {filename}")
                    logger.info(f"Exported {view_type} image to {filename}")
                    
        except Exception as e:
            logger.error(f"Error exporting image: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export image:\n{str(e)}")
    
    def export_data_as_npy(self):
        """Export the current view data as a NumPy array."""
        if not self.current_file:
            QMessageBox.warning(self, "Export Warning", "No audio file loaded to export.")
            return
        
        try:
            # Get current tab and data
            current_index = self.tab_widget.currentIndex()
            tab_names = ['spectrogram', 'cepstrogram', 'fk_transform']
            view_type = tab_names[current_index] if 0 <= current_index < len(tab_names) else 'spectrogram'
            
            # Get the last computed data for this view
            data = self.get_current_view_data(view_type)
            
            if data is None:
                QMessageBox.warning(self, "Export Warning", f"No {view_type} data available to export.")
                return
            
            # File dialog
            filename, _ = QFileDialog.getSaveFileName(
                self, f"Export {view_type.title()} Data",
                f"{view_type}_{os.path.splitext(os.path.basename(self.current_file))[0]}.npy",
                "NumPy Arrays (*.npy);;All Files (*)"
            )
            
            if filename:
                np.save(filename, data)
                self.statusBar().showMessage(f"Exported {view_type} data to {filename}")
                logger.info(f"Exported {view_type} data to {filename} (shape: {data.shape})")
                
        except Exception as e:
            logger.error(f"Error exporting NPY data: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{str(e)}")
    
    def export_data_as_csv(self):
        """Export the current view data as CSV."""
        if not self.current_file:
            QMessageBox.warning(self, "Export Warning", "No audio file loaded to export.")
            return
        
        try:
            # Get current tab and data
            current_index = self.tab_widget.currentIndex()
            tab_names = ['spectrogram', 'cepstrogram', 'fk_transform']
            view_type = tab_names[current_index] if 0 <= current_index < len(tab_names) else 'spectrogram'
            
            data = self.get_current_view_data(view_type)
            
            if data is None:
                QMessageBox.warning(self, "Export Warning", f"No {view_type} data available to export.")
                return
            
            # File dialog
            filename, _ = QFileDialog.getSaveFileName(
                self, f"Export {view_type.title()} Data",
                f"{view_type}_{os.path.splitext(os.path.basename(self.current_file))[0]}.csv",
                "CSV Files (*.csv);;All Files (*)"
            )
            
            if filename:
                import pandas as pd
                
                # Convert 2D array to DataFrame
                if data.ndim == 2:
                    df = pd.DataFrame(data)
                    df.index.name = 'frequency_bin' if view_type == 'spectrogram' else 'coefficient'
                    df.columns.name = 'time_frame'
                else:
                    df = pd.DataFrame(data.flatten(), columns=[view_type])
                
                df.to_csv(filename)
                self.statusBar().showMessage(f"Exported {view_type} data to {filename}")
                logger.info(f"Exported {view_type} data to {filename} (shape: {data.shape})")
                
        except ImportError:
            QMessageBox.critical(self, "Export Error", "pandas library not available for CSV export.")
        except Exception as e:
            logger.error(f"Error exporting CSV data: {e}")
            QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{str(e)}")
    
    def get_current_view_data(self, view_type: str):
        """Get the current view's data for export."""
        try:
            # For now, recompute the data for the current view range
            # In a full implementation, this could cache the last computed result
            
            if not self.current_file:
                return None
            
            # Get audio data for current view range
            time_range = self.current_view_range[0]
            start_sample = int(time_range[0] * self.spectrogram_engine.sample_rate)
            end_sample = int(time_range[1] * self.spectrogram_engine.sample_rate)
            
            total_samples = self.audio_loader.total_samples
            num_samples = min(end_sample - start_sample, total_samples - start_sample)
            
            if num_samples <= 0:
                return None
            
            audio_data = self.audio_loader.get_chunk(start_sample, num_samples)
            
            if view_type == 'spectrogram':
                magnitude_db, frequencies, times = self.spectrogram_engine.compute_stft_batched(audio_data)
                return magnitude_db
            elif view_type == 'cepstrogram':
                magnitude_db, frequencies, times = self.spectrogram_engine.compute_stft_batched(audio_data)
                # Clear cache to rebuild with correct dimensions
                self.cepstrogram_engine._mel_filterbank = None
                cepstral = self.cepstrogram_engine.compute_cepstrogram_from_spectrogram(magnitude_db, frequencies)
                return cepstral
            elif view_type == 'fk_transform':
                # F-K transform disabled for now
                return None
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting view data for {view_type}: {e}")
            return None
    
    def apply_modern_theme(self):
        """Apply modern glassmorphic dark theme inspired by PyQtGraph example."""
        self.setStyleSheet("""
            /* Main Window - Darker gradient background */
            QMainWindow {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #0A0A15,
                    stop: 0.5 #12121F,
                    stop: 1 #0E1628
                );
            }
            
            /* Central Widget */
            QWidget {
                background: transparent;
                color: rgba(255, 255, 255, 0.9);
                font-size: 10pt;
            }
            
            /* Visualization Container - Distinct darker tone */
            QFrame#viz_container {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(20, 20, 35, 0.95),
                    stop: 1 rgba(15, 15, 30, 0.95)
                );
                border: 1px solid rgba(100, 100, 150, 0.2);
                border-radius: 0px;
            }
            
            /* Tab Widget - Modern style */
            QTabWidget::pane {
                border: none;
                background: transparent;
            }
            
            QTabBar::tab {
                background: rgba(255, 255, 255, 0.05);
                color: rgba(255, 255, 255, 0.7);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 8px 20px;
                margin-right: 4px;
                font-weight: 500;
            }
            
            QTabBar::tab:selected {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 rgba(99, 102, 241, 0.3),
                    stop: 1 rgba(139, 92, 246, 0.3)
                );
                color: rgba(255, 255, 255, 0.95);
                border: 1px solid rgba(139, 92, 246, 0.5);
                border-bottom: none;
            }
            
            QTabBar::tab:hover {
                background: rgba(255, 255, 255, 0.1);
            }
            
            /* Control Frames */
            QFrame {
                background: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 8px;
                padding: 4px;
            }
            
            /* Labels */
            QLabel {
                color: rgba(255, 255, 255, 0.85);
                background: transparent;
                border: none;
                font-size: 10pt;
            }
            
            /* ComboBox - Dropdown */
            QComboBox {
                background: rgba(255, 255, 255, 0.05);
                color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                padding: 5px 10px;
                min-width: 80px;
            }
            
            QComboBox:hover {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid rgba(255, 255, 255, 0.7);
                margin-right: 5px;
            }
            
            QComboBox QAbstractItemView {
                background: rgba(20, 20, 30, 0.95);
                color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                selection-background-color: rgba(99, 102, 241, 0.3);
                padding: 4px;
            }
            
            /* Sliders - Modern style */
            QSlider::groove:horizontal {
                height: 6px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 3px;
            }
            
            QSlider::handle:horizontal {
                width: 16px;
                height: 16px;
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #8B5CF6,
                    stop: 1 #6366F1
                );
                border-radius: 8px;
                margin: -5px 0;
            }
            
            QSlider::handle:horizontal:hover {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #9F6FFF,
                    stop: 1 #7C7FFF
                );
            }
            
            QSlider::sub-page:horizontal {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #6366F1,
                    stop: 1 #8B5CF6
                );
                border-radius: 3px;
            }
            
            /* Buttons - Modern pill style */
            QPushButton {
                background: rgba(255, 255, 255, 0.05);
                color: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                padding: 6px 16px;
                font-weight: 500;
                font-size: 10pt;
            }
            
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            QPushButton:pressed {
                background: rgba(255, 255, 255, 0.03);
            }
            
            /* Progress Bar */
            QProgressBar {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                text-align: center;
                color: rgba(255, 255, 255, 0.9);
                height: 20px;
            }
            
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #6366F1,
                    stop: 1 #8B5CF6
                );
                border-radius: 5px;
            }
            
            /* Status Bar */
            QStatusBar {
                background: rgba(0, 0, 0, 0.3);
                color: rgba(255, 255, 255, 0.7);
                border-top: 1px solid rgba(255, 255, 255, 0.05);
                font-size: 9pt;
            }
            
            /* Menu Bar */
            QMenuBar {
                background: rgba(0, 0, 0, 0.3);
                color: rgba(255, 255, 255, 0.9);
                border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                padding: 4px;
            }
            
            QMenuBar::item {
                background: transparent;
                padding: 6px 12px;
                border-radius: 4px;
            }
            
            QMenuBar::item:selected {
                background: rgba(255, 255, 255, 0.1);
            }
            
            QMenu {
                background: rgba(20, 20, 30, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 4px;
            }
            
            QMenu::item {
                color: rgba(255, 255, 255, 0.9);
                padding: 8px 20px;
                border-radius: 4px;
            }
            
            QMenu::item:selected {
                background: rgba(99, 102, 241, 0.3);
            }
            
            /* Scrollbars - Sleek modern style */
            QScrollBar:vertical {
                background: rgba(255, 255, 255, 0.02);
                width: 12px;
                border-radius: 6px;
            }
            
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                min-height: 30px;
            }
            
            QScrollBar::handle:vertical:hover {
                background: rgba(255, 255, 255, 0.15);
            }
            
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            
            QScrollBar:horizontal {
                background: rgba(255, 255, 255, 0.02);
                height: 12px;
                border-radius: 6px;
            }
            
            QScrollBar::handle:horizontal {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                min-width: 30px;
            }
            
            QScrollBar::handle:horizontal:hover {
                background: rgba(255, 255, 255, 0.15);
            }
            
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
    
    def closeEvent(self, event):
        """Handle application close."""
        logger.info("Application closing - cleaning up resources...")
        
        # Stop performance timer first
        if hasattr(self, 'perf_timer'):
            self.perf_timer.stop()
        
        # Shutdown task manager and wait for threads to finish
        if hasattr(self, 'task_manager'):
            self.task_manager.shutdown(wait=True)
        
        # Clear tile cache
        if hasattr(self, 'tile_manager'):
            self.tile_manager.clear()
        
        # Clear old cache manager
        if hasattr(self, 'cache_manager'):
            self.cache_manager.clear()
        
        # Cleanup GPU memory
        if hasattr(self, 'gpu_memory_manager'):
            self.gpu_memory_manager.cleanup(aggressive=True)
            logger.info("GPU memory cleaned up")
        
        # Cleanup batched FFT engine
        if hasattr(self, 'spectrogram_engine') and hasattr(self.spectrogram_engine, 'batched_fft_engine'):
            self.spectrogram_engine.batched_fft_engine.cleanup()
        
        logger.info("Application closed cleanly")
        event.accept()

def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Audio Visualizer")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("Audio Visualization Team")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    return app.exec()