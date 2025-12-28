"""
VisPy Canvas for spectrogram visualization.
Handles zoom, pan, normalization, and display.
"""
import logging
from typing import Optional, Tuple, Callable
import numpy as np

from .qt_compat import QTimer, Signal

try:
    from vispy import scene
    from vispy.visuals.axis import AxisVisual
    HAS_VISPY = True
except ImportError:
    scene = None
    AxisVisual = None
    HAS_VISPY = False

from ..core.adaptive_spectrogram import ViewRegion, ZoomLevelDetector

logger = logging.getLogger(__name__)


class TimeAxisFormatter:
    """Custom tick formatter for time axis (displays mm:ss format)."""
    
    @staticmethod
    def format_tick(value: float) -> str:
        """Format time value as mm:ss or ss.ms depending on magnitude."""
        if value < 0:
            return ""
        if value < 60:
            return f"{value:.1f}s"
        minutes = int(value // 60)
        seconds = value % 60
        if seconds == int(seconds):
            return f"{minutes}:{int(seconds):02d}"
        return f"{minutes}:{seconds:04.1f}"
    
    @staticmethod
    def get_ticks(t_min: float, t_max: float, n_ticks: int = 8) -> list:
        """Generate nice tick positions for time axis."""
        span = t_max - t_min
        if span <= 0:
            return [t_min]
        
        # Choose interval based on span
        if span < 0.5:
            interval = 0.1
        elif span < 2:
            interval = 0.25
        elif span < 5:
            interval = 0.5
        elif span < 10:
            interval = 1
        elif span < 30:
            interval = 5
        elif span < 60:
            interval = 10
        elif span < 300:
            interval = 30
        elif span < 600:
            interval = 60
        else:
            interval = 120
        
        start = np.ceil(t_min / interval) * interval
        ticks = []
        current = start
        while current <= t_max and len(ticks) < n_ticks:
            ticks.append(current)
            current += interval
        return ticks


class FreqAxisFormatter:
    """Custom tick formatter for frequency axis (displays kHz format)."""
    
    @staticmethod
    def format_tick(value: float) -> str:
        """Format frequency value as kHz or Hz."""
        if value < 0:
            return ""
        if value >= 10000:
            return f"{value/1000:.0f}k"
        elif value >= 1000:
            return f"{value/1000:.1f}k"
        return f"{value:.0f}"
    
    @staticmethod
    def get_ticks(f_min: float, f_max: float, n_ticks: int = 8) -> list:
        """Generate nice tick positions for frequency axis."""
        span = f_max - f_min
        if span <= 0:
            return [f_min]
        
        # Choose interval based on span
        if span < 100:
            interval = 20
        elif span < 500:
            interval = 100
        elif span < 1000:
            interval = 200
        elif span < 5000:
            interval = 500
        elif span < 10000:
            interval = 1000
        elif span < 20000:
            interval = 2000
        else:
            interval = 5000
        
        start = np.ceil(f_min / interval) * interval
        ticks = []
        current = start
        while current <= f_max and len(ticks) < n_ticks:
            ticks.append(current)
            current += interval
        return ticks


class TiledImageRenderer:
    """Manages multiple VisPy Image visuals for tile-based rendering without downsampling."""
    
    def __init__(self, parent_view):
        """Initialize tiled renderer."""
        self.view = parent_view
        self.tiles = []
        self.enabled = False
        
    def clear_tiles(self):
        """Remove all tile visuals from the scene."""
        for tile_info in self.tiles:
            if tile_info['visual'].parent:
                tile_info['visual'].parent = None
        self.tiles.clear()
        logger.debug("Cleared all tile visuals")
    
    def render_tiles(self, tile_data_list: list, time_range: tuple, freq_range: tuple, 
                     tile_time_duration: float) -> bool:
        """Render multiple tiles as separate Image visuals positioned in the scene."""
        try:
            self.clear_tiles()
            
            if not tile_data_list:
                logger.warning("No tiles to render")
                return False
            
            time_start, time_end = time_range
            freq_start, freq_end = freq_range
            
            logger.info(f"Rendering {len(tile_data_list)} tiles")
            
            # Calculate global min/max for consistent normalization
            global_min = min(tile.min() for tile in tile_data_list if tile is not None and tile.size > 0)
            global_max = max(tile.max() for tile in tile_data_list if tile is not None and tile.size > 0)
            
            current_time = time_start
            for i, tile_data in enumerate(tile_data_list):
                if tile_data is None or tile_data.size == 0:
                    continue
                
                tile_time_end = min(current_time + tile_time_duration, time_end)
                tile_time_width = tile_time_end - current_time
                
                # Normalize using global range
                if global_max > global_min:
                    display_data = (tile_data - global_min) / (global_max - global_min)
                else:
                    display_data = np.zeros_like(tile_data)
                
                display_data = display_data.astype(np.float32)
                
                # Create Image visual
                tile_visual = scene.visuals.Image(
                    display_data,
                    parent=self.view.scene,
                    interpolation='nearest',
                    cmap='plasma'
                )
                tile_visual.clim = (0, 1)
                
                # Position tile
                freq_height = freq_end - freq_start
                x_scale = tile_time_width / tile_data.shape[1]
                y_scale = freq_height / tile_data.shape[0]
                
                transform = scene.STTransform(
                    scale=(x_scale, y_scale),
                    translate=(current_time, freq_start)
                )
                tile_visual.transform = transform
                
                self.tiles.append({
                    'visual': tile_visual,
                    'bounds': (current_time, tile_time_end, freq_start, freq_end),
                    'data_shape': tile_data.shape
                })
                
                current_time = tile_time_end
            
            self.enabled = True
            logger.info(f"Successfully rendered {len(self.tiles)} tiles")
            return True
            
        except Exception as e:
            logger.error(f"Error rendering tiles: {e}")
            return False
    
    def cull_tiles(self, viewport_time: tuple, viewport_freq: tuple):
        """Show/hide tiles based on viewport visibility."""
        if not self.enabled:
            return
        
        for tile_info in self.tiles:
            t0, t1, f0, f1 = tile_info['bounds']
            vt0, vt1 = viewport_time
            vf0, vf1 = viewport_freq
            
            time_overlap = not (t1 < vt0 or t0 > vt1)
            freq_overlap = not (f1 < vf0 or f0 > vf1)
            tile_info['visual'].visible = time_overlap and freq_overlap


class VisPyCanvas(scene.SceneCanvas):
    """Custom VisPy canvas for audio visualization with proper axes."""
    
    def __init__(self, view_type: str, parent=None):
        if not HAS_VISPY:
            raise RuntimeError("VisPy not available")
        
        super().__init__(keys='interactive', parent=parent, size=(800, 600))
        
        self.unfreeze()
        self.view_type = view_type
        
        # Grid layout for axes
        self.grid = self.central_widget.add_grid(margin=0)
        self.grid.spacing = 0
        
        # Create axis widgets
        self.y_axis = scene.AxisWidget(
            orientation='left', 
            axis_label='Frequency (Hz)',
            axis_font_size=6, 
            axis_label_margin=30,
            tick_label_margin=5
        )
        self.y_axis.width_max = 60
        self.y_axis.width_min = 55
        
        self.x_axis = scene.AxisWidget(
            orientation='bottom', 
            axis_label='Time (s)',
            axis_font_size=6, 
            axis_label_margin=20,
            tick_label_margin=5
        )
        self.x_axis.height_max = 40
        self.x_axis.height_min = 35
        
        # Create ViewBox
        self.view = scene.ViewBox(camera='panzoom', parent=None)
        
        # Grid layout
        self.grid.add_widget(self.y_axis, row=0, col=0)
        self.grid.add_widget(self.view, row=0, col=1)
        self.grid.add_widget(self.x_axis, row=1, col=1)
        
        # Link axes
        self.y_axis.link_view(self.view)
        self.x_axis.link_view(self.view)
        
        # Configure axis ticks
        self.y_axis.axis.tick_font_size = 6
        self.x_axis.axis.tick_font_size = 6
        
        # Configure camera
        self.view.camera.aspect = None
        self.view.camera.rect = (-1, -1, 2, 2)
        self.view.camera.interactive = False
        self.view.camera.flip = (False, False, False)
        self.view.camera.zoom_factor = 4.0
        
        # Data bounds
        self.data_bounds = None
        
        # Image visual
        self.image_visual = scene.visuals.Image(parent=self.view.scene, interpolation='bilinear', cmap='plasma')
        self.image_visual.order = 0  # Base layer
        self.current_interpolation = 'bilinear'
        
        # Tiled renderer
        self.tiled_renderer = TiledImageRenderer(self.view)
        
        # Zoom detector (disabled for performance)
        self.zoom_detector = ZoomLevelDetector(recompute_threshold=10.0)
        self.last_zoom_recompute_time = 0
        self.zoom_recompute_delay = 5.0
        self.pending_zoom_recompute = False
        self.enable_zoom_recompute = False
        self.on_zoom_recompute_callback = None
        
        # Callback for zoom level change (for status bar update)
        self.on_zoom_changed_callback = None
        
        # Callback for cursor position change
        self.on_cursor_moved_callback = None
        
        # Debounced normalization
        self.normalization_timer = QTimer()
        self.normalization_timer.setSingleShot(True)
        self.normalization_timer.timeout.connect(self._do_debounced_normalization)
        self.normalization_pending = False
        
        # Debounced auto-recompute on zoom (500ms delay)
        # DISABLED by default - causes alignment issues when computing partial regions
        # User can still press F5 to recompute at higher resolution
        self.recompute_timer = QTimer()
        self.recompute_timer.setSingleShot(True)
        self.recompute_timer.timeout.connect(self._do_auto_recompute)
        self.auto_recompute_enabled = False  # Disabled - use F5 instead
        self.on_auto_recompute_callback = None
        
        # Mouse state
        self.mouse_pos = (0, 0)
        self.crosshair_enabled = False
        self.is_panning = False
        self.last_mouse_pos = None
        self.pan_speed = 1.0
        
        # Crosshair visuals
        self.crosshair_v = scene.visuals.Line(color=(0.5, 0.3, 0.9, 0.8), width=1.5, parent=self.view.scene)
        self.crosshair_h = scene.visuals.Line(color=(0.5, 0.3, 0.9, 0.8), width=1.5, parent=self.view.scene)
        self.crosshair_v.visible = False
        self.crosshair_h.visible = False
        
        # Measurement tool state
        self.measurement_mode = False
        self.measurement_start = None  # (time, freq) - current start point
        self.measurement_end = None    # (time, freq) - current end point  
        self.measurement_all_points = []  # All points in current sequence [(t, f), ...]
        self._measurement_mode_callback = None
        self._on_measurement_callback = None  # Callback when measurement completes
        
        # Annotation drawing state
        self.annotation_mode = False
        self.annotation_drawing = False
        self.annotation_start = None  # (time, freq)
        self.annotation_end = None    # (time, freq)
        self._annotation_renderer = None
        self._on_annotation_created_callback = None
        self._on_annotation_clicked_callback = None
        self._on_annotation_context_menu_callback = None  # For right-click menu
        
        # Measurement visuals - using same setup as annotation rectangles
        # Line for all measurement segments
        self.measurement_line = scene.visuals.Line(parent=self.view.scene)
        self.measurement_line.set_data(color=(1.0, 0.8, 0.0, 0.9), width=2.0)
        self.measurement_line.visible = False
        self.measurement_line.order = 150  # Above spectrogram
        self.measurement_line.set_gl_state('translucent', depth_test=False)
        
        # Markers for all measurement points
        self.measurement_markers = scene.visuals.Markers(parent=self.view.scene)
        self.measurement_markers.visible = False
        self.measurement_markers.order = 200  # On top
        self.measurement_markers.set_gl_state('translucent', depth_test=False)
        
        # Legacy markers for backwards compatibility
        self.measurement_start_marker = scene.visuals.Markers(parent=self.view.scene)
        self.measurement_start_marker.visible = False
        self.measurement_start_marker.order = 200  # On top
        self.measurement_start_marker.set_gl_state('translucent', depth_test=False)
        
        self.measurement_end_marker = scene.visuals.Markers(parent=self.view.scene)
        self.measurement_end_marker.visible = False
        self.measurement_end_marker.order = 200  # On top
        self.measurement_end_marker.set_gl_state('translucent', depth_test=False)
        self.measurement_text = scene.visuals.Text('', color='yellow', font_size=12,
                                                    bold=True, parent=self.view.scene)
        self.measurement_text.visible = False
        
        # Curve Drawing State (Doppler)
        # Curve drawing visuals - using same setup as annotation rectangles
        self.curve_mode = False
        self.curve_points = []  # List of (time, freq)
        self.curve_visual = scene.visuals.Line(parent=self.view.scene, method='gl')
        self.curve_visual.set_data(color='cyan', width=2.0)
        self.curve_visual.visible = False
        self.curve_visual.order = 150  # Above spectrogram
        self.curve_visual.set_gl_state('translucent', depth_test=False)  # Same as annotations
        
        self.curve_markers = scene.visuals.Markers(parent=self.view.scene)
        self.curve_markers.visible = False
        self.curve_markers.order = 200  # On top
        self.curve_markers.set_gl_state('translucent', depth_test=False)
        
        self._on_curve_updated_callback = None  # Callback(points) when curve changes
        self._on_curve_completed_callback = None  # Callback(points) when Enter pressed when curve changes
        self._on_curve_completed_callback = None  # Callback(points) when user presses Enter

        # Event Tagging Mode (vertical lines for marking events)
        self.event_mode = False
        self.event_t1: Optional[float] = None     # First time marker (click 1)
        self.event_t2: Optional[float] = None     # Second time marker (click 2)
        self.event_line_1: Optional[scene.visuals.Line] = None
        self.event_line_2: Optional[scene.visuals.Line] = None
        self._on_event_created_callback = None    # Callback(t_start, t_end) when both lines placed
        self._on_event_mode_toggled_callback = None  # Callback(enabled: bool) when mode toggled via E key

        # Detected Doppler Tracks (from full spectrogram detection)
        self.detected_track_visuals = []  # List of Line visuals for detected tracks

        # Text readout
        self.text_visual = scene.visuals.Text('', color='white', font_size=11,
                                              pos=(10, 30), parent=self.view.scene)

        # Mode indicator (shows current mode in top-left corner)
        self.mode_indicator = scene.visuals.Text('', color='white', font_size=12,
                                                  bold=True, parent=self.view.scene)
        self.mode_indicator.order = 250  # On top of everything
        self._current_mode_name = None  # Track current mode for efficient updates

        # OpenGL limit
        self.max_texture_size = 16384
        
        # Data for normalization
        self.raw_display_data = None
        self.global_data_range = None
        self.current_clim = None
        self.db_range = None
        self.display_extent = None
        self.normalized_display_data = None
        self.global_percentiles = None
        self.global_mean_std = None
        self.local_mean_std = None  # Mean/std for visible region (for STD normalization)
        self.normalization_mode = 'std'  # 'minmax' or 'std' - default to STD (adaptive to zoom)
        self.std_scale = 2.5  # Scale factor for STD normalization

        # Zoom history for undo (Ctrl+Z or Backspace)
        self.zoom_history = []  # Stack of (x, y, width, height) tuples
        self.zoom_history_max = 20  # Maximum history size
        self.zoom_forward_history = []  # For redo (Ctrl+Y)

        # Connect events
        self.view.events.mouse_wheel.connect(self.on_mouse_wheel)
        self.events.key_press.connect(self.on_key_press)
        self.events.mouse_move.connect(self.on_mouse_move)
        self.events.mouse_press.connect(self.on_mouse_press)
        self.events.mouse_release.connect(self.on_mouse_release)
        
        self.freeze()
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for axis-specific zoom."""
        event.handled = True

        # Get modifiers
        modifiers = []
        try:
            if hasattr(event, 'modifiers') and event.modifiers:
                modifiers = event.modifiers
            elif hasattr(event, 'mouse_event') and hasattr(event.mouse_event, 'modifiers'):
                modifiers = event.mouse_event.modifiers or []
        except AttributeError:
            modifiers = []

        mod_strings = [str(mod).lower() for mod in modifiers]
        shift_pressed = any('shift' in s for s in mod_strings)
        ctrl_pressed = any('ctrl' in s or 'control' in s for s in mod_strings)

        # 15% zoom per scroll
        factor = 1.15 if event.delta[1] > 0 else 0.87

        if shift_pressed:
            scale_factors = [factor, 1.0]  # Time only
        elif ctrl_pressed:
            scale_factors = [1.0, factor]  # Freq only
        else:
            scale_factors = [factor, factor]  # Both

        self.zoom_with_center(scale_factors, event.pos)
    
    def _save_zoom_state(self):
        """Save current zoom state to history stack."""
        rect = self.view.camera.rect
        if rect is None:
            return
        state = (float(rect.left), float(rect.bottom), float(rect.width), float(rect.height))

        # Don't save if it's identical to the last state
        if self.zoom_history and self.zoom_history[-1] == state:
            return

        self.zoom_history.append(state)

        # Limit history size
        while len(self.zoom_history) > self.zoom_history_max:
            self.zoom_history.pop(0)

        # Clear forward history on new action
        self.zoom_forward_history.clear()

    def zoom_undo(self):
        """Undo last zoom - go back to previous zoom state."""
        if not self.zoom_history:
            logger.debug("No zoom history to undo")
            return False

        # Save current state for redo
        rect = self.view.camera.rect
        if rect is not None:
            current = (float(rect.left), float(rect.bottom), float(rect.width), float(rect.height))
            self.zoom_forward_history.append(current)

        # Restore previous state
        x, y, w, h = self.zoom_history.pop()
        self.view.camera.rect = (x, y, w, h)
        self.update_dynamic_clim()
        self.notify_zoom_changed()
        logger.debug(f"Zoom undo: restored to ({x:.2f}, {y:.2f}, {w:.2f}, {h:.2f})")
        return True

    def zoom_redo(self):
        """Redo zoom - go forward in zoom history."""
        if not self.zoom_forward_history:
            logger.debug("No zoom forward history to redo")
            return False

        # Save current state
        rect = self.view.camera.rect
        if rect is not None:
            current = (float(rect.left), float(rect.bottom), float(rect.width), float(rect.height))
            self.zoom_history.append(current)

        # Restore forward state
        x, y, w, h = self.zoom_forward_history.pop()
        self.view.camera.rect = (x, y, w, h)
        self.update_dynamic_clim()
        self.notify_zoom_changed()
        logger.debug(f"Zoom redo: restored to ({x:.2f}, {y:.2f}, {w:.2f}, {h:.2f})")
        return True

    def reset_zoom_history(self):
        """Clear zoom history (e.g., when loading new file)."""
        self.zoom_history.clear()
        self.zoom_forward_history.clear()

    def zoom_with_center(self, scale_factors, mouse_pos):
        """Zoom with center-based scaling."""
        try:
            rect = self.view.camera.rect
            if rect is None:
                return

            # Save current state before zooming
            self._save_zoom_state()

            current_width = float(rect.width)
            current_height = float(rect.height)
            current_x = float(rect.left)
            current_y = float(rect.bottom)

            if current_width <= 0 or current_height <= 0:
                return

            new_width = current_width / scale_factors[0]
            new_height = current_height / scale_factors[1]

            # Apply boundary constraints
            if self.data_bounds:
                time_min, time_max, freq_min, freq_max = self.data_bounds

                max_width = time_max - time_min
                min_width = max_width / 100
                new_width = max(min_width, min(new_width, max_width))

                max_height = freq_max - freq_min
                min_height = max_height / 100
                new_height = max(min_height, min(new_height, max_height))

            # Zoom around center
            center_x = current_x + current_width / 2
            center_y = current_y + current_height / 2
            new_x = center_x - new_width / 2
            new_y = center_y - new_height / 2

            # Constrain position
            if self.data_bounds:
                time_min, time_max, freq_min, freq_max = self.data_bounds

                max_x = time_max - new_width
                new_x = time_min if max_x < time_min else max(time_min, min(new_x, max_x))

                max_y = freq_max - new_height
                new_y = freq_min if max_y < freq_min else max(freq_min, min(new_y, max_y))

            if new_width > 0 and new_height > 0:
                self.view.camera.rect = (new_x, new_y, new_width, new_height)
                self.update_dynamic_clim()
                self.notify_zoom_changed()
                self.schedule_auto_recompute()
            
        except Exception:
            pass
    
    def on_key_press(self, event):
        """Handle key press events for zoom controls, navigation, and modes."""
        # Arrow keys for navigation - work in ALL modes (annotation, measurement, event, etc.)
        if event.key == 'Left':
            self._pan_by_fraction(-0.2, 0)  # Pan left 20% of view
            event.handled = True
            return
        elif event.key == 'Right':
            self._pan_by_fraction(0.2, 0)  # Pan right 20% of view
            event.handled = True
            return
        elif event.key == 'Up':
            self._pan_by_fraction(0, 0.2)  # Pan up 20% of view
            event.handled = True
            return
        elif event.key == 'Down':
            self._pan_by_fraction(0, -0.2)  # Pan down 20% of view
            event.handled = True
            return

        # Zoom controls
        if event.key == 'X':
            self.zoom_with_center([2/3, 1.0], None)
        elif event.key == 'x':
            self.zoom_with_center([1.5, 1.0], None)
        elif event.key == 'Y':
            self.zoom_with_center([1.0, 2/3], None)
        elif event.key == 'y':
            self.zoom_with_center([1.0, 1.5], None)
        elif event.key in ('R', 'r'):
            self.reset_camera_to_data_bounds()
        elif event.key in ('M', 'm'):
            # Toggle measurement mode
            self.toggle_measurement_mode()
        elif event.key in ('C', 'c'):
            # Toggle curve drawing mode
            self.set_curve_mode(not self.curve_mode)
        elif event.key == 'Return' or event.key == 'Enter':
            # Finish curve drawing (save to selected annotation)
            print(f"DEBUG: Enter key pressed! curve_mode={self.curve_mode}, points={len(self.curve_points)}")
            if self.curve_mode and len(self.curve_points) >= 2:
                print(f"DEBUG: Entering save flow with {len(self.curve_points)} points")
                # Show immediate feedback
                self.update_text_readout(f"Processing {len(self.curve_points)} points, please wait...", (10, 60))

                logger.info(f"Enter pressed: Curve completed with {len(self.curve_points)} points")
                # Store points before callback (in case callback clears them)
                points_copy = list(self.curve_points)
                logger.info(f"Points copy created: {points_copy[:2]}... (showing first 2)")

                # Call callback to save curve
                if self._on_curve_completed_callback:
                    logger.info("Calling curve completed callback...")
                    try:
                        self._on_curve_completed_callback(points_copy)
                        logger.info("Callback completed successfully")
                    except Exception as e:
                        logger.error(f"Error in curve completed callback: {e}", exc_info=True)
                else:
                    logger.warning("No curve completed callback set!")

                # Exit curve mode and clear curve
                self.set_curve_mode(False)
                self.clear_curve()
                self.update_text_readout(f"Curve saved: {len(points_copy)} points", (10, 60))
            elif self.curve_mode:
                logger.warning(f"Need at least 2 points (have {len(self.curve_points)})")
                self.update_text_readout(f"Need at least 2 points (have {len(self.curve_points)})", (10, 60))
        elif event.key in ('A', 'a'):
            # Toggle annotation mode
            self.set_annotation_mode(not self.annotation_mode)
        elif event.key in ('E', 'e'):
            # Toggle event tagging mode
            new_state = not self.event_mode
            self.set_event_mode(new_state)
            # Notify main window to show/hide event panel
            if self._on_event_mode_toggled_callback:
                self._on_event_mode_toggled_callback(new_state)
        elif event.key == 'Escape':
            # Clear measurement, curve, or event markers, but DON'T close the window
            if self.event_mode:
                self.clear_event_markers()
                event.handled = True
            elif self.measurement_mode:
                self.clear_measurement()
                # Don't exit measurement mode, just clear current measurement
                event.handled = True
            elif self.curve_mode:
                self.clear_curve()
                # Optionally exit curve mode
                self.set_curve_mode(False)
                event.handled = True
        elif event.key == 'Backspace':
            # Zoom undo - go back to previous zoom level
            if self.zoom_undo():
                event.handled = True
        # Note: Ctrl+Z/Ctrl+Y handled by main window for consistency

    def on_mouse_press(self, event):
        """Handle mouse press for panning, measurement, or annotation."""
        # Handle Right Click (Context Menu)
        if event.button == 2:  # 2 is Right Click in VisPy
            try:
                world_pos = self._screen_to_world(event.pos)
                if world_pos is not None:
                    time_pos, freq_pos = world_pos

                    # If in curve mode, right click clears the curve
                    if self.curve_mode:
                        logger.info("Right-click: Clearing curve")
                        self.clear_curve()
                        self.update_text_readout("CURVE MODE: Curve cleared | Click to add points (need 2 min) | Enter to finish", (10, 60))
                        event.handled = True
                        return

                    logger.info(f"Right click at t={time_pos:.3f}, f={freq_pos:.0f}")
                    if self._on_annotation_context_menu_callback:
                        self._on_annotation_context_menu_callback(time_pos, freq_pos)
                        event.handled = True
                        return
                    else:
                        logger.warning("Right click: no context menu callback set")
                else:
                    logger.warning("Right click: could not convert screen to world coordinates")
            except Exception as e:
                logger.warning(f"Error handling right click: {e}", exc_info=True)
        
        if event.button == 1:
            # Priority 0: Event Tagging Mode (vertical lines)
            if self.event_mode:
                world_pos = self._screen_to_world(event.pos)
                if world_pos is not None:
                    time_pos = world_pos[0]
                    self._add_event_marker(time_pos)
                    event.handled = True
                    return

            # Priority 1: Curve Drawing Mode
            if self.curve_mode:
                world_pos = self._screen_to_world(event.pos)
                if world_pos is not None:
                    self.add_curve_point(world_pos[0], world_pos[1])
                    logger.debug(f"Added curve point: t={world_pos[0]:.3f}s, f={world_pos[1]:.1f}Hz (total: {len(self.curve_points)})")
                    event.handled = True
                    return

            # Priority 2: Check if in annotation mode
            if self.annotation_mode:
                try:
                    # Convert screen coordinates to world coordinates (time, freq)
                    world_pos = self._screen_to_world(event.pos)
                    if world_pos is not None:
                        time_pos, freq_pos = world_pos
                        logger.debug(f"Annotation press at t={time_pos:.3f}s, f={freq_pos:.0f}Hz")
                        
                        # Check if clicking on existing annotation
                        if self._on_annotation_clicked_callback:
                            clicked_ann = self._on_annotation_clicked_callback(time_pos, freq_pos)
                            if clicked_ann:
                                event.handled = True
                                return
                        
                        # Start drawing new annotation
                        self.annotation_drawing = True
                        self.annotation_start = (time_pos, freq_pos)
                        self.annotation_end = None
                        
                        # Show initial temp rectangle
                        if self._annotation_renderer:
                            self._annotation_renderer.show_temp_rectangle(
                                time_pos, time_pos, freq_pos, freq_pos
                            )
                        
                        event.handled = True
                        return
                except Exception as e:
                    logger.warning(f"Error in annotation mouse press: {e}", exc_info=True)
            
            # Check if in measurement mode
            if self.measurement_mode:
                world_pos = self._screen_to_world(event.pos)
                if world_pos is not None:
                    self.set_measurement_point(world_pos[0], world_pos[1])
                    event.handled = True
                    return
            
            # Normal panning (only if not in special modes)
            if not self.annotation_mode and not self.measurement_mode and not self.curve_mode and not self.event_mode:
                self.is_panning = True
                self.last_mouse_pos = event.pos
                event.handled = True
    
    def on_mouse_release(self, event):
        """Handle mouse release."""
        if event.button == 1:
            # Priority 1: Finish annotation drawing
            if self.annotation_drawing and self.annotation_start is not None:
                try:
                    # Convert screen coordinates to world coordinates
                    world_pos = self._screen_to_world(event.pos)
                    if world_pos is not None:
                        time_pos, freq_pos = world_pos
                        self.annotation_end = (time_pos, freq_pos)
                        
                        # Get start and end positions
                        t_start, f_start = self.annotation_start
                        t_end, f_end = self.annotation_end
                        
                        # Calculate size
                        dt = abs(t_end - t_start)
                        df = abs(f_end - f_start)
                        
                        logger.debug(f"Annotation release: dt={dt:.3f}s, df={df:.0f}Hz")
                        
                        # Only create annotation if it has meaningful size
                        if dt > 0.001 and df > 1.0:
                            # Ensure correct ordering (min/max)
                            t_min = min(t_start, t_end)
                            t_max = max(t_start, t_end)
                            f_min = min(f_start, f_end)
                            f_max = max(f_start, f_end)
                            
                            logger.info(f"Creating annotation: time=[{t_min:.3f}, {t_max:.3f}]s, freq=[{f_min:.0f}, {f_max:.0f}]Hz")
                            
                            if self._on_annotation_created_callback:
                                self._on_annotation_created_callback(
                                    t_min, t_max, f_min, f_max
                                )
                        else:
                            logger.debug(f"Annotation too small (dt={dt:.4f}s, df={df:.1f}Hz), cancelled")
                    
                    # Always clean up drawing state
                    self.annotation_drawing = False
                    self.annotation_start = None
                    self.annotation_end = None
                    
                    # Hide temp rectangle
                    if self._annotation_renderer:
                        self._annotation_renderer.hide_temp_rectangle()
                    
                    # Force update to remove temp rectangle
                    self.update()
                    
                    event.handled = True
                    return
                    
                except Exception as e:
                    logger.warning(f"Error in annotation mouse release: {e}", exc_info=True)
                    # Clean up on error
                    self.annotation_drawing = False
                    self.annotation_start = None
                    self.annotation_end = None
                    if self._annotation_renderer:
                        self._annotation_renderer.hide_temp_rectangle()
                    self.update()
            
            # Clean up panning state
            self.is_panning = False
            self.last_mouse_pos = None
            event.handled = True
    
    def on_mouse_move(self, event):
        """Handle mouse movement for panning, crosshair, and annotation drawing."""
        if event.pos is None:
            return
        
        # Priority 1: Handle annotation drawing (live rectangle preview)
        if self.annotation_drawing and self.annotation_start is not None:
            try:
                world_pos = self._screen_to_world(event.pos)
                if world_pos is not None:
                    time_pos, freq_pos = world_pos
                    t_start, f_start = self.annotation_start
                    
                    # Update temporary rectangle in real-time
                    if self._annotation_renderer:
                        self._annotation_renderer.show_temp_rectangle(
                            t_start, time_pos, f_start, freq_pos
                        )
                        # Force canvas update for immediate visual feedback
                        self.update()
                    
                    # Also update cursor position for status bar
                    if self.on_cursor_moved_callback:
                        self.on_cursor_moved_callback(time_pos, freq_pos)
                    
                    event.handled = True
                    return
            except Exception as e:
                logger.debug(f"Error in annotation drawing: {e}")
        
        # Priority 2: Handle panning
        if self.is_panning and self.last_mouse_pos is not None:
            try:
                delta_screen = event.pos - self.last_mouse_pos
                rect = self.view.camera.rect
                if rect is None:
                    return
                
                canvas_size = self.size
                if canvas_size[0] <= 0 or canvas_size[1] <= 0:
                    return
                
                delta_x = -(delta_screen[0] / canvas_size[0]) * rect.width * self.pan_speed
                delta_y = (delta_screen[1] / canvas_size[1]) * rect.height * self.pan_speed
                
                new_left = rect.left + delta_x
                new_bottom = rect.bottom + delta_y
                
                if self.data_bounds:
                    time_min, time_max, freq_min, freq_max = self.data_bounds
                    
                    if new_left < time_min:
                        new_left = time_min
                    if new_left + rect.width > time_max:
                        new_left = time_max - rect.width
                    
                    if new_bottom < freq_min:
                        new_bottom = freq_min
                    if new_bottom + rect.height > freq_max:
                        new_bottom = freq_max - rect.height
                
                self.view.camera.rect = (new_left, new_bottom, rect.width, rect.height)
                self.update_dynamic_clim()
                self.last_mouse_pos = event.pos
                
            except Exception:
                pass
            return
        
        # Priority 3: Normal mouse hover (crosshair and status updates)
        try:
            world_pos = self._screen_to_world(event.pos)
            if world_pos is not None:
                time_pos, freq_pos = world_pos
                self.mouse_pos = (time_pos, freq_pos)
                
                # Update cursor position callback (for status bar)
                if self.on_cursor_moved_callback:
                    self.on_cursor_moved_callback(time_pos, freq_pos)
                
                # Update crosshair if enabled
                if self.crosshair_enabled:
                    self.set_crosshair(True, self.mouse_pos)
                    readout = f"Time: {self._format_time(time_pos)}, Freq: {self._format_freq(freq_pos)}"
                    self.update_text_readout(readout, (10, 30))
        except Exception as e:
            logger.debug(f"Error in mouse move: {e}")
    
    def set_data_bounds(self, time_min, time_max, freq_min, freq_max):
        """Set data bounds for zoom/pan constraints."""
        self.data_bounds = (time_min, time_max, freq_min, freq_max)
        logger.info(f"Data bounds set: time=[{time_min:.3f}, {time_max:.3f}], freq=[{freq_min:.1f}, {freq_max:.1f}]")
    
    def reset_camera_to_data_bounds(self):
        """Reset camera to show all data."""
        if self.data_bounds:
            time_min, time_max, freq_min, freq_max = self.data_bounds
            self.view.camera.rect = (time_min, freq_min, time_max - time_min, freq_max - freq_min)
            self.update_dynamic_clim()

    def _pan_by_fraction(self, dx_fraction: float, dy_fraction: float):
        """Pan the view by a fraction of the current view size.

        Works in all modes (annotation, measurement, event) - allows navigation
        while marking regions.

        Args:
            dx_fraction: Horizontal pan as fraction of view width (positive = right)
            dy_fraction: Vertical pan as fraction of view height (positive = up)
        """
        rect = self.view.camera.rect
        if rect is None:
            return

        # Calculate pan amounts
        dx = dx_fraction * rect.width
        dy = dy_fraction * rect.height

        # Calculate new position
        new_left = rect.left + dx
        new_bottom = rect.bottom + dy

        # Constrain to data bounds if set
        if self.data_bounds:
            time_min, time_max, freq_min, freq_max = self.data_bounds
            # Don't pan beyond data bounds
            new_left = max(time_min, min(new_left, time_max - rect.width))
            new_bottom = max(freq_min, min(new_bottom, freq_max - rect.height))

        # Apply new camera rect
        self.view.camera.rect = (new_left, new_bottom, rect.width, rect.height)

        # Update display
        self.notify_zoom_changed()
        self.schedule_auto_recompute()
        self.update()

    def update_image(self, data: np.ndarray, extent: tuple = None, preserve_view: bool = False):
        """Update the displayed image data.
        
        Args:
            data: Image data to display
            extent: (time_start, time_end, freq_start, freq_end)
            preserve_view: If True, don't change camera position (for zoomed recompute)
        """
        if data is None or data.size == 0:
            return
        
        # Save current camera position if preserving view
        saved_camera_rect = None
        if preserve_view and self.view.camera.rect is not None:
            rect = self.view.camera.rect
            saved_camera_rect = (rect.left, rect.bottom, rect.width, rect.height)
        
        display_data = data.astype(np.float32)
        
        # Handle 2D data
        if display_data.ndim == 2:
            # Check texture limits
            if display_data.shape[1] > self.max_texture_size:
                factor = int(np.ceil(display_data.shape[1] / self.max_texture_size))
                logger.warning(f"Downsampling by {factor}x in time")
                display_data = display_data[:, ::factor]
            
            if display_data.shape[0] > self.max_texture_size:
                factor = int(np.ceil(display_data.shape[0] / self.max_texture_size))
                logger.warning(f"Downsampling by {factor}x in frequency")
                display_data = display_data[::factor, :]
            
            # Handle non-finite values
            display_data = np.nan_to_num(display_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calculate statistics
            data_min = float(np.nanmin(display_data))
            data_max = float(np.nanmax(display_data))
            if data_max - data_min < 1e-6:
                data_max = data_min + 1.0
            
            self.global_data_range = (data_min, data_max)
            
            finite_values = display_data[np.isfinite(display_data)]
            if finite_values.size > 0:
                self.global_percentiles = (
                    float(np.percentile(finite_values, 5)),
                    float(np.percentile(finite_values, 95))
                )
                mean = float(np.mean(finite_values))
                std = max(float(np.std(finite_values)), 1e-6)
                self.global_mean_std = (mean, std)
            
            logger.info(f"Display shape: {display_data.shape}, range: [{data_min:.1f}, {data_max:.1f}]")
        
        self.raw_display_data = display_data
        self.display_extent = extent
        self.current_clim = None  # Reset clim to force re-normalization
        self.local_mean_std = None  # Reset local stats for new data
        self.normalized_display_data = None  # Clear normalized data
        
        if extent:
            time_start, time_end, freq_start, freq_end = extent
            time_width = time_end - time_start
            freq_height = freq_end - freq_start
            
            x_scale = time_width / display_data.shape[1]
            y_scale = freq_height / display_data.shape[0]
            
            transform = scene.STTransform(
                scale=(x_scale, y_scale),
                translate=(time_start, freq_start)
            )
            self.image_visual.transform = transform
            
            if not preserve_view:
                self.set_data_bounds(time_start, time_end, freq_start, freq_end)
                self.view.camera.rect = (time_start, freq_start, time_width, freq_height)
        
        # Restore camera position if preserving view
        if preserve_view and saved_camera_rect:
            self.view.camera.rect = saved_camera_rect
        
        # Force immediate update instead of waiting for timer
        self._update_dynamic_clim_now()
        self.update()

    def schedule_normalization_update(self):
        """Schedule debounced normalization update."""
        self.normalization_pending = True
        self.normalization_timer.start(150)
    
    def _do_debounced_normalization(self):
        """Perform normalization update (called by timer)."""
        if self.normalization_pending:
            self.normalization_pending = False
            self._update_dynamic_clim_now()
    
    def schedule_auto_recompute(self):
        """Schedule auto-recompute after zoom stops."""
        if self.auto_recompute_enabled and self.on_auto_recompute_callback:
            self.recompute_timer.start(500)  # 500ms delay
    
    def _do_auto_recompute(self):
        """Trigger recomputation at current zoom level."""
        if self.on_auto_recompute_callback:
            visible = self.get_visible_region()
            if visible:
                self.on_auto_recompute_callback(visible)
    
    def update_dynamic_clim(self):
        """Debounced normalization update."""
        self.schedule_normalization_update()
    
    def _update_dynamic_clim_now(self):
        """Immediately adjust color scaling based on visible region."""
        if self.raw_display_data is None:
            return
        
        clim_min, clim_max = None, None
        
        rect = getattr(self.view.camera, 'rect', None)
        if rect is not None and self.data_bounds is not None:
            time_min, time_max, freq_min, freq_max = self.data_bounds
            time_span = max(time_max - time_min, 1e-9)
            freq_span = max(freq_max - freq_min, 1e-9)
            
            x0 = max(float(rect.left), time_min)
            x1 = min(float(rect.right), time_max)
            y0 = max(float(rect.bottom), freq_min)
            y1 = min(float(rect.top), freq_max)
            
            if x1 > x0 and y1 > y0:
                num_rows, num_cols = self.raw_display_data.shape
                
                col_start = int((x0 - time_min) / time_span * num_cols)
                col_end = int((x1 - time_min) / time_span * num_cols)
                row_start = int((y0 - freq_min) / freq_span * num_rows)
                row_end = int((y1 - freq_min) / freq_span * num_rows)
                
                col_start = max(0, min(num_cols - 1, col_start))
                col_end = max(col_start + 1, min(num_cols, col_end))
                row_start = max(0, min(num_rows - 1, row_start))
                row_end = max(row_start + 1, min(num_rows, row_end))
                
                visible_patch = self.raw_display_data[row_start:row_end, col_start:col_end]
                finite_vals = visible_patch[np.isfinite(visible_patch)]
                if finite_vals.size > 10:
                    if self.normalization_mode == 'std':
                        # For STD normalization, compute mean and std from visible region
                        local_mean = float(np.mean(finite_vals))
                        local_std = max(float(np.std(finite_vals)), 1e-6)
                        self.local_mean_std = (local_mean, local_std)
                        # Still compute clim for fallback, but use local mean/std for normalization
                        clim_min = float(np.percentile(finite_vals, 2))
                        clim_max = float(np.percentile(finite_vals, 98))
                    else:
                        # Min-Max normalization
                        clim_min = float(np.percentile(finite_vals, 2))
                        clim_max = float(np.percentile(finite_vals, 98))
        
        # Fallback
        if clim_min is None or clim_max is None:
            if self.global_percentiles:
                clim_min, clim_max = self.global_percentiles
            elif self.global_data_range:
                clim_min, clim_max = self.global_data_range
            else:
                clim_min = float(np.nanmin(self.raw_display_data))
                clim_max = float(np.nanmax(self.raw_display_data))
        
        # For STD mode, if we don't have local mean/std, use global
        if self.normalization_mode == 'std' and self.local_mean_std is None:
            if self.global_mean_std:
                self.local_mean_std = self.global_mean_std
            else:
                # Compute from all data as fallback
                finite_all = self.raw_display_data[np.isfinite(self.raw_display_data)]
                if finite_all.size > 10:
                    mean = float(np.mean(finite_all))
                    std = max(float(np.std(finite_all)), 1e-6)
                    self.local_mean_std = (mean, std)
                else:
                    # Last resort fallback
                    self.local_mean_std = (clim_min, (clim_max - clim_min) / 4.0)
        
        if clim_max - clim_min < 1e-6:
            clim_max = clim_min + 1.0
        
        # Apply dB range if set (only for minmax mode)
        if self.db_range and self.normalization_mode == 'minmax':
            db_min, db_max = self.db_range
            if db_max > db_min:
                clim_min = max(clim_min, db_min)
                clim_max = min(clim_max, db_max)
        
        new_clim = (float(clim_min), float(clim_max))
        
        if self.current_clim is None or abs(new_clim[0] - self.current_clim[0]) > 0.01 or abs(new_clim[1] - self.current_clim[1]) > 0.01:
            self.current_clim = new_clim
            self._apply_normalized_data(new_clim[0], new_clim[1])
    
    def _apply_normalized_data(self, clim_min: float, clim_max: float):
        """Normalize and display data."""
        if self.raw_display_data is None:
            logger.warning("_apply_normalized_data: No raw_display_data!")
            return

        logger.info(f"Applying normalized data: clim=[{clim_min:.1f}, {clim_max:.1f}], shape={self.raw_display_data.shape}")
        
        if self.normalization_mode == 'std' and self.local_mean_std is not None:
            # STD-based normalization using local (visible region) mean/std
            # This makes it adaptive to zoom level, just like minmax
            mean, std = self.local_mean_std
            std = max(std, 1e-6)  # Avoid division by zero
            
            # Compute z-scores: (value - mean) / (std * scale)
            # Maps approximately [-scale, +scale] sigma range
            z_scores = (self.raw_display_data - mean) / (std * self.std_scale)
            
            # Map z-score to [0, 1]: z_score of -scale maps to 0, +scale maps to 1
            normalized = (z_scores + 1.0) * 0.5
            normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
        else:
            # Min-Max normalization (default)
            clim_range = max(clim_max - clim_min, 1e-6)
            normalized = (self.raw_display_data - clim_min) / clim_range
            normalized = np.clip(normalized, 0.0, 1.0).astype(np.float32)
        
        self.normalized_display_data = normalized
        
        # Preserve transform
        current_transform = self.image_visual.transform
        if current_transform is None and self.display_extent is not None:
            time_start, time_end, freq_start, freq_end = self.display_extent
            time_width = max(time_end - time_start, 1e-9)
            freq_height = max(freq_end - freq_start, 1e-9)
            current_transform = scene.STTransform(
                scale=(time_width / normalized.shape[1], freq_height / normalized.shape[0]),
                translate=(time_start, freq_start)
            )
        
        logger.info(f"Setting image_visual data: shape={self.normalized_display_data.shape}")
        self.image_visual.set_data(self.normalized_display_data)
        self.image_visual.clim = (0.0, 1.0)
        if current_transform is not None:
            self.image_visual.transform = current_transform

        logger.info("Calling canvas update()")
        self.update()
    
    def set_normalization_mode(self, mode: str, std_scale: float = 2.5):
        """Set normalization mode.
        
        Args:
            mode: 'minmax' or 'std'
            std_scale: Scale factor for STD normalization (typically 2.0-3.0)
        """
        if mode not in ('minmax', 'std'):
            logger.warning(f"Invalid normalization mode: {mode}, using 'minmax'")
            mode = 'minmax'
        
        self.normalization_mode = mode
        self.std_scale = max(0.5, min(5.0, std_scale))  # Clamp to reasonable range
        
        # Re-apply normalization with new mode
        if self.current_clim is not None:
            self._apply_normalized_data(self.current_clim[0], self.current_clim[1])
        
        logger.debug(f"Normalization mode changed to: {mode} (std_scale={self.std_scale})")
    
    def set_crosshair(self, enabled: bool, pos: tuple = None):
        """Enable/disable crosshair display."""
        self.crosshair_enabled = enabled
        
        if enabled and pos:
            x, y = pos
            rect = self.view.camera.rect
            x_range = (rect.left, rect.right)
            y_range = (rect.bottom, rect.top)
            
            self.crosshair_v.set_data(np.array([[x, y_range[0]], [x, y_range[1]]]))
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

    def update_mode_indicator(self):
        """Update the mode indicator overlay based on current mode state.

        Shows the current mode (annotation, measurement, curve, event) in the
        top-left corner with appropriate color coding.
        """
        # Determine current mode and color
        mode_name = None
        mode_color = 'white'

        if self.curve_mode:
            mode_name = "CURVE"
            mode_color = (0, 1, 1, 1)  # Cyan
        elif self.event_mode:
            mode_name = "EVENT"
            mode_color = (1, 0.5, 0, 1)  # Orange
        elif self.annotation_mode:
            mode_name = "ANNOTATION"
            mode_color = (1, 0.3, 0.3, 1)  # Red
        elif self.measurement_mode:
            mode_name = "MEASURE"
            mode_color = (1, 0.8, 0, 1)  # Yellow

        # Only update if mode changed (for efficiency)
        if mode_name == self._current_mode_name:
            return

        self._current_mode_name = mode_name

        if mode_name:
            self.mode_indicator.text = f"[ {mode_name} ]"
            self.mode_indicator.color = mode_color
            self.mode_indicator.pos = (10, 10)
            self.mode_indicator.visible = True
        else:
            self.mode_indicator.text = ''
            self.mode_indicator.visible = False

        self.update()

    def set_interpolation(self, mode: str):
        """Set image interpolation mode."""
        vispy_mode_map = {'nearest': 'nearest', 'bilinear': 'linear', 'bicubic': 'cubic'}
        try:
            vispy_mode = vispy_mode_map.get(mode, 'linear')
            self.image_visual.interpolation = vispy_mode
            self.current_interpolation = mode
            self.update()
        except Exception:
            pass
    
    # ==================== Curve Drawing (Doppler) ====================

    def set_curve_mode(self, enabled: bool):
        """Enable or disable curve drawing mode."""
        was_enabled = self.curve_mode
        self.curve_mode = enabled

        if enabled and not was_enabled:
            # Entering curve mode - disable other modes
            self.annotation_mode = False
            self.measurement_mode = False
            # Show visual feedback
            self.update_text_readout("CURVE MODE: Click to add points (need 2 min) | Right-click to clear | Enter to finish", (10, 60))
            logger.info("Curve mode: ON (Click to add points, Right-click to clear, Enter to finish)")
        elif not enabled and was_enabled:
            self.update_text_readout("", (10, 60))
            logger.info("Curve mode: OFF")

        # Update mode indicator and canvas
        self.update_mode_indicator()
        self.update()
    
    def set_curve_callback(self, callback):
        """Set callback for when curve drawing is completed (Enter pressed)."""
        self._on_curve_completed_callback = callback
        
    def add_curve_point(self, t: float, f: float):
        """Add a point to the current curve."""
        self.curve_points.append((t, f))
        # Sort points by time
        self.curve_points.sort(key=lambda p: p[0])
        logger.info(f"Added curve point: t={t:.3f}s, f={f:.1f}Hz, total points: {len(self.curve_points)}")
        self._update_curve_visuals()
        
        # Update status text
        n_points = len(self.curve_points)
        min_points = 2
        status = f"CURVE MODE: {n_points} points"
        if n_points < min_points:
            status += f" (need {min_points - n_points} more)"
        status += " | Click to add | Right-click to clear | Enter to finish"
        self.update_text_readout(status, (10, 60))
        
        if self._on_curve_updated_callback:
            self._on_curve_updated_callback(self.curve_points)
            
    def set_curve_points(self, points: list):
        """Set the curve points externally (e.g. loading from annotation)."""
        self.curve_points = list(points)
        # Sort by time
        self.curve_points.sort(key=lambda p: p[0])
        self._update_curve_visuals()
        
    def clear_curve(self):
        """Clear the current curve."""
        self.curve_points = []
        self._update_curve_visuals()
        if self._on_curve_updated_callback:
            self._on_curve_updated_callback([])
            
    def _update_curve_visuals(self):
        """Update the VisPy visuals for the curve with monotone cubic interpolation (PCHIP)."""
        if not self.curve_points:
            self.curve_visual.visible = False
            self.curve_markers.visible = False
            self.update()
            return

        points = np.array(self.curve_points)
        logger.info(f"Updating curve visuals with {len(points)} points")

        # If we have enough points, use monotone cubic interpolation (PCHIP)
        if len(points) >= 2:
            try:
                from scipy.interpolate import PchipInterpolator

                # Sort by time
                sorted_indices = np.argsort(points[:, 0])
                sorted_points = points[sorted_indices]

                times = sorted_points[:, 0]
                freqs = sorted_points[:, 1]

                # Check for duplicate times
                if len(np.unique(times)) < len(times):
                    logger.warning("Duplicate time values detected, removing duplicates")
                    unique_indices = np.unique(times, return_index=True)[1]
                    times = times[unique_indices]
                    freqs = freqs[unique_indices]

                if len(times) >= 2:
                    # Create PCHIP interpolator (monotone cubic - no oscillations)
                    interpolator = PchipInterpolator(times, freqs)

                    # Generate smooth curve
                    t_min, t_max = times[0], times[-1]
                    t_smooth = np.linspace(t_min, t_max, max(100, len(points) * 10))
                    f_smooth = interpolator(t_smooth)

                    smooth_curve = np.column_stack((t_smooth, f_smooth))

                    # Update Line with smooth curve - color set in set_data
                    self.curve_visual.set_data(pos=smooth_curve, color='yellow', width=2.5)
                    logger.debug(f"PCHIP interpolation: {len(points)} points -> {len(t_smooth)} smooth points")
                else:
                    # Fallback to linear
                    self.curve_visual.set_data(pos=points, color='cyan', width=2.0)

            except ImportError:
                logger.debug("scipy.interpolate.PchipInterpolator not available, using linear")
                # Fallback to linear
                self.curve_visual.set_data(pos=points, color='cyan', width=2.0)
            except Exception as e:
                logger.debug(f"PCHIP interpolation failed: {e}, using linear")
                # Fallback to linear
                self.curve_visual.set_data(pos=points, color='cyan', width=2.0)
        else:
            # Not enough points for interpolation, use linear
            self.curve_visual.set_data(pos=points, color='cyan', width=2.0)
            
        self.curve_visual.visible = True
        
        # Update Markers (always show the actual clicked points)
        self.curve_markers.set_data(
            pos=points,
            face_color='cyan',
            edge_color='white',
            size=15,  # Larger size for visibility
            edge_width=3,
            symbol='disc'  # Explicit symbol
        )
        self.curve_markers.visible = True
        self.curve_markers.order = 300  # Very high order to ensure on top
        logger.info(f"Curve markers set to visible with {len(points)} points at order 300")
        self.update()

    # ==================== Preview Curves (for DSP detection) ====================

    def draw_preview_curve(self, points: np.ndarray, color: str = 'yellow'):
        """Draw a preview curve on the spectrogram.

        Args:
            points: Nx2 array of (time, freq) points
            color: Color for the curve
        """
        if not hasattr(self, '_preview_curves'):
            self._preview_curves = []

        if len(points) < 2:
            return

        # Create a new line visual for preview
        preview_line = scene.visuals.Line(parent=self.view.scene, method='gl')
        preview_line.set_data(pos=points, color=color, width=3.0)
        preview_line.visible = True
        preview_line.order = 160  # Above annotations
        preview_line.set_gl_state('translucent', depth_test=False)

        self._preview_curves.append(preview_line)
        self.update()

    def clear_preview_curves(self):
        """Clear all preview curves from the spectrogram."""
        if hasattr(self, '_preview_curves'):
            for curve in self._preview_curves:
                curve.parent = None  # Remove from scene
            self._preview_curves = []
            self.update()

    # ==================== Event Tagging Mode (Vertical Lines) ====================

    def set_event_mode(self, enabled: bool):
        """Enable or disable event tagging mode.

        In event mode, clicking places vertical lines to mark event boundaries.
        First click places line 1, second click places line 2 and triggers callback.
        Lines stay visible until mode is exited.

        Args:
            enabled: True to enable event mode
        """
        was_enabled = self.event_mode
        self.event_mode = enabled

        if enabled and not was_enabled:
            # Entering event mode - disable other modes
            self.annotation_mode = False
            self.measurement_mode = False
            self.curve_mode = False
            # Show existing lines if any
            self._set_event_lines_visible(True)
            if self.event_t1 is None:
                self.update_text_readout("EVENT MODE: Click to place start line", (10, 60))
            elif self.event_t2 is None:
                self.update_text_readout(f"EVENT MODE: Start at {self.event_t1:.3f}s | Click for end line | ESC to clear", (10, 60))
            else:
                t_start = min(self.event_t1, self.event_t2)
                t_end = max(self.event_t1, self.event_t2)
                self.update_text_readout(f"EVENT: {t_start:.3f}s - {t_end:.3f}s | Click to start new event", (10, 60))
            logger.info("Event mode: ON (Click to place vertical lines)")
        elif not enabled and was_enabled:
            # Exiting event mode - hide lines but DON'T delete them
            self._set_event_lines_visible(False)
            self.update_text_readout("", (10, 60))
            logger.info("Event mode: OFF")

        # Update mode indicator and canvas
        self.update_mode_indicator()
        self.update()

    def set_event_created_callback(self, callback):
        """Set callback for when event region is marked (both lines placed).

        Args:
            callback: Function(t_start: float, t_end: float) -> None
        """
        self._on_event_created_callback = callback

    def set_event_mode_toggled_callback(self, callback):
        """Set callback for when event mode is toggled via E key.

        Args:
            callback: Function(enabled: bool) -> None
        """
        self._on_event_mode_toggled_callback = callback

    def _add_event_marker(self, time: float):
        """Add a vertical line at the given time position.

        Args:
            time: Time position in seconds
        """
        if self.event_t1 is None:
            # First click - place line 1
            self.event_t1 = time
            self.event_line_1 = self._create_event_vertical_line(time)
            self.update_text_readout(f"EVENT MODE: Start at {time:.3f}s | Click for end line | ESC to clear", (10, 60))
            logger.info(f"Event marker 1 placed at t={time:.3f}s")

        elif self.event_t2 is None:
            # Second click - place line 2 and trigger callback
            self.event_t2 = time
            self.event_line_2 = self._create_event_vertical_line(time)

            # Order times
            t_start = min(self.event_t1, self.event_t2)
            t_end = max(self.event_t1, self.event_t2)

            duration = t_end - t_start
            self.update_text_readout(f"EVENT: {t_start:.3f}s - {t_end:.3f}s ({duration:.3f}s)", (10, 60))
            logger.info(f"Event marker 2 placed at t={time:.3f}s, event duration: {duration:.3f}s")

            # Trigger callback
            if self._on_event_created_callback:
                self._on_event_created_callback(t_start, t_end)

        else:
            # Both lines already placed - user clicking to start new event
            # Clear old markers and place first line of new event
            self.clear_event_markers()
            self.event_t1 = time
            self.event_line_1 = self._create_event_vertical_line(time)
            self.update_text_readout(f"EVENT MODE: Start at {time:.3f}s | Click for end line | ESC to clear", (10, 60))
            logger.info(f"New event started, marker 1 placed at t={time:.3f}s")

    def _create_event_vertical_line(self, time: float) -> scene.visuals.Line:
        """Create a vertical line visual at the given time.

        Args:
            time: Time position for the line

        Returns:
            The created Line visual
        """
        # Get current frequency range from data bounds or camera
        if self.data_bounds:
            _, _, f_min, f_max = self.data_bounds
        else:
            rect = self.view.camera.rect
            if rect:
                f_min = rect.bottom
                f_max = rect.bottom + rect.height
            else:
                f_min, f_max = 0, 22050

        # Create line from bottom to top of frequency range
        line_data = np.array([[time, f_min], [time, f_max]], dtype=np.float32)

        line = scene.visuals.Line(
            pos=line_data,
            color=(0.0, 1.0, 0.0, 0.9),  # Bright green
            width=2.5,
            parent=self.view.scene
        )
        line.order = 160  # Above spectrogram, visible
        line.set_gl_state('translucent', depth_test=False)

        self.update()
        return line

    def clear_event_markers(self):
        """Clear current event markers (both vertical lines)."""
        if self.event_line_1 is not None:
            self.event_line_1.parent = None
            self.event_line_1 = None
        if self.event_line_2 is not None:
            self.event_line_2.parent = None
            self.event_line_2 = None

        self.event_t1 = None
        self.event_t2 = None

        if self.event_mode:
            self.update_text_readout("EVENT MODE: Click to place start line", (10, 60))

        self.update()
        logger.debug("Event markers cleared")

    def update_event_lines_frequency_range(self):
        """Update event lines to span current frequency range.

        Call this after zoom/pan to keep lines spanning full visible height.
        """
        if self.data_bounds:
            _, _, f_min, f_max = self.data_bounds
        else:
            return

        if self.event_line_1 is not None and self.event_t1 is not None:
            line_data = np.array([[self.event_t1, f_min], [self.event_t1, f_max]], dtype=np.float32)
            self.event_line_1.set_data(pos=line_data)

        if self.event_line_2 is not None and self.event_t2 is not None:
            line_data = np.array([[self.event_t2, f_min], [self.event_t2, f_max]], dtype=np.float32)
            self.event_line_2.set_data(pos=line_data)

    def _set_event_lines_visible(self, visible: bool):
        """Show or hide event lines without deleting them.

        Args:
            visible: True to show lines, False to hide
        """
        if self.event_line_1 is not None:
            self.event_line_1.visible = visible
        if self.event_line_2 is not None:
            self.event_line_2.visible = visible
        self.update()

    # ==================== Measurement Tool ====================
    
    def set_measurement_callback(self, callback):
        """Set callback for measurement mode changes."""
        self._measurement_mode_callback = callback
        
    def set_measurement_completed_callback(self, callback):
        """Set callback for when a measurement is completed."""
        self._on_measurement_callback = callback
    
    def toggle_measurement_mode(self):
        """Toggle measurement mode on/off."""
        self.measurement_mode = not self.measurement_mode
        if not self.measurement_mode:
            self.clear_measurement()
        logger.info(f"Measurement mode: {'ON' if self.measurement_mode else 'OFF'}")

        # Update mode indicator
        self.update_mode_indicator()

        # Notify callback
        if self._measurement_mode_callback:
            self._measurement_mode_callback(self.measurement_mode)

        return self.measurement_mode
    
    def clear_measurement(self):
        """Clear current measurement sequence."""
        self.measurement_start = None
        self.measurement_end = None
        self.measurement_all_points = []
        self.measurement_line.visible = False
        self.measurement_markers.visible = False
        self.measurement_start_marker.visible = False
        self.measurement_end_marker.visible = False
        self.measurement_text.visible = False
        self.update()
    
    def set_measurement_point(self, time_pos: float, freq_pos: float):
        """Set a measurement point - supports continuous measurements.
        
        First click = start point
        Second click = end point (completes first measurement)
        Third click = new end point (measurement from previous end to this point)
        And so on...
        
        All points and lines remain visible throughout the sequence.
        """
        # Add point to the sequence
        self.measurement_all_points.append((time_pos, freq_pos))
        
        if len(self.measurement_all_points) == 1:
            # First click - set start point
            self.measurement_start = (time_pos, freq_pos)
            logger.info(f"Measurement start: time={time_pos:.3f}s, freq={freq_pos:.1f}Hz")
        else:
            # Subsequent clicks - complete measurement segment
            prev_point = self.measurement_all_points[-2]
            self.measurement_start = prev_point
            self.measurement_end = (time_pos, freq_pos)
            
            logger.info(f"Measurement: ({prev_point[0]:.3f}s, {prev_point[1]:.1f}Hz) -> ({time_pos:.3f}s, {freq_pos:.1f}Hz)")
            
            # Notify callback for this segment
            if hasattr(self, '_on_measurement_callback') and self._on_measurement_callback:
                self._on_measurement_callback(prev_point[0], prev_point[1], time_pos, freq_pos)
        
        # Update all visuals to show all points and lines
        self._update_all_measurement_visuals()
        self.update()
    
    def _update_measurement_marker(self, marker, time_pos, freq_pos, color):
        """Update a measurement marker position."""
        marker.set_data(
            pos=np.array([[time_pos, freq_pos]]),
            face_color=color,
            edge_color='white',
            size=12,
            edge_width=2
        )
    
    def _update_all_measurement_visuals(self):
        """Update all measurement visuals to show all points and lines."""
        if not self.measurement_all_points:
            return
        
        points = np.array(self.measurement_all_points)
        
        # Draw all points with colors: green for first, yellow for middle, red for last
        n_points = len(points)
        colors = []
        for i in range(n_points):
            if i == 0:
                colors.append([0, 1, 0, 1])  # Green for first
            elif i == n_points - 1:
                colors.append([1, 0, 0, 1])  # Red for last
            else:
                colors.append([1, 1, 0, 1])  # Yellow for middle
        
        self.measurement_markers.set_data(
            pos=points,
            face_color=np.array(colors),
            edge_color='white',
            size=12,
            edge_width=2
        )
        self.measurement_markers.visible = True
        self.measurement_markers.order = 200
        
        # Draw lines connecting all points
        if n_points >= 2:
            self.measurement_line.set_data(
                pos=points,
                color='yellow',
                width=3.0,
                connect='strip'  # Connect all points in sequence
            )
            self.measurement_line.visible = True
            self.measurement_line.order = 180
        
        # Hide legacy markers
        self.measurement_start_marker.visible = False
        self.measurement_end_marker.visible = False
    
    def _update_measurement_display(self):
        """Update the measurement line and text display - legacy method."""
        # Now handled by _update_all_measurement_visuals
        self._update_all_measurement_visuals()
    
    def get_visible_region(self) -> Optional[ViewRegion]:
        """Get currently visible region."""
        if self.data_bounds is None:
            return None
        
        try:
            rect = self.view.camera.rect
            if rect is None:
                return None
            
            canvas_size = self.size
            return ViewRegion(
                time_start=float(rect.left),
                time_end=float(rect.right),
                freq_start=float(rect.bottom),
                freq_end=float(rect.top),
                canvas_width=int(canvas_size[0]),
                canvas_height=int(canvas_size[1])
            )
        except Exception:
            return None
    
    def get_zoom_level(self) -> tuple:
        """Get current zoom level as (time_zoom, freq_zoom)."""
        if self.data_bounds is None:
            return (1.0, 1.0)
        
        try:
            rect = self.view.camera.rect
            if rect is None:
                return (1.0, 1.0)
            
            time_min, time_max, freq_min, freq_max = self.data_bounds
            total_time = time_max - time_min
            total_freq = freq_max - freq_min
            
            visible_time = rect.width
            visible_freq = rect.height
            
            time_zoom = total_time / max(visible_time, 1e-9)
            freq_zoom = total_freq / max(visible_freq, 1e-9)
            
            return (time_zoom, freq_zoom)
        except Exception:
            return (1.0, 1.0)
    
    def notify_zoom_changed(self):
        """Notify callback about zoom level change and update axis labels."""
        if self.on_zoom_changed_callback:
            time_zoom, freq_zoom = self.get_zoom_level()
            self.on_zoom_changed_callback(time_zoom, freq_zoom)
        
        # Update axis labels if axes exist
        self._update_axis_labels()
    
    def _update_axis_labels(self):
        """Update axis tick labels with formatted values (mm:ss, kHz)."""
        try:
            rect = self.view.camera.rect
            if rect is None:
                return
            
            # The AxisWidget automatically updates ticks based on the linked view
            # But we can customize the axis labels here if needed
            t_min, t_max = float(rect.left), float(rect.right)
            f_min, f_max = float(rect.bottom), float(rect.top)
            
            # Update axis labels with appropriate formatting
            if hasattr(self, 'x_axis') and self.x_axis:
                # Format X axis label based on time range
                time_span = t_max - t_min
                if time_span < 60:
                    self.x_axis.axis.axis_label = 'Time (s)'
                else:
                    self.x_axis.axis.axis_label = 'Time (mm:ss)'
            
            if hasattr(self, 'y_axis') and self.y_axis:
                # Format Y axis label based on frequency range
                if f_max >= 1000:
                    self.y_axis.axis.axis_label = 'Frequency (kHz)'
                else:
                    self.y_axis.axis.axis_label = 'Frequency (Hz)'
                    
        except Exception as e:
            logger.debug(f"Error updating axis labels: {e}")
    
    def set_zoom_recompute_callback(self, callback):
        """Set callback for zoom-triggered recomputation."""
        self.on_zoom_recompute_callback = callback
    
    def clear(self):
        """Clear the display."""
        self.raw_display_data = None
        self.normalized_display_data = None
        self.image_visual.set_data(np.zeros((2, 2), dtype=np.float32))
        self.tiled_renderer.clear_tiles()
        self.update()
    
    # ==================== Annotation Drawing Methods ====================
    
    def set_annotation_renderer(self, renderer):
        """Set the annotation renderer for drawing rectangles."""
        self._annotation_renderer = renderer
    
    def set_annotation_mode(self, enabled: bool):
        """Enable or disable annotation drawing mode.

        Args:
            enabled: True to enable annotation mode, False to disable
        """
        self.annotation_mode = enabled
        if not enabled:
            # Clean up any in-progress drawing
            self.annotation_drawing = False
            self.annotation_start = None
            self.annotation_end = None
            if self._annotation_renderer:
                self._annotation_renderer.hide_temp_rectangle()
        logger.info(f"Annotation mode: {'ON' if enabled else 'OFF'}")
        self.update_mode_indicator()
    
    def set_annotation_callbacks(self, on_created=None, on_clicked=None, on_context_menu=None):
        """Set callbacks for annotation events.
        
        Args:
            on_created: Callback(time_start, time_end, freq_min, freq_max) called when annotation is created
            on_clicked: Callback(time, freq) -> annotation or None, called when clicking on canvas
            on_context_menu: Callback(time, freq) -> void, called on right-click
        """
        self._on_annotation_created_callback = on_created
        self._on_annotation_clicked_callback = on_clicked
        self._on_annotation_context_menu_callback = on_context_menu
    
    def _screen_to_world(self, screen_pos) -> Optional[Tuple[float, float]]:
        """Convert screen coordinates to world coordinates (time, frequency).
        
        This method properly handles the PanZoomCamera and grid layout
        to convert mouse screen pixels to spectrogram time/frequency values.
        
        Args:
            screen_pos: (x, y) screen coordinates from mouse event
        
        Returns:
            (time, frequency) tuple, or None if conversion fails
        """
        try:
            if screen_pos is None:
                return None
            
            screen_x = float(screen_pos[0])
            screen_y = float(screen_pos[1])
            
            # Get the camera rect (visible world coordinates)
            rect = self.view.camera.rect
            if rect is None:
                return None
            
            # Get the ViewBox size in its own coordinate system
            view_size = self.view.size
            if view_size[0] <= 0 or view_size[1] <= 0:
                return None
            
            # Get transform from ViewBox to canvas (screen)
            try:
                view_to_canvas = self.view.node_transform(self)
                
                # Map ViewBox corners to canvas coordinates
                # ViewBox internal coords: (0,0) to (width, height)
                corner_00 = view_to_canvas.map((0, 0, 0, 1))
                corner_11 = view_to_canvas.map((view_size[0], view_size[1], 0, 1))
                
                # Extract 2D coordinates
                vb_left = min(corner_00[0], corner_11[0])
                vb_right = max(corner_00[0], corner_11[0])
                vb_bottom = min(corner_00[1], corner_11[1])
                vb_top = max(corner_00[1], corner_11[1])
                
                vb_width = max(vb_right - vb_left, 1.0)
                vb_height = max(vb_top - vb_bottom, 1.0)
                
                # Check if click is within ViewBox bounds
                if screen_x < vb_left or screen_x > vb_right:
                    return None
                if screen_y < vb_bottom or screen_y > vb_top:
                    return None
                
                # Normalize position within ViewBox (0.0 to 1.0)
                norm_x = (screen_x - vb_left) / vb_width
                norm_y = (screen_y - vb_bottom) / vb_height
                
                # CRITICAL: In screen coordinates, Y=0 is at top, increasing downward
                # In world coordinates (spectrogram), Y=0 is at bottom (low freq)
                # So we need to flip Y
                norm_y = 1.0 - norm_y
                
                # Map normalized coords to world coords using camera rect
                # rect.left/bottom are the world coords of the view origin
                time = float(rect.left) + norm_x * float(rect.width)
                freq = float(rect.bottom) + norm_y * float(rect.height)
                
                return (time, freq)
                
            except Exception as e:
                logger.debug(f"ViewBox transform failed: {e}, trying fallback")
            
            # Fallback method: use canvas size directly
            # This assumes ViewBox fills most of the canvas (less accurate with axes)
            try:
                canvas_size = self.size
                if canvas_size[0] <= 0 or canvas_size[1] <= 0:
                    return None
                
                # Estimate axis margins (approximate)
                left_margin = 60   # Y-axis width
                bottom_margin = 40  # X-axis height
                
                # Effective ViewBox bounds
                vb_left = left_margin
                vb_right = canvas_size[0]
                vb_bottom = 0
                vb_top = canvas_size[1] - bottom_margin
                
                vb_width = max(vb_right - vb_left, 1.0)
                vb_height = max(vb_top - vb_bottom, 1.0)
                
                # Check bounds
                if screen_x < vb_left or screen_x > vb_right:
                    return None
                if screen_y < vb_bottom or screen_y > vb_top:
                    return None
                
                norm_x = (screen_x - vb_left) / vb_width
                norm_y = 1.0 - ((screen_y - vb_bottom) / vb_height)
                
                time = float(rect.left) + norm_x * float(rect.width)
                freq = float(rect.bottom) + norm_y * float(rect.height)
                
                return (time, freq)
                
            except Exception as e:
                logger.debug(f"Fallback transform also failed: {e}")
            
            return None
        
        except Exception as e:
            logger.error(f"Error in _screen_to_world: {e}")
            return None
    
    def get_audio_coordinates(self, screen_pos, sample_rate: int = None, 
                             hop_length: int = None, fft_size: int = None) -> Optional[dict]:
        """Get full audio coordinates for a screen position.
        
        Args:
            screen_pos: (x, y) screen coordinates
            sample_rate: Audio sample rate (default: from engine)
            hop_length: Hop length (default: from engine)
            fft_size: FFT size (default: from engine)
        
        Returns:
            Dictionary with time_sec, freq_hz, frame_idx, bin_idx, or None
        """
        world = self._screen_to_world(screen_pos)
        if world is None:
            return None
        
        time_sec, freq_hz = world
        
        # Use defaults if not provided
        sr = sample_rate or 44100
        hop = hop_length or 512
        fft = fft_size or 4096
        
        # Calculate spectrogram indices
        frame_idx = int(time_sec * sr / hop)
        bin_idx = int(freq_hz * fft / sr)
        
        return {
            'time_sec': time_sec,
            'freq_hz': freq_hz,
            'frame_idx': max(0, frame_idx),
            'bin_idx': max(0, bin_idx),
            'time_formatted': self._format_time(time_sec),
            'freq_formatted': self._format_freq(freq_hz)
        }
    
    def _format_time(self, time_sec: float) -> str:
        """Format time as mm:ss.ms or ss.ms"""
        if time_sec < 0:
            return "0:00"
        if time_sec < 60:
            return f"{time_sec:.2f}s"
        minutes = int(time_sec // 60)
        seconds = time_sec % 60
        return f"{minutes}:{seconds:05.2f}"
    
    def _format_freq(self, freq_hz: float) -> str:
        """Format frequency as kHz or Hz"""
        if freq_hz >= 1000:
            return f"{freq_hz/1000:.2f} kHz"
        return f"{freq_hz:.1f} Hz"

    def get_spectrogram_data(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get the current spectrogram data with time and frequency arrays.

        Returns:
            Tuple of (spectrogram, times, freqs) or None if no data.
            - spectrogram: 2D array (freq_bins x time_frames)
            - times: 1D array of time values for each column
            - freqs: 1D array of frequency values for each row
        """
        if self.raw_display_data is None or self.display_extent is None:
            return None

        time_start, time_end, freq_start, freq_end = self.display_extent
        n_rows, n_cols = self.raw_display_data.shape

        times = np.linspace(time_start, time_end, n_cols)
        freqs = np.linspace(freq_start, freq_end, n_rows)

        return self.raw_display_data, times, freqs

    def get_spectrogram_region(self, t_start: float, t_end: float,
                                f_min: float, f_max: float) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get spectrogram data for a specific region (e.g., annotation bounds).

        Args:
            t_start, t_end: Time bounds in seconds
            f_min, f_max: Frequency bounds in Hz

        Returns:
            Tuple of (region_spectrogram, region_times, region_freqs) or None.
            - region_spectrogram: 2D array subset
            - region_times: 1D array of times for the region
            - region_freqs: 1D array of frequencies for the region
        """
        full_data = self.get_spectrogram_data()
        if full_data is None:
            return None

        spectrogram, times, freqs = full_data

        # Find indices for the region
        t_mask = (times >= t_start) & (times <= t_end)
        f_mask = (freqs >= f_min) & (freqs <= f_max)

        t_indices = np.where(t_mask)[0]
        f_indices = np.where(f_mask)[0]

        if len(t_indices) == 0 or len(f_indices) == 0:
            return None

        t_start_idx = t_indices[0]
        t_end_idx = t_indices[-1] + 1
        f_start_idx = f_indices[0]
        f_end_idx = f_indices[-1] + 1

        region = spectrogram[f_start_idx:f_end_idx, t_start_idx:t_end_idx]
        region_times = times[t_start_idx:t_end_idx]
        region_freqs = freqs[f_start_idx:f_end_idx]

        return region, region_times, region_freqs

    # ==================== Detected Tracks Visualization ====================

    def add_detected_track(self, times: list, freqs: list, track_id: int = 0,
                          color: tuple = (1.0, 0.3, 0.3, 0.8)):
        """Add a detected Doppler track to the visualization.

        Args:
            times: List of time values in seconds
            freqs: List of frequency values in Hz
            track_id: Unique identifier for this track
            color: RGBA color tuple for the track line
        """
        if len(times) < 2 or len(freqs) < 2:
            return

        # Create points array
        points = np.column_stack([times, freqs]).astype(np.float32)

        # Create a new Line visual for this track
        track_visual = scene.visuals.Line(parent=self.view.scene, method='gl')
        track_visual.set_data(pos=points, color=color, width=2.5)
        track_visual.visible = True
        track_visual.order = 160  # Above spectrogram, below annotations
        track_visual.set_gl_state('translucent', depth_test=False)

        # Store reference
        self.detected_track_visuals.append({
            'visual': track_visual,
            'track_id': track_id,
            'times': times,
            'freqs': freqs
        })

        logger.debug(f"Added detected track {track_id} with {len(times)} points")

    def clear_detected_tracks(self):
        """Remove all detected track visuals from the display."""
        for track_data in self.detected_track_visuals:
            visual = track_data['visual']
            if visual is not None:
                visual.parent = None  # Remove from scene

        self.detected_track_visuals = []
        self.update()
        logger.debug("Cleared all detected track visuals")

    def set_detected_tracks_visible(self, visible: bool):
        """Show or hide all detected track visuals.

        Args:
            visible: True to show, False to hide
        """
        for track_data in self.detected_track_visuals:
            visual = track_data['visual']
            if visual is not None:
                visual.visible = visible
        self.update()

    def get_detected_track_count(self) -> int:
        """Get the number of detected tracks currently displayed."""
        return len(self.detected_track_visuals)

