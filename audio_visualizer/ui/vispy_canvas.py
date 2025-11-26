"""
VisPy Canvas for spectrogram visualization.
Handles zoom, pan, normalization, and display.
"""
import logging
from typing import Optional
import numpy as np

from PySide6.QtCore import QTimer

try:
    from vispy import scene
    HAS_VISPY = True
except ImportError:
    scene = None
    HAS_VISPY = False

from ..core.adaptive_spectrogram import ViewRegion, ZoomLevelDetector

logger = logging.getLogger(__name__)


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
        self.measurement_start = None  # (time, freq)
        self.measurement_end = None    # (time, freq)
        self._measurement_mode_callback = None
        
        # Measurement visuals
        self.measurement_line = scene.visuals.Line(color=(1.0, 0.8, 0.0, 0.9), width=2.0, parent=self.view.scene)
        self.measurement_line.visible = False
        self.measurement_start_marker = scene.visuals.Markers(parent=self.view.scene)
        self.measurement_start_marker.visible = False
        self.measurement_end_marker = scene.visuals.Markers(parent=self.view.scene)
        self.measurement_end_marker.visible = False
        self.measurement_text = scene.visuals.Text('', color='yellow', font_size=12,
                                                    bold=True, parent=self.view.scene)
        self.measurement_text.visible = False
        
        # Text readout
        self.text_visual = scene.visuals.Text('', color='white', font_size=11,
                                              pos=(10, 30), parent=self.view.scene)
        
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
    
    def zoom_with_center(self, scale_factors, mouse_pos):
        """Zoom with center-based scaling."""
        try:
            rect = self.view.camera.rect
            if rect is None:
                return
            
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
        """Handle key press events for zoom controls and measurement."""
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
        elif event.key == 'Escape':
            # Clear measurement
            if self.measurement_mode:
                self.clear_measurement()
    
    def on_mouse_press(self, event):
        """Handle mouse press for panning or measurement."""
        if event.button == 1:
            # Check if in measurement mode
            if self.measurement_mode:
                try:
                    tr = self.view.scene.node_transform(self.view.scene)
                    if tr is not None:
                        world_pos = tr.map(event.pos)
                        self.set_measurement_point(world_pos[0], world_pos[1])
                        event.handled = True
                        return
                except Exception:
                    pass
            
            # Normal panning
            self.is_panning = True
            self.last_mouse_pos = event.pos
            event.handled = True
    
    def on_mouse_release(self, event):
        """Handle mouse release."""
        if event.button == 1:
            self.is_panning = False
            self.last_mouse_pos = None
            event.handled = True
    
    def on_mouse_move(self, event):
        """Handle mouse movement for panning and crosshair."""
        if event.pos is None:
            return
        
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
        
        elif not self.is_panning:
            try:
                tr = self.view.scene.node_transform(self.view.scene)
                if tr is not None:
                    world_pos = tr.map(event.pos)
                    time_pos = world_pos[0]
                    freq_pos = world_pos[1]
                    self.mouse_pos = (time_pos, freq_pos)
                    
                    # Update cursor position callback (for status bar)
                    if self.on_cursor_moved_callback:
                        self.on_cursor_moved_callback(time_pos, freq_pos)
                    
                    # Update crosshair if enabled
                    if self.crosshair_enabled:
                        self.set_crosshair(True, self.mouse_pos)
                        readout = f"Time: {time_pos:.3f}s, Freq: {freq_pos:.0f}Hz"
                        self.update_text_readout(readout, (10, 30))
            except Exception:
                pass
    
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
        self.current_clim = None
        
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
        
        self.update_dynamic_clim()
    
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
        
        if clim_max - clim_min < 1e-6:
            clim_max = clim_min + 1.0
        
        # Apply dB range if set
        if self.db_range:
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
            return
        
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
        
        self.image_visual.set_data(self.normalized_display_data)
        self.image_visual.clim = (0.0, 1.0)
        if current_transform is not None:
            self.image_visual.transform = current_transform
        
        self.update()
    
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
    
    # ==================== Measurement Tool ====================
    
    def set_measurement_callback(self, callback):
        """Set callback for measurement mode changes."""
        self._measurement_mode_callback = callback
    
    def toggle_measurement_mode(self):
        """Toggle measurement mode on/off."""
        self.measurement_mode = not self.measurement_mode
        if not self.measurement_mode:
            self.clear_measurement()
        logger.info(f"Measurement mode: {'ON' if self.measurement_mode else 'OFF'}")
        
        # Notify callback
        if self._measurement_mode_callback:
            self._measurement_mode_callback(self.measurement_mode)
        
        return self.measurement_mode
    
    def clear_measurement(self):
        """Clear current measurement."""
        self.measurement_start = None
        self.measurement_end = None
        self.measurement_line.visible = False
        self.measurement_start_marker.visible = False
        self.measurement_end_marker.visible = False
        self.measurement_text.visible = False
        self.update()
    
    def set_measurement_point(self, time_pos: float, freq_pos: float):
        """Set a measurement point (first click = start, second = end)."""
        if self.measurement_start is None:
            # First click - set start point
            self.measurement_start = (time_pos, freq_pos)
            self._update_measurement_marker(self.measurement_start_marker, time_pos, freq_pos, (0, 1, 0, 1))
            self.measurement_start_marker.visible = True
            logger.debug(f"Measurement start: time={time_pos:.3f}s, freq={freq_pos:.1f}Hz")
        else:
            # Second click - set end point and show measurement
            self.measurement_end = (time_pos, freq_pos)
            self._update_measurement_marker(self.measurement_end_marker, time_pos, freq_pos, (1, 0, 0, 1))
            self.measurement_end_marker.visible = True
            self._update_measurement_display()
            logger.debug(f"Measurement end: time={time_pos:.3f}s, freq={freq_pos:.1f}Hz")
        
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
    
    def _update_measurement_display(self):
        """Update the measurement line and text display."""
        if self.measurement_start is None or self.measurement_end is None:
            return
        
        t1, f1 = self.measurement_start
        t2, f2 = self.measurement_end
        
        # Draw line between points
        self.measurement_line.set_data(pos=np.array([[t1, f1], [t2, f2]]))
        self.measurement_line.visible = True
        
        # Calculate differences
        delta_time = abs(t2 - t1)
        delta_freq = abs(f2 - f1)
        
        # Format time
        if delta_time >= 60:
            time_str = f"{int(delta_time // 60)}m {delta_time % 60:.2f}s"
        else:
            time_str = f"{delta_time:.3f}s"
        
        # Format frequency
        if delta_freq >= 1000:
            freq_str = f"{delta_freq / 1000:.2f} kHz"
        else:
            freq_str = f"{delta_freq:.1f} Hz"
        
        # Position text at midpoint
        mid_t = (t1 + t2) / 2
        mid_f = (f1 + f2) / 2
        
        self.measurement_text.text = f"Δt: {time_str}  |  Δf: {freq_str}"
        self.measurement_text.pos = (mid_t, mid_f + delta_freq * 0.1)  # Slightly above line
        self.measurement_text.visible = True
    
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
        """Notify callback about zoom level change."""
        if self.on_zoom_changed_callback:
            time_zoom, freq_zoom = self.get_zoom_level()
            self.on_zoom_changed_callback(time_zoom, freq_zoom)
    
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

