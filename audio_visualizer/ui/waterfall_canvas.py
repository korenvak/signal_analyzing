"""
Waterfall Canvas for DAS multi-channel visualization.

VisPy-based canvas for displaying waterfall data:
- X-axis: Sensors (integer IDs)
- Y-axis: Time (sample index)
- Smooth pan/zoom
- Crosshair cursor
"""

import numpy as np
import logging
from typing import Optional, Tuple, Callable, Dict, Any

logger = logging.getLogger(__name__)

try:
    from vispy import scene
    from vispy.scene import visuals
    HAS_VISPY = True
except ImportError:
    HAS_VISPY = False
    logger.warning("VisPy not available - waterfall canvas disabled")

from ..core.time_formatter import TimeFormatter


if HAS_VISPY:

    class WaterfallCanvas(scene.SceneCanvas):
        """
        VisPy canvas for waterfall display.

        Displays DAS matrix data with:
        - Sensors on X-axis (integers)
        - Time on Y-axis (sample index)
        - Scrollable/zoomable view
        """

        def __init__(
            self,
            parent=None,
            keys='interactive',
            show_crosshair: bool = True,
            **kwargs
        ):
            super().__init__(keys=keys, size=(800, 600), **kwargs)

            self.unfreeze()

            # Grid layout for axes (like VisPyCanvas)
            self.grid = self.central_widget.add_grid(margin=0)
            self.grid.spacing = 0

            # Create Y-axis (Time) on the left
            self.y_axis = scene.AxisWidget(
                orientation='left',
                axis_label='Time (samples)',
                axis_font_size=6,
                axis_label_margin=30,
                tick_label_margin=5
            )
            self.y_axis.width_max = 60
            self.y_axis.width_min = 55

            # Create X-axis (Sensors) at the bottom
            self.x_axis = scene.AxisWidget(
                orientation='bottom',
                axis_label='Sensor ID',
                axis_font_size=6,
                axis_label_margin=20,
                tick_label_margin=5
            )
            self.x_axis.height_max = 40
            self.x_axis.height_min = 35

            # Create ViewBox (like VisPyCanvas)
            self.view = scene.ViewBox(camera='panzoom', parent=None)

            # Grid layout
            self.grid.add_widget(self.y_axis, row=0, col=0)
            self.grid.add_widget(self.view, row=0, col=1)
            self.grid.add_widget(self.x_axis, row=1, col=1)

            # Link axes
            self.y_axis.link_view(self.view)
            self.x_axis.link_view(self.view)

            # Configure camera
            self.view.camera.aspect = None
            self.view.camera.rect = (0, 0, 100, 100)
            self.view.camera.interactive = False  # We handle interaction ourselves
            self.view.camera.flip = (False, False, False)

            # Image visual - will be created on first set_data call
            self.image_visual = None
            self._image_created = False

            # Crosshair lines - also created on demand
            self._show_crosshair = show_crosshair
            self.crosshair_h = None
            self.crosshair_v = None

            # State
            self._data: Optional[np.ndarray] = None
            self._raw_data: Optional[np.ndarray] = None
            self._data_extent: Tuple[int, int, int, int] = (0, 100, 0, 100)
            self._time_formatter: Optional[TimeFormatter] = None
            self._colormap = 'viridis'

            # Data bounds for zoom constraints
            self.data_bounds = None

            # Panning state
            self.is_panning = False
            self.last_mouse_pos = None

            # Callbacks
            self.on_cursor_moved_callback: Optional[Callable] = None

            # Connect events
            self.events.mouse_move.connect(self._on_mouse_move)
            self.events.mouse_press.connect(self._on_mouse_press)
            self.events.mouse_release.connect(self._on_mouse_release)
            self.view.events.mouse_wheel.connect(self._on_mouse_wheel)

            self.freeze()
            logger.info("WaterfallCanvas initialized")

        def _create_image_visual(self, data: np.ndarray):
            """Create the image visual with actual data."""
            if self.image_visual is not None:
                # Remove old visual
                self.image_visual.parent = None

            self.image_visual = scene.visuals.Image(
                data,
                parent=self.view.scene,
                interpolation='nearest',
                cmap=self._colormap
            )
            self.image_visual.clim = (0, 1)
            self.image_visual.order = 0
            self._image_created = True
            logger.debug(f"Image visual created with shape {data.shape}")

        def _create_crosshairs(self):
            """Create crosshair line visuals."""
            if self.crosshair_h is not None:
                return  # Already created

            self.crosshair_h = scene.visuals.Line(
                color=(1, 1, 0, 0.7),
                width=1,
                parent=self.view.scene
            )
            self.crosshair_v = scene.visuals.Line(
                color=(1, 1, 0, 0.7),
                width=1,
                parent=self.view.scene
            )
            self.crosshair_h.visible = False
            self.crosshair_v.visible = False

        def set_data(
            self,
            data: np.ndarray,
            sensor_start: int = 0,
            sensor_end: Optional[int] = None,
            time_start: int = 0,
            time_end: Optional[int] = None,
            time_formatter: Optional[TimeFormatter] = None
        ):
            """
            Set waterfall data for display.

            Args:
                data: 2D array of shape (n_time, n_sensors)
                sensor_start: Starting sensor ID
                sensor_end: Ending sensor ID
                time_start: Starting time sample index
                time_end: Ending time sample index
                time_formatter: Optional time formatter for display
            """
            if data is None or data.size == 0:
                logger.warning("Empty data provided")
                return

            # Ensure float32 and C-contiguous
            data = np.ascontiguousarray(data, dtype=np.float32)
            n_time, n_sensors = data.shape

            if n_time < 2 or n_sensors < 2:
                logger.warning(f"Data too small: {data.shape}")
                return

            logger.info(f"Setting waterfall data: shape={data.shape}, "
                       f"input range=[{data.min():.4f}, {data.max():.4f}]")

            # Store raw data
            self._raw_data = data.copy()

            if sensor_end is None:
                sensor_end = sensor_start + n_sensors
            if time_end is None:
                time_end = time_start + n_time

            self._data_extent = (sensor_start, sensor_end, time_start, time_end)
            self._time_formatter = time_formatter

            # Process data for display (normalize to 0-1)
            processed = self._process_for_display(data)
            self._data = processed

            logger.info(f"Processed data range: [{processed.min():.4f}, {processed.max():.4f}]")

            # Set data bounds
            self.data_bounds = (sensor_start, sensor_end, time_start, time_end)

            # Create image visual if not exists, or update data
            if not self._image_created:
                self._create_image_visual(processed)
                self._create_crosshairs()
            else:
                self.image_visual.set_data(processed)
                self.image_visual.clim = (0, 1)
                self.image_visual.cmap = self._colormap

            # Calculate transform to position image correctly
            # Image shape is (n_time, n_sensors) = (height, width)
            # We want to map pixel (col, row) to world (sensor, time)
            x_scale = (sensor_end - sensor_start) / n_sensors
            y_scale = (time_end - time_start) / n_time

            transform = scene.STTransform(
                scale=(x_scale, y_scale),
                translate=(sensor_start, time_start)
            )
            self.image_visual.transform = transform

            # Set camera to show all data
            self.view.camera.rect = (sensor_start, time_start,
                                     sensor_end - sensor_start,
                                     time_end - time_start)

            self.update()
            logger.info(f"Waterfall display updated: sensors=[{sensor_start}, {sensor_end}], "
                       f"time=[{time_start}, {time_end}]")

        def _process_for_display(self, data: np.ndarray) -> np.ndarray:
            """Process raw data for display (normalize to 0-1 range)."""
            # Take absolute value (common for DAS phase derivatives)
            processed = np.abs(data).astype(np.float32)

            # Robust normalization using percentiles
            p_low = np.percentile(processed, 2)
            p_high = np.percentile(processed, 98)

            logger.debug(f"Normalization: p2={p_low:.4f}, p98={p_high:.4f}")

            if p_high - p_low > 1e-10:
                processed = (processed - p_low) / (p_high - p_low)
            else:
                processed = np.full_like(processed, 0.5)

            processed = np.clip(processed, 0.0, 1.0)

            return np.ascontiguousarray(processed, dtype=np.float32)

        def set_colormap(self, colormap: str):
            """Set colormap for display."""
            self._colormap = colormap
            if self.image_visual is not None:
                try:
                    self.image_visual.cmap = colormap
                    self.update()
                except Exception as e:
                    logger.warning(f"Failed to set colormap: {e}")

        def set_clim(self, vmin: float, vmax: float):
            """Set color limits."""
            if self.image_visual is not None:
                self.image_visual.clim = (vmin, vmax)
                self.update()

        def set_time_formatter(self, formatter: TimeFormatter):
            """Set time formatter for display."""
            self._time_formatter = formatter

        # ==================== Mouse Events ====================

        def _on_mouse_wheel(self, event):
            """Handle mouse wheel for zoom."""
            event.handled = True

            factor = 1.15 if event.delta[1] > 0 else 0.87

            rect = self.view.camera.rect
            if rect is None:
                return

            # Zoom centered
            center_x = rect.left + rect.width / 2
            center_y = rect.bottom + rect.height / 2

            new_width = rect.width / factor
            new_height = rect.height / factor

            # Constrain to data bounds
            if self.data_bounds:
                s_start, s_end, t_start, t_end = self.data_bounds
                max_width = s_end - s_start
                max_height = t_end - t_start
                new_width = max(max_width / 100, min(new_width, max_width))
                new_height = max(max_height / 100, min(new_height, max_height))

            new_x = center_x - new_width / 2
            new_y = center_y - new_height / 2

            self.view.camera.rect = (new_x, new_y, new_width, new_height)
            self.update()

        def _on_mouse_press(self, event):
            """Handle mouse press for panning."""
            if event.button == 1:
                self.is_panning = True
                self.last_mouse_pos = event.pos
                event.handled = True

        def _on_mouse_release(self, event):
            """Handle mouse release."""
            if event.button == 1:
                self.is_panning = False
                self.last_mouse_pos = None
                event.handled = True

        def _on_mouse_move(self, event):
            """Handle mouse move for panning and crosshair."""
            if event.pos is None:
                return

            # Handle panning
            if self.is_panning and self.last_mouse_pos is not None:
                delta = event.pos - self.last_mouse_pos
                rect = self.view.camera.rect
                if rect is None:
                    return

                canvas_size = self.size
                if canvas_size[0] <= 0 or canvas_size[1] <= 0:
                    return

                # Convert screen delta to world delta
                dx = -(delta[0] / canvas_size[0]) * rect.width
                dy = (delta[1] / canvas_size[1]) * rect.height

                new_left = rect.left + dx
                new_bottom = rect.bottom + dy

                # Constrain to data bounds
                if self.data_bounds:
                    s_start, s_end, t_start, t_end = self.data_bounds
                    new_left = max(s_start, min(new_left, s_end - rect.width))
                    new_bottom = max(t_start, min(new_bottom, t_end - rect.height))

                self.view.camera.rect = (new_left, new_bottom, rect.width, rect.height)
                self.last_mouse_pos = event.pos
                self.update()
                return

            # Update crosshair and cursor info
            world_pos = self._screen_to_world(event.pos)
            if world_pos is not None:
                sensor, time_idx = world_pos

                if self._show_crosshair and self._data is not None:
                    self._update_crosshair(sensor, time_idx)

                if self.on_cursor_moved_callback:
                    info = self._get_cursor_info(sensor, time_idx)
                    self.on_cursor_moved_callback(info)

        def _screen_to_world(self, screen_pos) -> Optional[Tuple[float, float]]:
            """Convert screen coordinates to world coordinates."""
            try:
                rect = self.view.camera.rect
                if rect is None:
                    return None

                # Get ViewBox bounds in screen coordinates
                view_size = self.view.size
                if view_size[0] <= 0 or view_size[1] <= 0:
                    return None

                # Get transform from view to canvas
                view_to_canvas = self.view.node_transform(self)
                corner_00 = view_to_canvas.map((0, 0, 0, 1))
                corner_11 = view_to_canvas.map((view_size[0], view_size[1], 0, 1))

                vb_left = min(corner_00[0], corner_11[0])
                vb_right = max(corner_00[0], corner_11[0])
                vb_bottom = min(corner_00[1], corner_11[1])
                vb_top = max(corner_00[1], corner_11[1])

                vb_width = max(vb_right - vb_left, 1.0)
                vb_height = max(vb_top - vb_bottom, 1.0)

                screen_x = float(screen_pos[0])
                screen_y = float(screen_pos[1])

                if screen_x < vb_left or screen_x > vb_right:
                    return None
                if screen_y < vb_bottom or screen_y > vb_top:
                    return None

                norm_x = (screen_x - vb_left) / vb_width
                norm_y = 1.0 - ((screen_y - vb_bottom) / vb_height)

                world_x = float(rect.left) + norm_x * float(rect.width)
                world_y = float(rect.bottom) + norm_y * float(rect.height)

                return (world_x, world_y)

            except Exception as e:
                logger.debug(f"Screen to world conversion failed: {e}")
                return None

        def _get_cursor_info(self, sensor: float, time_idx: float) -> Dict[str, Any]:
            """Get information at cursor position."""
            sensor_int = int(round(sensor))
            time_int = int(round(time_idx))

            info = {
                'sensor': sensor_int,
                'time_idx': time_int,
                'time_str': '',
                'value': None
            }

            if self._time_formatter:
                info['time_str'] = self._time_formatter.format_sample(time_int)
            else:
                info['time_str'] = f"{time_int}"

            if self._raw_data is not None:
                s_start, s_end, t_start, t_end = self._data_extent
                local_sensor = sensor_int - s_start
                local_time = time_int - t_start

                if 0 <= local_sensor < self._raw_data.shape[1] and \
                   0 <= local_time < self._raw_data.shape[0]:
                    info['value'] = float(self._raw_data[local_time, local_sensor])

            return info

        def _update_crosshair(self, sensor: float, time_idx: float):
            """Update crosshair position."""
            if self.crosshair_h is None or self.crosshair_v is None:
                return

            s_start, s_end, t_start, t_end = self._data_extent

            if not (s_start <= sensor <= s_end and t_start <= time_idx <= t_end):
                self.crosshair_h.visible = False
                self.crosshair_v.visible = False
                return

            self.crosshair_h.set_data(pos=np.array([
                [s_start, time_idx],
                [s_end, time_idx]
            ], dtype=np.float32))
            self.crosshair_h.visible = True

            self.crosshair_v.set_data(pos=np.array([
                [sensor, t_start],
                [sensor, t_end]
            ], dtype=np.float32))
            self.crosshair_v.visible = True

        def reset_view(self):
            """Reset camera to show all data."""
            if self.data_bounds:
                s_start, s_end, t_start, t_end = self.data_bounds
                self.view.camera.rect = (s_start, t_start,
                                         s_end - s_start,
                                         t_end - t_start)
                self.update()

        # ==================== Memory Management ====================

        def clear(self):
            """Clear the display."""
            self._data = None
            self._raw_data = None
            if self.crosshair_h is not None:
                self.crosshair_h.visible = False
            if self.crosshair_v is not None:
                self.crosshair_v.visible = False
            logger.debug("WaterfallCanvas cleared")

        def cleanup_memory(self):
            """Release memory resources."""
            self._data = None
            self._raw_data = None

            import gc
            gc.collect()

            logger.debug("WaterfallCanvas memory cleaned up")


    class WaterfallWidget:
        """Wrapper widget for WaterfallCanvas."""

        def __init__(self, parent=None):
            self.canvas = WaterfallCanvas()
            self.native = self.canvas.native

        def set_data(self, *args, **kwargs):
            self.canvas.set_data(*args, **kwargs)

        def set_colormap(self, *args, **kwargs):
            self.canvas.set_colormap(*args, **kwargs)

        def set_clim(self, *args, **kwargs):
            self.canvas.set_clim(*args, **kwargs)

        def clear(self):
            self.canvas.clear()

        def cleanup_memory(self):
            self.canvas.cleanup_memory()

else:
    class WaterfallCanvas:
        def __init__(self, *args, **kwargs):
            raise ImportError("VisPy is required for WaterfallCanvas")

    class WaterfallWidget:
        def __init__(self, *args, **kwargs):
            raise ImportError("VisPy is required for WaterfallWidget")
