"""
Waterfall Canvas for DAS multi-channel visualization.

VisPy-based canvas for displaying waterfall data:
- X-axis: Sensors (integer IDs)
- Y-axis: Time (HH:MM:SS format)
- Smooth pan/zoom with resolution switching
- Annotation support
- Crosshair cursor
"""

import numpy as np
import logging
from typing import Optional, Tuple, Callable, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    from vispy import scene, app
    from vispy.scene import visuals
    from vispy.color import Colormap
    HAS_VISPY = True
except ImportError:
    HAS_VISPY = False
    logger.warning("VisPy not available - waterfall canvas disabled")

from ..core.time_formatter import TimeFormatter
from ..core.waterfall_tile_manager import (
    WaterfallTileManager, ViewRect, TileConfig,
    get_waterfall_tile_manager
)
from ..engines.waterfall_engine import (
    WaterfallEngine, WaterfallParams, NormalizationMode,
    get_waterfall_engine
)


if HAS_VISPY:

    class WaterfallCanvas(scene.SceneCanvas):
        """
        VisPy canvas for waterfall display.

        Displays DAS matrix data with:
        - Sensors on X-axis (integers)
        - Time on Y-axis (HH:MM:SS)
        - Scrollable/zoomable view
        """

        def __init__(
            self,
            parent=None,
            keys='interactive',
            show_crosshair: bool = True,
            **kwargs
        ):
            super().__init__(keys=keys, **kwargs)

            # Create view
            self.unfreeze()
            self.view = self.central_widget.add_view()
            self.view.camera = scene.PanZoomCamera(aspect=1)
            self.view.camera.flip = (False, True, False)  # Flip Y for top-down time

            # Image visual for waterfall display
            self._image = visuals.Image(
                data=np.zeros((100, 100), dtype=np.float32),
                cmap='viridis',
                clim=(0, 1),
                parent=self.view.scene
            )

            # Crosshair
            self._show_crosshair = show_crosshair
            self._crosshair_h = visuals.Line(
                pos=np.array([[0, 0], [1, 0]]),
                color=(1, 1, 0, 0.5),
                parent=self.view.scene
            )
            self._crosshair_v = visuals.Line(
                pos=np.array([[0, 0], [0, 1]]),
                color=(1, 1, 0, 0.5),
                parent=self.view.scene
            )
            self._crosshair_h.visible = False
            self._crosshair_v.visible = False

            # Axis labels (text visuals)
            self._x_labels: list = []
            self._y_labels: list = []

            # State
            self._data: Optional[np.ndarray] = None
            self._data_extent: Tuple[int, int, int, int] = (0, 100, 0, 100)  # sensor_start, end, time_start, end
            self._time_formatter: Optional[TimeFormatter] = None
            self._colormap = 'viridis'
            self._clim = (0.0, 1.0)

            # Callbacks
            self.on_cursor_moved_callback: Optional[Callable] = None
            self.on_zoom_changed_callback: Optional[Callable] = None
            self.on_region_selected_callback: Optional[Callable] = None

            # Tile manager and engine
            self._tile_manager: Optional[WaterfallTileManager] = None
            self._engine: Optional[WaterfallEngine] = None

            # Connect events
            self.events.mouse_move.connect(self._on_mouse_move)
            self.events.mouse_wheel.connect(self._on_mouse_wheel)

            self.freeze()
            logger.info("WaterfallCanvas initialized")

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
                sensor_end: Ending sensor ID (defaults to sensor_start + n_sensors)
                time_start: Starting time sample index
                time_end: Ending time sample index (defaults to time_start + n_time)
                time_formatter: Optional time formatter for Y-axis labels
            """
            if data is None or data.size == 0:
                logger.warning("Empty data provided")
                return

            self._data = data.astype(np.float32)
            n_time, n_sensors = data.shape

            if sensor_end is None:
                sensor_end = sensor_start + n_sensors
            if time_end is None:
                time_end = time_start + n_time

            self._data_extent = (sensor_start, sensor_end, time_start, time_end)
            self._time_formatter = time_formatter

            # Process data through engine
            engine = self._engine or get_waterfall_engine()
            processed, stats = engine.process(data)

            # Update image
            self._image.set_data(processed)
            self._image.clim = self._clim

            # Set image transform to match data coordinates
            # Image covers (0, 0) to (n_sensors, n_time) in image coords
            # We want it to cover (sensor_start, time_start) to (sensor_end, time_end) in data coords
            from vispy.visuals.transforms import STTransform
            scale_x = (sensor_end - sensor_start) / n_sensors
            scale_y = (time_end - time_start) / n_time
            self._image.transform = STTransform(
                translate=(sensor_start, time_start),
                scale=(scale_x, scale_y)
            )

            # Update camera to show all data
            self.view.camera.set_range(
                x=(sensor_start, sensor_end),
                y=(time_start, time_end)
            )

            self.update()
            logger.info(f"Waterfall data set: shape={data.shape}, "
                       f"sensors=[{sensor_start}, {sensor_end}], "
                       f"time=[{time_start}, {time_end}]")

        def set_colormap(self, colormap: str):
            """Set colormap for display."""
            self._colormap = colormap
            self._image.cmap = colormap
            self.update()

        def set_clim(self, vmin: float, vmax: float):
            """Set color limits."""
            self._clim = (vmin, vmax)
            self._image.clim = self._clim
            self.update()

        def set_time_formatter(self, formatter: TimeFormatter):
            """Set time formatter for Y-axis labels."""
            self._time_formatter = formatter

        def set_tile_manager(self, manager: WaterfallTileManager):
            """Set tile manager for large data handling."""
            self._tile_manager = manager

        def set_engine(self, engine: WaterfallEngine):
            """Set waterfall processing engine."""
            self._engine = engine

        # ==================== View Helpers ====================

        def get_view_rect(self) -> ViewRect:
            """Get current view rectangle in data coordinates."""
            rect = self.view.camera.rect
            return ViewRect(
                sensor_start=int(max(0, rect.left)),
                sensor_end=int(rect.right),
                time_start=int(max(0, rect.top)),
                time_end=int(rect.bottom),
                zoom_level=self._calculate_zoom_level()
            )

        def _calculate_zoom_level(self) -> float:
            """Calculate current zoom level (1.0 = full view)."""
            rect = self.view.camera.rect
            if self._data is None:
                return 1.0

            n_time, n_sensors = self._data.shape
            view_sensors = rect.width
            view_time = rect.height

            # Zoom is ratio of data size to view size
            zoom_x = n_sensors / max(1, view_sensors)
            zoom_y = n_time / max(1, view_time)

            return min(zoom_x, zoom_y)

        def zoom_to_region(
            self,
            sensor_start: int,
            sensor_end: int,
            time_start: int,
            time_end: int
        ):
            """Zoom camera to specific region."""
            self.view.camera.set_range(
                x=(sensor_start, sensor_end),
                y=(time_start, time_end)
            )
            self.update()

        def reset_zoom(self):
            """Reset to show all data."""
            if self._data_extent:
                s_start, s_end, t_start, t_end = self._data_extent
                self.zoom_to_region(s_start, s_end, t_start, t_end)

        # ==================== Coordinate Conversion ====================

        def pixel_to_data(self, pos: Tuple[float, float]) -> Tuple[float, float]:
            """Convert pixel position to data coordinates (sensor, time)."""
            tr = self.view.scene.transform
            data_pos = tr.imap(pos)[:2]
            return data_pos[0], data_pos[1]

        def data_to_pixel(self, sensor: float, time: float) -> Tuple[float, float]:
            """Convert data coordinates to pixel position."""
            tr = self.view.scene.transform
            pixel_pos = tr.map([sensor, time, 0, 1])
            return pixel_pos[0], pixel_pos[1]

        def get_cursor_info(self, pos: Tuple[float, float]) -> Dict[str, Any]:
            """Get information at cursor position."""
            sensor, time_idx = self.pixel_to_data(pos)
            sensor = int(sensor)
            time_idx = int(time_idx)

            info = {
                'sensor': sensor,
                'time_idx': time_idx,
                'time_str': '',
                'value': None
            }

            # Format time
            if self._time_formatter:
                info['time_str'] = self._time_formatter.format_sample(time_idx)
            else:
                info['time_str'] = f"{time_idx}"

            # Get value from data
            if self._data is not None:
                s_start, s_end, t_start, t_end = self._data_extent
                local_sensor = sensor - s_start
                local_time = time_idx - t_start

                if 0 <= local_sensor < self._data.shape[1] and \
                   0 <= local_time < self._data.shape[0]:
                    info['value'] = float(self._data[local_time, local_sensor])

            return info

        # ==================== Events ====================

        def _on_mouse_move(self, event):
            """Handle mouse move for crosshair and info display."""
            if event.pos is None:
                return

            pos = event.pos
            sensor, time_idx = self.pixel_to_data(pos)

            # Update crosshair
            if self._show_crosshair:
                self._update_crosshair(sensor, time_idx)

            # Callback with cursor info
            if self.on_cursor_moved_callback:
                info = self.get_cursor_info(pos)
                self.on_cursor_moved_callback(info)

        def _update_crosshair(self, sensor: float, time_idx: float):
            """Update crosshair position."""
            if self._data is None:
                self._crosshair_h.visible = False
                self._crosshair_v.visible = False
                return

            s_start, s_end, t_start, t_end = self._data_extent

            # Horizontal line (across sensors at time position)
            self._crosshair_h.set_data(pos=np.array([
                [s_start, time_idx],
                [s_end, time_idx]
            ]))
            self._crosshair_h.visible = True

            # Vertical line (across time at sensor position)
            self._crosshair_v.set_data(pos=np.array([
                [sensor, t_start],
                [sensor, t_end]
            ]))
            self._crosshair_v.visible = True

            self.update()

        def _on_mouse_wheel(self, event):
            """Handle mouse wheel for vertical time scrolling."""
            # Default behavior is zoom, but we can add shift+wheel for time scroll
            pass

        # ==================== Cleanup ====================

        def clear(self):
            """Clear the display."""
            self._data = None
            self._image.set_data(np.zeros((10, 10), dtype=np.float32))
            self._crosshair_h.visible = False
            self._crosshair_v.visible = False
            self.update()


    class WaterfallWidget:
        """
        Wrapper widget for WaterfallCanvas with Qt integration.

        Provides a native Qt widget that can be added to layouts.
        """

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

else:
    # Fallback if VisPy not available
    class WaterfallCanvas:
        def __init__(self, *args, **kwargs):
            raise ImportError("VisPy is required for WaterfallCanvas")

    class WaterfallWidget:
        def __init__(self, *args, **kwargs):
            raise ImportError("VisPy is required for WaterfallWidget")
