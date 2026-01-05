"""
Annotation renderer for drawing annotation rectangles and ridges on VisPy canvas.
"""
import logging
from typing import Optional, Dict, List
import numpy as np

try:
    from vispy import scene
    HAS_VISPY = True
except ImportError:
    scene = None
    HAS_VISPY = False

from .annotation_data import Annotation

logger = logging.getLogger(__name__)


class AnnotationRenderer:
    """Renders annotation rectangles, ridges, and Doppler curves on a VisPy canvas."""
    
    def __init__(self, canvas_view):
        """Initialize annotation renderer.
        
        Args:
            canvas_view: The VisPy ViewBox where rectangles will be drawn
        """
        if not HAS_VISPY:
            raise RuntimeError("VisPy not available")
        
        self.view = canvas_view
        self.annotations: Dict[int, dict] = {}  # annotation_id -> {rect, ridge, doppler_curve, doppler_markers, ...}
        self.ridges: Dict[int, dict] = {}  # annotation_id -> ridge visual info
        
        # Store annotation references for visibility updates
        self._annotation_refs: Dict[int, Annotation] = {}
        
        # Temporary rectangle for drawing
        self.temp_border = None
        self.temp_fill = None
    
    def create_rectangle_visuals(self, annotation: Annotation, 
                                  is_selected: bool = False) -> Dict:
        """Create rectangle visuals (border + fill) for an annotation.
        
        Uses Rectangle visual for proper alpha blending (Raven Pro style).
        
        Args:
            annotation: The annotation to visualize
            is_selected: Whether this annotation is currently selected
        
        Returns:
            Dictionary with 'rect' visual (Rectangle has both fill and border)
        """
        if not HAS_VISPY:
            return {}
        
        # Calculate rectangle bounds
        t_min = min(annotation.t_start, annotation.t_end)
        t_max = max(annotation.t_start, annotation.t_end)
        f_min = min(annotation.f_min, annotation.f_max)
        f_max = max(annotation.f_min, annotation.f_max)
        
        # Calculate center and dimensions
        width = t_max - t_min
        height = f_max - f_min
        center_x = (t_min + t_max) / 2
        center_y = (f_min + f_max) / 2
        
        logger.debug(f"Creating annotation visual: center=({center_x:.3f}, {center_y:.0f}), size=({width:.3f}, {height:.0f})")
        
        # Choose colors based on selection state
        if is_selected:
            fill_color = (1.0, 0.4, 0.0, 0.4)    # Orange tint, 40% opacity
            border_color = (1.0, 0.4, 0.0, 1.0)  # Orange border
            border_width = 3.0
        else:
            fill_color = (1.0, 0.0, 0.0, 0.3)    # Red tint, 30% opacity
            border_color = (1.0, 0.0, 0.0, 1.0)  # Solid red border
            border_width = 2.0
        
        # Create Rectangle visual with fill and border
        rect = scene.visuals.Rectangle(
            center=(center_x, center_y),
            width=width,
            height=height,
            color=fill_color,
            border_color=border_color,
            border_width=border_width,
            parent=self.view.scene
        )
        rect.order = 100  # High order to render on top of spectrogram
        rect.set_gl_state('translucent', depth_test=False)  # Ensure transparency works and draws on top
        
        return {'rect': rect, 'border': None, 'fill': None}
    
    def create_doppler_curve_visuals(self, annotation: Annotation) -> Dict:
        """Create Doppler curve visuals (line + markers) for an annotation.
        
        Uses PCHIP (monotone cubic) interpolation for smooth curves, same as curve mode.
        
        Args:
            annotation: The annotation with Doppler curve points
        
        Returns:
            Dictionary with 'doppler_curve' and 'doppler_markers' visuals
        """
        if not HAS_VISPY or not annotation.points:
            logger.debug(f"No points for annotation {annotation.id}, skipping curve creation")
            return {}
        
        logger.info(f"Creating doppler curve for annotation {annotation.id} with {len(annotation.points)} points")
        points = np.array(annotation.points)
        
        # Apply PCHIP monotone cubic interpolation for smooth curve (same as curve mode)
        smooth_curve = points  # Default to raw points
        if len(points) >= 2:
            try:
                from scipy.interpolate import PchipInterpolator
                
                # Sort by time
                sorted_indices = np.argsort(points[:, 0])
                times = points[sorted_indices, 0]
                freqs = points[sorted_indices, 1]
                
                # Remove duplicate times (required for interpolation)
                unique_indices = np.where(np.diff(times, prepend=times[0]-1) > 0)[0]
                if len(unique_indices) > 1:
                    times = times[unique_indices]
                    freqs = freqs[unique_indices]
                
                if len(times) >= 2:
                    # Create PCHIP interpolator (monotone cubic - no oscillations)
                    interpolator = PchipInterpolator(times, freqs)
                    
                    # Generate smooth curve with many points
                    t_min, t_max = times[0], times[-1]
                    t_smooth = np.linspace(t_min, t_max, max(100, len(points) * 10))
                    f_smooth = interpolator(t_smooth)
                    
                    smooth_curve = np.column_stack((t_smooth, f_smooth))
                    logger.debug(f"PCHIP interpolation: {len(points)} points -> {len(t_smooth)} smooth points")
            except ImportError:
                logger.debug("scipy.interpolate.PchipInterpolator not available, using linear")
            except Exception as e:
                logger.debug(f"PCHIP interpolation failed: {e}, using linear")
        
        # Validate smooth_curve data - must have valid finite values
        if not np.all(np.isfinite(smooth_curve)):
            logger.warning(f"smooth_curve contains NaN/Inf values, filtering them out")
            valid_mask = np.all(np.isfinite(smooth_curve), axis=1)
            smooth_curve = smooth_curve[valid_mask]
            if len(smooth_curve) < 2:
                logger.warning("Not enough valid points for doppler curve after filtering")
                return {}
        
        # Ensure at least 2 points for a valid line
        if len(smooth_curve) < 2:
            logger.warning(f"Not enough points for doppler curve: {len(smooth_curve)}")
            return {}
        
        # Create curve line with smooth interpolated points
        curve = scene.visuals.Line(parent=self.view.scene, method='gl')
        curve.set_data(pos=smooth_curve.astype(np.float32), color='cyan', width=2.5)
        curve.order = 120  # Above rectangle but below markers
        curve.set_gl_state('translucent', depth_test=False)
        
        # Validate points data for markers
        valid_points = points[np.all(np.isfinite(points), axis=1)] if len(points) > 0 else points
        if len(valid_points) == 0:
            logger.warning("No valid points for markers")
            return {'doppler_curve': curve, 'doppler_markers': None}
        
        # Create markers for the ORIGINAL control points (not interpolated)
        markers = scene.visuals.Markers(parent=self.view.scene)
        markers.set_data(
            pos=valid_points.astype(np.float32),
            face_color=(0, 1, 1, 0.8),  # Cyan
            edge_color='white',
            size=10,
            edge_width=2
        )
        markers.order = 130  # On top of curve
        markers.set_gl_state('translucent', depth_test=False)
        
        return {'doppler_curve': curve, 'doppler_markers': markers}
    
    def add_annotation(self, annotation: Annotation, is_selected: bool = False):
        """Add an annotation rectangle (and Doppler curve if exists) to the display.
        
        Args:
            annotation: The annotation to display
            is_selected: Whether this annotation is selected
        """
        logger.debug(f"Adding annotation {annotation.id} to renderer (points={len(annotation.points) if annotation.points else 0}, "
                    f"is_visible={annotation.is_visible}, show_doppler={annotation.show_doppler_curve})")
        
        if annotation.id in self.annotations:
            self.remove_annotation(annotation.id)
        
        visuals = self.create_rectangle_visuals(annotation, is_selected)
        
        # Store annotation reference for later visibility updates
        self._annotation_refs[annotation.id] = annotation
        
        # Add Doppler curve if annotation has points
        if annotation.points:
            doppler_visuals = self.create_doppler_curve_visuals(annotation)
            visuals.update(doppler_visuals)
            logger.info(f"Added doppler curve for annotation {annotation.id}")
        
        # Apply visibility settings from annotation data
        rect_visible = annotation.is_visible
        curve_visible = annotation.is_visible and annotation.show_doppler_curve
        
        if visuals.get('rect'):
            visuals['rect'].visible = rect_visible
        if visuals.get('doppler_curve'):
            visuals['doppler_curve'].visible = curve_visible
            logger.debug(f"Curve visibility set to {curve_visible} for annotation {annotation.id}")
        if visuals.get('doppler_markers'):
            visuals['doppler_markers'].visible = curve_visible
        
        self.annotations[annotation.id] = visuals
        annotation.graphics_handle = visuals
        
        # Force canvas update
        self._update_canvas()
    
    def _remove_annotation_visuals(self, annotation_id: int):
        """Internal method to remove annotation visuals without canvas update.
        
        This is the core removal logic without triggering canvas updates,
        used by both remove_annotation() and clear_all().
        
        Args:
            annotation_id: ID of the annotation to remove
        """
        if annotation_id not in self.annotations:
            return
            
        visuals = self.annotations[annotation_id]
        
        # IMPORTANT: For Line visuals, we must set invisible and clear data BEFORE
        # removing from parent to prevent "Error drawing visual" during render queue flush
        
        # Remove Doppler curve visuals FIRST (most error-prone)
        if visuals.get('doppler_curve'):
            try:
                curve = visuals['doppler_curve']
                curve.visible = False
                # Try to set minimal data instead of zeros to see if it's more stable
                # Some VisPy versions prefer at least 2 points for Line visuals
                empty_data = np.array([[0, 0], [0.001, 0.001]], dtype=np.float32)
                curve.set_data(pos=empty_data)
                if curve.parent:
                    curve.parent = None
            except Exception as e:
                logger.debug(f"Error removing doppler_curve: {e}")
        
        if visuals.get('doppler_markers'):
            try:
                markers = visuals['doppler_markers']
                markers.visible = False
                # Use a small finite position instead of zeros
                empty_markers = np.array([[0, 0]], dtype=np.float32)
                markers.set_data(pos=empty_markers)
                if markers.parent:
                    markers.parent = None
            except Exception as e:
                logger.debug(f"Error removing doppler_markers: {e}")
        
        # Remove rectangle visuals from scene
        if visuals.get('rect'):
            try:
                visuals['rect'].visible = False
                if visuals['rect'].parent:
                    visuals['rect'].parent = None
            except Exception as e:
                logger.debug(f"Error removing rect: {e}")
            
        if visuals.get('border') and visuals['border']:
            try:
                visuals['border'].visible = False
                if visuals['border'].parent:
                    visuals['border'].parent = None
            except Exception as e:
                logger.debug(f"Error removing border: {e}")
            
        if visuals.get('fill') and visuals['fill']:
            try:
                visuals['fill'].visible = False
                if visuals['fill'].parent:
                    visuals['fill'].parent = None
            except Exception as e:
                logger.debug(f"Error removing fill: {e}")
        
        # Remove annotation reference
        if annotation_id in self._annotation_refs:
            del self._annotation_refs[annotation_id]
        
        logger.debug(f"Removed visual for annotation {annotation_id}")
    
    def remove_annotation(self, annotation_id: int):
        """Remove an annotation rectangle and Doppler curve from the display.
        
        Args:
            annotation_id: ID of the annotation to remove
        """
        if annotation_id not in self.annotations:
            return
            
        self._remove_annotation_visuals(annotation_id)
        del self.annotations[annotation_id]
        
        # Force canvas update to ensure visual removal
        self._update_canvas()
    
    def update_annotation(self, annotation: Annotation, is_selected: bool = False):
        """Update an existing annotation rectangle.

        Args:
            annotation: The annotation with updated bounds
            is_selected: Whether this annotation is selected
        """
        if annotation.id not in self.annotations:
            self.add_annotation(annotation, is_selected)
            return

        visuals = self.annotations[annotation.id]
        rect = visuals.get('rect')

        if rect:
            # Calculate new bounds
            t_min = min(annotation.t_start, annotation.t_end)
            t_max = max(annotation.t_start, annotation.t_end)
            f_min = min(annotation.f_min, annotation.f_max)
            f_max = max(annotation.f_min, annotation.f_max)

            width = t_max - t_min
            height = f_max - f_min
            center_x = (t_min + t_max) / 2
            center_y = (f_min + f_max) / 2

            # Check if geometry changed
            old_center = rect.center
            geometry_changed = (
                abs(old_center[0] - center_x) > 1e-6 or
                abs(old_center[1] - center_y) > 1e-6 or
                abs(rect.width - width) > 1e-6 or
                abs(rect.height - height) > 1e-6
            )

            if geometry_changed:
                # Geometry changed - need to recreate (VisPy limitation)
                self.remove_annotation(annotation.id)
                self.add_annotation(annotation, is_selected)
            else:
                # Only selection state changed - update colors in-place
                # Note: border_width cannot be changed after creation in VisPy
                if is_selected:
                    rect.color = (1.0, 0.4, 0.0, 0.4)
                    rect.border_color = (1.0, 0.4, 0.0, 1.0)
                else:
                    rect.color = (1.0, 0.0, 0.0, 0.3)
                    rect.border_color = (1.0, 0.0, 0.0, 1.0)

                # Update visibility
                rect.visible = annotation.is_visible

                # Update Doppler curve if needed
                if annotation.points:
                    self.update_doppler_curve(annotation)
        else:
            # No rect exists, create fresh
            self.remove_annotation(annotation.id)
            self.add_annotation(annotation, is_selected)
    
    def set_selected(self, annotation_id: Optional[int]):
        """Set which annotation is selected (highlighted).

        Args:
            annotation_id: ID of annotation to select, or None to deselect all
        """
        for ann_id, visuals in self.annotations.items():
            is_selected = (ann_id == annotation_id)

            rect = visuals.get('rect')
            if rect:
                # Update colors in-place (more efficient than recreating)
                # Note: border_width cannot be changed after creation in VisPy
                if is_selected:
                    rect.color = (1.0, 0.4, 0.0, 0.4)    # Orange tint
                    rect.border_color = (1.0, 0.4, 0.0, 1.0)  # Orange border
                else:
                    rect.color = (1.0, 0.0, 0.0, 0.3)    # Red tint
                    rect.border_color = (1.0, 0.0, 0.0, 1.0)  # Red border
    
    def clear_all(self):
        """Remove all annotation rectangles.
        
        This method freezes the canvas during removal to prevent draw errors
        when visuals are in an inconsistent state.
        """
        # Freeze canvas to prevent draws during batch removal
        canvas = None
        try:
            if hasattr(self.view, 'canvas') and self.view.canvas:
                canvas = self.view.canvas
                canvas.freeze()
        except Exception as e:
            logger.debug(f"Could not freeze canvas: {e}")
        
        try:
            annotation_ids = list(self.annotations.keys())
            for ann_id in annotation_ids:
                self._remove_annotation_visuals(ann_id)
            self.annotations.clear()
            self._annotation_refs.clear()
        finally:
            # Always unfreeze canvas
            if canvas:
                try:
                    canvas.unfreeze()
                    canvas.update()
                except Exception as e:
                    logger.debug(f"Could not unfreeze canvas: {e}")
    
    def show_temp_rectangle(self, t_start: float, t_end: float, 
                            f_min: float, f_max: float):
        """Show/update a temporary rectangle while drawing.
        
        Uses Rectangle visual for reliable rendering with proper alpha blending.
        
        Args:
            t_start: Start time
            t_end: End time
            f_min: Min frequency
            f_max: Max frequency
        """
        if not HAS_VISPY:
            return
        
        # Calculate bounds (ensure proper ordering)
        t_min = min(t_start, t_end)
        t_max = max(t_start, t_end)
        f_min_val = min(f_min, f_max)
        f_max_val = max(f_min, f_max)
        
        # Ensure minimum size for visibility
        if t_max - t_min < 0.001:
            t_max = t_min + 0.001
        if f_max_val - f_min_val < 1.0:
            f_max_val = f_min_val + 1.0
        
        width = t_max - t_min
        height = f_max_val - f_min_val
        center_x = (t_min + t_max) / 2
        center_y = (f_min_val + f_max_val) / 2
        
        # Remove old visuals first (more reliable than updating)
        self._remove_temp_visuals()
        
        # Create filled rectangle using Rectangle visual
        # This gives us a proper filled rectangle with border
        self.temp_fill = scene.visuals.Rectangle(
            center=(center_x, center_y),
            width=width,
            height=height,
            color=(1.0, 0.0, 0.0, 0.4),  # Red tint with 40% opacity - HIGH VISIBILITY
            border_color=(1.0, 0.0, 0.0, 1.0),  # Solid red border
            border_width=2.5,
            parent=self.view.scene
        )
        self.temp_fill.order = 100  # High order to render on top
        self.temp_fill.set_gl_state('translucent', depth_test=False)  # Ensure it draws on top
        
        logger.debug(f"Temp rect: center=({center_x:.2f}, {center_y:.0f}), size=({width:.3f}, {height:.0f})")
    
    def _remove_temp_visuals(self):
        """Remove temporary visuals from scene."""
        if self.temp_border is not None:
            if self.temp_border.parent is not None:
                self.temp_border.parent = None
            self.temp_border = None
        
        if self.temp_fill is not None:
            if self.temp_fill.parent is not None:
                self.temp_fill.parent = None
            self.temp_fill = None
    
    def hide_temp_rectangle(self):
        """Hide the temporary rectangle."""
        self._remove_temp_visuals()

    def set_visible(self, visible: bool):
        """Set visibility of all annotations."""
        for visuals in self.annotations.values():
            if visuals.get('rect'):
                visuals['rect'].visible = visible
            if visuals.get('doppler_curve'):
                visuals['doppler_curve'].visible = visible
            if visuals.get('doppler_markers'):
                visuals['doppler_markers'].visible = visible
    
    def set_annotation_visible(self, annotation_id: int, visible: bool):
        """Set visibility of a specific annotation (rectangle + Doppler curve based on curve flag).
        
        Args:
            annotation_id: ID of the annotation
            visible: Whether to show or hide the annotation
        """
        logger.info(f"set_annotation_visible called: annotation_id={annotation_id}, visible={visible}")
        
        if annotation_id not in self.annotations:
            logger.warning(f"Annotation {annotation_id} not found in renderer")
            return
        
        visuals = self.annotations[annotation_id]
        annotation = self._annotation_refs.get(annotation_id)
        
        # Update rectangle visibility
        if visuals.get('rect'):
            visuals['rect'].visible = visible
        
        # Update curve visibility based on both is_visible and show_doppler_curve
        if annotation:
            curve_visible = visible and annotation.show_doppler_curve
        else:
            curve_visible = visible
            
        if visuals.get('doppler_curve'):
            visuals['doppler_curve'].visible = curve_visible
            logger.debug(f"Set doppler_curve.visible = {curve_visible}")
        if visuals.get('doppler_markers'):
            visuals['doppler_markers'].visible = curve_visible
        
        self._update_canvas()
    
    def set_doppler_curve_visible(self, annotation_id: int, visible: bool):
        """Set visibility of just the Doppler curve for an annotation.
        
        Args:
            annotation_id: ID of the annotation
            visible: Whether to show or hide the Doppler curve
        """
        logger.info(f"set_doppler_curve_visible called: annotation_id={annotation_id}, visible={visible}")
        
        if annotation_id not in self.annotations:
            logger.warning(f"Annotation {annotation_id} not found in renderer (known ids: {list(self.annotations.keys())})")
            return
        
        visuals = self.annotations[annotation_id]
        annotation = self._annotation_refs.get(annotation_id)
        
        # Check if annotation is visible - curve should only show if annotation is also visible
        is_annotation_visible = True
        if annotation:
            is_annotation_visible = annotation.is_visible
        
        final_visibility = visible and is_annotation_visible
        
        if visuals.get('doppler_curve'):
            visuals['doppler_curve'].visible = final_visibility
            logger.info(f"Set doppler_curve.visible = {final_visibility} (annotation_visible={is_annotation_visible}, show_curve={visible})")
        else:
            logger.warning(f"No doppler_curve visual found for annotation {annotation_id}")
            
        if visuals.get('doppler_markers'):
            visuals['doppler_markers'].visible = final_visibility
        else:
            logger.warning(f"No doppler_markers visual found for annotation {annotation_id}")
        
        self._update_canvas()
    
    def update_doppler_curve(self, annotation: Annotation):
        """Update or add the Doppler curve for an annotation.
        
        Args:
            annotation: The annotation with updated Doppler points
        """
        if annotation.id not in self.annotations:
            return
        
        visuals = self.annotations[annotation.id]
        
        # Remove old Doppler visuals - MUST set invisible and clear data before removing
        if visuals.get('doppler_curve'):
            try:
                curve = visuals['doppler_curve']
                curve.visible = False
                curve.set_data(pos=np.zeros((2, 2), dtype=np.float32))
                if curve.parent:
                    curve.parent = None
            except Exception as e:
                logger.debug(f"Error removing old doppler_curve: {e}")
            del visuals['doppler_curve']
        
        if visuals.get('doppler_markers'):
            try:
                markers = visuals['doppler_markers']
                markers.visible = False
                markers.set_data(pos=np.zeros((1, 2), dtype=np.float32))
                if markers.parent:
                    markers.parent = None
            except Exception as e:
                logger.debug(f"Error removing old doppler_markers: {e}")
            del visuals['doppler_markers']
        
        # Add new Doppler visuals if points exist
        if annotation.points:
            doppler_visuals = self.create_doppler_curve_visuals(annotation)
            visuals.update(doppler_visuals)
            
            # Apply visibility based on annotation settings
            curve_visible = annotation.is_visible and annotation.show_doppler_curve
            if visuals.get('doppler_curve'):
                visuals['doppler_curve'].visible = curve_visible
            if visuals.get('doppler_markers'):
                visuals['doppler_markers'].visible = curve_visible
        
        # Update annotation reference
        self._annotation_refs[annotation.id] = annotation
        
        self._update_canvas()
    
    def batch_add_annotations(self, annotations: list, selected_id: Optional[int] = None):
        """Add multiple annotations in a single batch with canvas frozen.
        
        This is more efficient than calling add_annotation repeatedly,
        and prevents draw errors during batch operations.
        
        Args:
            annotations: List of Annotation objects to add
            selected_id: Optional ID of the selected annotation
        """
        if not annotations:
            return
            
        # Freeze canvas to prevent draws during batch addition
        canvas = None
        try:
            if hasattr(self.view, 'canvas') and self.view.canvas:
                canvas = self.view.canvas
                canvas.freeze()
        except Exception as e:
            logger.debug(f"Could not freeze canvas: {e}")
        
        try:
            for annotation in annotations:
                is_selected = (annotation.id == selected_id)
                
                if annotation.id in self.annotations:
                    self._remove_annotation_visuals(annotation.id)
                    del self.annotations[annotation.id]
                
                visuals = self.create_rectangle_visuals(annotation, is_selected)
                
                # Store annotation reference for later visibility updates
                self._annotation_refs[annotation.id] = annotation
                
                # Add Doppler curve if annotation has points
                if annotation.points:
                    doppler_visuals = self.create_doppler_curve_visuals(annotation)
                    visuals.update(doppler_visuals)
                    logger.debug(f"Added doppler curve for annotation {annotation.id}")
                
                # Apply visibility settings from annotation data
                rect_visible = annotation.is_visible
                curve_visible = annotation.is_visible and annotation.show_doppler_curve
                
                if visuals.get('rect'):
                    visuals['rect'].visible = rect_visible
                if visuals.get('doppler_curve'):
                    visuals['doppler_curve'].visible = curve_visible
                if visuals.get('doppler_markers'):
                    visuals['doppler_markers'].visible = curve_visible
                
                self.annotations[annotation.id] = visuals
                annotation.graphics_handle = visuals
                
        finally:
            # Always unfreeze canvas
            if canvas:
                try:
                    canvas.unfreeze()
                    canvas.update()
                except Exception as e:
                    logger.debug(f"Could not unfreeze canvas: {e}")
        
        logger.info(f"Batch added {len(annotations)} annotations")
    
    def _update_canvas(self):
        """Force canvas update."""
        try:
            if hasattr(self.view, 'canvas') and self.view.canvas:
                self.view.canvas.update()
        except Exception as e:
            logger.debug(f"Canvas update failed: {e}")
