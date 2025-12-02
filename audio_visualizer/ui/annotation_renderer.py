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
        
        Args:
            annotation: The annotation with Doppler curve points
        
        Returns:
            Dictionary with 'doppler_curve' and 'doppler_markers' visuals
        """
        if not HAS_VISPY or not annotation.points:
            return {}
        
        points = np.array(annotation.points)
        
        # Create curve line
        curve = scene.visuals.Line(parent=self.view.scene, method='gl')
        curve.set_data(pos=points, color='cyan', width=2.5)
        curve.order = 120  # Above rectangle but below markers
        curve.set_gl_state('translucent', depth_test=False)
        curve.visible = annotation.show_doppler_curve and annotation.is_visible
        
        # Create markers for points
        markers = scene.visuals.Markers(parent=self.view.scene)
        markers.set_data(
            pos=points,
            face_color=(0, 1, 1, 0.8),  # Cyan
            edge_color='white',
            size=10,
            edge_width=2
        )
        markers.order = 130  # On top of curve
        markers.set_gl_state('translucent', depth_test=False)
        markers.visible = annotation.show_doppler_curve and annotation.is_visible
        
        return {'doppler_curve': curve, 'doppler_markers': markers}
    
    def add_annotation(self, annotation: Annotation, is_selected: bool = False):
        """Add an annotation rectangle (and Doppler curve if exists) to the display.
        
        Args:
            annotation: The annotation to display
            is_selected: Whether this annotation is selected
        """
        if annotation.id in self.annotations:
            self.remove_annotation(annotation.id)
        
        visuals = self.create_rectangle_visuals(annotation, is_selected)
        
        # Add Doppler curve if annotation has points
        if annotation.points:
            doppler_visuals = self.create_doppler_curve_visuals(annotation)
            visuals.update(doppler_visuals)
        
        # Apply visibility settings
        if visuals.get('rect'):
            visuals['rect'].visible = annotation.is_visible
        
        self.annotations[annotation.id] = visuals
        annotation.graphics_handle = visuals
    
    def remove_annotation(self, annotation_id: int):
        """Remove an annotation rectangle and Doppler curve from the display.
        
        Args:
            annotation_id: ID of the annotation to remove
        """
        if annotation_id in self.annotations:
            visuals = self.annotations[annotation_id]
            
            # Remove rectangle visuals from scene
            if visuals.get('rect'):
                if visuals['rect'].parent:
                    visuals['rect'].parent = None
                visuals['rect'].visible = False
                
            if visuals.get('border') and visuals['border']:
                if visuals['border'].parent:
                    visuals['border'].parent = None
                visuals['border'].visible = False
                
            if visuals.get('fill') and visuals['fill']:
                if visuals['fill'].parent:
                    visuals['fill'].parent = None
                visuals['fill'].visible = False
            
            # Remove Doppler curve visuals
            if visuals.get('doppler_curve'):
                if visuals['doppler_curve'].parent:
                    visuals['doppler_curve'].parent = None
                visuals['doppler_curve'].visible = False
            
            if visuals.get('doppler_markers'):
                if visuals['doppler_markers'].parent:
                    visuals['doppler_markers'].parent = None
                visuals['doppler_markers'].visible = False
            
            del self.annotations[annotation_id]
            
            # Force canvas update to ensure visual removal
            if hasattr(self.view, 'canvas'):
                self.view.canvas.update()
            
            logger.debug(f"Removed visual for annotation {annotation_id}")
    
    def update_annotation(self, annotation: Annotation, is_selected: bool = False):
        """Update an existing annotation rectangle.
        
        Args:
            annotation: The annotation with updated bounds
            is_selected: Whether this annotation is selected
        """
        if annotation.id not in self.annotations:
            self.add_annotation(annotation, is_selected)
        else:
            # Remove old visuals
            self.remove_annotation(annotation.id)
            # Add new visuals
            self.add_annotation(annotation, is_selected)
    
    def set_selected(self, annotation_id: Optional[int]):
        """Set which annotation is selected (highlighted).
        
        Args:
            annotation_id: ID of annotation to select, or None to deselect all
        """
        for ann_id, visuals in self.annotations.items():
            is_selected = (ann_id == annotation_id)
            
            # We must recreate the visual because VisPy visuals are often frozen
            # or don't support dynamic property updates reliably
            old_rect = visuals.get('rect')
            if old_rect:
                # Capture current geometry
                center = old_rect.center
                width = old_rect.width
                height = old_rect.height
                parent = old_rect.parent
                
                # Remove old
                old_rect.parent = None
                
                # Define new colors
                if is_selected:
                    fill_color = (1.0, 0.4, 0.0, 0.4)    # Orange tint
                    border_color = (1.0, 0.4, 0.0, 1.0)  # Orange border
                    border_width = 3.0
                else:
                    fill_color = (1.0, 0.0, 0.0, 0.3)    # Red tint
                    border_color = (1.0, 0.0, 0.0, 1.0)  # Red border
                    border_width = 2.0
                
                # Create new visual
                new_rect = scene.visuals.Rectangle(
                    center=center,
                    width=width,
                    height=height,
                    color=fill_color,
                    border_color=border_color,
                    border_width=border_width,
                    parent=parent
                )
                new_rect.order = 100
                new_rect.set_gl_state('translucent', depth_test=False)
                
                # Update storage
                visuals['rect'] = new_rect
    
    def clear_all(self):
        """Remove all annotation rectangles."""
        annotation_ids = list(self.annotations.keys())
        for ann_id in annotation_ids:
            self.remove_annotation(ann_id)
    
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
        """Set visibility of a specific annotation (rectangle + Doppler curve).
        
        Args:
            annotation_id: ID of the annotation
            visible: Whether to show or hide
        """
        if annotation_id in self.annotations:
            visuals = self.annotations[annotation_id]
            if visuals.get('rect'):
                visuals['rect'].visible = visible
            if visuals.get('doppler_curve'):
                visuals['doppler_curve'].visible = visible
            if visuals.get('doppler_markers'):
                visuals['doppler_markers'].visible = visible
            
            if hasattr(self.view, 'canvas'):
                self.view.canvas.update()
    
    def set_doppler_curve_visible(self, annotation_id: int, visible: bool):
        """Set visibility of just the Doppler curve for an annotation.
        
        Args:
            annotation_id: ID of the annotation
            visible: Whether to show or hide the Doppler curve
        """
        if annotation_id in self.annotations:
            visuals = self.annotations[annotation_id]
            if visuals.get('doppler_curve'):
                visuals['doppler_curve'].visible = visible
            if visuals.get('doppler_markers'):
                visuals['doppler_markers'].visible = visible
            
            if hasattr(self.view, 'canvas'):
                self.view.canvas.update()
    
    def update_doppler_curve(self, annotation: Annotation):
        """Update or add the Doppler curve for an annotation.
        
        Args:
            annotation: The annotation with updated Doppler points
        """
        if annotation.id not in self.annotations:
            return
        
        visuals = self.annotations[annotation.id]
        
        # Remove old Doppler visuals
        if visuals.get('doppler_curve'):
            if visuals['doppler_curve'].parent:
                visuals['doppler_curve'].parent = None
            del visuals['doppler_curve']
        
        if visuals.get('doppler_markers'):
            if visuals['doppler_markers'].parent:
                visuals['doppler_markers'].parent = None
            del visuals['doppler_markers']
        
        # Add new Doppler visuals if points exist
        if annotation.points:
            doppler_visuals = self.create_doppler_curve_visuals(annotation)
            visuals.update(doppler_visuals)
        
        if hasattr(self.view, 'canvas'):
            self.view.canvas.update()

