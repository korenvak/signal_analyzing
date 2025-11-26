"""
Shader-based Renderer with Real-time Uniform Updates
Provides instant colormap and dB range changes without recomputation.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

try:
    import vispy
    from vispy import gloo
    from vispy.gloo import Program
    HAS_VISPY = True
except ImportError:
    vispy = None
    gloo = None
    Program = None
    HAS_VISPY = False

from .shader_colormap import COLORMAP_FRAGMENT_SHADER


class ShaderRenderer:
    """High-performance shader-based renderer with real-time parameter updates."""
    
    def __init__(self, canvas=None):
        """Initialize shader renderer.
        
        Args:
            canvas: VisPy canvas to render to
        """
        self.canvas = canvas
        self.program = None
        self.texture = None
        self.vertices = None
        
        # Current uniforms
        self.uniforms = {
            'u_db_min': -80.0,
            'u_db_max': 0.0,
            'u_colormap_type': 0,  # 0=viridis, 1=plasma, 2=jet, 3=magma, 4=inferno
            'u_alpha': 1.0,
            'u_crosshair_pos': [0.0, 0.0, 0.0, 1.0],  # x, y, enabled, width
            'u_crosshair_color': [1.0, 1.0, 1.0, 0.8]  # white, semi-transparent
        }
        
        # Data bounds for coordinate transformation
        self.data_bounds = {
            'time_start': 0.0,
            'time_end': 1.0,
            'freq_start': 0.0,
            'freq_end': 1.0
        }
        
        # Colormap mapping
        self.colormap_names = {
            'viridis': 0,
            'plasma': 1,
            'jet': 2,
            'magma': 3,
            'inferno': 4,
            'hot': 5,
            'cool': 6,
            'grayscale': 7
        }
        
        self.enabled = HAS_VISPY
        
        if self.enabled:
            self._initialize_shader()
        
        logger.info(f"ShaderRenderer initialized: enabled={self.enabled}")
    
    def _initialize_shader(self):
        """Initialize the shader program."""
        try:
            # Vertex shader (simple quad)
            vertex_shader = """
            #version 330 core
            
            attribute vec2 a_position;
            attribute vec2 a_texcoord;
            
            varying vec2 v_texcoord;
            varying vec2 v_world_pos;
            
            uniform vec4 u_data_bounds;  // time_start, time_end, freq_start, freq_end
            
            void main() {
                gl_Position = vec4(a_position, 0.0, 1.0);
                v_texcoord = a_texcoord;
                
                // Transform texture coordinates to world coordinates
                v_world_pos.x = mix(u_data_bounds.x, u_data_bounds.y, a_texcoord.x);
                v_world_pos.y = mix(u_data_bounds.z, u_data_bounds.w, a_texcoord.y);
            }
            """
            
            # Create shader program
            self.program = Program(vertex_shader, COLORMAP_FRAGMENT_SHADER)
            
            # Create quad vertices - separate buffers for each attribute
            positions = np.array([
                [-1, -1],  # bottom-left
                [ 1, -1],  # bottom-right
                [-1,  1],  # top-left
                [ 1,  1],  # top-right
            ], dtype=np.float32)
            
            texcoords = np.array([
                [0, 0],  # bottom-left
                [1, 0],  # bottom-right
                [0, 1],  # top-left
                [1, 1],  # top-right
            ], dtype=np.float32)
            
            # Create vertex buffers
            self.pos_buffer = gloo.VertexBuffer(positions)
            self.tex_buffer = gloo.VertexBuffer(texcoords)
            
            # Set vertex attributes
            self.program['a_position'] = self.pos_buffer
            self.program['a_texcoord'] = self.tex_buffer
            
            # Initialize uniforms
            self._update_all_uniforms()
            
            logger.debug("Shader program initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize shader program: {e}")
            self.enabled = False
    
    def set_data(self, data: np.ndarray, data_bounds: Dict[str, float] = None):
        """Set the data texture and bounds.
        
        Args:
            data: 2D numpy array (frequency, time)
            data_bounds: Dictionary with time_start, time_end, freq_start, freq_end
        """
        if not self.enabled or self.program is None:
            return False
        
        try:
            # Update data bounds
            if data_bounds:
                self.data_bounds.update(data_bounds)
            
            # Create or update texture
            if self.texture is None:
                self.texture = gloo.Texture2D(data, format='luminance')
            else:
                self.texture.set_data(data)
            
            # Set texture uniform
            self.program['u_spectrogram_data'] = self.texture
            
            # Update data bounds uniform
            bounds_array = np.array([
                self.data_bounds['time_start'],
                self.data_bounds['time_end'],
                self.data_bounds['freq_start'],
                self.data_bounds['freq_end']
            ], dtype=np.float32)
            self.program['u_data_bounds'] = bounds_array
            
            logger.debug(f"Shader data updated: shape={data.shape}, bounds={self.data_bounds}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set shader data: {e}")
            return False
    
    def set_colormap(self, colormap_name: str):
        """Set the colormap type.
        
        Args:
            colormap_name: Name of colormap ('viridis', 'plasma', 'jet', etc.)
        """
        if not self.enabled or self.program is None:
            return False
        
        colormap_id = self.colormap_names.get(colormap_name, 0)
        
        if self.uniforms['u_colormap_type'] != colormap_id:
            self.uniforms['u_colormap_type'] = colormap_id
            self.program['u_colormap_type'] = colormap_id
            logger.debug(f"Colormap changed to: {colormap_name} (id={colormap_id})")
            return True
        
        return False
    
    def set_db_range(self, db_min: float, db_max: float):
        """Set the dB range for color mapping.
        
        Args:
            db_min: Minimum dB value
            db_max: Maximum dB value
        """
        if not self.enabled or self.program is None:
            return False
        
        # Ensure valid range
        if db_min >= db_max:
            db_max = db_min + 1.0
        
        changed = False
        
        if self.uniforms['u_db_min'] != db_min:
            self.uniforms['u_db_min'] = db_min
            self.program['u_db_min'] = float(db_min)
            changed = True
        
        if self.uniforms['u_db_max'] != db_max:
            self.uniforms['u_db_max'] = db_max
            self.program['u_db_max'] = float(db_max)
            changed = True
        
        if changed:
            logger.debug(f"dB range changed to: [{db_min:.1f}, {db_max:.1f}] dB")
        
        return changed
    
    def set_alpha(self, alpha: float):
        """Set the transparency level.
        
        Args:
            alpha: Alpha value (0.0 = transparent, 1.0 = opaque)
        """
        if not self.enabled or self.program is None:
            return False
        
        alpha = max(0.0, min(1.0, alpha))
        
        if self.uniforms['u_alpha'] != alpha:
            self.uniforms['u_alpha'] = alpha
            self.program['u_alpha'] = float(alpha)
            logger.debug(f"Alpha changed to: {alpha:.2f}")
            return True
        
        return False
    
    def set_crosshair(self, x: float, y: float, enabled: bool = True, width: float = 1.0):
        """Set crosshair position and properties.
        
        Args:
            x: X position in world coordinates
            y: Y position in world coordinates
            enabled: Whether crosshair is visible
            width: Line width in pixels
        """
        if not self.enabled or self.program is None:
            return False
        
        crosshair_data = [float(x), float(y), 1.0 if enabled else 0.0, float(width)]
        
        if self.uniforms['u_crosshair_pos'] != crosshair_data:
            self.uniforms['u_crosshair_pos'] = crosshair_data
            self.program['u_crosshair_pos'] = np.array(crosshair_data, dtype=np.float32)
            return True
        
        return False
    
    def render(self):
        """Render the current data with shaders."""
        if not self.enabled or self.program is None or self.texture is None:
            return False
        
        try:
            # Enable blending for transparency
            gloo.set_state(blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
            
            # Render the quad
            self.program.draw('triangle_strip')
            
            return True
            
        except Exception as e:
            logger.error(f"Shader render failed: {e}")
            return False
    
    def _update_all_uniforms(self):
        """Update all uniform values in the shader."""
        if not self.enabled or self.program is None:
            return
        
        try:
            for name, value in self.uniforms.items():
                if isinstance(value, list):
                    self.program[name] = np.array(value, dtype=np.float32)
                else:
                    self.program[name] = float(value)
            
            logger.debug("All shader uniforms updated")
            
        except Exception as e:
            logger.error(f"Failed to update shader uniforms: {e}")
    
    def get_available_colormaps(self) -> list:
        """Get list of available colormap names."""
        return list(self.colormap_names.keys())
    
    def get_current_uniforms(self) -> Dict[str, Any]:
        """Get current uniform values."""
        return self.uniforms.copy()
    
    def is_enabled(self) -> bool:
        """Check if shader rendering is enabled and working."""
        return self.enabled and self.program is not None


# Global instance
_shader_renderer = None

def get_shader_renderer(canvas=None) -> ShaderRenderer:
    """Get the global shader renderer instance."""
    global _shader_renderer
    if _shader_renderer is None:
        _shader_renderer = ShaderRenderer(canvas)
    return _shader_renderer