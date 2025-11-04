import numpy as np
import threading
from typing import Dict, Tuple, Optional, List, Any
import os
import time

try:
    import OpenGL.GL as gl
    import OpenGL.GL.shaders as shaders
    HAS_OPENGL = True
except ImportError:
    gl = None
    shaders = None
    HAS_OPENGL = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False

from ..core.cache_manager import CacheManager

class GPUTexture:
    """Wrapper for OpenGL texture with GPU memory management."""
    
    def __init__(self, width: int, height: int, texture_format=None):
        if not HAS_OPENGL:
            raise RuntimeError("OpenGL not available")
        
        self.width = width
        self.height = height
        self.texture_id = gl.glGenTextures(1)
        self.format = texture_format or gl.GL_R32F
        self.created_time = time.time()
        self.last_access = time.time()
        
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, self.format, width, height, 0, 
                       gl.GL_RED, gl.GL_FLOAT, None)
        
        # Set texture parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    
    def update_data(self, data: np.ndarray):
        """Update texture with new data."""
        if data.shape != (self.height, self.width):
            raise ValueError(f"Data shape {data.shape} doesn't match texture size {(self.height, self.width)}")
        
        self.last_access = time.time()
        
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                          gl.GL_RED, gl.GL_FLOAT, data.astype(np.float32))
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    
    def bind(self, texture_unit: int = 0):
        """Bind texture to specified unit."""
        gl.glActiveTexture(gl.GL_TEXTURE0 + texture_unit)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        self.last_access = time.time()
    
    def unbind(self):
        """Unbind texture."""
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    
    def delete(self):
        """Delete GPU texture."""
        if self.texture_id:
            gl.glDeleteTextures([self.texture_id])
            self.texture_id = None

class ShaderProgram:
    """OpenGL shader program wrapper."""
    
    def __init__(self, vertex_source: str, fragment_source: str):
        if not HAS_OPENGL:
            raise RuntimeError("OpenGL not available")
        
        self.program_id = None
        self.uniforms = {}
        self.attributes = {}
        
        try:
            vertex_shader = shaders.compileShader(vertex_source, gl.GL_VERTEX_SHADER)
            fragment_shader = shaders.compileShader(fragment_source, gl.GL_FRAGMENT_SHADER)
            
            self.program_id = shaders.compileProgram(vertex_shader, fragment_shader)
            
            # Clean up individual shaders
            gl.glDeleteShader(vertex_shader)
            gl.glDeleteShader(fragment_shader)
            
            self._cache_uniform_locations()
            
        except Exception as e:
            raise RuntimeError(f"Shader compilation failed: {e}")
    
    def _cache_uniform_locations(self):
        """Cache uniform locations for faster access."""
        uniform_names = [
            'u_projection', 'u_view', 'u_tile_bounds', 'u_viewport_size',
            'u_spectrogram_data', 'u_db_min', 'u_db_max', 'u_colormap_type',
            'u_crosshair_pos', 'u_crosshair_color', 'u_alpha'
        ]
        
        for name in uniform_names:
            location = gl.glGetUniformLocation(self.program_id, name)
            if location != -1:
                self.uniforms[name] = location
    
    def use(self):
        """Activate shader program."""
        gl.glUseProgram(self.program_id)
    
    def set_uniform_matrix4(self, name: str, matrix: np.ndarray):
        """Set 4x4 matrix uniform."""
        if name in self.uniforms:
            gl.glUniformMatrix4fv(self.uniforms[name], 1, gl.GL_FALSE, matrix.astype(np.float32))
    
    def set_uniform_vec4(self, name: str, vector: np.ndarray):
        """Set vec4 uniform."""
        if name in self.uniforms:
            gl.glUniform4f(self.uniforms[name], *vector.astype(np.float32))
    
    def set_uniform_vec2(self, name: str, vector: np.ndarray):
        """Set vec2 uniform."""
        if name in self.uniforms:
            gl.glUniform2f(self.uniforms[name], *vector.astype(np.float32))
    
    def set_uniform_float(self, name: str, value: float):
        """Set float uniform."""
        if name in self.uniforms:
            gl.glUniform1f(self.uniforms[name], float(value))
    
    def set_uniform_int(self, name: str, value: int):
        """Set int uniform."""
        if name in self.uniforms:
            gl.glUniform1i(self.uniforms[name], int(value))
    
    def delete(self):
        """Delete shader program."""
        if self.program_id:
            gl.glDeleteProgram(self.program_id)
            self.program_id = None

class RenderManager:
    """High-performance GPU rendering manager for spectrogram visualization."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        
        # Rendering state
        self.viewport_size = (1920, 1080)
        self.projection_matrix = np.eye(4, dtype=np.float32)
        self.view_matrix = np.eye(4, dtype=np.float32)
        
        # Shader programs
        self.shaders = {}
        self.current_shader = None
        
        # Texture management
        self.textures = {}
        self.texture_pool = []
        self.max_textures = 64
        
        # Vertex buffers
        self.quad_vao = None
        self.quad_vbo = None
        
        # Rendering parameters
        self.db_range = (-80.0, 0.0)
        self.colormap_type = 0  # 0=viridis, 1=plasma, 2=jet, 3=magma
        self.global_alpha = 1.0
        
        # Crosshair
        self.crosshair_enabled = False
        self.crosshair_pos = (0.0, 0.0)
        self.crosshair_color = (1.0, 1.0, 1.0, 0.8)
        self.crosshair_width = 2.0
        
        # Threading
        self._render_lock = threading.RLock()
        
        # Performance tracking
        self.frame_times = []
        self.max_frame_history = 60
        
    def initialize(self):
        """Initialize OpenGL resources."""
        if not HAS_OPENGL:
            raise RuntimeError("OpenGL not available")
        
        with self._render_lock:
            self._load_shaders()
            self._create_quad_geometry()
            self._setup_opengl_state()
    
    def _load_shaders(self):
        """Load and compile shader programs."""
        shader_dir = os.path.join(os.path.dirname(__file__), 'shaders')
        
        # Load spectrogram shaders
        vertex_path = os.path.join(shader_dir, 'spectrogram_vertex.glsl')
        fragment_path = os.path.join(shader_dir, 'spectrogram_fragment.glsl')
        
        with open(vertex_path, 'r') as f:
            vertex_source = f.read()
        
        with open(fragment_path, 'r') as f:
            fragment_source = f.read()
        
        self.shaders['spectrogram'] = ShaderProgram(vertex_source, fragment_source)
        self.current_shader = self.shaders['spectrogram']
    
    def _create_quad_geometry(self):
        """Create quad geometry for tile rendering."""
        # Quad vertices: position (x, y) and texture coordinates (u, v)
        quad_vertices = np.array([
            # Position  # TexCoord
            -1.0, -1.0,  0.0, 0.0,  # Bottom-left
             1.0, -1.0,  1.0, 0.0,  # Bottom-right
             1.0,  1.0,  1.0, 1.0,  # Top-right
            -1.0,  1.0,  0.0, 1.0   # Top-left
        ], dtype=np.float32)
        
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)
        
        # Create VAO
        self.quad_vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.quad_vao)
        
        # Create VBO
        self.quad_vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.quad_vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, quad_vertices.nbytes, quad_vertices, gl.GL_STATIC_DRAW)
        
        # Create EBO
        self.quad_ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.quad_ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
        
        # Set vertex attributes
        stride = 4 * 4  # 4 floats * 4 bytes
        
        # Position attribute (location 0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, None)
        gl.glEnableVertexAttribArray(0)
        
        # Texture coordinate attribute (location 1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, 
                               gl.ctypes.c_void_p(2 * 4))
        gl.glEnableVertexAttribArray(1)
        
        gl.glBindVertexArray(0)
    
    def _setup_opengl_state(self):
        """Setup default OpenGL state."""
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    
    def set_viewport(self, width: int, height: int):
        """Set rendering viewport."""
        with self._render_lock:
            self.viewport_size = (width, height)
            gl.glViewport(0, 0, width, height)
            
            # Update projection matrix for 2D rendering
            self.projection_matrix = np.array([
                [2.0/width, 0.0, 0.0, -1.0],
                [0.0, 2.0/height, 0.0, -1.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32)
    
    def set_view_transform(self, time_range: Tuple[float, float], 
                          freq_range: Tuple[float, float]):
        """Set view transformation for time-frequency space."""
        with self._render_lock:
            time_span = time_range[1] - time_range[0]
            freq_span = freq_range[1] - freq_range[0]
            
            # Create view matrix to map world coordinates to normalized coordinates
            self.view_matrix = np.array([
                [2.0/time_span, 0.0, 0.0, -2.0*time_range[0]/time_span - 1.0],
                [0.0, 2.0/freq_span, 0.0, -2.0*freq_range[0]/freq_span - 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ], dtype=np.float32)
    
    def render_spectrogram_tile(self, data: np.ndarray, 
                               time_range: Tuple[float, float],
                               freq_range: Tuple[float, float]):
        """Render a single spectrogram tile."""
        with self._render_lock:
            frame_start = time.time()
            
            # Get or create texture
            texture_key = f"tile_{time_range[0]:.2f}_{time_range[1]:.2f}_{freq_range[0]:.0f}_{freq_range[1]:.0f}"
            
            if texture_key not in self.textures:
                self.textures[texture_key] = GPUTexture(data.shape[1], data.shape[0])
            
            texture = self.textures[texture_key]
            texture.update_data(data)
            
            # Use spectrogram shader
            self.current_shader.use()
            
            # Set uniforms
            self.current_shader.set_uniform_matrix4('u_projection', self.projection_matrix)
            self.current_shader.set_uniform_matrix4('u_view', self.view_matrix)
            
            tile_bounds = np.array([time_range[0], time_range[1], 
                                  freq_range[0], freq_range[1]], dtype=np.float32)
            self.current_shader.set_uniform_vec4('u_tile_bounds', tile_bounds)
            
            viewport_size = np.array(self.viewport_size, dtype=np.float32)
            self.current_shader.set_uniform_vec2('u_viewport_size', viewport_size)
            
            self.current_shader.set_uniform_float('u_db_min', self.db_range[0])
            self.current_shader.set_uniform_float('u_db_max', self.db_range[1])
            self.current_shader.set_uniform_int('u_colormap_type', self.colormap_type)
            self.current_shader.set_uniform_float('u_alpha', self.global_alpha)
            
            # Crosshair uniforms
            crosshair_pos = np.array([
                self.crosshair_pos[0], self.crosshair_pos[1],
                1.0 if self.crosshair_enabled else 0.0,
                self.crosshair_width
            ], dtype=np.float32)
            self.current_shader.set_uniform_vec4('u_crosshair_pos', crosshair_pos)
            
            crosshair_color = np.array(self.crosshair_color, dtype=np.float32)
            self.current_shader.set_uniform_vec4('u_crosshair_color', crosshair_color)
            
            # Bind texture
            texture.bind(0)
            self.current_shader.set_uniform_int('u_spectrogram_data', 0)
            
            # Render quad
            gl.glBindVertexArray(self.quad_vao)
            gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)
            
            # Unbind texture
            texture.unbind()
            
            # Track frame time
            frame_time = time.time() - frame_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > self.max_frame_history:
                self.frame_times.pop(0)
    
    def set_colormap(self, colormap_name: str):
        """Set active colormap."""
        colormap_map = {
            'viridis': 0,
            'plasma': 1,
            'jet': 2,
            'magma': 3
        }
        
        if colormap_name in colormap_map:
            self.colormap_type = colormap_map[colormap_name]
    
    def set_db_range(self, db_min: float, db_max: float):
        """Set dB range for colormap scaling."""
        self.db_range = (db_min, db_max)
    
    def set_crosshair(self, enabled: bool, time_pos: float = 0.0, 
                     freq_pos: float = 0.0, color: Tuple[float, float, float, float] = None):
        """Set crosshair parameters."""
        self.crosshair_enabled = enabled
        self.crosshair_pos = (time_pos, freq_pos)
        if color:
            self.crosshair_color = color
    
    def clear_frame(self):
        """Clear the frame buffer."""
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get rendering performance statistics."""
        if not self.frame_times:
            return {'fps': 0.0, 'avg_frame_time_ms': 0.0, 'min_frame_time_ms': 0.0, 'max_frame_time_ms': 0.0}
        
        avg_frame_time = np.mean(self.frame_times)
        fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        return {
            'fps': fps,
            'avg_frame_time_ms': avg_frame_time * 1000,
            'min_frame_time_ms': np.min(self.frame_times) * 1000,
            'max_frame_time_ms': np.max(self.frame_times) * 1000
        }
    
    def cleanup_old_textures(self, max_age_seconds: float = 300.0):
        """Clean up old unused textures."""
        current_time = time.time()
        textures_to_remove = []
        
        for key, texture in self.textures.items():
            if current_time - texture.last_access > max_age_seconds:
                textures_to_remove.append(key)
        
        for key in textures_to_remove:
            texture = self.textures.pop(key)
            texture.delete()
    
    def shutdown(self):
        """Clean up all OpenGL resources."""
        with self._render_lock:
            # Delete all textures
            for texture in self.textures.values():
                texture.delete()
            self.textures.clear()
            
            # Delete shaders
            for shader in self.shaders.values():
                shader.delete()
            self.shaders.clear()
            
            # Delete vertex objects
            if self.quad_vao:
                gl.glDeleteVertexArrays([self.quad_vao])
            if self.quad_vbo:
                gl.glDeleteBuffers([self.quad_vbo])
            if hasattr(self, 'quad_ebo'):
                gl.glDeleteBuffers([self.quad_ebo])