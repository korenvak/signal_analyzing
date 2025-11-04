// Vertex shader for spectrogram rendering
#version 330 core

// Input attributes
in vec2 position;      // Vertex position in tile space
in vec2 texcoord;      // Texture coordinates

// Uniforms for transformation
uniform mat4 u_projection;     // Projection matrix
uniform mat4 u_view;          // View matrix
uniform vec4 u_tile_bounds;   // [time_start, time_end, freq_start, freq_end]
uniform vec2 u_viewport_size; // Viewport dimensions

// Output to fragment shader
out vec2 v_texcoord;
out vec2 v_world_pos;    // World position for crosshair calculations

void main() {
    // Transform position from tile space to world space
    vec2 world_pos = vec2(
        mix(u_tile_bounds.x, u_tile_bounds.y, position.x),  // Time
        mix(u_tile_bounds.z, u_tile_bounds.w, position.y)   // Frequency
    );
    
    // Apply view and projection transformations
    vec4 view_pos = u_view * vec4(world_pos, 0.0, 1.0);
    gl_Position = u_projection * view_pos;
    
    // Pass texture coordinates and world position to fragment shader
    v_texcoord = texcoord;
    v_world_pos = world_pos;
}