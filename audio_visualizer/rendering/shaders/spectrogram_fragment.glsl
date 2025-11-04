// Fragment shader for spectrogram rendering with dynamic colormaps
#version 330 core

// Input from vertex shader
in vec2 v_texcoord;
in vec2 v_world_pos;

// Uniforms
uniform sampler2D u_spectrogram_data;  // Spectrogram magnitude data
uniform float u_db_min;               // Minimum dB value for color mapping
uniform float u_db_max;               // Maximum dB value for color mapping
uniform int u_colormap_type;          // Colormap selection (0=viridis, 1=plasma, 2=jet, etc.)
uniform vec4 u_crosshair_pos;         // [time, freq, enabled, line_width]
uniform vec4 u_crosshair_color;       // Crosshair color
uniform float u_alpha;               // Global alpha

// Output
out vec4 fragColor;

// Colormap functions
vec3 viridis_colormap(float t) {
    // Viridis colormap approximation
    const vec3 c0 = vec3(0.2777273272234177, 0.005407344544966578, 0.3340998053353061);
    const vec3 c1 = vec3(0.1050930431085774, 1.404613529898575, 1.384590162594685);
    const vec3 c2 = vec3(-0.3308618287255563, 0.214847559468213, 0.09509516302823659);
    const vec3 c3 = vec3(-4.634230498983486, -5.799100973351585, -19.33244095627987);
    const vec3 c4 = vec3(6.228269936347081, 14.17993336680509, 56.69055260068105);
    const vec3 c5 = vec3(4.776384997670288, -13.74514537774601, -65.35303263337234);
    const vec3 c6 = vec3(-5.435455855934631, 4.645852612178535, 26.3124352495832);
    
    return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}

vec3 plasma_colormap(float t) {
    // Plasma colormap approximation
    const vec3 c0 = vec3(0.05873234392399702, 0.02333670892565664, 0.5433401826748754);
    const vec3 c1 = vec3(2.176514634195958, 0.2383834171260182, 0.7539604599784036);
    const vec3 c2 = vec3(-2.689460476458092, -7.455851135738909, 3.110799939717086);
    const vec3 c3 = vec3(6.130348345893603, 42.3461881477227, -28.51885465332158);
    const vec3 c4 = vec3(-11.10743619062271, -82.66631109428045, 60.13984767418263);
    const vec3 c5 = vec3(10.02306557647065, 71.41361770095349, -54.07218655560067);
    const vec3 c6 = vec3(-3.658713842777788, -22.93153465461149, 18.19190778539828);
    
    return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}

vec3 jet_colormap(float t) {
    // Classic jet colormap
    float r = clamp(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0);
    float g = clamp(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0);
    float b = clamp(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0);
    return vec3(r, g, b);
}

vec3 magma_colormap(float t) {
    // Magma colormap approximation
    const vec3 c0 = vec3(-0.002136485053939582, -0.000749655052795221, -0.005386127855323933);
    const vec3 c1 = vec3(0.2516605407371642, 0.6775232436837668, 2.494026599312351);
    const vec3 c2 = vec3(8.353717279216625, -3.577719514958484, 0.3144679030132573);
    const vec3 c3 = vec3(-27.66873308576866, 14.26473078096533, -13.64921318813922);
    const vec3 c4 = vec3(52.17613981234068, -27.94360607168351, 12.94416944238394);
    const vec3 c5 = vec3(-50.76852536473588, 29.04658282127291, 4.23415299384598);
    const vec3 c6 = vec3(18.65570506591883, -11.48977351997711, -5.601961508734096);
    
    return c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6)))));
}

vec3 apply_colormap(float value, int colormap_type) {
    // Clamp and normalize value
    float t = clamp(value, 0.0, 1.0);
    
    if (colormap_type == 0) {
        return viridis_colormap(t);
    } else if (colormap_type == 1) {
        return plasma_colormap(t);
    } else if (colormap_type == 2) {
        return jet_colormap(t);
    } else if (colormap_type == 3) {
        return magma_colormap(t);
    } else {
        // Default grayscale
        return vec3(t, t, t);
    }
}

void main() {
    // Sample spectrogram data
    float magnitude_db = texture(u_spectrogram_data, v_texcoord).r;
    
    // Normalize to [0, 1] range based on dB limits
    float normalized_db = (magnitude_db - u_db_min) / (u_db_max - u_db_min);
    
    // Apply colormap
    vec3 color = apply_colormap(normalized_db, u_colormap_type);
    
    // Initialize output color
    fragColor = vec4(color, u_alpha);
    
    // Add crosshair if enabled
    if (u_crosshair_pos.z > 0.5) {
        float crosshair_time = u_crosshair_pos.x;
        float crosshair_freq = u_crosshair_pos.y;
        float line_width = u_crosshair_pos.w;
        
        // Calculate distance to crosshair lines
        float time_dist = abs(v_world_pos.x - crosshair_time);
        float freq_dist = abs(v_world_pos.y - crosshair_freq);
        
        // Check if we're on crosshair lines
        bool on_time_line = time_dist < line_width;
        bool on_freq_line = freq_dist < line_width;
        
        if (on_time_line || on_freq_line) {
            // Blend crosshair color
            float crosshair_alpha = u_crosshair_color.a;
            fragColor.rgb = mix(fragColor.rgb, u_crosshair_color.rgb, crosshair_alpha);
        }
    }
}