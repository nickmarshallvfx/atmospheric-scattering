"""
GPU-accelerated LUT precomputation using Blender's gpu module.

Uses fragment shaders rendered to offscreen buffers for GPU acceleration
without requiring external dependencies like CuPy.

Note: Blender 4.0+ uses GPUShaderCreateInfo API for shader creation.

VERSION: 43 - Remove max(1e-10) protection in GetTransmittance (not in reference)
# V34: 4D interpolation. V35: Mie combination. V37: Push constant fix. V43: Remove transmittance protection
"""
print("[Helios GPU] Module loaded - VERSION 43 (remove trans protection)")

import os
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

try:
    import bpy
    import gpu
    from gpu_extras.batch import batch_for_shader
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

from .constants import (
    TRANSMITTANCE_TEXTURE_WIDTH,
    TRANSMITTANCE_TEXTURE_HEIGHT,
    SCATTERING_TEXTURE_WIDTH,
    SCATTERING_TEXTURE_HEIGHT,
    SCATTERING_TEXTURE_DEPTH,
    IRRADIANCE_TEXTURE_WIDTH,
    IRRADIANCE_TEXTURE_HEIGHT,
)
from .parameters import AtmosphereParameters


@dataclass
class PrecomputedTextures:
    """Container for precomputed LUT textures."""
    transmittance: np.ndarray
    scattering: np.ndarray
    irradiance: np.ndarray
    single_mie_scattering: Optional[np.ndarray] = None


def create_transmittance_shader() -> 'gpu.types.GPUShader':
    """Create the transmittance precomputation shader."""
    shader_info = gpu.types.GPUShaderCreateInfo()
    
    # Vertex input
    shader_info.vertex_in(0, 'VEC2', 'pos')
    
    # Vertex-fragment interface
    vert_out = gpu.types.GPUStageInterfaceInfo("vert_iface")
    vert_out.smooth('VEC2', 'uv')
    shader_info.vertex_out(vert_out)
    
    # Push constants (uniforms)
    shader_info.push_constant('FLOAT', 'bottom_radius')
    shader_info.push_constant('FLOAT', 'top_radius')
    shader_info.push_constant('VEC3', 'rayleigh_scattering')
    shader_info.push_constant('FLOAT', 'rayleigh_scale_height')
    shader_info.push_constant('VEC3', 'mie_extinction')
    shader_info.push_constant('FLOAT', 'mie_scale_height')
    shader_info.push_constant('VEC3', 'absorption_extinction')
    
    # Fragment output
    shader_info.fragment_out(0, 'VEC4', 'fragColor')
    
    # Vertex shader
    shader_info.vertex_source('''
void main() {
    uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
''')
    
    # Fragment shader - matches reference Bruneton implementation
    shader_info.fragment_source('''
#define TRANSMITTANCE_TEXTURE_WIDTH 256
#define TRANSMITTANCE_TEXTURE_HEIGHT 64
#define SAMPLE_COUNT 500

// Ozone layer parameters (must match CPU constants)
#define OZONE_CENTER_ALTITUDE 25000.0
#define OZONE_WIDTH 15000.0

float SafeSqrt(float x) { return sqrt(max(x, 0.0)); }

float DistanceToTopAtmosphereBoundary(float r, float mu) {
    float discriminant = r * r * (mu * mu - 1.0) + top_radius * top_radius;
    return max(0.0, -r * mu + SafeSqrt(discriminant));
}

float GetUnitRangeFromTextureCoord(float u, float tex_size) {
    return (u - 0.5 / tex_size) / (1.0 - 1.0 / tex_size);
}

// Rayleigh/Mie density: simple exponential profile
float GetExpDensity(float altitude, float scale_height) {
    return exp(-altitude / scale_height);
}

// Ozone density: two-layer profile matching Bruneton reference exactly
// Layer 0: altitude < 25km, density = altitude/15000 - 2/3
// Layer 1: altitude >= 25km, density = -altitude/15000 + 8/3  (uses absolute altitude!)
float GetOzoneDensity(float altitude) {
    float density;
    if (altitude < OZONE_CENTER_ALTITUDE) {
        // Lower layer: linear_term = 1/15000, constant_term = -2/3
        density = altitude / OZONE_WIDTH - 2.0 / 3.0;
    } else {
        // Upper layer: linear_term = -1/15000, constant_term = 8/3 (absolute altitude)
        density = -altitude / OZONE_WIDTH + 8.0 / 3.0;
    }
    return clamp(density, 0.0, 1.0);
}

void main() {
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    
    float x_mu = GetUnitRangeFromTextureCoord(uv.x, float(TRANSMITTANCE_TEXTURE_WIDTH));
    float x_r = GetUnitRangeFromTextureCoord(uv.y, float(TRANSMITTANCE_TEXTURE_HEIGHT));
    
    float rho = H * x_r;
    float r = sqrt(rho * rho + bottom_radius * bottom_radius);
    
    float d_min = top_radius - r;
    float d_max = rho + H;
    float d = d_min + x_mu * (d_max - d_min);
    
    float mu = (d == 0.0) ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
    mu = clamp(mu, -1.0, 1.0);
    
    float dist = DistanceToTopAtmosphereBoundary(r, mu);
    float dx = dist / float(SAMPLE_COUNT);
    
    // Use trapezoidal integration like reference
    vec3 optical_depth = vec3(0.0);
    
    for (int i = 0; i <= SAMPLE_COUNT; ++i) {
        float d_i = float(i) * dx;
        float r_i = sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r);
        float altitude = r_i - bottom_radius;
        
        float rayleigh_density = GetExpDensity(altitude, rayleigh_scale_height);
        float mie_density = GetExpDensity(altitude, mie_scale_height);
        float ozone_density = GetOzoneDensity(altitude);
        
        // Trapezoidal weight
        float weight = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
        
        optical_depth += (
            rayleigh_scattering * rayleigh_density +
            mie_extinction * mie_density +
            absorption_extinction * ozone_density
        ) * weight * dx;
    }
    
    fragColor = vec4(exp(-optical_depth), 1.0);
}
''')
    
    return gpu.shader.create_from_info(shader_info)


def create_single_scattering_shader() -> 'gpu.types.GPUShader':
    """Create the single scattering precomputation shader (one depth slice)."""
    shader_info = gpu.types.GPUShaderCreateInfo()
    
    shader_info.vertex_in(0, 'VEC2', 'pos')
    
    vert_out = gpu.types.GPUStageInterfaceInfo("vert_iface")
    vert_out.smooth('VEC2', 'uv')
    shader_info.vertex_out(vert_out)
    
    # Uniforms - reduced to fit within 128 byte push constant limit
    # Constants (rayleigh_scattering, mie_scattering, solar_irradiance) are now hardcoded in shader
    shader_info.push_constant('FLOAT', 'bottom_radius')
    shader_info.push_constant('FLOAT', 'top_radius')
    shader_info.push_constant('FLOAT', 'rayleigh_scale_height')
    shader_info.push_constant('FLOAT', 'mie_scale_height')
    shader_info.push_constant('INT', 'current_layer')
    
    shader_info.sampler(0, 'FLOAT_2D', 'transmittance_texture')
    
    shader_info.fragment_out(0, 'VEC4', 'fragColor')
    
    shader_info.vertex_source('''
void main() {
    uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
''')
    
    shader_info.fragment_source('''
#define TRANSMITTANCE_TEXTURE_WIDTH 256
#define TRANSMITTANCE_TEXTURE_HEIGHT 64
#define SCATTERING_TEXTURE_R_SIZE 32
#define SCATTERING_TEXTURE_MU_SIZE 128
#define SCATTERING_TEXTURE_MU_S_SIZE 32
#define SCATTERING_TEXTURE_NU_SIZE 8
#define SCATTERING_TEXTURE_WIDTH 256
#define SCATTERING_TEXTURE_HEIGHT 128
#define SAMPLE_COUNT 50
#define SUN_ANGULAR_RADIUS 0.004675

// Hardcoded constants to avoid push constant overflow
// Rayleigh scattering coefficients (m^-1) at 680nm, 550nm, 440nm
const vec3 rayleigh_scattering = vec3(5.802e-6, 13.558e-6, 33.1e-6);
// Mie scattering coefficient (m^-1)
const vec3 mie_scattering = vec3(4.0e-6, 4.0e-6, 4.0e-6);
// Mie extinction coefficient (m^-1)
const vec3 mie_extinction = vec3(4.44e-6, 4.44e-6, 4.44e-6);
// Solar irradiance (W/m^2/nm) at 680nm, 550nm, 440nm
const vec3 solar_irradiance = vec3(1.474, 1.8504, 1.91198);

float SafeSqrt(float x) { return sqrt(max(x, 0.0)); }

float GetUnitRangeFromTextureCoord(float u, float tex_size) {
    return (u - 0.5 / tex_size) / (1.0 - 1.0 / tex_size);
}

float GetTextureCoordFromUnitRange(float x, float tex_size) {
    return 0.5 / tex_size + x * (1.0 - 1.0 / tex_size);
}

float DistanceToTop(float r, float mu) {
    float disc = r * r * (mu * mu - 1.0) + top_radius * top_radius;
    return max(0.0, -r * mu + SafeSqrt(disc));
}

float DistanceToBottom(float r, float mu) {
    float disc = r * r * (mu * mu - 1.0) + bottom_radius * bottom_radius;
    return max(0.0, -r * mu - SafeSqrt(disc));
}

float GetExpDensity(float altitude, float scale_height) {
    return exp(-altitude / scale_height);
}

vec3 GetTransmittanceToTopBoundary(float r, float mu) {
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    float rho = SafeSqrt(r * r - bottom_radius * bottom_radius);
    float d = DistanceToTop(r, mu);
    float d_min = top_radius - r;
    float d_max = rho + H;
    float x_mu = (d_max > d_min) ? (d - d_min) / (d_max - d_min) : 0.0;
    float x_r = rho / H;
    
    vec2 tex_uv = vec2(
        GetTextureCoordFromUnitRange(x_mu, float(TRANSMITTANCE_TEXTURE_WIDTH)),
        GetTextureCoordFromUnitRange(x_r, float(TRANSMITTANCE_TEXTURE_HEIGHT))
    );
    return texture(transmittance_texture, tex_uv).rgb;
}

vec3 GetTransmittanceToSun(float r, float mu_s) {
    float sin_theta_h = bottom_radius / r;
    float cos_theta_h = -sqrt(max(1.0 - sin_theta_h * sin_theta_h, 0.0));
    return GetTransmittanceToTopBoundary(r, mu_s) *
        smoothstep(-sin_theta_h * SUN_ANGULAR_RADIUS, sin_theta_h * SUN_ANGULAR_RADIUS, mu_s - cos_theta_h);
}

vec3 GetTransmittance(float r, float mu, float d, bool intersects_ground) {
    float r_d = clamp(sqrt(d * d + 2.0 * r * mu * d + r * r), bottom_radius, top_radius);
    float mu_d = clamp((r * mu + d) / r_d, -1.0, 1.0);
    
    if (intersects_ground) {
        return min(GetTransmittanceToTopBoundary(r_d, -mu_d) / GetTransmittanceToTopBoundary(r, -mu), vec3(1.0));
    } else {
        return min(GetTransmittanceToTopBoundary(r, mu) / GetTransmittanceToTopBoundary(r_d, mu_d), vec3(1.0));
    }
}

void main() {
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    
    // Decode r from layer
    float z = (float(current_layer) + 0.5) / float(SCATTERING_TEXTURE_R_SIZE);
    float x_r = GetUnitRangeFromTextureCoord(z, float(SCATTERING_TEXTURE_R_SIZE));
    float rho = H * x_r;
    float r = sqrt(rho * rho + bottom_radius * bottom_radius);
    
    // Decode mu from y coordinate
    float w = uv.y;
    float mu;
    bool ray_intersects_ground;
    
    if (w < 0.5) {
        float d_min = r - bottom_radius;
        float d_max = rho;
        float x = GetUnitRangeFromTextureCoord(1.0 - 2.0 * w, float(SCATTERING_TEXTURE_MU_SIZE) / 2.0);
        float d = d_min + (d_max - d_min) * x;
        mu = (d == 0.0) ? -1.0 : -(rho * rho + d * d) / (2.0 * r * d);
        ray_intersects_ground = true;
    } else {
        float d_min = top_radius - r;
        float d_max = rho + H;
        float x = GetUnitRangeFromTextureCoord(2.0 * w - 1.0, float(SCATTERING_TEXTURE_MU_SIZE) / 2.0);
        float d = d_min + (d_max - d_min) * x;
        mu = (d == 0.0) ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
        ray_intersects_ground = false;
    }
    mu = clamp(mu, -1.0, 1.0);
    
    // Decode mu_s and nu from x coordinate
    // CPU uses: nu_idx = ii // MU_S_SIZE, mu_s_idx = ii % MU_S_SIZE
    int tex_x = int(uv.x * float(SCATTERING_TEXTURE_WIDTH));
    int mu_s_idx = tex_x % SCATTERING_TEXTURE_MU_S_SIZE;
    int nu_idx = tex_x / SCATTERING_TEXTURE_MU_S_SIZE;
    
    float v_s = (float(mu_s_idx) + 0.5) / float(SCATTERING_TEXTURE_MU_S_SIZE);
    float x_mu_s = GetUnitRangeFromTextureCoord(v_s, float(SCATTERING_TEXTURE_MU_S_SIZE));
    float d_min_s = top_radius - bottom_radius;
    float d_max_s = H;
    float D = DistanceToTop(bottom_radius, -0.2);
    float A = (D - d_min_s) / (d_max_s - d_min_s);
    float a = (1.0 + x_mu_s * A != 0.0) ? (A - x_mu_s * A) / (1.0 + x_mu_s * A) : 0.0;
    float d_s = d_min_s + min(a, A) * (d_max_s - d_min_s);
    float mu_s = (d_s == 0.0) ? 1.0 : (H * H - d_s * d_s) / (2.0 * bottom_radius * d_s);
    mu_s = clamp(mu_s, -1.0, 1.0);
    
    // Reference uses: nu = clamp(frag_coord_nu / (NU_SIZE - 1) * 2.0 - 1.0, -1, 1)
    float nu = clamp(float(nu_idx) / float(SCATTERING_TEXTURE_NU_SIZE - 1) * 2.0 - 1.0, -1.0, 1.0);
    
    // Distance to boundary
    float d_max_ray = ray_intersects_ground ? DistanceToBottom(r, mu) : DistanceToTop(r, mu);
    float dx = d_max_ray / float(SAMPLE_COUNT);
    
    // Integrate along ray using trapezoidal rule
    vec3 rayleigh_sum = vec3(0.0);
    vec3 mie_sum = vec3(0.0);
    
    for (int i = 0; i <= SAMPLE_COUNT; ++i) {
        float d_i = float(i) * dx;
        float r_i = clamp(sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r), bottom_radius, top_radius);
        float altitude = r_i - bottom_radius;
        
        // Sun zenith cosine at sample point
        float mu_s_i = clamp((r * mu_s + d_i * nu) / r_i, -1.0, 1.0);
        
        // Transmittance from p to sample, and from sample to sun
        vec3 trans_to_i = GetTransmittance(r, mu, d_i, ray_intersects_ground);
        vec3 trans_to_sun = GetTransmittanceToSun(r_i, mu_s_i);
        vec3 trans = trans_to_i * trans_to_sun;
        
        float rayleigh_density = GetExpDensity(altitude, rayleigh_scale_height);
        float mie_density = GetExpDensity(altitude, mie_scale_height);
        
        // Trapezoidal weight
        float weight = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
        
        rayleigh_sum += trans * rayleigh_density * weight;
        mie_sum += trans * mie_density * weight;
    }
    
    // Multiply by step size, solar irradiance, and scattering coefficients
    rayleigh_sum *= dx * solar_irradiance * rayleigh_scattering;
    mie_sum *= dx * solar_irradiance * mie_scattering;
    
    // Pack: RGB = Rayleigh, A = Mie.r
    fragColor = vec4(rayleigh_sum, mie_sum.r);
}
''')
    
    return gpu.shader.create_from_info(shader_info)


def create_direct_irradiance_shader() -> 'gpu.types.GPUShader':
    """Create the direct irradiance precomputation shader."""
    shader_info = gpu.types.GPUShaderCreateInfo()
    
    shader_info.vertex_in(0, 'VEC2', 'pos')
    
    vert_out = gpu.types.GPUStageInterfaceInfo("vert_iface")
    vert_out.smooth('VEC2', 'uv')
    shader_info.vertex_out(vert_out)
    
    shader_info.push_constant('FLOAT', 'bottom_radius')
    shader_info.push_constant('FLOAT', 'top_radius')
    shader_info.push_constant('VEC3', 'solar_irradiance')
    shader_info.push_constant('FLOAT', 'sun_angular_radius')
    
    shader_info.sampler(0, 'FLOAT_2D', 'transmittance_texture')
    
    shader_info.fragment_out(0, 'VEC4', 'fragColor')
    
    shader_info.vertex_source('''
void main() {
    uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
''')
    
    shader_info.fragment_source('''
#define IRRADIANCE_TEXTURE_WIDTH 64
#define IRRADIANCE_TEXTURE_HEIGHT 16
#define TRANSMITTANCE_TEXTURE_WIDTH 256
#define TRANSMITTANCE_TEXTURE_HEIGHT 64

float SafeSqrt(float x) { return sqrt(max(x, 0.0)); }

float GetUnitRangeFromTextureCoord(float u, float tex_size) {
    return (u - 0.5 / tex_size) / (1.0 - 1.0 / tex_size);
}

float GetTextureCoordFromUnitRange(float x, float tex_size) {
    return 0.5 / tex_size + x * (1.0 - 1.0 / tex_size);
}

float DistanceToTop(float r, float mu) {
    float disc = r * r * (mu * mu - 1.0) + top_radius * top_radius;
    return max(0.0, -r * mu + SafeSqrt(disc));
}

vec3 GetTransmittanceToSun(float r, float mu_s) {
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    float rho = SafeSqrt(r * r - bottom_radius * bottom_radius);
    float d = DistanceToTop(r, mu_s);
    float d_min = top_radius - r;
    float d_max = rho + H;
    float x_mu = (d_max > d_min) ? (d - d_min) / (d_max - d_min) : 0.0;
    float x_r = rho / H;
    
    vec2 tex_uv = vec2(
        GetTextureCoordFromUnitRange(x_mu, float(TRANSMITTANCE_TEXTURE_WIDTH)),
        GetTextureCoordFromUnitRange(x_r, float(TRANSMITTANCE_TEXTURE_HEIGHT))
    );
    
    vec3 trans = texture(transmittance_texture, tex_uv).rgb;
    
    float sin_h = bottom_radius / r;
    float cos_h = -sqrt(max(1.0 - sin_h * sin_h, 0.0));
    return trans * smoothstep(-sin_h * sun_angular_radius, sin_h * sun_angular_radius, mu_s - cos_h);
}

void main() {
    // Reference uses: frag_coord / IRRADIANCE_TEXTURE_SIZE then GetRMuSFromIrradianceTextureUv
    vec2 frag_uv = gl_FragCoord.xy / vec2(float(IRRADIANCE_TEXTURE_WIDTH), float(IRRADIANCE_TEXTURE_HEIGHT));
    
    float x_mu_s = GetUnitRangeFromTextureCoord(frag_uv.x, float(IRRADIANCE_TEXTURE_WIDTH));
    float x_r = GetUnitRangeFromTextureCoord(frag_uv.y, float(IRRADIANCE_TEXTURE_HEIGHT));
    
    float r = bottom_radius + x_r * (top_radius - bottom_radius);
    float mu_s = clamp(2.0 * x_mu_s - 1.0, -1.0, 1.0);  // Reference range [-1, 1]
    
    // Compute transmittance to top of atmosphere
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    float rho = SafeSqrt(r * r - bottom_radius * bottom_radius);
    float d = DistanceToTop(r, mu_s);
    float d_min = top_radius - r;
    float d_max = rho + H;
    float x_mu = (d_max > d_min) ? (d - d_min) / (d_max - d_min) : 0.0;
    float x_r_trans = rho / H;
    
    vec2 tex_uv = vec2(
        GetTextureCoordFromUnitRange(x_mu, float(TRANSMITTANCE_TEXTURE_WIDTH)),
        GetTextureCoordFromUnitRange(x_r_trans, float(TRANSMITTANCE_TEXTURE_HEIGHT))
    );
    vec3 trans = texture(transmittance_texture, tex_uv).rgb;
    
    // Reference average_cosine_factor formula
    float alpha_s = sun_angular_radius;
    float average_cosine_factor = (mu_s < -alpha_s) ? 0.0 : 
        ((mu_s > alpha_s) ? mu_s : (mu_s + alpha_s) * (mu_s + alpha_s) / (4.0 * alpha_s));
    
    // Direct irradiance goes to delta_irradiance (computed here)
    vec3 delta_irradiance = solar_irradiance * trans * average_cosine_factor;
    
    // IMPORTANT: Reference kComputeDirectIrradianceShader writes irradiance = vec3(0.0)
    // The irradiance texture accumulates INDIRECT irradiance only (from multiple scattering)
    // Direct irradiance is used for scattering density computation but not stored in irradiance texture
    // Output: R,G,B = delta_irradiance (for scattering density), A = 0 (placeholder for irradiance)
    fragColor = vec4(delta_irradiance, 0.0);
}
''')
    
    return gpu.shader.create_from_info(shader_info)


def create_scattering_density_shader():
    """
    Create shader for computing scattering density (first step of multiple scattering).
    
    This computes the radiance scattered at each point towards each direction,
    by integrating over all incident directions.
    """
    shader_info = gpu.types.GPUShaderCreateInfo()
    
    shader_info.vertex_in(0, 'VEC2', 'pos')
    
    vert_out = gpu.types.GPUStageInterfaceInfo("sd_vert_iface")
    vert_out.smooth('VEC2', 'uv')
    shader_info.vertex_out(vert_out)
    
    # Uniforms - atmosphere parameters
    shader_info.push_constant('FLOAT', 'bottom_radius')
    shader_info.push_constant('FLOAT', 'top_radius')
    shader_info.push_constant('VEC3', 'rayleigh_scattering')
    shader_info.push_constant('VEC3', 'mie_scattering')
    shader_info.push_constant('FLOAT', 'mie_phase_g')
    shader_info.push_constant('VEC3', 'ground_albedo')
    shader_info.push_constant('INT', 'current_layer')
    shader_info.push_constant('INT', 'scattering_order')
    
    # Input textures
    shader_info.sampler(0, 'FLOAT_2D', 'transmittance_texture')
    shader_info.sampler(1, 'FLOAT_2D', 'single_rayleigh_scattering_texture')
    shader_info.sampler(2, 'FLOAT_2D', 'single_mie_scattering_texture')
    shader_info.sampler(3, 'FLOAT_2D', 'multiple_scattering_texture')
    shader_info.sampler(4, 'FLOAT_2D', 'irradiance_texture')
    
    shader_info.fragment_out(0, 'VEC4', 'fragColor')
    
    shader_info.vertex_source('''
void main() {
    uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
''')
    
    shader_info.fragment_source('''
#define PI 3.14159265358979323846
#define TRANSMITTANCE_TEXTURE_WIDTH 256
#define TRANSMITTANCE_TEXTURE_HEIGHT 64
#define SCATTERING_TEXTURE_R_SIZE 32
#define SCATTERING_TEXTURE_MU_SIZE 128
#define SCATTERING_TEXTURE_MU_S_SIZE 32
#define SCATTERING_TEXTURE_NU_SIZE 8
#define SCATTERING_TEXTURE_WIDTH 256
#define SCATTERING_TEXTURE_HEIGHT 128
#define SCATTERING_TEXTURE_DEPTH 32
#define IRRADIANCE_TEXTURE_WIDTH 64
#define IRRADIANCE_TEXTURE_HEIGHT 16
#define SAMPLE_COUNT 16

float SafeSqrt(float x) { return sqrt(max(x, 0.0)); }

float GetUnitRangeFromTextureCoord(float u, float tex_size) {
    return (u - 0.5 / tex_size) / (1.0 - 1.0 / tex_size);
}

float GetTextureCoordFromUnitRange(float x, float tex_size) {
    return 0.5 / tex_size + x * (1.0 - 1.0 / tex_size);
}

float ClampRadius(float r) {
    return clamp(r, bottom_radius, top_radius);
}

float ClampCosine(float mu) {
    return clamp(mu, -1.0, 1.0);
}

float DistanceToTop(float r, float mu) {
    float disc = r * r * (mu * mu - 1.0) + top_radius * top_radius;
    return max(0.0, -r * mu + SafeSqrt(disc));
}

float DistanceToBottom(float r, float mu) {
    float disc = r * r * (mu * mu - 1.0) + bottom_radius * bottom_radius;
    return max(0.0, -r * mu - SafeSqrt(disc));
}

bool RayIntersectsGround(float r, float mu) {
    return (mu < 0.0) && (r * r * (mu * mu - 1.0) + bottom_radius * bottom_radius >= 0.0);
}

float RayleighPhaseFunction(float nu) {
    return (3.0 / (16.0 * PI)) * (1.0 + nu * nu);
}

float MiePhaseFunction(float g, float nu) {
    float g2 = g * g;
    return (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + nu * nu)) /
           ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * nu, 1.5));
}

vec3 GetTransmittanceToTopBoundary(float r, float mu) {
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    float rho = SafeSqrt(r * r - bottom_radius * bottom_radius);
    float d = DistanceToTop(r, mu);
    float d_min = top_radius - r;
    float d_max = rho + H;
    float x_mu = (d_max > d_min) ? (d - d_min) / (d_max - d_min) : 0.0;
    float x_r = rho / H;
    
    vec2 tex_uv = vec2(
        GetTextureCoordFromUnitRange(x_mu, float(TRANSMITTANCE_TEXTURE_WIDTH)),
        GetTextureCoordFromUnitRange(x_r, float(TRANSMITTANCE_TEXTURE_HEIGHT))
    );
    return texture(transmittance_texture, tex_uv).rgb;
}

vec3 GetTransmittance(float r, float mu, float d, bool intersects_ground) {
    float r_d = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
    float mu_d = ClampCosine((r * mu + d) / r_d);
    
    if (intersects_ground) {
        return min(GetTransmittanceToTopBoundary(r_d, -mu_d) / 
                   max(GetTransmittanceToTopBoundary(r, -mu), vec3(1e-10)), vec3(1.0));
    } else {
        return min(GetTransmittanceToTopBoundary(r, mu) / 
                   max(GetTransmittanceToTopBoundary(r_d, mu_d), vec3(1e-10)), vec3(1.0));
    }
}

// Sample scattering from tiled 2D texture
vec4 SampleScattering4D(sampler2D tex, float r, float mu, float mu_s, float nu, bool ray_intersects_ground) {
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    float rho = SafeSqrt(r * r - bottom_radius * bottom_radius);
    float u_r = GetTextureCoordFromUnitRange(rho / H, float(SCATTERING_TEXTURE_R_SIZE));
    
    float r_mu = r * mu;
    float discriminant = r_mu * r_mu - r * r + bottom_radius * bottom_radius;
    float u_mu;
    if (ray_intersects_ground) {
        float d = -r_mu - SafeSqrt(discriminant);
        float d_min = r - bottom_radius;
        float d_max = rho;
        u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(
            (d_max > d_min) ? (d - d_min) / (d_max - d_min) : 0.0,
            float(SCATTERING_TEXTURE_MU_SIZE) / 2.0);
    } else {
        float d = -r_mu + SafeSqrt(discriminant + H * H);
        float d_min = top_radius - r;
        float d_max = rho + H;
        u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
            (d_max > d_min) ? (d - d_min) / (d_max - d_min) : 0.0,
            float(SCATTERING_TEXTURE_MU_SIZE) / 2.0);
    }
    
    float d = DistanceToTop(bottom_radius, mu_s);
    float d_min = top_radius - bottom_radius;
    float d_max = H;
    float a = (d - d_min) / (d_max - d_min);
    float D = DistanceToTop(bottom_radius, -0.2);  // mu_s_min
    float A = (D - d_min) / (d_max - d_min);
    float u_mu_s = GetTextureCoordFromUnitRange(max(1.0 - a / A, 0.0) / (1.0 + a), float(SCATTERING_TEXTURE_MU_S_SIZE));
    float u_nu = (nu + 1.0) / 2.0;
    
    // Sample from tiled 2D texture
    // The scattering texture is stored with depth slices tiled horizontally
    // Within each slice: x encodes nu (high bits) and mu_s (low bits)
    // x = nu_idx * MU_S_SIZE + mu_s_idx, so u_within_slice = u_nu + u_mu_s / NU_SIZE
    int layer = int(u_r * float(SCATTERING_TEXTURE_DEPTH));
    layer = clamp(layer, 0, SCATTERING_TEXTURE_DEPTH - 1);
    
    float u_within_slice = clamp(u_nu + u_mu_s / float(SCATTERING_TEXTURE_NU_SIZE), 0.0, 0.9999);
    float tex_x = (float(layer) + u_within_slice) / float(SCATTERING_TEXTURE_DEPTH);
    float tex_y = u_mu;
    
    return texture(tex, vec2(tex_x, tex_y));
}

vec3 SampleIrradiance(float r, float mu_s) {
    float x_r = (r - bottom_radius) / (top_radius - bottom_radius);
    float x_mu_s = mu_s * 0.5 + 0.5;
    float u = GetTextureCoordFromUnitRange(x_mu_s, float(IRRADIANCE_TEXTURE_WIDTH));
    float v = GetTextureCoordFromUnitRange(x_r, float(IRRADIANCE_TEXTURE_HEIGHT));
    return texture(irradiance_texture, vec2(u, v)).rgb;
}

void main() {
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    
    // Decode r from layer
    float z = (float(current_layer) + 0.5) / float(SCATTERING_TEXTURE_R_SIZE);
    float x_r = GetUnitRangeFromTextureCoord(z, float(SCATTERING_TEXTURE_R_SIZE));
    float rho = H * x_r;
    float r = sqrt(rho * rho + bottom_radius * bottom_radius);
    
    // Decode mu from y coordinate
    float w = uv.y;
    float mu;
    bool ray_intersects_ground;
    
    if (w < 0.5) {
        float d_min = r - bottom_radius;
        float d_max = rho;
        float x = GetUnitRangeFromTextureCoord(1.0 - 2.0 * w, float(SCATTERING_TEXTURE_MU_SIZE) / 2.0);
        float d = d_min + (d_max - d_min) * x;
        mu = (d == 0.0) ? -1.0 : -(rho * rho + d * d) / (2.0 * r * d);
        ray_intersects_ground = true;
    } else {
        float d_min = top_radius - r;
        float d_max = rho + H;
        float x = GetUnitRangeFromTextureCoord(2.0 * w - 1.0, float(SCATTERING_TEXTURE_MU_SIZE) / 2.0);
        float d = d_min + (d_max - d_min) * x;
        mu = (d == 0.0) ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
        ray_intersects_ground = false;
    }
    mu = ClampCosine(mu);
    
    // Decode mu_s and nu from x coordinate
    int tex_x = int(uv.x * float(SCATTERING_TEXTURE_WIDTH));
    int mu_s_idx = tex_x % SCATTERING_TEXTURE_MU_S_SIZE;
    int nu_idx = tex_x / SCATTERING_TEXTURE_MU_S_SIZE;
    
    float x_mu_s = GetUnitRangeFromTextureCoord(
        (float(mu_s_idx) + 0.5) / float(SCATTERING_TEXTURE_MU_S_SIZE),
        float(SCATTERING_TEXTURE_MU_S_SIZE)
    );
    
    float d_s = DistanceToTop(bottom_radius, -0.2);  // mu_s_min
    float d_s_min = top_radius - bottom_radius;
    float d_s_max = H;
    float A_s = (d_s - d_s_min) / (d_s_max - d_s_min);
    float a_s = (1.0 - x_mu_s * (1.0 + A_s)) / (1.0 + x_mu_s * A_s);
    float mu_s = ClampCosine(cos(acos((top_radius - bottom_radius) / H) * (1.0 - a_s)));
    
    // Reference uses: nu = clamp(frag_coord_nu / (NU_SIZE - 1) * 2.0 - 1.0, -1, 1)
    float nu = clamp(float(nu_idx) / float(SCATTERING_TEXTURE_NU_SIZE - 1) * 2.0 - 1.0, -1.0, 1.0);
    
    // Compute scattering density by integrating over hemisphere
    vec3 zenith_dir = vec3(0.0, 0.0, 1.0);
    vec3 omega = vec3(sqrt(max(1.0 - mu * mu, 0.0)), 0.0, mu);
    
    // Safe sun direction calculation - avoid division by very small numbers
    float sun_dir_x = 0.0;
    float sun_dir_y = 0.0;
    if (abs(omega.x) > 1e-6) {
        sun_dir_x = clamp((nu - mu * mu_s) / omega.x, -1.0, 1.0);
    }
    float sun_dir_y_sq = max(1.0 - sun_dir_x * sun_dir_x - mu_s * mu_s, 0.0);
    sun_dir_y = sqrt(sun_dir_y_sq);
    vec3 omega_s = vec3(sun_dir_x, sun_dir_y, mu_s);
    
    float dphi = PI / float(SAMPLE_COUNT);
    float dtheta = PI / float(SAMPLE_COUNT);
    vec3 rayleigh_mie = vec3(0.0);
    
    for (int l = 0; l < SAMPLE_COUNT; ++l) {
        float theta = (float(l) + 0.5) * dtheta;
        float cos_theta = cos(theta);
        float sin_theta = sin(theta);
        bool ray_theta_intersects_ground = RayIntersectsGround(r, cos_theta);
        
        float dist_to_ground = 0.0;
        vec3 trans_to_ground = vec3(0.0);
        if (ray_theta_intersects_ground) {
            dist_to_ground = DistanceToBottom(r, cos_theta);
            trans_to_ground = GetTransmittance(r, cos_theta, dist_to_ground, true);
        }
        
        for (int m = 0; m < 2 * SAMPLE_COUNT; ++m) {
            float phi = (float(m) + 0.5) * dphi;
            vec3 omega_i = vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
            float domega_i = dtheta * dphi * sin_theta;
            
            float nu1 = clamp(dot(omega_s, omega_i), -1.0, 1.0);
            
            // Get incident radiance from scattering textures
            vec3 incident_radiance;
            if (scattering_order == 2) {
                // For order 2, use single scattering
                vec4 rayleigh = SampleScattering4D(single_rayleigh_scattering_texture, 
                    r, omega_i.z, mu_s, nu1, ray_theta_intersects_ground);
                vec4 mie = SampleScattering4D(single_mie_scattering_texture,
                    r, omega_i.z, mu_s, nu1, ray_theta_intersects_ground);
                incident_radiance = rayleigh.rgb * RayleighPhaseFunction(nu1) +
                                    mie.rgb * MiePhaseFunction(mie_phase_g, nu1);
            } else {
                // For higher orders, use multiple scattering texture
                vec4 ms = SampleScattering4D(multiple_scattering_texture,
                    r, omega_i.z, mu_s, nu1, ray_theta_intersects_ground);
                incident_radiance = ms.rgb;
            }
            
            // Add ground contribution
            if (ray_theta_intersects_ground) {
                vec3 ground_pos = zenith_dir * r + omega_i * dist_to_ground;
                float ground_pos_len = length(ground_pos);
                if (ground_pos_len > 1e-6) {
                    vec3 ground_normal = ground_pos / ground_pos_len;
                    float ground_mu_s = clamp(dot(ground_normal, omega_s), -1.0, 1.0);
                    vec3 ground_irradiance = SampleIrradiance(bottom_radius, ground_mu_s);
                    incident_radiance += trans_to_ground * ground_albedo * (1.0 / PI) * ground_irradiance;
                }
            }
            
            float nu2 = clamp(dot(omega, omega_i), -1.0, 1.0);
            float altitude = max(r - bottom_radius, 0.0);
            float rayleigh_density = exp(-altitude / 8000.0);  // Rayleigh scale height
            float mie_density = exp(-altitude / 1200.0);       // Mie scale height
            
            rayleigh_mie += incident_radiance * (
                rayleigh_scattering * rayleigh_density * RayleighPhaseFunction(nu2) +
                mie_scattering * mie_density * MiePhaseFunction(mie_phase_g, nu2)
            ) * domega_i;
        }
    }
    
    fragColor = vec4(rayleigh_mie, 1.0);
}
''')
    
    return gpu.shader.create_from_info(shader_info)


def create_indirect_irradiance_shader():
    """
    Create shader for computing indirect ground irradiance.
    
    This integrates scattered light over the hemisphere to compute
    the irradiance received at the ground from scattered light.
    """
    shader_info = gpu.types.GPUShaderCreateInfo()
    
    shader_info.vertex_in(0, 'VEC2', 'pos')
    
    vert_out = gpu.types.GPUStageInterfaceInfo("ii_vert_iface")
    vert_out.smooth('VEC2', 'uv')
    shader_info.vertex_out(vert_out)
    
    shader_info.push_constant('FLOAT', 'bottom_radius')
    shader_info.push_constant('FLOAT', 'top_radius')
    shader_info.push_constant('FLOAT', 'mie_phase_g')
    shader_info.push_constant('INT', 'scattering_order')
    
    shader_info.sampler(0, 'FLOAT_2D', 'single_rayleigh_scattering_texture')
    shader_info.sampler(1, 'FLOAT_2D', 'single_mie_scattering_texture')
    shader_info.sampler(2, 'FLOAT_2D', 'multiple_scattering_texture')
    
    shader_info.fragment_out(0, 'VEC4', 'fragColor')
    
    shader_info.vertex_source('''
void main() {
    uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
''')
    
    shader_info.fragment_source('''
#define PI 3.14159265358979323846
#define SCATTERING_TEXTURE_R_SIZE 32
#define SCATTERING_TEXTURE_MU_SIZE 128
#define SCATTERING_TEXTURE_MU_S_SIZE 32
#define SCATTERING_TEXTURE_NU_SIZE 8
#define SCATTERING_TEXTURE_WIDTH 256
#define SCATTERING_TEXTURE_HEIGHT 128
#define SCATTERING_TEXTURE_DEPTH 32
#define IRRADIANCE_TEXTURE_WIDTH 64
#define IRRADIANCE_TEXTURE_HEIGHT 16
#define SAMPLE_COUNT 32

float SafeSqrt(float x) { return sqrt(max(x, 0.0)); }

float GetUnitRangeFromTextureCoord(float u, float tex_size) {
    return (u - 0.5 / tex_size) / (1.0 - 1.0 / tex_size);
}

float GetTextureCoordFromUnitRange(float x, float tex_size) {
    return 0.5 / tex_size + x * (1.0 - 1.0 / tex_size);
}

float ClampCosine(float mu) {
    return clamp(mu, -1.0, 1.0);
}

float DistanceToTop(float r, float mu) {
    float disc = r * r * (mu * mu - 1.0) + top_radius * top_radius;
    return max(0.0, -r * mu + SafeSqrt(disc));
}

float RayleighPhaseFunction(float nu) {
    return (3.0 / (16.0 * PI)) * (1.0 + nu * nu);
}

float MiePhaseFunction(float g, float nu) {
    float g2 = g * g;
    return (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + nu * nu)) /
           ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * nu, 1.5));
}

// Sample scattering from tiled 2D texture with proper interpolation
// Texture format: tiled by r (32 layers side by side), each 256x128
// Within each layer: x = nu_idx * MU_S_SIZE + mu_s_idx, y = mu
const float MU_S_MIN = -0.2;  // Same as reference
vec3 SampleScattering(sampler2D tex, float r, float mu, float mu_s, float nu) {
    // Return zero for sun below horizon (mu_s < mu_s_min)
    if (mu_s < MU_S_MIN) {
        return vec3(0.0);
    }
    
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    float rho = SafeSqrt(r * r - bottom_radius * bottom_radius);
    
    // 1. Compute continuous r coordinate and layer interpolation
    float u_r = GetTextureCoordFromUnitRange(rho / H, float(SCATTERING_TEXTURE_R_SIZE));
    float layer_coord = u_r * float(SCATTERING_TEXTURE_R_SIZE) - 0.5;
    int layer0 = int(floor(layer_coord));
    int layer1 = layer0 + 1;
    float layer_lerp = layer_coord - float(layer0);
    layer0 = clamp(layer0, 0, SCATTERING_TEXTURE_R_SIZE - 1);
    layer1 = clamp(layer1, 0, SCATTERING_TEXTURE_R_SIZE - 1);
    
    // 2. Compute u_mu (y coordinate) - for upward rays only
    float r_mu = r * mu;
    float discriminant = r_mu * r_mu - r * r + bottom_radius * bottom_radius;
    float d = -r_mu + SafeSqrt(discriminant + H * H);
    float d_min = top_radius - r;
    float d_max = rho + H;
    float u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
        (d_max > d_min) ? (d - d_min) / (d_max - d_min) : 0.0,
        float(SCATTERING_TEXTURE_MU_SIZE) / 2.0);
    
    // 3. Compute continuous mu_s coordinate (matching reference exactly)
    float d_s = DistanceToTop(bottom_radius, mu_s);
    float d_s_min = top_radius - bottom_radius;
    float d_s_max = H;
    float a = (d_s - d_s_min) / (d_s_max - d_s_min);
    float D = DistanceToTop(bottom_radius, MU_S_MIN);
    float A = (D - d_s_min) / (d_s_max - d_s_min);
    float x_mu_s = max(1.0 - a / A, 0.0) / (1.0 + a);
    float u_mu_s = GetTextureCoordFromUnitRange(x_mu_s, float(SCATTERING_TEXTURE_MU_S_SIZE));
    
    // 4. Compute continuous nu coordinate with interpolation
    float nu_clamped = clamp(nu, -1.0, 1.0);
    float u_nu = (nu_clamped + 1.0) * 0.5;  // Map [-1,1] to [0,1]
    float nu_coord = u_nu * float(SCATTERING_TEXTURE_NU_SIZE - 1);
    int nu0 = int(floor(nu_coord));
    int nu1 = nu0 + 1;
    float nu_lerp = nu_coord - float(nu0);
    nu0 = clamp(nu0, 0, SCATTERING_TEXTURE_NU_SIZE - 1);
    nu1 = clamp(nu1, 0, SCATTERING_TEXTURE_NU_SIZE - 1);
    
    // 5. Sample with bilinear filtering on (mu, mu_s) and manual lerp on (nu, r)
    // Each nu slice is MU_S_SIZE wide, mu_s varies within the slice
    float mu_s_in_slice = u_mu_s;  // Already in [0,1] range for the slice
    
    // Sample 4 points: (nu0,layer0), (nu1,layer0), (nu0,layer1), (nu1,layer1)
    vec3 s00 = texture(tex, vec2(
        (float(layer0) + (float(nu0) + mu_s_in_slice) / float(SCATTERING_TEXTURE_NU_SIZE)) / float(SCATTERING_TEXTURE_DEPTH),
        u_mu)).rgb;
    vec3 s10 = texture(tex, vec2(
        (float(layer0) + (float(nu1) + mu_s_in_slice) / float(SCATTERING_TEXTURE_NU_SIZE)) / float(SCATTERING_TEXTURE_DEPTH),
        u_mu)).rgb;
    vec3 s01 = texture(tex, vec2(
        (float(layer1) + (float(nu0) + mu_s_in_slice) / float(SCATTERING_TEXTURE_NU_SIZE)) / float(SCATTERING_TEXTURE_DEPTH),
        u_mu)).rgb;
    vec3 s11 = texture(tex, vec2(
        (float(layer1) + (float(nu1) + mu_s_in_slice) / float(SCATTERING_TEXTURE_NU_SIZE)) / float(SCATTERING_TEXTURE_DEPTH),
        u_mu)).rgb;
    
    // Bilinear interpolation across nu and r
    vec3 s0 = mix(s00, s10, nu_lerp);
    vec3 s1 = mix(s01, s11, nu_lerp);
    return mix(s0, s1, layer_lerp);
}

void main() {
    // Get r, mu_s from irradiance texture coordinates
    float x_mu_s = GetUnitRangeFromTextureCoord(uv.x, float(IRRADIANCE_TEXTURE_WIDTH));
    float x_r = GetUnitRangeFromTextureCoord(uv.y, float(IRRADIANCE_TEXTURE_HEIGHT));
    
    float r = bottom_radius + x_r * (top_radius - bottom_radius);
    float mu_s = ClampCosine(2.0 * x_mu_s - 1.0);
    
    // Sun direction
    vec3 omega_s = vec3(sqrt(1.0 - mu_s * mu_s), 0.0, mu_s);
    
    float dphi = PI / float(SAMPLE_COUNT);
    float dtheta = PI / float(SAMPLE_COUNT);
    vec3 result = vec3(0.0);
    
    // Integrate over upper hemisphere only (j < SAMPLE_COUNT/2)
    for (int j = 0; j < SAMPLE_COUNT / 2; ++j) {
        float theta = (float(j) + 0.5) * dtheta;
        float cos_theta = cos(theta);
        float sin_theta = sin(theta);
        
        for (int i = 0; i < 2 * SAMPLE_COUNT; ++i) {
            float phi = (float(i) + 0.5) * dphi;
            vec3 omega = vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
            float domega = dtheta * dphi * sin_theta;
            
            float nu = dot(omega, omega_s);
            
            vec3 scattering;
            if (scattering_order == 1) {
                vec3 rayleigh = SampleScattering(single_rayleigh_scattering_texture, r, omega.z, mu_s, nu);
                vec3 mie = SampleScattering(single_mie_scattering_texture, r, omega.z, mu_s, nu);
                scattering = rayleigh * RayleighPhaseFunction(nu) + mie * MiePhaseFunction(mie_phase_g, nu);
            } else {
                scattering = SampleScattering(multiple_scattering_texture, r, omega.z, mu_s, nu);
            }
            
            result += scattering * omega.z * domega;
        }
    }
    
    fragColor = vec4(result, 1.0);
}
''')
    
    return gpu.shader.create_from_info(shader_info)


def create_multiple_scattering_shader():
    """
    Create shader for computing multiple scattering (second step).
    
    This integrates the scattering density along rays to compute
    the radiance from multiple scattering.
    """
    shader_info = gpu.types.GPUShaderCreateInfo()
    
    shader_info.vertex_in(0, 'VEC2', 'pos')
    
    vert_out = gpu.types.GPUStageInterfaceInfo("ms_vert_iface")
    vert_out.smooth('VEC2', 'uv')
    shader_info.vertex_out(vert_out)
    
    shader_info.push_constant('FLOAT', 'bottom_radius')
    shader_info.push_constant('FLOAT', 'top_radius')
    shader_info.push_constant('INT', 'current_layer')
    
    shader_info.sampler(0, 'FLOAT_2D', 'transmittance_texture')
    shader_info.sampler(1, 'FLOAT_2D', 'scattering_density_texture')
    
    shader_info.fragment_out(0, 'VEC4', 'fragColor')
    
    shader_info.vertex_source('''
void main() {
    uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
''')
    
    shader_info.fragment_source('''
#define TRANSMITTANCE_TEXTURE_WIDTH 256
#define TRANSMITTANCE_TEXTURE_HEIGHT 64
#define SCATTERING_TEXTURE_R_SIZE 32
#define SCATTERING_TEXTURE_MU_SIZE 128
#define SCATTERING_TEXTURE_MU_S_SIZE 32
#define SCATTERING_TEXTURE_NU_SIZE 8
#define SCATTERING_TEXTURE_WIDTH 256
#define SCATTERING_TEXTURE_HEIGHT 128
#define SCATTERING_TEXTURE_DEPTH 32
#define SAMPLE_COUNT 50

float SafeSqrt(float x) { return sqrt(max(x, 0.0)); }

float GetUnitRangeFromTextureCoord(float u, float tex_size) {
    return (u - 0.5 / tex_size) / (1.0 - 1.0 / tex_size);
}

float GetTextureCoordFromUnitRange(float x, float tex_size) {
    return 0.5 / tex_size + x * (1.0 - 1.0 / tex_size);
}

float ClampRadius(float r) {
    return clamp(r, bottom_radius, top_radius);
}

float ClampCosine(float mu) {
    return clamp(mu, -1.0, 1.0);
}

float DistanceToTop(float r, float mu) {
    float disc = r * r * (mu * mu - 1.0) + top_radius * top_radius;
    return max(0.0, -r * mu + SafeSqrt(disc));
}

float DistanceToBottom(float r, float mu) {
    float disc = r * r * (mu * mu - 1.0) + bottom_radius * bottom_radius;
    return max(0.0, -r * mu - SafeSqrt(disc));
}

float DistanceToNearestBoundary(float r, float mu, bool ray_intersects_ground) {
    if (ray_intersects_ground) {
        return DistanceToBottom(r, mu);
    } else {
        return DistanceToTop(r, mu);
    }
}

vec3 GetTransmittanceToTopBoundary(float r, float mu) {
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    float rho = SafeSqrt(r * r - bottom_radius * bottom_radius);
    float d = DistanceToTop(r, mu);
    float d_min = top_radius - r;
    float d_max = rho + H;
    float x_mu = (d_max > d_min) ? (d - d_min) / (d_max - d_min) : 0.0;
    float x_r = rho / H;
    
    vec2 tex_uv = vec2(
        GetTextureCoordFromUnitRange(x_mu, float(TRANSMITTANCE_TEXTURE_WIDTH)),
        GetTextureCoordFromUnitRange(x_r, float(TRANSMITTANCE_TEXTURE_HEIGHT))
    );
    return texture(transmittance_texture, tex_uv).rgb;
}

vec3 GetTransmittance(float r, float mu, float d, bool intersects_ground) {
    float r_d = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
    float mu_d = ClampCosine((r * mu + d) / r_d);
    
    if (intersects_ground) {
        return min(GetTransmittanceToTopBoundary(r_d, -mu_d) / 
                   max(GetTransmittanceToTopBoundary(r, -mu), vec3(1e-10)), vec3(1.0));
    } else {
        return min(GetTransmittanceToTopBoundary(r, mu) / 
                   max(GetTransmittanceToTopBoundary(r_d, mu_d), vec3(1e-10)), vec3(1.0));
    }
}

// Sample scattering density from tiled 2D texture
vec3 SampleScatteringDensity(float r, float mu, float mu_s, float nu, bool ray_intersects_ground) {
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    float rho = SafeSqrt(r * r - bottom_radius * bottom_radius);
    float u_r = GetTextureCoordFromUnitRange(rho / H, float(SCATTERING_TEXTURE_R_SIZE));
    
    float r_mu = r * mu;
    float discriminant = r_mu * r_mu - r * r + bottom_radius * bottom_radius;
    float u_mu;
    if (ray_intersects_ground) {
        float d = -r_mu - SafeSqrt(discriminant);
        float d_min = r - bottom_radius;
        float d_max = rho;
        u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(
            (d_max > d_min) ? (d - d_min) / (d_max - d_min) : 0.0,
            float(SCATTERING_TEXTURE_MU_SIZE) / 2.0);
    } else {
        float d = -r_mu + SafeSqrt(discriminant + H * H);
        float d_min = top_radius - r;
        float d_max = rho + H;
        u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
            (d_max > d_min) ? (d - d_min) / (d_max - d_min) : 0.0,
            float(SCATTERING_TEXTURE_MU_SIZE) / 2.0);
    }
    
    float d = DistanceToTop(bottom_radius, mu_s);
    float d_min = top_radius - bottom_radius;
    float d_max = H;
    float a = (d - d_min) / (d_max - d_min);
    float D = DistanceToTop(bottom_radius, -0.2);
    float A = (D - d_min) / (d_max - d_min);
    float u_mu_s = GetTextureCoordFromUnitRange(max(1.0 - a / A, 0.0) / (1.0 + a), float(SCATTERING_TEXTURE_MU_S_SIZE));
    float u_nu = (nu + 1.0) / 2.0;
    
    int layer = int(u_r * float(SCATTERING_TEXTURE_DEPTH));
    layer = clamp(layer, 0, SCATTERING_TEXTURE_DEPTH - 1);
    
    // Within each slice: x encodes nu (high bits) and mu_s (low bits)
    float u_within_slice = clamp(u_nu + u_mu_s / float(SCATTERING_TEXTURE_NU_SIZE), 0.0, 0.9999);
    float tex_x = (float(layer) + u_within_slice) / float(SCATTERING_TEXTURE_DEPTH);
    float tex_y = u_mu;
    
    return texture(scattering_density_texture, vec2(tex_x, tex_y)).rgb;
}

void main() {
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    
    // Decode r from layer
    float z = (float(current_layer) + 0.5) / float(SCATTERING_TEXTURE_R_SIZE);
    float x_r = GetUnitRangeFromTextureCoord(z, float(SCATTERING_TEXTURE_R_SIZE));
    float rho = H * x_r;
    float r = sqrt(rho * rho + bottom_radius * bottom_radius);
    
    // Decode mu from y coordinate
    float w = uv.y;
    float mu;
    bool ray_intersects_ground;
    
    if (w < 0.5) {
        float d_min = r - bottom_radius;
        float d_max = rho;
        float x = GetUnitRangeFromTextureCoord(1.0 - 2.0 * w, float(SCATTERING_TEXTURE_MU_SIZE) / 2.0);
        float d = d_min + (d_max - d_min) * x;
        mu = (d == 0.0) ? -1.0 : -(rho * rho + d * d) / (2.0 * r * d);
        ray_intersects_ground = true;
    } else {
        float d_min = top_radius - r;
        float d_max = rho + H;
        float x = GetUnitRangeFromTextureCoord(2.0 * w - 1.0, float(SCATTERING_TEXTURE_MU_SIZE) / 2.0);
        float d = d_min + (d_max - d_min) * x;
        mu = (d == 0.0) ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
        ray_intersects_ground = false;
    }
    mu = ClampCosine(mu);
    
    // Decode mu_s and nu from x coordinate
    int tex_x = int(uv.x * float(SCATTERING_TEXTURE_WIDTH));
    int mu_s_idx = tex_x % SCATTERING_TEXTURE_MU_S_SIZE;
    int nu_idx = tex_x / SCATTERING_TEXTURE_MU_S_SIZE;
    
    float x_mu_s = GetUnitRangeFromTextureCoord(
        (float(mu_s_idx) + 0.5) / float(SCATTERING_TEXTURE_MU_S_SIZE),
        float(SCATTERING_TEXTURE_MU_S_SIZE)
    );
    
    float d_s = DistanceToTop(bottom_radius, -0.2);
    float d_s_min = top_radius - bottom_radius;
    float d_s_max = H;
    float A_s = (d_s - d_s_min) / (d_s_max - d_s_min);
    float a_s = (1.0 - x_mu_s * (1.0 + A_s)) / (1.0 + x_mu_s * A_s);
    float mu_s = ClampCosine(cos(acos((top_radius - bottom_radius) / H) * (1.0 - a_s)));
    
    // Reference uses: nu = clamp(frag_coord_nu / (NU_SIZE - 1) * 2.0 - 1.0, -1, 1)
    float nu = clamp(float(nu_idx) / float(SCATTERING_TEXTURE_NU_SIZE - 1) * 2.0 - 1.0, -1.0, 1.0);
    
    // Integration along the ray
    float dx = DistanceToNearestBoundary(r, mu, ray_intersects_ground) / float(SAMPLE_COUNT);
    vec3 rayleigh_mie_sum = vec3(0.0);
    
    for (int i = 0; i <= SAMPLE_COUNT; ++i) {
        float d_i = float(i) * dx;
        
        float r_i = ClampRadius(sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r));
        float mu_i = ClampCosine((r * mu + d_i) / r_i);
        float mu_s_i = ClampCosine((r * mu_s + d_i * nu) / r_i);
        
        vec3 scattering_density = SampleScatteringDensity(r_i, mu_i, mu_s_i, nu, ray_intersects_ground);
        vec3 transmittance = GetTransmittance(r, mu, d_i, ray_intersects_ground);
        
        float weight = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
        rayleigh_mie_sum += scattering_density * transmittance * weight * dx;
    }
    
    fragColor = vec4(rayleigh_mie_sum, 1.0);
}
''')
    
    return gpu.shader.create_from_info(shader_info)


class GPUPrecompute:
    """
    GPU-accelerated atmosphere LUT precomputation.
    
    Uses Blender's gpu module to render fragment shaders to offscreen buffers.
    """
    
    def __init__(self, params: AtmosphereParameters):
        self.params = params
        self._shaders = {}
        self._transmittance_texture = None
        self._delta_irradiance = None  # Direct irradiance for scattering density computation
        # Multiple scattering textures
        self._single_rayleigh_texture = None
        self._single_mie_texture = None
        self._multiple_scattering_texture = None
        self._irradiance_texture = None
        self._scattering_density_texture = None
        # Use raw spectral solar_irradiance values (reference uses these directly)
        print(f"  [DEBUG] solar_irradiance (raw): {params.solar_irradiance[:3]}")
    
    def _render_to_texture(self, shader: 'gpu.types.GPUShader', 
                           width: int, height: int,
                           uniforms: dict,
                           textures: dict = None) -> np.ndarray:
        """
        Render a shader to an offscreen buffer and read back pixels.
        
        Args:
            shader: Compiled GPU shader
            width, height: Output texture dimensions
            uniforms: Dict of uniform name -> value
            textures: Dict of sampler name -> GPUTexture (optional)
        """
        # Create offscreen buffer with float format for HDR values
        offscreen = gpu.types.GPUOffScreen(width, height, format='RGBA32F')
        
        # Full-screen quad vertices (CCW winding)
        vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        indices = [(0, 1, 2), (0, 2, 3)]
        
        batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)
        
        with offscreen.bind():
            # Set GPU state
            gpu.state.depth_test_set('NONE')
            gpu.state.blend_set('NONE')
            
            # Bind shader
            shader.bind()
            
            # Set uniforms
            for name, value in uniforms.items():
                try:
                    if isinstance(value, int):
                        shader.uniform_int(name, value)
                    elif isinstance(value, float):
                        shader.uniform_float(name, value)
                    elif isinstance(value, (tuple, list)):
                        shader.uniform_float(name, value)
                except (ValueError, TypeError) as e:
                    pass  # Uniform not found or wrong type
            
            # Bind textures
            if textures:
                for name, tex in textures.items():
                    try:
                        shader.uniform_sampler(name, tex)
                    except (ValueError, TypeError):
                        pass
            
            # Draw full-screen quad
            batch.draw(shader)
        
        # Read pixels from offscreen texture
        buffer = offscreen.texture_color.read()
        offscreen.free()
        
        # Convert buffer to numpy array
        # buffer.to_list() returns flat list of RGBA values
        pixels = np.array(buffer.to_list(), dtype=np.float32)
        pixels = pixels.reshape(height, width, 4)
        
        # Keep original orientation - Y-flip was wrong
        return pixels
    
    def _set_atmosphere_uniforms(self, shader: 'gpu.types.GPUShader'):
        """Set common atmosphere parameter uniforms."""
        p = self.params
        
        uniforms = {
            'bottom_radius': p.bottom_radius,
            'top_radius': p.top_radius,
            'rayleigh_scattering': tuple(p.rayleigh_scattering[:3]),
            'rayleigh_scale_height': p.rayleigh_density[0].exp_scale * -1.0 if p.rayleigh_density else 8000.0,
            'mie_scattering': tuple(p.mie_scattering[:3]),
            'mie_extinction': tuple(p.mie_extinction[:3]),
            'mie_scale_height': p.mie_density[0].exp_scale * -1.0 if p.mie_density else 1200.0,
            'mie_phase_g': p.mie_phase_function_g,
            'absorption_extinction': tuple(p.absorption_extinction[:3]),
            'ground_albedo': tuple(p.ground_albedo[:3]),
            'solar_irradiance': tuple(p.solar_irradiance[:3]),
            'sun_angular_radius': p.sun_angular_radius,
        }
        
        return uniforms
    
    def precompute_transmittance(self) -> np.ndarray:
        """Precompute transmittance LUT using GPU."""
        print("[Helios GPU] Computing transmittance...")
        
        shader = create_transmittance_shader()
        
        p = self.params
        # Get scale heights from density profile
        rayleigh_scale = 8000.0
        mie_scale = 1200.0
        if p.rayleigh_density and len(p.rayleigh_density) > 0:
            rayleigh_scale = -1.0 / p.rayleigh_density[0].exp_scale if p.rayleigh_density[0].exp_scale != 0 else 8000.0
        if p.mie_density and len(p.mie_density) > 0:
            mie_scale = -1.0 / p.mie_density[0].exp_scale if p.mie_density[0].exp_scale != 0 else 1200.0
        
        # Debug: print values being passed to shader
        print(f"  [DEBUG] bottom_radius: {p.bottom_radius}")
        print(f"  [DEBUG] top_radius: {p.top_radius}")
        print(f"  [DEBUG] rayleigh_scattering: {p.rayleigh_scattering[:3]}")
        print(f"  [DEBUG] rayleigh_scale_height: {rayleigh_scale}")
        print(f"  [DEBUG] mie_extinction: {p.mie_extinction[:3]}")
        print(f"  [DEBUG] mie_scale_height: {mie_scale}")
        print(f"  [DEBUG] absorption_extinction: {p.absorption_extinction[:3]}")
        
        uniforms = {
            'bottom_radius': float(p.bottom_radius),
            'top_radius': float(p.top_radius),
            'rayleigh_scattering': tuple(p.rayleigh_scattering[:3]),
            'rayleigh_scale_height': float(rayleigh_scale),
            'mie_extinction': tuple(p.mie_extinction[:3]),
            'mie_scale_height': float(mie_scale),
            'absorption_extinction': tuple(p.absorption_extinction[:3]),
        }
        
        pixels = self._render_to_texture(
            shader,
            TRANSMITTANCE_TEXTURE_WIDTH,
            TRANSMITTANCE_TEXTURE_HEIGHT,
            uniforms
        )
        
        # Store for use by other shaders
        self._transmittance_pixels = pixels[:, :, :3].copy()
        
        result = pixels[:, :, :3].astype(np.float32)
        print(f"  [DEBUG] Transmittance range: min={result.min():.6f}, max={result.max():.6f}")
        print(f"  [DEBUG] Transmittance center pixel: {result[32, 128, :]}")
        
        return result
    
    def _create_transmittance_texture(self, transmittance: np.ndarray):
        """Create a GPU texture from transmittance data for use in other shaders."""
        height, width = transmittance.shape[:2]
        
        # Pad to RGBA - no flip needed since we're keeping OpenGL orientation
        rgba = np.zeros((height, width, 4), dtype=np.float32)
        rgba[:, :, :3] = transmittance
        rgba[:, :, 3] = 1.0
        
        # Create GPU texture
        buf = gpu.types.Buffer('FLOAT', width * height * 4, rgba.flatten().tolist())
        self._transmittance_texture = gpu.types.GPUTexture(
            size=(width, height),
            format='RGBA32F',
            data=buf
        )
    
    def precompute_direct_irradiance(self, transmittance: np.ndarray) -> np.ndarray:
        """Precompute direct irradiance LUT using GPU."""
        print("[Helios GPU] Computing direct irradiance...")
        
        if self._transmittance_texture is None:
            self._create_transmittance_texture(transmittance)
        
        shader = create_direct_irradiance_shader()
        
        p = self.params
        uniforms = {
            'bottom_radius': float(p.bottom_radius),
            'top_radius': float(p.top_radius),
            'solar_irradiance': tuple(self.params.solar_irradiance[:3]),  # Raw spectral values
            'sun_angular_radius': float(p.sun_angular_radius),
        }
        
        textures = {'transmittance_texture': self._transmittance_texture}
        
        pixels = self._render_to_texture(
            shader,
            IRRADIANCE_TEXTURE_WIDTH,
            IRRADIANCE_TEXTURE_HEIGHT,
            uniforms,
            textures
        )
        
        # Shader outputs: RGB = delta_irradiance (direct), A = 0 (placeholder)
        delta_irradiance = pixels[:, :, :3].astype(np.float32)
        
        print(f"  [DEBUG] Delta irradiance (direct) range: min={delta_irradiance.min():.6f}, max={delta_irradiance.max():.6f}")
        print(f"  [DEBUG] Delta irradiance top-right pixel: {delta_irradiance[-1, -1, :]}")
        
        # Store delta_irradiance for use in multiple scattering
        self._delta_irradiance = delta_irradiance
        
        # Return delta_irradiance as the irradiance texture
        # Note: Reference stores indirect only, but irradiance_texture is not used in Helios OSL rendering
        # This stores direct irradiance for potential future use
        return delta_irradiance
    
    def precompute_single_scattering(self, transmittance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precompute single scattering LUT using GPU.
        
        Renders 32 passes (one per depth slice) to build the 3D texture.
        """
        print("[Helios GPU] Computing single scattering (32 GPU passes)...")
        
        if self._transmittance_texture is None:
            self._create_transmittance_texture(transmittance)
        
        shader = create_single_scattering_shader()
        
        p = self.params
        rayleigh_scale = 8000.0
        mie_scale = 1200.0
        if p.rayleigh_density and len(p.rayleigh_density) > 0:
            rayleigh_scale = -1.0 / p.rayleigh_density[0].exp_scale if p.rayleigh_density[0].exp_scale != 0 else 8000.0
        if p.mie_density and len(p.mie_density) > 0:
            mie_scale = -1.0 / p.mie_density[0].exp_scale if p.mie_density[0].exp_scale != 0 else 1200.0
        
        # Reduced uniforms - constants are now hardcoded in shader to avoid push constant overflow
        base_uniforms = {
            'bottom_radius': float(p.bottom_radius),
            'top_radius': float(p.top_radius),
            'rayleigh_scale_height': float(rayleigh_scale),
            'mie_scale_height': float(mie_scale),
        }
        
        textures = {'transmittance_texture': self._transmittance_texture}
        
        # Allocate output arrays
        scattering = np.zeros((SCATTERING_TEXTURE_DEPTH, SCATTERING_TEXTURE_HEIGHT, 
                               SCATTERING_TEXTURE_WIDTH, 4), dtype=np.float32)
        delta_mie = np.zeros((SCATTERING_TEXTURE_DEPTH, SCATTERING_TEXTURE_HEIGHT,
                              SCATTERING_TEXTURE_WIDTH, 3), dtype=np.float32)
        
        # Render each depth slice
        for layer in range(SCATTERING_TEXTURE_DEPTH):
            uniforms = base_uniforms.copy()
            uniforms['current_layer'] = layer
            
            pixels = self._render_to_texture(
                shader,
                SCATTERING_TEXTURE_WIDTH,
                SCATTERING_TEXTURE_HEIGHT,
                uniforms,
                textures
            )
            
            scattering[layer] = pixels.astype(np.float32)
            # Mie red channel is in alpha
            delta_mie[layer, :, :, 0] = pixels[:, :, 3]
            delta_mie[layer, :, :, 1] = pixels[:, :, 3]  # Approximate G/B from R
            delta_mie[layer, :, :, 2] = pixels[:, :, 3]
        
        print(f"  Single scattering complete ({SCATTERING_TEXTURE_DEPTH} slices)")
        print(f"  [DEBUG] Scattering range: min={scattering[:,:,:,:3].min():.6f}, max={scattering[:,:,:,:3].max():.6f}")
        
        # Detailed per-channel analysis at test point (layer=0, y=96, x=255 = nu7_mus31)
        test_pixel = scattering[0, 96, 255, :3]
        print(f"  [DEBUG] Test point (layer0, y96, nu7_mus31): R={test_pixel[0]:.6f}, G={test_pixel[1]:.6f}, B={test_pixel[2]:.6f}")
        print(f"  [DEBUG] Test point B/R ratio: {test_pixel[2]/max(test_pixel[0], 1e-10):.4f} (reference B/R=3.77)")
        
        # Per-channel max to see wavelength distribution
        r_max = scattering[:,:,:,0].max()
        g_max = scattering[:,:,:,1].max()
        b_max = scattering[:,:,:,2].max()
        print(f"  [DEBUG] Per-channel max: R={r_max:.6f}, G={g_max:.6f}, B={b_max:.6f}")
        print(f"  [DEBUG] Channel ratios: G/R={g_max/r_max:.4f}, B/R={b_max/r_max:.4f}")
        
        # Debug layer 0 (ground level) vs layer 1
        layer0_max = scattering[0, :, :, :3].max()
        layer1_max = scattering[1, :, :, :3].max()
        layer0_upper_half_max = scattering[0, 64:, :, :3].max()  # Upper half = upward rays
        layer0_lower_half_max = scattering[0, :64, :, :3].max()  # Lower half = downward rays
        print(f"  [DEBUG] Layer 0 (ground): max={layer0_max:.6f}, upper_half_max={layer0_upper_half_max:.6f}, lower_half_max={layer0_lower_half_max:.6f}")
        return scattering, delta_mie
    
    def _create_scattering_texture(self, scattering: np.ndarray):
        """
        Create a tiled 2D GPU texture from 3D scattering data.
        
        The 3D texture (D, H, W, 4) is stored as 2D (H, W*D, 4) with
        depth slices arranged horizontally.
        """
        depth, height, width, channels = scattering.shape
        
        # Create tiled 2D array
        tiled_width = width * depth
        tiled = np.zeros((height, tiled_width, 4), dtype=np.float32)
        
        for d in range(depth):
            x_start = d * width
            x_end = (d + 1) * width
            tiled[:, x_start:x_end, :] = scattering[d]
        
        # No flip needed - keep OpenGL orientation consistent with transmittance texture
        tiled = tiled.copy()
        
        buf = gpu.types.Buffer('FLOAT', height * tiled_width * 4, tiled.flatten().tolist())
        return gpu.types.GPUTexture(
            size=(tiled_width, height),
            format='RGBA32F',
            data=buf
        )
    
    def _create_scattering_textures(self, delta_rayleigh: np.ndarray, delta_mie: np.ndarray, 
                                     scattering: np.ndarray):
        """Create GPU textures for scattering data used in multiple scattering."""
        # Pad delta_rayleigh to RGBA
        delta_rayleigh_rgba = np.zeros((*delta_rayleigh.shape[:3], 4), dtype=np.float32)
        delta_rayleigh_rgba[:, :, :, :3] = delta_rayleigh
        delta_rayleigh_rgba[:, :, :, 3] = 1.0
        
        # Pad delta_mie to RGBA
        delta_mie_rgba = np.zeros((*delta_mie.shape[:3], 4), dtype=np.float32)
        delta_mie_rgba[:, :, :, :3] = delta_mie
        delta_mie_rgba[:, :, :, 3] = 1.0
        
        self._single_rayleigh_texture = self._create_scattering_texture(delta_rayleigh_rgba)
        self._single_mie_texture = self._create_scattering_texture(delta_mie_rgba)
        # Initialize multiple_scattering_texture with single scattering (for order 2 indirect irradiance)
        # This will be updated with delta_multiple_scattering after each order
        self._multiple_scattering_texture = self._create_scattering_texture(scattering)
    
    def _update_multiple_scattering_texture(self, delta_multiple_scattering: np.ndarray):
        """Update multiple scattering texture with DELTA (not accumulated) for next order's indirect irradiance."""
        # Pad to RGBA if needed
        if delta_multiple_scattering.shape[-1] == 3:
            delta_rgba = np.zeros((*delta_multiple_scattering.shape[:3], 4), dtype=np.float32)
            delta_rgba[:, :, :, :3] = delta_multiple_scattering
            delta_rgba[:, :, :, 3] = 1.0
        else:
            delta_rgba = delta_multiple_scattering
        
        self._multiple_scattering_texture = self._create_scattering_texture(delta_rgba)
    
    def _create_irradiance_texture(self, irradiance: np.ndarray):
        """Create GPU texture for irradiance data."""
        height, width = irradiance.shape[:2]
        
        # Pad to RGBA
        rgba = np.zeros((height, width, 4), dtype=np.float32)
        rgba[:, :, :3] = irradiance
        rgba[:, :, 3] = 1.0
        
        # No flip needed - keep OpenGL orientation consistent with transmittance texture
        rgba = rgba.copy()
        
        buf = gpu.types.Buffer('FLOAT', height * width * 4, rgba.flatten().tolist())
        self._irradiance_texture = gpu.types.GPUTexture(
            size=(width, height),
            format='RGBA32F',
            data=buf
        )
    
    def _create_scattering_density_texture(self, scattering_density: np.ndarray):
        """Create GPU texture for scattering density data."""
        # Pad to RGBA if needed
        if scattering_density.shape[-1] == 3:
            density_rgba = np.zeros((*scattering_density.shape[:3], 4), dtype=np.float32)
            density_rgba[:, :, :, :3] = scattering_density
            density_rgba[:, :, :, 3] = 1.0
        else:
            density_rgba = scattering_density
        
        self._scattering_density_texture = self._create_scattering_texture(density_rgba)
    
    def _compute_scattering_density(self, scattering_order: int) -> np.ndarray:
        """Compute scattering density using GPU shader."""
        shader = create_scattering_density_shader()
        
        p = self.params
        base_uniforms = {
            'bottom_radius': float(p.bottom_radius),
            'top_radius': float(p.top_radius),
            'rayleigh_scattering': tuple(p.rayleigh_scattering[:3]),
            'mie_scattering': tuple(p.mie_scattering[:3]),
            'mie_phase_g': float(p.mie_phase_function_g),
            'ground_albedo': tuple(p.ground_albedo[:3]) if hasattr(p.ground_albedo, '__len__') else (p.ground_albedo,) * 3,
            'scattering_order': scattering_order,
        }
        
        textures = {
            'transmittance_texture': self._transmittance_texture,
            'single_rayleigh_scattering_texture': self._single_rayleigh_texture,
            'single_mie_scattering_texture': self._single_mie_texture,
            'multiple_scattering_texture': self._multiple_scattering_texture,
            'irradiance_texture': self._irradiance_texture,
        }
        
        # Compute each layer
        scattering_density = np.zeros((SCATTERING_TEXTURE_DEPTH, SCATTERING_TEXTURE_HEIGHT,
                                       SCATTERING_TEXTURE_WIDTH, 3), dtype=np.float32)
        
        for layer in range(SCATTERING_TEXTURE_DEPTH):
            uniforms = {**base_uniforms, 'current_layer': layer}
            pixels = self._render_to_texture(
                shader,
                SCATTERING_TEXTURE_WIDTH,
                SCATTERING_TEXTURE_HEIGHT,
                uniforms,
                textures
            )
            scattering_density[layer] = pixels[:, :, :3].astype(np.float32)
        
        return scattering_density
    
    def _compute_indirect_irradiance(self, scattering_order: int) -> np.ndarray:
        """Compute indirect irradiance using GPU shader."""
        shader = create_indirect_irradiance_shader()
        
        p = self.params
        uniforms = {
            'bottom_radius': float(p.bottom_radius),
            'top_radius': float(p.top_radius),
            'mie_phase_g': float(p.mie_phase_function_g),
            'scattering_order': scattering_order,
        }
        
        textures = {
            'single_rayleigh_scattering_texture': self._single_rayleigh_texture,
            'single_mie_scattering_texture': self._single_mie_texture,
            'multiple_scattering_texture': self._multiple_scattering_texture,
        }
        
        pixels = self._render_to_texture(
            shader,
            IRRADIANCE_TEXTURE_WIDTH,
            IRRADIANCE_TEXTURE_HEIGHT,
            uniforms,
            textures
        )
        
        return pixels[:, :, :3].astype(np.float32)
    
    def _compute_multiple_scattering(self) -> np.ndarray:
        """Compute multiple scattering using GPU shader."""
        shader = create_multiple_scattering_shader()
        
        p = self.params
        base_uniforms = {
            'bottom_radius': float(p.bottom_radius),
            'top_radius': float(p.top_radius),
        }
        
        textures = {
            'transmittance_texture': self._transmittance_texture,
            'scattering_density_texture': self._scattering_density_texture,
        }
        
        # Compute each layer
        multiple_scattering = np.zeros((SCATTERING_TEXTURE_DEPTH, SCATTERING_TEXTURE_HEIGHT,
                                        SCATTERING_TEXTURE_WIDTH, 3), dtype=np.float32)
        
        for layer in range(SCATTERING_TEXTURE_DEPTH):
            uniforms = {**base_uniforms, 'current_layer': layer}
            pixels = self._render_to_texture(
                shader,
                SCATTERING_TEXTURE_WIDTH,
                SCATTERING_TEXTURE_HEIGHT,
                uniforms,
                textures
            )
            multiple_scattering[layer] = pixels[:, :, :3].astype(np.float32)
        
        return multiple_scattering
    
    def precompute(self, num_scattering_orders: int = 4, 
                   progress_callback=None) -> PrecomputedTextures:
        """
        Precompute all atmosphere LUTs using GPU acceleration.
        
        Args:
            num_scattering_orders: Number of scattering orders to compute (1-4)
            progress_callback: Optional callback(progress, message)
            
        Returns:
            PrecomputedTextures with all LUT data
        """
        import time
        
        if not IN_BLENDER:
            raise RuntimeError("GPU precomputation requires Blender's gpu module")
        
        total_start = time.perf_counter()
        
        if progress_callback:
            progress_callback(0.0, "Starting GPU precomputation...")
        
        # 1. Transmittance
        t0 = time.perf_counter()
        if progress_callback:
            progress_callback(0.1, "Computing transmittance (GPU)...")
        transmittance = self.precompute_transmittance()
        print(f"  Transmittance: {time.perf_counter() - t0:.2f}s")
        
        # 2. Direct irradiance (stored separately, NOT in irradiance texture)
        t0 = time.perf_counter()
        if progress_callback:
            progress_callback(0.15, "Computing direct irradiance (GPU)...")
        delta_irradiance_direct = self.precompute_direct_irradiance(transmittance)
        # Irradiance texture starts at 0 - it only stores INDIRECT irradiance
        # Direct irradiance is used for scattering density but not stored in texture
        irradiance = np.zeros((IRRADIANCE_TEXTURE_HEIGHT, IRRADIANCE_TEXTURE_WIDTH, 3), dtype=np.float32)
        print(f"  Direct irradiance: {time.perf_counter() - t0:.2f}s")
        
        # 3. Single scattering
        t0 = time.perf_counter()
        if progress_callback:
            progress_callback(0.2, "Computing single scattering (GPU)...")
        scattering, delta_mie = self.precompute_single_scattering(transmittance)
        delta_rayleigh = scattering[:, :, :, :3].copy()
        
        # Combine Rayleigh + Mie for the scattering texture
        # Reference formula: scattering = rayleigh + mie * (mie_scattering / mie_extinction)
        p = self.params
        mie_scattering = np.array(p.mie_scattering[:3], dtype=np.float32)
        mie_extinction = np.array(p.mie_extinction[:3], dtype=np.float32)
        mie_factor = mie_scattering / np.maximum(mie_extinction, 1e-10)
        print(f"  [DEBUG] Mie factor (scattering/extinction): {mie_factor}")
        # Add Mie contribution to scattering RGB
        scattering[:, :, :, :3] += delta_mie[:, :, :, :3] * mie_factor
        print(f"  [DEBUG] Combined scattering (Rayleigh+Mie): min={scattering[:,:,:,:3].min():.6f}, max={scattering[:,:,:,:3].max():.6f}")
        print(f"  Single scattering: {time.perf_counter() - t0:.2f}s")
        
        # 4. Multiple scattering orders (if requested)
        if num_scattering_orders > 1:
            print(f"[Helios GPU] Computing {num_scattering_orders - 1} additional scattering orders...")
            
            # Create textures for multiple scattering computation
            self._create_scattering_textures(delta_rayleigh, delta_mie, scattering)
            
            # IMPORTANT: For scattering density, we need delta_irradiance (not accumulated):
            # - Order 2: uses direct irradiance
            # - Order 3+: uses previous order's indirect irradiance
            # Initialize with direct irradiance for order 2's scattering density
            current_delta_irradiance = delta_irradiance_direct.copy()
            self._create_irradiance_texture(current_delta_irradiance)
            
            print(f"  [DEBUG] Delta irradiance for order 2 (direct): min={current_delta_irradiance.min():.6f}, max={current_delta_irradiance.max():.6f}")
            
            for order in range(2, num_scattering_orders + 1):
                t0 = time.perf_counter()
                progress = 0.2 + 0.7 * (order - 2) / max(1, num_scattering_orders - 1)
                
                if progress_callback:
                    progress_callback(progress, f"Computing scattering order {order} (GPU)...")
                
                # Step 1: Compute scattering density (uses current_delta_irradiance via irradiance_texture)
                scattering_density = self._compute_scattering_density(order)
                print(f"    [DEBUG] Order {order} scattering_density: min={scattering_density.min():.6f}, max={scattering_density.max():.6f}")
                
                # Step 2: Compute indirect irradiance for THIS order
                # Uses single scattering (order-1=1) or previous delta_multiple_scattering (order-1>1)
                # IMPORTANT: multiple_scattering_texture should contain DELTA, not accumulated!
                delta_irradiance = self._compute_indirect_irradiance(order - 1)
                print(f"    [DEBUG] Order {order} delta_irradiance: min={delta_irradiance.min():.6f}, max={delta_irradiance.max():.6f}")
                
                # Accumulate into final irradiance (only indirect, not direct)
                irradiance += delta_irradiance
                print(f"    [DEBUG] Order {order} irradiance after accumulation: min={irradiance.min():.6f}, max={irradiance.max():.6f}")
                
                # Update irradiance texture with THIS order's delta for NEXT order's scattering density
                current_delta_irradiance = delta_irradiance.copy()
                self._create_irradiance_texture(current_delta_irradiance)
                
                # Step 3: Compute multiple scattering and accumulate
                self._create_scattering_density_texture(scattering_density)
                delta_multiple_scattering = self._compute_multiple_scattering()
                print(f"    [DEBUG] Order {order} delta_multiple_scattering: min={delta_multiple_scattering.min():.6f}, max={delta_multiple_scattering.max():.6f}")
                
                # Accumulate into scattering texture (for final output)
                scattering[:, :, :, :3] += delta_multiple_scattering
                print(f"    [DEBUG] Order {order} scattering after accumulation: min={scattering[:,:,:,:3].min():.6f}, max={scattering[:,:,:,:3].max():.6f}")
                
                # Update multiple_scattering_texture with DELTA (not accumulated) for next order's indirect irradiance
                # This is critical - indirect irradiance should sample from delta, not accumulated
                self._update_multiple_scattering_texture(delta_multiple_scattering)
                
                print(f"  Order {order}: {time.perf_counter() - t0:.2f}s")
            
            print(f"  [DEBUG] Irradiance after MS loop (indirect only): min={irradiance.min():.6f}, max={irradiance.max():.6f}")
        
        total_time = time.perf_counter() - total_start
        print(f"[Helios GPU] Total precomputation time: {total_time:.2f}s")
        
        # Debug: Check for inf/nan in final outputs
        print(f"  [DEBUG] Final scattering: shape={scattering.shape}, min={scattering.min():.6f}, max={scattering.max():.6f}")
        print(f"  [DEBUG] Final scattering has inf: {np.any(np.isinf(scattering))}, has nan: {np.any(np.isnan(scattering))}")
        print(f"  [DEBUG] Final irradiance: shape={irradiance.shape}, min={irradiance.min():.6f}, max={irradiance.max():.6f}")
        print(f"  [DEBUG] Final irradiance has inf: {np.any(np.isinf(irradiance))}, has nan: {np.any(np.isnan(irradiance))}")
        
        if progress_callback:
            progress_callback(1.0, f"GPU precomputation complete ({total_time:.1f}s)")
        
        return PrecomputedTextures(
            transmittance=transmittance,
            scattering=scattering,
            irradiance=irradiance,
            single_mie_scattering=delta_mie
        )


class BlenderGPUAtmosphereModel:
    """
    Atmosphere model using Blender's built-in GPU for precomputation.
    
    This is the preferred model when running inside Blender as it requires
    no external dependencies.
    """
    
    def __init__(self, params: Optional[AtmosphereParameters] = None):
        self.params = params or AtmosphereParameters.earth_default()
        self.textures: Optional[PrecomputedTextures] = None
        self._is_initialized = False
        self._solar_irradiance_rgb: Optional[np.ndarray] = None
        
        print("[Helios] Using Blender GPU backend (no external dependencies)")
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    def init(self, num_scattering_orders: int = 4, progress_callback=None) -> None:
        """Precompute atmosphere LUTs using GPU."""
        from .constants import convert_spectrum_to_linear_srgb
        
        # Compute solar irradiance RGB
        self._solar_irradiance_rgb = convert_spectrum_to_linear_srgb(
            self.params.wavelengths,
            self.params.solar_irradiance
        )
        
        # Run GPU precomputation
        gpu_precompute = GPUPrecompute(self.params)
        self.textures = gpu_precompute.precompute(
            num_scattering_orders=num_scattering_orders,
            progress_callback=progress_callback
        )
        
        self._is_initialized = True
        print("[Helios] GPU precomputation complete!")
    
    def get_shader_uniforms(self) -> dict:
        """Get dictionary of uniform values for shaders."""
        if not self._is_initialized:
            raise RuntimeError("Model not initialized.")
        
        p = self.params
        return {
            'bottom_radius': p.bottom_radius / p.length_unit_in_meters,
            'top_radius': p.top_radius / p.length_unit_in_meters,
            'rayleigh_scattering': p.rayleigh_scattering * p.length_unit_in_meters,
            'mie_scattering': p.mie_scattering * p.length_unit_in_meters,
            'mie_extinction': p.mie_extinction * p.length_unit_in_meters,
            'mie_phase_function_g': p.mie_phase_function_g,
            'absorption_extinction': p.absorption_extinction * p.length_unit_in_meters,
            'ground_albedo': p.ground_albedo,
            'sun_angular_radius': p.sun_angular_radius,
            'solar_irradiance': self._solar_irradiance_rgb,
        }
    
    def save_textures(self, filepath: str) -> None:
        """Save precomputed textures to NPZ file."""
        if not self._is_initialized:
            raise RuntimeError("Model not initialized.")
        
        np.savez_compressed(
            filepath,
            transmittance=self.textures.transmittance,
            scattering=self.textures.scattering,
            irradiance=self.textures.irradiance,
            single_mie=self.textures.single_mie_scattering
        )
    
    def save_textures_exr(self, output_dir: str) -> None:
        """Save precomputed textures as EXR files."""
        from .model import AtmosphereModel
        temp_model = AtmosphereModel(self.params)
        temp_model.textures = self.textures
        temp_model._is_initialized = True
        temp_model._solar_irradiance_rgb = self._solar_irradiance_rgb
        temp_model.save_textures_exr(output_dir)
