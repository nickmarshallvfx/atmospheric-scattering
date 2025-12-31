"""
GPU-accelerated LUT precomputation - Fresh implementation v2.

APPROACH: Direct, faithful port of Bruneton reference_functions.glsl
- No optimizations until baseline matches reference
- Each step verified against reference before proceeding
- Git commit at each verified milestone

VERSION: 2 - Transmittance + Single Scattering
"""
print("[Helios GPU v2] Module loaded - VERSION 2 (transmittance + single scattering)")

import os
import numpy as np
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

try:
    import bpy
    import gpu
    from gpu_extras.batch import batch_for_shader
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False
    print("[Helios GPU v2] Not running in Blender - GPU functions unavailable")

# =============================================================================
# TEXTURE DIMENSIONS - Match reference exactly
# =============================================================================
TRANSMITTANCE_TEXTURE_WIDTH = 256
TRANSMITTANCE_TEXTURE_HEIGHT = 64

SCATTERING_TEXTURE_R_SIZE = 32
SCATTERING_TEXTURE_MU_SIZE = 128
SCATTERING_TEXTURE_MU_S_SIZE = 32
SCATTERING_TEXTURE_NU_SIZE = 8
SCATTERING_TEXTURE_WIDTH = SCATTERING_TEXTURE_MU_S_SIZE * SCATTERING_TEXTURE_NU_SIZE  # 256
SCATTERING_TEXTURE_HEIGHT = SCATTERING_TEXTURE_MU_SIZE  # 128
SCATTERING_TEXTURE_DEPTH = SCATTERING_TEXTURE_R_SIZE  # 32

IRRADIANCE_TEXTURE_WIDTH = 64
IRRADIANCE_TEXTURE_HEIGHT = 16

# =============================================================================
# ATMOSPHERE PARAMETERS - Match reference demo.cc exactly
# =============================================================================
@dataclass
class AtmosphereParams:
    """Atmosphere parameters matching reference demo.cc"""
    # Radii in meters
    bottom_radius: float = 6360000.0
    top_radius: float = 6420000.0
    
    # Rayleigh scattering coefficients at RGB wavelengths (m^-1)
    # Computed as: kRayleigh * pow(lambda, -4) where lambda is in micrometers
    # kRayleigh = 1.24062e-6
    # Red (680nm=0.68um): 1.24062e-6 * 0.68^-4 = 5.802e-6
    # Green (550nm=0.55um): 1.24062e-6 * 0.55^-4 = 13.558e-6
    # Blue (440nm=0.44um): 1.24062e-6 * 0.44^-4 = 33.1e-6
    rayleigh_scattering: Tuple[float, float, float] = (5.802e-6, 13.558e-6, 33.1e-6)
    rayleigh_scale_height: float = 8000.0
    
    # Mie scattering/extinction coefficients (m^-1)
    # mie = kMieAngstromBeta / kMieScaleHeight * pow(lambda, -kMieAngstromAlpha)
    # With kMieAngstromAlpha=0, kMieAngstromBeta=5.328e-3, kMieScaleHeight=1200:
    # mie = 5.328e-3 / 1200 = 4.44e-6 (wavelength independent)
    # mie_scattering = mie * kMieSingleScatteringAlbedo = 4.44e-6 * 0.9 = 4.0e-6
    mie_scattering: Tuple[float, float, float] = (4.0e-6, 4.0e-6, 4.0e-6)
    mie_extinction: Tuple[float, float, float] = (4.44e-6, 4.44e-6, 4.44e-6)
    mie_scale_height: float = 1200.0
    mie_phase_g: float = 0.8
    
    # Ozone absorption extinction at RGB wavelengths (m^-1)
    # Computed from kMaxOzoneNumberDensity * kOzoneCrossSection[wavelength_index]
    # These values are for use_ozone=true
    absorption_extinction: Tuple[float, float, float] = (0.650e-6, 1.881e-6, 0.085e-6)
    
    # Solar irradiance at RGB wavelengths (W/m^2/nm)
    # From kSolarIrradiance array at indices for 680nm, 550nm, 440nm
    solar_irradiance: Tuple[float, float, float] = (1.474, 1.8504, 1.91198)
    
    # Sun angular radius
    sun_angular_radius: float = 0.004675  # radians
    
    # Ground albedo
    ground_albedo: float = 0.1


# =============================================================================
# TRANSMITTANCE SHADER - Direct port of reference
# =============================================================================
def create_transmittance_shader() -> 'gpu.types.GPUShader':
    """
    Create transmittance precomputation shader.
    
    Direct port of reference ComputeTransmittanceToTopAtmosphereBoundary.
    Uses GetRMuFromTransmittanceTextureUv for coordinate mapping.
    """
    shader_info = gpu.types.GPUShaderCreateInfo()
    
    shader_info.vertex_in(0, 'VEC2', 'pos')
    
    vert_out = gpu.types.GPUStageInterfaceInfo("vert_iface")
    vert_out.smooth('VEC2', 'uv')
    shader_info.vertex_out(vert_out)
    
    # Push constants for atmosphere parameters
    shader_info.push_constant('FLOAT', 'bottom_radius')
    shader_info.push_constant('FLOAT', 'top_radius')
    shader_info.push_constant('VEC3', 'rayleigh_scattering')
    shader_info.push_constant('FLOAT', 'rayleigh_scale_height')
    shader_info.push_constant('VEC3', 'mie_extinction')
    shader_info.push_constant('FLOAT', 'mie_scale_height')
    shader_info.push_constant('VEC3', 'absorption_extinction')
    
    shader_info.fragment_out(0, 'VEC4', 'fragColor')
    
    shader_info.vertex_source('''
void main() {
    uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
''')
    
    # Fragment shader - DIRECT PORT of reference functions.glsl
    shader_info.fragment_source('''
#define TRANSMITTANCE_TEXTURE_WIDTH 256
#define TRANSMITTANCE_TEXTURE_HEIGHT 64
#define SAMPLE_COUNT 500

// =============================================================================
// Utility functions - exact match to reference
// =============================================================================

float SafeSqrt(float x) {
    return sqrt(max(x, 0.0));
}

float ClampCosine(float mu) {
    return clamp(mu, -1.0, 1.0);
}

float ClampDistance(float d) {
    return max(d, 0.0);
}

float ClampRadius(float r) {
    return clamp(r, bottom_radius, top_radius);
}

// Reference: GetUnitRangeFromTextureCoord (line 346)
float GetUnitRangeFromTextureCoord(float u, float texture_size) {
    return (u - 0.5 / texture_size) / (1.0 - 1.0 / texture_size);
}

// Reference: DistanceToTopAtmosphereBoundary (line 207)
float DistanceToTopAtmosphereBoundary(float r, float mu) {
    float discriminant = r * r * (mu * mu - 1.0) + top_radius * top_radius;
    return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

// =============================================================================
// Density profiles - exact match to reference demo.cc
// =============================================================================

// Reference: GetLayerDensity (line 263)
// For Rayleigh: exp_term=1.0, exp_scale=-1/8000, linear_term=0, constant_term=0
// For Mie: exp_term=1.0, exp_scale=-1/1200, linear_term=0, constant_term=0
float GetRayleighDensity(float altitude) {
    float density = exp(-altitude / rayleigh_scale_height);
    return clamp(density, 0.0, 1.0);
}

float GetMieDensity(float altitude) {
    float density = exp(-altitude / mie_scale_height);
    return clamp(density, 0.0, 1.0);
}

// Ozone density profile from reference demo.cc (lines 246-250):
// Layer 0 (altitude < 25km): width=25000, linear_term=1/15000, constant_term=-2/3
// Layer 1 (altitude >= 25km): width=0, linear_term=-1/15000, constant_term=8/3
// Note: Layer 1 uses ABSOLUTE altitude in the linear term (not relative)
float GetOzoneDensity(float altitude) {
    float density;
    if (altitude < 25000.0) {
        // Layer 0: density = linear_term * altitude + constant_term
        density = altitude / 15000.0 - 2.0 / 3.0;
    } else {
        // Layer 1: density = linear_term * altitude + constant_term
        // Uses absolute altitude (not relative to layer boundary)
        density = -altitude / 15000.0 + 8.0 / 3.0;
    }
    return clamp(density, 0.0, 1.0);
}

// =============================================================================
// Main transmittance computation - reference lines 275-319
// =============================================================================

void main() {
    // Reference: GetRMuFromTransmittanceTextureUv (lines 427-447)
    float x_mu = GetUnitRangeFromTextureCoord(uv.x, float(TRANSMITTANCE_TEXTURE_WIDTH));
    float x_r = GetUnitRangeFromTextureCoord(uv.y, float(TRANSMITTANCE_TEXTURE_HEIGHT));
    
    // Distance to top atmosphere boundary for a horizontal ray at ground level
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    
    // Distance to the horizon, from which we can compute r
    float rho = H * x_r;
    float r = sqrt(rho * rho + bottom_radius * bottom_radius);
    
    // Distance to the top atmosphere boundary for the ray (r,mu)
    float d_min = top_radius - r;
    float d_max = rho + H;
    float d = d_min + x_mu * (d_max - d_min);
    
    // Recover mu from d
    float mu = (d == 0.0) ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
    mu = ClampCosine(mu);
    
    // Reference: ComputeOpticalLengthToTopAtmosphereBoundary (lines 275-298)
    // Compute for each density profile separately, then combine
    float dist = DistanceToTopAtmosphereBoundary(r, mu);
    float dx = dist / float(SAMPLE_COUNT);
    
    // Integration using trapezoidal rule
    float rayleigh_optical_length = 0.0;
    float mie_optical_length = 0.0;
    float ozone_optical_length = 0.0;
    
    for (int i = 0; i <= SAMPLE_COUNT; ++i) {
        float d_i = float(i) * dx;
        
        // Distance from current sample point to planet center
        float r_i = sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r);
        
        // Altitude above ground
        float altitude = r_i - bottom_radius;
        
        // Sample densities at this altitude
        float rayleigh_density = GetRayleighDensity(altitude);
        float mie_density = GetMieDensity(altitude);
        float ozone_density = GetOzoneDensity(altitude);
        
        // Trapezoidal weight
        float weight = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
        
        // Accumulate optical lengths
        rayleigh_optical_length += rayleigh_density * weight * dx;
        mie_optical_length += mie_density * weight * dx;
        ozone_optical_length += ozone_density * weight * dx;
    }
    
    // Reference: ComputeTransmittanceToTopAtmosphereBoundary (lines 306-319)
    // transmittance = exp(-(rayleigh_scattering * rayleigh_length + 
    //                       mie_extinction * mie_length +
    //                       absorption_extinction * ozone_length))
    vec3 optical_depth = rayleigh_scattering * rayleigh_optical_length +
                         mie_extinction * mie_optical_length +
                         absorption_extinction * ozone_optical_length;
    
    fragColor = vec4(exp(-optical_depth), 1.0);
}
''')
    
    return gpu.shader.create_from_info(shader_info)


# =============================================================================
# SINGLE SCATTERING SHADER - Direct port of reference
# =============================================================================
def create_single_scattering_shader() -> 'gpu.types.GPUShader':
    """
    Create single scattering precomputation shader.
    
    Direct port of reference ComputeSingleScatteringTexture.
    Renders one depth slice at a time (for 2D tiled texture output).
    """
    shader_info = gpu.types.GPUShaderCreateInfo()
    
    shader_info.vertex_in(0, 'VEC2', 'pos')
    
    vert_out = gpu.types.GPUStageInterfaceInfo("vert_iface")
    vert_out.smooth('VEC2', 'uv')
    shader_info.vertex_out(vert_out)
    
    # Push constants
    shader_info.push_constant('FLOAT', 'bottom_radius')
    shader_info.push_constant('FLOAT', 'top_radius')
    shader_info.push_constant('FLOAT', 'mu_s_min')
    shader_info.push_constant('INT', 'current_layer')
    
    # Transmittance texture sampler
    shader_info.sampler(0, 'FLOAT_2D', 'transmittance_texture')
    
    shader_info.fragment_out(0, 'VEC4', 'fragColor')
    
    shader_info.vertex_source('''
void main() {
    uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
''')
    
    # Fragment shader - DIRECT PORT of reference functions.glsl
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

// Atmosphere constants - hardcoded to match reference exactly
const float rayleigh_scale_height = 8000.0;
const float mie_scale_height = 1200.0;
const float sun_angular_radius = 0.004675;

// Scattering coefficients at RGB wavelengths (m^-1)
const vec3 rayleigh_scattering = vec3(5.802e-6, 13.558e-6, 33.1e-6);
const vec3 mie_scattering = vec3(4.0e-6, 4.0e-6, 4.0e-6);

// Solar irradiance at RGB wavelengths
const vec3 solar_irradiance = vec3(1.474, 1.8504, 1.91198);

// =============================================================================
// Utility functions - exact match to reference
// =============================================================================

float SafeSqrt(float x) {
    return sqrt(max(x, 0.0));
}

float ClampCosine(float mu) {
    return clamp(mu, -1.0, 1.0);
}

float ClampDistance(float d) {
    return max(d, 0.0);
}

float ClampRadius(float r) {
    return clamp(r, bottom_radius, top_radius);
}

float GetTextureCoordFromUnitRange(float x, float texture_size) {
    return 0.5 / texture_size + x * (1.0 - 1.0 / texture_size);
}

float GetUnitRangeFromTextureCoord(float u, float texture_size) {
    return (u - 0.5 / texture_size) / (1.0 - 1.0 / texture_size);
}

// Reference: DistanceToTopAtmosphereBoundary (line 207)
float DistanceToTopAtmosphereBoundary(float r, float mu) {
    float discriminant = r * r * (mu * mu - 1.0) + top_radius * top_radius;
    return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

// Reference: DistanceToBottomAtmosphereBoundary (line 222)
float DistanceToBottomAtmosphereBoundary(float r, float mu) {
    float discriminant = r * r * (mu * mu - 1.0) + bottom_radius * bottom_radius;
    return ClampDistance(-r * mu - SafeSqrt(discriminant));
}

// Reference: RayIntersectsGround (line 240)
bool RayIntersectsGround(float r, float mu) {
    return mu < 0.0 && r * r * (mu * mu - 1.0) + bottom_radius * bottom_radius >= 0.0;
}

// =============================================================================
// Density profiles - exact match to reference demo.cc
// =============================================================================

float GetRayleighDensity(float altitude) {
    return clamp(exp(-altitude / rayleigh_scale_height), 0.0, 1.0);
}

float GetMieDensity(float altitude) {
    return clamp(exp(-altitude / mie_scale_height), 0.0, 1.0);
}

// =============================================================================
// Transmittance texture lookups - reference lines 460-519
// =============================================================================

vec3 GetTransmittanceToTopAtmosphereBoundary(float r, float mu) {
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    float rho = SafeSqrt(r * r - bottom_radius * bottom_radius);
    float d = DistanceToTopAtmosphereBoundary(r, mu);
    float d_min = top_radius - r;
    float d_max = rho + H;
    float x_mu = (d - d_min) / (d_max - d_min);
    float x_r = rho / H;
    vec2 tex_uv = vec2(
        GetTextureCoordFromUnitRange(x_mu, float(TRANSMITTANCE_TEXTURE_WIDTH)),
        GetTextureCoordFromUnitRange(x_r, float(TRANSMITTANCE_TEXTURE_HEIGHT))
    );
    return texture(transmittance_texture, tex_uv).rgb;
}

// Reference: GetTransmittance (line 483-519)
vec3 GetTransmittance(float r, float mu, float d, bool ray_r_mu_intersects_ground) {
    float r_d = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
    float mu_d = ClampCosine((r * mu + d) / r_d);
    
    if (ray_r_mu_intersects_ground) {
        return min(
            GetTransmittanceToTopAtmosphereBoundary(r_d, -mu_d) /
            GetTransmittanceToTopAtmosphereBoundary(r, -mu),
            vec3(1.0));
    } else {
        return min(
            GetTransmittanceToTopAtmosphereBoundary(r, mu) /
            GetTransmittanceToTopAtmosphereBoundary(r_d, mu_d),
            vec3(1.0));
    }
}

// Reference: GetTransmittanceToSun (line 552-563)
vec3 GetTransmittanceToSun(float r, float mu_s) {
    float sin_theta_h = bottom_radius / r;
    float cos_theta_h = -sqrt(max(1.0 - sin_theta_h * sin_theta_h, 0.0));
    return GetTransmittanceToTopAtmosphereBoundary(r, mu_s) *
        smoothstep(-sin_theta_h * sun_angular_radius,
                   sin_theta_h * sun_angular_radius,
                   mu_s - cos_theta_h);
}

// =============================================================================
// Single scattering computation - reference lines 650-730
// =============================================================================

// Reference: ComputeSingleScatteringIntegrand (line 650-668)
void ComputeSingleScatteringIntegrand(
    float r, float mu, float mu_s, float nu, float d,
    bool ray_r_mu_intersects_ground,
    out vec3 rayleigh, out vec3 mie) {
    
    float r_d = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
    float mu_s_d = ClampCosine((r * mu_s + d * nu) / r_d);
    
    vec3 transmittance = 
        GetTransmittance(r, mu, d, ray_r_mu_intersects_ground) *
        GetTransmittanceToSun(r_d, mu_s_d);
    
    float altitude = r_d - bottom_radius;
    rayleigh = transmittance * GetRayleighDensity(altitude);
    mie = transmittance * GetMieDensity(altitude);
}

// Reference: ComputeSingleScattering (line 695-730)
void ComputeSingleScattering(
    float r, float mu, float mu_s, float nu,
    bool ray_r_mu_intersects_ground,
    out vec3 rayleigh, out vec3 mie) {
    
    // Distance to nearest atmosphere boundary
    float dx;
    if (ray_r_mu_intersects_ground) {
        dx = DistanceToBottomAtmosphereBoundary(r, mu) / float(SAMPLE_COUNT);
    } else {
        dx = DistanceToTopAtmosphereBoundary(r, mu) / float(SAMPLE_COUNT);
    }
    
    // Integration using trapezoidal rule
    vec3 rayleigh_sum = vec3(0.0);
    vec3 mie_sum = vec3(0.0);
    
    for (int i = 0; i <= SAMPLE_COUNT; ++i) {
        float d_i = float(i) * dx;
        vec3 rayleigh_i, mie_i;
        ComputeSingleScatteringIntegrand(r, mu, mu_s, nu, d_i,
            ray_r_mu_intersects_ground, rayleigh_i, mie_i);
        float weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
        rayleigh_sum += rayleigh_i * weight_i;
        mie_sum += mie_i * weight_i;
    }
    
    // Reference line 727-729: multiply by dx, solar_irradiance, scattering coefficients
    rayleigh = rayleigh_sum * dx * solar_irradiance * rayleigh_scattering;
    mie = mie_sum * dx * solar_irradiance * mie_scattering;
}

// =============================================================================
// Texture coordinate mapping - reference lines 837-926
// =============================================================================

// Reference: GetRMuMuSNuFromScatteringTextureUvwz (line 837-890)
void GetRMuMuSNuFromScatteringTextureUvwz(
    vec4 uvwz,
    out float r, out float mu, out float mu_s, out float nu,
    out bool ray_r_mu_intersects_ground) {
    
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    float rho = H * GetUnitRangeFromTextureCoord(uvwz.w, float(SCATTERING_TEXTURE_R_SIZE));
    r = sqrt(rho * rho + bottom_radius * bottom_radius);
    
    if (uvwz.z < 0.5) {
        float d_min = r - bottom_radius;
        float d_max = rho;
        float d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
            1.0 - 2.0 * uvwz.z, float(SCATTERING_TEXTURE_MU_SIZE) / 2.0);
        mu = (d == 0.0) ? -1.0 : ClampCosine(-(rho * rho + d * d) / (2.0 * r * d));
        ray_r_mu_intersects_ground = true;
    } else {
        float d_min = top_radius - r;
        float d_max = rho + H;
        float d = d_min + (d_max - d_min) * GetUnitRangeFromTextureCoord(
            2.0 * uvwz.z - 1.0, float(SCATTERING_TEXTURE_MU_SIZE) / 2.0);
        mu = (d == 0.0) ? 1.0 : ClampCosine((H * H - rho * rho - d * d) / (2.0 * r * d));
        ray_r_mu_intersects_ground = false;
    }
    
    float x_mu_s = GetUnitRangeFromTextureCoord(uvwz.y, float(SCATTERING_TEXTURE_MU_S_SIZE));
    float d_min = top_radius - bottom_radius;
    float d_max = H;
    float D = DistanceToTopAtmosphereBoundary(bottom_radius, mu_s_min);
    float A = (D - d_min) / (d_max - d_min);
    float a = (A - x_mu_s * A) / (1.0 + x_mu_s * A);
    float d = d_min + min(a, A) * (d_max - d_min);
    mu_s = (d == 0.0) ? 1.0 : ClampCosine((H * H - d * d) / (2.0 * bottom_radius * d));
    
    nu = ClampCosine(uvwz.x * 2.0 - 1.0);
}

// Reference: GetRMuMuSNuFromScatteringTextureFragCoord (line 905-926)
void GetRMuMuSNuFromScatteringTextureFragCoord(
    vec3 frag_coord,
    out float r, out float mu, out float mu_s, out float nu,
    out bool ray_r_mu_intersects_ground) {
    
    vec4 SCATTERING_TEXTURE_SIZE = vec4(
        float(SCATTERING_TEXTURE_NU_SIZE - 1),
        float(SCATTERING_TEXTURE_MU_S_SIZE),
        float(SCATTERING_TEXTURE_MU_SIZE),
        float(SCATTERING_TEXTURE_R_SIZE));
    
    float frag_coord_nu = floor(frag_coord.x / float(SCATTERING_TEXTURE_MU_S_SIZE));
    float frag_coord_mu_s = mod(frag_coord.x, float(SCATTERING_TEXTURE_MU_S_SIZE));
    
    vec4 uvwz = vec4(frag_coord_nu, frag_coord_mu_s, frag_coord.y, frag_coord.z) /
                SCATTERING_TEXTURE_SIZE;
    
    GetRMuMuSNuFromScatteringTextureUvwz(uvwz, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    
    // Clamp nu to its valid range
    nu = clamp(nu, mu * mu_s - sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)),
                   mu * mu_s + sqrt((1.0 - mu * mu) * (1.0 - mu_s * mu_s)));
}

// =============================================================================
// Main - compute single scattering for one texel
// =============================================================================

void main() {
    // Convert 2D texture coords to 3D frag_coord
    // x = [0, SCATTERING_TEXTURE_WIDTH), y = [0, SCATTERING_TEXTURE_HEIGHT), z = current_layer
    vec3 frag_coord = vec3(
        uv.x * float(SCATTERING_TEXTURE_WIDTH),
        uv.y * float(SCATTERING_TEXTURE_HEIGHT),
        float(current_layer) + 0.5
    );
    
    float r, mu, mu_s, nu;
    bool ray_r_mu_intersects_ground;
    GetRMuMuSNuFromScatteringTextureFragCoord(frag_coord, r, mu, mu_s, nu,
        ray_r_mu_intersects_ground);
    
    vec3 rayleigh, mie;
    ComputeSingleScattering(r, mu, mu_s, nu, ray_r_mu_intersects_ground,
        rayleigh, mie);
    
    // Pack Rayleigh RGB + Mie red channel in alpha (for single Mie extraction)
    fragColor = vec4(rayleigh, mie.r);
}
''')
    
    return gpu.shader.create_from_info(shader_info)


# =============================================================================
# PRECOMPUTED TEXTURES CONTAINER
# =============================================================================
@dataclass
class PrecomputedTexturesV2:
    """Container for precomputed LUT textures."""
    transmittance: np.ndarray
    scattering: Optional[np.ndarray] = None
    irradiance: Optional[np.ndarray] = None
    single_mie_scattering: Optional[np.ndarray] = None


# =============================================================================
# GPU PRECOMPUTE ENGINE
# =============================================================================
class GPUPrecomputeV2:
    """
    Fresh GPU LUT precomputation engine - faithful to Bruneton reference.
    
    Each step is verified against reference before proceeding.
    """
    
    def __init__(self, params: AtmosphereParams):
        self.params = params
        self._transmittance_texture = None
    
    def _render_to_texture(self, shader: 'gpu.types.GPUShader',
                           width: int, height: int,
                           uniforms: dict,
                           textures: dict = None) -> np.ndarray:
        """Render shader to offscreen buffer and read back pixels."""
        offscreen = gpu.types.GPUOffScreen(width, height, format='RGBA32F')
        
        vertices = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        indices = [(0, 1, 2), (0, 2, 3)]
        
        batch = batch_for_shader(shader, 'TRIS', {"pos": vertices}, indices=indices)
        
        with offscreen.bind():
            gpu.state.depth_test_set('NONE')
            gpu.state.blend_set('NONE')
            
            shader.bind()
            
            for name, value in uniforms.items():
                try:
                    if isinstance(value, int):
                        shader.uniform_int(name, value)
                    elif isinstance(value, float):
                        shader.uniform_float(name, value)
                    elif isinstance(value, (tuple, list)):
                        shader.uniform_float(name, value)
                except (ValueError, TypeError):
                    pass
            
            if textures:
                for name, tex in textures.items():
                    try:
                        shader.uniform_sampler(name, tex)
                    except (ValueError, TypeError):
                        pass
            
            batch.draw(shader)
        
        buffer = offscreen.texture_color.read()
        offscreen.free()
        
        pixels = np.array(buffer.to_list(), dtype=np.float32)
        pixels = pixels.reshape(height, width, 4)
        
        return pixels
    
    def precompute_transmittance(self) -> np.ndarray:
        """
        Precompute transmittance LUT.
        
        This is the first step - must match reference before proceeding.
        """
        print("[Helios GPU v2] Computing transmittance...")
        
        shader = create_transmittance_shader()
        
        p = self.params
        uniforms = {
            'bottom_radius': float(p.bottom_radius),
            'top_radius': float(p.top_radius),
            'rayleigh_scattering': p.rayleigh_scattering,
            'rayleigh_scale_height': float(p.rayleigh_scale_height),
            'mie_extinction': p.mie_extinction,
            'mie_scale_height': float(p.mie_scale_height),
            'absorption_extinction': p.absorption_extinction,
        }
        
        print(f"  [DEBUG] bottom_radius={p.bottom_radius}, top_radius={p.top_radius}")
        print(f"  [DEBUG] rayleigh_scattering={p.rayleigh_scattering}")
        print(f"  [DEBUG] mie_extinction={p.mie_extinction}")
        print(f"  [DEBUG] absorption_extinction={p.absorption_extinction}")
        
        pixels = self._render_to_texture(
            shader,
            TRANSMITTANCE_TEXTURE_WIDTH,
            TRANSMITTANCE_TEXTURE_HEIGHT,
            uniforms
        )
        
        transmittance = pixels[:, :, :3].astype(np.float32)
        
        print(f"  [DEBUG] Transmittance range: min={transmittance.min():.6f}, max={transmittance.max():.6f}")
        print(f"  [DEBUG] Transmittance center pixel [32, 128]: {transmittance[32, 128, :]}")
        
        return transmittance
    
    def _create_transmittance_gpu_texture(self, transmittance: np.ndarray):
        """Create GPU texture from transmittance data for use in other shaders."""
        height, width = transmittance.shape[:2]
        
        # Pad to RGBA
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
    
    def precompute_single_scattering(self, transmittance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precompute single scattering LUT.
        
        Direct port of reference ComputeSingleScatteringTexture.
        Returns (scattering, delta_mie) arrays.
        """
        print("[Helios GPU v2] Computing single scattering...")
        
        # Create GPU texture from transmittance
        self._create_transmittance_gpu_texture(transmittance)
        
        shader = create_single_scattering_shader()
        
        p = self.params
        
        # mu_s_min from reference - cos(102 degrees) for half precision
        import math
        mu_s_min = math.cos(102.0 / 180.0 * math.pi)
        
        # Allocate output arrays
        scattering = np.zeros((SCATTERING_TEXTURE_DEPTH, SCATTERING_TEXTURE_HEIGHT,
                               SCATTERING_TEXTURE_WIDTH, 4), dtype=np.float32)
        delta_mie = np.zeros((SCATTERING_TEXTURE_DEPTH, SCATTERING_TEXTURE_HEIGHT,
                              SCATTERING_TEXTURE_WIDTH, 3), dtype=np.float32)
        
        # Render each depth slice
        for layer in range(SCATTERING_TEXTURE_DEPTH):
            uniforms = {
                'bottom_radius': float(p.bottom_radius),
                'top_radius': float(p.top_radius),
                'mu_s_min': float(mu_s_min),
                'current_layer': layer,
            }
            
            pixels = self._render_to_texture(
                shader,
                SCATTERING_TEXTURE_WIDTH,
                SCATTERING_TEXTURE_HEIGHT,
                uniforms,
                textures={'transmittance_texture': self._transmittance_texture}
            )
            
            # Store: RGB = Rayleigh, A = Mie.r
            scattering[layer] = pixels
            
            # Extract Mie from alpha and reconstruct full RGB
            # Mie is wavelength-independent (same coefficient for RGB)
            mie_r = pixels[:, :, 3]
            delta_mie[layer, :, :, 0] = mie_r
            delta_mie[layer, :, :, 1] = mie_r
            delta_mie[layer, :, :, 2] = mie_r
            
            if layer == 0:
                print(f"  [DEBUG] Layer 0 scattering range: min={pixels.min():.6f}, max={pixels.max():.6f}")
                # Test pixel at same location as before
                test_y, test_x = 96, 255
                print(f"  [DEBUG] Test pixel [0, {test_y}, {test_x}]: {pixels[test_y, test_x, :]}")
        
        print(f"  [DEBUG] Scattering shape: {scattering.shape}")
        print(f"  [DEBUG] Scattering range: min={scattering.min():.6f}, max={scattering.max():.6f}")
        
        return scattering, delta_mie
    
    def precompute(self, num_scattering_orders: int = 4,
                   progress_callback: Callable = None) -> PrecomputedTexturesV2:
        """
        Run full precomputation.
        
        V2: Transmittance + Single Scattering
        """
        import time
        start_time = time.perf_counter()
        
        print("[Helios GPU v2] Starting precomputation...")
        
        if progress_callback:
            progress_callback(0.0, "Computing transmittance (GPU v2)...")
        
        # Step 1: Transmittance
        transmittance = self.precompute_transmittance()
        
        if progress_callback:
            progress_callback(0.1, "Computing single scattering (GPU v2)...")
        
        # Step 2: Single Scattering
        scattering, delta_mie = self.precompute_single_scattering(transmittance)
        
        if progress_callback:
            progress_callback(0.5, "Single scattering complete")
        
        # TODO: Add direct irradiance after single scattering verified
        # TODO: Add multiple scattering after direct irradiance verified
        
        # Placeholder for irradiance until we implement it
        irradiance = np.zeros((IRRADIANCE_TEXTURE_HEIGHT, IRRADIANCE_TEXTURE_WIDTH, 3), 
                              dtype=np.float32)
        
        if progress_callback:
            progress_callback(1.0, "GPU v2 precomputation complete")
        
        total_time = time.perf_counter() - start_time
        print(f"[Helios GPU v2] Complete in {total_time:.2f}s")
        
        return PrecomputedTexturesV2(
            transmittance=transmittance,
            scattering=scattering,
            irradiance=irradiance,
            single_mie_scattering=delta_mie
        )


# =============================================================================
# BLENDER GPU ATMOSPHERE MODEL V2 - Main interface for operator
# =============================================================================
class BlenderGPUAtmosphereModelV2:
    """
    Atmosphere model using Blender's built-in GPU for precomputation.
    
    Fresh implementation v2 - faithful port of Bruneton reference.
    Drop-in replacement for BlenderGPUAtmosphereModel.
    """
    
    def __init__(self, params=None):
        # Accept either AtmosphereParams or the existing AtmosphereParameters
        if params is None:
            self.params = AtmosphereParams()
            self._external_params = None
        elif hasattr(params, 'rayleigh_scattering'):
            # It's an AtmosphereParameters from the existing system
            self._external_params = params
            # Convert to our internal AtmosphereParams
            self.params = AtmosphereParams(
                bottom_radius=params.bottom_radius,
                top_radius=params.top_radius,
                rayleigh_scattering=tuple(params.rayleigh_scattering[:3]),
                rayleigh_scale_height=8000.0,  # Default
                mie_scattering=tuple(params.mie_scattering[:3]),
                mie_extinction=tuple(params.mie_extinction[:3]),
                mie_scale_height=1200.0,  # Default
                mie_phase_g=params.mie_phase_function_g,
                absorption_extinction=tuple(params.absorption_extinction[:3]),
                solar_irradiance=(1.474, 1.8504, 1.91198),  # Default RGB
                sun_angular_radius=params.sun_angular_radius,
                ground_albedo=params.ground_albedo if isinstance(params.ground_albedo, float) else 0.1,
            )
        else:
            self.params = params
            self._external_params = None
        
        self.textures: Optional[PrecomputedTexturesV2] = None
        self._is_initialized = False
        self._solar_irradiance_rgb = np.array(self.params.solar_irradiance)
        
        print("[Helios] Using Blender GPU backend V2 (fresh implementation)")
    
    @property
    def is_initialized(self) -> bool:
        return self._is_initialized
    
    def init(self, num_scattering_orders: int = 4, progress_callback=None) -> None:
        """Precompute atmosphere LUTs using GPU."""
        # Run GPU precomputation with our fresh v2 engine
        gpu_precompute = GPUPrecomputeV2(self.params)
        self.textures = gpu_precompute.precompute(
            num_scattering_orders=num_scattering_orders,
            progress_callback=progress_callback
        )
        
        self._is_initialized = True
        print("[Helios GPU v2] Precomputation complete!")
    
    def get_shader_uniforms(self) -> dict:
        """Get dictionary of uniform values for shaders."""
        if not self._is_initialized:
            raise RuntimeError("Model not initialized.")
        
        p = self.params
        length_unit = 1000.0  # km
        return {
            'bottom_radius': p.bottom_radius / length_unit,
            'top_radius': p.top_radius / length_unit,
            'rayleigh_scattering': tuple(x * length_unit for x in p.rayleigh_scattering),
            'mie_scattering': tuple(x * length_unit for x in p.mie_scattering),
            'mie_extinction': tuple(x * length_unit for x in p.mie_extinction),
            'mie_phase_function_g': p.mie_phase_g,
            'absorption_extinction': tuple(x * length_unit for x in p.absorption_extinction),
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
        
        # Create a temporary model to use its EXR saving functionality
        if self._external_params:
            temp_model = AtmosphereModel(self._external_params)
        else:
            # Create minimal params for saving
            from .parameters import AtmosphereParameters
            temp_params = AtmosphereParameters.earth_default()
            temp_model = AtmosphereModel(temp_params)
        
        # Create a compatible textures object
        from .gpu_precompute import PrecomputedTextures
        temp_model.textures = PrecomputedTextures(
            transmittance=self.textures.transmittance,
            scattering=self.textures.scattering,
            irradiance=self.textures.irradiance,
            single_mie_scattering=self.textures.single_mie_scattering
        )
        temp_model._is_initialized = True
        temp_model._solar_irradiance_rgb = self._solar_irradiance_rgb
        temp_model.save_textures_exr(output_dir)


# =============================================================================
# EXPORTS - Make v2 available as drop-in replacement
# =============================================================================
# Alias for compatibility - can switch back by changing this
BlenderGPUAtmosphereModel = BlenderGPUAtmosphereModelV2
GPUPrecompute = GPUPrecomputeV2


# =============================================================================
# TEST FUNCTION
# =============================================================================
def test_transmittance():
    """Test transmittance computation and compare with reference."""
    print("=" * 60)
    print("TRANSMITTANCE BASELINE TEST")
    print("=" * 60)
    
    model = BlenderGPUAtmosphereModelV2()
    model.init(num_scattering_orders=4)
    
    transmittance = model.textures.transmittance
    
    print(f"\nTransmittance shape: {transmittance.shape}")
    print(f"Expected shape: ({TRANSMITTANCE_TEXTURE_HEIGHT}, {TRANSMITTANCE_TEXTURE_WIDTH}, 3)")
    
    # Load reference for comparison
    ref_path = r"c:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-4\cache\reference_transmittance.npy"
    if os.path.exists(ref_path):
        reference = np.load(ref_path)
        print(f"\nReference shape: {reference.shape}")
        
        # Compare
        diff = np.abs(transmittance - reference[:, :, :3])
        print(f"Max difference: {diff.max():.6f}")
        print(f"Mean difference: {diff.mean():.6f}")
        
        # Per-channel comparison
        for i, c in enumerate(['R', 'G', 'B']):
            ch_diff = diff[:, :, i]
            print(f"  {c}: max={ch_diff.max():.6f}, mean={ch_diff.mean():.6f}")
    else:
        print(f"\nReference not found at {ref_path}")
        print("Run the reference app to generate comparison data.")
    
    print("=" * 60)
    return transmittance


if __name__ == "__main__":
    if IN_BLENDER:
        test_transmittance()
    else:
        print("Run this script in Blender's Python console.")
