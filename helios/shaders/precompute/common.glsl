/**
 * Helios Precomputation - Common GLSL code
 * Shared definitions and functions for all precomputation shaders.
 */

// Constants
const float PI = 3.14159265358979323846;

// Texture dimensions
const int TRANSMITTANCE_TEXTURE_WIDTH = 256;
const int TRANSMITTANCE_TEXTURE_HEIGHT = 64;
const int SCATTERING_TEXTURE_R_SIZE = 32;
const int SCATTERING_TEXTURE_MU_SIZE = 128;
const int SCATTERING_TEXTURE_MU_S_SIZE = 32;
const int SCATTERING_TEXTURE_NU_SIZE = 8;
const int SCATTERING_TEXTURE_WIDTH = 256;  // NU_SIZE * MU_S_SIZE
const int SCATTERING_TEXTURE_HEIGHT = 128; // MU_SIZE
const int SCATTERING_TEXTURE_DEPTH = 32;   // R_SIZE
const int IRRADIANCE_TEXTURE_WIDTH = 64;
const int IRRADIANCE_TEXTURE_HEIGHT = 16;

// Atmosphere uniforms
uniform float bottom_radius;
uniform float top_radius;
uniform vec3 rayleigh_scattering;
uniform float rayleigh_scale_height;
uniform vec3 mie_scattering;
uniform vec3 mie_extinction;
uniform float mie_scale_height;
uniform float mie_phase_g;
uniform vec3 absorption_extinction;
uniform vec3 ground_albedo;
uniform vec3 solar_irradiance;

// For 3D texture slices
uniform int current_layer;

// Helper functions
float SafeSqrt(float x) {
    return sqrt(max(x, 0.0));
}

float ClampCosine(float mu) {
    return clamp(mu, -1.0, 1.0);
}

float ClampRadius(float r) {
    return clamp(r, bottom_radius, top_radius);
}

float ClampDistance(float d) {
    return max(d, 0.0);
}

// Distance calculations
float DistanceToTopAtmosphereBoundary(float r, float mu) {
    float discriminant = r * r * (mu * mu - 1.0) + top_radius * top_radius;
    return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

float DistanceToBottomAtmosphereBoundary(float r, float mu) {
    float discriminant = r * r * (mu * mu - 1.0) + bottom_radius * bottom_radius;
    return ClampDistance(-r * mu - SafeSqrt(discriminant));
}

bool RayIntersectsGround(float r, float mu) {
    return mu < 0.0 && r * r * (mu * mu - 1.0) + bottom_radius * bottom_radius >= 0.0;
}

// Density profile (exponential)
float GetDensityExponential(float altitude, float scale_height) {
    return exp(-altitude / scale_height);
}

// Ozone density profile (layer between 10-40km, peak at 25km)
float GetOzoneDensity(float altitude) {
    float layer_center = 25000.0;  // 25km
    float layer_width = 15000.0;   // 15km half-width
    if (altitude < 10000.0 || altitude > 40000.0) return 0.0;
    return 1.0 - abs(altitude - layer_center) / layer_width;
}

// Texture coordinate conversions
float GetTextureCoordFromUnitRange(float x, int texture_size) {
    return 0.5 / float(texture_size) + x * (1.0 - 1.0 / float(texture_size));
}

float GetUnitRangeFromTextureCoord(float u, int texture_size) {
    return (u - 0.5 / float(texture_size)) / (1.0 - 1.0 / float(texture_size));
}

// Phase functions
float RayleighPhaseFunction(float nu) {
    float k = 3.0 / (16.0 * PI);
    return k * (1.0 + nu * nu);
}

float MiePhaseFunction(float g, float nu) {
    float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);
}
