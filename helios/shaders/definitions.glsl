/**
 * Helios Atmospheric Scattering - GLSL Definitions
 * 
 * Ported from atmosphere/definitions.glsl by Eric Bruneton
 * Adapted for Blender integration with Z-up coordinate system
 *
 * Copyright (c) 2017 Eric Bruneton (BSD License)
 * Copyright (c) 2024 MattePaint
 */

#ifndef HELIOS_DEFINITIONS_GLSL
#define HELIOS_DEFINITIONS_GLSL

// Constants
const float PI = 3.14159265358979323846;
const float INV_PI = 0.31830988618379067154;

// Physical unit types (for clarity, all are float in GLSL)
#define Length float
#define Wavelength float
#define Angle float
#define SolidAngle float
#define Power float
#define LuminousPower float
#define Area float
#define Volume float
#define Irradiance float
#define Radiance float
#define SpectralPower float
#define SpectralIrradiance float
#define SpectralRadiance float
#define SpectralRadianceDensity float
#define ScatteringCoefficient float
#define InverseSolidAngle float
#define LuminousIntensity float
#define Luminance float
#define Illuminance float
#define Number float
#define AbstractSpectrum vec3
#define DimensionlessSpectrum vec3
#define PowerSpectrum vec3
#define IrradianceSpectrum vec3
#define RadianceSpectrum vec3
#define RadianceDensitySpectrum vec3
#define ScatteringSpectrum vec3
#define Position vec3
#define Direction vec3
#define Luminance3 vec3
#define Illuminance3 vec3

// LUT texture dimensions
const int TRANSMITTANCE_TEXTURE_WIDTH = 256;
const int TRANSMITTANCE_TEXTURE_HEIGHT = 64;
const int SCATTERING_TEXTURE_R_SIZE = 32;
const int SCATTERING_TEXTURE_MU_SIZE = 128;
const int SCATTERING_TEXTURE_MU_S_SIZE = 32;
const int SCATTERING_TEXTURE_NU_SIZE = 8;
const int SCATTERING_TEXTURE_WIDTH = SCATTERING_TEXTURE_NU_SIZE * SCATTERING_TEXTURE_MU_S_SIZE;
const int SCATTERING_TEXTURE_HEIGHT = SCATTERING_TEXTURE_MU_SIZE;
const int SCATTERING_TEXTURE_DEPTH = SCATTERING_TEXTURE_R_SIZE;
const int IRRADIANCE_TEXTURE_WIDTH = 64;
const int IRRADIANCE_TEXTURE_HEIGHT = 16;

// Atmosphere parameters structure
struct AtmosphereParameters {
    // Radii
    float bottom_radius;  // Planet surface radius
    float top_radius;     // Top of atmosphere radius
    
    // Rayleigh scattering
    vec3 rayleigh_scattering;
    float rayleigh_scale_height;
    
    // Mie scattering
    vec3 mie_scattering;
    vec3 mie_extinction;
    float mie_phase_function_g;
    float mie_scale_height;
    
    // Absorption (ozone)
    vec3 absorption_extinction;
    
    // Ground
    vec3 ground_albedo;
    
    // Sun
    float sun_angular_radius;
    vec3 solar_irradiance;
};

// Density profile layer
struct DensityProfileLayer {
    float width;
    float exp_term;
    float exp_scale;
    float linear_term;
    float constant_term;
};

// Unit length in meters used by the shader
// This is set by the model and affects all distance calculations
uniform float kLengthUnitInMeters;

// Atmosphere parameters uniform
uniform AtmosphereParameters atmosphere;

// Precomputed textures
uniform sampler2D transmittance_texture;
uniform sampler3D scattering_texture;
uniform sampler2D irradiance_texture;
uniform sampler3D single_mie_scattering_texture;  // Only if not combined

// Rendering uniforms
uniform vec3 camera;           // Camera position (in length units)
uniform vec3 earth_center;     // Planet center (usually 0,0,-radius)
uniform vec3 sun_direction;    // Unit vector towards sun
uniform vec2 sun_size;         // x=tan(angular_radius), y=cos(angular_radius)
uniform float exposure;
uniform vec3 white_point;

// Helper functions for coordinate system
// Blender uses Z-up, our reference uses Z-up as well

/**
 * Get atmospheric density at given altitude for exponential profile
 */
float GetLayerDensity(DensityProfileLayer layer, float altitude) {
    float density = layer.exp_term * exp(layer.exp_scale * altitude) +
                    layer.linear_term * altitude + layer.constant_term;
    return clamp(density, 0.0, 1.0);
}

/**
 * Clamp cosine to valid range to avoid NaN from numerical issues
 */
float ClampCosine(float mu) {
    return clamp(mu, -1.0, 1.0);
}

/**
 * Clamp distance to positive
 */
float ClampDistance(float d) {
    return max(d, 0.0);
}

/**
 * Clamp radius to atmosphere bounds
 */
float ClampRadius(float r) {
    return clamp(r, atmosphere.bottom_radius, atmosphere.top_radius);
}

/**
 * Safe square root (clamp negative to zero)
 */
float SafeSqrt(float a) {
    return sqrt(max(a, 0.0));
}

#endif // HELIOS_DEFINITIONS_GLSL
