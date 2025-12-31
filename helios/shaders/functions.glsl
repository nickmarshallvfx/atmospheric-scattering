/**
 * Helios Atmospheric Scattering - Core GLSL Functions
 * 
 * Ported from atmosphere/functions.glsl by Eric Bruneton
 * Implements transmittance, scattering, and irradiance lookups
 *
 * Copyright (c) 2017 Eric Bruneton (BSD License)
 * Copyright (c) 2024 MattePaint
 */

#include "definitions.glsl"

// ============================================================================
// TRANSMITTANCE FUNCTIONS
// ============================================================================

/**
 * Compute distance to top atmosphere boundary
 */
float DistanceToTopAtmosphereBoundary(float r, float mu) {
    float discriminant = r * r * (mu * mu - 1.0) + 
                         atmosphere.top_radius * atmosphere.top_radius;
    return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

/**
 * Compute distance to bottom atmosphere boundary (ground)
 */
float DistanceToBottomAtmosphereBoundary(float r, float mu) {
    float discriminant = r * r * (mu * mu - 1.0) + 
                         atmosphere.bottom_radius * atmosphere.bottom_radius;
    return ClampDistance(-r * mu - SafeSqrt(discriminant));
}

/**
 * Check if ray intersects ground
 */
bool RayIntersectsGround(float r, float mu) {
    return mu < 0.0 && r * r * (mu * mu - 1.0) + 
           atmosphere.bottom_radius * atmosphere.bottom_radius >= 0.0;
}

/**
 * Get UV coordinates for transmittance texture from (r, mu)
 */
vec2 GetTransmittanceTextureUvFromRMu(float r, float mu) {
    float H = sqrt(atmosphere.top_radius * atmosphere.top_radius - 
                   atmosphere.bottom_radius * atmosphere.bottom_radius);
    float rho = SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
    
    float d = DistanceToTopAtmosphereBoundary(r, mu);
    float d_min = atmosphere.top_radius - r;
    float d_max = rho + H;
    
    float x_mu = (d - d_min) / (d_max - d_min);
    float x_r = rho / H;
    
    return vec2(
        0.5 / float(TRANSMITTANCE_TEXTURE_WIDTH) + 
            x_mu * (1.0 - 1.0 / float(TRANSMITTANCE_TEXTURE_WIDTH)),
        0.5 / float(TRANSMITTANCE_TEXTURE_HEIGHT) + 
            x_r * (1.0 - 1.0 / float(TRANSMITTANCE_TEXTURE_HEIGHT))
    );
}

/**
 * Get transmittance to top of atmosphere from texture
 */
vec3 GetTransmittanceToTopAtmosphereBoundary(float r, float mu) {
    vec2 uv = GetTransmittanceTextureUvFromRMu(r, mu);
    return texture(transmittance_texture, uv).rgb;
}

/**
 * Get transmittance between two points
 */
vec3 GetTransmittance(float r, float mu, float d, bool ray_r_mu_intersects_ground) {
    float r_d = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
    float mu_d = ClampCosine((r * mu + d) / r_d);
    
    if (ray_r_mu_intersects_ground) {
        return min(
            GetTransmittanceToTopAtmosphereBoundary(r_d, -mu_d) /
            GetTransmittanceToTopAtmosphereBoundary(r, -mu),
            vec3(1.0)
        );
    } else {
        return min(
            GetTransmittanceToTopAtmosphereBoundary(r, mu) /
            GetTransmittanceToTopAtmosphereBoundary(r_d, mu_d),
            vec3(1.0)
        );
    }
}

/**
 * Get transmittance to sun
 */
vec3 GetTransmittanceToSun(float r, float mu_s) {
    float sin_theta_h = atmosphere.bottom_radius / r;
    float cos_theta_h = -sqrt(max(1.0 - sin_theta_h * sin_theta_h, 0.0));
    
    return GetTransmittanceToTopAtmosphereBoundary(r, mu_s) *
           smoothstep(-sin_theta_h * atmosphere.sun_angular_radius,
                      sin_theta_h * atmosphere.sun_angular_radius,
                      mu_s - cos_theta_h);
}

// ============================================================================
// SCATTERING FUNCTIONS
// ============================================================================

/**
 * Rayleigh phase function
 */
float RayleighPhaseFunction(float nu) {
    float k = 3.0 / (16.0 * PI);
    return k * (1.0 + nu * nu);
}

/**
 * Mie phase function (Cornette-Shanks)
 */
float MiePhaseFunction(float g, float nu) {
    float k = 3.0 / (8.0 * PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);
}

/**
 * Get 4D texture coordinates for scattering lookup
 */
vec4 GetScatteringTextureUvwzFromRMuMuSNu(float r, float mu, float mu_s, float nu,
                                          bool ray_r_mu_intersects_ground) {
    float H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
                   atmosphere.bottom_radius * atmosphere.bottom_radius);
    float rho = SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
    
    float u_r = 0.5 / float(SCATTERING_TEXTURE_R_SIZE) +
                rho / H * (1.0 - 1.0 / float(SCATTERING_TEXTURE_R_SIZE));
    
    float r_mu = r * mu;
    float discriminant = r_mu * r_mu - r * r + atmosphere.bottom_radius * atmosphere.bottom_radius;
    float u_mu;
    if (ray_r_mu_intersects_ground) {
        float d = -r_mu - SafeSqrt(discriminant);
        float d_min = r - atmosphere.bottom_radius;
        float d_max = rho;
        u_mu = 0.5 - 0.5 / float(SCATTERING_TEXTURE_MU_SIZE) -
               (d_max == d_min ? 0.0 : (d - d_min) / (d_max - d_min)) *
               (0.5 - 1.0 / float(SCATTERING_TEXTURE_MU_SIZE));
    } else {
        float d = -r_mu + SafeSqrt(discriminant + H * H);
        float d_min = atmosphere.top_radius - r;
        float d_max = rho + H;
        u_mu = 0.5 + 0.5 / float(SCATTERING_TEXTURE_MU_SIZE) +
               (d - d_min) / (d_max - d_min) *
               (0.5 - 1.0 / float(SCATTERING_TEXTURE_MU_SIZE));
    }
    
    float d = DistanceToTopAtmosphereBoundary(atmosphere.bottom_radius, mu_s);
    float d_min = atmosphere.top_radius - atmosphere.bottom_radius;
    float d_max = H;
    float a = (d - d_min) / (d_max - d_min);
    float D = DistanceToTopAtmosphereBoundary(atmosphere.bottom_radius, 
              cos(max(acos(mu_s) - PI, 0.0)));  // mu_s for horizon
    float A = (D - d_min) / (d_max - d_min);
    float u_mu_s = max(1.0 - a / A, 0.0) / (1.0 + a);
    u_mu_s = 0.5 / float(SCATTERING_TEXTURE_MU_S_SIZE) +
             u_mu_s * (1.0 - 1.0 / float(SCATTERING_TEXTURE_MU_S_SIZE));
    
    float u_nu = (nu + 1.0) / 2.0;
    
    return vec4(u_nu, u_mu_s, u_mu, u_r);
}

/**
 * Sample scattering from 4D texture (packed into 3D)
 */
vec3 GetScattering(float r, float mu, float mu_s, float nu,
                   bool ray_r_mu_intersects_ground) {
    vec4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(r, mu, mu_s, nu, 
                                                     ray_r_mu_intersects_ground);
    
    float tex_coord_x = uvwz.x * float(SCATTERING_TEXTURE_NU_SIZE - 1);
    float tex_x = floor(tex_coord_x);
    float lerp = tex_coord_x - tex_x;
    
    vec3 uvw0 = vec3((tex_x + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), 
                     uvwz.z, uvwz.w);
    vec3 uvw1 = vec3((tex_x + 1.0 + uvwz.y) / float(SCATTERING_TEXTURE_NU_SIZE), 
                     uvwz.z, uvwz.w);
    
    return texture(scattering_texture, uvw0).rgb * (1.0 - lerp) +
           texture(scattering_texture, uvw1).rgb * lerp;
}

// ============================================================================
// IRRADIANCE FUNCTIONS
// ============================================================================

/**
 * Get UV for irradiance texture
 */
vec2 GetIrradianceTextureUvFromRMuS(float r, float mu_s) {
    float x_r = (r - atmosphere.bottom_radius) /
                (atmosphere.top_radius - atmosphere.bottom_radius);
    float x_mu_s = mu_s * 0.5 + 0.5;
    
    return vec2(
        0.5 / float(IRRADIANCE_TEXTURE_WIDTH) +
            x_mu_s * (1.0 - 1.0 / float(IRRADIANCE_TEXTURE_WIDTH)),
        0.5 / float(IRRADIANCE_TEXTURE_HEIGHT) +
            x_r * (1.0 - 1.0 / float(IRRADIANCE_TEXTURE_HEIGHT))
    );
}

/**
 * Get ground irradiance from texture
 */
vec3 GetIrradiance(float r, float mu_s) {
    vec2 uv = GetIrradianceTextureUvFromRMuS(r, mu_s);
    return texture(irradiance_texture, uv).rgb;
}

// ============================================================================
// SKY RENDERING FUNCTIONS
// ============================================================================

/**
 * Get sky radiance along a view ray
 */
vec3 GetSkyRadiance(vec3 camera, vec3 view_ray, float shadow_length,
                    vec3 sun_direction, out vec3 transmittance) {
    // Position relative to planet center
    float r = length(camera);
    float rmu = dot(camera, view_ray);
    float distance_to_top = -rmu - sqrt(rmu * rmu - r * r + 
                            atmosphere.top_radius * atmosphere.top_radius);
    
    // If ray doesn't intersect atmosphere, return zero
    if (distance_to_top < 0.0) {
        transmittance = vec3(1.0);
        return vec3(0.0);
    }
    
    // Move to atmosphere entry point if outside
    if (distance_to_top > 0.0) {
        camera = camera + view_ray * distance_to_top;
        r = atmosphere.top_radius;
        rmu += distance_to_top;
    }
    
    // Compute view parameters
    float mu = rmu / r;
    float mu_s = dot(camera, sun_direction) / r;
    float nu = dot(view_ray, sun_direction);
    bool ray_r_mu_intersects_ground = RayIntersectsGround(r, mu);
    
    // Get transmittance
    transmittance = ray_r_mu_intersects_ground ? vec3(0.0) :
                    GetTransmittanceToTopAtmosphereBoundary(r, mu);
    
    // Get scattering with phase functions
    vec3 scattering = GetScattering(r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    
    // Apply shadow if needed
    if (shadow_length > 0.0) {
        // Reduce scattering in shadowed region
        float shadow_transmittance = exp(-shadow_length * 0.1);
        scattering *= shadow_transmittance;
    }
    
    return scattering;
}

/**
 * Get sky radiance to a point (aerial perspective)
 */
vec3 GetSkyRadianceToPoint(vec3 camera, vec3 point, float shadow_length,
                           vec3 sun_direction, out vec3 transmittance) {
    vec3 view_ray = normalize(point - camera);
    float r = length(camera);
    float rmu = dot(camera, view_ray);
    float distance_to_point = length(point - camera);
    
    float mu = rmu / r;
    float mu_s = dot(camera, sun_direction) / r;
    float nu = dot(view_ray, sun_direction);
    bool ray_r_mu_intersects_ground = false;
    
    // Get transmittance to point
    transmittance = GetTransmittance(r, mu, distance_to_point, ray_r_mu_intersects_ground);
    
    // Get inscattered light
    vec3 scattering = GetScattering(r, mu, mu_s, nu, ray_r_mu_intersects_ground);
    
    // Subtract scattering beyond the point
    float r_p = length(point);
    float mu_p = dot(point, view_ray) / r_p;
    float mu_s_p = dot(point, sun_direction) / r_p;
    
    vec3 scattering_p = GetScattering(r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground);
    scattering = scattering - transmittance * scattering_p;
    
    return scattering;
}

/**
 * Get sun and sky irradiance at a point on a surface
 */
vec3 GetSunAndSkyIrradiance(vec3 point, vec3 normal, vec3 sun_direction,
                            out vec3 sky_irradiance) {
    float r = length(point);
    float mu_s = dot(point, sun_direction) / r;
    
    // Direct sun irradiance
    vec3 sun_irradiance = atmosphere.solar_irradiance * 
                          GetTransmittanceToSun(r, mu_s) *
                          max(dot(normal, sun_direction), 0.0);
    
    // Sky irradiance from ground illumination
    sky_irradiance = GetIrradiance(r, mu_s) * 
                     (1.0 + dot(normal, point / r)) * 0.5;
    
    return sun_irradiance;
}

/**
 * Get solar radiance (for sun disk rendering)
 */
vec3 GetSolarRadiance() {
    return atmosphere.solar_irradiance / 
           (PI * atmosphere.sun_angular_radius * atmosphere.sun_angular_radius);
}

#endif // HELIOS_FUNCTIONS_GLSL
