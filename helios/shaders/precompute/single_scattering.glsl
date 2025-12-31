/**
 * Helios Precomputation - Single Scattering Fragment Shader
 * Computes single scattering for one depth slice of the 3D texture.
 */

uniform sampler2D transmittance_texture;
uniform int current_layer;  // Current depth slice (0 to SCATTERING_TEXTURE_DEPTH-1)

in vec2 uv;
out vec4 fragColor;

// Constants
const float PI = 3.14159265358979323846;
const int TRANSMITTANCE_TEXTURE_WIDTH = 256;
const int TRANSMITTANCE_TEXTURE_HEIGHT = 64;
const int SCATTERING_TEXTURE_R_SIZE = 32;
const int SCATTERING_TEXTURE_MU_SIZE = 128;
const int SCATTERING_TEXTURE_MU_S_SIZE = 32;
const int SCATTERING_TEXTURE_NU_SIZE = 8;
const int SCATTERING_TEXTURE_WIDTH = 256;
const int SCATTERING_TEXTURE_HEIGHT = 128;

// Atmosphere uniforms
uniform float bottom_radius;
uniform float top_radius;
uniform vec3 rayleigh_scattering;
uniform float rayleigh_scale_height;
uniform vec3 mie_scattering;
uniform vec3 mie_extinction;
uniform float mie_scale_height;
uniform vec3 absorption_extinction;
uniform vec3 solar_irradiance;

// Helpers
float SafeSqrt(float x) { return sqrt(max(x, 0.0)); }
float ClampDistance(float d) { return max(d, 0.0); }
float ClampCosine(float mu) { return clamp(mu, -1.0, 1.0); }
float ClampRadius(float r) { return clamp(r, bottom_radius, top_radius); }

float GetUnitRangeFromTextureCoord(float u, int texture_size) {
    return (u - 0.5 / float(texture_size)) / (1.0 - 1.0 / float(texture_size));
}

float GetTextureCoordFromUnitRange(float x, int texture_size) {
    return 0.5 / float(texture_size) + x * (1.0 - 1.0 / float(texture_size));
}

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

float GetDensityExponential(float altitude, float scale_height) {
    return exp(-altitude / scale_height);
}

float GetOzoneDensity(float altitude) {
    if (altitude < 10000.0 || altitude > 40000.0) return 0.0;
    return max(0.0, 1.0 - abs(altitude - 25000.0) / 15000.0);
}

// Transmittance texture sampling
vec2 GetTransmittanceTextureUvFromRMu(float r, float mu) {
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    float rho = SafeSqrt(r * r - bottom_radius * bottom_radius);
    float d = DistanceToTopAtmosphereBoundary(r, mu);
    float d_min = top_radius - r;
    float d_max = rho + H;
    float x_mu = (d_max > d_min) ? (d - d_min) / (d_max - d_min) : 0.0;
    float x_r = rho / H;
    return vec2(
        GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH),
        GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT)
    );
}

vec3 GetTransmittanceToTopAtmosphereBoundary(float r, float mu) {
    vec2 uv = GetTransmittanceTextureUvFromRMu(r, mu);
    return texture(transmittance_texture, uv).rgb;
}

vec3 GetTransmittance(float r, float mu, float d, bool ray_intersects_ground) {
    float r_d = ClampRadius(sqrt(d * d + 2.0 * r * mu * d + r * r));
    float mu_d = ClampCosine((r * mu + d) / r_d);
    
    vec3 trans_full = GetTransmittanceToTopAtmosphereBoundary(r, mu);
    vec3 trans_d = GetTransmittanceToTopAtmosphereBoundary(r_d, mu_d);
    
    if (ray_intersects_ground) {
        return min(GetTransmittanceToTopAtmosphereBoundary(r_d, -mu_d) /
                   max(GetTransmittanceToTopAtmosphereBoundary(r, -mu), vec3(1e-10)), vec3(1.0));
    } else {
        return min(trans_full / max(trans_d, vec3(1e-10)), vec3(1.0));
    }
}

// Decode scattering texture coordinates
void GetRMuMuSNuFromScatteringTextureUvwz(vec2 uv_xy, int layer,
    out float r, out float mu, out float mu_s, out float nu, out bool ray_r_mu_intersects_ground) {
    
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    
    // r from layer (z coordinate)
    float x_r = GetUnitRangeFromTextureCoord((float(layer) + 0.5) / float(SCATTERING_TEXTURE_R_SIZE), SCATTERING_TEXTURE_R_SIZE);
    float rho = H * x_r;
    r = sqrt(rho * rho + bottom_radius * bottom_radius);
    
    // mu from y coordinate
    float x_mu = uv_xy.y;
    if (x_mu < 0.5) {
        float d_min = r - bottom_radius;
        float d_max = rho;
        float x = GetUnitRangeFromTextureCoord(1.0 - 2.0 * x_mu, SCATTERING_TEXTURE_MU_SIZE / 2);
        float d = d_min + (d_max - d_min) * x;
        mu = (d == 0.0) ? -1.0 : -(rho * rho + d * d) / (2.0 * r * d);
        ray_r_mu_intersects_ground = true;
    } else {
        float d_min = top_radius - r;
        float d_max = rho + H;
        float x = GetUnitRangeFromTextureCoord(2.0 * x_mu - 1.0, SCATTERING_TEXTURE_MU_SIZE / 2);
        float d = d_min + (d_max - d_min) * x;
        mu = (d == 0.0) ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
        ray_r_mu_intersects_ground = false;
    }
    mu = ClampCosine(mu);
    
    // mu_s and nu from x coordinate
    int tex_x = int(uv_xy.x * float(SCATTERING_TEXTURE_WIDTH));
    int nu_idx = tex_x % SCATTERING_TEXTURE_NU_SIZE;
    int mu_s_idx = tex_x / SCATTERING_TEXTURE_NU_SIZE;
    
    float x_mu_s = GetUnitRangeFromTextureCoord((float(mu_s_idx) + 0.5) / float(SCATTERING_TEXTURE_MU_S_SIZE), SCATTERING_TEXTURE_MU_S_SIZE);
    float d_min = top_radius - bottom_radius;
    float d_max = H;
    float D = DistanceToTopAtmosphereBoundary(bottom_radius, -0.2);  // mu_s_min
    float A = (D - d_min) / (d_max - d_min);
    float a = (1.0 + x_mu_s * A != 0.0) ? (A - x_mu_s * A) / (1.0 + x_mu_s * A) : 0.0;
    float d = d_min + min(a, A) * (d_max - d_min);
    mu_s = (d == 0.0) ? 1.0 : (H * H - d * d) / (2.0 * bottom_radius * d);
    mu_s = ClampCosine(mu_s);
    
    float x_nu = GetUnitRangeFromTextureCoord((float(nu_idx) + 0.5) / float(SCATTERING_TEXTURE_NU_SIZE), SCATTERING_TEXTURE_NU_SIZE);
    nu = x_nu * 2.0 - 1.0;
}

void ComputeSingleScattering(float r, float mu, float mu_s, float nu, bool ray_intersects_ground,
    out vec3 rayleigh, out vec3 mie) {
    
    const int SAMPLE_COUNT = 50;
    
    float d_max = ray_intersects_ground ? 
        DistanceToBottomAtmosphereBoundary(r, mu) :
        DistanceToTopAtmosphereBoundary(r, mu);
    float dx = d_max / float(SAMPLE_COUNT);
    
    rayleigh = vec3(0.0);
    mie = vec3(0.0);
    
    for (int i = 0; i < SAMPLE_COUNT; ++i) {
        float d_i = (float(i) + 0.5) * dx;
        float r_i = ClampRadius(sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r));
        float mu_i = ClampCosine((r * mu + d_i) / r_i);
        float mu_s_i = ClampCosine((r * mu_s + d_i * nu) / r_i);
        float altitude = r_i - bottom_radius;
        
        vec3 trans_to_i = GetTransmittance(r, mu, d_i, ray_intersects_ground);
        vec3 trans_to_sun = GetTransmittanceToTopAtmosphereBoundary(r_i, mu_s_i);
        vec3 trans = trans_to_i * trans_to_sun;
        
        float rayleigh_density = GetDensityExponential(altitude, rayleigh_scale_height);
        float mie_density = GetDensityExponential(altitude, mie_scale_height);
        
        rayleigh += trans * rayleigh_density * rayleigh_scattering * dx;
        mie += trans * mie_density * mie_scattering * dx;
    }
    
    rayleigh *= solar_irradiance;
    mie *= solar_irradiance;
}

void main() {
    float r, mu, mu_s, nu;
    bool ray_intersects_ground;
    
    GetRMuMuSNuFromScatteringTextureUvwz(uv, current_layer, r, mu, mu_s, nu, ray_intersects_ground);
    
    vec3 rayleigh, mie;
    ComputeSingleScattering(r, mu, mu_s, nu, ray_intersects_ground, rayleigh, mie);
    
    // Pack: RGB = Rayleigh, A = Mie.r (for single channel Mie)
    fragColor = vec4(rayleigh, mie.r);
}
