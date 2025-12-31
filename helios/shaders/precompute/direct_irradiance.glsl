/**
 * Helios Precomputation - Direct Irradiance Fragment Shader
 * Computes direct sun irradiance at ground level.
 */

uniform sampler2D transmittance_texture;

in vec2 uv;
out vec4 fragColor;

// Constants
const float PI = 3.14159265358979323846;
const int TRANSMITTANCE_TEXTURE_WIDTH = 256;
const int TRANSMITTANCE_TEXTURE_HEIGHT = 64;
const int IRRADIANCE_TEXTURE_WIDTH = 64;
const int IRRADIANCE_TEXTURE_HEIGHT = 16;

uniform float bottom_radius;
uniform float top_radius;
uniform vec3 solar_irradiance;
uniform float sun_angular_radius;

float SafeSqrt(float x) { return sqrt(max(x, 0.0)); }
float ClampDistance(float d) { return max(d, 0.0); }

float GetTextureCoordFromUnitRange(float x, int texture_size) {
    return 0.5 / float(texture_size) + x * (1.0 - 1.0 / float(texture_size));
}

float GetUnitRangeFromTextureCoord(float u, int texture_size) {
    return (u - 0.5 / float(texture_size)) / (1.0 - 1.0 / float(texture_size));
}

float DistanceToTopAtmosphereBoundary(float r, float mu) {
    float discriminant = r * r * (mu * mu - 1.0) + top_radius * top_radius;
    return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

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

vec3 GetTransmittanceToSun(float r, float mu_s) {
    float sin_theta_h = bottom_radius / r;
    float cos_theta_h = -sqrt(max(1.0 - sin_theta_h * sin_theta_h, 0.0));
    return GetTransmittanceToTopAtmosphereBoundary(r, mu_s) *
           smoothstep(-sin_theta_h * sun_angular_radius,
                      sin_theta_h * sun_angular_radius,
                      mu_s - cos_theta_h);
}

void GetRMuSFromIrradianceTextureUv(vec2 uv, out float r, out float mu_s) {
    float x_mu_s = GetUnitRangeFromTextureCoord(uv.x, IRRADIANCE_TEXTURE_WIDTH);
    float x_r = GetUnitRangeFromTextureCoord(uv.y, IRRADIANCE_TEXTURE_HEIGHT);
    
    r = bottom_radius + x_r * (top_radius - bottom_radius);
    mu_s = clamp(2.0 * x_mu_s - 1.0, -1.0, 1.0);
}

void main() {
    float r, mu_s;
    GetRMuSFromIrradianceTextureUv(uv, r, mu_s);
    
    vec3 direct_irradiance = solar_irradiance * GetTransmittanceToSun(r, mu_s) * max(mu_s, 0.0);
    fragColor = vec4(direct_irradiance, 1.0);
}
