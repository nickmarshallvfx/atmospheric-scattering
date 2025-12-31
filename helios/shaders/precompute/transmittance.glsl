/**
 * Helios Precomputation - Transmittance Fragment Shader
 * Computes optical depth and transmittance to top of atmosphere.
 * 
 * For Blender GPUShaderCreateInfo: in/out defined via API.
 * - Input: uv (from vertex shader)
 * - Output: fragColor
 */

// Constants
const float PI = 3.14159265358979323846;
const int TRANSMITTANCE_TEXTURE_WIDTH = 256;
const int TRANSMITTANCE_TEXTURE_HEIGHT = 64;

float SafeSqrt(float x) { return sqrt(max(x, 0.0)); }
float ClampDistance(float d) { return max(d, 0.0); }

float DistanceToTopAtmosphereBoundary(float r, float mu) {
    float discriminant = r * r * (mu * mu - 1.0) + top_radius * top_radius;
    return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

float GetUnitRangeFromTextureCoord(float u, int texture_size) {
    return (u - 0.5 / float(texture_size)) / (1.0 - 1.0 / float(texture_size));
}

float GetDensityExponential(float altitude, float scale_height) {
    return exp(-altitude / scale_height);
}

float GetOzoneDensity(float altitude) {
    float layer_center = 25000.0;
    float layer_width = 15000.0;
    if (altitude < 10000.0 || altitude > 40000.0) return 0.0;
    return max(0.0, 1.0 - abs(altitude - layer_center) / layer_width);
}

void GetRMuFromTransmittanceTextureUv(vec2 uv, out float r, out float mu) {
    float H = sqrt(top_radius * top_radius - bottom_radius * bottom_radius);
    
    float x_mu = GetUnitRangeFromTextureCoord(uv.x, TRANSMITTANCE_TEXTURE_WIDTH);
    float x_r = GetUnitRangeFromTextureCoord(uv.y, TRANSMITTANCE_TEXTURE_HEIGHT);
    
    float rho = H * x_r;
    r = sqrt(rho * rho + bottom_radius * bottom_radius);
    
    float d_min = top_radius - r;
    float d_max = rho + H;
    float d = d_min + x_mu * (d_max - d_min);
    
    mu = (d == 0.0) ? 1.0 : (H * H - rho * rho - d * d) / (2.0 * r * d);
    mu = clamp(mu, -1.0, 1.0);
}

vec3 ComputeTransmittanceToTopAtmosphereBoundary(float r, float mu) {
    const int SAMPLE_COUNT = 500;
    
    float dx = DistanceToTopAtmosphereBoundary(r, mu) / float(SAMPLE_COUNT);
    vec3 optical_depth = vec3(0.0);
    
    for (int i = 0; i < SAMPLE_COUNT; ++i) {
        float d_i = (float(i) + 0.5) * dx;
        float r_i = sqrt(d_i * d_i + 2.0 * r * mu * d_i + r * r);
        float altitude = r_i - bottom_radius;
        
        float rayleigh_density = GetDensityExponential(altitude, rayleigh_scale_height);
        float mie_density = GetDensityExponential(altitude, mie_scale_height);
        float ozone_density = GetOzoneDensity(altitude);
        
        optical_depth += (
            rayleigh_scattering * rayleigh_density +
            mie_extinction * mie_density +
            absorption_extinction * ozone_density
        ) * dx;
    }
    
    return exp(-optical_depth);
}

void main() {
    float r, mu;
    GetRMuFromTransmittanceTextureUv(uv, r, mu);
    
    vec3 transmittance = ComputeTransmittanceToTopAtmosphereBoundary(r, mu);
    fragColor = vec4(transmittance, 1.0);
}
