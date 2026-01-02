# Aerial Perspective Reference Analysis

## Source: atmospheric-scattering-2-export/atmosphere/functions.glsl

## GetSkyRadianceToPoint Algorithm

### Inputs
- `camera`: Position relative to earth center (km)
- `point`: Surface point relative to earth center (km)
- `sun_direction`: Normalized sun direction
- `shadow_length`: For light shafts (usually 0)

### Step 1: View Ray and Distance
```glsl
Direction view_ray = normalize(point - camera);
Length r = length(camera);
Length rmu = dot(camera, view_ray);
Length d = length(point - camera);
```

### Step 2: Camera Parameters
```glsl
Number mu = rmu / r;
Number mu_s = dot(camera, sun_direction) / r;
Number nu = dot(view_ray, sun_direction);
bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);
```

### Step 3: Transmittance Lookup
```glsl
transmittance = GetTransmittance(atmosphere, transmittance_texture, r, mu, d, ray_r_mu_intersects_ground);
```

### Step 4: Scattering at Camera
```glsl
scattering_cam = GetCombinedScattering(r, mu, mu_s, nu, ray_r_mu_intersects_ground);
```

### Step 5: Point Parameters - LAW OF COSINES (CRITICAL!)
```glsl
Length r_p = ClampRadius(sqrt(d*d + 2.0*r*mu*d + r*r));  // NOT length(point)!
Number mu_p = (r*mu + d) / r_p;
Number mu_s_p = (r*mu_s + d*nu) / r_p;
```

### Step 6: Scattering at Point
```glsl
scattering_point = GetCombinedScattering(r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground);
```

### Step 7: Inscatter Calculation
```glsl
inscatter = scattering_cam - transmittance * scattering_point;
```

### Step 8: Phase Functions
```glsl
result = inscatter * RayleighPhaseFunction(nu) + single_mie * MiePhaseFunction(g, nu);
```

## GetCombinedScattering - Texture UV Mapping

### Input: (r, mu, mu_s, nu)
### Output: uvwz = (u_nu, u_mu_s, u_mu, u_r)

The scattering texture is 4D, stored as 3D with horizontal tiling:
- Width: NU_SIZE * MU_S_SIZE = 256
- Height: MU_SIZE = 128
- Depth: R_SIZE = 32 (tiled horizontally in our 2D atlas)

### UV Calculation
```glsl
vec4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(r, mu, mu_s, nu);
// uvwz.x = u_nu (view-sun angle)
// uvwz.y = u_mu_s (sun zenith)
// uvwz.z = u_mu (view zenith)
// uvwz.w = u_r (altitude)

Number tex_coord_x = uvwz.x * (NU_SIZE - 1);
Number tex_x = floor(tex_coord_x);
Number lerp = tex_coord_x - tex_x;

// Sample two adjacent nu slices and interpolate
vec3 uvw0 = vec3((tex_x + uvwz.y) / NU_SIZE, uvwz.z, uvwz.w);
vec3 uvw1 = vec3((tex_x + 1.0 + uvwz.y) / NU_SIZE, uvwz.z, uvwz.w);
result = texture(tex, uvw0) * (1 - lerp) + texture(tex, uvw1) * lerp;
```

## Key Differences from My Implementation

1. **r_p calculation**: I was using `length(point)` directly. WRONG!
   - Reference: `r_p = sqrt(d² + 2*r*mu*d + r²)` (law of cosines)

2. **mu_p calculation**: I was using `dot(point, view_ray) / r_p`. WRONG!
   - Reference: `mu_p = (r*mu + d) / r_p`

3. **mu_s_p calculation**: I was using `dot(point, sun_direction) / r_p`. WRONG!
   - Reference: `mu_s_p = (r*mu_s + d*nu) / r_p`

These formulas preserve the geometric relationships along the view ray rather than
recomputing them from the world position, which is essential for the scattering 
subtraction to work correctly.
