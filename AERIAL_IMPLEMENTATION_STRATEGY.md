# Bruneton Aerial Perspective - Step-by-Step Implementation Strategy

## Current Status (2026-01-08 10:30)

### Completed Steps
- **Step 1.1-1.5**: Transmittance pipeline - VALIDATED
- **Step 2.1-2.3**: Sun params, UV mapping, scattering sampling - VALIDATED
- **Step 2.4**: Full inscatter computation - **VALIDATED** (B/R=1.59, correct Rayleigh blue)
- **Step 3.1**: Rayleigh phase function - **VALIDATED** (responds to sun position)
- **Step 3.2**: Mie phase function - **WORKING** (Henyey-Greenstein g=0.8)

### Key Fixes Applied
1. Proper Bruneton u_mu formula (not simplified `(1+mu)/2`)
2. Depth slice interpolation (fixed stair-stepping)
3. Mix node indices: inputs[6,7] for colors, outputs[2] for color result
4. RGB clamp via SeparateColor + MAXIMUM + CombineColor
5. Sun direction read from scene Sun light object
6. Blender Sun rotation X=120° = 30° elevation (X=0 points DOWN)

### Known Limitations
- **Transmittance**: Using simplified exponential approximation, not full LUT-based
- Minor foreground stair-stepping (acceptable for close objects)

### Pending
- **Step 4**: Apply transmittance to object surface (T × surface_color)
- **Step 5**: AOV outputs (Sky, Transmittance, Rayleigh, Mie, Sun Disk)
- **Step 6**: Integration with addon creative controls
- **Step 7**: Validate sky material consistency (main branch)

## Document Status

| Step | Status | Notes |
|------|--------|-------|
| 1.1 Distance | ✅ Complete | Mean error 4.6m, median 0.25% |
| 1.2 r, mu | ✅ Complete | mu: 0.1% error |
| 1.3 Trans UV | ✅ Complete | U in [0,1] for atmos rays |
| 1.4 Trans Sample | ✅ Complete | R>G>B for long paths |
| 1.5 Point Trans | ✅ Complete | Mean ~0.98 for nearby |
| 2.1 mu_s, nu | ✅ Complete | Sun/view angle params |
| 2.2 Scatter UV | ✅ Complete | 4D UV with Y-flip |
| 2.3 Scatter Sample | ✅ Complete | B>G>R Rayleigh correct |
| 2.4 Inscatter | ✅ Complete | B/R=1.59 validated |
| 3.1 Rayleigh Phase | ✅ Complete | (3/16π)(1 + nu²) |
| 3.2 Mie Phase | ✅ Complete | Henyey-Greenstein g=0.8 |
| 3.3 Apply Phase | ✅ Complete | Rayleigh×RGB + Mie×Alpha |
| 4.1 Surface Trans | 🔴 Pending | T × surface_color |
| 5.1 AOV: Sky | 🔴 Pending | Sky without sun disk |
| 5.2 AOV: Trans | 🔴 Pending | Transmittance pass |
| 5.3 AOV: Rayleigh | 🔴 Pending | Rayleigh component |
| 5.4 AOV: Mie | 🔴 Pending | Mie component |
| 5.5 AOV: Sun Disk | 🔴 Pending | Sun disk only |
| 6.1 Sun Light Link | 🔴 Pending | Read from scene Sun |
| 6.2 Creative Sync | 🔴 Pending | Share params with sky |

---

## Validation Data

| Resource | Location | Purpose |
|----------|----------|---------|
| **TEST_SCENE_REFERENCE.md** | Repository root | Scene geometry extracted from Position AOV |
| **Reference EXR** | `tests/base_render_reference_noAtmos_v001_01.exr` | Source Position data |
| **EXR Analyzer** | `tests/analyze_scene_exr.py` | Re-analyze if scene changes |

---

## Overview

This document outlines a methodical approach to implementing the Bruneton atmospheric haze (aerial perspective) in Blender. Each step is designed to be independently testable before moving on.

**Primary Reference**: `reference/atmospheric-scattering-2-export/atmosphere/functions.glsl`

**Core Function**: `GetSkyRadianceToPoint()` (lines 1787-1863)

**Core Equation** (from Bruneton):
```
L_final = L_surface * T(camera→point) + S(camera→point)
```
Where:
- `L_surface` = original surface radiance (beauty pass)
- `T` = transmittance along view ray (how much light is absorbed/scattered out)
- `S` = inscattered light along view ray (light scattered INTO the view ray)

---

## Testing Approach

**We do NOT need AOVs for testing.** Each step outputs directly to material emission:

```
[Intermediate Value] → [Emission Shader] → [Material Output]
```

This allows us to:
1. Render and inspect pixel values
2. Compare against reference values
3. Iterate quickly without AOV infrastructure

---

## Constants (from Bruneton)

**Reference**: `reference/atmospheric-scattering-2-export/atmosphere/constants.h`

```
TRANSMITTANCE_TEXTURE_WIDTH  = 256
TRANSMITTANCE_TEXTURE_HEIGHT = 64

SCATTERING_TEXTURE_R_SIZE    = 32
SCATTERING_TEXTURE_MU_SIZE   = 128
SCATTERING_TEXTURE_MU_S_SIZE = 32
SCATTERING_TEXTURE_NU_SIZE   = 8

SCATTERING_TEXTURE_WIDTH  = 256  (NU_SIZE * MU_S_SIZE)
SCATTERING_TEXTURE_HEIGHT = 128  (MU_SIZE)
SCATTERING_TEXTURE_DEPTH  = 32   (R_SIZE)

bottom_radius = 6360.0 km (Earth surface)
top_radius    = 6420.0 km (top of atmosphere)
```

---

## Prerequisites (Assumed Working)

Before starting, we assume these are already validated:
- [x] LUT precomputation produces valid `transmittance.exr`, `scattering.exr`
- [x] Sky shader renders correctly and matches reference at various sun angles
- [x] Basic Blender integration (addon loads, UI works)

---

## Test Scene Reference

**Reference File**: `TEST_SCENE_REFERENCE.md` (auto-generated from Position AOV)
**Source EXR**: `tests/base_render_reference_noAtmos_v001_01.exr`

### Scene Geometry (Extracted from Position AOV)

| Property | Value |
|----------|-------|
| Camera Position | (-3.5, -8.5, 218.8) meters |
| Camera Altitude | ~218m above ground |
| Resolution | 960×540 |

### Distance Distribution

| Statistic | Distance | Distance (km) |
|-----------|----------|---------------|
| Minimum | 161m | 0.16km |
| 5th percentile | 182m | 0.18km |
| 25th percentile | 206m | 0.21km |
| **Median** | **220m** | **0.22km** |
| 75th percentile | 226m | 0.23km |
| 95th percentile | 683m | 0.68km |
| **Maximum** | **10,645m** | **10.6km** |

### Scene Bounds

| Axis | Min | Max | Range |
|------|-----|-----|-------|
| X | -2,895m | 51m | 2,946m |
| Y | -89m | 10,235m | 10,324m |
| Z | -1m | 419m | 420m |

### Coverage

- **Geometry**: 73.6% of pixels
- **Sky**: 26.4% of pixels

### Expected Aerial Perspective Behavior

| Distance Range | Transmittance | Expected Appearance |
|----------------|---------------|---------------------|
| < 200m (near) | ~0.99+ | Nearly original color |
| 200m - 700m | ~0.95-0.99 | Slight haze, reduced contrast |
| > 700m | < 0.95 | Visible haze |
| ~10km (max) | ~0.85 | Heavy haze, blending with sky |

### Validation Approach

**Step 1.1 (Distance)**: Output `distance / 10.6km` as emission color
- Near geometry (~200m): R ≈ 0.02 (dark)
- Mid geometry (~700m): R ≈ 0.07
- Far geometry (~10km): R ≈ 1.0 (bright)
- Sky: Black (filtered out)

**Step 1.5 (Transmittance)**: Objects fade toward black with distance
- Near (~200m): almost original brightness
- Far (~10km): significantly darkened, slight blue tint

**Step 2.5 (Full Aerial)**: Objects blend into sky
- Near: original color
- Far: heavy haze, almost blending with sky color

### Render Checklist

When sharing a test render, include:
1. ☐ Which step is being tested
2. ☐ What output is connected to emission
3. ☐ Sun position if relevant
4. ☐ Any scene changes from reference

---

## Phase 1: Transmittance Only (No Inscatter)

**Goal**: Objects fade toward black with distance, matching the transmittance LUT.

### Step 1.1: Distance Calculation

**What**: Calculate the distance from camera to surface point in kilometers.

**Blender Implementation**:
```python
# Inputs
world_position = Geometry.Position  # meters
camera_position = camera.location   # meters

# Convert to km and compute distance
distance_m = length(world_position - camera_position)
distance_km = distance_m * 0.001
```

**Validation**:
1. Create test scene with spheres at 1km, 5km, 10km, 20km
2. Output: `emission_color = (distance_km / 20.0, 0, 0)`
3. **Expected pixel values**:
   - 1km sphere: R ≈ 0.05
   - 5km sphere: R ≈ 0.25
   - 10km sphere: R ≈ 0.5
   - 20km sphere: R ≈ 1.0

---

### Step 1.2: r, mu Calculation (View Ray Parameters)

**What**: Convert camera position and view direction to Bruneton's (r, mu) parameters.

**Bruneton Reference** (`functions.glsl` lines 1797-1815):
```glsl
// camera = position relative to planet center (in km)
// view_ray = normalized direction from camera to point

Length r = length(camera);              // distance from planet center
Length rmu = dot(camera, view_ray);     // r * mu
Number mu = rmu / r;                    // cos(zenith angle)
```

**Blender Implementation**:
```python
# Planet center in Blender coords (Z-up)
planet_center = (0, 0, -6360000)  # -6360 km in meters

# Camera position relative to planet center (convert to km)
camera_rel = (camera.location - planet_center) * 0.001  # km
r = length(camera_rel)  # Should be ~6360.5 for 500m altitude

# View direction (normalized)
view_dir = normalize(world_position - camera.location)

# mu = cos(angle between view ray and "up" direction from planet center)
# "up" at camera = normalize(camera_rel)
up_at_camera = normalize(camera_rel)
mu = dot(view_dir, up_at_camera)
```

**Validation**:
1. Output: `emission_color = (r / 6400.0, (mu + 1) / 2, 0)`
2. **Expected for camera at 500m altitude**:
   - R channel ≈ 0.994 (r ≈ 6360.5 km)
   - G channel varies 0→1 based on view direction (0.5 = horizontal)

---

### Step 1.3: Transmittance UV Mapping

**What**: Convert (r, mu) to UV coordinates for sampling `transmittance.exr`.

**Bruneton Reference** (`functions.glsl` lines 402-421):
```glsl
vec2 GetTransmittanceTextureUvFromRMu(AtmosphereParameters atmosphere,
    Length r, Number mu) {
  // Distance to top atmosphere boundary for a horizontal ray at ground level.
  Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
      atmosphere.bottom_radius * atmosphere.bottom_radius);
  // Distance to the horizon.
  Length rho = SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
  // Distance to the top atmosphere boundary for the ray (r,mu), and its minimum
  // and maximum values over all mu - obtained for (r,1) and (r,mu_horizon).
  Length d = DistanceToTopAtmosphereBoundary(atmosphere, r, mu);
  Length d_min = atmosphere.top_radius - r;
  Length d_max = rho + H;
  Number x_mu = (d - d_min) / (d_max - d_min);
  Number x_r = rho / H;
  return vec2(GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_TEXTURE_WIDTH),
              GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_TEXTURE_HEIGHT));
}
```

**Supporting Functions** (`functions.glsl` lines 207-214, 342-348):
```glsl
Length DistanceToTopAtmosphereBoundary(AtmosphereParameters atmosphere,
    Length r, Number mu) {
  Area discriminant = r * r * (mu * mu - 1.0) +
      atmosphere.top_radius * atmosphere.top_radius;
  return ClampDistance(-r * mu + SafeSqrt(discriminant));
}

Number GetTextureCoordFromUnitRange(Number x, int texture_size) {
  return 0.5 / Number(texture_size) + x * (1.0 - 1.0 / Number(texture_size));
}
```

**Blender Implementation**:
```python
# Constants
bottom_radius = 6360.0  # km
top_radius = 6420.0     # km
TRANSMITTANCE_WIDTH = 256
TRANSMITTANCE_HEIGHT = 64

# H = distance to top for horizontal ray at ground
H = sqrt(top_radius² - bottom_radius²)  # ≈ 797.66 km

# rho = distance to horizon
rho = sqrt(max(r² - bottom_radius², 0))

# d = distance to top atmosphere boundary
discriminant = r² * (mu² - 1) + top_radius²
d = max(0, -r * mu + sqrt(max(discriminant, 0)))

# Mapping
d_min = top_radius - r
d_max = rho + H
x_mu = (d - d_min) / (d_max - d_min)
x_r = rho / H

# Texture coord with half-texel offset
u = 0.5/TRANSMITTANCE_WIDTH + x_mu * (1 - 1/TRANSMITTANCE_WIDTH)
v = 0.5/TRANSMITTANCE_HEIGHT + x_r * (1 - 1/TRANSMITTANCE_HEIGHT)
```

**Validation**:
1. Output: `emission_color = (u, v, 0)`
2. **Expected**: Both U and V in range [0, 1]
3. **Expected for camera at 500m looking up (mu=1)**:
   - d ≈ 59.5 km (atmosphere thickness)
   - x_mu ≈ 0 (short path)
   - U ≈ 0.002
4. **Expected for camera at 500m looking horizontal (mu≈0)**:
   - d much larger
   - x_mu closer to 1
   - U closer to 1

---

### Step 1.4: Sample Transmittance LUT

**What**: Sample `transmittance.exr` using the computed UVs.

**Blender Implementation**:
```python
# Use Image Texture node
# - Image: transmittance.exr
# - Interpolation: Linear
# - Extension: Extend
# - UV input: from Step 1.3
```

**Validation**:
1. Output: `emission_color = sampled_transmittance_rgb`
2. **Expected for looking up (short path)**: Nearly white (0.95+, 0.95+, 0.95+)
3. **Expected for looking horizontal**: Darker, with blue > green > red (more red absorption)
4. **Cross-check**: Sample same UV in Nuke, compare values

---

### Step 1.5: Transmittance Between Two Points

**What**: Compute transmittance from camera to a specific surface point.

**Bruneton Reference** (`functions.glsl` lines 493-519):
```glsl
DimensionlessSpectrum GetTransmittance(
    AtmosphereParameters atmosphere,
    TransmittanceTexture transmittance_texture,
    Length r, Number mu, Length d, bool ray_r_mu_intersects_ground) {
  
  Length r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
  Number mu_d = ClampCosine((r * mu + d) / r_d);

  if (ray_r_mu_intersects_ground) {
    return min(
        GetTransmittanceToTopAtmosphereBoundary(atmosphere, transmittance_texture, r_d, -mu_d) /
        GetTransmittanceToTopAtmosphereBoundary(atmosphere, transmittance_texture, r, -mu),
        DimensionlessSpectrum(1.0));
  } else {
    return min(
        GetTransmittanceToTopAtmosphereBoundary(atmosphere, transmittance_texture, r, mu) /
        GetTransmittanceToTopAtmosphereBoundary(atmosphere, transmittance_texture, r_d, mu_d),
        DimensionlessSpectrum(1.0));
  }
}
```

**Key Insight**: Transmittance between camera and point = T(camera→top) / T(point→top)

**Blender Implementation**:
```python
# Given: r (camera radius), mu (view angle), d (distance to point in km)

# Compute r and mu at the point
r_d = clamp(sqrt(d² + 2*r*mu*d + r²), bottom_radius, top_radius)
mu_d = clamp((r*mu + d) / r_d, -1, 1)

# Sample transmittance at both positions
T_camera = sample_transmittance(r, mu)      # from Step 1.4
T_point = sample_transmittance(r_d, mu_d)   # same function, different inputs

# Final transmittance (for non-ground-intersecting rays)
transmittance = min(T_camera / T_point, 1.0)
```

**Validation**:
1. Sphere at 1km: transmittance ≈ (0.99, 0.99, 0.99)
2. Sphere at 10km: transmittance ≈ (0.95, 0.97, 0.98) (redder = more absorbed)
3. Sphere at 50km: transmittance significantly darker

---

### Step 1.6: Apply Transmittance to Surface

**What**: Multiply surface color by transmittance.

**Formula**:
```
L_partial = L_surface * transmittance
```

**Validation**:
1. Colored spheres at various distances
2. Near: retain original color
3. Far: fade toward black with slight blue tint

---

## Phase 2: Inscatter

**Goal**: Add atmospheric glow/haze that increases with distance.

### Step 2.1: Sun Parameters (mu_s, nu)

**What**: Compute sun-related parameters for scattering lookup.

**Bruneton Reference** (`functions.glsl` lines 1811-1813):
```glsl
Number mu_s = dot(camera, sun_direction) / r;
Number nu = dot(view_ray, sun_direction);
```

**Definitions**:
- `mu_s` = cos(sun zenith angle at camera position)
- `nu` = cos(angle between view ray and sun direction)

**Blender Implementation**:
```python
# sun_direction = normalized vector pointing TO the sun
sun_direction = normalize(sun.location)  # or from heading/elevation

# mu_s = dot(up_at_camera, sun_direction)
up_at_camera = normalize(camera_rel)  # from Step 1.2
mu_s = dot(up_at_camera, sun_direction)

# nu = dot(view_direction, sun_direction)
nu = dot(view_dir, sun_direction)
```

**Validation**:
1. Output: `emission_color = ((mu_s+1)/2, (nu+1)/2, 0)`
2. **Expected mu_s**: Constant across frame (depends only on sun position)
   - Sun at 45° elevation → mu_s ≈ 0.707
3. **Expected nu**: Varies across frame
   - Looking toward sun → nu ≈ 1 (bright)
   - Looking away → nu ≈ -1 (dark)

---

### Step 2.2: Scattering 4D→2D UV Mapping

**What**: Convert (r, mu, mu_s, nu) to texture coordinates for `scattering.exr`.

**Bruneton Reference** (`functions.glsl` lines 773-831):
```glsl
vec4 GetScatteringTextureUvwzFromRMuMuSNu(AtmosphereParameters atmosphere,
    Length r, Number mu, Number mu_s, Number nu,
    bool ray_r_mu_intersects_ground) {
  
  Length H = sqrt(atmosphere.top_radius * atmosphere.top_radius -
      atmosphere.bottom_radius * atmosphere.bottom_radius);
  Length rho = SafeSqrt(r * r - atmosphere.bottom_radius * atmosphere.bottom_radius);
  Number u_r = GetTextureCoordFromUnitRange(rho / H, SCATTERING_TEXTURE_R_SIZE);

  Length r_mu = r * mu;
  Area discriminant = r_mu * r_mu - r * r + atmosphere.bottom_radius * atmosphere.bottom_radius;
  Number u_mu;
  if (ray_r_mu_intersects_ground) {
    Length d = -r_mu - SafeSqrt(discriminant);
    Length d_min = r - atmosphere.bottom_radius;
    Length d_max = rho;
    u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(d_max == d_min ? 0.0 :
        (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
  } else {
    Length d = -r_mu + SafeSqrt(discriminant + H * H);
    Length d_min = atmosphere.top_radius - r;
    Length d_max = rho + H;
    u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
        (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
  }

  Length d = DistanceToTopAtmosphereBoundary(atmosphere, atmosphere.bottom_radius, mu_s);
  Length d_min = atmosphere.top_radius - atmosphere.bottom_radius;
  Length d_max = H;
  Number a = (d - d_min) / (d_max - d_min);
  Length D = DistanceToTopAtmosphereBoundary(atmosphere, atmosphere.bottom_radius, atmosphere.mu_s_min);
  Number A = (D - d_min) / (d_max - d_min);
  Number u_mu_s = GetTextureCoordFromUnitRange(
      max(1.0 - a / A, 0.0) / (1.0 + a), SCATTERING_TEXTURE_MU_S_SIZE);

  Number u_nu = (nu + 1.0) / 2.0;
  return vec4(u_nu, u_mu_s, u_mu, u_r);
}
```

**Ground Intersection Test** (`functions.glsl` lines 240-246):
```glsl
bool RayIntersectsGround(AtmosphereParameters atmosphere, Length r, Number mu) {
  return mu < 0.0 && r * r * (mu * mu - 1.0) +
      atmosphere.bottom_radius * atmosphere.bottom_radius >= 0.0;
}
```

**This is complex - implement carefully with validation at each sub-step.**

---

### Step 2.3: Sample Scattering LUT with Nu Interpolation

**What**: Sample scattering texture with proper 4D addressing.

**Bruneton Reference** (`functions.glsl` lines 958-976):
```glsl
AbstractSpectrum GetScattering(AtmosphereParameters atmosphere,
    AbstractScatteringTexture scattering_texture,
    Length r, Number mu, Number mu_s, Number nu,
    bool ray_r_mu_intersects_ground) {
  vec4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(
      atmosphere, r, mu, mu_s, nu, ray_r_mu_intersects_ground);
  Number tex_coord_x = uvwz.x * Number(SCATTERING_TEXTURE_NU_SIZE - 1);
  Number tex_x = floor(tex_coord_x);
  Number lerp = tex_coord_x - tex_x;
  vec3 uvw0 = vec3((tex_x + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
      uvwz.z, uvwz.w);
  vec3 uvw1 = vec3((tex_x + 1.0 + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE),
      uvwz.z, uvwz.w);
  return AbstractSpectrum(texture(scattering_texture, uvw0) * (1.0 - lerp) +
      texture(scattering_texture, uvw1) * lerp);
}
```

**Key Points**:
- Nu requires interpolation between two texture samples
- The texture is 3D (packed from 4D), indexed as (x, y, z) = (nu*mu_s, mu, r)

---

### Step 2.4: GetSkyRadianceToPoint Implementation

**What**: Full aerial perspective scattering calculation.

**Bruneton Reference** (`functions.glsl` lines 1787-1863) - THE KEY FUNCTION:
```glsl
RadianceSpectrum GetSkyRadianceToPoint(
    AtmosphereParameters atmosphere,
    TransmittanceTexture transmittance_texture,
    ReducedScatteringTexture scattering_texture,
    ReducedScatteringTexture single_mie_scattering_texture,
    Position camera, Position point, Length shadow_length,
    Direction sun_direction, out DimensionlessSpectrum transmittance) {
  
  Direction view_ray = normalize(point - camera);
  Length r = length(camera);
  Length rmu = dot(camera, view_ray);
  
  // ... handle viewer in space case ...
  
  Number mu = rmu / r;
  Number mu_s = dot(camera, sun_direction) / r;
  Number nu = dot(view_ray, sun_direction);
  Length d = length(point - camera);
  bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);

  // Get transmittance from camera to point
  transmittance = GetTransmittance(atmosphere, transmittance_texture,
      r, mu, d, ray_r_mu_intersects_ground);

  // Sample scattering at camera position
  IrradianceSpectrum single_mie_scattering;
  IrradianceSpectrum scattering = GetCombinedScattering(
      atmosphere, scattering_texture, single_mie_scattering_texture,
      r, mu, mu_s, nu, ray_r_mu_intersects_ground,
      single_mie_scattering);

  // Compute parameters at point position
  Length r_p = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
  Number mu_p = (r * mu + d) / r_p;
  Number mu_s_p = (r * mu_s + d * nu) / r_p;

  // Sample scattering at point position
  IrradianceSpectrum single_mie_scattering_p;
  IrradianceSpectrum scattering_p = GetCombinedScattering(
      atmosphere, scattering_texture, single_mie_scattering_texture,
      r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,
      single_mie_scattering_p);

  // Inscatter = scattering_camera - transmittance * scattering_point
  scattering = scattering - transmittance * scattering_p;
  single_mie_scattering = single_mie_scattering - transmittance * single_mie_scattering_p;

  // Apply phase functions
  return scattering * RayleighPhaseFunction(nu) + 
         single_mie_scattering * MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
}
```

---

## Phase 3: Phase Functions

### Step 3.1: Rayleigh Phase Function

**Bruneton Reference** (`functions.glsl` lines 739-742):
```glsl
InverseSolidAngle RayleighPhaseFunction(Number nu) {
  InverseSolidAngle k = 3.0 / (16.0 * PI * sr);
  return k * (1.0 + nu * nu);
}
```

**Values**:
- nu = 1 (toward sun): k * 2 ≈ 0.1194
- nu = 0 (perpendicular): k * 1 ≈ 0.0597
- nu = -1 (away from sun): k * 2 ≈ 0.1194

---

### Step 3.2: Mie Phase Function

**Bruneton Reference** (`functions.glsl` lines 744-747):
```glsl
InverseSolidAngle MiePhaseFunction(Number g, Number nu) {
  InverseSolidAngle k = 3.0 / (8.0 * PI * sr) * (1.0 - g * g) / (2.0 + g * g);
  return k * (1.0 + nu * nu) / pow(1.0 + g * g - 2.0 * g * nu, 1.5);
}
```

**Values for g=0.8**:
- nu = 1 (toward sun): VERY LARGE (forward scattering peak)
- nu = 0: moderate
- nu = -1 (away): small

---

### Step 3.3: Combined Scattering with Mie Extrapolation

**Bruneton Reference** (`functions.glsl` lines 1633-1644):
```glsl
#ifdef COMBINED_SCATTERING_TEXTURES
vec3 GetExtrapolatedSingleMieScattering(
    AtmosphereParameters atmosphere, vec4 scattering) {
  if (scattering.r <= 0.0) {
    return vec3(0.0);
  }
  return scattering.rgb * scattering.a / scattering.r *
      (atmosphere.rayleigh_scattering.r / atmosphere.mie_scattering.r) *
      (atmosphere.mie_scattering / atmosphere.rayleigh_scattering);
}
#endif
```

**Note**: This extrapolates full RGB Mie from the red channel stored in alpha.

---

## Phase 4: Ground Intersection & Edge Cases

### Step 4.1: Ray-Ground Intersection

**Bruneton Reference** (`functions.glsl` lines 240-246):
```glsl
bool RayIntersectsGround(AtmosphereParameters atmosphere, Length r, Number mu) {
  return mu < 0.0 && r * r * (mu * mu - 1.0) +
      atmosphere.bottom_radius * atmosphere.bottom_radius >= 0.0;
}
```

---

## Known Pitfalls

1. **Coordinate systems**: Blender is Z-up, Bruneton reference is Y-up in some docs
2. **Units**: Bruneton uses km internally, Blender uses meters - convert with * 0.001
3. **Planet center**: Must be at (0, 0, -6360000) in Blender coords (meters)
4. **mu clamping**: Always clamp to [-1, 1] to avoid NaN
5. **Texture addressing**: Half-texel offsets (`0.5/size + x * (1-1/size)`) are CRITICAL
6. **Ground intersection**: Uses different u_mu mapping for below-horizon rays
7. **Division by zero**: Check for r=0, d=0, scattering.r=0 cases

---

## Validation Reference Values

For camera at (0, 0, 500m) = (0, 0, 0.5km) from planet surface:
- **r** = 6360.5 km
- **H** = sqrt(6420² - 6360²) ≈ 797.66 km
- **rho** = sqrt(6360.5² - 6360²) ≈ 79.77 km

For sun at 45° elevation:
- **mu_s** ≈ 0.707

---

## AOV Output Requirements

The final addon must output the following AOVs in a single multi-layer EXR:

| AOV | Content | Purpose |
|-----|---------|---------|
| **Sky** | Sky radiance WITHOUT sun disk | Background atmosphere |
| **Transmittance** | Per-pixel T(camera→point) | Object color attenuation |
| **Rayleigh** | Rayleigh scattering component | Blue atmospheric scatter |
| **Mie** | Mie scattering component | Forward scatter / sun halo |
| **Sun Disk** | Sun disk only (no sky) | Separate for artistic control |

**Channel Naming Convention (for Nuke):**
```
helios.sky.R, helios.sky.G, helios.sky.B
helios.transmittance.R, helios.transmittance.G, helios.transmittance.B
helios.rayleigh.R, helios.rayleigh.G, helios.rayleigh.B
helios.mie.R, helios.mie.G, helios.mie.B
helios.sun_disk.R, helios.sun_disk.G, helios.sun_disk.B
```

---

## Integration Requirements

### Sun Light Integration
- Addon must read sun position from scene's Sun light object
- Sun direction derived from Sun light's world transform matrix
- No separate/independent sun position controls in addon
- Sun light rotation X=90° = horizon, X=120° = 30° elevation

### Creative Controls Sharing
The following parameters must be shared between sky material and aerial perspective:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rayleigh_strength` | Rayleigh scattering multiplier | 1.0 |
| `mie_strength` | Mie scattering multiplier | 1.0 |
| `mie_phase_g` | Henyey-Greenstein asymmetry (-1 to 1) | 0.8 |
| `mie_density` | Aerosol density multiplier | 1.0 |
| `rayleigh_density` | Molecular scattering multiplier | 1.0 |
| `exposure` | Scene exposure adjustment | 1.0 |

Both sky and aerial perspective must use:
- Same LUT textures (transmittance.exr, scattering.exr, etc.)
- Same creative parameter values
- Same sun direction from scene Sun light

### Existing Sky Material
- Working sky material exists in `main` branch
- Must verify it uses identical math to aerial perspective implementation
- Integration work to be done in separate branch to recombine developments

---

## Next Steps

1. ✅ Steps 1-3: Core pipeline (Transmittance, Inscatter, Phase Functions) - COMPLETE
2. 🔴 Step 4: Apply transmittance to object surface (T × surface_color)
3. 🔴 Step 5: Implement AOV outputs
4. 🔴 Step 6: Sun light integration (read from scene)
5. 🔴 Step 7: Creative controls sync with sky addon
6. 🔴 Step 8: Validate sky material consistency (separate branch)

