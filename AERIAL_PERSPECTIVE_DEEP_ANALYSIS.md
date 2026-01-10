# Aerial Perspective Deep Analysis

## Purpose
Complete line-by-line analysis of Bruneton's `GetSkyRadianceToPoint` function to create a bulletproof implementation plan for Blender.

---

## PART 1: Reference Implementation Analysis

### Source: `reference/functions.glsl` lines 1787-1863

### Algorithm Overview

The `GetSkyRadianceToPoint` function computes the **inscattered light** between a camera and a point in the scene. The key insight is:

```
inscatter = S_camera - transmittance × S_point
```

Where:
- `S_camera` = scattering from camera to atmosphere boundary
- `S_point` = scattering from point to atmosphere boundary  
- `transmittance` = transmittance between camera and point

### Step-by-Step Algorithm

#### Step 1: Compute View Ray and Handle Space Viewers
```glsl
Direction view_ray = normalize(point - camera);
Length r = length(camera);
Length rmu = dot(camera, view_ray);
Length distance_to_top = -rmu - sqrt(rmu*rmu - r*r + top_radius*top_radius);
if (distance_to_top > 0.0) {
    camera = camera + view_ray * distance_to_top;
    r = top_radius;
    rmu += distance_to_top;
}
```
**Key insight**: Camera position is relative to planet center. `r` is distance from planet center.

#### Step 2: Compute r, mu, mu_s, nu for CAMERA
```glsl
Number mu = rmu / r;           // cosine of view zenith angle
Number mu_s = dot(camera, sun_direction) / r;  // cosine of sun zenith angle
Number nu = dot(view_ray, sun_direction);      // cosine of view-sun angle
Length d = length(point - camera);             // distance to point
```

#### Step 3: Determine if Ray Intersects Ground
```glsl
bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);
```

**RayIntersectsGround** (lines 240-246):
```glsl
return mu < 0.0 && r*r*(mu*mu - 1.0) + bottom_radius*bottom_radius >= 0.0;
```

**CRITICAL**: This is a MATHEMATICAL determination based on viewing angle and camera altitude. It does NOT consider actual scene geometry.

#### Step 4: Compute Transmittance (camera → point)
```glsl
transmittance = GetTransmittance(atmosphere, transmittance_texture,
    r, mu, d, ray_r_mu_intersects_ground);
```

**GetTransmittance** (lines 493-519):
```glsl
Length r_d = ClampRadius(sqrt(d*d + 2.0*r*mu*d + r*r));
Number mu_d = ClampCosine((r*mu + d) / r_d);

if (ray_r_mu_intersects_ground) {
    return T(r_d, -mu_d) / T(r, -mu);  // NEGATED mu values, SWAPPED order
} else {
    return T(r, mu) / T(r_d, mu_d);    // Normal order
}
```

**WHY TWO FORMULAS?**
- The transmittance texture only stores T(r, mu) to the TOP atmosphere boundary
- For ground-intersecting rays, we need T to the BOTTOM boundary
- The trick: T(r, mu) to bottom = T(r, -mu) to top (by symmetry)
- And we swap the division order to get the ratio correctly

#### Step 5: Sample Scattering at Camera Position
```glsl
IrradianceSpectrum scattering = GetCombinedScattering(
    atmosphere, scattering_texture, single_mie_scattering_texture,
    r, mu, mu_s, nu, ray_r_mu_intersects_ground,
    single_mie_scattering);
```

**GetCombinedScattering** calls **GetScatteringTextureUvwzFromRMuMuSNu** (lines 773-831):

```glsl
// Distance to horizon
Length rho = SafeSqrt(r*r - bottom_radius*bottom_radius);
Number u_r = GetTextureCoordFromUnitRange(rho / H, R_SIZE);

// THIS IS THE CRITICAL PART - u_mu calculation
if (ray_r_mu_intersects_ground) {
    // Ground path: maps to [0, 0.5]
    Length d = -r_mu - SafeSqrt(discriminant);  // distance to GROUND
    Length d_min = r - bottom_radius;
    Length d_max = rho;
    u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(
        (d - d_min) / (d_max - d_min), MU_SIZE / 2);
} else {
    // Non-ground path: maps to [0.5, 1.0]
    Length d = -r_mu + SafeSqrt(discriminant + H*H);  // distance to TOP
    Length d_min = top_radius - r;
    Length d_max = rho + H;
    u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
        (d - d_min) / (d_max - d_min), MU_SIZE / 2);
}
```

**KEY INSIGHT**: The scattering texture is SPLIT:
- u_mu in [0, 0.5] = ground-intersecting rays (scattering to ground)
- u_mu in [0.5, 1.0] = non-ground rays (scattering to top atmosphere)

#### Step 6: Compute r_p, mu_p, mu_s_p for POINT Position
```glsl
d = max(d - shadow_length, 0.0);  // Adjust for light shafts
Length r_p = ClampRadius(sqrt(d*d + 2.0*r*mu*d + r*r));
Number mu_p = (r*mu + d) / r_p;
Number mu_s_p = (r*mu_s + d*nu) / r_p;
```

These are the parameters at the target point, computed using law of cosines.

#### Step 7: Sample Scattering at Point Position
```glsl
IrradianceSpectrum scattering_p = GetCombinedScattering(
    atmosphere, scattering_texture, single_mie_scattering_texture,
    r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,  // SAME ray_intersects_ground!
    single_mie_scattering_p);
```

**CRITICAL**: Uses the SAME `ray_r_mu_intersects_ground` value as the camera lookup! This ensures both lookups use the same texture half.

#### Step 8: Compute Final Inscatter
```glsl
scattering = scattering - shadow_transmittance * scattering_p;
single_mie_scattering = single_mie_scattering - shadow_transmittance * single_mie_scattering_p;

// Apply phase functions
return scattering * RayleighPhaseFunction(nu) + 
       single_mie_scattering * MiePhaseFunction(mie_g, nu);
```

---

## PART 2: Why Bruneton's Approach Works

### The Scattering Subtraction Trick

The scattering texture stores **cumulative scattering from point to boundary**:
- For non-ground rays: scattering from (r, mu) to top atmosphere
- For ground rays: scattering from (r, mu) to ground surface

To get scattering between camera and point:
```
S(camera→point) = S(camera→boundary) - T(camera→point) × S(point→boundary)
```

This only works if both lookups sample the SAME half of the texture (same boundary target).

### Why the Horizon Causes Problems

At the mathematical horizon:
- `ray_r_mu_intersects_ground` switches from false to true
- This causes u_mu to jump from ~1.0 to ~0.0 (or vice versa)
- Different texture halves store different physical quantities
- The subtraction S_cam - T × S_point becomes meaningless

### In the Reference Application

The reference application renders a **spherical planet**:
- Objects below the horizon are hidden by the planet surface
- Ground-intersecting rays hit actual ground geometry
- The `ray_r_mu_intersects_ground` boolean matches reality

### In Our Blender Scene

We have a **flat scene**:
- Objects below the mathematical horizon are still visible
- Rays that "intersect ground" mathematically don't hit any actual geometry
- The `ray_r_mu_intersects_ground` boolean doesn't match visible geometry

---

## PART 3: Current Implementation Issues

### Issue 1: Flat Scene vs Spherical Planet

Our scene geometry is flat, but we're using Bruneton's spherical planet model. At the horizon:
- Mathematical: ray intersects ground (mu < mu_horizon)
- Reality: ray still sees sky/objects

### Issue 2: u_mu Discontinuity

When `ray_r_mu_intersects_ground` switches:
- u_mu jumps by ~0.99 (our validation showed this)
- Scattering values change dramatically
- Result: visible banding at horizon

### Issue 3: Texture Half Mismatch

The scattering texture halves store fundamentally different data:
- Ground half [0, 0.5]: scattering to ground surface
- Non-ground half [0.5, 1.0]: scattering to top atmosphere

Blending between them produces physically incorrect results.

---

## PART 4: Possible Solutions

### Solution A: Always Use Non-Ground (V36 approach)
- Use only the non-ground scattering formula
- Problem: Produces incorrect values below horizon (our validation showed clamping to u_mu=1.0)

### Solution B: Blend Both Halves (V37 approach)
- Sample both texture halves and blend
- Problem: Blending physically different quantities produces incorrect results

### Solution C: Spherical Planet Geometry
- Make scene geometry actually spherical
- Problem: Impractical for VFX production

### Solution D: Different Parameterization
- Use a different texture parameterization that doesn't have the ground/non-ground split
- Would require regenerating LUTs
- This is what Sebastien Hillaire did for UE4

### Solution E: Distance-Based Fallback
- For near objects: use Bruneton inscatter (geometry is above horizon)
- For far objects: simple exponential fog fallback
- Problem: Visible transition, not physically accurate

### Solution F: Re-examine the Problem
- **The sky shader works correctly** - validates our LUTs
- Maybe aerial perspective needs a fundamentally different approach for flat scenes
- Sebastien Hillaire's UE4 approach uses a 3D LUT indexed by (distance, height, cos_view_angle)

---

## PART 4B: Critical Finding - Reference Demo Geometry

### Object Positioning in Reference Demo

From `demo.glsl` and `demo.cc`:

```glsl
const float kLengthUnitInMeters = 1000.0;  // 1 unit = 1 km
const vec3 kSphereCenter = vec3(0.0, 0.0, 1000.0) / kLengthUnitInMeters;  // (0, 0, 1) km
earth_center = (0, 0, -kBottomRadius / kLengthUnitInMeters);  // (0, 0, -6360) km
```

**The sphere floats 1km ABOVE the ground surface!**

### Camera Setup

```cpp
view_distance_meters_ = 9000.0;  // 9 km from origin
view_zenith_angle_radians_ = 1.47;  // ~84° from vertical (looking slightly down)
```

### Ground Rendering

The reference demo renders ground by intersecting the view ray with a **spherical planet**:

```glsl
float discriminant = earth_center.z * earth_center.z - ray_earth_center_squared_distance;
if (discriminant >= 0.0) {
    // Ray intersects the spherical planet
    vec3 point = camera + view_direction * distance_to_intersection;
    // ... GetSkyRadianceToPoint called for this ground point
}
```

### Why This Works in Reference But Not For Us

| Aspect | Reference Demo | Our Blender Scene |
|--------|---------------|-------------------|
| Object position | 1km above ground (above horizon) | Arbitrary (can be at/below horizon) |
| Ground geometry | Spherical planet | Flat plane |
| Ground intersection | Ray truly hits planet sphere | Flat geometry visible below horizon |
| `ray_r_mu_intersects_ground` | Matches reality | Doesn't match visible geometry |

### The Fundamental Mismatch

In reference demo:
- When ray mathematically intersects ground → it ACTUALLY hits spherical planet geometry
- The ground scattering formula is CORRECT for points on the planet surface

In our scene:
- When ray mathematically intersects ground → geometry is still visible (flat scene)
- Using ground formula for visible geometry produces wrong results
- Using non-ground formula for below-horizon geometry also produces wrong results

**This is not a bug in our implementation. It's a fundamental incompatibility between Bruneton's spherical planet model and our flat scene geometry.**

---

## PART 5: Questions That Need Answers

1. **What does the reference demo actually render?**
   - Is the sphere/cube in the demo ABOVE the horizon?
   - What happens if we render objects BELOW the mathematical horizon in the reference?

2. **Do our LUTs have boundary continuity?**
   - Does Scattering(u_mu=0.5-epsilon) ≈ Scattering(u_mu=0.5+epsilon)?
   - If not, blending will never work

3. **Is the Hillaire approach viable?**
   - UE4 uses a volumetric approach with a 3D froxel LUT
   - More suited to game engines but could work for offline rendering

4. **Can we modify the LUT precomputation?**
   - Generate a seamless texture without the ground/non-ground split
   - Would require deep changes to precomputation

---

## PART 6: Next Steps

Before any implementation:

1. **Test reference demo with objects below horizon**
   - See if it exhibits the same artifacts
   
2. **Validate LUT boundary continuity**
   - Sample scattering texture at u_mu = 0.499 and 0.501
   - If different, blending can never work

3. **Study Sebastien Hillaire's UE4 paper**
   - "A Scalable and Production Ready Sky and Atmosphere Rendering Technique"
   - May provide alternative approach

4. **Consider hybrid approach**
   - Use Bruneton for sky (working)
   - Use simpler volumetric fog for aerial perspective on geometry

---

*Document created during deep analysis session. No implementation until plan is complete and approved.*
