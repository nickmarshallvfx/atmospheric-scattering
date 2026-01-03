# Aerial Perspective Implementation Analysis

**Complete analysis of atmospheric-scattering-2 reference implementation**

---

## Key Reference Functions Read

| Function | Lines | Purpose |
|----------|-------|---------|
| `DistanceToTopAtmosphereBoundary` | 207-214 | d = -r*mu + sqrt(disc) |
| `DistanceToBottomAtmosphereBoundary` | 222-228 | d = -r*mu - sqrt(disc) ← **SUBTRACTION** |
| `RayIntersectsGround` | 240-246 | mu < 0 && r²(μ²-1) + bottom² >= 0 |
| `GetTextureCoordFromUnitRange` | 342-344 | 0.5/n + x*(1 - 1/n) |
| `GetTransmittanceTextureUvFromRMu` | 402-421 | Transmittance UV mapping |
| `GetTransmittance` | 493-518 | **Negates mu for ground rays** |
| `GetScatteringTextureUvwzFromRMuMuSNu` | 773-830 | **Two different u_mu formulas** |
| `GetScattering` | 958-976 | Texture lookup with nu interpolation |
| `GetCombinedScattering` | 1658-1690 | Combined Rayleigh+Mie lookup |
| `GetSkyRadianceToPoint` | 1787-1863 | **Main aerial perspective function** |

---

## Reference: GetSkyRadianceToPoint (functions.glsl lines 1787-1863)

### Algorithm Summary

```
1. view_ray = normalize(point - camera)
2. r = length(camera)
3. rmu = dot(camera, view_ray)
4. mu = rmu / r
5. mu_s = dot(camera, sun_direction) / r
6. nu = dot(view_ray, sun_direction)
7. d = length(point - camera)
8. ray_r_mu_intersects_ground = RayIntersectsGround(r, mu)

9. transmittance = GetTransmittance(r, mu, d, ray_r_mu_intersects_ground)

10. S_cam = GetCombinedScattering(r, mu, mu_s, nu, ray_r_mu_intersects_ground)

11. r_p = ClampRadius(sqrt(d² + 2*r*mu*d + r²))
12. mu_p = (r*mu + d) / r_p
13. mu_s_p = (r*mu_s + d*nu) / r_p

14. S_point = GetCombinedScattering(r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground)
    ^^^ SAME ray_r_mu_intersects_ground from camera!

15. inscatter = S_cam - transmittance * S_point

16. Return inscatter * RayleighPhaseFunction(nu) + mie_scattering * MiePhaseFunction(nu)
```

---

## Reference: GetTransmittance (functions.glsl lines 493-518)

### Key Insight: Different formulas for ground vs non-ground rays!

```glsl
r_d = ClampRadius(sqrt(d² + 2*r*mu*d + r²))
mu_d = ClampCosine((r*mu + d) / r_d)

if (ray_r_mu_intersects_ground) {
    // NEGATED mu values!
    return T(r_d, -mu_d) / T(r, -mu)
} else {
    // Normal mu values
    return T(r, mu) / T(r_d, mu_d)
}
```

**CRITICAL**: For ground-intersecting rays, the transmittance lookup uses **negated mu**!

---

## Reference: GetScatteringTextureUvwzFromRMuMuSNu (lines 773-830)

### u_r calculation (same for both cases):
```
H = sqrt(top² - bottom²)
rho = SafeSqrt(r² - bottom²)
u_r = GetTextureCoordFromUnitRange(rho / H, R_SIZE)
```

### u_mu calculation - TWO DIFFERENT FORMULAS:

**For ground-intersecting rays (ray_r_mu_intersects_ground = true):**
```
discriminant = r*mu*r*mu - r*r + bottom²
d = -r*mu - SafeSqrt(discriminant)    ← SUBTRACTION! Distance to GROUND
d_min = r - bottom_radius
d_max = rho
u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange((d-d_min)/(d_max-d_min), MU_SIZE/2)
       ↑ Maps to [0, 0.5]
```

**For non-ground-intersecting rays:**
```
discriminant = r*mu*r*mu - r*r + bottom²
d = -r*mu + SafeSqrt(discriminant + H*H)  ← ADDITION! Distance to TOP
d_min = top_radius - r
d_max = rho + H
u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange((d-d_min)/(d_max-d_min), MU_SIZE/2)
       ↑ Maps to [0.5, 1.0]
```

### u_mu_s calculation:
```
d = DistanceToTopAtmosphereBoundary(bottom_radius, mu_s)
d_min = top_radius - bottom_radius
d_max = H
a = (d - d_min) / (d_max - d_min)
D = DistanceToTopAtmosphereBoundary(bottom_radius, mu_s_min)
A = (D - d_min) / (d_max - d_min)
u_mu_s = GetTextureCoordFromUnitRange(max(1 - a/A, 0) / (1 + a), MU_S_SIZE)
```

### u_nu calculation:
```
u_nu = (nu + 1) / 2
```

### Return value order: vec4(u_nu, u_mu_s, u_mu, u_r)

---

## Reference: RayIntersectsGround (lines 240-246)

```glsl
bool RayIntersectsGround(r, mu) {
    return mu < 0.0 && r*r*(mu*mu - 1.0) + bottom² >= 0.0
}
```

---

## ISSUES IN CURRENT IMPLEMENTATION

### Issue 1: Ground-intersecting u_mu formula is WRONG

My current implementation in `_compute_scattering_uvwz`:
```python
# I'm using: d = -r*mu + sqrt(discriminant + top²)
# For BOTH ground and non-ground cases!

# But reference uses DIFFERENT formulas:
# Ground: d = -r*mu - sqrt(r²*mu² - r² + bottom²)  ← to GROUND
# Non-ground: d = -r*mu + sqrt(r²*mu² - r² + top²)  ← to TOP
```

The discriminant is also different:
- Ground case uses: `r²*mu² - r² + bottom²` (intersection with ground sphere)
- Non-ground case uses: `r²*mu² - r² + top²` (intersection with top atmosphere)

### Issue 2: Transmittance calculation for ground rays

Reference uses negated mu for ground-intersecting rays:
```
T = T(r_d, -mu_d) / T(r, -mu)
```

My implementation doesn't negate mu for ground rays.

### Issue 3: Scattering texture sampling

The reference uses 3D texture sampling with uvw coordinates:
```
uvw0 = vec3((tex_x + u_mu_s) / NU_SIZE, u_mu, u_r)
```

My implementation stores as 2D tiled texture and may have coordinate mapping issues.

---

## Demo Usage (demo.glsl lines 503-505)

```glsl
vec3 in_scatter = GetSkyRadianceToPoint(camera - earth_center,
    point - earth_center, shadow_length, sun_direction, transmittance);
object_radiance = object_radiance * transmittance + in_scatter;
```

**Key**: Both camera and point are relative to planet center (earth_center).

---

## CRITICAL BUGS IDENTIFIED

### Bug 1: Ground u_mu formula completely wrong

**Reference** (lines 795-802 for ground, 803-811 for non-ground):
```glsl
// GROUND-INTERSECTING (u_mu maps to [0, 0.5])
discriminant = r*mu*r*mu - r*r + bottom²  // Uses BOTTOM radius
d = -r*mu - SafeSqrt(discriminant)         // SUBTRACTION - distance to GROUND
d_min = r - bottom_radius
d_max = rho
u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(...)

// NON-GROUND (u_mu maps to [0.5, 1.0])  
discriminant = r*mu*r*mu - r*r + bottom²  // Same discriminant base
d = -r*mu + SafeSqrt(discriminant + H*H)   // ADDITION - distance to TOP
d_min = top_radius - r
d_max = rho + H
u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(...)
```

**My implementation**: Only implements non-ground formula for ALL cases.

### Bug 2: Transmittance uses negated mu for ground rays

**Reference** (lines 504-517):
```glsl
if (ray_r_mu_intersects_ground) {
    return T(r_d, -mu_d) / T(r, -mu)   // NEGATED mu!
} else {
    return T(r, mu) / T(r_d, mu_d)     // Normal mu
}
```

**My implementation**: Uses same formula (no negation) for both cases.

### Bug 3: Coordinate system verification needed

Reference uses camera/point positions relative to planet center.
Need to verify my Blender implementation matches this coordinate system.

---

## FIX PLAN

### Step 1: Fix u_mu calculation in `_compute_scattering_uvwz`

Rewrite to compute BOTH formulas and select based on `ray_intersects_ground_socket`:

```python
# For GROUND rays:
disc_ground = r²*(μ²-1) + bottom²
d_ground = -r*mu - sqrt(disc_ground)  # SUBTRACTION
d_min_ground = r - bottom
d_max_ground = rho
x_mu_ground = (d_ground - d_min_ground) / (d_max_ground - d_min_ground)
u_mu_ground = 0.5 - 0.5 * GetTextureCoordFromUnitRange(x_mu_ground, MU_SIZE/2)

# For NON-GROUND rays:
disc_nonground = r²*(μ²-1) + top²  # or equivalently disc_ground + H²
d_nonground = -r*mu + sqrt(disc_nonground)  # ADDITION
d_min_nonground = top - r
d_max_nonground = rho + H
x_mu_nonground = (d_nonground - d_min_nonground) / (d_max_nonground - d_min_nonground)
u_mu_nonground = 0.5 + 0.5 * GetTextureCoordFromUnitRange(x_mu_nonground, MU_SIZE/2)

# Select based on flag
u_mu = mix(u_mu_nonground, u_mu_ground, ray_intersects_ground)
```

### Step 2: Fix transmittance calculation

Add negation of mu for ground-intersecting rays in transmittance ratio computation.

### Step 3: Verify coordinate system

Ensure camera_km and point_km are computed correctly relative to planet center.

### Step 4: Test with known values

Create a test case with specific r, mu values and verify intermediate results match reference.
