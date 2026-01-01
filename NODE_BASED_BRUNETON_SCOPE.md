# Node-Based Bruneton Aerial Perspective - Full Scope Analysis

## Executive Summary

This document analyzes what is required to implement **complete Bruneton parity** in a node-based 
Blender shader, without OSL. This is necessary because AOV Output nodes do not work when OSL is 
enabled (known Blender bug #79942, #100282).

**Conclusion: Full Bruneton IS achievable in nodes, but requires significant complexity.**

---

## Reference Implementation Analysis

### Our OSL aerial_perspective.osl (Full Bruneton)
- ✅ Transmittance texture lookup with proper UV mapping
- ✅ 4D Scattering texture lookup with nu interpolation  
- ✅ Combined scattering (Rayleigh + Mie extrapolation)
- ✅ Rayleigh and Mie phase functions
- ✅ GetSkyRadianceToPoint algorithm
- ⚠️ Shadow length (marked TODO, not implemented)

### atmospheric-scattering-3's bruneton_node_group.py (Simplified)
- ❌ Uses simple Beer-Lambert: `T = exp(-distance × extinction)`
- ❌ Uses fog approximation: `inscatter = sky_color × (1 - T)`
- ✅ Phase functions (Rayleigh, simplified Mie HG)
- ❌ Does NOT implement GetSkyRadianceToPoint
- ❌ Does NOT do proper 4D scattering lookup
- **This is NOT full Bruneton - it's an approximation**

---

## Full Bruneton Algorithm Requirements

### 1. Transmittance Texture Lookup

**Function: GetTransmittanceTextureUvFromRMu(r, mu)**

```
H = sqrt(top_radius² - bottom_radius²)
rho = sqrt(r² - bottom_radius²)
d = DistanceToTopAtmosphereBoundary(r, mu)
d_min = top_radius - r
d_max = rho + H
x_mu = (d - d_min) / (d_max - d_min)
x_r = rho / H
u = GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_WIDTH)
v = GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_HEIGHT)
```

**Node complexity: ~15 Math nodes**

### 2. GetTransmittance (between two points)

**Function: GetTransmittance(r, mu, d, ray_intersects_ground)**

```
r_d = sqrt(d² + 2×r×mu×d + r²)  # clamped to [bottom, top]
mu_d = (r×mu + d) / r_d

if ray_intersects_ground:
    T = T_boundary(r_d, -mu_d) / T_boundary(r, -mu)
else:
    T = T_boundary(r, mu) / T_boundary(r_d, mu_d)
```

**Challenge: Conditional logic**
- Solution: Compute both branches, use lerp with binary mask

**Node complexity: ~25 Math nodes + 2 texture lookups**

### 3. Scattering Texture Lookup (4D)

**Function: GetScatteringTextureUvwzFromRMuMuSNu(r, mu, mu_s, nu)**

This is the most complex part - maps 4 parameters to texture coordinates:
- `u_r` - altitude coordinate
- `u_mu` - view zenith coordinate (different for ground/sky intersection)
- `u_mu_s` - sun zenith coordinate  
- `u_nu` - sun-view angle coordinate

**Challenge: Requires interpolation between adjacent nu slices**

```
tex_coord_x = u_nu × (NU_SIZE - 1)
tex_x = floor(tex_coord_x)
lerp_factor = tex_coord_x - tex_x

uvw0_x = (tex_x + u_mu_s) / NU_SIZE
uvw1_x = (tex_x + 1 + u_mu_s) / NU_SIZE

s0 = Sample3D(uvw0_x, u_mu, u_r)
s1 = Sample3D(uvw1_x, u_mu, u_r)
result = mix(s0, s1, lerp_factor)
```

**Node complexity: ~40 Math nodes + 4 texture lookups (2 per nu slice × 2 for depth interpolation)**

### 4. GetCombinedScattering

**Function: GetCombinedScattering(...)**

Two modes:
1. **Combined textures**: Extract Mie from combined using extrapolation
2. **Separate textures**: Sample Rayleigh and Mie separately

```
if use_combined_textures:
    scattering = sample_combined
    single_mie = extrapolate_mie(scattering, rayleigh_coeff, mie_coeff)
else:
    scattering = sample_rayleigh
    single_mie = sample_mie
```

**Node complexity: ~20 Math nodes + extrapolation logic**

### 5. Phase Functions

**Rayleigh:** `3/(16π) × (1 + cos²θ)`
**Mie (HG):** `3/(8π) × (1-g²)/(2+g²) × (1+cos²θ) / (1+g²-2g×cosθ)^1.5`

**Node complexity: ~15 Math nodes**

### 6. GetSkyRadianceToPoint (Core Algorithm)

**The heart of aerial perspective:**

```
view_ray = normalize(point - camera)
distance = length(point - camera)

r = length(camera)
mu = dot(camera, view_ray) / r
mu_s = dot(camera, sun_dir) / r
nu = dot(view_ray, sun_dir)

# Transmittance from camera to point
T = GetTransmittance(r, mu, distance, false)

# Scattering from camera to infinity
scattering_cam, mie_cam = GetCombinedScattering(r, mu, mu_s, nu, ...)

# Scattering from point to infinity
r_p = length(point)
mu_p = dot(point, view_ray) / r_p
mu_s_p = dot(point, sun_dir) / r_p
scattering_point, mie_point = GetCombinedScattering(r_p, mu_p, mu_s_p, nu, ...)

# Inscatter = scattering along segment [camera, point]
rayleigh_inscatter = scattering_cam - T × scattering_point
mie_inscatter = mie_cam - T × mie_point

# Apply phase functions
inscatter = rayleigh_inscatter × RayleighPhase(nu) + mie_inscatter × MiePhase(g, nu)
```

**Node complexity: ~30 Math nodes + calls to above functions**

### 7. Shadow Length (Light Shafts/God Rays)

**From Bruneton reference:**

```
if shadow_length > 0:
    # Subtract shadow_length from distance to ignore shadowed segment
    d = max(d - shadow_length, 0)
    
    # Recompute point position at shadow boundary
    r_p = sqrt(d² + 2×r×mu×d + r²)
    mu_p = (r×mu + d) / r_p
    mu_s_p = (r×mu_s + d×nu) / r_p
    
    # Get transmittance through shadowed region
    shadow_transmittance = GetTransmittance(r, mu, d, ...)
    
    # Attenuate scattering in shadow
    scattering = scattering × shadow_transmittance
```

**Requirement:** Needs shadow map/depth from light's perspective
**Node complexity: ~20 additional Math nodes + shadow map texture**

---

## Total Node Complexity Estimate

| Component | Math Nodes | Texture Lookups |
|-----------|------------|-----------------|
| Transmittance UV | 15 | 0 |
| GetTransmittance | 25 | 2-4 |
| Scattering UV (4D) | 40 | 0 |
| Scattering Sample | 20 | 4-8 |
| GetCombinedScattering | 20 | (included above) |
| Phase Functions | 15 | 0 |
| GetSkyRadianceToPoint | 30 | (calls above) |
| Shadow Length | 20 | 1 (shadow map) |
| **TOTAL** | **~185 nodes** | **~8-12 textures** |

---

## Critical Challenges for Node Implementation

### 1. Conditional Logic
**Problem:** Nodes don't have if/else
**Solution:** Compute both branches, multiply by mask, add results
```
result = branch_true × mask + branch_false × (1 - mask)
```
**Impact:** Doubles computation for each conditional

### 2. 3D/4D Texture Sampling
**Problem:** Blender Image Texture node only samples 2D
**Solution:** Use tiled 2D atlas (we already do this for LUTs)
- Pack Z slices horizontally
- Manually interpolate between slices
**Impact:** 2× texture lookups per interpolation axis

### 3. Division Safety
**Problem:** Division by zero in edge cases
**Solution:** Use `max(denominator, epsilon)` everywhere
**Impact:** Extra Math nodes

### 4. Coordinate System Transform
**Problem:** Blender world coords → atmosphere coords (km, planet-centered)
**Solution:** Node group handles transform at entry
**Impact:** ~10 additional nodes

### 5. Dynamic Camera Position
**Problem:** Camera moves, need to update shader inputs
**Solution:** Use driver/frame change handler (we already do this)
**Impact:** Python handler code

---

## What atmospheric-scattering-3 Got Wrong

| Aspect | atmospheric-scattering-3 | Full Bruneton |
|--------|-------------------------|---------------|
| Transmittance | `exp(-d × β)` (Beer-Lambert) | LUT lookup with proper UV |
| Inscatter | `fog_color × (1-T)` | `S(cam→∞) - T × S(point→∞)` |
| Color variation | Fixed fog color | View/sun angle dependent |
| Wavelength dependence | Single extinction | RGB from LUT |
| Multiple scattering | None | Included in LUT |

**Key insight:** atmospheric-scattering-3's inscatter is fundamentally wrong for Bruneton.
It approximates with a static fog color rather than computing the integral of scattering along the ray.

---

## Implementation Options

### Option A: Full Node-Based Bruneton
**Pros:**
- Complete parity with reference
- Works with stock Blender
- AOVs work correctly

**Cons:**
- ~200+ nodes per material
- Complex to debug/maintain
- Performance impact (many texture lookups)

**Estimated effort:** 3-5 days

### Option B: Hybrid Approach
**What:** Use full Bruneton for sky (world shader with OSL), simplified aerial for objects
**Pros:**
- Faster to implement
- Less node complexity

**Cons:**
- Objects won't have full color accuracy
- Noticeable at long distances

**Estimated effort:** 1-2 days

### Option C: Bake Aerial to 3D LUT
**What:** Precompute aerial perspective into a 3D lookup texture (distance × view_angle × sun_angle)
**Pros:**
- Single texture lookup at runtime
- Fast and simple nodes

**Cons:**
- Fixed camera altitude
- Large texture memory
- Complex precomputation

**Estimated effort:** 2-3 days

---

## Recommendation

For **complete Bruneton parity** with your requirements:
1. Atmospheric haze ✓
2. Atmospheric shadows ✓
3. Blender compositor preview ✓
4. Nuke AOV workflow ✓

**I recommend Option A: Full Node-Based Bruneton**

Despite the complexity, it's the only approach that guarantees:
- Exact visual match to OSL/reference
- Working AOVs in stock Blender
- Clear path to Houdini (convert node logic back to OSL)

---

## Shadow Implementation Note

The shadow_length feature requires:
1. Shadow map from sun's perspective (Blender's shadow pass or custom)
2. Computation of shadow ray intersection with atmosphere
3. Additional scattering calculation for unshadowed segment

**This is the most complex part and may require:**
- Custom compositor setup to extract shadow data
- Multiple render passes
- Significant additional node complexity

**Recommendation:** Implement base aerial perspective first, add shadow support as Phase 2.

---

## Next Steps

1. [ ] Create node group framework with inputs/outputs matching OSL shader
2. [ ] Implement transmittance UV calculation
3. [ ] Implement transmittance lookup (GetTransmittance)
4. [ ] Implement scattering UV calculation (4D mapping)
5. [ ] Implement scattering lookup with nu interpolation
6. [ ] Implement combined scattering with Mie extrapolation
7. [ ] Implement phase functions
8. [ ] Implement GetSkyRadianceToPoint core algorithm
9. [ ] Test against OSL shader output for validation
10. [ ] Add shadow length support (Phase 2)
11. [ ] Performance optimization

---

## Verification Strategy

To ensure parity:
1. Render same scene with OSL shader (save transmittance/inscatter)
2. Render same scene with node-based shader
3. Compute per-pixel difference
4. Target: < 1% error in well-lit regions

---

*Document created: 2026-01-01*
*Author: Cascade AI*
*Project: Helios Atmospheric Scattering v4*
