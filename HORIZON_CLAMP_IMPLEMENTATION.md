# Horizon-Clamp Implementation Plan

## Document Purpose

This document provides a comprehensive implementation plan for fixing the aerial perspective inscatter banding artifacts. It serves as a permanent record of:
1. The problem analysis and root cause
2. The chosen solution approach
3. Detailed implementation steps
4. Testing and validation criteria

**Created:** January 4, 2026  
**Version:** 1.0  
**Status:** Approved for implementation

---

## Part 1: Problem Statement

### Symptoms
- Highly saturated blue band visible above the mathematical horizon
- Thin bright line at the exact horizon level
- Inscatter values jump dramatically at horizon boundary
- Banding artifacts on geometry near horizon level

### Root Cause Analysis

The Bruneton atmospheric scattering model was designed for a **spherical planet** where:
1. The ground is a sphere at radius 6360 km
2. Objects "below the horizon" are hidden by the planet surface
3. Rays that mathematically intersect ground actually hit the planet geometry

Our Blender scenes have **flat geometry** where:
1. Objects at/below the mathematical horizon are still visible
2. There is no spherical planet surface to occlude them
3. The `ray_r_mu_intersects_ground` boolean doesn't match visible reality

### Technical Details

The scattering texture is split into two halves:
- **u_mu ∈ [0, 0.5]**: Ground-intersecting rays (scattering TO planet surface)
- **u_mu ∈ [0.5, 1.0]**: Non-ground rays (scattering TO top of atmosphere)

These store **physically different quantities**:
- Ground half: Cumulative scattering along ray until it hits the planet
- Non-ground half: Cumulative scattering along ray until it exits atmosphere

At the u_mu = 0.5 boundary, there is a **massive discontinuity**:
```
Validated discontinuity examples:
- u_r=0.5: Ground side = 66,602,848 | Non-ground side = 57.1 | Diff = 66,602,790
- u_r=0.9: Ground side = inf | Non-ground side = 76.3 | Diff = inf
```

Blending between these values is physically meaningless and produces artifacts.

---

## Part 2: Solution Overview

### The Horizon-Clamp Approach

**Core Idea:** Always use the non-ground half of the scattering texture by clamping the viewing angle (mu) to never go below the horizon threshold.

**Key Insight:** 
- At the mathematical horizon, mu = mu_horizon = -rho/r
- At this angle, u_mu = 0.5 exactly (boundary between halves)
- By clamping mu to this minimum value, we always sample u_mu ≥ 0.5
- Geometry holdout masks whatever would be "underground" anyway

### Why This Works

1. **Value Continuity**: At the horizon, values transition smoothly (no jump)
2. **Below Horizon**: Gets constant inscatter (the horizon value)
3. **Geometry Holdout**: Masks underground regions, so constant values are acceptable
4. **Physical Accuracy**: Preserved for all above-horizon geometry (the common case)

### Trade-offs

| Aspect | Above Horizon | Below Horizon |
|--------|---------------|---------------|
| Physical accuracy | ✓ Correct | ~ Approximated (constant) |
| Visual continuity | ✓ Smooth | ✓ Smooth (clamped) |
| Artifacts | ✓ None | ✓ None (geometry masks) |

---

## Part 3: Mathematical Foundation

### Horizon Angle Calculation

For a viewer at altitude r (distance from planet center):

```
rho = sqrt(r² - bottom_radius²)     // Distance to horizon point
mu_horizon = -rho / r                // Cosine of horizon angle (negative = looking down)
```

Where:
- `bottom_radius = 6360 km` (planet surface)
- `r = bottom_radius + altitude` (viewer distance from planet center)

Example values:
| Altitude | r (km) | rho (km) | mu_horizon |
|----------|--------|----------|------------|
| 0.5 km | 6360.5 | 79.75 | -0.01254 |
| 1 km | 6361 | 112.77 | -0.01773 |
| 10 km | 6370 | 356.93 | -0.05603 |
| 100 km | 6460 | 1131.37 | -0.17513 |

### Clamping Rule

For any viewing direction with cosine mu:
```
mu_clamped = max(mu, mu_horizon)
```

This ensures:
- mu_clamped >= mu_horizon always
- ray_r_mu_intersects_ground = false always
- u_mu >= 0.5 always (non-ground texture half)

### Application to Both Lookups

The aerial perspective algorithm requires TWO scattering lookups:
1. **Camera lookup**: At position (r, mu, mu_s, nu)
2. **Point lookup**: At position (r_p, mu_p, mu_s_p, nu)

Both mu values must be clamped:
```
mu_clamped = max(mu, mu_horizon_at_r)
mu_p_clamped = max(mu_p, mu_horizon_at_r_p)
```

Note: mu_horizon is different at each altitude, so we compute it for both r and r_p.

---

## Part 4: Implementation Steps

### Step 1: Add mu_horizon Computation

**File:** `helios/aerial_nodes.py`

**Location:** After computing r (camera altitude), add nodes to compute mu_horizon.

**Node graph:**
```
r² = r × r
bottom² = 6360 × 6360 = 40449600
rho² = r² - bottom²
rho² = max(rho², 0)           // Clamp to prevent negative sqrt
rho = sqrt(rho²)
mu_horizon = -rho / r
```

**Implementation notes:**
- Use existing NodeBuilder pattern
- Place nodes near the r computation section
- Label clearly: "mu_horizon" for debugging

### Step 2: Clamp mu for Camera Lookup

**File:** `helios/aerial_nodes.py`

**Location:** After computing mu from dot(camera, view_ray) / r

**Node graph:**
```
mu_clamped = max(mu, mu_horizon)
```

**Implementation notes:**
- Use Math node with MAXIMUM operation
- Replace all downstream uses of mu with mu_clamped for scattering
- Keep original mu for other calculations if needed

### Step 3: Clamp mu_p for Point Lookup

**File:** `helios/aerial_nodes.py`

**Location:** After computing r_p and mu_p for the point position

**Node graph:**
```
// First compute mu_horizon at r_p altitude
r_p² = r_p × r_p
rho_p² = r_p² - bottom²
rho_p² = max(rho_p², 0)
rho_p = sqrt(rho_p²)
mu_horizon_p = -rho_p / r_p

// Then clamp mu_p
mu_p_clamped = max(mu_p, mu_horizon_p)
```

**Implementation notes:**
- Similar pattern to Step 1
- Place near the r_p, mu_p computation section

### Step 4: Force Non-Ground Formula Everywhere

**File:** `helios/aerial_nodes.py`

**Changes required:**
1. Remove `ray_intersects_ground` computation entirely (or set to constant false)
2. Remove ground-formula branches in `_compute_scattering_uvwz`
3. Remove ground-formula branches in transmittance computation
4. Simplify `sample_scattering_texture` to only use non-ground path

**Specific code sections:**
- `_compute_scattering_uvwz`: Remove ground path (lines ~620-660)
- `sample_scattering_texture`: Remove blend_at_horizon logic
- Transmittance: Ensure only non-ground formula is used

### Step 5: Update Version Number

**File:** `helios/aerial_nodes.py`

```python
AERIAL_NODE_VERSION = 38  # Horizon-clamp approach for flat scene compatibility
```

### Step 6: Add Explanatory Comments

Add clear comments explaining the horizon-clamp approach:

```python
# =============================================================================
# HORIZON-CLAMP APPROACH (V38+)
# =============================================================================
# The Bruneton scattering texture has two halves:
#   - u_mu in [0, 0.5]: Ground-intersecting rays (to planet surface)
#   - u_mu in [0.5, 1.0]: Non-ground rays (to top of atmosphere)
#
# These halves are DISCONTINUOUS at u_mu = 0.5 (values differ by 10^6+).
#
# For flat scenes (not spherical planets), we clamp mu to mu_horizon
# so we always sample the non-ground half. This provides:
#   - Continuous values at the horizon
#   - Constant inscatter below horizon (acceptable for holdout geometry)
#   - Physical accuracy for all above-horizon geometry
#
# See: HORIZON_CLAMP_IMPLEMENTATION.md for full analysis
# =============================================================================
```

---

## Part 5: Testing Plan

### Test 1: Horizon Continuity

**Setup:**
- Camera at 1km altitude looking toward horizon
- Flat ground plane at Z=0
- Render with inscatter AOV

**Expected result:**
- Smooth gradient across horizon
- No banding or bright lines
- Inscatter values continuous

### Test 2: Above-Horizon Objects

**Setup:**
- Camera at 1km altitude
- Sphere floating 500m above ground (clearly above horizon)
- Compare to reference Bruneton demo

**Expected result:**
- Inscatter matches reference behavior
- Physically correct aerial perspective

### Test 3: At-Horizon Objects

**Setup:**
- Camera at 1km altitude
- Sphere at exact horizon level

**Expected result:**
- Smooth transition
- No discontinuities
- Reasonable inscatter values

### Test 4: Below-Horizon Objects (Flat Scene)

**Setup:**
- Camera at 1km altitude
- Objects placed below mathematical horizon but visible in flat scene
- Geometry holdout enabled

**Expected result:**
- Constant inscatter (clamped to horizon value)
- No artifacts
- Geometry holdout masks correctly

### Test 5: Various Sun Angles

**Setup:**
- Test with sun at high, medium, low, and below-horizon angles
- Check for artifacts at each

**Expected result:**
- Smooth inscatter at all sun angles
- No new artifacts introduced

---

## Part 6: Rollback Plan

If this implementation causes unexpected issues:

### Quick Rollback
1. Revert `aerial_nodes.py` to V37
2. Re-enable ground formula paths
3. Accept horizon banding as known limitation

### Diagnostic Mode
Add debug output to visualize:
- mu vs mu_horizon values
- Which rays are being clamped
- Raw u_mu values before/after clamping

### Alternative Approaches (If Needed)
1. **Sebastien Hillaire approach**: 3D froxel LUT (major rewrite)
2. **Distance-based fallback**: Simple fog for far objects
3. **Blender-native fog**: Use Blender's mist pass for below-horizon

---

## Part 7: Implementation Checklist

- [ ] Read current `aerial_nodes.py` to identify exact locations for changes
- [ ] Implement mu_horizon computation for camera
- [ ] Implement mu clamping for camera lookup
- [ ] Implement mu_horizon_p computation for point
- [ ] Implement mu_p clamping for point lookup
- [ ] Remove/bypass ground formula in `_compute_scattering_uvwz`
- [ ] Remove/bypass ground formula in transmittance
- [ ] Remove blend_at_horizon logic (no longer needed)
- [ ] Update version to V38
- [ ] Add explanatory comments
- [ ] Test: Horizon continuity
- [ ] Test: Above-horizon objects
- [ ] Test: At-horizon objects
- [ ] Test: Below-horizon objects
- [ ] Test: Various sun angles
- [ ] Document any unexpected behaviors

---

## Part 8: Code Reference (Exact Line Numbers from V37)

### Current Code Locations

**r computation (camera altitude):** lines 816-827
```python
r_raw = builder.vec_math('LENGTH', -1400, 300, 'r_raw')
r_min = builder.math('MAXIMUM', -1200, 300, 'r_min', v1=min_r)
r = builder.math('MINIMUM', -1000, 300, 'r', v1=TOP_RADIUS)
```

**mu computation (view zenith cosine):** lines 829-841
```python
mu = builder.math('DIVIDE', -1000, 200, 'mu')
mu_clamped = builder.math('MINIMUM', -800, 200, 'mu_clamp', v1=1.0)
mu_final = builder.math('MAXIMUM', -600, 200, 'mu_final', v1=-1.0)
```
→ **CHANGE POINT A**: Insert mu_horizon computation after r (line 827)
→ **CHANGE POINT B**: Replace mu_final with horizon clamp (line 841)

**r_p computation (point altitude):** lines 901-910
```python
r_p_raw = builder.math('SQRT', 600, -150, 'r_p_raw')
r_p_min = builder.math('MAXIMUM', 800, -150, 'r_p_min', v1=BOTTOM_RADIUS)
r_p = builder.math('MINIMUM', 1000, -150, 'r_p', v1=TOP_RADIUS)
```

**mu_p computation:** lines 912-928
```python
mu_p = builder.math('DIVIDE', 200, -300, 'μ_p')
mu_p_clamped = builder.math('MINIMUM', 400, -300, 'μ_p_clamp', v1=1.0)
mu_p_final = builder.math('MAXIMUM', 600, -300, 'μ_p_final', v1=-1.0)
```
→ **CHANGE POINT C**: Insert mu_horizon_p computation after r_p (line 910)
→ **CHANGE POINT D**: Replace mu_p_final with horizon clamp (line 928)

**ray_intersects_ground computation:** lines 952-987
```python
mu_negative = builder.math('LESS_THAN', 2000, 600, 'mu<0', v1=0.0)
# ... discriminant calculation ...
ray_intersects_ground = builder.math('MULTIPLY', 2750, 550, 'ray_hits_ground')
```
→ **CHANGE POINT E**: Remove or replace with constant 0.0

**Transmittance - Ground path:** lines 1056-1098
```python
# --- GROUND PATH: T(r_p, -mu_p) / T(r, -mu) ---
trans_uv_cam_g = create_transmittance_uv(...)
# ... entire ground transmittance calculation ...
```
→ **CHANGE POINT F**: Remove entirely (dead code after ray_intersects_ground = 0)

**Transmittance - Select nodes:** lines 1100-1116
```python
T_sel_r = builder.mix('FLOAT', 'MIX', 1400, 550, 'T_sel_r')
builder.link(ray_intersects_ground.outputs[0], T_sel_r.inputs['Factor'])
```
→ **CHANGE POINT G**: Remove select, wire non-ground directly to clamp

**Scattering lookups:** lines 1141-1149+
```python
scat_cam_color = sample_scattering_texture(
    builder, r.outputs[0], mu_final.outputs[0], ...
    ray_intersects_ground.outputs[0]  # Remove this parameter
)
```
→ **CHANGE POINT H**: Remove ray_intersects_ground parameter, simplify function

---

## Appendix A: Reference Formulas

### From Bruneton functions.glsl

**RayIntersectsGround (line 240-246):**
```glsl
bool RayIntersectsGround(AtmosphereParameters atmosphere, Length r, Number mu) {
  return mu < 0.0 && r * r * (mu * mu - 1.0) +
      atmosphere.bottom_radius * atmosphere.bottom_radius >= 0.0;
}
```

**GetScatteringTextureUvwzFromRMuMuSNu (line 773-831):**
```glsl
// Non-ground u_mu (lines 803-811):
Length d = -r_mu + SafeSqrt(discriminant + H * H);
Length d_min = atmosphere.top_radius - r;
Length d_max = rho + H;
u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(
    (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);

// Ground u_mu (lines 795-802):
Length d = -r_mu - SafeSqrt(discriminant);
Length d_min = r - atmosphere.bottom_radius;
Length d_max = rho;
u_mu = 0.5 - 0.5 * GetTextureCoordFromUnitRange(
    (d - d_min) / (d_max - d_min), SCATTERING_TEXTURE_MU_SIZE / 2);
```

**GetTransmittance with distance (line 493-519):**
```glsl
if (ray_r_mu_intersects_ground) {
    return min(T(r_d, -mu_d) / T(r, -mu), 1.0);  // Ground formula
} else {
    return min(T(r, mu) / T(r_d, mu_d), 1.0);    // Non-ground formula
}
```

---

## Appendix B: Version History

| Version | Date | Description |
|---------|------|-------------|
| V36 | Jan 3, 2026 | Always use non-ground (clamped u_mu to 0.5) |
| V37 | Jan 3, 2026 | Blend both texture halves (failed - discontinuous) |
| V38 | Jan 4, 2026 | Horizon-clamp approach (this implementation) |

---

*End of Implementation Plan*
