# Aerial Perspective Failed Approaches Log

## Purpose
Document all attempted fixes (V38-V45) that failed, with analysis of why they failed.
This serves as reference for future decision-making.

---

## Background: The Original Problem

Before V38, the implementation had a **discontinuity at the horizon**. This occurred because:

1. The Bruneton scattering texture has two halves:
   - **Ground half** (u_mu in [0, 0.5]): For rays that intersect the spherical planet surface
   - **Non-ground half** (u_mu in [0.5, 1.0]): For rays that go to the top atmosphere boundary

2. The `ray_r_mu_intersects_ground` flag determines which half to sample

3. For rays exactly at the horizon (mu = mu_horizon), there's a mathematical discontinuity where the texture transitions between these halves

4. With **flat geometry** (buildings on a flat plane), rays looking down don't actually hit a spherical planet - they hit flat geometry instead. This creates a mismatch between what the Bruneton model expects and what our scene provides.

---

## V38: Horizon Clamp Approach

### What it did:
- Clamped `mu` (view zenith cosine) to `mu_horizon` minimum
- This forced all rays to be "above horizon" mathematically
- Removed the ground texture half sampling entirely
- Only used non-ground formulas for transmittance and scattering

### Why it failed:
- Created a **bright ground artifact** - values at ground level became too bright
- Introduced a **saturated blue band** above the horizon
- The transmittance ratio `T_cam / T_pt` was incorrect for ground-level geometry because:
  - The point (r_p, mu_p) was computed using law-of-cosines assuming the point is ON the clamped ray
  - But the actual geometry position doesn't match this assumption
  - This creates inconsistent transmittance values

### Lesson learned:
Clamping mu doesn't make the geometry match the spherical model. The underlying mismatch remains.

---

## V39: Adjustments to Horizon Clamp

### What it did:
- Minor adjustments to the horizon clamp approach
- Attempted to smooth the transition

### Why it failed:
- Same fundamental issues as V38
- The blue band and bright ground persisted
- No meaningful improvement

---

## V40: Below-Horizon Inscatter Blending

### What it did:
- Computed a `below_horizon_factor` based on how far below horizon the view was
- For below-horizon regions, used simplified inscatter: `S_cam × (1 - T)`
- Blended between normal inscatter and simplified inscatter based on this factor

### Why it failed:
- The simplified inscatter formula doesn't match the physically correct formula
- Created color/intensity discontinuities at the blend boundary
- The blue band was still visible
- This was treating symptoms, not the root cause (transmittance was still wrong)

---

## V41: Actual Geometry Positions

### What it did:
- Instead of using law-of-cosines to compute r_p, mu_p, mu_s_p:
- Computed actual geometry position in spherical coordinates
- Used actual length from planet center for r_p
- Used actual dot products for mu_p and mu_s_p

### Why it failed:
- **Completely broke rendering** - strong cyan band appeared
- The Bruneton inscatter formula assumes:
  ```
  inscatter = S(r, mu) - T(r, mu, d) × S(r_p, mu_p)
  ```
- This formula is ONLY valid when (r_p, mu_p) are computed using law-of-cosines FROM (r, mu, d)
- Using actual geometry positions violates this mathematical relationship
- The LUT parameterization requires consistent ray geometry

### Lesson learned:
**Critical insight**: The Bruneton model's inscatter formula has a mathematical derivation that REQUIRES the point parameters to be derived from the camera parameters via law-of-cosines. You cannot substitute actual geometry positions.

---

## V42: Revert to Law-of-Cosines

### What it did:
- Reverted V41 changes
- Back to law-of-cosines for r_p, mu_p, mu_s_p
- Still had horizon clamp from V38-V40

### Result:
- Back to V40 behavior (blue band, bright ground)
- Confirmed that V41 approach was fundamentally wrong

---

## V43: Ground Detection + Ground Transmittance Formula

### What it did:
- Detected "ground geometry" when r_p ≈ BOTTOM_RADIUS
- For ground geometry, used the GROUND transmittance formula:
  ```
  T = T(r_p, -mu_p) / T(r, -mu)  // Note: NEGATED mu, SWAPPED division
  ```
- For non-ground geometry, used normal formula:
  ```
  T = T(r, mu) / T(r_p, mu_p)
  ```

### Partial success:
- The thin discontinuity line was gone
- The saturated blue band was less intense
- Ground haze looked more natural

### Why it still had issues:
- Scattering was still using non-ground formula
- Inconsistency between transmittance (ground) and scattering (non-ground)
- This created "sky ghosting" - scattering values that looked like sky bleeding through
- Distant objects became too warm (transmittance too strong relative to inscatter)

---

## V44: Ground Scattering with Negated Mu

### What it did:
- Applied same logic as V43 transmittance to scattering
- Used -mu and -mu_p for ground scattering lookups
- Blended between ground and non-ground scattering

### Why it failed:
- **Fundamental misunderstanding**: Scattering doesn't use negated mu for ground rays
- Scattering uses DIFFERENT UV FORMULAS (different d, d_min, d_max calculations)
- Ground scattering maps to u_mu in [0, 0.5]
- Non-ground scattering maps to u_mu in [0.5, 1.0]
- The mu value itself is NOT negated - the UV computation formula changes
- Negating mu for scattering gave completely wrong texture coordinates

---

## V45: Pass is_ground to Scattering UV Selection

### What it did:
- Fixed the V44 mistake - don't negate mu for scattering
- Pass `is_ground` socket to `sample_scattering_texture`
- Let the function select between ground UV formula (u_mu in [0, 0.5]) and non-ground UV formula (u_mu in [0.5, 1.0])

### Why it failed:
- **Massive discontinuities** appeared in scattering
- The `is_ground` detection was based on r_p (geometry altitude)
- But `ray_r_mu_intersects_ground` in Bruneton is based on CAMERA parameters (r, mu)
- These are fundamentally different concepts:
  - Bruneton: "Does the camera's view RAY intersect the mathematical planet sphere?"
  - Our detection: "Is the geometry at ground level?"
- Mixing ground/non-ground formulas per-pixel based on geometry position creates discontinuities at the transition boundary

---

## Root Cause Analysis

### The fundamental mismatch:

1. **Bruneton assumes spherical planet geometry**: The scattering/transmittance textures encode values for rays that either:
   - Hit a perfect sphere at BOTTOM_RADIUS, or
   - Go to infinity at TOP_RADIUS

2. **Our scene has flat geometry**: Buildings sitting on a flat plane don't match the spherical model

3. **The inscatter formula derivation**:
   ```
   inscatter = S_cam - T × S_pt
   ```
   This is derived from:
   ```
   ∫[camera to point] scattering × transmittance ds
   ```
   The formula ONLY works when all parameters are consistently parameterized on the same ray.

4. **Horizon clamp doesn't solve the geometry mismatch**: It just forces different (also incorrect) values.

5. **Ground detection creates discontinuities**: Switching formulas per-pixel based on geometry creates visible seams.

---

## Possible Future Directions

### Option 1: Accept the discontinuity
- Implement pure Bruneton exactly as reference
- Accept that there will be a discontinuity at the horizon for flat geometry
- May be acceptable for many shots where horizon isn't prominent

### Option 2: Hillaire/UE4 approach
- Different atmospheric model designed for real-time rendering
- Uses aerial perspective LUT that's view-dependent
- May handle flat geometry better
- Requires implementing a different system

### Option 3: Geometry-aware blending
- Near the horizon, blend between aerial perspective and simpler fog
- Requires careful tuning to hide the transition
- Not physically accurate but may be visually acceptable

### Option 4: Spherical geometry proxy
- Render atmospheric effects on a sphere that matches Bruneton's model
- Composite with flat geometry
- Complex pipeline but mathematically correct

---

## Conclusion

The core issue is a **model mismatch**: Bruneton's precomputed atmospheric scattering assumes a spherical planet, but our CG scenes have flat geometry. All attempts to work around this within the Bruneton framework have introduced new artifacts because they break the mathematical assumptions the model relies on.

The next step is to implement pure Bruneton exactly as reference, accept its limitations, and then make an informed decision about whether to:
- Live with the discontinuity
- Try a different atmospheric model
- Develop a hybrid approach

---

## V128-V134: Node Injection Approach (January 2026)

### What was attempted:
Step 11 tried to combine working Step 2.4 (LUT scattering + exponential transmittance) with working Step 6 (LUT transmittance) by:
1. Running Step 2.4 to create scattering nodes
2. Finding specific nodes in the graph (d, mu, t_rgb, t_times_spt)
3. Injecting Step 6's LUT transmittance nodes
4. Connecting the new transmittance to replace exponential transmittance

### Iterations:
- **V128**: Initial implementation - node finding failed
- **V129**: Fixed UV calculation for dynamic r_d
- **V130**: Fixed d node finding (was LENGTH, should be MULTIPLY with 0.001)
- **V131**: Fixed mu node finding (MULTIPLY pass-through, not CLAMP)
- **V132**: Fixed mu_p node finding (MAXIMUM chain, not CLAMP)
- **V133**: Fixed output socket name (MATH uses 'Value', not 'Result')
- **V134**: Rewrote to build geometry from scratch - still broken

### Why all iterations failed:

1. **Fragile node-finding**: Finding nodes by type and input values is error-prone. Node graph structure can vary.

2. **Duplicate geometry nodes**: When we build new geometry nodes for transmittance, they're separate from Step 2.4's geometry nodes. Any subtle difference causes inconsistency.

3. **Variable name confusion**: V134 created a new `d` node but old code still referenced `d_node`.

4. **Fundamental architecture problem**: Trying to modify an existing node graph is inherently risky. The graph has internal dependencies that are easy to break.

### Visual symptoms:
- Very dark scene (transmittance near 1.0 everywhere → inscatter ≈ 0)
- Bright band at horizon (exponential fallback issue)
- Buildings barely visible

### Lesson learned:
**Node injection is not a viable approach.** The correct solution is to build the entire shader from scratch in one function, sharing geometry nodes between scattering and transmittance calculations.

### Resolution:
Step 2.4b - complete clean-build implementation (see STEP_2_4B_IMPLEMENTATION_SPEC.md)
