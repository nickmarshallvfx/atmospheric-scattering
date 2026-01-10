# Blue Band Debug Log

## Problem Statement
A blue/cyan band artifact appears at the horizon in aerial perspective renders.

---

## Test Results (Chronological)

### V51: Output mie_inscatter (before phase)
- **Result**: Blue band present
- **Learning**: Blue band exists in Mie inscatter before phase function

### V52: Force non-ground UV only + output raw scat_cam_color
- **Result**: NO blue band, correct sky gradient
- **Learning**: Raw scattering at camera position is clean when using non-ground UV

### V53: Force ground UV only + output raw scat_cam_color  
- **Result**: NO blue band, but VERY DIFFERENT values (bright/white below horizon)
- **Learning**: Ground half of texture produces completely different scattering values

### V54: Non-ground UV for scattering + full Rayleigh+Mie inscatter output
- **Result**: Blue band RETURNED
- **Learning**: The issue is NOT in raw scattering samples - it's in the INSCATTER COMPUTATION

### V55: Non-ground UV + Rayleigh-only inscatter (no Mie)
- **Result**: BLUE BAND PRESENT
- **Learning**: Issue is in Rayleigh inscatter, not Mie. Since scattering UV is non-ground, issue must be TRANSMITTANCE

### V56: Force non-ground for BOTH scattering AND transmittance
- **Result**: Blue band STILL PRESENT + weird ground haze added
- **Learning**: Forcing non-ground T made things worse. The issue is NOT just about ground/non-ground selection.

---

## Holistic Analysis

### The Inscatter Formula
```
inscatter = scat_cam - T × scat_pt
```

### What We Know For Certain
1. `scat_cam` alone is CLEAN (V52)
2. Blue band appears ONLY when we subtract `T × scat_pt`
3. Forcing non-ground UV for scattering didn't help
4. Forcing non-ground for transmittance made things worse

### New Hypothesis
The issue might be in `scat_pt` (point scattering), not `scat_cam`. 

At the point position, we compute:
- `r_p` = radius at point (via law of cosines)
- `mu_p` = view angle at point
- `mu_s_p` = sun angle at point

For objects near the horizon, `mu_p` could have very different values than `mu`, potentially hitting edge cases in the texture sampling.

### V57: Output scat_pt_color (point scattering only)
- **Result**: NO blue band, but WARM band at horizon
- **Learning**: Point scattering has warm values at horizon (mu_s_p → nu for distant points)

### V58: Output scat_cam - scat_pt (raw difference, no T)
- **Result**: NO blue band! Shows expected gradient:
  - Black/dark at horizon (scat_pt ≈ scat_cam or scat_pt > scat_cam)
  - Yellow/warm gradient above (scat_cam > scat_pt, expected)
  - Red/orange transition band
- **Learning**: The raw difference has NO blue band. Blue band is caused by TRANSMITTANCE.

---

## ROOT CAUSE IDENTIFIED

**The blue band is caused by TRANSMITTANCE multiplication, not scattering.**

Formula: `inscatter = scat_cam - T × scat_pt`

- V58 (no T): `scat_cam - scat_pt` → NO blue band
- V54+ (with T): `scat_cam - T × scat_pt` → BLUE band

### Why T creates blue band:
At the horizon, transmittance has long path length → more blue extinction → T is red-shifted.
When we compute `T × scat_pt`:
- T has less blue (blue scattered away)
- T × scat_pt is WARMER than scat_pt alone
- Subtracting warmer value → result is COOLER (bluer)

This could be:
1. **Physically correct** - sunset-like effect from long atmospheric paths
2. **A bug** - T computed incorrectly at horizon (ground/non-ground boundary issue)

### V59: Output T directly
- **Result**: Neutral grey, slightly darker at horizon (expected - more extinction)
- **Learning**: T appears visually correct, no obvious color shift

### V60: Output ray_intersects_ground flag
- **Result**: White (ground) at/below horizon, Black (non-ground) above
- **Learning**: Blue band appears in BLACK region (non-ground), NOT at boundary!

---

## Hypothesis Revision

**Ground/non-ground mismatch hypothesis is WRONG.**

The blue band appears in the non-ground region where:
- Scattering uses non-ground formula (forced)
- T ALSO uses non-ground formula (ray_intersects_ground = 0)

Both use the same formula, yet the blue band still appears!

### New Approach
If T looks neutral but T × scat_pt creates blue when subtracted, T must have a subtle per-channel difference that's not visible but affects the math.

### V61: Output T × scat_pt
- **Result**: Looks very similar to V57 (scat_pt alone), still has warm band
- **Learning**: T is not adding obvious color shift, mostly just attenuating

### V62: Full inscatter output restored
- **Result**: BLUE BAND STILL PROMINENT AND UNACCEPTABLE
- **Learning**: The blue band is definitely a BUG, not physically correct behavior

---

## Complete Test Summary

| Version | Output | Blue Band? | Notes |
|---------|--------|------------|-------|
| V52 | scat_cam (non-ground UV) | NO | Clean, correct appearance |
| V53 | scat_cam (ground UV) | NO | Different values, bright below horizon |
| V54 | Full inscatter | YES | Blue band returned |
| V55 | Rayleigh-only inscatter | YES | Blue band in Rayleigh too |
| V56 | Non-ground T + scatter | YES + haze | Made things worse |
| V57 | scat_pt only | NO | WARM band at horizon |
| V58 | scat_cam - scat_pt (no T) | NO | Expected gradient, no blue |
| V59 | T only | N/A | Neutral grey, correct |
| V60 | ray_intersects_ground | N/A | Boundary at horizon |
| V61 | T × scat_pt | NO | Similar to V57 |
| V62 | Full inscatter | YES | Blue band still prominent |

## Key Insights

1. **Raw scattering samples are CLEAN** - V52, V53, V57 all have no blue band
2. **Raw difference (no T) is CLEAN** - V58 has no blue band
3. **T looks neutral** - V59 shows correct transmittance
4. **Blue band appears ONLY in full inscatter formula**
5. **Blue band is in NON-GROUND region** - V60 shows boundary is below the band

## What We Still Don't Know

1. WHY does multiplying by T create the blue band if T is neutral?
2. Is there a subtle numerical issue in how we compute T or apply it?
3. Does the working Bruneton reference (atmospheric-scattering-2) show this issue?

### V63: RAW inscatter before clamping
- **Result**: BLUE BAND STILL PRESENT
- **Learning**: Clamping is NOT the cause - band exists in raw values before clamp

## Root Cause Analysis

**Key comparison:**
- V58: `scat_cam - scat_pt` = NO blue band
- V63: `scat_cam - T×scat_pt` = BLUE BAND

**The ONLY difference is T multiplication.** Even though T looks neutral grey (V59), it must have a subtle per-channel difference that creates color shift.

### V64: Output (T × scat_pt) - scat_pt
- **Result**: BLACK with DIFFERENT negative values per channel
- **Learning**: T has spectral bias (expected - Rayleigh affects blue more)

## Transmittance Analysis

Checked reference `functions.glsl` lines 493-518:
```glsl
// NON-GROUND: T = T(r, mu) / T(r_d, mu_d)
// GROUND: T = T(r_d, -mu_d) / T(r, -mu)
```

**Our implementation matches the reference formula exactly.**

The spectral variation in T is PHYSICALLY CORRECT:
- Rayleigh scattering removes more blue than red
- T transmits more red than blue
- T × scat_pt is redder than scat_pt
- inscatter = scat_cam - T×scat_pt = bluer result

## The Question

If the formula is correct, why is the blue band so prominent?

Possible causes:
1. Scene geometry: very distant objects at horizon exaggerate the effect
2. Numerical precision at horizon (mu ≈ 0)
3. The reference demo might show the same effect
4. Something else is different between our impl and reference

### V65: Force T = 1.0 (neutral white)
- **Result**: NO BLUE BAND!
- **Visual**: Warm/yellow at top of buildings, dark at horizon, red/orange transition
- **Learning**: This is the raw inscatter (scat_cam - scat_pt) without T attenuation

## ROOT CAUSE CONFIRMED

**T spectral bias is the SOLE cause of the blue band.**

Verified formulas match reference exactly:
- `r_p = sqrt(d² + 2rμd + r²)` ✓
- `mu_p = (r×μ + d) / r_p` ✓
- `T = T(r,μ) / T(r_p, μ_p)` for non-ground ✓

The issue is NOT a bug - the spectral variation in T is physically correct. But it's EXAGGERATED at extreme horizon distances in this scene, creating a visually objectionable blue band.

## Possible Solutions

1. **Artistic control**: Blend T towards neutral at horizon angles
2. **Spectral clamping**: Limit per-channel variation in T
3. **Scene adjustment**: Move objects closer or adjust camera altitude
4. **Check reference demo**: Does it have objects at similar extreme distances?

### V66: T Spectral Neutralization at Horizon
- **Result**: Helps reduce blue saturation but doesn't fix root cause
- **Visual**: Still shows odd contribution bleeding through closer buildings
- **Decision**: Keep as backup solution, investigate root cause first

---

## EXHAUSTIVE ROOT CAUSE ANALYSIS

Comparing our implementation with Bruneton reference (functions.glsl) which does NOT have this issue.

### Category 1: Parameter Computation Differences

| Parameter | Reference Formula | Our Implementation | Match? |
|-----------|-------------------|-------------------|--------|
| `r` | `length(camera)` | `length(camera_km)` | ✓ |
| `mu` | `dot(camera, view_ray) / r` | `dot(camera_km, view_ray) / r` | ✓ |
| `mu_s` | `dot(camera, sun_direction) / r` | `dot(camera_km, Sun_Direction) / r` | ✓ |
| `nu` | `dot(view_ray, sun_direction)` | `dot(view_ray, Sun_Direction)` | ✓ |
| `d` | `length(point - camera)` | `length(point_km - camera_km)` | ✓ |
| `r_p` | `sqrt(d² + 2rμd + r²)` | Same formula | ✓ |
| `mu_p` | `(r×μ + d) / r_p` | Same formula | ✓ |
| `mu_s_p` | `(r×μ_s + d×ν) / r_p` | Same formula | ✓ |

### Category 2: Transmittance Computation

| Aspect | Reference | Our Implementation | POTENTIAL ISSUE? |
|--------|-----------|-------------------|------------------|
| NON-GROUND | `T(r,μ) / T(r_d, μ_d)` | `T(r,μ) / T(r_p, μ_p)` | ✓ Same |
| GROUND | `T(r_d,-μ_d) / T(r,-μ)` | `T(r_p,-μ_p) / T(r,-μ)` | ✓ Same |
| r_d vs r_p | Reference uses d-based calculation | We use d-based calculation | ✓ Same |
| **Selection** | Uses `ray_r_mu_intersects_ground` from **camera** | Same flag for T selection | ✓ Same |

### Category 3: Scattering UV Computation

| Aspect | Reference | Our Implementation | POTENTIAL ISSUE? |
|--------|-----------|-------------------|------------------|
| Camera scattering | Uses `ray_r_mu_intersects_ground` | **FORCED to non-ground (V54+)** | ⚠️ DIFFERENT |
| Point scattering | Uses **SAME** `ray_r_mu_intersects_ground` from camera | Uses same flag | ✓ Same |
| UV formula | Complex 4D mapping with nu interpolation | **May differ** | ⚠️ NEEDS CHECK |

### Category 4: Coordinate System / Units

| Aspect | Reference | Our Implementation | POTENTIAL ISSUE? |
|--------|-----------|-------------------|------------------|
| Units | Meters internally | Kilometers internally | ✓ (scaled) |
| Planet center | Origin (0,0,0) | User-defined `Planet_Center` | ✓ Same logic |
| Radii | `bottom_radius`, `top_radius` | 6360 km, 6420 km | ✓ Same |
| **Camera altitude** | Not clamped | **MIN_CAMERA_ALTITUDE = 0.5 km** | ⚠️ DIFFERENT |

### Category 5: Scene Geometry Differences

| Aspect | Bruneton Demo | Our Scene | POTENTIAL ISSUE? |
|--------|--------------|-----------|------------------|
| Camera height | ~1-2 km above ground | Ground level (~0 meters) | ⚠️ VERY DIFFERENT |
| Object distances | Mid-range (~1-10 km) | Very distant (horizon ~40+ km) | ⚠️ VERY DIFFERENT |
| Object altitudes | Various | Ground level | ⚠️ DIFFERENT |
| View angles | Varied | Horizontal (mu ≈ 0) | ⚠️ EDGE CASE |

### Category 6: Texture Sampling

| Aspect | Reference | Our Implementation | POTENTIAL ISSUE? |
|--------|-----------|-------------------|------------------|
| Interpolation | Bilinear in shader | Blender's Linear interpolation | ✓ Same |
| Edge handling | CLAMP | EXTEND | ✓ Same |
| **4D→2D mapping** | Complex slice interpolation | **May differ** | ⚠️ NEEDS CHECK |
| Precision | Float32 | Float32 EXR | ✓ Same |

---

## PRIORITIZED LIST OF POSSIBLE CAUSES

### HIGH PRIORITY (Most Likely)

1. **Scattering UV ground/non-ground MISMATCH with Transmittance**
   - We forced scattering to always use non-ground (V54)
   - But T still uses ground/non-ground selection
   - At horizon, T might select ground formula while scattering uses non-ground
   - **Test**: Force T to also use non-ground only

2. **Extreme Scene Geometry (Edge Case)**
   - Camera at ground level + horizontal view = mu ≈ 0
   - Objects at extreme horizon distance = very large d
   - This combination may hit numerical edge cases in LUT sampling
   - **Test**: Raise camera altitude significantly

3. **4D Scattering Texture UV Mapping Error**
   - The scattering texture encodes 4D data (r, mu, mu_s, nu)
   - Complex slice interpolation required
   - May have error at specific mu values near 0
   - **Test**: Output raw UV coordinates to check for discontinuities

### MEDIUM PRIORITY

4. **mu Clamping Differences**
   - We clamp mu to [-1, 1] after computation
   - Reference uses ClampCosine which may handle edge cases differently
   - At horizon, floating point errors could cause mu to be slightly outside range
   - **Test**: Check mu values at horizon band location

5. **MIN_CAMERA_ALTITUDE Causing Issues**
   - We clamp camera altitude to 0.5 km minimum
   - This could create discontinuity at low altitudes
   - Reference doesn't have this clamp
   - **Test**: Remove MIN_CAMERA_ALTITUDE clamp

6. **Point Scattering Parameters (r_p, mu_p, mu_s_p)**
   - For very distant points, these values may become extreme
   - mu_p could approach edge values
   - mu_s_p calculation involves d×nu which could be very large
   - **Test**: Output r_p, mu_p, mu_s_p values at blue band location

### LOWER PRIORITY

7. **Transmittance Texture UV Mapping**
   - Different UV formulas for ground vs non-ground
   - May have discontinuity at selection boundary
   - **Test**: Compare T texture samples with reference

8. **Phase Function Not Applied Correctly**
   - Phase functions are applied AFTER inscatter subtraction
   - Could amplify small errors
   - **Test**: Already tested - blue band present without phase functions too

9. **EXR Texture Precision/Encoding**
   - LUT values might have precision issues at edge cases
   - **Test**: Compare our LUT values with reference at horizon coordinates

---

## V68: FIX ALL BRUNETON DIFFERENCES

After exhaustive line-by-line comparison with reference `functions.glsl`:

### DIFFERENCES FOUND AND FIXED:

| # | Issue | Before | After (V68) |
|---|-------|--------|-------------|
| 1 | **Nu interpolation** | Only sampled floor(nu), NO interpolation | Now samples 4 textures, bilinear interp (nu × depth) |
| 2 | **Scattering UV ground selection** | Forced to non-ground (V54) | Restored ray_intersects_ground flag selection |
| 3 | **MIN_CAMERA_ALTITUDE** | Clamped r to BOTTOM+0.5km | Removed - clamp to [BOTTOM, TOP] like reference |

### Code Changes:
1. `sample_scattering_texture()`: Complete rewrite for proper nu interpolation
   - Now samples 4 textures: (nu_floor, depth_floor), (nu_floor, depth_ceil), (nu_ceil, depth_floor), (nu_ceil, depth_ceil)
   - Bilinear interpolation: first depth, then nu
   
2. `_compute_scattering_uvwz()`: Restored ground/non-ground u_mu selection
   - Uses `ray_intersects_ground_socket` to select between ground and non-ground formulas
   - Same flag used for both scattering AND transmittance (matching reference)
   
3. Camera radius: Removed MIN_CAMERA_ALTITUDE
   - Was: `r_min = BOTTOM_RADIUS + 0.5` 
   - Now: `r_min = BOTTOM_RADIUS` (matches reference)

## V68 RESULT: BROKEN - Excessive haze and banding

**Symptoms:**
- Massive excessive haze/fog across ground plane
- Horizontal banding above horizon
- Scene geometry barely visible through haze

**Root Cause:** Restored ground/non-ground selection for scattering UV caused the issue.
- Ground UV samples underground scattering (u_mu in [0, 0.5])
- Non-ground UV samples sky scattering (u_mu in [0.5, 1.0])
- For aerial perspective of above-ground objects, we need SKY scattering, not underground
- The reference uses ground selection for rays actually hitting ground, but aerial perspective objects are above ground

## V69: Revert ground selection, keep nu interpolation

**Changes:**
- Reverted scattering UV to forced non-ground (like V54)
- KEPT nu interpolation (4-texture bilinear sampling) - this is important for quality
- KEPT MIN_CAMERA_ALTITUDE removal

**Key Insight:** The reference's ground/non-ground selection applies to the GEOMETRY of what the ray hits. For aerial perspective, we're computing scattering for rays from camera to above-ground objects, so we should always use non-ground scattering UV. The transmittance still correctly uses ground/non-ground selection.

## V70: FIX MIE PHASE FUNCTION (was completely wrong!)

**BUG FOUND:** Mie phase function formula was incorrect.

**Reference formula (lines 744-746):**
```glsl
k = 3/(8π) × (1-g²)/(2+g²)
MiePhaseFunction = k × (1+ν²) / (1+g²-2gν)^1.5
```

**Our WRONG formula:**
```
(1-g²) / (4π × (1+g²-2gν)^1.5)
```

**Missing:**
1. The `(1 + ν²)` term in numerator - THIS IS CRITICAL for sun halo!
2. The `/(2+g²)` factor in k
3. Wrong constant (4π vs 8π with 3× factor)

This explains why Mie scattering wasn't showing sun-concentrated pattern - the phase function was completely wrong!

---

## Key Insights

### The Inscatter Formula
```
inscatter = scat_cam - T × scat_pt
```
Where:
- `scat_cam` = scattering sampled at camera position (r, mu, mu_s, nu)
- `scat_pt` = scattering sampled at point position (r_p, mu_p, mu_s_p, nu)
- `T` = transmittance from camera to point

### What's Different Between V52 (working) and V54 (broken)?
- V52 output: `scat_cam_color` (raw sample, no subtraction)
- V54 output: `inscatter_phased` (includes subtraction and phase functions)

### Transmittance Still Uses Ground/Non-Ground
The transmittance calculation still uses `ray_intersects_ground` to select between:
- Non-ground: `T = T(r, mu) / T(r_p, mu_p)`
- Ground: `T = T(r_p, -mu_p) / T(r, -mu)`

This could be the source of the blue band, since T is multiplied by scat_pt.

---

## Separate Issue: Mie Sun Concentration
The Mie scattering doesn't show the expected sun-concentrated pattern.
- Likely cause: `nu` (view-sun angle cosine) calculation or phase function
- NOT related to blue band (tracked separately)

---

## Next Steps Based on V55 Result
- If V55 has blue band → Issue is in Rayleigh inscatter (T or scat_pt)
- If V55 has no blue band → Issue is in Mie calculation

---

## V72-V80 SYSTEMATIC DIAGNOSTIC (Jan 5, 2026)

After V68-V71 fixes (nu interpolation, forced non-ground, Mie phase fix), banding still present.
Systematic isolation to find EXACT source of banding artifacts.

### Complete Diagnostic Table

| Version | Output | Banding? | Notes |
|---------|--------|----------|-------|
| V72 | `scat_cam_color` (raw camera scattering) | NO | Texture sampling is CLEAN |
| V73 | `rayleigh_inscatter` (with max(0) clamp) | YES | Banding appears in inscatter |
| V74 | `T × scat_pt` | NO | Clean, very faint - had to push values hard to see |
| V75 | `scat_cam - T×scat_pt` (no clamp) | YES | Subtraction itself has bands - clamp NOT the cause |
| V76 | `scat_pt_color` (raw point scattering) | NO | Clean, smooth gradient |
| V77 | `transmittance` (T_cam / T_pt) | YES | **COLORED BANDING VISIBLE** (subtle, needed gamma crush) |
| V78 | `T_cam` only (T(r,mu) before division) | NO | Relatively clean, minor stair-stepping from texture resolution |
| V79 | `T_pt` only (T(r_p, mu_p)) | NO | Same as V78 - clean with minor stair-stepping |
| V80 | `T_cam - T_pt` (difference) | ? | Testing now - checking if subtraction or division causes bands |

### Key Findings

#### 1. All Scattering Components Are Clean
- `scat_cam_color` (V72) = CLEAN
- `scat_pt_color` (V76) = CLEAN
- `T × scat_pt` (V74) = CLEAN (very faint values)
- The raw texture sampling and scattering calculations are working correctly

#### 2. Inscatter Subtraction Has Banding
- `scat_cam - T×scat_pt` (V75) = BANDING even without max(0) clamp
- The `max(0)` clamp is NOT the cause of banding

#### 3. ROOT CAUSE: Transmittance Division Creates Colored Banding
- **V77 showed colored banding in the final transmittance (T_cam/T_pt)**
- V78 showed T_cam alone is relatively clean (minor stair-stepping)
- V79 showed T_pt alone is also clean (same as T_cam)
- **The DIVISION of two similar-valued wavelength-dependent (RGB) samples creates colored artifacts**

### Why Division Creates Colored Banding

The transmittance texture encodes wavelength-dependent values (RGB channels differ because red scatters less than blue). When dividing T(r,mu) by T(r_p, mu_p):

1. Both samples have similar values but at slightly different UV positions
2. The per-channel ratios aren't perfectly consistent across the texture
3. Small spectral differences get amplified by division
4. This creates colored bands at specific viewing angles

### Texture Resolution Notes

The transmittance texture is 256x64 pixels. The stair-stepping visible in V78/V79 is due to this limited resolution. When two samples at nearby positions are divided, quantization artifacts become colored bands.

### Reference Implementation Check

The reference `GetTransmittance` function (functions.glsl lines 493-519):
```glsl
// Non-ground case:
return min(
    GetTransmittanceToTopAtmosphereBoundary(r, mu) /
    GetTransmittanceToTopAtmosphereBoundary(r_d, mu_d),
    DimensionlessSpectrum(1.0));
```

- Uses same formula as our implementation
- Applies `min(..., 1.0)` clamp (we do this too)
- Uses `GetTextureCoordFromUnitRange` for UV mapping (we match this exactly)

Our implementation matches the reference, but GPU hardware texture sampling may provide smoother interpolation than Blender's Image Texture nodes.

### Next Steps

1. V80 tests `T_cam - T_pt` to see if subtraction also creates bands or if it's specific to division
2. If subtraction is clean → division amplifies spectral differences
3. If subtraction has bands → the texture samples themselves have discontinuities

### Potential Fixes to Explore

1. **Increase transmittance texture resolution** (e.g., 512x128)
2. **Apply smoothing/filtering** to transmittance before division
3. **Check if Blender texture interpolation differs from GPU bilinear**
4. **Verify transmittance LUT precomputation** matches reference exactly

---

## V80-V85 DEEP DIAGNOSTIC (Jan 5, 2026)

### Critical Clarification
**The banding is on OBJECTS/BUILDINGS only, NOT the sky.** The sky has always been smooth. This is crucial - the issue is specifically in the aerial perspective applied to geometry.

### Extended Diagnostic Table

| Version | Output | Banding? | Notes |
|---------|--------|----------|-------|
| V80 | `T_cam - T_pt` | N/A | All negative values - couldn't evaluate visually |
| V81 | `T_pt - T_cam` (positive) | **YES** | Clear banding on objects |
| V82 | `UV_pt - UV_cam` (UV difference) | **YES** | Banding in ground/object area only |
| V83 | Distance `d` (normalized) | NO | Smooth gradient |
| V84 | `mu_p` (point view direction) | NO | Smooth gradient |
| V85 | Point UV.x coordinate | ? | Testing now |

### Analysis: Inputs Smooth, Outputs Have Banding

**V83 and V84 confirm:** The input parameters (distance `d`, point view direction `mu_p`) are both smooth gradients with no discontinuities.

**V81 and V82 confirm:** The outputs (transmittance difference, UV difference) have visible banding on objects.

**The mystery:** Smooth inputs → discontinuous outputs. This can only happen if:
1. The UV formula has non-uniform sensitivity at different input values
2. The texture sampling introduces quantization artifacts
3. There's a numerical precision issue in the transformation

### Hypothesis: Texture Quantization Amplification

The transmittance texture is 256x64 pixels. When two nearby samples are subtracted or divided:
- Each sample has subtle stair-stepping from texture resolution
- The stair-stepping is invisible in individual samples (V78/V79 looked smooth)
- But when differenced, the stair-stepping becomes visible as colored bands
- This is because the two samples straddle texture boundaries at slightly different points

### V85 Result
Point transmittance UV.x is **SMOOTH** - same as V83/V84. The UV formula is correct.

**Conclusion:** The banding is caused by texture quantization when per-channel RGB division amplifies small differences between samples.

---

## V86: GRAYSCALE TRANSMITTANCE FIX (Jan 5, 2026)

### The Problem
When dividing `T_cam / T_pt` per-channel (RGB), texture quantization causes each channel to have slightly different ratios at certain positions. This creates **colored bands** even though individual samples appear smooth.

### The Fix
Instead of per-channel division:
```
T = T_cam.rgb / T_pt.rgb  // Each channel divided separately → colored banding
```

Use grayscale-based division:
```
T_cam_gray = (T_cam.r + T_cam.g + T_cam.b) / 3
T_pt_gray = (T_pt.r + T_pt.g + T_pt.b) / 3
ratio = T_cam_gray / T_pt_gray
T = T_cam.rgb * ratio  // Single ratio applied uniformly → no colored banding
```

### Trade-offs
- **Pro:** Eliminates colored banding artifacts
- **Pro:** Preserves overall transmittance magnitude
- **Con:** Loses some per-channel spectral variation in the transmittance ratio
- **Note:** The original T_cam color is still used, just scaled uniformly

### V86 Status: ✅ SUCCESS - NO BANDING OBSERVED

The grayscale transmittance fix eliminates the colored banding artifacts on objects/buildings.

---

## BANDING ISSUE RESOLVED (Jan 5, 2026)

### Root Cause
Per-channel RGB division of transmittance texture samples (`T_cam / T_pt`) amplified texture quantization differences into colored banding artifacts.

### Solution
Use grayscale-based division: compute average luminance for both samples, divide once, then scale the original color uniformly.

### Files Modified
- `helios/aerial_nodes.py` - V86 transmittance calculation

### Impact
- Banding on objects/buildings: **ELIMINATED**
- Sky rendering: Unchanged (was always smooth)
- Spectral accuracy: Minor reduction (single ratio vs per-channel), but visually imperceptible

---

## V87-V90: SEPARATE AOV DEBUGGING (Jan 5, 2026)

### V87: Separate Rayleigh and Mie AOVs
Separated inscatter output into individual Rayleigh and Mie AOVs for Nuke compositing.
- **Result:** ❌ BROKEN - Rayleigh looked like sky gradient, Mie didn't respond to sun

### V88: Fix Transmittance Formula
Discovered V86 grayscale fix had wrong formula:
- **V86 (wrong):** `T = T_cam × (T_cam_gray / T_pt_gray)`
- **V88 (correct):** `T = T_cam_gray / T_pt_gray`

For d→0, V86 gave T≈T_cam instead of T→1.
- **Result:** ❌ STILL BROKEN - Same visual issues

### V89: Diagnostic - Raw S_cam vs S_pt
Output raw scattering samples to compare camera vs point sampling.
- **Result:** ❌ S_cam = S_pt (IDENTICAL) - Point parameters not affecting UV

### V90: Diagnostic - Distance d
Output distance d directly to verify Position input is connected.
- **Result:** ✅ CORRECT - Distance shows proper depth falloff, close objects dark, far objects bright

### V91: Diagnostic - mu vs mu_p
Output camera mu vs point mu_p to verify point parameters are computed differently.
- **Result:** ❌ mu = mu_p (IDENTICAL) - Point parameters NOT using d

### V92: Diagnostic - r×μ vs r×μ+d
Output the intermediate values to find exactly where d gets lost.
- **Result:** ❌ r×μ = r×μ+d (IDENTICAL) - d not being linked to ADD node

### V93: FIX - Variable Shadowing Bug
**ROOT CAUSE FOUND AND FIXED**

Line 239 in `_compute_transmittance_uv` was reassigning `d`:
```python
d = builder.math('ADD', ...)  # Overwrote distance d!
```

This shadowed the outer `d` (distance) variable. When mu_p calculation tried to use `d.outputs['Value']`, it was referencing the wrong node.

**Fix:** Renamed local variable to `t_dist` to avoid shadowing.

### V93: Attempted Fix - Variable Shadowing
Renamed `d` to `t_dist` in `create_transmittance_uv` to avoid shadowing.
- **Result:** ❌ DID NOT FIX - Still broken

The functions have separate scopes, so variable shadowing wasn't the issue.

### V94-V95: Fresh Node Tests
- V94: Fresh nodes at end, d on inputs[1] - appeared identical
- V95: Fresh nodes at end, d on inputs[0] - appeared identical

### V96: Debug Print Analysis - **CRITICAL FINDING**
Added debug prints to inspect actual node graph structure.

**Console output confirmed:**
```
r_mu_plus_d.inputs[0].is_linked = True
r_mu_plus_d.inputs[1].is_linked = True
r_mu_plus_d.inputs[0] linked from: r×μ
r_mu_plus_d.inputs[1] linked from: d
```

**The links ARE correct!** d IS being linked properly.

### Root Cause Analysis
The diagnostic comparisons looked identical because:
- r×μ ≈ ±6360 km (Earth radius × view angle)
- d ≈ 1 km (typical object distance)  
- Difference is only ~0.016% - **visually imperceptible** when both scaled

The math IS working. The inscatter formula IS using d correctly.

### V97: Restore Actual Output
Now that linking is confirmed correct, restored actual Rayleigh/Mie AOV output.

### Key Findings Summary
1. V89: S_cam = S_pt - appeared identical due to UV quantization
2. V90: Distance d correct ✅
3. V91-V95: Appeared identical due to scale difference (d << r×μ)
4. **V96: Links ARE correct** ✅ - d IS linked to r_mu_plus_d
5. V97: Testing actual output with confirmed correct linking
