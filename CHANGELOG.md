# Helios Aerial Perspective Changelog

## V126 - January 8, 2026
- **Step 9 created**: Step 2.4 + wavelength-dependent transmittance
  - Calls Step 2.4 then modifies transmittance nodes in-place
  - Grayscale T (k=-0.1) -> RGB T (k_r=0.02, k_g=0.03, k_b=0.05)
  - Blue attenuates faster than red (physically correct)
  - Safe approach: Step 2.4 preserved, Step 9 modifies copy

## V125 - January 8, 2026
- **Step 8 replaced with Step 9**: Proper implementation instead of wrapper

## V124 - January 8, 2026
- **Step 7 scattering UV fix FAILED**: Made output completely broken
  - Attempted to simplify Bruneton UV but formula was wrong
  - Step 2.4's complex create_scatter_uv helper is required

## V123 - January 8, 2026
- **Step 7: Full LUT Inscatter** implemented (HAD BUG - scattering UV was wrong)
  - Combines LUT scattering with wavelength-dependent exponential T
  - LUT-based Rayleigh (RGB) and Mie (Alpha) with phase functions
  - Inscatter formula: S_cam - T × S_pt with positive clamping
  - Debug modes: 0=inscatter, 1=T, 2=S_cam, 3=S_pt, 4=phase

## V122 - January 8, 2026
- **Horizon fallback**: Reverted to hard mu<0 cutoff, added exponential fallback
  - When |mu| < 0.1: blend to wavelength-dependent exponential T
  - Exponential uses k_r=0.02, k_g=0.03, k_b=0.05
  - This avoids LUT instability near horizon while giving plausible values

## V121 - January 8, 2026
- **Horizon seam fix FAILED**: Smoothstep blend made horizon worse (0.98→0.63)
  - Both sky and ground formulas give bad values at mu≈0
  - Blending two bad values doesn't help

## V120 - January 8, 2026
- **Step 6: Bruneton Transmittance** implemented with ground intersection handling
  - `apply_step_6_bruneton_transmittance(debug_mode)` function added
  - Computes both sky and ground formulas, selects based on mu < 0
  - Sky: `T = T(r, mu) / T(r_d, mu_d)`
  - Ground: `T = T(r_d, -mu_d) / T(r, -mu)` (negated mu!)
  - Debug modes: 0=full, 1=T_sky, 2=T_gnd, 3=ground_flag
- Created `TRANSMITTANCE_DEBUG_LOG.md` for tracking attempts

## V119 - January 8, 2026
- **ROOT CAUSE FOUND** for LUT transmittance failure
  - Failed implementation used sky formula for all rays
  - Ground-hitting rays need: `T = T(r_d, -mu_d) / T(r, -mu)` (negated mu!)
  - This flips lookup to "looking up" side of LUT, avoiding near-zero values
  - Reference: Bruneton `functions.glsl` lines 493-518
- Created `debug_transmittance.py` for detailed LUT analysis
- Copied `functions.glsl` to reference/ for comparison

## V118 - January 8, 2026
- **REVERTED** LUT-based transmittance - broke render (nearly black output)
- Back to simplified exponential: `T = exp(-d × 0.1)`
- LUT-based transmittance needs debugging before re-implementation

## V117 - January 8, 2026
- **LUT-based Transmittance ATTEMPTED** in Step 2.4 - FAILED
  - Implementation broke render (values ~0.01 instead of ~1.0)
  - Issue: Used wrong formula for ground-intersecting rays
  - Reverted in V118

## V116 - January 8, 2026
- **Step 4 VALIDATED**: Full aerial perspective working correctly
  - Near objects: (0.997, 0.500, 0.300) ≈ input (1.0, 0.5, 0.3)
  - Further objects: (0.932, 0.470, 0.288) - attenuated by transmittance
  - Formula `L_final = L_surface × T + inscatter` confirmed working

## V115 - January 8, 2026
- Step 4: Full aerial perspective formula implemented
  - `apply_step_4_full_aerial(surface_color, debug_mode)` function added
  - Formula: `L_final = L_surface × T + inscatter`
  - Surface color configurable via parameter (default gray 0.8)
  - Simplified inscatter using (1-T) approximation with phase functions
  - Debug modes: 0=full, 1=surface×T, 2=T, 3=inscatter, 4=surface, 5=distance

## V114 - January 8, 2026
- Updated AERIAL_IMPLEMENTATION_STRATEGY.md and DEVELOPMENT_PLAN.md with new requirements:
  - **AOV Requirements**: Sky (no sun disk), Transmittance, Rayleigh, Mie, Sun Disk
  - **Sun Light Integration**: Addon reads sun position from scene Sun light object
  - **Creative Controls Sharing**: Sky and aerial perspective share same parameters
  - **Sky Material Validation**: Note to verify main branch sky material uses same math
- Updated status tables to reflect completed Steps 1-3

## V113 - January 8, 2026
- **Step 3.2 WORKING**: Mie phase function integrated
  - Sample alpha channel from scattering texture (stores Mie.r)
  - Compute Mie inscatter: `alpha_cam - T × alpha_pt`
  - Apply Henyey-Greenstein phase (g=0.8) to Mie
  - Final output = Rayleigh×RayleighPhase + Mie×MiePhase
  - Debug modes: 4=Rayleigh only, 5=Mie only (10x scaled)
  - User confirmed output responds to sun position changes
  - Next: validate against sky/transmittance for consistency

## V112 - January 8, 2026
- **Step 3.1 VALIDATED**: Rayleigh phase function responds to sun position
  - Sun rotation X=120° gives 30° elevation (Z=0.5 positive)
  - Confirmed render varies with sun direction changes
  - Note: Blender Sun at X=0 points DOWN; need X≈90-150 for daytime

## V111 - January 8, 2026
- Integrated phase functions into working Step 2.4 inscatter
  - Added Rayleigh phase: `(3/16π)(1 + nu²)` directly to `apply_step_2_4_inscatter()`
  - Added Mie phase nodes (computed but not yet applied - needs alpha channel work)
  - Reuses working UV calculation instead of simplified version
  - debug_mode=4 added to see inscatter without phase for comparison
  - Step 3 Full had black output due to simplified UV not handling sun below horizon

## V110 - January 7, 2026
- Step 3 Full: Integrated phase functions with inscatter
  - `apply_step_3_full()` combines phase functions with dynamic scattering UV
  - Rayleigh phase applied to RGB channels
  - Mie phase applied to alpha channel (single_mie_scattering)
  - Distance modulation: `1 - exp(-d * 0.5)` for aerial perspective falloff
  - Sun direction read from scene Sun light (validated: nu changes with sun rotation)

## V109 - January 7, 2026
- Step 3: Phase functions implementation
  - Rayleigh phase: `(3/16π)(1 + nu²)` where nu = dot(view, sun)
  - Mie phase: Henyey-Greenstein with g=0.8
  - Combined scattering: RGB = Rayleigh, Alpha = Mie.r
  - Distance-based falloff for aerial perspective effect
  - NOTE: This is a simplified test - uses fixed UV, not full inscatter pipeline

## V108 - January 7, 2026
- Sun direction now reads from scene Sun light
  - Finds first LIGHT object with type='SUN' in scene
  - Extracts world-space direction from matrix_world
  - Falls back to 15° elevation if no sun found
  - Prints sun direction to console for verification

## V107 - January 7, 2026
- **Step 2.4 VALIDATED**: Full inscatter computation working
  - B/R ratio: 1.59 (expected ~1.69 for Rayleigh) ✓
  - G/R ratio: 1.38 ✓
  - Proper blue-shifted inscatter with distance variation
  - Fixed RGB clamp using SeparateColor + MAXIMUM + CombineColor

## V106 - January 7, 2026
- Step 2.4: Fixed Mix node input/output indices for RGBA color data
  - **Bug found**: Using `inputs['A']` and `outputs['Result']` for Mix RGBA nodes
  - **Fix**: RGBA Mix nodes use indices 6,7 for A,B color inputs, index 2 for color output
  - Fixed t_times_spt multiply: `inputs[6]` = T_rgb, `inputs[7]` = S_pt
  - Fixed inscatter subtract: `inputs[6]` = S_cam, `inputs[7]` = T×S_pt, `outputs[2]` for result
  - Initial clamp attempt with Mix clamp_result failed (output black)

## V105 - January 7, 2026
- Step 2.4: Debug validation of S_cam and S_pt sampling
  - **S_cam**: B/R=1.69 (correct Rayleigh), range 0-0.68
  - **S_pt**: B/R=1.69 (correct Rayleigh), range 0-0.70
  - Both samples show correct color ratios - scattering LUT sampling is working
  - **Issue**: S_cam ≈ S_pt at ground level, so inscatter subtraction cancels color
  - **Analysis needed**: The inscatter formula S_cam - T×S_pt should produce blue-shifted result when T < 1

## V104 - January 7, 2026
- Step 2.4: Added depth slice interpolation to fix stair-stepping
  - **Issue identified**: Multiple horizontal discontinuities at different altitudes
  - **Root cause**: Sampling only at depth_floor without interpolation between depth slices
  - **Fix**: Sample at both depth_floor and depth_ceil, interpolate using depth_frac
  - Returns (uv_floor, uv_ceil, depth_frac) for proper bilinear interpolation
  - **Result**: Major stair-stepping resolved
  - **Note for later**: Minor subtle stepping visible in very close foreground objects - acceptable since aerial perspective is negligible at close range

## V103 - January 7, 2026
- Step 2.4: Fixed u_mu calculation with proper Bruneton formula
  - **Issue identified**: Grayscale output (B/R=1.0) and visible discontinuity at horizon
  - **Root cause**: Simplified u_mu formula `(1+mu)/2` was completely wrong
  - **Fix**: Implemented proper non-ground u_mu from Bruneton:
    - d = -r*mu + sqrt(r²(μ²-1) + top²)
    - d_min = top - r, d_max = rho + H
    - x_mu = (d - d_min) / (d_max - d_min)
    - u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(x_mu, MU_SIZE/2)

## V102 - January 7, 2026
- Step 2.3: Fixed scattering LUT sampling with proper 3D→2D UV mapping
  - Added Y-flip (V = 1.0 - u_mu) for Blender texture convention
  - Added proper depth indexing: U = (depth_floor + uvw_x) / DEPTH
  - Validated correct Rayleigh color ratios: B/R=1.66, G/R=1.41 (B>G>R)

## V101 - January 7, 2026
- Step 1.2: mu validated (0.1% median error), r has Blender render caching issue (noted for later)
- Implemented Step 1.3: Transmittance UV calculation
- Fixed __main__ to not auto-run (was causing confusion with exec())

---

## V100 - January 7, 2026
- Step 1.1 VALIDATED: Distance calculation correct (mean error 4.6m, median 0.25%)
- Implemented Step 1.2: r, mu calculation test node group
- Updated TEST_SCENE_REFERENCE.md with actual camera position (37.069, -44.786, 6.0)

---

## V99 - January 7, 2026
- Implemented Step 1.1: Distance calculation test node group
- Created helios/aerial_test_steps.py for step-by-step validation
- Test outputs distance/10.6km as grayscale emission for visual validation

---

## V98 - January 7, 2026
- Created AERIAL_IMPLEMENTATION_STRATEGY.md - methodical step-by-step rebuild of aerial perspective
- Document includes exact Bruneton code references from functions.glsl with line numbers
- 14 discrete steps with validation criteria for each
- New approach: test via Emission shader output (no AOV infrastructure needed)
- Added standardized test scene specification with documented geometry for validation context
- Created tests/create_test_scene.py - Blender script to generate consistent test scene
- Created tests/analyze_scene_exr.py - Analyzes Position AOV from EXR to extract scene geometry
- Analyzed user's test scene (base_render_reference_noAtmos_v001_01.exr) and generated TEST_SCENE_REFERENCE.md
- Scene: camera at 218m altitude, geometry from 161m to 10.6km distance, 73.6% geometry / 26.4% sky

---

## V97 - January 5, 2026
- Restored actual Rayleigh/Mie AOV output after V96 confirmed node links are correct
- **Status:** Still broken - horizontal gradient, no depth-dependent scattering

## V96 - January 5, 2026
- Added debug prints to trace node graph structure
- **CONFIRMED:** d IS correctly linked to r_mu_plus_d.inputs[1]
- All 5 links from d.outputs['Value'] verified correct
- Issue is NOT with node linking

## V94-V95 - January 5, 2026
- Tested fresh r×μ and r×μ+d nodes at end of function
- V94: d on inputs[1], V95: d on inputs[0]
- Both appeared identical (scale difference: d ~1km vs r×μ ~6360km = 0.016%)

## V93 - January 5, 2026
- Attempted variable shadowing fix (renamed local 'd' to 't_dist')
- Did not resolve issue - functions have separate scopes

## V92 - Diagnostic: r×μ vs r×μ+d (Jan 5, 2026)

**DIAGNOSTIC** - Found r×μ = r×μ+d, confirming d not being added.

---

## V91 - Diagnostic: mu vs mu_p (Jan 5, 2026)

**DIAGNOSTIC** - Compare camera mu vs point mu_p to verify point parameters differ.

### Finding
mu = mu_p (IDENTICAL) - confirms d is not being used in point parameter calculation.

---

## V90 - Diagnostic: Distance d (Jan 5, 2026)

**DIAGNOSTIC** - Output distance d to verify Position input is connected.

### Finding
Distance d is CORRECT - shows proper depth falloff. Issue is NOT Position input.

---

## V89 - Diagnostic: Raw Scattering (Jan 5, 2026)

**DIAGNOSTIC** - Output raw S_cam and S_pt to compare scattering samples.

### Finding
S_cam and S_pt are IDENTICAL - the point position is not affecting texture sampling.
This indicates either d=0, Position not connected, or Scene_Scale=0.

---

## V88 - Fix Transmittance Formula (Jan 5, 2026)

**Critical bug fix** - V86 grayscale banding fix broke the transmittance calculation.

### The Bug
V86 computed: `T = T_cam × (T_cam_gray / T_pt_gray)`
Correct formula: `T = T_cam_gray / T_pt_gray`

For nearby points (d→0), V86 gave T ≈ T_cam (~0.5-0.9) instead of T → 1.
This caused inscatter = S_cam - T×S_pt to be dominated by S_cam, making output look like sky gradient.

### The Fix
- Compute grayscale transmittance: `T_gray = T_cam_gray / T_pt_gray`
- Output uniform RGB: `T = (T_gray, T_gray, T_gray)`
- Clamp to [0, 1]

### Behavior
- For d→0: T→1 (correct - no attenuation for nearby objects)
- For d→∞: T→0 (correct - full attenuation for distant objects)
- Inscatter now properly depends on distance, not just view direction

---

## V87 - Separate Rayleigh and Mie AOVs (Jan 5, 2026)

Separated inscatter output into individual Rayleigh and Mie AOVs for Nuke compositing.

### Changes
- Replaced single `Inscatter` output with separate `Rayleigh` and `Mie` outputs
- Both outputs have phase functions already applied
- Mie includes smoothstep fade for sun below horizon

### Nuke Compositing Formula
```
final_color = object_color * Transmittance + Rayleigh + Mie
```

### Outputs
- **Transmittance** - RGB transmittance for object color attenuation
- **Rayleigh** - Rayleigh scattering with phase function applied
- **Mie** - Mie scattering with phase function and horizon fade applied

---

## V86 - Fix Colored Banding on Objects (Jan 5, 2026)

Fixed colored banding artifacts appearing on objects/buildings in aerial perspective.

### Root Cause
Per-channel RGB division of transmittance samples (`T_cam / T_pt`) amplified texture quantization differences into visible colored bands. The 256x64 transmittance texture has subtle stair-stepping that becomes visible when two nearby samples are divided per-channel.

### Fix
Use grayscale-based transmittance division:
1. Convert both T_cam and T_pt to grayscale (average of RGB)
2. Compute single ratio from grayscale values
3. Apply ratio uniformly to original T_cam color

This eliminates colored banding while preserving overall transmittance magnitude and spectral color.

### Trade-offs
- Eliminates colored banding artifacts ✓
- Minor loss of per-channel spectral variation (visually imperceptible)

---

## V54 - Fix Blue Band Artifact (Jan 4, 2025)

Fixed the blue/cyan band artifact at the horizon.

### Root Cause
The scattering texture has two halves:
- u_mu [0, 0.5]: Ground-intersecting rays
- u_mu [0.5, 1.0]: Non-ground rays

For **aerial perspective** on objects above ground, using the ground formula caused sampling from the wrong texture region, producing very different (incorrect) values at the horizon boundary.

### Fix
Always use non-ground formula for scattering UV in aerial perspective. Objects are above ground, so we sample atmospheric scattering in the sky region. Transmittance still uses proper ground/non-ground selection.

### Debug Tests (V52-V53)
- V52: Force non-ground only → blue band **gone**, correct appearance
- V53: Force ground only → very different values (bright below horizon)

---

## V47 - Full Mie Scattering (Jan 4, 2025)

Added complete Mie scattering implementation.

### Changes
- **Single Mie scattering texture**: Now samples `single_mie_scattering.exr`
- **Mie inscatter**: Computed separately as `mie_cam - T × mie_pt`
- **Mie phase function**: Full Henyey-Greenstein: `(1-g²) / (4π × (1 + g² - 2g×ν)^1.5)`
- **mu_s smoothstep fade**: Mie fades out when sun below horizon (reference lines 1858-1859)
- **Combined output**: `scattering × RayleighPhase + mie × MiePhase` (reference lines 1861-1862)

### Reference Lines
- Single Mie inscatter: lines 1850-1851
- Smoothstep fade: lines 1858-1859
- Phase function combination: lines 1861-1862

### Note
Requires `single_mie_scattering.exr` in LUT cache. This file exists in atmospheric-scattering-3 but may need to be copied or regenerated for atmospheric-scattering-4.

---

## V46 - Pure Bruneton Implementation (Jan 4, 2025)

Complete rewrite to match Eric Bruneton's `GetSkyRadianceToPoint` exactly.

### Changes
- **Removed all workarounds**: No horizon clamping, no geometry-based ground detection
- **ray_r_mu_intersects_ground**: Now computed from camera (r, mu) using exact Bruneton formula:
  ```
  ray_r_mu_intersects_ground = (mu < 0) AND (r²(μ²-1) + bottom² >= 0)
  ```
- **Same flag for transmittance AND scattering**: Both lookups use the same `ray_r_mu_intersects_ground`
- **Transmittance**: Exact implementation of `GetTransmittance` (reference lines 493-519)
  - Non-ground: `T = T(r, mu) / T(r_p, mu_p)`
  - Ground: `T = T(r_p, -mu_p) / T(r, -mu)`
- **Scattering**: Uses correct ground/non-ground UV formula (reference lines 773-831)
- **Inscatter**: Simple `S_cam - T × S_pt` with no blending workarounds

### Reference Lines
- GetSkyRadianceToPoint: lines 1787-1863
- RayIntersectsGround: lines 240-246
- GetTransmittance: lines 493-519
- GetScatteringTextureUvwzFromRMuMuSNu: lines 773-831

---

## V38-V45 - Failed Approaches (Dec 2024 - Jan 2025)

See `FAILED_APPROACHES_LOG.md` for detailed documentation of what was tried and why it failed.

Summary:
- V38-39: Horizon clamp (created bright ground + blue band)
- V40: Below-horizon inscatter blending (treated symptoms, not cause)
- V41: Actual geometry positions (broke LUT parameterization)
- V42: Revert to law-of-cosines
- V43: Ground transmittance formula based on geometry r_p
- V44: Negated mu for ground scattering (wrong approach)
- V45: is_ground based on geometry position (created discontinuities)
