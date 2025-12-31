# Irradiance LUT Debugging Analysis

## Current Status Summary

**What matches:**
- Transmittance LUT: Visual match confirmed
- Irradiance gradients and overall shape: Correct direction (ground brightest, TOA darkest)
- Nighttime cutoff: Working (mu_s < -0.2 → zero)

**What doesn't match:**
- Irradiance values are ~78% of reference maximum (0.192 vs 0.246)
- Scattering values are ~58% of reference maximum (1.85 vs 3.18)
- 17 spurious pixels at mu_s ≈ -0.175 transition region

---

## Architecture Comparison

### Reference Implementation (Bruneton)
- Uses **3D textures** for scattering (GLSL `sampler3D`)
- Stores scattering as `(NU_SIZE, MU_S_SIZE, MU_SIZE, R_SIZE)` → 3D texture with NU packed into X
- GPU hardware does **trilinear interpolation** automatically
- Manual lerp only needed for **nu** dimension (two 3D texture samples)

### Our Implementation (Helios)
- Uses **tiled 2D textures** (Blender GPU limitation - no 3D texture support)
- Layout: `(R_SIZE * WIDTH, HEIGHT)` = `(32 * 256, 128)` = `(8192, 128)`
- Within each R-layer (256 wide): `x = nu_idx * MU_S_SIZE + mu_s_idx`, `y = mu`
- GPU does **bilinear interpolation** on the 2D texture
- Manual lerp needed for **nu** AND **r** dimensions

---

## Detailed Code Comparison

### 1. Single Scattering Computation

**Reference** (`reference_functions.glsl:700-730`):
```glsl
void ComputeSingleScattering(..., OUT rayleigh, OUT mie) {
    const int SAMPLE_COUNT = 50;
    Length dx = DistanceToNearestAtmosphereBoundary(...) / Number(SAMPLE_COUNT);
    
    for (int i = 0; i <= SAMPLE_COUNT; ++i) {
        // ... integration ...
        Number weight_i = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
        rayleigh_sum += rayleigh_i * weight_i;
        mie_sum += mie_i * weight_i;
    }
    rayleigh = rayleigh_sum * dx * atmosphere.solar_irradiance * atmosphere.rayleigh_scattering;
    mie = mie_sum * dx * atmosphere.solar_irradiance * atmosphere.mie_scattering;
}
```

**Ours** (`gpu_precompute.py:326-358`):
```glsl
for (int i = 0; i <= SAMPLE_COUNT; ++i) {
    // ... integration ...
    float weight = (i == 0 || i == SAMPLE_COUNT) ? 0.5 : 1.0;
    rayleigh_sum += trans * rayleigh_density * weight;
    mie_sum += trans * mie_density * weight;
}
rayleigh_sum *= dx * solar_irradiance * rayleigh_scattering;
mie_sum *= dx * solar_irradiance * mie_scattering;
```

**Assessment:** ✅ Logic matches. Both use trapezoidal integration with SAMPLE_COUNT=50.

---

### 2. Scattering Texture Storage

**Reference stores TWO separate outputs:**
- `rayleigh` texture (RGB)
- `mie` texture (RGB)
- Combined scattering: `rayleigh + mie * (mie_scattering / mie_extinction)`

**Ours stores:**
- `scattering` array with RGB = Rayleigh, Alpha = Mie.r
- `delta_mie` array (separate)

**POTENTIAL ISSUE:** When we save the final `scattering.exr`, are we combining Rayleigh + Mie correctly?

Looking at code flow:
1. `precompute_single_scattering()` returns `scattering` (Rayleigh in RGB) and `delta_mie`
2. V35 added: `scattering[:,:,:,:3] += delta_mie[:,:,:,:3] * mie_factor`
3. But this happens BEFORE the multiple scattering loop

**⚠️ ISSUE FOUND:** The Mie combination in V35 happens at line 1755, but then at line 1795 we do:
```python
scattering[:, :, :, :3] += delta_multiple_scattering
```
This overwrites/accumulates into scattering, but `delta_multiple_scattering` already includes both Rayleigh and Mie contributions from the multiple scattering shader. So the V35 fix may not be having the expected effect because:
1. The initial combination is correct
2. But subsequent multiple scattering accumulations may be structured differently

---

### 3. GetScattering (Texture Sampling)

**Reference** (`reference_functions.glsl:958-976`):
```glsl
AbstractSpectrum GetScattering(...) {
    vec4 uvwz = GetScatteringTextureUvwzFromRMuMuSNu(...);  // Returns (u_nu, u_mu_s, u_mu, u_r)
    Number tex_coord_x = uvwz.x * Number(SCATTERING_TEXTURE_NU_SIZE - 1);
    Number tex_x = floor(tex_coord_x);
    Number lerp = tex_coord_x - tex_x;
    vec3 uvw0 = vec3((tex_x + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
    vec3 uvw1 = vec3((tex_x + 1.0 + uvwz.y) / Number(SCATTERING_TEXTURE_NU_SIZE), uvwz.z, uvwz.w);
    return texture(scattering_texture, uvw0) * (1.0 - lerp) + texture(scattering_texture, uvw1) * lerp;
}
```

**Key insight:** Reference uses `uvwz.y` (which is `u_mu_s` in [0,1]) directly within the nu-slice calculation. The formula `(tex_x + uvwz.y) / NU_SIZE` means mu_s varies continuously within each nu slice.

**Ours** (`gpu_precompute.py:876-945`):
```glsl
vec3 SampleScattering(...) {
    // ... compute u_r, layer interpolation ...
    // ... compute u_mu ...
    // ... compute u_mu_s (continuous) ...
    // ... compute nu interpolation (nu0, nu1, nu_lerp) ...
    
    float mu_s_in_slice = u_mu_s;  // [0,1] for the slice
    
    vec3 s00 = texture(tex, vec2(
        (float(layer0) + (float(nu0) + mu_s_in_slice) / float(SCATTERING_TEXTURE_NU_SIZE)) / float(SCATTERING_TEXTURE_DEPTH),
        u_mu)).rgb;
    // ... 4 samples with bilinear lerp ...
}
```

**⚠️ CRITICAL DIFFERENCE:** 
- Reference: `uvw.x = (tex_x + uvwz.y) / NU_SIZE` where `uvwz.y = u_mu_s ∈ [0,1]`
- Ours: `(float(nu0) + mu_s_in_slice) / NU_SIZE` where `mu_s_in_slice = u_mu_s`

The formulas look equivalent, BUT:
- Reference `u_mu_s` comes from `GetTextureCoordFromUnitRange(x_mu_s, MU_S_SIZE)` which maps to **texel centers**
- Our `u_mu_s` also uses `GetTextureCoordFromUnitRange`, so this should be correct

**However**, there's a subtle issue: The reference samples a 3D texture where hardware trilinear interpolation handles mu_s, mu, and r. We're sampling a 2D texture where only bilinear interpolation on (x, y) is available.

---

### 4. Indirect Irradiance Computation

**Reference** (`reference_functions.glsl:1477-1511`):
```glsl
IrradianceSpectrum ComputeIndirectIrradiance(...) {
    const int SAMPLE_COUNT = 32;
    const Angle dphi = pi / Number(SAMPLE_COUNT);
    const Angle dtheta = pi / Number(SAMPLE_COUNT);
    
    vec3 omega_s = vec3(sqrt(1.0 - mu_s * mu_s), 0.0, mu_s);
    for (int j = 0; j < SAMPLE_COUNT / 2; ++j) {  // Upper hemisphere only
        Angle theta = (Number(j) + 0.5) * dtheta;
        for (int i = 0; i < 2 * SAMPLE_COUNT; ++i) {
            Angle phi = (Number(i) + 0.5) * dphi;
            vec3 omega = vec3(cos(phi) * sin(theta), sin(phi) * sin(theta), cos(theta));
            SolidAngle domega = (dtheta / rad) * (dphi / rad) * sin(theta) * sr;
            Number nu = dot(omega, omega_s);
            result += GetScattering(..., r, omega.z, mu_s, nu, false, scattering_order) * omega.z * domega;
        }
    }
    return result;
}
```

**Ours** (`gpu_precompute.py:948-990`):
```glsl
void main() {
    // ... decode r, mu_s from texture coords ...
    vec3 omega_s = vec3(sqrt(1.0 - mu_s * mu_s), 0.0, mu_s);
    
    float dphi = PI / float(SAMPLE_COUNT);  // SAMPLE_COUNT = 32
    float dtheta = PI / float(SAMPLE_COUNT);
    
    for (int j = 0; j < SAMPLE_COUNT / 2; ++j) {
        float theta = (float(j) + 0.5) * dtheta;
        for (int i = 0; i < 2 * SAMPLE_COUNT; ++i) {
            float phi = (float(i) + 0.5) * dphi;
            vec3 omega = vec3(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta);
            float domega = dtheta * dphi * sin_theta;
            float nu = dot(omega, omega_s);
            // ... sample scattering and accumulate ...
            result += scattering * omega.z * domega;
        }
    }
}
```

**Assessment:** ✅ Loop structure matches. The `(dtheta/rad) * (dphi/rad) * sr` in reference simplifies to `dtheta * dphi` in dimensionless code.

---

## Hypothesis: Root Cause of ~78% Scaling

### Hypothesis 1: Scattering Texture Content
Our scattering max is 1.85 vs reference 3.18 (58%). This propagates to irradiance.

**Test:** Compare raw single scattering values at specific (r, mu, mu_s, nu) coordinates before any texture sampling.

### Hypothesis 2: Texture Coordinate Mapping Differences
Our tiled 2D texture may have different effective resolution or interpolation artifacts.

**Test:** Sample both textures at identical (r, mu, mu_s, nu) values and compare.

### Hypothesis 3: Missing Mie Contribution in Final Output
The Mie contribution may not be correctly included in the scattering texture that's saved/compared.

**Test:** Check if reference scattering.npy includes combined Rayleigh+Mie or separate.

### Hypothesis 4: Phase Function Application Timing
Reference applies phase functions at render time for single scattering lookup in indirect irradiance.
We do the same, but there might be a mismatch in when/where this happens.

**Test:** Trace phase function application through both codepaths.

---

## Proposed Systematic Tests

### Test 1: Single Scattering Point Comparison
Create a Python script that:
1. Picks 10 specific (r, mu, mu_s, nu) test points spanning the parameter space
2. Computes single scattering analytically using our GLSL formulas (via Python)
3. Compares with reference values at the same points
4. Reports any discrepancy

### Test 2: Texture Coordinate Mapping Verification
Create a test that:
1. For a given (r, mu, mu_s, nu), compute texture coordinates using our formula
2. Compute texture coordinates using reference formula
3. Compare the resulting (u, v) or (u, v, w, z) values
4. If different, the sampling will be wrong

### Test 3: Reference Scattering Analysis
Analyze the reference scattering.npy to determine:
1. Is it Rayleigh only, Mie only, or combined?
2. What's the actual content at known coordinates?
3. Compare with what we're generating at the same coordinates

### Test 4: Integration Verification
Test the hemisphere integration by:
1. Integrating a known function (e.g., constant = 1) over the hemisphere
2. Should equal 2π (hemisphere solid angle)
3. Verify our domega calculation gives correct total

---

## Recommended Next Steps

1. **Run Test 3 first** - Determine what the reference scattering.npy actually contains
2. **Run Test 1** - Verify single scattering computation matches at specific points
3. If single scattering matches, the issue is in texture sampling/storage
4. If single scattering doesn't match, trace the integration to find the divergence

---

## Files Involved

| Component | Our File | Reference |
|-----------|----------|-----------|
| Transmittance | `gpu_precompute.py:80-166` | `reference_functions.glsl:400-520` |
| Single Scattering | `gpu_precompute.py:169-362` | `reference_functions.glsl:600-730` |
| Texture Coords | `gpu_precompute.py:872-945` | `reference_functions.glsl:773-890` |
| Indirect Irradiance | `gpu_precompute.py:948-1000` | `reference_functions.glsl:1477-1511` |
| GetScattering | `gpu_precompute.py:872-945` | `reference_functions.glsl:958-1008` |

---

## Known Intentional Differences

1. **2D tiled texture vs 3D texture** - Required due to Blender GPU limitations
2. **Manual 4D interpolation** - Compensates for lack of 3D textures
3. **mu_s_min cutoff** - Added to eliminate spurious nighttime energy

---

## Version History

- V32: Removed flipud from texture creation
- V33: Added mu_s < -0.2 cutoff in SampleScattering  
- V34: Added 4D interpolation (lerp across nu and r)
- V35: Added Mie contribution to scattering (may not be effective - see analysis above)
