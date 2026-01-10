# Scattering AOV Debug Handoff Document

**Date:** January 5, 2026  
**Status:** UNRESOLVED after V89-V97 debugging  
**Primary File:** `helios/aerial_nodes.py`

---

## Problem Statement

The Rayleigh and Mie scattering AOVs are broken:
1. **Rayleigh** appears as sky gradient, not scattered light over objects
2. **Mie** appears as grayscale copy of Rayleigh, not responding to sun position
3. Objects below horizon are not correctly held out
4. The combined inscatter looks nothing like Bruneton reference output

Visual symptom: Horizontal banding/gradient that doesn't respect object depth or atmospheric physics.

---

## What Was Confirmed Working

### V96 Debug Output Confirmed:
```
r_mu_plus_d.inputs[0].is_linked = True
r_mu_plus_d.inputs[1].is_linked = True
r_mu_plus_d.inputs[0] linked from: r×μ
r_mu_plus_d.inputs[1] linked from: d
```

**The Blender node graph links ARE being created correctly.**

### V90 Confirmed:
- Distance `d` (camera to point) is calculated correctly
- Shows proper depth falloff when visualized directly
- `d = builder.vec_math('LENGTH', ...)` at line 884 works

---

## What Was NOT Working

### V89 Finding:
- `S_cam` (scattering at camera) and `S_pt` (scattering at point) are **identical**
- This means the scattering texture is being sampled at the same UV for both

### V91-V92 Finding:
- `mu` and `mu_p` appear identical when visualized
- `r×μ` and `r×μ+d` appear identical when visualized

### Root Cause Hypothesis (UNVERIFIED):
The parameters differ by such tiny amounts that:
1. Visual comparison is imperceptible (d ~1km vs r×μ ~6360km = 0.016% difference)
2. Scattering texture UV quantization may map both to same texel

**However**, the inscatter formula `inscatter = S_cam - T × S_pt` should still work because transmittance T varies significantly with distance. This is NOT happening correctly.

---

## Key Code Locations

### Main Implementation
`helios/aerial_nodes.py`

### Critical Sections:

**Distance calculation (line ~884):**
```python
d = builder.vec_math('LENGTH', -1600, 50, 'd')
builder.link(pt_minus_cam.outputs[0], d.inputs[0])
```

**Point parameters (lines ~1000-1080):**
```python
# r_p = sqrt(d² + 2·r·μ·d + r²)
# mu_p = (r·μ + d) / r_p
# mu_s_p = (r·μ_s + d·ν) / r_p
```

**r×μ+d calculation (lines 1042-1044):**
```python
r_mu_plus_d = builder.math('ADD', 0, -300, 'r×μ+d')
builder.link(r_mu.outputs[0], r_mu_plus_d.inputs[0])
builder.link(d.outputs['Value'], r_mu_plus_d.inputs[1])
```

**Inscatter subtraction (lines ~1300-1340):**
```python
# rayleigh_inscatter = scat_cam_rayleigh - trans_to_pt × scat_pt_rayleigh
# mie_inscatter = scat_cam_mie - trans_to_pt × scat_pt_mie
```

**Scattering UV creation:**
- Function `create_scattering_uv()` starting around line 479
- Uses `ray_intersects_ground_socket` to select ground vs non-ground formula
- Ground rays: u_mu in [0, 0.5]
- Non-ground rays: u_mu in [0.5, 1.0]

**Transmittance UV creation:**
- Function `create_transmittance_uv()` starting around line 148

---

## Diagnostic Versions Summary

| Version | Test | Result |
|---------|------|--------|
| V89 | Output raw S_cam vs S_pt | Identical |
| V90 | Output distance d | ✅ Correct depth falloff |
| V91 | Output mu vs mu_p | Identical |
| V92 | Output r×μ vs r×μ+d | Identical |
| V93 | Fix variable shadowing | No change |
| V94 | Fresh nodes at end (inputs[1]) | Identical |
| V95 | Fresh nodes at end (inputs[0]) | Identical |
| V96 | Debug prints | ✅ Links ARE correct |
| V97 | Restore actual output | Still broken |

---

## Potential Issues NOT Yet Investigated

### 1. Transmittance Calculation
The transmittance path has complex ground/non-ground switching:
- `trans_uv_cam_ng`, `trans_uv_cam_g` (camera, non-ground/ground)
- `trans_uv_pt_ng`, `trans_uv_pt_g` (point, non-ground/ground)
- Mix node selects based on `ray_intersects_ground`

**Possible issue:** Ground intersection flag may not be correctly computed or applied.

### 2. Scattering UV Calculation
Similar ground/non-ground switching for scattering:
- Different formulas for ground vs non-ground rays
- u_mu ranges are different ([0, 0.5] vs [0.5, 1.0])

**Possible issue:** The UV formulas may have errors in the ground/non-ground path selection.

### 3. Phase Function Application
```python
rayleigh_phased = builder.vec_math('SCALE', 6700, 50, 'Rayleigh_Phased')
mie_phased = builder.vec_math('SCALE', 6700, -50, 'Mie_Phased')
```

**Possible issue:** Phase functions may be incorrectly calculated or applied.

### 4. Scene Scale
The `Scene_Scale` input converts Blender units to km:
```python
camera_km = builder.vec_math('SCALE', -2000, 200, 'Camera_km')
builder.link(group_input.outputs['Scene_Scale'], camera_km.inputs['Scale'])
```

**Possible issue:** If Scene_Scale is wrong, all distance-dependent calculations will be incorrect.

### 5. The Inscatter Subtraction Formula
```python
# inscatter = S_cam - T × S_pt
rayleigh_inscatter = builder.vec_math('SUBTRACT', ...)
builder.link(scat_cam_rayleigh.outputs[0], rayleigh_inscatter.inputs[0])
builder.link(scat_pt_trans.outputs[0], rayleigh_inscatter.inputs[1])
```

**Possible issue:** The subtraction order or transmittance application may be wrong.

---

## Reference Implementation

The Bruneton reference GLSL is in:
- `reference/functions.glsl`
- Key function: `GetSkyRadianceToPoint` (around line 1790)

Reference formula:
```glsl
// Compute inscattering between camera and point
vec3 transmittance;
vec3 in_scatter = GetSkyRadianceToPoint(camera, point, shadow_length, sun_direction, transmittance);
```

---

## Coordinate System Notes

- **Blender:** Z-up
- **Nuke:** Y-up
- **Bruneton:** Uses planet-centered coordinates where camera/point positions are relative to planet center

---

## Files Modified During Debug

1. `helios/aerial_nodes.py` - Main node group (currently V97)
2. `BLUE_BAND_DEBUG_LOG.md` - Detailed debug log
3. `CHANGELOG.md` - Version history

---

## Recommended Next Steps

1. **Compare against working reference:** The `atmospheric-scattering-2` repository has a working implementation. Direct comparison of UV calculations may reveal differences.

2. **Test with extreme distances:** Use objects at 100km+ to see if inscatter becomes visible (would confirm scale issue).

3. **Output intermediate UVs:** Visualize the actual UV coordinates being sent to scattering texture for camera vs point.

4. **Check transmittance path separately:** The transmittance AOV may also be incorrect, which would break inscatter.

5. **Verify Scene_Scale value:** Confirm what value is being passed and if it correctly converts to km.

6. **Review ground intersection logic:** The `ray_intersects_ground` calculation determines which formula branch is used.

---

## Constants

```python
BOTTOM_RADIUS = 6360.0  # km (Earth surface)
TOP_RADIUS = 6420.0     # km (top of atmosphere)
MU_S_MIN = -0.2         # Minimum sun zenith cosine
H = sqrt(TOP² - BOTTOM²) ≈ 79.8 km
```

---

## Summary

After 9 diagnostic versions (V89-V97), confirmed that:
- Node links ARE being created correctly
- Distance `d` IS being calculated correctly
- The math operations ARE connected properly

The issue appears to be that despite correct linking, the **output is still wrong**. This suggests either:
1. A formula error in the inscatter calculation itself
2. Incorrect UV mapping causing same texels to be sampled
3. Ground/non-ground path selection issues
4. Transmittance calculation errors affecting the subtraction

The visual symptom (horizontal gradient ignoring object depth) suggests the point-specific calculations are not producing meaningfully different results from camera calculations.
