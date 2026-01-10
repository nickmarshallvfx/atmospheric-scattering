# Step 2.4b Implementation Specification

## Overview

This document specifies the implementation of **Step 2.4b**, a unified inscatter shader that combines LUT-based scattering AND LUT-based transmittance in a single, cleanly-built function.

## Context

### Project: Helios Atmospheric Scattering for Blender

We are implementing Eric Bruneton's precomputed atmospheric scattering model as a Blender shader node graph. The goal is to produce physically accurate aerial perspective for film VFX, with AOV outputs for compositing in Nuke.

### Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| **Step 2.4** (LUT Scattering + Exponential Transmittance) | ✅ Working | Produces correct inscatter with exponential T approximation |
| **Step 6** (LUT Transmittance standalone) | ✅ Working | Correct T with ground handling and horizon fallback |
| **Step 11** (Attempted integration) | ❌ Broken | Node injection approach failed multiple times |

### Why Step 11 Failed

Step 11 attempted to:
1. Run Step 2.4 (creates scattering nodes)
2. Find existing nodes in the graph
3. Inject LUT transmittance nodes

**Problems:**
- Fragile node-finding based on node types and values
- Duplicate geometry calculations (Step 2.4's + new injected ones)
- Inconsistent variable references between the two sets of nodes
- Multiple V130-V134 fixes still didn't resolve issues

**Root cause:** Node injection into an existing graph is inherently fragile.

---

## Step 2.4b Design

### Philosophy

Build everything **from scratch in one function**. No node finding. No injection. All geometry nodes are created once and shared by both scattering and transmittance calculations.

### Architecture

```
apply_step_2_4b_full_inscatter()
│
├── 1. SETUP
│   ├── Load LUTs (scattering.exr, transmittance.exr)
│   ├── Get camera position → r_cam, rho_cam
│   └── Get sun direction → sun_dir vector
│
├── 2. GEOMETRY (built ONCE, shared by scattering AND transmittance)
│   ├── geo.Position → world position of shading point
│   ├── cam_pos → camera world position
│   ├── view_vec = Position - cam_pos
│   ├── d = length(view_vec) * 0.001  [distance in km]
│   ├── view_dir = normalize(view_vec)
│   ├── up = (0, 0, 1)
│   └── mu = dot(view_dir, up)  [view zenith cosine]
│
├── 3. SUN PARAMETERS
│   ├── mu_s = dot(sun_dir, up)  [sun zenith cosine]
│   └── nu = dot(view_dir, sun_dir)  [view-sun angle cosine]
│
├── 4. POINT PARAMETERS (r_d, mu_d, mu_s_d) - used by BOTH scattering and transmittance
│   ├── r_d² = d² + 2*r*mu*d + r²
│   ├── r_d = sqrt(r_d²), clamped to [BOTTOM, TOP]
│   ├── mu_d = (r*mu + d) / r_d, clamped to [-1, 1]
│   └── mu_s_d = (r*mu_s + d*nu) / r_d, clamped to [-1, 1]
│
├── 5. SCATTERING AT CAMERA (S_cam)
│   ├── create_scatter_uv(r_cam, mu, mu_s, nu) → UV coordinates
│   ├── Sample scattering.exr RGB → Rayleigh
│   └── Sample scattering.exr Alpha → Mie.r
│
├── 6. SCATTERING AT POINT (S_pt)
│   ├── create_scatter_uv(r_d, mu_d, mu_s_d, nu) → UV coordinates
│   ├── Sample scattering.exr RGB → Rayleigh_pt
│   └── Sample scattering.exr Alpha → Mie_pt.r
│
├── 7. TRANSMITTANCE (T) - uses SAME r_d, mu_d from step 4
│   │
│   ├── 7a. Ground flag (FULL Bruneton check)
│   │   └── ground_flag = 1 if (mu < 0 AND r²(mu²-1) + bottom² >= 0), else 0
│   │
│   ├── 7b. Transmittance UV for 4 sample points
│   │   ├── SKY numerator: create_trans_uv(r_cam, mu)
│   │   ├── SKY denominator: create_trans_uv(r_d, mu_d)
│   │   ├── GROUND numerator: create_trans_uv(r_d, -mu_d)
│   │   └── GROUND denominator: create_trans_uv(r_cam, -mu)
│   │
│   ├── 7c. Sample transmittance.exr 4 times
│   │
│   ├── 7d. Compute ratios (with epsilon=1e-6 to prevent div-by-zero)
│   │   ├── T_sky = T(r,mu) / max(T(r_d,mu_d), 1e-6)
│   │   └── T_gnd = T(r_d,-mu_d) / max(T(r,-mu), 1e-6)
│   │
│   ├── 7e. Select based on ground_flag
│   │   └── T_lut = mix(T_sky, T_gnd, ground_flag)
│   │
│   ├── 7f. Horizon fallback (exponential when |mu| < HORIZON_EPSILON)
│   │   ├── HORIZON_EPSILON = 0.02 (configurable, ~1° from horizon)
│   │   ├── horizon_factor = 1 - clamp(|mu| / HORIZON_EPSILON, 0, 1)
│   │   ├── T_exp = exp(-d * [k_r, k_g, k_b])  where k=[0.02, 0.03, 0.05]
│   │   └── T_final = mix(T_lut, T_exp, horizon_factor)
│   │
│   └── T output: T_final
│
├── 8. INSCATTER COMPUTATION
│   ├── Rayleigh_inscatter = S_cam.rgb - T × S_pt.rgb
│   ├── Mie_inscatter = S_cam.a - T.r × S_pt.a  [use red channel for Mie]
│   └── Clamp to >= 0
│
├── 9. PHASE FUNCTIONS
│   ├── Rayleigh_phase = (3/16π) × (1 + nu²)
│   ├── Mie_phase = k × (1 + nu²) / (1 + g² - 2*g*nu)^1.5
│   │   where k = (3/8π) × (1-g²)/(2+g²), g = 0.8
│   └── Combined = Rayleigh × Rayleigh_phase + Mie × Mie_phase
│
├── 10. OUTPUT
│   ├── Emission node → Material Output
│   └── Debug modes: 0=inscatter, 1=T, 2=S_cam, 3=S_pt, 4=phase
│
└── 11. AOVs
    ├── Helios_Sky: placeholder (black)
    ├── Helios_Transmittance: T_final
    ├── Helios_Rayleigh: Rayleigh_inscatter × Rayleigh_phase
    ├── Helios_Mie: Mie_inscatter × Mie_phase
    └── Helios_SunDisk: placeholder (black)
```

---

## Key Formulas

### Bruneton Reference: `functions.glsl`

**Transmittance UV (lines 175-205):**
```glsl
// For texture coordinate from unit range [0,1] to texture coordinate
float GetTextureCoordFromUnitRange(float x, int size) {
    return 0.5 / float(size) + x * (1.0 - 1.0 / float(size));
}

// Transmittance texture parameterization
rho = sqrt(r*r - bottom*bottom);
H = sqrt(top*top - bottom*bottom);
x_r = rho / H;
v = GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_HEIGHT);

// For mu (zenith angle)
d = DistanceToTopAtmosphereBoundary(r, mu);  // = -r*mu + sqrt(r²(mu²-1) + top²)
d_min = top - r;
d_max = rho + H;
x_mu = (d - d_min) / (d_max - d_min);
u = GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_WIDTH);
```

**GetTransmittance (lines 493-518):**
```glsl
float r_d = ClampRadius(sqrt(d*d + 2*r*mu*d + r*r));
float mu_d = ClampCosine((r*mu + d) / r_d);

if (ray_r_mu_intersects_ground) {
    return GetTransmittanceToTopAtmosphereBoundary(r_d, -mu_d) /
           GetTransmittanceToTopAtmosphereBoundary(r, -mu);
} else {
    return GetTransmittanceToTopAtmosphereBoundary(r, mu) /
           GetTransmittanceToTopAtmosphereBoundary(r_d, mu_d);
}
```

**RayIntersectsGround (lines 240-246):**
```glsl
bool RayIntersectsGround(float r, float mu) {
    return mu < 0.0 && r*r*(mu*mu - 1.0) + bottom*bottom >= 0.0;
}
```

---

## Implementation Notes

### Blender Shader Nodes

| Operation | Node Type | Notes |
|-----------|-----------|-------|
| Math operations | ShaderNodeMath | Operations: ADD, MULTIPLY, DIVIDE, SQRT, etc. |
| Vector math | ShaderNodeVectorMath | DOT_PRODUCT, NORMALIZE, LENGTH, SUBTRACT |
| Clamp | ShaderNodeClamp | Min/Max inputs, outputs 'Result' |
| Image texture | ShaderNodeTexImage | Set colorspace to 'Non-Color' for LUTs |
| Separate RGB | ShaderNodeSeparateColor | Outputs: Red, Green, Blue |
| Combine RGB | ShaderNodeCombineColor | Inputs: Red, Green, Blue |
| Mix | ShaderNodeMix | data_type='RGBA', blend_type='MIX' or 'MULTIPLY' |

### Critical Details

1. **Output socket names:**
   - CLAMP nodes: `outputs['Result']`
   - MATH nodes: `outputs['Value']`
   - Mix RGBA nodes: `outputs[2]` for color output

2. **Mix node inputs for RGBA:**
   - `inputs['Factor']` for blend factor
   - `inputs[6]` for color A
   - `inputs[7]` for color B

3. **Distance units:**
   - Blender uses meters, Bruneton uses kilometers
   - Convert: `d_km = d_blender * 0.001`

4. **Coordinate system:**
   - Blender: Z-up
   - up vector = (0, 0, 1)

---

## Constants

```python
BOTTOM_RADIUS = 6360.0  # km
TOP_RADIUS = 6420.0     # km
H = sqrt(TOP² - BOTTOM²) ≈ 79.8 km

TRANSMITTANCE_WIDTH = 256
TRANSMITTANCE_HEIGHT = 64

SCATTERING_R = 32
SCATTERING_MU = 128
SCATTERING_MU_S = 32
SCATTERING_NU = 8

MIE_G = 0.8
MU_S_MIN = -0.2

HORIZON_EPSILON = 0.1   # Start conservative (~5.7°), reduce later if needed
DIV_EPSILON = 1e-6      # Safe division epsilon
```

---

## Validation Criteria

### Visual Tests

1. **Distant objects should be blue-tinted** (Rayleigh scattering)
2. **Near objects should have minimal inscatter**
3. **No bright band at horizon**
4. **No discontinuities at ground level**
5. **Inscatter should increase with distance**

### Numerical Tests

1. **B/R ratio ≈ 1.69** for inscatter (Rayleigh characteristic)
2. **Transmittance → 1.0 for d → 0** (no attenuation for nearby objects)
3. **Transmittance → 0 for very large d** (full attenuation)

### Debug Modes

| Mode | Output | Expected |
|------|--------|----------|
| 0 | Full inscatter | Blue-tinted, distance-dependent |
| 1 | Transmittance only | White near, darker far |
| 2 | S_cam only | Blue gradient (sky scattering) |
| 3 | S_pt only | Similar to S_cam but per-object |
| 4 | Phase only | Sun-dependent brightness variation |

---

## LUT Files

Located at: `C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-3\helios_cache\luts\`

| File | Dimensions | Contents |
|------|------------|----------|
| `transmittance.exr` | 256×64 | RGB transmittance to atmosphere top |
| `scattering.exr` | 256×128×32 (flattened) | RGB=Rayleigh, A=Mie.r |

---

## Questions for Review (RESOLVED)

1. **Ground detection formula** → **USE FULL CHECK**
   - Simple `mu < 0` fails for high-altitude cameras
   - Full: `mu < 0 AND r²(μ²-1) + bottom² >= 0`

2. **Horizon fallback threshold** → **0.02 (~1°)**
   - 0.1 was too wide (5.7°), causes visible softness
   - Make configurable for tuning

3. **Share r_d/mu_d** → **YES**
   - Ensures mathematical coherence
   - Prevents fireflies from floating-point drift

4. **Mie phase function** → **STANDARD HG IS FINE**
   - g=0.8 with Bruneton's k factor
   - Easier to tweak for VFX

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| V134 | Jan 9, 2026 | Failed node injection approach |
| V135 | Jan 9, 2026 | Step 2.4b spec created (this document) |
| V135.1 | Jan 9, 2026 | Gemini review incorporated (full ground check, 0.02 horizon, epsilon) |

---

## Implementation Checklist

- [ ] Create `apply_step_2_4b_full_inscatter()` function
- [ ] Implement geometry section (d, mu, view_dir)
- [ ] Implement sun parameters (mu_s, nu)
- [ ] Implement point parameters (r_d, mu_d, mu_s_d)
- [ ] Implement scattering at camera (S_cam)
- [ ] Implement scattering at point (S_pt)
- [ ] Implement transmittance with ground handling
- [ ] Implement horizon fallback
- [ ] Implement inscatter formula
- [ ] Implement phase functions
- [ ] Add debug modes
- [ ] Register AOVs
- [ ] Test and validate
