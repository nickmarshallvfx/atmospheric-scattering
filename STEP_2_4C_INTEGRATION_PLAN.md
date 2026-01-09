# Step 2.4c Integration Plan

## The Bruneton Aerial Perspective Formula

```
inscatter = S_cam - T × S_pt
```

Where:
- `S_cam` = Scattering LUT sampled at camera position
- `S_pt` = Scattering LUT sampled at geometry point
- `T` = Transmittance from camera to point

---

## ALL PARAMETERS - Complete List

### Geometry Parameters (Shared Between Scattering and Transmittance)

| Parameter | Definition | Computed From | Validated? |
|-----------|------------|---------------|------------|
| `r_cam` | Camera radius from planet center | `BOTTOM_RADIUS + camera_altitude` | Constant (6360.006 km) |
| `d` | Distance from camera to point | `length(camera_pos - geometry_pos)` in km | YES - vertical profile correct |
| `r_p` | Point radius from planet center | `sqrt(d² + 2*r_cam*mu*d + r_cam²)` | YES - matches law-of-cosines |
| `mu` | View zenith cosine at camera | `dot(view_dir, up_vec)` | YES - -1 to +1 range correct |
| `mu_p` | View zenith cosine at point | `(r_cam*mu + d) / r_p` | YES - derived from r_p |
| `mu_s` | Sun zenith cosine at camera | `dot(sun_dir, up_vec)` | Constant per frame |
| `mu_s_p` | Sun zenith cosine at point | `(r_cam*mu_s + d*nu) / r_p` | Derived (law-of-cosines) |
| `nu` | View-sun angle cosine | `dot(view_dir, sun_dir)` | Constant per pixel |

### Derived Flags

| Flag | Definition | Used By |
|------|------------|---------|
| `ground_flag` | `mu < 0` | Transmittance formula selection |
| `horizon_factor` | `1 - clamp(\|mu\|/0.1, 0, 1)` | Horizon blending |

---

## CRITICAL: Which Parameters Are Shared?

### Parameters that MUST be identical between Scattering and Transmittance:

1. **`r_cam`** - Same camera radius
2. **`mu`** - Same view zenith cosine
3. **`d`** - Same distance (in km)
4. **`r_p`** - Same point radius (computed from r_cam, mu, d)
5. **`mu_p`** - Same point zenith (computed from r_cam, mu, d, r_p)

### Parameters only used by Scattering:

1. **`mu_s`** - Sun zenith at camera
2. **`mu_s_p`** - Sun zenith at point
3. **`nu`** - View-sun angle

### Parameters only used by Transmittance:

1. **`ground_flag`** - For formula selection
2. **`horizon_factor`** - For exponential fallback blending
3. **`neg_mu`** and **`neg_mu_p`** - Negated values for ground case

---

## VALIDATION SUMMARY

### What Each Validation Proved:

| Validation | Result | What It Proves |
|------------|--------|----------------|
| `mu` | PASSED | View direction dot product is computed correctly. Range [-1, +1]. Red at zenith (mu=1), green at nadir (mu=-1), horizon at mu≈0. |
| `d` | PASSED | Distance in km computed correctly from geometry. Near objects (1-10m) show ~0.001-0.01 km. Far objects show larger values. |
| `r_p` | PASSED | Point radius follows law-of-cosines: `sqrt(d² + 2*r_cam*mu*d + r_cam²)`. Values near r_cam for close objects, varies with distance and angle. |
| `mu_p` | PASSED | Point zenith follows: `(r_cam*mu + d) / r_p`. Values derived correctly from shared parameters. |
| `trans_uv_cam` | PASSED | UV coordinates for transmittance LUT correct at camera position. U varies with mu, V constant for fixed altitude. |
| `trans_ratio` | PASSED (with horizon blend) | Full Bruneton transmittance ratio working. Horizon discontinuity smoothed with exponential fallback. |

### What Remains Unvalidated (but working in Step 2.4):

- `mu_s`, `mu_s_p`, `nu` - These are used in scattering UV calculation
- Scattering UV coordinates - Working in existing Step 2.4
- Scattering texture sampling - Working in existing Step 2.4

---

## THE INTEGRATION APPROACH

### Key Insight: Don't Rebuild Everything

The existing Step 2.4 (`apply_step_2_4_lut_scattering`) already has **working scattering**:
- Correct scattering UV calculation for camera (S_cam)
- Correct scattering UV calculation for point (S_pt)  
- Correct geometry parameters (r_cam, mu, d, r_p, mu_p, mu_s, mu_s_p, nu)
- Working exponential transmittance approximation

**The only thing that needs to change**: Replace the exponential transmittance with LUT transmittance.

### Integration Strategy

1. **Keep ALL geometry node creation from Step 2.4** - proven working
2. **Keep ALL scattering UV and sampling from Step 2.4** - proven working
3. **Replace ONLY the transmittance section** with our validated trans_ratio logic
4. **Add horizon blending** to the transmittance

---

## DETAILED NODE STRUCTURE

### Phase 1: Geometry (from Step 2.4 - keep as-is)

```
Camera Position → d (distance in km)
                → view_dir
                → up_vec = (0, 0, 1)

view_dir · up_vec → mu (view zenith)

r_cam = BOTTOM_RADIUS + altitude (constant 6360.006)

d, r_cam, mu → r_p = sqrt(d² + 2*r_cam*mu*d + r_cam²)
            → mu_p = (r_cam*mu + d) / r_p

sun_dir · up_vec → mu_s
view_dir · sun_dir → nu
mu_s, nu, d, r_p → mu_s_p = (r_cam*mu_s + d*nu) / r_p
```

### Phase 2: Scattering (from Step 2.4 - keep as-is)

```
r_cam, mu, mu_s, nu → scatter_uv_cam → S_cam (LUT sample)
r_p, mu_p, mu_s_p, nu → scatter_uv_pt → S_pt (LUT sample)
```

### Phase 3: Transmittance (REPLACE with validated trans_ratio)

```
ground_flag = (mu < 0)
horizon_factor = 1 - clamp(|mu|/0.1, 0, 1)

neg_mu = -mu
neg_mu_p = -mu_p

# Sky case: T = T(r_cam, mu) / T(r_p, mu_p)
uv_sky_num = trans_uv(r_cam, mu)
uv_sky_den = trans_uv(r_p, mu_p)
t_sky = sample(uv_sky_num) / sample(uv_sky_den)

# Ground case: T = T(r_p, -mu_p) / T(r_cam, -mu)  
uv_gnd_num = trans_uv(r_p, neg_mu_p)
uv_gnd_den = trans_uv(r_cam, neg_mu)
t_gnd = sample(uv_gnd_num) / sample(uv_gnd_den)

# LUT result
t_lut = mix(t_sky, t_gnd, ground_flag)

# Exponential fallback for horizon
t_exp = exp(-d * [0.02, 0.03, 0.05])

# Final transmittance
T = mix(t_lut, t_exp, horizon_factor)
```

### Phase 4: Combine (from Step 2.4 - modify to use new T)

```
inscatter = S_cam - T × S_pt
final_color = inscatter (with phase functions if needed)
```

---

## MATH VERIFICATION CHECKLIST

### Law-of-Cosines Formulas (from Bruneton reference):

1. **r_p** (point radius):
   ```
   r_p = sqrt(d² + 2*r*mu*d + r²)
   ```
   ✅ Verified in validation - matches reference `functions.glsl` line 501

2. **mu_p** (point zenith):
   ```
   mu_p = (r*mu + d) / r_p
   ```
   ✅ Verified in validation - matches reference `functions.glsl` line 502

3. **Transmittance UV** (from `GetTransmittanceTextureUvFromRMu`):
   ```
   H = sqrt(top² - bottom²)
   rho = sqrt(r² - bottom²)
   d_to_top = -r*mu + sqrt(r²*(mu²-1) + top²)
   d_min = top - r
   d_max = rho + H
   x_mu = (d_to_top - d_min) / (d_max - d_min)
   x_r = rho / H
   U = x_mu * (WIDTH-1)/WIDTH + 0.5/WIDTH
   V = x_r * (HEIGHT-1)/HEIGHT + 0.5/HEIGHT
   ```
   ✅ Verified in trans_uv_cam validation

4. **Transmittance Ratio** (from `GetTransmittance`):
   ```
   if ray_intersects_ground:
       T = T(r_p, -mu_p) / T(r, -mu)
   else:
       T = T(r, mu) / T(r_p, mu_p)
   ```
   ✅ Verified in trans_ratio validation

---

## IMPLEMENTATION STEPS

### Step 1: Copy Working Geometry from Step 2.4

- [ ] Copy camera position calculation
- [ ] Copy d (distance) calculation  
- [ ] Copy view_dir, up_vec, mu calculation
- [ ] Copy r_p, mu_p calculation (law-of-cosines)
- [ ] Copy mu_s, mu_s_p, nu calculation

### Step 2: Copy Working Scattering from Step 2.4

- [ ] Copy create_scatter_uv helper function
- [ ] Copy S_cam sampling with depth interpolation
- [ ] Copy S_pt sampling with depth interpolation

### Step 3: Add Validated Transmittance

- [ ] Add ground_flag (mu < 0)
- [ ] Add horizon_factor (1 - clamp(|mu|/0.1, 0, 1))
- [ ] Add neg_mu, neg_mu_p
- [ ] Add create_dynamic_trans_uv helper (from trans_ratio validation)
- [ ] Add sky case UV and sampling
- [ ] Add ground case UV and sampling
- [ ] Add safe division (clamp to 0-1)
- [ ] Add t_lut mix (sky vs ground based on ground_flag)
- [ ] Add exponential fallback
- [ ] Add t_final mix (lut vs exp based on horizon_factor)

### Step 4: Combine

- [ ] T × S_pt multiplication
- [ ] S_cam - T×S_pt subtraction
- [ ] Output to emission

---

## CONSTANTS (must match between components)

```python
BOTTOM_RADIUS = 6360.0  # km
TOP_RADIUS = 6420.0     # km
H = sqrt(TOP_RADIUS² - BOTTOM_RADIUS²) ≈ 79.79 km

TRANSMITTANCE_WIDTH = 256
TRANSMITTANCE_HEIGHT = 64

SCATTERING_R_SIZE = 32
SCATTERING_MU_SIZE = 128
SCATTERING_MU_S_SIZE = 32
SCATTERING_NU_SIZE = 8
```

---

## POTENTIAL FAILURE MODES TO AVOID

1. **Different d values**: Scattering and transmittance must use the SAME distance node
2. **Different mu values**: Must share the SAME mu_dot node
3. **Different r_p values**: Must share the SAME r_p calculation
4. **Unit mismatch**: d must be in km everywhere
5. **Output socket mismatch**: Clamp nodes use 'Result', Math nodes use 'Value'
6. **Horizon discontinuity**: Must have horizon blending enabled

---

## SUCCESS CRITERIA

1. Close objects: High transmittance (~1), minimal inscatter contribution
2. Far objects: Lower transmittance, visible blue haze
3. Horizon region: Smooth transition, no red spike or discontinuity
4. Sky-looking vs ground-looking: Both cases handled correctly
5. Overall appearance: Similar to working Step 2.4 but with better transmittance accuracy

---

## EXISTING STEP 2.4 NODE STRUCTURE (lines 2500-3400)

### Verified Working Nodes to KEEP:

```
Lines 2557-2573: d (distance)
  - pos_to_cam (SUBTRACT: geometry_pos - cam_pos)
  - d_vec_len (LENGTH)
  - d (MULTIPLY by 0.001 → km)

Lines 2579-2582: view_dir
  - view_dir (NORMALIZE pos_to_cam)

Lines 2588-2614: r_cam, mu
  - r (VALUE: BOTTOM_RADIUS + altitude)
  - up_at_cam (COMBINE_XYZ: 0, 0, 1)
  - mu_dot (DOT_PRODUCT: view_dir, up_at_cam)
  - mu (MULTIPLY by 1.0 passthrough)

Lines 2620-2644: mu_s, nu
  - mu_s_dot, mu_s (sun zenith)
  - nu_dot, nu (view-sun angle)

Lines 2651-2742: r_p, mu_p
  - Law-of-cosines calculation
  - Clamped to valid ranges

Lines 2744-2780: mu_s_p
  - (r*mu_s + d*nu) / r_p

Lines 2782-3172: create_scatter_uv helper
  - Full scattering UV calculation

Lines 3178-3230: S_cam, S_pt sampling
  - With depth interpolation
```

### Nodes to REPLACE (lines 3236-3253):

```
Lines 3236-3253: TRANSMITTANCE (currently exponential)
  - neg_d (MULTIPLY by -0.1)
  - trans_approx (EXPONENT)
  - t_rgb (COMBINE_COLOR)

REPLACE WITH: Full LUT transmittance ratio + horizon blending
```

### Nodes to KEEP (lines 3259-end):

```
Lines 3259-3275: INSCATTER combination
  - t_times_spt (MULTIPLY: T × S_pt)
  - inscatter (SUBTRACT: S_cam - T×S_pt)

Lines 3282-3400: Phase functions
  - Rayleigh and Mie phase calculations
```

---

## EXACT REPLACEMENT PLAN

### What Gets Removed:
- `neg_d` node (line 3237)
- `trans_approx` node (line 3243)
- `t_rgb` node (line 3249)

### What Gets Added (from validated trans_ratio):

1. **ground_flag**: `mu < 0` detection
2. **horizon_factor**: `1 - clamp(|mu|/0.1, 0, 1)`
3. **neg_mu, neg_mu_p**: Negated values for ground case
4. **create_dynamic_trans_uv**: Helper for LUT UV calculation
5. **Sky case sampling**: T(r_cam, mu) / T(r_p, mu_p)
6. **Ground case sampling**: T(r_p, -mu_p) / T(r_cam, -mu)
7. **t_lut**: Mix of sky/ground based on ground_flag
8. **t_exp_rgb**: Exponential fallback
9. **t_final**: Mix of LUT/exponential based on horizon_factor

### Connection Points:
- INPUT: `d` node, `mu` node (from mu_dot), `r` node, `r_p` node, `mu_p` node
- OUTPUT: `t_rgb` (replaced by `t_final`) → connects to `t_times_spt`
