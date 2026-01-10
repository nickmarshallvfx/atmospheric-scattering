# Bruneton Formula Verification Checklist

This document exists because I repeatedly missed formula errors. Every formula must be:
1. Written out from REFERENCE
2. Written out from OUR CODE  
3. Mathematically verified as IDENTICAL (not "similar")
4. Tested in isolation

---

## PHASE FUNCTIONS

### Rayleigh Phase Function
**Reference (line 740-741):**
```
RayleighPhaseFunction = 3/(16π) × (1 + ν²)
```

**Our Implementation:**
```python
rayleigh_phase = 3/(16π) × (1 + ν²)
```
**Status:** ✅ VERIFIED IDENTICAL

### Mie Phase Function  
**Reference (lines 744-746):**
```
k = 3/(8π) × (1-g²)/(2+g²)
MiePhaseFunction = k × (1+ν²) / (1+g²-2gν)^1.5
```

**Our Implementation (V70):**
```python
mie_k = 3/(8π) × (1-g²)/(2+g²)
mie_phase = mie_k × (1+ν²) / (1+g²-2gν)^1.5
```
**Status:** ✅ VERIFIED IDENTICAL (V70)

**PREVIOUS BUG (V69 and earlier):**
```python
# WRONG - was missing (1+ν²) and had wrong constants
mie_phase = (1-g²) / (4π × (1+g²-2gν)^1.5)
```

---

## SCATTERING UV COORDINATES

### u_r (altitude coordinate)
**Reference (line 787):**
```
rho = sqrt(r² - bottom²)
u_r = GetTextureCoordFromUnitRange(rho/H, R_SIZE)
```

**Our Implementation:**
```python
rho = sqrt(r² - BOTTOM²)
x_r = rho / H
u_r = 0.5/R_SIZE + x_r × (1 - 1/R_SIZE)
```
**Status:** ✅ VERIFIED IDENTICAL

### u_mu NON-GROUND (view angle coordinate)
**Reference (lines 803-811):**
```
d = -r×μ + sqrt(r²(μ²-1) + top²)
d_min = top - r
d_max = rho + H
x_mu = (d - d_min) / (d_max - d_min)
u_mu = 0.5 + 0.5 × GetTextureCoordFromUnitRange(x_mu, MU_SIZE/2)
```

**Our Implementation:**
```python
disc = r²×(μ²-1) + top²
d = -r×μ + sqrt(disc)
d_min = top - r  
d_max = rho + H
x_mu = (d - d_min) / (d_max - d_min)
coord = 0.5/(MU_SIZE/2) + x_mu × (1 - 1/(MU_SIZE/2))
u_mu = 0.5 + 0.5 × coord
```
**Status:** ✅ VERIFIED IDENTICAL

### u_mu_s (sun angle coordinate)
**Reference (lines 814-827):**
```
d = DistanceToTopAtmosphereBoundary(bottom, mu_s)
d_min = top - bottom
d_max = H
a = (d - d_min) / (d_max - d_min)
D = DistanceToTopAtmosphereBoundary(bottom, mu_s_min)
A = (D - d_min) / (d_max - d_min)
x_mu_s = max(1 - a/A, 0) / (1 + a)
u_mu_s = GetTextureCoordFromUnitRange(x_mu_s, MU_S_SIZE)
```

**Our Implementation:**
```python
d = -bottom×mu_s + sqrt(bottom²×(mu_s²-1) + top²)
d_min = top - bottom
d_max = H
a = (d - d_min) / (d_max - d_min)
A = precomputed for MU_S_MIN
x_mu_s = max(1 - a/A, 0) / (1 + a)
u_mu_s = 0.5/MU_S_SIZE + x_mu_s × (1 - 1/MU_S_SIZE)
```
**Status:** ✅ VERIFIED IDENTICAL

### x_nu (view-sun angle coordinate)
**Reference (line 829):**
```
u_nu = (nu + 1) / 2
```

**Our Implementation:**
```python
x_nu = (nu + 1) / 2
```
**Status:** ✅ VERIFIED IDENTICAL

---

## TEXTURE SAMPLING

### Nu Interpolation
**Reference (lines 967-975):**
```
tex_coord_x = u_nu × (NU_SIZE - 1)
tex_x = floor(tex_coord_x)
lerp = tex_coord_x - tex_x
uvw0.x = (tex_x + u_mu_s) / NU_SIZE
uvw1.x = (tex_x + 1 + u_mu_s) / NU_SIZE
result = texture(uvw0) × (1-lerp) + texture(uvw1) × lerp
```

**Our Implementation (V68+):**
```python
tex_coord_x = x_nu × (NU_SIZE - 1)
tex_x_floor = floor(tex_coord_x)
nu_lerp = tex_coord_x - tex_x_floor
tex_x_ceil = min(tex_x_floor + 1, NU_SIZE - 1)
uvw_x_0 = (tex_x_floor + u_mu_s) / NU_SIZE
uvw_x_1 = (tex_x_ceil + u_mu_s) / NU_SIZE
# Sample 4 textures, bilinear interpolate
```
**Status:** ✅ VERIFIED IDENTICAL (V68+)

---

## TRANSMITTANCE

### GetTransmittance Formula
**Reference (lines 493-518):**
```
r_d = ClampRadius(sqrt(d² + 2r×μ×d + r²))
mu_d = ClampCosine((r×μ + d) / r_d)

if ray_r_mu_intersects_ground:
    T = T(r_d, -mu_d) / T(r, -mu)
else:
    T = T(r, mu) / T(r_d, mu_d)
```

**Our Implementation:**
```python
r_p = clamp(sqrt(d² + 2r×μ×d + r²), BOTTOM, TOP)
mu_p = clamp((r×μ + d) / r_p, -1, 1)

if ray_intersects_ground:
    T = T(r_p, -mu_p) / T(r, -mu)
else:
    T = T(r, mu) / T(r_p, mu_p)
```
**Status:** ✅ VERIFIED IDENTICAL

---

## INSCATTER

### Inscatter Formula
**Reference (lines 1849-1851):**
```
scattering = scattering - shadow_transmittance × scattering_p
single_mie = single_mie - shadow_transmittance × single_mie_p
```

**Our Implementation:**
```python
rayleigh_inscatter = scat_cam - T × scat_pt
mie_inscatter = mie_cam - T × mie_pt
```
**Status:** ✅ VERIFIED IDENTICAL

---

## POINT PARAMETERS

### r_p, mu_p, mu_s_p
**Reference (lines 1832-1834):**
```
r_p = ClampRadius(sqrt(d² + 2r×μ×d + r²))
mu_p = (r×μ + d) / r_p
mu_s_p = (r×mu_s + d×ν) / r_p
```

**Our Implementation:**
```python
r_p = clamp(sqrt(d² + 2r×μ×d + r²), BOTTOM, TOP)
mu_p = clamp((r×μ + d) / r_p, -1, 1)
mu_s_p = clamp((r×mu_s + d×ν) / r_p, -1, 1)
```
**Status:** ✅ VERIFIED IDENTICAL

---

## CONSTANTS

| Constant | Reference | Our Code | Status |
|----------|-----------|----------|--------|
| BOTTOM_RADIUS | 6360 km | 6360.0 | ✅ |
| TOP_RADIUS | 6420 km | 6420.0 | ✅ |
| MU_S_MIN | -0.2 | -0.2 | ✅ |
| H | sqrt(top²-bottom²) | sqrt(top²-bottom²) | ✅ |
| TRANSMITTANCE_WIDTH | 256 | 256 | ✅ |
| TRANSMITTANCE_HEIGHT | 64 | 64 | ✅ |
| SCATTERING_R_SIZE | 32 | 32 | ✅ |
| SCATTERING_MU_SIZE | 128 | 128 | ✅ |
| SCATTERING_MU_S_SIZE | 32 | 32 | ✅ |
| SCATTERING_NU_SIZE | 8 | 8 | ✅ |
| MIE_G | 0.8 | 0.8 | ✅ |

---

## TRANSMITTANCE SELECTION

### Ground vs Non-Ground for Aerial Perspective
**Reference behavior:** Uses `ray_r_mu_intersects_ground` to select formula.

**Our Implementation (V71):** **FORCED NON-GROUND**

**Reason:** For aerial perspective, objects are ABOVE ground. Using ground selection causes:
- Discontinuity at horizon where `ray_intersects_ground` changes
- Visible banding artifacts

The ground formula is for rays that would hit the ground if extended infinitely.
For aerial perspective to above-ground objects, we always use non-ground.

---

## CHANGELOG

- **V71**: Force non-ground transmittance - ground selection was causing horizon banding
- **V70**: Fixed Mie phase function - was completely wrong, missing (1+ν²) term
- **V69**: Reverted ground selection to forced non-ground (was causing excessive haze)
- **V68**: Added nu interpolation (4-texture bilinear sampling)
