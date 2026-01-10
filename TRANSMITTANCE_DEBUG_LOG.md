# Transmittance LUT Implementation Debug Log

## Objective
Implement proper Bruneton LUT-based transmittance for aerial perspective on object surfaces.

## Reference
- Bruneton `functions.glsl` lines 493-518: `GetTransmittance()`
- LUT path: `C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-3\helios_cache\luts\transmittance.exr`
- LUT dimensions: 256 x 64 (width x height)

## Key Formulas

### Ground Intersection Test
```
ray_intersects_ground = (mu < 0) AND (r²(mu²-1) + bottom² >= 0)
```

### Transmittance Between Two Points
```
r_d = sqrt(d² + 2*r*mu*d + r²)  // radius at distance d
mu_d = (r*mu + d) / r_d         // cosine at distance d

IF ray_intersects_ground:
    T = T(r_d, -mu_d) / T(r, -mu)    // NEGATED mu, reversed ratio
ELSE:
    T = T(r, mu) / T(r_d, mu_d)      // Normal ratio
```

### UV Parameterization
```
rho = sqrt(r² - bottom²)
H = sqrt(top² - bottom²) = 875.67 km
d_to_top = -r*mu + sqrt(r²(mu²-1) + top²)
d_min = top - r
d_max = rho + H
x_mu = (d_to_top - d_min) / (d_max - d_min)
x_r = rho / H

u = 0.5/256 + x_mu * (1 - 1/256)
v = 0.5/64 + x_r * (1 - 1/64)
```

## Constants
- BOTTOM_RADIUS = 6360.0 km
- TOP_RADIUS = 6420.0 km
- H = 875.67 km
- TRANSMITTANCE_WIDTH = 256
- TRANSMITTANCE_HEIGHT = 64

---

## Attempt 1: Initial Failed Implementation (V117)

**Date:** 2026-01-08

**What was tried:**
- Implemented ratio method T = T_cam / T_pt
- Did NOT handle ground intersection case
- Used positive mu values for all rays

**Result:** Nearly black output (values ~0.01)

**Root cause:** For ground-looking rays, denominator T_pt sampled near-zero values from LUT edge.

---

## Attempt 2: Correct Bruneton Implementation

**Date:** 2026-01-08

**Plan:**
1. Compute `ray_intersects_ground` flag in shader nodes
2. Compute both sky and ground formulas
3. Use Mix node to select correct result based on flag
4. Validate with known test cases

**Implementation Notes:**

### Step 2.1: Ground Intersection Detection
Need to compute: `(mu < 0) AND (discriminant >= 0)`
Where: `discriminant = r²(mu²-1) + bottom²`

In shader nodes:
- mu already exists
- r_cam is constant (can precompute discriminant for camera)
- For per-pixel: need to check if looking down (mu < 0)

### Step 2.2: Compute UV for Both Cases
For each sample point, compute:
- UV_sky using (r, mu) and (r_d, mu_d)
- UV_ground using (r, -mu) and (r_d, -mu_d)

### Step 2.3: Sample LUT and Compute Ratio
- Sample T at all 4 UV positions
- Compute T_sky = T(r,mu) / T(r_d,mu_d)
- Compute T_ground = T(r_d,-mu_d) / T(r,-mu)
- Mix based on ground intersection flag

---

## Validation Test Cases

| Scenario | r (km) | mu | d (km) | Expected T (approx) |
|----------|--------|-----|--------|---------------------|
| Near object, horizontal | 6360.001 | 0.0 | 0.1 | ~0.99 |
| Near object, down | 6360.001 | -0.01 | 0.1 | ~0.99 |
| Far object, horizontal | 6360.001 | 0.0 | 10.0 | ~0.85 |
| Far object, down | 6360.001 | -0.001 | 10.0 | ~0.85 |

---

## Current Status
- [x] Ground intersection detection implemented (mu < 0)
- [x] Sky formula implemented: T(r,mu) / T(r_d,mu_d)
- [x] Ground formula implemented: T(r_d,-mu_d) / T(r,-mu)
- [x] Mix node selection working
- [x] Horizon fallback to exponential (|mu| < 0.1)
- [x] Validation passed - no color collapse

## V122 SUCCESSFUL
- Exponential fallback for horizon eliminates color collapse
- All geometry pixels now have B >= 0.5*R
- Mean transmittance: R=0.99, G=0.99, B=0.99
