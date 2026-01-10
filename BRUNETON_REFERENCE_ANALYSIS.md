# Bruneton GetSkyRadianceToPoint - Line-by-Line Analysis

## Purpose
Complete analysis of the reference implementation to ensure 100% accurate port.

---

## Function Signature (lines 1787-1793)

```glsl
RadianceSpectrum GetSkyRadianceToPoint(
    IN(AtmosphereParameters) atmosphere,
    IN(TransmittanceTexture) transmittance_texture,
    IN(ReducedScatteringTexture) scattering_texture,
    IN(ReducedScatteringTexture) single_mie_scattering_texture,
    Position camera, IN(Position) point, Length shadow_length,
    IN(Direction) sun_direction, OUT(DimensionlessSpectrum) transmittance)
```

**Inputs:**
- `camera`: 3D position of camera (relative to planet center, in km)
- `point`: 3D position of surface point (relative to planet center, in km)
- `shadow_length`: For light shafts (we use 0)
- `sun_direction`: Normalized sun direction vector

**Outputs:**
- `transmittance`: Transmittance between camera and point (OUT parameter)
- Return value: Inscattered radiance

---

## Step 1: Compute View Ray (lines 1797-1808)

```glsl
Direction view_ray = normalize(point - camera);
Length r = length(camera);
Length rmu = dot(camera, view_ray);
Length distance_to_top_atmosphere_boundary = -rmu -
    sqrt(rmu * rmu - r * r + atmosphere.top_radius * atmosphere.top_radius);
if (distance_to_top_atmosphere_boundary > 0.0 * m) {
    camera = camera + view_ray * distance_to_top_atmosphere_boundary;
    r = atmosphere.top_radius;
    rmu += distance_to_top_atmosphere_boundary;
}
```

**What it does:**
1. `view_ray` = normalized direction from camera to point
2. `r` = distance from planet center to camera
3. `rmu` = r × μ (where μ = cos(view zenith angle))
4. If camera is in space, move it to atmosphere boundary

**For our case:** Camera is always inside atmosphere, so the `if` block doesn't execute.

---

## Step 2: Compute Camera Parameters (lines 1811-1815)

```glsl
Number mu = rmu / r;
Number mu_s = dot(camera, sun_direction) / r;
Number nu = dot(view_ray, sun_direction);
Length d = length(point - camera);
bool ray_r_mu_intersects_ground = RayIntersectsGround(atmosphere, r, mu);
```

**Parameters:**
- `mu` = cos(view zenith angle) = dot(camera_normalized, view_ray)
- `mu_s` = cos(sun zenith angle) = dot(camera_normalized, sun_direction)
- `nu` = cos(view-sun angle) = dot(view_ray, sun_direction)
- `d` = distance from camera to point
- `ray_r_mu_intersects_ground` = does view ray hit planet sphere?

**CRITICAL:** `ray_r_mu_intersects_ground` is computed from (r, mu) only - it's about whether the MATHEMATICAL RAY intersects the planet sphere, NOT about the actual geometry.

---

## Step 3: RayIntersectsGround (lines 240-246)

```glsl
bool RayIntersectsGround(IN(AtmosphereParameters) atmosphere,
    Length r, Number mu) {
  return mu < 0.0 && r * r * (mu * mu - 1.0) +
      atmosphere.bottom_radius * atmosphere.bottom_radius >= 0.0 * m2;
}
```

**Logic:**
- Only possible if `mu < 0` (looking downward)
- AND discriminant >= 0 (ray actually reaches planet surface)

**For μ = mu_horizon:** discriminant = 0 exactly (tangent to surface)
**For μ < mu_horizon:** discriminant > 0 (intersects ground)
**For μ > mu_horizon:** discriminant < 0 (misses ground)

---

## Step 4: Get Transmittance (lines 1817-1818)

```glsl
transmittance = GetTransmittance(atmosphere, transmittance_texture,
    r, mu, d, ray_r_mu_intersects_ground);
```

This calls `GetTransmittance` (lines 493-519) which computes transmittance between camera and point.

---

## Step 5: GetTransmittance Details (lines 493-519)

```glsl
Length r_d = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
Number mu_d = ClampCosine((r * mu + d) / r_d);

if (ray_r_mu_intersects_ground) {
    return min(
        GetTransmittanceToTopAtmosphereBoundary(r_d, -mu_d) /
        GetTransmittanceToTopAtmosphereBoundary(r, -mu),
        DimensionlessSpectrum(1.0));
} else {
    return min(
        GetTransmittanceToTopAtmosphereBoundary(r, mu) /
        GetTransmittanceToTopAtmosphereBoundary(r_d, mu_d),
        DimensionlessSpectrum(1.0));
}
```

**Key insight:** 
- `r_d` and `mu_d` are computed via **law of cosines** from camera parameters
- For ground rays: use NEGATED mu values, SWAP numerator/denominator
- For non-ground rays: normal formula

**Why the swap for ground rays?**
- Ground rays go DOWN then UP to reach top atmosphere
- Must compute path through atmosphere correctly
- Using -mu converts downward ray to equivalent upward ray

---

## Step 6: Get Camera Scattering (lines 1820-1824)

```glsl
IrradianceSpectrum single_mie_scattering;
IrradianceSpectrum scattering = GetCombinedScattering(
    atmosphere, scattering_texture, single_mie_scattering_texture,
    r, mu, mu_s, nu, ray_r_mu_intersects_ground,
    single_mie_scattering);
```

**Note:** Uses SAME `ray_r_mu_intersects_ground` flag as transmittance.

---

## Step 7: Compute Point Parameters (lines 1831-1834)

```glsl
d = max(d - shadow_length, 0.0 * m);  // We use shadow_length=0
Length r_p = ClampRadius(atmosphere, sqrt(d * d + 2.0 * r * mu * d + r * r));
Number mu_p = (r * mu + d) / r_p;
Number mu_s_p = (r * mu_s + d * nu) / r_p;
```

**CRITICAL:** Point parameters are computed via **law of cosines**, NOT from actual point position!
- `r_p` = radius at point along the ray
- `mu_p` = view zenith cosine at point
- `mu_s_p` = sun zenith cosine at point

This ensures consistency with the inscatter formula derivation.

---

## Step 8: Get Point Scattering (lines 1836-1840)

```glsl
IrradianceSpectrum single_mie_scattering_p;
IrradianceSpectrum scattering_p = GetCombinedScattering(
    atmosphere, scattering_texture, single_mie_scattering_texture,
    r_p, mu_p, mu_s_p, nu, ray_r_mu_intersects_ground,
    single_mie_scattering_p);
```

**CRITICAL:** Uses SAME `ray_r_mu_intersects_ground` flag as camera scattering!

---

## Step 9: Combine Results (lines 1842-1851)

```glsl
DimensionlessSpectrum shadow_transmittance = transmittance;
if (shadow_length > 0.0 * m) {
    shadow_transmittance = GetTransmittance(atmosphere, transmittance_texture,
        r, mu, d, ray_r_mu_intersects_ground);
}
scattering = scattering - shadow_transmittance * scattering_p;
single_mie_scattering =
    single_mie_scattering - shadow_transmittance * single_mie_scattering_p;
```

**The inscatter formula:**
```
inscatter = S_camera - transmittance × S_point
```

For shadow_length=0, this simplifies to:
```
inscatter = S_cam - T × S_pt
```

---

## Step 10: Apply Phase Functions (lines 1857-1862)

```glsl
single_mie_scattering = single_mie_scattering *
    smoothstep(Number(0.0), Number(0.01), mu_s);

return scattering * RayleighPhaseFunction(nu) + single_mie_scattering *
    MiePhaseFunction(atmosphere.mie_phase_function_g, nu);
```

**Notes:**
- Mie scattering is faded out when sun is below horizon (mu_s < 0)
- Phase functions are applied to final result

---

## Summary of Critical Requirements

1. **ray_r_mu_intersects_ground** is computed from (r, mu) ONLY
2. **Same flag** used for BOTH transmittance AND scattering
3. **Law of cosines** for all point parameters (r_p, mu_p, mu_s_p)
4. **Ground transmittance**: uses -mu, -mu_d, swapped division
5. **Ground scattering**: uses different UV formula (u_mu in [0, 0.5])
6. **Non-ground scattering**: uses u_mu in [0.5, 1.0]
7. **No clamping of mu** to horizon - let the model work as designed

---

## Implementation Checklist

- [ ] Camera position relative to planet center (add BOTTOM_RADIUS to Z)
- [ ] view_ray = normalize(point - camera)
- [ ] r = length(camera)
- [ ] mu = dot(camera, view_ray) / r
- [ ] mu_s = dot(camera, sun_direction) / r
- [ ] nu = dot(view_ray, sun_direction)
- [ ] d = length(point - camera)
- [ ] ray_r_mu_intersects_ground = (mu < 0) AND (r²(μ²-1) + bottom² >= 0)
- [ ] Transmittance with correct ground/non-ground formula
- [ ] Scattering with correct ground/non-ground UV
- [ ] r_p, mu_p, mu_s_p via law of cosines
- [ ] Point scattering with SAME ray_r_mu_intersects_ground flag
- [ ] inscatter = S_cam - T × S_pt
- [ ] Phase functions applied
