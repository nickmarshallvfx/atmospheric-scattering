# Test Scene Geometry Reference

**Generated**: 2026-01-07 07:55:50
**Source EXR**: `base_render_reference_noAtmos_v001_01.exr`
**Resolution**: 960x540

---

## Camera

| Parameter | Value |
|-----------|-------|
| Position (actual) | (37.069, -44.786, 6.0) meters |
| Altitude above ground | ~6 meters |
| Rotation | (100°, 0°, 46.692°) |

---

## Scene Bounds

| Axis | Min | Max | Range |
|------|-----|-----|-------|
| X | -2894.9m | 50.7m | 2945.6m |
| Y | -88.8m | 10235.1m | 10323.9m |
| Z | -1.4m | 418.8m | 420.1m |

---

## Distance from Camera

| Statistic | Value | Value (km) |
|-----------|-------|------------|
| Minimum | 160.9m | 0.161km |
| 5th percentile | 181.5m | 0.182km |
| 25th percentile | 206.2m | 0.206km |
| Median | 220.3m | 0.220km |
| Mean | 326.7m | 0.327km |
| 75th percentile | 225.8m | 0.226km |
| 95th percentile | 682.9m | 0.683km |
| Maximum | 10644.6m | 10.645km |

---

## Coverage

| Type | Pixels | Percentage |
|------|--------|------------|
| Geometry | 381,631 | 73.6% |
| Sky | 136,769 | 26.4% |

---

## Expected Aerial Perspective Behavior

Based on distance statistics:

### Near Objects (< 206m)
- **Transmittance**: ~0.99+ (almost no absorption)
- **Inscatter**: Minimal
- **Expected appearance**: Nearly original color

### Mid-range Objects (206m - 226m)
- **Transmittance**: ~0.95-0.99
- **Inscatter**: Moderate
- **Expected appearance**: Slight haze, reduced contrast

### Far Objects (> 226m)
- **Transmittance**: ~0.85-0.95
- **Inscatter**: Significant
- **Expected appearance**: Visible haze, blending toward sky color

### Very Far Objects (> 683m)
- **Transmittance**: < 0.85
- **Inscatter**: Heavy
- **Expected appearance**: Strong haze, significantly blended with sky

---

## Validation Reference

When validating Step 1.1 (Distance), output `distance / 10.6km`:
- Near geometry should appear **dark** (low distance value)
- Far geometry should appear **bright** (high distance value)
- Sky pixels should be **black** (filtered out or clamped)

