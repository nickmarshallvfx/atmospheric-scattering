# Bruneton Implementation TODO

Tracking file for features from the Eric Bruneton Atmospheric Scattering model that need implementation or refinement.

---

## ✅ Recently Implemented

### Multiple Scattering (Orders 2+)
- **File:** `helios/core/model.py`
- **Methods:**
  - `_precompute_scattering_density()` - Full spherical integration over incident directions
  - `_precompute_indirect_irradiance()` - Hemisphere integration for ground irradiance
  - `_precompute_multiple_scattering()` - Ray march integration with trapezoidal rule
- **Status:** Implemented based on Bruneton reference. Needs testing.
- **Note:** Precomputation will be slower (nested loops over 4D texture + spherical sampling)

---

## ❌ Not Implemented (Placeholders)

### Aerial Perspective for Scene Objects
- **File:** Not yet created
- **Description:** Objects in the scene should have atmospheric fog/haze applied based on distance. Requires:
  - `GetSkyRadianceToPoint()` - radiance between camera and a point
  - `GetSunAndSkyIrradiance()` - ground illumination
- **Impact:** Scene objects won't fade into atmosphere with distance
- **Reference:** `functions.glsl` `GetSkyRadianceToPoint`, `GetSunAndSkyIrradiance`

### Ground Radiance
- **File:** `helios/shaders/bruneton_sky.osl`
- **Description:** `GetGroundRadiance()` function for rendering ground when looking down
- **Impact:** Ground plane won't be lit correctly when visible from altitude
- **Reference:** `functions.glsl` `GetSolarRadiance`, `GetGroundRadiance`

---

## ⚠️ Implemented but Needs Verification

### Single Scattering Precomputation
- **File:** `helios/core/model.py`
- **Method:** `_precompute_single_scattering()` 
- **Status:** Just implemented, needs visual verification
- **Concern:** Coordinate mapping between Python precomputation and OSL shader must match exactly

### Transmittance Precomputation
- **File:** `helios/core/model.py`
- **Method:** `_precompute_transmittance()`
- **Status:** Implemented with numerical integration
- **Concern:** Need to verify UV mapping matches Bruneton reference

### OSL Tiled 2D Texture Sampling
- **File:** `helios/shaders/bruneton_sky.osl`
- **Function:** `GetScatteringTiled2D()`
- **Status:** Implemented as workaround for OSL lacking 3D texture support
- **Concern:** Interpolation between depth slices may have edge artifacts

---

## 🔧 Known Issues / Incomplete

### Sun Direction Controls
- **File:** `helios/world.py`
- **Status:** ✅ FIXED - Sun heading/elevation now working correctly

### Sun Disk Rendering
- **File:** `helios/shaders/bruneton_sky.osl`
- **Status:** Code exists but untested
- **Concern:** May need limb darkening adjustment

### Exposure / Tone Mapping
- **File:** `helios/shaders/bruneton_sky.osl`
- **Status:** Basic exposure multiplier exists
- **Concern:** Reference uses specific tone mapping that may not be replicated

---

## ✅ Implemented

- Transmittance LUT precomputation
- Direct irradiance precomputation  
- Single scattering precomputation (vectorized)
- OSL shader structure with texture lookups
- Rayleigh and Mie phase functions
- EXR export using Blender's image API
- Tiled 2D representation of 3D textures
- Basic sun disk rendering code
- Sun direction controls (heading/elevation)
- Sun intensity multiplier
- Exposure control
- Auto-enable OSL in Cycles
- Development symlink workflow

---

## 🚀 Future Performance Improvements

### GPU-Based Precomputation
- **Description:** Use Blender compute shaders to recompute LUTs on GPU in real-time
- **Benefit:** Would allow instant parameter updates (density, scale height) like the web demo
- **Complexity:** High - requires porting numerical integration to GPU compute shaders
- **Priority:** Nice-to-have after core features complete

---

## Reference Files

- **Original GLSL:** `reference/atmospheric-scattering-2/src/atmosphere/shaders/functions.glsl`
- **Original C++ Model:** `reference/atmospheric-scattering-2/src/atmosphere/model.cc`
- **OSL Port:** `helios/shaders/bruneton_sky.osl`
- **Python Model:** `helios/core/model.py`

---

*Last Updated: 2025-12-28*
