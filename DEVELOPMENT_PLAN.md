# Helios - Atmospheric Scattering for Blender

## Overview

Helios is a Blender 5.0 addon that ports the Eric Bruneton Precomputed Atmospheric Scattering system for use in film VFX pipelines. It prioritizes physical accuracy and quality over real-time performance while retaining the efficient precomputed LUT approach.

## Reference Implementation

The working standalone implementation is located at:
- `atmospheric-scattering-2/` (DO NOT MODIFY)

### Key Components to Port

| Component | Source File | Purpose |
|-----------|-------------|---------|
| Atmosphere Model | `atmosphere/model.h`, `model.cc` | C++ API, LUT precomputation |
| GLSL Functions | `atmosphere/functions.glsl` | Core scattering computations |
| GLSL Definitions | `atmosphere/definitions.glsl` | Types and constants |
| Demo Shader | `atmosphere/demo/demo.glsl` | Scene rendering with AOVs |
| Renderer | `atmosphere/demo/atmospheric_renderer.h` | Parameter structure |
| GUI Controls | `atmosphere/demo/gui_prototype.h` | Creative controls interface |

### Creative Controls (from GUI Prototype)

**Camera Parameters:**
- Position (X, Y, Z) in meters
- Rotation (Pitch, Yaw, Roll) in degrees
- Field of View

**Environment Parameters:**
- Sun Zenith angle
- Sun Azimuth angle
- Sun Intensity

**Atmospheric Composition (Artistic Controls):**
- `mie_phase_g` (-1 to 1): Scattering directionality
- `mie_density`: Aerosol density multiplier
- `rayleigh_density`: Molecular scattering multiplier
- `mie_height`: Altitude falloff for aerosols (meters)
- `rayleigh_height`: Altitude falloff for air (meters)
- `ground_albedo`: Ground reflectivity (0-1)

**Rendering Options:**
- Exposure
- Use Ozone layer
- Luminance mode (None, Approximate, Precomputed)
- White Balance
- Render mode (Perspective, Latlong)

## Architecture Design

### Blender Integration Strategy

**Option A: Custom Render Engine** (Complex but full control)
- Register as `bpy.types.RenderEngine`
- Full control over render passes and AOVs
- Requires implementing entire render pipeline

**Option B: Compositor Nodes** (Recommended)
- Create custom compositor nodes
- Works with any render engine (Cycles, EEVEE)
- Processes depth/normal passes to add atmosphere
- More flexible for VFX pipelines

**Option C: World Shader Nodes** (Limited)
- Custom shader nodes for background
- Limited to sky only, no object interaction

### Recommended Approach: Hybrid Compositor + World

1. **World Environment**: Generate sky dome texture using precomputed LUTs
2. **Compositor Nodes**: Apply transmittance and inscattering to rendered layers
3. **AOV Outputs**: Export separate passes for Nuke

### Directory Structure

```
atmospheric-scattering-4/
├── helios/                      # Blender addon package
│   ├── __init__.py              # Addon registration
│   ├── core/                    # Core atmospheric model
│   │   ├── __init__.py
│   │   ├── constants.py         # Physical constants
│   │   ├── model.py             # Atmosphere model (LUT generation)
│   │   ├── precompute.py        # LUT precomputation
│   │   └── parameters.py        # Atmosphere parameters dataclass
│   ├── shaders/                 # GLSL/OSL shaders
│   │   ├── functions.glsl       # Ported scattering functions
│   │   ├── definitions.glsl     # Types and constants
│   │   └── atmosphere.osl       # OSL version for Cycles
│   ├── nodes/                   # Custom Blender nodes
│   │   ├── __init__.py
│   │   ├── sky_node.py          # Sky radiance node
│   │   ├── transmittance_node.py
│   │   └── inscatter_node.py
│   ├── operators/               # Blender operators
│   │   ├── __init__.py
│   │   ├── precompute.py        # Precompute LUTs operator
│   │   └── export.py            # Export EXR operator
│   ├── panels/                  # UI panels
│   │   ├── __init__.py
│   │   └── atmosphere_panel.py  # Main control panel
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── gpu.py               # GPU texture utilities
│       └── exr.py               # EXR export utilities
├── tests/                       # Validation tests
│   ├── compare_reference.py     # Compare with standalone
│   └── test_luts.py             # LUT generation tests
├── presets/                     # Atmosphere presets
│   ├── earth.json
│   ├── mars.json
│   └── sunset.json
└── docs/
    └── user_guide.md
```

## Implementation Phases

### Phase 1: Core Model Port (Python)
1. Port `constants.h` → `core/constants.py`
2. Port `DensityProfileLayer` and parameter structures
3. Port LUT precomputation from `model.cc`
4. Validate LUT output matches reference

### Phase 2: Shader Port
1. Port `definitions.glsl` → `shaders/definitions.glsl`
2. Port `functions.glsl` → `shaders/functions.glsl`
3. Create OSL version for Cycles compatibility
4. Test shader functions against reference

### Phase 3: Blender Integration
1. Create addon structure with proper registration
2. Implement UI panels matching GUI prototype
3. Create custom nodes for compositor workflow
4. Implement world shader for sky background

### Phase 4: AOV System
1. Implement separate render passes:
   - `Sky`: Background sky radiance
   - `Transmittance`: Per-pixel atmospheric transmittance
   - `Inscattering`: Atmospheric inscattering contribution
2. Multi-layer EXR export with proper channel naming

### Phase 5: Validation & Polish
1. Visual comparison with reference implementation
2. Performance optimization
3. User documentation
4. Presets for common scenarios

## Technical Notes

### Coordinate Systems
- **Blender**: Z-up, right-handed
- **Bruneton Reference**: Y-up converted in demo.glsl
- **Nuke**: Y-up

The port must handle coordinate system conversion in the shader.

### LUT Textures
The Bruneton model precomputes 4 lookup textures:
1. **Transmittance** (2D): 256×64, optical depth lookup
2. **Scattering** (4D → 3D): 256×128×32, single + multiple scattering
3. **Irradiance** (2D): 64×16, ground irradiance
4. **Optional Single Mie** (4D → 3D): If not combined with scattering

### GPU Considerations
- Blender 5.0 uses Vulkan/Metal, need GPU texture upload
- Use `gpu` module for texture management
- Consider baking LUTs to image files for persistence

## AOV Channel Naming Convention (for Nuke)

```
helios.sky.R, helios.sky.G, helios.sky.B
helios.transmittance.R, helios.transmittance.G, helios.transmittance.B
helios.inscatter.R, helios.inscatter.G, helios.inscatter.B
```

## Dependencies

- Blender 5.0 (Python 3.11+)
- NumPy (bundled with Blender)
- OpenEXR (for multi-layer export)

## Next Steps

1. Create the addon package structure
2. Port physical constants
3. Implement `DensityProfileLayer` and atmosphere parameters
4. Begin LUT precomputation port
