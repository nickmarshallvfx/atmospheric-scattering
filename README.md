# Helios - Atmospheric Scattering for Blender

A Blender 5.0 addon implementing Eric Bruneton's Precomputed Atmospheric Scattering system for film VFX pipelines.

## Features

- **Physically-based** atmospheric scattering using precomputed lookup tables
- **Artistic controls** matching the standalone Bruneton implementation
- **AOV output** (Sky, Transmittance, Inscattering) for Nuke compositing
- **Multi-layer EXR** export with proper channel naming
- **Planet presets** (Earth, Mars, Titan)
- **Z-up coordinate system** native to Blender

## Installation

1. Download or clone this repository
2. In Blender: Edit → Preferences → Add-ons → Install
3. Select the `helios` folder
4. Enable "Helios Atmospheric Scattering"

## Quick Start

1. Open **Properties → World → Helios Atmosphere**
2. Adjust sun position and atmosphere parameters
3. Click **Precompute Atmosphere** (takes 10-30 seconds)
4. The sky will render with physically-accurate scattering

## Usage

### UI Panels

Located in **Properties → World → Helios Atmosphere**:

- **Sun & Environment**: Sun zenith, azimuth, intensity
- **Atmosphere Composition**: Rayleigh/Mie density, scale heights, phase function
- **Rendering**: Exposure, white balance, luminance mode
- **Planet**: Radius, atmosphere height, presets

### Artistic Controls

| Parameter | Effect |
|-----------|--------|
| Rayleigh Density | Sky blue intensity (higher = bluer sky) |
| Mie Density | Haze/fog amount (higher = hazier) |
| Mie Phase G | Sun glare directionality (-1 to 1) |
| Scale Heights | Altitude falloff of scattering |
| Ground Albedo | Ground light reflection |

### AOV Channels

For Nuke compositing, export multi-layer EXR with channels:
- `helios.sky.R/G/B` - Sky radiance
- `helios.transmittance.R/G/B` - Atmospheric transmittance
- `helios.inscatter.R/G/B` - Inscattered light

## Requirements

- Blender 5.0+
- NumPy (bundled with Blender)
- OpenEXR (optional, for multi-layer export)

## Project Structure

```
helios/
├── __init__.py           # Addon registration
├── core/                 # Atmospheric model
│   ├── constants.py      # Physical constants
│   ├── parameters.py     # Atmosphere parameters
│   └── model.py          # LUT precomputation
├── shaders/              # GLSL/OSL shaders
│   ├── definitions.glsl  # Type definitions
│   ├── functions.glsl    # Scattering functions
│   └── atmosphere.osl    # Cycles shader
├── nodes/                # Custom Blender nodes
├── operators/            # Blender operators
├── panels/               # UI panels
└── utils/                # GPU and EXR utilities
```

## Technical Details

### LUT Textures

The Bruneton model precomputes 4 lookup tables:
- **Transmittance** (256×64): Optical depth
- **Scattering** (256×128×32): Single + multiple scattering
- **Irradiance** (64×16): Ground illumination

### Coordinate System

- Blender uses Z-up (handled natively)
- Nuke uses Y-up (coordinate transform on export)

## Credits

- **Eric Bruneton** - Original atmospheric scattering implementation
- **Fabrice Neyret** - Co-author of the research paper
- **Sebastien Hillaire** - Unreal Engine adaptations

## License

BSD License (matching original Bruneton implementation)

## References

- [Precomputed Atmospheric Scattering](https://hal.inria.fr/inria-00288758/en) (Bruneton & Neyret, 2008)
- [A Scalable and Production Ready Sky and Atmosphere Rendering Technique](https://sebh.github.io/publications/egsr2020.pdf) (Hillaire, 2020)
