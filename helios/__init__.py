"""
Helios - Physically-based Atmospheric Scattering for Blender

A Blender 5.0 addon implementing Eric Bruneton's Precomputed Atmospheric Scattering
for film VFX pipelines. Outputs multi-layer EXR with AOVs for Nuke compositing.

Copyright (c) 2024 MattePaint
Based on work by Eric Bruneton (BSD License)
"""

bl_info = {
    "name": "Helios Atmospheric Scattering",
    "author": "MattePaint",
    "version": (1, 0, 0),
    "blender": (5, 0, 0),
    "location": "Properties > World > Helios Atmosphere",
    "description": "Physically-based atmospheric scattering with AOV output for VFX",
    "category": "Render",
}

import bpy
from bpy.props import (
    FloatProperty,
    FloatVectorProperty,
    BoolProperty,
    IntProperty,
    EnumProperty,
    PointerProperty,
)
from bpy.types import PropertyGroup

# Import submodules
from . import core
from . import operators
from . import panels
from . import nodes
from . import world


# Update callback for real-time preview
def _update_preview(self, context):
    """Update preview when parameter changes."""
    from . import world
    world.update_atmosphere_world(context)


# Update callback that triggers LUT recomputation
def _update_preview_invalidate(self, context):
    """Update preview and trigger LUT recomputation for parameters baked into LUTs."""
    from . import core
    from . import world as world_module
    
    scene = context.scene
    settings = scene.helios
    
    # Mark LUTs as invalid
    settings.luts_valid = False
    
    # Auto-recompute LUTs (this is fast with vectorized code)
    try:
        # Get LUT cache directory
        lut_dir = world_module.get_lut_cache_dir()
        
        # Create atmosphere parameters from current settings
        params = core.parameters.AtmosphereParameters.from_blender_settings(settings)
        
        # Precompute LUTs
        print("Helios: Auto-recomputing LUTs...")
        model = core.model.AtmosphereModel(params)
        model.init()
        
        # Save LUTs as EXR files
        import os
        os.makedirs(lut_dir, exist_ok=True)
        model.save_textures_exr(lut_dir)
        
        settings.luts_valid = True
        print("Helios: LUT recomputation complete")
        
        # Force shader rebuild to refresh texture cache
        world = context.scene.world
        if world and world.use_nodes and world.get("is_helios"):
            # Clear nodes and rebuild to force texture reload
            world.node_tree.nodes.clear()
            world_module._build_sky_nodes(
                world.node_tree.nodes,
                world.node_tree.links,
                settings
            )
            world_module._force_viewport_update(context, world)
            print("Helios: Shader rebuilt with fresh textures")
        
    except Exception as e:
        print(f"Helios: Auto-recomputation failed: {e}")
        import traceback
        traceback.print_exc()
        # Fall back to just updating preview with stale LUTs
    
    # Update the world shader
    _update_preview(self, context)


class HeliosAtmosphereSettings(PropertyGroup):
    """Main property group for Helios atmosphere settings."""
    
    # Sun/Environment (preview-only, no rebake needed)
    sun_elevation: FloatProperty(
        name="Sun Elevation",
        description="Sun elevation angle in degrees (0 = horizon, 90 = overhead, negative = below horizon)",
        default=45.0,
        min=-10.0,
        max=90.0,
        update=_update_preview,
    )
    sun_heading: FloatProperty(
        name="Sun Heading",
        description="Sun heading/azimuth in degrees (0 = North, 90 = East, 180 = South, 270 = West)",
        default=180.0,
        min=0.0,
        max=360.0,
        update=_update_preview,
    )
    sun_intensity: FloatProperty(
        name="Sun Intensity",
        description="Sun intensity multiplier",
        default=1.0,
        min=0.0,
        max=10.0,
        update=_update_preview,
    )
    
    # Rendering (preview-only)
    exposure: FloatProperty(
        name="Exposure",
        description="Exposure adjustment",
        default=10.0,
        min=0.01,
        max=100.0,
        update=_update_preview,
    )
    white_balance: BoolProperty(
        name="White Balance",
        description="Apply white balance correction",
        default=True,
    )
    use_ozone: BoolProperty(
        name="Use Ozone",
        description="Include ozone absorption layer",
        default=True,
    )
    luminance_mode: EnumProperty(
        name="Luminance Mode",
        description="How to compute luminance values",
        items=[
            ('NONE', "Radiance", "Use spectral radiance directly"),
            ('APPROXIMATE', "Approximate", "Approximate luminance conversion"),
            ('PRECOMPUTED', "Precomputed", "Full spectral precomputation"),
        ],
        default='PRECOMPUTED',
    )
    render_mode: EnumProperty(
        name="Render Mode",
        description="Camera projection mode",
        items=[
            ('PERSPECTIVE', "Perspective", "Standard perspective projection"),
            ('LATLONG', "Lat-Long", "Equirectangular panorama"),
        ],
        default='PERSPECTIVE',
    )
    
    # Atmospheric Composition (Artistic Controls)
    # Note: mie_phase_g is runtime-only, doesn't need LUT rebuild
    mie_phase_g: FloatProperty(
        name="Mie Phase G",
        description="Mie scattering directionality (-1 to 1). Higher = more forward scattering",
        default=0.8,
        min=-0.999,
        max=0.999,
        update=_update_preview,  # Runtime parameter, no recompute needed
    )
    mie_density: FloatProperty(
        name="Mie Density",
        description="Aerosol density multiplier (affects haze/fog)",
        default=1.0,
        min=0.0,
        max=10.0,
        update=_update_preview_invalidate,
    )
    rayleigh_density: FloatProperty(
        name="Rayleigh Density",
        description="Air molecule density multiplier (affects sky blue color)",
        default=1.0,
        min=0.0,
        max=10.0,
        update=_update_preview_invalidate,
    )
    mie_scale_height: FloatProperty(
        name="Mie Scale Height",
        description="Altitude falloff for aerosols in meters",
        default=1200.0,
        min=100.0,
        max=10000.0,
        update=_update_preview_invalidate,
    )
    rayleigh_scale_height: FloatProperty(
        name="Rayleigh Scale Height",
        description="Altitude falloff for air molecules in meters",
        default=8000.0,
        min=1000.0,
        max=20000.0,
        update=_update_preview_invalidate,
    )
    ground_albedo: FloatProperty(
        name="Ground Albedo",
        description="Ground reflectivity (0-1)",
        default=0.1,
        min=0.0,
        max=1.0,
        update=_update_preview_invalidate,
    )
    
    # Planet Parameters
    planet_radius: FloatProperty(
        name="Planet Radius",
        description="Planet radius in kilometers",
        default=6360.0,
        min=100.0,
        max=100000.0,
    )
    atmosphere_height: FloatProperty(
        name="Atmosphere Height",
        description="Atmosphere thickness in kilometers",
        default=60.0,
        min=1.0,
        max=1000.0,
    )
    
    # LUT Status
    luts_valid: BoolProperty(
        name="LUTs Valid",
        description="Whether precomputed LUTs are current",
        default=False,
    )
    
    # GPU Acceleration
    use_gpu: BoolProperty(
        name="Use GPU",
        description="Use GPU acceleration (CuPy) for precomputation if available",
        default=True,
    )


classes = (
    HeliosAtmosphereSettings,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.helios = PointerProperty(type=HeliosAtmosphereSettings)
    
    operators.register()
    panels.register()
    nodes.register()
    world.register()


def unregister():
    world.unregister()
    nodes.unregister()
    panels.unregister()
    operators.unregister()
    
    del bpy.types.Scene.helios
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
