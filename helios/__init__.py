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


# Debounce timer for LUT recomputation
_debounce_timer = None

# Cache of previous LUT-affecting parameter values to detect actual changes
_last_lut_params = {}

def _get_lut_param_hash(settings):
    """Get a hashable tuple of all LUT-affecting parameters."""
    return (
        round(settings.rayleigh_density, 6),
        round(settings.mie_density, 6),
        round(settings.rayleigh_scale_height, 1),
        round(settings.mie_scale_height, 1),
        round(settings.ground_albedo, 4),
        settings.use_ozone,
        round(getattr(settings, 'ozone_density', 1.0), 4),
        round(getattr(settings, 'mie_angstrom_beta', 0.04), 6),
    )

def _do_recompute_luts(context):
    """Actually perform the LUT recomputation (called after debounce delay).
    
    Uses the operator for proper GPU context (timer callbacks have degraded GPU performance).
    The operator handles clearing/rebuilding the sky shader for faster baking.
    """
    try:
        # Call the precompute operator - this has proper GPU context
        # The operator clears sky shader before baking and rebuilds after
        print("Helios: Auto-recomputing LUTs via operator...")
        bpy.ops.helios.precompute_luts('EXEC_DEFAULT')
        
    except Exception as e:
        print(f"Helios: Auto-recomputation failed: {e}")
        import traceback
        traceback.print_exc()
    
    return None  # Don't repeat timer


# Update callback that triggers LUT recomputation
def _update_preview_invalidate(self, context):
    """Update preview and trigger LUT recomputation for parameters baked into LUTs."""
    global _debounce_timer, _last_lut_params
    import bpy
    
    scene = context.scene
    settings = scene.helios
    
    # Check if values actually changed
    current_hash = _get_lut_param_hash(settings)
    scene_key = scene.name  # Use name, not id() which can change
    
    print(f"Helios: DEBUG scene_key={scene_key}")
    print(f"Helios: DEBUG current_hash={current_hash}")
    print(f"Helios: DEBUG cached_hash={_last_lut_params.get(scene_key, 'NOT CACHED')}")
    
    if scene_key in _last_lut_params and _last_lut_params[scene_key] == current_hash:
        # Values haven't changed, skip rebake
        print(f"Helios: Values unchanged, skipping rebake")
        return
    
    print(f"Helios: Values changed, scheduling rebake")
    # Update cached values
    _last_lut_params[scene_key] = current_hash
    
    # Mark LUTs as invalid
    settings.luts_valid = False
    
    # Cancel any pending recompute
    if _debounce_timer is not None:
        try:
            bpy.app.timers.unregister(_debounce_timer)
        except:
            pass
        _debounce_timer = None
    
    # Schedule recompute after debounce delay (0.5 seconds)
    # This prevents multiple rebakes when dragging sliders
    def debounced_recompute():
        global _debounce_timer
        _debounce_timer = None
        _do_recompute_luts(context)
        return None
    
    _debounce_timer = debounced_recompute
    bpy.app.timers.register(debounced_recompute, first_interval=0.5)
    
    # Update the world shader immediately (with stale LUTs for now)
    _update_preview(self, context)


# Update callback for quality change - always triggers rebake
def _update_quality_change(self, context):
    """Trigger rebake when quality setting changes."""
    global _debounce_timer
    import bpy
    
    scene = context.scene
    settings = scene.helios
    
    # Mark LUTs as invalid
    settings.luts_valid = False
    
    # Cancel any pending recompute
    if _debounce_timer is not None:
        try:
            bpy.app.timers.unregister(_debounce_timer)
        except:
            pass
        _debounce_timer = None
    
    # Schedule recompute immediately (no debounce needed for dropdown)
    def do_quality_recompute():
        global _debounce_timer
        _debounce_timer = None
        _do_recompute_luts(context)
        return None
    
    _debounce_timer = do_quality_recompute
    bpy.app.timers.register(do_quality_recompute, first_interval=0.1)


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
    preview_quality: EnumProperty(
        name="Quality Level",
        description="LUT precomputation quality. Preview is faster but less accurate for multiple scattering",
        items=[
            ('PREVIEW', "Preview (2 orders)", "Fast preview with 2 scattering orders (~10s)"),
            ('FINAL', "Final (4 orders)", "Full quality with 4 scattering orders (~40s)"),
        ],
        default='PREVIEW',
        update=_update_quality_change,
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
    ozone_density: FloatProperty(
        name="Ozone Density",
        description="Ozone layer density multiplier (affects sunset colors)",
        default=1.0,
        min=0.0,
        max=5.0,
        update=_update_preview_invalidate,
    )
    mie_angstrom_beta: FloatProperty(
        name="Mie Angstrom Beta",
        description="Aerosol optical thickness at reference wavelength (higher = denser atmosphere)",
        default=0.04,
        min=0.0,
        max=0.5,
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
    
    # Aerial Perspective Mode
    aerial_mode: EnumProperty(
        name="Aerial Mode",
        description="Shader implementation for aerial perspective (Node-based supports AOVs, OSL is reference)",
        items=[
            ('NODE', "Node-Based", "Use shader nodes (supports AOV output)"),
            ('OSL', "OSL", "Use OSL shader (reference, no AOV support in Blender 5.0)"),
        ],
        default='NODE',
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
