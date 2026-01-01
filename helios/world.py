"""
Helios World Integration - Creates and manages the World shader for atmosphere rendering.

This module handles:
- Creating the World node tree for atmosphere rendering
- Using OSL shader with precomputed Bruneton LUTs for final rendering
- Sun direction calculation from heading/elevation

The shader is a direct port of Eric Bruneton's GLSL atmospheric scattering to OSL.
"""

import bpy
import math
import os
from mathutils import Vector

# Constants
OSL_NODE_NAME = "Helios_OSL"
BACKGROUND_NODE_NAME = "Helios_Background"

# Get path to OSL shader
def get_osl_shader_path():
    """Get the path to the Bruneton OSL shader."""
    shader_dir = os.path.join(os.path.dirname(__file__), "shaders")
    return os.path.join(shader_dir, "bruneton_sky.osl")


def get_lut_cache_dir():
    """Get the directory where precomputed LUTs are stored."""
    # Check blend file directory first, then user config
    blend_path = bpy.data.filepath
    if blend_path:
        cache_dir = os.path.join(os.path.dirname(blend_path), "helios_cache", "luts")
        if os.path.exists(cache_dir):
            return cache_dir
    
    # Fallback to Blender's user config directory
    config_dir = bpy.utils.user_resource('CONFIG')
    return os.path.join(config_dir, "helios_cache", "luts")


def get_sun_direction(settings) -> Vector:
    """Calculate sun direction from heading and elevation angles."""
    elevation = math.radians(settings.sun_elevation)
    heading = math.radians(settings.sun_heading)
    
    # Z-up coordinate system (Blender native)
    # Elevation: 0 = horizon, 90 = overhead
    # Heading: 0 = North (+Y), 90 = East (+X), 180 = South (-Y), 270 = West (-X)
    cos_elev = math.cos(elevation)
    sin_elev = math.sin(elevation)
    cos_head = math.cos(heading)
    sin_head = math.sin(heading)
    
    return Vector((
        cos_elev * sin_head,   # X: East-West
        cos_elev * cos_head,   # Y: North-South  
        sin_elev               # Z: Up
    ))


def setup_helios_aovs(context):
    """Set up all Helios AOVs in the view layer.
    
    Based on working implementation from atmospheric-scattering-3.
    """
    view_layer = context.view_layer
    scene = context.scene
    
    # Enable Film Transparent - removes sky from beauty pass for clean compositing
    scene.render.film_transparent = True
    print("Helios: Enabled Film Transparent")
    
    # Enable Environment pass - provides world background for all pixels
    view_layer.use_pass_environment = True
    print("Helios: Enabled Environment pass")
    
    aovs = view_layer.aovs
    
    # Define all Helios AOVs - names must match exactly what AOV Output nodes use
    helios_aovs = [
        "Helios_Sky",
        "Helios_Transmittance", 
        "Helios_Inscatter",
    ]
    
    for aov_name in helios_aovs:
        # Check if already exists
        exists = False
        for aov in aovs:
            if aov.name == aov_name:
                exists = True
                break
        
        if not exists:
            aov = aovs.add()
            aov.name = aov_name
            aov.type = 'COLOR'
            print(f"Helios: Created AOV '{aov_name}'")
        else:
            print(f"Helios: AOV already exists: '{aov_name}'")
    
    return True


def is_helios_world(world) -> bool:
    """Check if a world is a Helios atmosphere world."""
    if world is None:
        return False
    return world.get("is_helios", False)


def get_or_create_helios_world(context):
    """Get existing Helios world or create a new one."""
    scene = context.scene
    
    # Check if current world is Helios
    if is_helios_world(scene.world):
        return scene.world
    
    # Check if a Helios world exists in the blend file
    for world in bpy.data.worlds:
        if is_helios_world(world):
            scene.world = world
            return world
    
    # Create new world
    world = bpy.data.worlds.new("Helios Atmosphere")
    world["is_helios"] = True
    scene.world = world
    return world


SKY_NODE_GROUP_INSTANCE = "Helios_Sky_Group"


def create_atmosphere_world(context, use_preview=True):
    """
    Create or update the World shader for atmosphere rendering.
    
    Supports two modes based on settings.aerial_mode:
    - NODE: Uses shader node group (supports AOV output, no OSL required)
    - OSL: Uses OSL shader (reference implementation)
    
    Args:
        context: Blender context
        use_preview: Currently unused, reserved for future preview mode.
    """
    scene = context.scene
    settings = scene.helios
    
    # Ensure Cycles is set
    if scene.render.engine != 'CYCLES':
        scene.render.engine = 'CYCLES'
        print("Helios: Switched render engine to Cycles")
    
    # Handle OSL based on mode
    if settings.aerial_mode == 'OSL':
        # Enable OSL for OSL shader mode
        if hasattr(scene, 'cycles') and hasattr(scene.cycles, 'shading_system'):
            if not scene.cycles.shading_system:
                scene.cycles.shading_system = True
                print("Helios: Enabled Open Shading Language (OSL)")
    else:
        # Disable OSL for node-based mode (required for AOVs to work)
        if hasattr(scene, 'cycles') and hasattr(scene.cycles, 'shading_system'):
            if scene.cycles.shading_system:
                scene.cycles.shading_system = False
                print("Helios: Disabled OSL for node-based mode (AOVs require this)")
    
    world = get_or_create_helios_world(context)
    world.use_nodes = True
    
    # Set up AOVs in view layer
    setup_helios_aovs(context)
    
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    # Route to appropriate implementation based on mode
    if settings.aerial_mode == 'NODE':
        # Check if node-based sky exists
        sky_group = nodes.get(SKY_NODE_GROUP_INSTANCE)
        if sky_group is None:
            # Clear and rebuild with node-based sky
            nodes.clear()
            _build_sky_nodes_node_based(nodes, links, settings, context)
        else:
            _update_sky_nodes_node_based(nodes, settings, context)
    else:
        # Check if OSL sky exists
        osl_node = nodes.get(OSL_NODE_NAME)
        if osl_node is None:
            # Clear and rebuild with OSL
            nodes.clear()
            _build_sky_nodes(nodes, links, settings)
        else:
            _update_sky_nodes(nodes, settings)
    
    # Force viewport update
    _force_viewport_update(context, world)
    
    return world


def _build_sky_nodes(nodes, links, settings):
    """Build the sky node tree using OSL Bruneton shader."""
    
    osl_path = get_osl_shader_path()
    lut_dir = get_lut_cache_dir()
    
    print(f"Helios: OSL shader path: {osl_path}")
    print(f"Helios: LUT cache dir: {lut_dir}")
    
    # Check if OSL shader exists
    if not os.path.exists(osl_path):
        print(f"Helios: OSL shader not found!")
        _build_fallback_nodes(nodes, links, settings)
        return
    
    # Check if LUTs exist
    if not os.path.exists(lut_dir):
        print(f"Helios: LUT cache dir not found - run Precompute first")
    else:
        lut_files = ['transmittance.exr', 'scattering.exr', 'irradiance.exr']
        for f in lut_files:
            path = os.path.join(lut_dir, f)
            exists = os.path.exists(path)
            print(f"Helios: LUT {f}: {'FOUND' if exists else 'MISSING'}")
    
    # === OSL Script Node ===
    osl_node = nodes.new('ShaderNodeScript')
    osl_node.name = OSL_NODE_NAME
    osl_node.location = (-400, 0)
    osl_node.mode = 'EXTERNAL'
    osl_node.filepath = osl_path
    
    # Force OSL script to compile by updating the node tree
    # This is needed for inputs/outputs to become available
    osl_node.update()
    
    print(f"Helios: OSL outputs: {[o.name for o in osl_node.outputs]}")
    print(f"Helios: OSL inputs: {[i.name for i in osl_node.inputs]}")
    
    # === Background Node ===
    background = nodes.new('ShaderNodeBackground')
    background.name = BACKGROUND_NODE_NAME
    background.location = (200, 0)
    
    # === Output Node ===
    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (400, 0)
    
    # Connect OSL Sky output to Background
    if 'Sky' in osl_node.outputs:
        links.new(osl_node.outputs['Sky'], background.inputs['Color'])
        print("Helios: Connected Sky output to Background")
    else:
        print("Helios: WARNING - 'Sky' output not found in OSL node!")
        # Fallback: connect first color output if available
        for out in osl_node.outputs:
            if out.type == 'RGBA' or out.type == 'COLOR':
                links.new(out, background.inputs['Color'])
                print(f"Helios: Connected fallback output '{out.name}' to Background")
                break
    
    links.new(background.outputs['Background'], output.inputs['Surface'])
    
    # === Create AOV outputs for Sky ===
    # Sky AOV
    if 'Sky' in osl_node.outputs:
        aov_sky = nodes.new('ShaderNodeOutputAOV')
        aov_sky.name = "Helios_AOV_Sky"
        aov_sky.label = "Sky AOV"
        aov_sky.location = (200, -150)
        aov_sky.aov_name = "Helios_Sky"
        links.new(osl_node.outputs['Sky'], aov_sky.inputs['Color'])
        print("Helios: Created Sky AOV output")
    
    # Set LUT texture paths
    _set_lut_paths(osl_node, lut_dir)
    
    # Apply initial settings
    _update_sky_nodes(nodes, settings)


def _build_fallback_nodes(nodes, links, settings):
    """Build bright pink fallback when OSL shader fails - makes errors obvious."""
    # Bright pink = obvious failure indicator for visual testing
    background = nodes.new('ShaderNodeBackground')
    background.name = BACKGROUND_NODE_NAME
    background.location = (0, 0)
    background.inputs['Color'].default_value = (1.0, 0.0, 1.0, 1.0)  # Bright pink
    background.inputs['Strength'].default_value = 1.0
    
    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (200, 0)
    links.new(background.outputs['Background'], output.inputs['Surface'])


def _set_lut_paths(osl_node, lut_dir):
    """Set the LUT texture paths on the OSL node."""
    if not os.path.exists(lut_dir):
        print(f"Helios: LUT dir does not exist: {lut_dir}")
        return
    
    # Map OSL input names to LUT file names
    lut_files = {
        'transmittance_texture': 'transmittance.exr',
        'scattering_texture': 'scattering.exr',
        'single_mie_scattering_texture': 'single_mie_scattering.exr',
        'irradiance_texture': 'irradiance.exr',
    }
    
    for input_name, filename in lut_files.items():
        filepath = os.path.join(lut_dir, filename)
        if input_name in osl_node.inputs:
            if os.path.exists(filepath):
                osl_node.inputs[input_name].default_value = filepath
                print(f"Helios: Set {input_name} = {filepath}")
            else:
                print(f"Helios: LUT file missing: {filepath}")
        else:
            print(f"Helios: Input '{input_name}' not found in OSL node")


def _update_sky_nodes(nodes, settings):
    """Update sky node parameters from Helios settings."""
    
    osl_node = nodes.get(OSL_NODE_NAME)
    background = nodes.get(BACKGROUND_NODE_NAME)
    
    # Calculate sun direction from heading and elevation
    sun_dir = get_sun_direction(settings)
    
    # Update OSL shader inputs if available
    if osl_node is not None:
        # Sun direction (normalized vector)
        if 'sun_direction' in osl_node.inputs:
            osl_node.inputs['sun_direction'].default_value = (sun_dir.x, sun_dir.y, sun_dir.z)
        
        # Mie phase function g
        if 'mie_phase_g' in osl_node.inputs:
            osl_node.inputs['mie_phase_g'].default_value = settings.mie_phase_g
        
        # Exposure
        if 'exposure' in osl_node.inputs:
            osl_node.inputs['exposure'].default_value = settings.exposure
        
        # Sun intensity (affects solar irradiance)
        if 'sun_intensity' in osl_node.inputs:
            osl_node.inputs['sun_intensity'].default_value = settings.sun_intensity
        
        # Camera altitude (observer position)
        # Default to 500m above sea level = 6360.5 km from planet center
        if 'camera_position' in osl_node.inputs:
            altitude_km = 6360.0 + 0.5  # Earth radius + 500m
            osl_node.inputs['camera_position'].default_value = (0, 0, altitude_km)
    
    # Update background strength
    if background:
        background.inputs['Strength'].default_value = 1.0  # OSL handles exposure internally


def _build_sky_nodes_node_based(nodes, links, settings, context):
    """Build the sky node tree using node-based Bruneton shader (no OSL)."""
    from . import sky_nodes
    
    lut_dir = get_lut_cache_dir()
    print(f"Helios: Building node-based sky, LUT dir: {lut_dir}")
    
    # Get or create the sky node group
    sky_group = sky_nodes.get_or_create_sky_node_group(lut_dir)
    
    # Add sky node group instance
    group_node = nodes.new('ShaderNodeGroup')
    group_node.name = SKY_NODE_GROUP_INSTANCE
    group_node.label = "Helios Sky (Node)"
    group_node.node_tree = sky_group
    group_node.location = (-200, 0)
    
    # Geometry node for view direction
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.name = "Helios_Geom"
    geom.location = (-600, 0)
    
    # Negate incoming ray to get view direction (I points toward surface)
    negate = nodes.new('ShaderNodeVectorMath')
    negate.operation = 'SCALE'
    negate.location = (-400, 0)
    negate.inputs['Scale'].default_value = -1.0
    links.new(geom.outputs['Incoming'], negate.inputs[0])
    
    # Normalize view direction
    normalize = nodes.new('ShaderNodeVectorMath')
    normalize.operation = 'NORMALIZE'
    normalize.location = (-200, -100)
    links.new(negate.outputs[0], normalize.inputs[0])
    
    # Connect view direction to sky group
    links.new(normalize.outputs[0], group_node.inputs['View_Direction'])
    
    # Background node
    background = nodes.new('ShaderNodeBackground')
    background.name = BACKGROUND_NODE_NAME
    background.location = (200, 0)
    
    # Output node
    output = nodes.new('ShaderNodeOutputWorld')
    output.location = (400, 0)
    
    # Connect sky to background
    links.new(group_node.outputs['Sky'], background.inputs['Color'])
    links.new(background.outputs['Background'], output.inputs['Surface'])
    
    # Create AOV outputs
    aov_sky = nodes.new('ShaderNodeOutputAOV')
    aov_sky.name = "Helios_AOV_Sky"
    aov_sky.location = (200, -150)
    aov_sky.aov_name = "Helios_Sky"
    links.new(group_node.outputs['Sky'], aov_sky.inputs['Color'])
    
    aov_trans = nodes.new('ShaderNodeOutputAOV')
    aov_trans.name = "Helios_AOV_Sky_Trans"
    aov_trans.location = (200, -250)
    aov_trans.aov_name = "Helios_Sky_Transmittance"
    links.new(group_node.outputs['Transmittance'], aov_trans.inputs['Color'])
    
    aov_inscatter = nodes.new('ShaderNodeOutputAOV')
    aov_inscatter.name = "Helios_AOV_Sky_Inscatter"
    aov_inscatter.location = (200, -350)
    aov_inscatter.aov_name = "Helios_Sky_Inscatter"
    links.new(group_node.outputs['Inscatter'], aov_inscatter.inputs['Color'])
    
    # Apply initial settings
    _update_sky_nodes_node_based(nodes, settings, context)
    
    print("Helios: Built node-based sky shader with AOVs")


def _update_sky_nodes_node_based(nodes, settings, context):
    """Update node-based sky parameters from Helios settings."""
    
    group_node = nodes.get(SKY_NODE_GROUP_INSTANCE)
    if group_node is None:
        return
    
    # Calculate sun direction
    sun_dir = get_sun_direction(settings)
    
    # Camera position (500m above sea level by default)
    camera = context.scene.camera
    if camera:
        cam_pos = camera.matrix_world.translation
        # Convert to km, add Earth radius
        cam_km = (0, 0, 6360.0 + cam_pos.z * 0.001)
    else:
        cam_km = (0, 0, 6360.5)  # Default 500m above sea level
    
    # Update node group inputs
    if 'Sun_Direction' in group_node.inputs:
        group_node.inputs['Sun_Direction'].default_value = (sun_dir.x, sun_dir.y, sun_dir.z)
    
    if 'Camera_Position' in group_node.inputs:
        group_node.inputs['Camera_Position'].default_value = cam_km
    
    if 'Mie_Phase_G' in group_node.inputs:
        group_node.inputs['Mie_Phase_G'].default_value = settings.mie_phase_g
    
    if 'Sun_Intensity' in group_node.inputs:
        group_node.inputs['Sun_Intensity'].default_value = settings.sun_intensity
    
    if 'Exposure' in group_node.inputs:
        group_node.inputs['Exposure'].default_value = settings.exposure


def _force_viewport_update(context, world):
    """Force Blender to update the viewport with new sky settings."""
    import os
    
    # Tag the node tree as needing update
    if world and world.node_tree:
        world.node_tree.update_tag()
    
    # Force reload of Helios LUT textures from disk
    # This is needed because Cycles caches textures
    lut_dir = get_lut_cache_dir()
    lut_files = ['transmittance.exr', 'scattering.exr', 'irradiance.exr', 'single_mie_scattering.exr']
    
    for img in bpy.data.images:
        if img.filepath:
            img_path = bpy.path.abspath(img.filepath)
            img_name = os.path.basename(img_path)
            if img_name in lut_files:
                img.reload()
                print(f"Helios: Reloaded texture {img_name}")
    
    # For OSL shaders, force texture cache invalidation by re-setting the paths
    # This tricks Cycles into re-reading the texture files
    if world and world.node_tree:
        osl_node = world.node_tree.nodes.get(OSL_NODE_NAME)
        if osl_node:
            lut_inputs = {
                'transmittance_texture': 'transmittance.exr',
                'scattering_texture': 'scattering.exr',
                'single_mie_scattering_texture': 'single_mie_scattering.exr',
                'irradiance_texture': 'irradiance.exr',
            }
            for input_name, filename in lut_inputs.items():
                if input_name in osl_node.inputs:
                    filepath = os.path.join(lut_dir, filename)
                    # Re-set the path to force Cycles to reload
                    osl_node.inputs[input_name].default_value = filepath
    
    # Tag depsgraph for update - this forces Cycles to re-evaluate
    context.view_layer.update()
    
    # Tag scene for update
    if hasattr(context, 'scene') and context.scene:
        context.scene.update_tag()
    
    # Tag all 3D viewports for redraw and force shading mode toggle
    # This is needed because Cycles caches world textures aggressively
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
                for region in area.regions:
                    region.tag_redraw()
                
                # Force viewport refresh by briefly toggling shading mode
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        current_shading = space.shading.type
                        if current_shading == 'RENDERED':
                            # Toggle to SOLID and back to force texture cache invalidation
                            space.shading.type = 'SOLID'
                            space.shading.type = 'RENDERED'
                            print("Helios: Toggled viewport shading to force texture reload")


def update_atmosphere_world(context):
    """
    Update the world shader with current settings.
    Called from property update callbacks for real-time preview.
    """
    scene = context.scene
    
    # Only update if we have a Helios world
    if not is_helios_world(scene.world):
        return
    
    settings = scene.helios
    nodes = scene.world.node_tree.nodes
    
    _update_sky_nodes(nodes, settings)
    _force_viewport_update(context, scene.world)


# =============================================================================
# OPERATORS
# =============================================================================

class HELIOS_OT_create_world(bpy.types.Operator):
    """Create Helios atmosphere world shader"""
    bl_idname = "helios.create_world"
    bl_label = "Create Atmosphere"
    bl_description = "Create a World shader with Helios atmospheric scattering"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        create_atmosphere_world(context, use_preview=True)
        self.report({'INFO'}, "Created Helios atmosphere world")
        return {'FINISHED'}


class HELIOS_OT_update_world(bpy.types.Operator):
    """Update Helios atmosphere world shader with current settings"""
    bl_idname = "helios.update_world"
    bl_label = "Update Atmosphere"
    bl_description = "Rebuild the atmosphere world shader with current settings"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        # Force full rebuild
        if context.scene.world and is_helios_world(context.scene.world):
            nodes = context.scene.world.node_tree.nodes
            nodes.clear()
            _build_sky_nodes(nodes, context.scene.world.node_tree.links, context.scene.helios)
            _force_viewport_update(context, context.scene.world)
        else:
            create_atmosphere_world(context, use_preview=True)
        
        self.report({'INFO'}, "Updated atmosphere")
        return {'FINISHED'}


# =============================================================================
# REGISTRATION
# =============================================================================

# Note: Operators are registered via operators/__init__.py
# This module only provides the classes and functions

def register():
    pass


def unregister():
    pass
