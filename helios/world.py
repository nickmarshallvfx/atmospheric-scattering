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


def create_atmosphere_world(context, use_preview=True):
    """
    Create or update the World shader for atmosphere rendering.
    Uses OSL shader with precomputed Bruneton LUTs.
    
    Args:
        context: Blender context
        use_preview: Currently unused, reserved for future preview mode.
    """
    scene = context.scene
    settings = scene.helios
    
    # Ensure Cycles is set and OSL is enabled
    if scene.render.engine != 'CYCLES':
        scene.render.engine = 'CYCLES'
        print("Helios: Switched render engine to Cycles")
    
    # Enable OSL - required for OSL shaders to work
    if hasattr(scene, 'cycles') and hasattr(scene.cycles, 'shading_system'):
        if not scene.cycles.shading_system:
            scene.cycles.shading_system = True
            print("Helios: Enabled Open Shading Language (OSL)")
    
    world = get_or_create_helios_world(context)
    world.use_nodes = True
    
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    # Check if we need to rebuild or just update
    osl_node = nodes.get(OSL_NODE_NAME)
    
    if osl_node is None:
        # Clear and rebuild
        nodes.clear()
        _build_sky_nodes(nodes, links, settings)
    else:
        # Just update existing nodes
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


def _force_viewport_update(context, world):
    """Force Blender to update the viewport with new sky settings."""
    # Tag the node tree as needing update
    if world and world.node_tree:
        world.node_tree.update_tag()
    
    # Tag all 3D viewports for redraw
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


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
