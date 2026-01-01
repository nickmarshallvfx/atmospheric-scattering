"""
Helios Aerial Perspective Integration

This module handles:
- Setting up AOVs for aerial perspective outputs
- Adding aerial perspective shader to materials
- Managing parameters for the aerial perspective shader

The aerial perspective shader computes transmittance and inscatter for scene objects,
matching Eric Bruneton's GetSkyRadianceToPoint() implementation.
"""

import bpy
import math
import os
from mathutils import Vector

# Constants
AERIAL_OSL_NODE_NAME = "Helios_Aerial_OSL"
# AOV names must match exactly what's registered in the view layer
AERIAL_AOV_TRANSMITTANCE = "Helios_Transmittance"
AERIAL_AOV_INSCATTER = "Helios_Inscatter"


def get_aerial_shader_path():
    """Get the path to the aerial perspective OSL shader."""
    shader_dir = os.path.join(os.path.dirname(__file__), "shaders")
    return os.path.join(shader_dir, "aerial_perspective.osl")


def get_lut_cache_dir():
    """Get the directory where precomputed LUTs are stored."""
    blend_path = bpy.data.filepath
    if blend_path:
        cache_dir = os.path.join(os.path.dirname(blend_path), "helios_cache", "luts")
        if os.path.exists(cache_dir):
            return cache_dir
    
    config_dir = bpy.utils.user_resource('CONFIG')
    return os.path.join(config_dir, "helios_cache", "luts")


def get_camera_position_km(context, settings):
    """
    Get camera position in atmosphere coordinates (km, relative to planet center).
    
    Blender uses Z-up coordinate system.
    Camera altitude is converted from Blender units to km.
    """
    camera = context.scene.camera
    if camera is None:
        # Default: 500m above sea level
        return Vector((0, 0, 6360.5))
    
    # Get camera world position
    cam_pos = camera.matrix_world.translation
    
    # Convert to atmosphere coordinates
    # Assuming: planet center at origin, surface at Z=0, 1 Blender unit = 1 meter
    # Earth radius = 6360 km
    bottom_radius = 6360.0  # km
    
    # Camera position in km relative to planet center
    # Z coordinate: Blender Z=0 is Earth surface, so add bottom_radius
    cam_km = Vector((
        cam_pos.x * 0.001,  # X in km
        cam_pos.y * 0.001,  # Y in km
        cam_pos.z * 0.001 + bottom_radius  # Z in km (altitude + Earth radius)
    ))
    
    return cam_km


def get_sun_direction(settings) -> Vector:
    """Calculate sun direction from heading and elevation angles."""
    elevation = math.radians(settings.sun_elevation)
    heading = math.radians(settings.sun_heading)
    
    cos_elev = math.cos(elevation)
    sin_elev = math.sin(elevation)
    cos_head = math.cos(heading)
    sin_head = math.sin(heading)
    
    return Vector((
        cos_elev * sin_head,
        cos_elev * cos_head,
        sin_elev
    ))


def setup_aerial_aovs(context):
    """
    Set up AOV outputs for aerial perspective.
    Creates AOVs for transmittance and inscatter.
    """
    scene = context.scene
    view_layer = context.view_layer
    
    # Get or create AOVs
    aovs = view_layer.aovs
    
    # Transmittance AOV
    if AERIAL_AOV_TRANSMITTANCE not in aovs:
        aov = aovs.add()
        aov.name = AERIAL_AOV_TRANSMITTANCE
        aov.type = 'COLOR'
        print(f"Helios: Created AOV '{AERIAL_AOV_TRANSMITTANCE}'")
    
    # Inscatter AOV
    if AERIAL_AOV_INSCATTER not in aovs:
        aov = aovs.add()
        aov.name = AERIAL_AOV_INSCATTER
        aov.type = 'COLOR'
        print(f"Helios: Created AOV '{AERIAL_AOV_INSCATTER}'")
    
    return True


def remove_aerial_aovs(context):
    """Remove aerial perspective AOVs."""
    view_layer = context.view_layer
    aovs = view_layer.aovs
    
    for aov_name in [AERIAL_AOV_TRANSMITTANCE, AERIAL_AOV_INSCATTER]:
        if aov_name in aovs:
            idx = aovs.find(aov_name)
            if idx >= 0:
                aovs.remove(aovs[idx])
                print(f"Helios: Removed AOV '{aov_name}'")


def add_aerial_to_material(material, context):
    """
    Add aerial perspective shader to a material.
    
    This adds an OSL script node that outputs transmittance and inscatter AOVs.
    The AOVs can then be composited: beauty * transmittance + inscatter
    """
    if material is None or not material.use_nodes:
        return False
    
    settings = context.scene.helios
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    
    # Check if aerial node already exists
    if AERIAL_OSL_NODE_NAME in nodes:
        # Update existing node
        _update_aerial_node(nodes[AERIAL_OSL_NODE_NAME], context, settings)
        return True
    
    # Get shader path
    osl_path = get_aerial_shader_path()
    if not os.path.exists(osl_path):
        print(f"Helios: Aerial shader not found: {osl_path}")
        return False
    
    # Create OSL node
    osl_node = nodes.new('ShaderNodeScript')
    osl_node.name = AERIAL_OSL_NODE_NAME
    osl_node.label = "Helios Aerial Perspective"
    osl_node.location = (400, -200)
    osl_node.mode = 'EXTERNAL'
    osl_node.filepath = osl_path
    osl_node.update()
    
    # Debug: print available inputs/outputs
    print(f"Helios: OSL inputs: {[i.name for i in osl_node.inputs]}")
    print(f"Helios: OSL outputs: {[o.name for o in osl_node.outputs]}")
    
    # Set LUT paths and parameters
    _update_aerial_node(osl_node, context, settings)
    
    # Create AOV output nodes
    _create_aov_outputs(nodes, links, osl_node)
    
    # IMPORTANT: OSL nodes are only evaluated if connected to the material output chain.
    # Connect via a transparent mix to ensure evaluation without affecting appearance.
    _connect_to_material_output(nodes, links, osl_node)
    
    print(f"Helios: Added aerial perspective to material '{material.name}'")
    return True


def _update_aerial_node(osl_node, context, settings):
    """Update aerial perspective node parameters."""
    lut_dir = get_lut_cache_dir()
    print(f"Helios: LUT dir = {lut_dir}")
    
    # Set LUT texture paths
    lut_files = {
        'transmittance_texture': 'transmittance.exr',
        'scattering_texture': 'scattering.exr',
        'single_mie_scattering_texture': 'single_mie_scattering.exr',
    }
    
    for input_name, filename in lut_files.items():
        filepath = os.path.join(lut_dir, filename)
        exists = os.path.exists(filepath)
        print(f"Helios: {input_name} = {filepath} (exists: {exists})")
        if input_name in osl_node.inputs and exists:
            osl_node.inputs[input_name].default_value = filepath
        elif input_name not in osl_node.inputs:
            print(f"Helios: WARNING - input '{input_name}' not found in OSL node")
    
    # Set atmosphere parameters
    if 'bottom_radius' in osl_node.inputs:
        osl_node.inputs['bottom_radius'].default_value = 6360.0
    if 'top_radius' in osl_node.inputs:
        osl_node.inputs['top_radius'].default_value = 6420.0
    if 'mu_s_min' in osl_node.inputs:
        osl_node.inputs['mu_s_min'].default_value = -0.2
    
    # Set scattering coefficients (scaled by density)
    # OSL color inputs expect 4 values (RGBA)
    rayleigh_base = Vector((0.0058, 0.0135, 0.0331))
    mie_base = 0.004
    
    if 'rayleigh_scattering' in osl_node.inputs:
        scaled = rayleigh_base * settings.rayleigh_density
        osl_node.inputs['rayleigh_scattering'].default_value = (scaled.x, scaled.y, scaled.z, 1.0)
    
    if 'mie_scattering' in osl_node.inputs:
        scaled_mie = mie_base * settings.mie_density
        osl_node.inputs['mie_scattering'].default_value = (scaled_mie, scaled_mie, scaled_mie, 1.0)
    
    if 'mie_phase_g' in osl_node.inputs:
        osl_node.inputs['mie_phase_g'].default_value = settings.mie_phase_g
    
    # Set sun direction
    sun_dir = get_sun_direction(settings)
    if 'sun_direction' in osl_node.inputs:
        osl_node.inputs['sun_direction'].default_value = (sun_dir.x, sun_dir.y, sun_dir.z)
    
    if 'sun_intensity' in osl_node.inputs:
        osl_node.inputs['sun_intensity'].default_value = settings.sun_intensity
    
    # Set camera position
    cam_pos = get_camera_position_km(context, settings)
    print(f"Helios: Camera position (km): {cam_pos}")
    if 'camera_position' in osl_node.inputs:
        osl_node.inputs['camera_position'].default_value = (cam_pos.x, cam_pos.y, cam_pos.z)
    
    # Set scene scale and planet center
    if 'scene_scale' in osl_node.inputs:
        osl_node.inputs['scene_scale'].default_value = 0.001  # 1 Blender unit = 1 meter = 0.001 km
    
    if 'planet_center' in osl_node.inputs:
        # Planet center in Blender coords: Z = -6360 km = -6360000 meters
        osl_node.inputs['planet_center'].default_value = (0, 0, -6360000.0)
    
    # Debug mode - set to 1 to verify shader is running
    if 'debug_mode' in osl_node.inputs:
        osl_node.inputs['debug_mode'].default_value = 1  # TEMP: Enable debug to verify AOV connection
    
    print(f"Helios: Sun direction: {sun_dir}")


def _connect_to_material_output(nodes, links, osl_node):
    """
    Connect OSL node to material output chain to ensure it gets evaluated.
    
    We use a Mix Shader approach: mix the original material with an inscatter
    emission at factor 0.0001 (nearly invisible but forces evaluation).
    This also provides a visible debug mode when factor is increased.
    """
    # Find Material Output node
    output_node = None
    for node in nodes:
        if node.type == 'OUTPUT_MATERIAL' and node.is_active_output:
            output_node = node
            break
    
    if output_node is None:
        print("Helios: No active Material Output node found")
        return
    
    # Get what's currently connected to the output
    surface_input = output_node.inputs.get('Surface')
    if surface_input is None:
        print("Helios: Material Output has no Surface input")
        return
    
    original_socket = None
    for link in surface_input.links:
        original_socket = link.from_socket
        break
    
    if original_socket is None:
        print("Helios: Nothing connected to Material Output - skipping aerial integration")
        return
    
    # Create Emission shader for inscatter visualization
    emission = nodes.new('ShaderNodeEmission')
    emission.name = "Helios_Aerial_Emission"
    emission.location = (osl_node.location.x + 200, osl_node.location.y - 200)
    emission.inputs['Strength'].default_value = 1.0
    
    # Connect inscatter to emission color (this makes the haze visible)
    if 'AerialInscatter' in osl_node.outputs:
        links.new(osl_node.outputs['AerialInscatter'], emission.inputs['Color'])
    
    # Create Mix Shader - this evaluates BOTH inputs regardless of factor
    mix_shader = nodes.new('ShaderNodeMixShader')
    mix_shader.name = "Helios_Aerial_Mix"
    mix_shader.location = (output_node.location.x - 200, output_node.location.y)
    # Factor 0.5 = visible blend for debugging; set to 0.0001 for production
    mix_shader.inputs['Fac'].default_value = 0.5  # TEMP: Visible for debugging
    
    # Connect: original -> Mix Shader input 1
    links.new(original_socket, mix_shader.inputs[1])
    
    # Connect: emission (inscatter) -> Mix Shader input 2
    links.new(emission.outputs['Emission'], mix_shader.inputs[2])
    
    # Connect: Mix Shader -> Material Output
    links.new(mix_shader.outputs['Shader'], surface_input)
    
    print("Helios: Connected aerial perspective to material output chain (Mix factor=0.5 for debug)")


def _create_aov_outputs(nodes, links, osl_node):
    """Create AOV output nodes for transmittance and inscatter."""
    
    print(f"Helios: Creating AOV outputs...")
    print(f"Helios: OSL output sockets: {[(o.name, o.type) for o in osl_node.outputs]}")
    
    # MINIMAL TEST: Use Geometry node position (known to work in Blender)
    # This tests if AOV Output nodes work at all, independent of OSL
    geom_node = nodes.new('ShaderNodeNewGeometry')
    geom_node.name = "Helios_Test_Geom"
    geom_node.location = (osl_node.location.x + 100, osl_node.location.y + 100)
    
    # Transmittance AOV output - TEST: Use geometry position (should show world coords as color)
    aov_trans = nodes.new('ShaderNodeOutputAOV')
    aov_trans.name = "Helios_AOV_Transmittance"
    aov_trans.label = "Transmittance AOV"
    aov_trans.location = (osl_node.location.x + 300, osl_node.location.y)
    aov_trans.aov_name = AERIAL_AOV_TRANSMITTANCE
    
    # Debug: Check what inputs the AOV node has
    print(f"Helios: AOV node inputs: {[(i.name, i.type) for i in aov_trans.inputs]}")
    print(f"Helios: AOV node aov_name = '{aov_trans.aov_name}'")
    
    # Connect geometry position to AOV - this should show XYZ as RGB
    links.new(geom_node.outputs['Position'], aov_trans.inputs['Color'])
    print(f"Helios: Connected Geometry.Position -> {AERIAL_AOV_TRANSMITTANCE}")
    
    # Inscatter AOV output - use geometry normal for another test
    aov_inscatter = nodes.new('ShaderNodeOutputAOV')
    aov_inscatter.name = "Helios_AOV_Inscatter"
    aov_inscatter.label = "Inscatter AOV"
    aov_inscatter.location = (osl_node.location.x + 300, osl_node.location.y - 100)
    aov_inscatter.aov_name = AERIAL_AOV_INSCATTER
    
    links.new(geom_node.outputs['Normal'], aov_inscatter.inputs['Color'])
    print(f"Helios: Connected Geometry.Normal -> {AERIAL_AOV_INSCATTER}")


def remove_aerial_from_material(material):
    """Remove aerial perspective shader from a material."""
    if material is None or not material.use_nodes:
        return False
    
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    
    # First, restore original connection if we added the Mix/Add Shader
    add_shader = nodes.get("Helios_Aerial_Mix") or nodes.get("Helios_Aerial_Add")
    if add_shader:
        # Find what was connected to the shader's first input (original material)
        # Mix Shader uses index 1, Add Shader uses index 0
        input_index = 1 if add_shader.name == "Helios_Aerial_Mix" else 0
        original_socket = None
        for link in add_shader.inputs[input_index].links:
            original_socket = link.from_socket
            break
        
        # Find the Material Output
        output_node = None
        for node in nodes:
            if node.type == 'OUTPUT_MATERIAL' and node.is_active_output:
                output_node = node
                break
        
        # Restore original connection
        if original_socket and output_node:
            surface_input = output_node.inputs.get('Surface')
            if surface_input:
                links.new(original_socket, surface_input)
    
    # Remove all Helios aerial nodes
    nodes_to_remove = [
        AERIAL_OSL_NODE_NAME,
        "Helios_AOV_Transmittance",
        "Helios_AOV_Inscatter",
        "Helios_Aerial_Emission",
        "Helios_Aerial_Add",
        "Helios_Aerial_Mix",
        "Helios_Test_RGB",
        "Helios_Test_Geom",
    ]
    
    for node_name in nodes_to_remove:
        if node_name in nodes:
            nodes.remove(nodes[node_name])
    
    print(f"Helios: Removed aerial perspective from material '{material.name}'")
    return True


def add_aerial_to_all_materials(context):
    """Add aerial perspective to all materials in the scene."""
    count = 0
    for material in bpy.data.materials:
        if material.use_nodes:
            if add_aerial_to_material(material, context):
                count += 1
    
    print(f"Helios: Added aerial perspective to {count} materials")
    return count


def remove_aerial_from_all_materials():
    """Remove aerial perspective from all materials."""
    count = 0
    for material in bpy.data.materials:
        if material.use_nodes:
            if remove_aerial_from_material(material):
                count += 1
    
    print(f"Helios: Removed aerial perspective from {count} materials")
    return count


def update_all_aerial_nodes(context):
    """Update all aerial perspective nodes with current settings."""
    settings = context.scene.helios
    
    for material in bpy.data.materials:
        if material.use_nodes:
            nodes = material.node_tree.nodes
            if AERIAL_OSL_NODE_NAME in nodes:
                _update_aerial_node(nodes[AERIAL_OSL_NODE_NAME], context, settings)
    
    print("Helios: Updated all aerial perspective nodes")
