"""
Helios Node-Based Bruneton Aerial Perspective

This module creates a shader node group that implements the full Bruneton
GetSkyRadianceToPoint algorithm using standard Blender shader nodes.

This is necessary because AOV Output nodes don't work with OSL enabled
(Blender bugs #79942, #100282).

The node group outputs:
- Transmittance: RGB extinction from camera to surface point
- Inscatter: RGB light scattered into the view ray

Usage in Nuke: final = beauty * transmittance + inscatter

Copyright (c) 2017 Eric Bruneton (BSD License)
Copyright (c) 2024 MattePaint - Node-based implementation
"""

import bpy
import os
import math
from .utils import get_lut_cache_dir


# =============================================================================
# TEXTURE SIZE CONSTANTS (must match precomputation)
# =============================================================================

TRANSMITTANCE_TEXTURE_WIDTH = 256
TRANSMITTANCE_TEXTURE_HEIGHT = 64

SCATTERING_TEXTURE_R_SIZE = 32
SCATTERING_TEXTURE_MU_SIZE = 128
SCATTERING_TEXTURE_MU_S_SIZE = 32
SCATTERING_TEXTURE_NU_SIZE = 8

SCATTERING_TEXTURE_WIDTH = SCATTERING_TEXTURE_NU_SIZE * SCATTERING_TEXTURE_MU_S_SIZE
SCATTERING_TEXTURE_HEIGHT = SCATTERING_TEXTURE_MU_SIZE
SCATTERING_TEXTURE_DEPTH = SCATTERING_TEXTURE_R_SIZE


# =============================================================================
# NODE GROUP NAME
# =============================================================================

AERIAL_NODE_GROUP_NAME = "Helios_Aerial_Perspective"


# =============================================================================
# HELPER FUNCTIONS FOR NODE CREATION
# =============================================================================

class NodeBuilder:
    """Helper class for building shader node trees."""
    
    def __init__(self, node_tree):
        self.nodes = node_tree.nodes
        self.links = node_tree.links
        self._x = 0
        self._y = 0
    
    def set_position(self, x, y):
        """Set current position for next node."""
        self._x = x
        self._y = y
        return self
    
    def math(self, operation, x=None, y=None, label=None, v0=None, v1=None, v2=None):
        """Create a Math node."""
        node = self.nodes.new('ShaderNodeMath')
        node.operation = operation
        node.location = (x if x is not None else self._x, 
                        y if y is not None else self._y)
        if label:
            node.label = label
        if v0 is not None:
            node.inputs[0].default_value = v0
        if v1 is not None:
            node.inputs[1].default_value = v1
        if v2 is not None and len(node.inputs) > 2:
            node.inputs[2].default_value = v2
        return node
    
    def vec_math(self, operation, x=None, y=None, label=None):
        """Create a Vector Math node."""
        node = self.nodes.new('ShaderNodeVectorMath')
        node.operation = operation
        node.location = (x if x is not None else self._x,
                        y if y is not None else self._y)
        if label:
            node.label = label
        return node
    
    def combine_xyz(self, x=None, y=None, label=None):
        """Create a Combine XYZ node."""
        node = self.nodes.new('ShaderNodeCombineXYZ')
        node.location = (x if x is not None else self._x,
                        y if y is not None else self._y)
        if label:
            node.label = label
        return node
    
    def separate_xyz(self, x=None, y=None, label=None):
        """Create a Separate XYZ node."""
        node = self.nodes.new('ShaderNodeSeparateXYZ')
        node.location = (x if x is not None else self._x,
                        y if y is not None else self._y)
        if label:
            node.label = label
        return node
    
    def combine_color(self, x=None, y=None, label=None):
        """Create a Combine Color node."""
        node = self.nodes.new('ShaderNodeCombineColor')
        node.location = (x if x is not None else self._x,
                        y if y is not None else self._y)
        if label:
            node.label = label
        return node
    
    def separate_color(self, x=None, y=None, label=None):
        """Create a Separate Color node."""
        node = self.nodes.new('ShaderNodeSeparateColor')
        node.location = (x if x is not None else self._x,
                        y if y is not None else self._y)
        if label:
            node.label = label
        return node
    
    def mix(self, data_type='RGBA', blend_type='MIX', x=None, y=None, label=None, fac=None):
        """Create a Mix node."""
        node = self.nodes.new('ShaderNodeMix')
        node.data_type = data_type
        node.blend_type = blend_type
        node.location = (x if x is not None else self._x,
                        y if y is not None else self._y)
        if label:
            node.label = label
        if fac is not None:
            node.inputs[0].default_value = fac
        return node
    
    def image_texture(self, x=None, y=None, label=None, filepath=None):
        """Create an Image Texture node."""
        node = self.nodes.new('ShaderNodeTexImage')
        node.location = (x if x is not None else self._x,
                        y if y is not None else self._y)
        node.interpolation = 'Linear'
        node.extension = 'EXTEND'
        if label:
            node.label = label
        if filepath and os.path.exists(filepath):
            img = bpy.data.images.load(filepath, check_existing=True)
            img.colorspace_settings.name = 'Non-Color'
            node.image = img
        return node
    
    def value(self, x=None, y=None, label=None, val=0.0):
        """Create a Value node."""
        node = self.nodes.new('ShaderNodeValue')
        node.location = (x if x is not None else self._x,
                        y if y is not None else self._y)
        node.outputs[0].default_value = val
        if label:
            node.label = label
        return node
    
    def rgb(self, x=None, y=None, label=None, color=(1, 1, 1, 1)):
        """Create an RGB node."""
        node = self.nodes.new('ShaderNodeRGB')
        node.location = (x if x is not None else self._x,
                        y if y is not None else self._y)
        node.outputs[0].default_value = color
        if label:
            node.label = label
        return node
    
    def link(self, from_socket, to_socket):
        """Create a link between sockets."""
        self.links.new(from_socket, to_socket)
        return self


# =============================================================================
# TRANSMITTANCE UV CALCULATION
# =============================================================================

def create_transmittance_uv_nodes(builder, r_socket, mu_socket, 
                                   bottom_radius, top_radius,
                                   base_x=0, base_y=0):
    """
    Create nodes for GetTransmittanceTextureUvFromRMu.
    
    Returns: (u_socket, v_socket)
    """
    # H = sqrt(top_radius² - bottom_radius²)
    H = math.sqrt(top_radius * top_radius - bottom_radius * bottom_radius)
    
    # rho = sqrt(r² - bottom_radius²)
    r_sq = builder.math('POWER', base_x, base_y, 'r²', v1=2.0)
    builder.link(r_socket, r_sq.inputs[0])
    
    rho_sq = builder.math('SUBTRACT', base_x + 200, base_y, 'rho²', 
                          v1=bottom_radius * bottom_radius)
    builder.link(r_sq.outputs[0], rho_sq.inputs[0])
    
    rho = builder.math('SQRT', base_x + 400, base_y, 'rho')
    builder.link(rho_sq.outputs[0], rho.inputs[0])
    
    # x_r = rho / H
    x_r = builder.math('DIVIDE', base_x + 600, base_y, 'x_r', v1=H)
    builder.link(rho.outputs[0], x_r.inputs[0])
    
    # d = DistanceToTopAtmosphereBoundary(r, mu)
    # discriminant = r² * (mu² - 1) + top_radius²
    mu_sq = builder.math('POWER', base_x, base_y - 100, 'mu²', v1=2.0)
    builder.link(mu_socket, mu_sq.inputs[0])
    
    mu_sq_m1 = builder.math('SUBTRACT', base_x + 200, base_y - 100, 'mu²-1', v1=1.0)
    builder.link(mu_sq.outputs[0], mu_sq_m1.inputs[0])
    
    r_sq_term = builder.math('MULTIPLY', base_x + 400, base_y - 100, 'r²×(mu²-1)')
    builder.link(r_sq.outputs[0], r_sq_term.inputs[0])
    builder.link(mu_sq_m1.outputs[0], r_sq_term.inputs[1])
    
    discrim = builder.math('ADD', base_x + 600, base_y - 100, 'discriminant',
                           v1=top_radius * top_radius)
    builder.link(r_sq_term.outputs[0], discrim.inputs[0])
    
    discrim_sqrt = builder.math('SQRT', base_x + 800, base_y - 100, 'sqrt(disc)')
    builder.link(discrim.outputs[0], discrim_sqrt.inputs[0])
    
    # d = -r*mu + sqrt(discriminant)
    r_mu = builder.math('MULTIPLY', base_x + 200, base_y - 200, 'r×mu')
    builder.link(r_socket, r_mu.inputs[0])
    builder.link(mu_socket, r_mu.inputs[1])
    
    neg_r_mu = builder.math('MULTIPLY', base_x + 400, base_y - 200, '-r×mu', v1=-1.0)
    builder.link(r_mu.outputs[0], neg_r_mu.inputs[0])
    
    d = builder.math('ADD', base_x + 1000, base_y - 150, 'd')
    builder.link(neg_r_mu.outputs[0], d.inputs[0])
    builder.link(discrim_sqrt.outputs[0], d.inputs[1])
    
    # d_min = top_radius - r
    d_min = builder.math('SUBTRACT', base_x + 800, base_y, 'd_min', v0=top_radius)
    builder.link(r_socket, d_min.inputs[1])
    
    # d_max = rho + H
    d_max = builder.math('ADD', base_x + 800, base_y + 100, 'd_max', v1=H)
    builder.link(rho.outputs[0], d_max.inputs[0])
    
    # x_mu = (d - d_min) / (d_max - d_min)
    d_minus_dmin = builder.math('SUBTRACT', base_x + 1200, base_y - 100, 'd-d_min')
    builder.link(d.outputs[0], d_minus_dmin.inputs[0])
    builder.link(d_min.outputs[0], d_minus_dmin.inputs[1])
    
    dmax_minus_dmin = builder.math('SUBTRACT', base_x + 1200, base_y, 'dmax-dmin')
    builder.link(d_max.outputs[0], dmax_minus_dmin.inputs[0])
    builder.link(d_min.outputs[0], dmax_minus_dmin.inputs[1])
    
    x_mu = builder.math('DIVIDE', base_x + 1400, base_y - 50, 'x_mu')
    builder.link(d_minus_dmin.outputs[0], x_mu.inputs[0])
    builder.link(dmax_minus_dmin.outputs[0], x_mu.inputs[1])
    
    # u = GetTextureCoordFromUnitRange(x_mu, TRANSMITTANCE_WIDTH)
    # = 0.5/size + x * (1 - 1/size)
    u_scale = 1.0 - 1.0 / TRANSMITTANCE_TEXTURE_WIDTH
    u_offset = 0.5 / TRANSMITTANCE_TEXTURE_WIDTH
    
    u_scaled = builder.math('MULTIPLY', base_x + 1600, base_y - 50, 'u_scaled', v1=u_scale)
    builder.link(x_mu.outputs[0], u_scaled.inputs[0])
    
    u = builder.math('ADD', base_x + 1800, base_y - 50, 'u', v1=u_offset)
    builder.link(u_scaled.outputs[0], u.inputs[0])
    
    # v = GetTextureCoordFromUnitRange(x_r, TRANSMITTANCE_HEIGHT)
    v_scale = 1.0 - 1.0 / TRANSMITTANCE_TEXTURE_HEIGHT
    v_offset = 0.5 / TRANSMITTANCE_TEXTURE_HEIGHT
    
    v_scaled = builder.math('MULTIPLY', base_x + 1600, base_y + 50, 'v_scaled', v1=v_scale)
    builder.link(x_r.outputs[0], v_scaled.inputs[0])
    
    v = builder.math('ADD', base_x + 1800, base_y + 50, 'v', v1=v_offset)
    builder.link(v_scaled.outputs[0], v.inputs[0])
    
    return u.outputs[0], v.outputs[0]


# =============================================================================
# MAIN NODE GROUP CREATION
# =============================================================================

def create_aerial_perspective_node_group(lut_dir=None):
    """
    Create the Helios Aerial Perspective node group.
    
    This implements the full Bruneton GetSkyRadianceToPoint algorithm.
    """
    if lut_dir is None:
        lut_dir = get_lut_cache_dir()
    
    # Remove existing group if present
    if AERIAL_NODE_GROUP_NAME in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups[AERIAL_NODE_GROUP_NAME])
    
    # Create new node group
    group = bpy.data.node_groups.new(AERIAL_NODE_GROUP_NAME, 'ShaderNodeTree')
    builder = NodeBuilder(group)
    
    # =========================================================================
    # INTERFACE - Inputs and Outputs
    # =========================================================================
    
    group_input = group.nodes.new('NodeGroupInput')
    group_input.location = (-2000, 0)
    
    group_output = group.nodes.new('NodeGroupOutput')
    group_output.location = (3000, 0)
    
    # Create interface sockets (Blender 4.0+ API)
    if hasattr(group, 'interface'):
        # Inputs
        group.interface.new_socket('Position', in_out='INPUT', socket_type='NodeSocketVector')
        group.interface.new_socket('Camera_Position', in_out='INPUT', socket_type='NodeSocketVector')
        group.interface.new_socket('Sun_Direction', in_out='INPUT', socket_type='NodeSocketVector')
        group.interface.new_socket('Planet_Center', in_out='INPUT', socket_type='NodeSocketVector')
        group.interface.new_socket('Scene_Scale', in_out='INPUT', socket_type='NodeSocketFloat')
        group.interface.new_socket('Bottom_Radius', in_out='INPUT', socket_type='NodeSocketFloat')
        group.interface.new_socket('Top_Radius', in_out='INPUT', socket_type='NodeSocketFloat')
        group.interface.new_socket('Mie_Phase_G', in_out='INPUT', socket_type='NodeSocketFloat')
        group.interface.new_socket('Sun_Intensity', in_out='INPUT', socket_type='NodeSocketFloat')
        
        # Outputs  
        group.interface.new_socket('Transmittance', in_out='OUTPUT', socket_type='NodeSocketColor')
        group.interface.new_socket('Inscatter', in_out='OUTPUT', socket_type='NodeSocketColor')
    else:
        # Blender 3.x fallback
        group.inputs.new('NodeSocketVector', 'Position')
        group.inputs.new('NodeSocketVector', 'Camera_Position')
        group.inputs.new('NodeSocketVector', 'Sun_Direction')
        group.inputs.new('NodeSocketVector', 'Planet_Center')
        group.inputs.new('NodeSocketFloat', 'Scene_Scale')
        group.inputs.new('NodeSocketFloat', 'Bottom_Radius')
        group.inputs.new('NodeSocketFloat', 'Top_Radius')
        group.inputs.new('NodeSocketFloat', 'Mie_Phase_G')
        group.inputs.new('NodeSocketFloat', 'Sun_Intensity')
        
        group.outputs.new('NodeSocketColor', 'Transmittance')
        group.outputs.new('NodeSocketColor', 'Inscatter')
    
    # Set default values
    for socket in group_input.outputs:
        if socket.name == 'Scene_Scale':
            socket.default_value = 0.001  # 1 Blender unit = 1 meter = 0.001 km
        elif socket.name == 'Bottom_Radius':
            socket.default_value = 6360.0
        elif socket.name == 'Top_Radius':
            socket.default_value = 6420.0
        elif socket.name == 'Mie_Phase_G':
            socket.default_value = 0.8
        elif socket.name == 'Sun_Intensity':
            socket.default_value = 1.0
    
    # =========================================================================
    # LOAD LUT TEXTURES
    # =========================================================================
    
    transmittance_path = os.path.join(lut_dir, "transmittance.exr")
    scattering_path = os.path.join(lut_dir, "scattering.exr")
    
    tex_transmittance = builder.image_texture(-1600, 400, 'Transmittance_LUT', transmittance_path)
    tex_scattering = builder.image_texture(-1600, 200, 'Scattering_LUT', scattering_path)
    
    # =========================================================================
    # COORDINATE TRANSFORMS
    # =========================================================================
    
    # Convert world position to atmosphere coordinates (km, relative to planet center)
    # surface_point_km = (P_world - planet_center) * scene_scale
    
    pos_minus_center = builder.vec_math('SUBTRACT', -1600, 0, 'P - Planet_Center')
    builder.link(group_input.outputs['Position'], pos_minus_center.inputs[0])
    builder.link(group_input.outputs['Planet_Center'], pos_minus_center.inputs[1])
    
    surface_point_km = builder.vec_math('SCALE', -1400, 0, 'Surface_Point_km')
    builder.link(pos_minus_center.outputs[0], surface_point_km.inputs[0])
    builder.link(group_input.outputs['Scene_Scale'], surface_point_km.inputs['Scale'])
    
    # Camera in atmosphere coordinates
    cam_minus_center = builder.vec_math('SUBTRACT', -1600, -100, 'Cam - Planet_Center')
    builder.link(group_input.outputs['Camera_Position'], cam_minus_center.inputs[0])
    builder.link(group_input.outputs['Planet_Center'], cam_minus_center.inputs[1])
    
    camera_km = builder.vec_math('SCALE', -1400, -100, 'Camera_km')
    builder.link(cam_minus_center.outputs[0], camera_km.inputs[0])
    builder.link(group_input.outputs['Scene_Scale'], camera_km.inputs['Scale'])
    
    # =========================================================================
    # VIEW RAY AND DISTANCE
    # =========================================================================
    
    # view_ray = normalize(surface_point - camera)
    view_vec = builder.vec_math('SUBTRACT', -1200, -50, 'Surface - Camera')
    builder.link(surface_point_km.outputs[0], view_vec.inputs[0])
    builder.link(camera_km.outputs[0], view_vec.inputs[1])
    
    view_ray = builder.vec_math('NORMALIZE', -1000, -50, 'View_Ray')
    builder.link(view_vec.outputs[0], view_ray.inputs[0])
    
    # distance = length(surface_point - camera)
    distance = builder.vec_math('LENGTH', -1000, -150, 'Distance')
    builder.link(view_vec.outputs[0], distance.inputs[0])
    
    # =========================================================================
    # CAMERA PARAMETERS (r, mu, mu_s, nu)
    # =========================================================================
    
    # r = length(camera)
    r = builder.vec_math('LENGTH', -800, 0, 'r')
    builder.link(camera_km.outputs[0], r.inputs[0])
    
    # rmu = dot(camera, view_ray)
    rmu = builder.vec_math('DOT_PRODUCT', -800, -100, 'r×mu')
    builder.link(camera_km.outputs[0], rmu.inputs[0])
    builder.link(view_ray.outputs[0], rmu.inputs[1])
    
    # mu = rmu / r
    mu = builder.math('DIVIDE', -600, -50, 'mu')
    builder.link(rmu.outputs['Value'], mu.inputs[0])
    builder.link(r.outputs['Value'], mu.inputs[1])
    
    # mu_s = dot(camera, sun_direction) / r
    cam_dot_sun = builder.vec_math('DOT_PRODUCT', -800, -200, 'cam·sun')
    builder.link(camera_km.outputs[0], cam_dot_sun.inputs[0])
    builder.link(group_input.outputs['Sun_Direction'], cam_dot_sun.inputs[1])
    
    mu_s = builder.math('DIVIDE', -600, -200, 'mu_s')
    builder.link(cam_dot_sun.outputs['Value'], mu_s.inputs[0])
    builder.link(r.outputs['Value'], mu_s.inputs[1])
    
    # nu = dot(view_ray, sun_direction)
    nu = builder.vec_math('DOT_PRODUCT', -800, -300, 'nu')
    builder.link(view_ray.outputs[0], nu.inputs[0])
    builder.link(group_input.outputs['Sun_Direction'], nu.inputs[1])
    
    # =========================================================================
    # TRANSMITTANCE LOOKUP
    # =========================================================================
    
    # For now, use constant radii - we'll make these dynamic later
    BOTTOM_RADIUS = 6360.0
    TOP_RADIUS = 6420.0
    
    u_socket, v_socket = create_transmittance_uv_nodes(
        builder, r.outputs['Value'], mu.outputs[0],
        BOTTOM_RADIUS, TOP_RADIUS,
        base_x=-400, base_y=400
    )
    
    # Combine UV for texture lookup
    trans_uv = builder.combine_xyz(1600, 400, 'Trans_UV')
    builder.link(u_socket, trans_uv.inputs['X'])
    builder.link(v_socket, trans_uv.inputs['Y'])
    
    builder.link(trans_uv.outputs[0], tex_transmittance.inputs['Vector'])
    
    # =========================================================================
    # PHASE FUNCTIONS
    # =========================================================================
    
    # Rayleigh phase: 3/(16π) × (1 + nu²)
    nu_sq = builder.math('POWER', -400, -400, 'nu²', v1=2.0)
    builder.link(nu.outputs['Value'], nu_sq.inputs[0])
    
    ray_ph_term = builder.math('ADD', -200, -400, '1+nu²', v0=1.0)
    builder.link(nu_sq.outputs[0], ray_ph_term.inputs[1])
    
    rayleigh_phase = builder.math('MULTIPLY', 0, -400, 'Rayleigh_Phase', 
                                   v0=3.0 / (16.0 * math.pi))
    builder.link(ray_ph_term.outputs[0], rayleigh_phase.inputs[1])
    
    # Mie phase: 3/(8π) × (1-g²)/(2+g²) × (1+nu²) / (1+g²-2g×nu)^1.5
    # Using fixed g=0.8 for now
    G = 0.8
    mie_k = 3.0 / (8.0 * math.pi) * (1.0 - G*G) / (2.0 + G*G)
    
    # (1 + g² - 2g×nu)
    two_g_nu = builder.math('MULTIPLY', -400, -500, '2g×nu', v0=2.0 * G)
    builder.link(nu.outputs['Value'], two_g_nu.inputs[1])
    
    denom_base = builder.math('SUBTRACT', -200, -500, '1+g²-2gν', v0=1.0 + G*G)
    builder.link(two_g_nu.outputs[0], denom_base.inputs[1])
    
    denom_pow = builder.math('POWER', 0, -500, 'denom^1.5', v1=1.5)
    builder.link(denom_base.outputs[0], denom_pow.inputs[0])
    
    mie_num = builder.math('MULTIPLY', 200, -450, 'k×(1+nu²)', v0=mie_k)
    builder.link(ray_ph_term.outputs[0], mie_num.inputs[1])
    
    mie_phase = builder.math('DIVIDE', 400, -450, 'Mie_Phase')
    builder.link(mie_num.outputs[0], mie_phase.inputs[0])
    builder.link(denom_pow.outputs[0], mie_phase.inputs[1])
    
    # =========================================================================
    # TEMPORARY: Simple output for testing
    # Full implementation continues in next iteration
    # =========================================================================
    
    # For now, output transmittance from LUT lookup
    builder.link(tex_transmittance.outputs['Color'], group_output.inputs['Transmittance'])
    
    # Placeholder inscatter (will be replaced with full algorithm)
    placeholder_inscatter = builder.rgb(2800, -100, 'Placeholder_Inscatter', (0, 0, 0, 1))
    builder.link(placeholder_inscatter.outputs[0], group_output.inputs['Inscatter'])
    
    print(f"Helios: Created node group '{AERIAL_NODE_GROUP_NAME}'")
    return group


def get_or_create_aerial_node_group(lut_dir=None):
    """Get existing node group or create a new one."""
    if AERIAL_NODE_GROUP_NAME in bpy.data.node_groups:
        return bpy.data.node_groups[AERIAL_NODE_GROUP_NAME]
    return create_aerial_perspective_node_group(lut_dir)


# =============================================================================
# REGISTRATION
# =============================================================================

def register():
    pass


def unregister():
    pass
