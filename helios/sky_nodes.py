"""
Helios Node-Based Bruneton Sky Shader

This module creates a shader node group that implements the full Bruneton
GetSkyRadiance algorithm using standard Blender shader nodes.

This is necessary because AOV Output nodes don't work with OSL enabled
(Blender bugs #79942, #100282).

The node group outputs:
- Sky: RGB radiance of the sky
- Transmittance: RGB transmittance to top of atmosphere
- Inscatter: RGB scattered light (same as sky without sun disk)

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

SKY_NODE_GROUP_NAME = "Helios_Sky"
SKY_NODE_VERSION = 7  # Increment this to verify code changes are picked up


# =============================================================================
# HELPER CLASS FOR NODE CREATION
# =============================================================================

class NodeBuilder:
    """Helper class for building shader node trees."""
    
    def __init__(self, node_tree):
        self.nodes = node_tree.nodes
        self.links = node_tree.links
    
    def math(self, operation, x, y, label=None, v0=None, v1=None, v2=None):
        """Create a Math node."""
        node = self.nodes.new('ShaderNodeMath')
        node.operation = operation
        node.location = (x, y)
        if label:
            node.label = label
        if v0 is not None:
            node.inputs[0].default_value = v0
        if v1 is not None:
            node.inputs[1].default_value = v1
        if v2 is not None and len(node.inputs) > 2:
            node.inputs[2].default_value = v2
        return node
    
    def vec_math(self, operation, x, y, label=None):
        """Create a Vector Math node."""
        node = self.nodes.new('ShaderNodeVectorMath')
        node.operation = operation
        node.location = (x, y)
        if label:
            node.label = label
        return node
    
    def combine_xyz(self, x, y, label=None):
        """Create a Combine XYZ node."""
        node = self.nodes.new('ShaderNodeCombineXYZ')
        node.location = (x, y)
        if label:
            node.label = label
        return node
    
    def separate_xyz(self, x, y, label=None):
        """Create a Separate XYZ node."""
        node = self.nodes.new('ShaderNodeSeparateXYZ')
        node.location = (x, y)
        if label:
            node.label = label
        return node
    
    def combine_color(self, x, y, label=None):
        """Create a Combine Color node."""
        node = self.nodes.new('ShaderNodeCombineColor')
        node.location = (x, y)
        if label:
            node.label = label
        return node
    
    def separate_color(self, x, y, label=None):
        """Create a Separate Color node."""
        node = self.nodes.new('ShaderNodeSeparateColor')
        node.location = (x, y)
        if label:
            node.label = label
        return node
    
    def mix(self, data_type='RGBA', blend_type='MIX', x=None, y=None, label=None):
        """Create a Mix node."""
        node = self.nodes.new('ShaderNodeMix')
        node.data_type = data_type
        node.blend_type = blend_type
        node.location = (x, y)
        if label:
            node.label = label
        return node
    
    def image_texture(self, x, y, label=None, filepath=None):
        """Create an Image Texture node."""
        node = self.nodes.new('ShaderNodeTexImage')
        node.location = (x, y)
        node.interpolation = 'Linear'
        node.extension = 'EXTEND'
        if label:
            node.label = label
        if filepath and os.path.exists(filepath):
            img = bpy.data.images.load(filepath, check_existing=True)
            img.colorspace_settings.name = 'Non-Color'
            node.image = img
        return node
    
    def value(self, x, y, label=None, val=0.0):
        """Create a Value node."""
        node = self.nodes.new('ShaderNodeValue')
        node.location = (x, y)
        node.outputs[0].default_value = val
        if label:
            node.label = label
        return node
    
    def rgb(self, x, y, label=None, color=(1, 1, 1, 1)):
        """Create an RGB node."""
        node = self.nodes.new('ShaderNodeRGB')
        node.location = (x, y)
        node.outputs[0].default_value = color
        if label:
            node.label = label
        return node
    
    def link(self, from_socket, to_socket):
        """Create a link between sockets."""
        self.links.new(from_socket, to_socket)
        return self


# =============================================================================
# MAIN NODE GROUP CREATION
# =============================================================================

def create_sky_node_group(lut_dir=None):
    """
    Create the Helios Sky node group.
    
    This implements the full Bruneton GetSkyRadiance algorithm for world shaders.
    """
    if lut_dir is None:
        lut_dir = get_lut_cache_dir()
    
    # Remove existing group if present
    if SKY_NODE_GROUP_NAME in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups[SKY_NODE_GROUP_NAME])
    
    # Create new node group
    group = bpy.data.node_groups.new(SKY_NODE_GROUP_NAME, 'ShaderNodeTree')
    builder = NodeBuilder(group)
    
    # =========================================================================
    # INTERFACE - Inputs and Outputs
    # =========================================================================
    
    group_input = group.nodes.new('NodeGroupInput')
    group_input.location = (-2000, 0)
    
    group_output = group.nodes.new('NodeGroupOutput')
    group_output.location = (3500, 0)
    
    # Create interface sockets (Blender 4.0+ API)
    if hasattr(group, 'interface'):
        # Inputs
        group.interface.new_socket('View_Direction', in_out='INPUT', socket_type='NodeSocketVector')
        group.interface.new_socket('Sun_Direction', in_out='INPUT', socket_type='NodeSocketVector')
        group.interface.new_socket('Camera_Position', in_out='INPUT', socket_type='NodeSocketVector')
        group.interface.new_socket('Bottom_Radius', in_out='INPUT', socket_type='NodeSocketFloat')
        group.interface.new_socket('Top_Radius', in_out='INPUT', socket_type='NodeSocketFloat')
        group.interface.new_socket('Mie_Phase_G', in_out='INPUT', socket_type='NodeSocketFloat')
        group.interface.new_socket('Sun_Intensity', in_out='INPUT', socket_type='NodeSocketFloat')
        group.interface.new_socket('Exposure', in_out='INPUT', socket_type='NodeSocketFloat')
        group.interface.new_socket('Sun_Angular_Radius', in_out='INPUT', socket_type='NodeSocketFloat')
        group.interface.new_socket('Solar_Irradiance', in_out='INPUT', socket_type='NodeSocketColor')
        group.interface.new_socket('Add_Sun_Disk', in_out='INPUT', socket_type='NodeSocketFloat')
        
        # Outputs
        group.interface.new_socket('Sky', in_out='OUTPUT', socket_type='NodeSocketColor')
        group.interface.new_socket('Transmittance', in_out='OUTPUT', socket_type='NodeSocketColor')
        group.interface.new_socket('Inscatter', in_out='OUTPUT', socket_type='NodeSocketColor')
    else:
        # Blender 3.x fallback
        group.inputs.new('NodeSocketVector', 'View_Direction')
        group.inputs.new('NodeSocketVector', 'Sun_Direction')
        group.inputs.new('NodeSocketVector', 'Camera_Position')
        group.inputs.new('NodeSocketFloat', 'Bottom_Radius')
        group.inputs.new('NodeSocketFloat', 'Top_Radius')
        group.inputs.new('NodeSocketFloat', 'Mie_Phase_G')
        group.inputs.new('NodeSocketFloat', 'Sun_Intensity')
        group.inputs.new('NodeSocketFloat', 'Exposure')
        group.inputs.new('NodeSocketFloat', 'Sun_Angular_Radius')
        group.inputs.new('NodeSocketColor', 'Solar_Irradiance')
        group.inputs.new('NodeSocketFloat', 'Add_Sun_Disk')
        
        group.outputs.new('NodeSocketColor', 'Sky')
        group.outputs.new('NodeSocketColor', 'Transmittance')
        group.outputs.new('NodeSocketColor', 'Inscatter')
    
    # Set default values
    for socket in group_input.outputs:
        if socket.name == 'Bottom_Radius':
            socket.default_value = 6360.0
        elif socket.name == 'Top_Radius':
            socket.default_value = 6420.0
        elif socket.name == 'Mie_Phase_G':
            socket.default_value = 0.8
        elif socket.name == 'Sun_Intensity':
            socket.default_value = 1.0
        elif socket.name == 'Exposure':
            socket.default_value = 10.0
        elif socket.name == 'Sun_Angular_Radius':
            socket.default_value = 0.004675  # ~0.27 degrees (matches SUN_ANGULAR_RADIUS)
        elif socket.name == 'Solar_Irradiance':
            socket.default_value = (1.474, 1.8504, 1.91198, 1.0)  # Precomputed RGB
        elif socket.name == 'Add_Sun_Disk':
            socket.default_value = 1.0  # Enabled by default
    
    # =========================================================================
    # LOAD LUT TEXTURES
    # =========================================================================
    
    transmittance_path = os.path.join(lut_dir, "transmittance.exr")
    
    tex_transmittance = builder.image_texture(-1600, 400, 'Transmittance_LUT', transmittance_path)
    # Note: Scattering textures created later for proper 4-sample interpolation
    
    # =========================================================================
    # ATMOSPHERE CONSTANTS
    # =========================================================================
    
    BOTTOM_RADIUS = 6360.0
    TOP_RADIUS = 6420.0
    MU_S_MIN = -0.2
    H = math.sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS)
    
    # =========================================================================
    # CAMERA PARAMETERS (r, mu, mu_s, nu)
    # =========================================================================
    
    # r = length(camera_position)
    r = builder.vec_math('LENGTH', -1400, 0, 'r')
    builder.link(group_input.outputs['Camera_Position'], r.inputs[0])
    
    # Clamp r to atmosphere bounds
    r_clamped = builder.math('MINIMUM', -1200, 0, 'r_clamp_max', v1=TOP_RADIUS)
    builder.link(r.outputs['Value'], r_clamped.inputs[0])
    
    r_final = builder.math('MAXIMUM', -1000, 0, 'r_clamp_min', v1=BOTTOM_RADIUS + 0.001)
    builder.link(r_clamped.outputs[0], r_final.inputs[0])
    
    # mu = dot(camera, view_ray) / r
    # But view_ray points FROM camera, so we need dot(camera_normalized, view)
    cam_normalized = builder.vec_math('NORMALIZE', -1400, -100, 'cam_norm')
    builder.link(group_input.outputs['Camera_Position'], cam_normalized.inputs[0])
    
    cam_dot_view = builder.vec_math('DOT_PRODUCT', -1200, -100, 'cam·view')
    builder.link(cam_normalized.outputs[0], cam_dot_view.inputs[0])
    builder.link(group_input.outputs['View_Direction'], cam_dot_view.inputs[1])
    
    mu = builder.math('MULTIPLY', -1000, -100, 'mu', v1=1.0)  # Already normalized
    builder.link(cam_dot_view.outputs['Value'], mu.inputs[0])
    
    # Clamp mu to [-1, 1]
    mu_clamped = builder.math('MINIMUM', -800, -100, 'mu_max', v1=1.0)
    builder.link(mu.outputs[0], mu_clamped.inputs[0])
    
    mu_final = builder.math('MAXIMUM', -600, -100, 'mu_min', v1=-1.0)
    builder.link(mu_clamped.outputs[0], mu_final.inputs[0])
    
    # mu_s = dot(camera_normalized, sun_direction)
    cam_dot_sun = builder.vec_math('DOT_PRODUCT', -1200, -200, 'cam·sun')
    builder.link(cam_normalized.outputs[0], cam_dot_sun.inputs[0])
    builder.link(group_input.outputs['Sun_Direction'], cam_dot_sun.inputs[1])
    
    mu_s = builder.math('MULTIPLY', -1000, -200, 'mu_s', v1=1.0)
    builder.link(cam_dot_sun.outputs['Value'], mu_s.inputs[0])
    
    # Clamp mu_s
    mu_s_clamped = builder.math('MINIMUM', -800, -200, 'mu_s_max', v1=1.0)
    builder.link(mu_s.outputs[0], mu_s_clamped.inputs[0])
    
    mu_s_final = builder.math('MAXIMUM', -600, -200, 'mu_s_min', v1=MU_S_MIN)
    builder.link(mu_s_clamped.outputs[0], mu_s_final.inputs[0])
    
    # nu = dot(view_direction, sun_direction)
    nu = builder.vec_math('DOT_PRODUCT', -1200, -300, 'nu')
    builder.link(group_input.outputs['View_Direction'], nu.inputs[0])
    builder.link(group_input.outputs['Sun_Direction'], nu.inputs[1])
    
    # =========================================================================
    # TRANSMITTANCE UV CALCULATION
    # =========================================================================
    
    # rho = sqrt(r² - bottom_radius²)
    r_sq = builder.math('POWER', -400, 300, 'r²', v1=2.0)
    builder.link(r_final.outputs[0], r_sq.inputs[0])
    
    rho_sq = builder.math('SUBTRACT', -200, 300, 'rho²', v1=BOTTOM_RADIUS * BOTTOM_RADIUS)
    builder.link(r_sq.outputs[0], rho_sq.inputs[0])
    
    rho_sq_safe = builder.math('MAXIMUM', 0, 300, 'rho²_safe', v1=0.0)
    builder.link(rho_sq.outputs[0], rho_sq_safe.inputs[0])
    
    rho = builder.math('SQRT', 200, 300, 'rho')
    builder.link(rho_sq_safe.outputs[0], rho.inputs[0])
    
    # x_r = rho / H
    x_r = builder.math('DIVIDE', 400, 300, 'x_r', v1=H)
    builder.link(rho.outputs[0], x_r.inputs[0])
    
    # d = DistanceToTopAtmosphereBoundary
    # discriminant = r² * (mu² - 1) + top_radius²
    mu_sq = builder.math('POWER', -400, 200, 'mu²', v1=2.0)
    builder.link(mu_final.outputs[0], mu_sq.inputs[0])
    
    mu_sq_m1 = builder.math('SUBTRACT', -200, 200, 'mu²-1', v1=1.0)
    builder.link(mu_sq.outputs[0], mu_sq_m1.inputs[0])
    
    r_sq_mu_term = builder.math('MULTIPLY', 0, 200, 'r²(mu²-1)')
    builder.link(r_sq.outputs[0], r_sq_mu_term.inputs[0])
    builder.link(mu_sq_m1.outputs[0], r_sq_mu_term.inputs[1])
    
    discrim = builder.math('ADD', 200, 200, 'discriminant', v1=TOP_RADIUS * TOP_RADIUS)
    builder.link(r_sq_mu_term.outputs[0], discrim.inputs[0])
    
    discrim_safe = builder.math('MAXIMUM', 400, 200, 'disc_safe', v1=0.0)
    builder.link(discrim.outputs[0], discrim_safe.inputs[0])
    
    discrim_sqrt = builder.math('SQRT', 600, 200, 'sqrt_disc')
    builder.link(discrim_safe.outputs[0], discrim_sqrt.inputs[0])
    
    # d = -r*mu + sqrt(discriminant)
    r_mu = builder.math('MULTIPLY', 0, 100, 'r×mu')
    builder.link(r_final.outputs[0], r_mu.inputs[0])
    builder.link(mu_final.outputs[0], r_mu.inputs[1])
    
    neg_r_mu = builder.math('MULTIPLY', 200, 100, '-r×mu', v1=-1.0)
    builder.link(r_mu.outputs[0], neg_r_mu.inputs[0])
    
    d = builder.math('ADD', 800, 150, 'd')
    builder.link(neg_r_mu.outputs[0], d.inputs[0])
    builder.link(discrim_sqrt.outputs[0], d.inputs[1])
    
    # d_min = top_radius - r, d_max = rho + H
    d_min = builder.math('SUBTRACT', 600, 300, 'd_min', v0=TOP_RADIUS)
    builder.link(r_final.outputs[0], d_min.inputs[1])
    
    d_max = builder.math('ADD', 600, 350, 'd_max', v1=H)
    builder.link(rho.outputs[0], d_max.inputs[0])
    
    # x_mu = (d - d_min) / (d_max - d_min)
    d_minus_dmin = builder.math('SUBTRACT', 1000, 200, 'd-d_min')
    builder.link(d.outputs[0], d_minus_dmin.inputs[0])
    builder.link(d_min.outputs[0], d_minus_dmin.inputs[1])
    
    dmax_minus_dmin = builder.math('SUBTRACT', 1000, 300, 'dmax-dmin')
    builder.link(d_max.outputs[0], dmax_minus_dmin.inputs[0])
    builder.link(d_min.outputs[0], dmax_minus_dmin.inputs[1])
    
    denom_safe = builder.math('MAXIMUM', 1200, 300, 'denom_safe', v1=0.0001)
    builder.link(dmax_minus_dmin.outputs[0], denom_safe.inputs[0])
    
    x_mu = builder.math('DIVIDE', 1400, 250, 'x_mu')
    builder.link(d_minus_dmin.outputs[0], x_mu.inputs[0])
    builder.link(denom_safe.outputs[0], x_mu.inputs[1])
    
    # Clamp x_mu to [0, 1]
    x_mu_clamped = builder.math('MINIMUM', 1600, 250, 'x_mu_max', v1=1.0)
    builder.link(x_mu.outputs[0], x_mu_clamped.inputs[0])
    
    x_mu_final = builder.math('MAXIMUM', 1800, 250, 'x_mu_clamp', v1=0.0)
    builder.link(x_mu_clamped.outputs[0], x_mu_final.inputs[0])
    
    # UV texture coordinates
    u_scale = 1.0 - 1.0 / TRANSMITTANCE_TEXTURE_WIDTH
    u_offset = 0.5 / TRANSMITTANCE_TEXTURE_WIDTH
    v_scale = 1.0 - 1.0 / TRANSMITTANCE_TEXTURE_HEIGHT
    v_offset = 0.5 / TRANSMITTANCE_TEXTURE_HEIGHT
    
    trans_u_scaled = builder.math('MULTIPLY', 2000, 250, 'u_scaled', v1=u_scale)
    builder.link(x_mu_final.outputs[0], trans_u_scaled.inputs[0])
    
    trans_u = builder.math('ADD', 2200, 250, 'trans_u', v1=u_offset)
    builder.link(trans_u_scaled.outputs[0], trans_u.inputs[0])
    
    trans_v_scaled = builder.math('MULTIPLY', 2000, 350, 'v_scaled', v1=v_scale)
    builder.link(x_r.outputs[0], trans_v_scaled.inputs[0])
    
    trans_v = builder.math('ADD', 2200, 350, 'trans_v', v1=v_offset)
    builder.link(trans_v_scaled.outputs[0], trans_v.inputs[0])
    
    # Combine UV and sample transmittance
    trans_uv = builder.combine_xyz(2400, 300, 'Trans_UV')
    builder.link(trans_u.outputs[0], trans_uv.inputs['X'])
    builder.link(trans_v.outputs[0], trans_uv.inputs['Y'])
    
    builder.link(trans_uv.outputs[0], tex_transmittance.inputs['Vector'])
    
    # =========================================================================
    # SCATTERING UV CALCULATION (simplified - single slice for now)
    # =========================================================================
    
    # u_r for scattering
    scat_u_r_raw = builder.math('DIVIDE', -400, -400, 'scat_u_r', v1=H)
    builder.link(rho.outputs[0], scat_u_r_raw.inputs[0])
    
    # u_mu for scattering (using same x_mu but scaled for scattering texture)
    # For sky viewing (not intersecting ground), u_mu = 0.5 + 0.5 * x_mu_scaled
    scat_mu_scale = 1.0 - 1.0 / (SCATTERING_TEXTURE_MU_SIZE / 2)
    scat_mu_offset = 0.5 / (SCATTERING_TEXTURE_MU_SIZE / 2)
    
    x_mu_scat_scaled = builder.math('MULTIPLY', -200, -400, 'x_mu_scat', v1=scat_mu_scale)
    builder.link(x_mu_final.outputs[0], x_mu_scat_scaled.inputs[0])
    
    x_mu_scat_offset = builder.math('ADD', 0, -400, 'x_mu_scat_off', v1=scat_mu_offset)
    builder.link(x_mu_scat_scaled.outputs[0], x_mu_scat_offset.inputs[0])
    
    u_mu_scat = builder.math('MULTIPLY', 200, -400, 'u_mu_scat', v0=0.5)
    builder.link(x_mu_scat_offset.outputs[0], u_mu_scat.inputs[1])
    
    u_mu_scat_final = builder.math('ADD', 400, -400, 'u_mu_final', v0=0.5)
    builder.link(u_mu_scat.outputs[0], u_mu_scat_final.inputs[1])
    
    # u_mu_s for scattering - Bruneton non-linear mapping
    # d = DistanceToTopAtmosphereBoundary(bottom_radius, mu_s, top_radius)
    #   = -bottom_radius*mu_s + sqrt(bottom_radius²*(mu_s²-1) + top_radius²)
    # a = (d - d_min) / (d_max - d_min) where d_min = top-bottom, d_max = H
    # x_mu_s = max(1 - a/A, 0) / (1 + a) where A is computed for mu_s_min
    
    # Compute d = DistanceToTopAtmosphereBoundary(bottom_radius, mu_s, top_radius)
    mu_s_sq = builder.math('POWER', -600, -500, 'mu_s²', v1=2.0)
    builder.link(mu_s_final.outputs[0], mu_s_sq.inputs[0])
    
    mu_s_sq_m1 = builder.math('SUBTRACT', -400, -500, 'mu_s²-1', v1=1.0)
    builder.link(mu_s_sq.outputs[0], mu_s_sq_m1.inputs[0])
    
    # bottom_radius² * (mu_s² - 1) + top_radius²
    br_sq_term = builder.math('MULTIPLY', -200, -500, 'br²×term', v0=BOTTOM_RADIUS * BOTTOM_RADIUS)
    builder.link(mu_s_sq_m1.outputs[0], br_sq_term.inputs[1])
    
    d_discrim = builder.math('ADD', 0, -500, 'd_discrim', v1=TOP_RADIUS * TOP_RADIUS)
    builder.link(br_sq_term.outputs[0], d_discrim.inputs[0])
    
    d_discrim_safe = builder.math('MAXIMUM', 200, -500, 'd_disc_safe', v1=0.0)
    builder.link(d_discrim.outputs[0], d_discrim_safe.inputs[0])
    
    d_sqrt = builder.math('SQRT', 400, -500, 'd_sqrt')
    builder.link(d_discrim_safe.outputs[0], d_sqrt.inputs[0])
    
    # -bottom_radius * mu_s
    neg_br_mus = builder.math('MULTIPLY', 200, -550, '-br×mu_s', v0=-BOTTOM_RADIUS)
    builder.link(mu_s_final.outputs[0], neg_br_mus.inputs[1])
    
    # d = -br*mu_s + sqrt(discriminant)
    d_mus = builder.math('ADD', 600, -500, 'd_mus')
    builder.link(neg_br_mus.outputs[0], d_mus.inputs[0])
    builder.link(d_sqrt.outputs[0], d_mus.inputs[1])
    
    d_mus_safe = builder.math('MAXIMUM', 800, -500, 'd_mus_safe', v1=0.0)
    builder.link(d_mus.outputs[0], d_mus_safe.inputs[0])
    
    # Constants for mapping
    d_mus_min = TOP_RADIUS - BOTTOM_RADIUS  # = 60 km
    d_mus_max = H  # = sqrt(top² - bottom²)
    
    # Precompute A for mu_s_min
    D_const = -BOTTOM_RADIUS * MU_S_MIN + math.sqrt(
        BOTTOM_RADIUS * BOTTOM_RADIUS * (MU_S_MIN * MU_S_MIN - 1.0) + TOP_RADIUS * TOP_RADIUS)
    A_const = (D_const - d_mus_min) / (d_mus_max - d_mus_min)
    
    # a = (d - d_min) / (d_max - d_min)
    d_mus_shifted = builder.math('SUBTRACT', 1000, -500, 'd-dmin', v1=d_mus_min)
    builder.link(d_mus_safe.outputs[0], d_mus_shifted.inputs[0])
    
    a_val = builder.math('DIVIDE', 1200, -500, 'a', v1=(d_mus_max - d_mus_min))
    builder.link(d_mus_shifted.outputs[0], a_val.inputs[0])
    
    # x_mu_s = max(1 - a/A, 0) / (1 + a)
    a_over_A = builder.math('DIVIDE', 1400, -500, 'a/A', v1=A_const)
    builder.link(a_val.outputs[0], a_over_A.inputs[0])
    
    one_minus_aA = builder.math('SUBTRACT', 1600, -500, '1-a/A', v0=1.0)
    builder.link(a_over_A.outputs[0], one_minus_aA.inputs[1])
    
    one_minus_aA_safe = builder.math('MAXIMUM', 1800, -500, 'max(1-a/A,0)', v1=0.0)
    builder.link(one_minus_aA.outputs[0], one_minus_aA_safe.inputs[0])
    
    one_plus_a = builder.math('ADD', 1600, -550, '1+a', v0=1.0)
    builder.link(a_val.outputs[0], one_plus_a.inputs[1])
    
    x_mu_s = builder.math('DIVIDE', 2000, -500, 'x_mu_s')
    builder.link(one_minus_aA_safe.outputs[0], x_mu_s.inputs[0])
    builder.link(one_plus_a.outputs[0], x_mu_s.inputs[1])
    
    # Apply GetTextureCoordFromUnitRange
    mu_s_tex_scale = 1.0 - 1.0 / SCATTERING_TEXTURE_MU_S_SIZE
    mu_s_tex_offset = 0.5 / SCATTERING_TEXTURE_MU_S_SIZE
    
    u_mu_s_scaled = builder.math('MULTIPLY', 2200, -500, 'u_mu_s_sc', v1=mu_s_tex_scale)
    builder.link(x_mu_s.outputs[0], u_mu_s_scaled.inputs[0])
    
    u_mu_s = builder.math('ADD', 2400, -500, 'u_mu_s', v1=mu_s_tex_offset)
    builder.link(u_mu_s_scaled.outputs[0], u_mu_s.inputs[0])
    
    # u_nu (view-sun angle)
    nu_plus1 = builder.math('ADD', -400, -600, 'nu+1', v0=1.0)
    builder.link(nu.outputs['Value'], nu_plus1.inputs[1])
    
    x_nu = builder.math('MULTIPLY', -200, -600, 'x_nu', v1=0.5)
    builder.link(nu_plus1.outputs[0], x_nu.inputs[0])
    
    # Texture coord for nu with interpolation
    tex_coord_x = builder.math('MULTIPLY', 0, -600, 'tex_x', v1=float(SCATTERING_TEXTURE_NU_SIZE - 1))
    builder.link(x_nu.outputs[0], tex_coord_x.inputs[0])
    
    tex_x_floor = builder.math('FLOOR', 200, -600, 'tex_x_floor')
    builder.link(tex_coord_x.outputs[0], tex_x_floor.inputs[0])
    
    lerp_factor = builder.math('SUBTRACT', 400, -600, 'lerp_fac')
    builder.link(tex_coord_x.outputs[0], lerp_factor.inputs[0])
    builder.link(tex_x_floor.outputs[0], lerp_factor.inputs[1])
    
    # uvw0_x = (tex_x + u_mu_s) / NU_SIZE
    tex_x_plus_mus = builder.math('ADD', 600, -600, 'tex_x+mus')
    builder.link(tex_x_floor.outputs[0], tex_x_plus_mus.inputs[0])
    builder.link(u_mu_s.outputs[0], tex_x_plus_mus.inputs[1])
    
    uvw0_x = builder.math('DIVIDE', 800, -600, 'uvw0_x', v1=float(SCATTERING_TEXTURE_NU_SIZE))
    builder.link(tex_x_plus_mus.outputs[0], uvw0_x.inputs[0])
    
    # uvw1_x = (tex_x + 1 + u_mu_s) / NU_SIZE
    tex_x_plus1 = builder.math('ADD', 600, -700, 'tex_x+1', v1=1.0)
    builder.link(tex_x_floor.outputs[0], tex_x_plus1.inputs[0])
    
    tex_x_plus1_mus = builder.math('ADD', 800, -700, 'tex_x+1+mus')
    builder.link(tex_x_plus1.outputs[0], tex_x_plus1_mus.inputs[0])
    builder.link(u_mu_s.outputs[0], tex_x_plus1_mus.inputs[1])
    
    uvw1_x = builder.math('DIVIDE', 1000, -700, 'uvw1_x', v1=float(SCATTERING_TEXTURE_NU_SIZE))
    builder.link(tex_x_plus1_mus.outputs[0], uvw1_x.inputs[0])
    
    # =========================================================================
    # TILED 3D TEXTURE SAMPLING WITH DEPTH (u_r) INTERPOLATION
    # =========================================================================
    # The scattering texture is 3D stored as horizontally tiled 2D:
    # - Layout: depth slices side by side, total width = slice_width * DEPTH
    # - Each slice has width = NU_SIZE * MU_S_SIZE = 256
    # - We need to interpolate across both nu (already computed) and depth (u_r)
    
    # Apply GetTextureCoordFromUnitRange to u_r
    r_tex_scale = 1.0 - 1.0 / SCATTERING_TEXTURE_R_SIZE
    r_tex_offset = 0.5 / SCATTERING_TEXTURE_R_SIZE
    
    u_r_scaled = builder.math('MULTIPLY', -200, -350, 'u_r_scaled', v1=r_tex_scale)
    builder.link(scat_u_r_raw.outputs[0], u_r_scaled.inputs[0])
    
    u_r = builder.math('ADD', 0, -350, 'u_r', v1=r_tex_offset)
    builder.link(u_r_scaled.outputs[0], u_r.inputs[0])
    
    # Clamp u_r to [0, 1]
    u_r_clamped = builder.math('MINIMUM', 200, -350, 'u_r_max', v1=1.0)
    builder.link(u_r.outputs[0], u_r_clamped.inputs[0])
    
    u_r_final = builder.math('MAXIMUM', 400, -350, 'u_r_clamp', v1=0.0)
    builder.link(u_r_clamped.outputs[0], u_r_final.inputs[0])
    
    # Compute depth slice indices for interpolation
    # w_scaled = u_r * (DEPTH - 1)
    depth_scaled = builder.math('MULTIPLY', 600, -350, 'depth_scaled', 
                                 v1=float(SCATTERING_TEXTURE_DEPTH - 1))
    builder.link(u_r_final.outputs[0], depth_scaled.inputs[0])
    
    depth_floor = builder.math('FLOOR', 800, -350, 'depth_floor')
    builder.link(depth_scaled.outputs[0], depth_floor.inputs[0])
    
    depth_frac = builder.math('SUBTRACT', 1000, -350, 'depth_frac')
    builder.link(depth_scaled.outputs[0], depth_frac.inputs[0])
    builder.link(depth_floor.outputs[0], depth_frac.inputs[1])
    
    # depth_floor + 1, clamped to max
    depth_ceil = builder.math('ADD', 800, -300, 'depth_ceil', v1=1.0)
    builder.link(depth_floor.outputs[0], depth_ceil.inputs[0])
    
    depth_ceil_clamped = builder.math('MINIMUM', 1000, -300, 'depth_ceil_clamp', 
                                       v1=float(SCATTERING_TEXTURE_DEPTH - 1))
    builder.link(depth_ceil.outputs[0], depth_ceil_clamped.inputs[0])
    
    # Compute X coordinates including depth slice offset
    # Final X = (depth_slice + uvw_x) / DEPTH
    # For nu=0 slice:
    slice0_plus_uvw0 = builder.math('ADD', 1200, -600, 'slice0+uvw0')
    builder.link(depth_floor.outputs[0], slice0_plus_uvw0.inputs[0])
    builder.link(uvw0_x.outputs[0], slice0_plus_uvw0.inputs[1])
    
    final_x0_slice0 = builder.math('DIVIDE', 1400, -600, 'x0_s0', 
                                    v1=float(SCATTERING_TEXTURE_DEPTH))
    builder.link(slice0_plus_uvw0.outputs[0], final_x0_slice0.inputs[0])
    
    # For nu=1 slice:
    slice0_plus_uvw1 = builder.math('ADD', 1200, -650, 'slice0+uvw1')
    builder.link(depth_floor.outputs[0], slice0_plus_uvw1.inputs[0])
    builder.link(uvw1_x.outputs[0], slice0_plus_uvw1.inputs[1])
    
    final_x1_slice0 = builder.math('DIVIDE', 1400, -650, 'x1_s0', 
                                    v1=float(SCATTERING_TEXTURE_DEPTH))
    builder.link(slice0_plus_uvw1.outputs[0], final_x1_slice0.inputs[0])
    
    # Second depth slice (ceil)
    slice1_plus_uvw0 = builder.math('ADD', 1200, -700, 'slice1+uvw0')
    builder.link(depth_ceil_clamped.outputs[0], slice1_plus_uvw0.inputs[0])
    builder.link(uvw0_x.outputs[0], slice1_plus_uvw0.inputs[1])
    
    final_x0_slice1 = builder.math('DIVIDE', 1400, -700, 'x0_s1', 
                                    v1=float(SCATTERING_TEXTURE_DEPTH))
    builder.link(slice1_plus_uvw0.outputs[0], final_x0_slice1.inputs[0])
    
    slice1_plus_uvw1 = builder.math('ADD', 1200, -750, 'slice1+uvw1')
    builder.link(depth_ceil_clamped.outputs[0], slice1_plus_uvw1.inputs[0])
    builder.link(uvw1_x.outputs[0], slice1_plus_uvw1.inputs[1])
    
    final_x1_slice1 = builder.math('DIVIDE', 1400, -750, 'x1_s1', 
                                    v1=float(SCATTERING_TEXTURE_DEPTH))
    builder.link(slice1_plus_uvw1.outputs[0], final_x1_slice1.inputs[0])
    
    # Flip Y coordinate - Blender Image Texture uses opposite V convention from OSL
    # OSL: V=0 at bottom, V=1 at top
    # Blender: V=0 at top, V=1 at bottom (for EXR images)
    u_mu_flipped = builder.math('SUBTRACT', 1500, -400, 'u_mu_flip', v0=1.0)
    builder.link(u_mu_scat_final.outputs[0], u_mu_flipped.inputs[1])
    
    # Create UV coordinates for all 4 samples (2 nu × 2 depth)
    scat_uv_nu0_d0 = builder.combine_xyz(1600, -600, 'UV_nu0_d0')
    builder.link(final_x0_slice0.outputs[0], scat_uv_nu0_d0.inputs['X'])
    builder.link(u_mu_flipped.outputs[0], scat_uv_nu0_d0.inputs['Y'])
    
    scat_uv_nu1_d0 = builder.combine_xyz(1600, -650, 'UV_nu1_d0')
    builder.link(final_x1_slice0.outputs[0], scat_uv_nu1_d0.inputs['X'])
    builder.link(u_mu_flipped.outputs[0], scat_uv_nu1_d0.inputs['Y'])
    
    scat_uv_nu0_d1 = builder.combine_xyz(1600, -700, 'UV_nu0_d1')
    builder.link(final_x0_slice1.outputs[0], scat_uv_nu0_d1.inputs['X'])
    builder.link(u_mu_flipped.outputs[0], scat_uv_nu0_d1.inputs['Y'])
    
    scat_uv_nu1_d1 = builder.combine_xyz(1600, -750, 'UV_nu1_d1')
    builder.link(final_x1_slice1.outputs[0], scat_uv_nu1_d1.inputs['X'])
    builder.link(u_mu_flipped.outputs[0], scat_uv_nu1_d1.inputs['Y'])
    
    # Need 4 texture nodes for bilinear interpolation
    tex_scat_nu0_d0 = builder.image_texture(1800, -550, 'Scat_nu0_d0', 
                                             os.path.join(lut_dir, "scattering.exr"))
    tex_scat_nu1_d0 = builder.image_texture(1800, -650, 'Scat_nu1_d0', 
                                             os.path.join(lut_dir, "scattering.exr"))
    tex_scat_nu0_d1 = builder.image_texture(1800, -750, 'Scat_nu0_d1', 
                                             os.path.join(lut_dir, "scattering.exr"))
    tex_scat_nu1_d1 = builder.image_texture(1800, -850, 'Scat_nu1_d1', 
                                             os.path.join(lut_dir, "scattering.exr"))
    
    builder.link(scat_uv_nu0_d0.outputs[0], tex_scat_nu0_d0.inputs['Vector'])
    builder.link(scat_uv_nu1_d0.outputs[0], tex_scat_nu1_d0.inputs['Vector'])
    builder.link(scat_uv_nu0_d1.outputs[0], tex_scat_nu0_d1.inputs['Vector'])
    builder.link(scat_uv_nu1_d1.outputs[0], tex_scat_nu1_d1.inputs['Vector'])
    
    # Interpolate: first along nu for each depth slice
    scat_nu_interp_d0 = builder.mix('RGBA', 'MIX', 2100, -600, 'Scat_nu_d0')
    builder.link(lerp_factor.outputs[0], scat_nu_interp_d0.inputs[0])
    builder.link(tex_scat_nu0_d0.outputs['Color'], scat_nu_interp_d0.inputs[6])
    builder.link(tex_scat_nu1_d0.outputs['Color'], scat_nu_interp_d0.inputs[7])
    
    scat_nu_interp_d1 = builder.mix('RGBA', 'MIX', 2100, -750, 'Scat_nu_d1')
    builder.link(lerp_factor.outputs[0], scat_nu_interp_d1.inputs[0])
    builder.link(tex_scat_nu0_d1.outputs['Color'], scat_nu_interp_d1.inputs[6])
    builder.link(tex_scat_nu1_d1.outputs['Color'], scat_nu_interp_d1.inputs[7])
    
    # Then interpolate along depth
    scat_interp = builder.mix('RGBA', 'MIX', 2300, -675, 'Scattering_Final')
    builder.link(depth_frac.outputs[0], scat_interp.inputs[0])
    builder.link(scat_nu_interp_d0.outputs[2], scat_interp.inputs[6])
    builder.link(scat_nu_interp_d1.outputs[2], scat_interp.inputs[7])
    
    # =========================================================================
    # PHASE FUNCTIONS
    # =========================================================================
    
    # Rayleigh phase: 3/(16π) × (1 + nu²)
    nu_sq = builder.math('POWER', 1800, -400, 'nu²', v1=2.0)
    builder.link(nu.outputs['Value'], nu_sq.inputs[0])
    
    ray_ph_term = builder.math('ADD', 2000, -400, '1+nu²', v0=1.0)
    builder.link(nu_sq.outputs[0], ray_ph_term.inputs[1])
    
    rayleigh_phase = builder.math('MULTIPLY', 2200, -400, 'Ray_Phase', 
                                   v0=3.0 / (16.0 * math.pi))
    builder.link(ray_ph_term.outputs[0], rayleigh_phase.inputs[1])
    
    # Mie phase (using fixed g=0.8)
    G = 0.8
    mie_k = 3.0 / (8.0 * math.pi) * (1.0 - G*G) / (2.0 + G*G)
    
    two_g_nu = builder.math('MULTIPLY', 1800, -500, '2g×nu', v0=2.0 * G)
    builder.link(nu.outputs['Value'], two_g_nu.inputs[1])
    
    denom_base = builder.math('SUBTRACT', 2000, -500, '1+g²-2gν', v0=1.0 + G*G)
    builder.link(two_g_nu.outputs[0], denom_base.inputs[1])
    
    denom_pow = builder.math('POWER', 2200, -500, 'denom^1.5', v1=1.5)
    builder.link(denom_base.outputs[0], denom_pow.inputs[0])
    
    mie_num = builder.math('MULTIPLY', 2400, -450, 'k×(1+nu²)', v0=mie_k)
    builder.link(ray_ph_term.outputs[0], mie_num.inputs[1])
    
    mie_phase = builder.math('DIVIDE', 2600, -450, 'Mie_Phase')
    builder.link(mie_num.outputs[0], mie_phase.inputs[0])
    builder.link(denom_pow.outputs[0], mie_phase.inputs[1])
    
    # =========================================================================
    # EXTRACT MIE SCATTERING (same extrapolation as OSL)
    # =========================================================================
    # OSL extrapolates single Mie from combined scattering texture.
    # Formula: mie = scattering * (rayleigh[0]/mie[0]) * (mie/rayleigh)
    # With default coefficients: rayleigh = (0.0058, 0.0135, 0.0331), mie = (0.004, 0.004, 0.004)
    # Resulting factors: (1.0, 0.43, 0.176)
    
    sep_scat = builder.separate_color(2500, -600, 'Sep_Scattering')
    builder.link(scat_interp.outputs[2], sep_scat.inputs['Color'])
    
    mie_r = builder.math('MULTIPLY', 2700, -550, 'Mie_R', v1=1.0)
    builder.link(sep_scat.outputs['Red'], mie_r.inputs[0])
    
    mie_g = builder.math('MULTIPLY', 2700, -600, 'Mie_G', v1=0.43)
    builder.link(sep_scat.outputs['Green'], mie_g.inputs[0])
    
    mie_b = builder.math('MULTIPLY', 2700, -650, 'Mie_B', v1=0.176)
    builder.link(sep_scat.outputs['Blue'], mie_b.inputs[0])
    
    mie_color = builder.combine_color(2900, -600, 'Mie_Scattering')
    builder.link(mie_r.outputs[0], mie_color.inputs['Red'])
    builder.link(mie_g.outputs[0], mie_color.inputs['Green'])
    builder.link(mie_b.outputs[0], mie_color.inputs['Blue'])
    
    # =========================================================================
    # APPLY PHASE FUNCTIONS TO SCATTERING COMPONENTS
    # =========================================================================
    
    # Rayleigh component: scattering * Rayleigh_phase
    rayleigh_contrib = builder.vec_math('SCALE', 3100, -450, 'Rayleigh_Contrib')
    builder.link(scat_interp.outputs[2], rayleigh_contrib.inputs[0])
    builder.link(rayleigh_phase.outputs[0], rayleigh_contrib.inputs['Scale'])
    
    # Mie component: mie_scattering * Mie_phase
    mie_contrib = builder.vec_math('SCALE', 3100, -550, 'Mie_Contrib')
    builder.link(mie_color.outputs['Color'], mie_contrib.inputs[0])
    builder.link(mie_phase.outputs[0], mie_contrib.inputs['Scale'])
    
    # Total radiance = Rayleigh + Mie
    radiance = builder.vec_math('ADD', 3300, -500, 'Radiance')
    builder.link(rayleigh_contrib.outputs[0], radiance.inputs[0])
    builder.link(mie_contrib.outputs[0], radiance.inputs[1])
    
    # Apply sun intensity
    radiance_scaled = builder.vec_math('SCALE', 3500, -500, 'Radiance_Scaled')
    builder.link(radiance.outputs[0], radiance_scaled.inputs[0])
    builder.link(group_input.outputs['Sun_Intensity'], radiance_scaled.inputs['Scale'])
    
    # =========================================================================
    # SUN DISK RENDERING
    # =========================================================================
    # Add sun disk if ray points at sun: cos(view, sun) > cos(sun_angular_radius)
    # Sun radiance = solar_irradiance / (PI * sun_angular_radius^2)
    
    # cos(angle) = dot(view_direction, sun_direction)
    cos_sun_angle = builder.vec_math('DOT_PRODUCT', 3600, -200, 'Cos_Sun_Angle')
    builder.link(group_input.outputs['View_Direction'], cos_sun_angle.inputs[0])
    builder.link(group_input.outputs['Sun_Direction'], cos_sun_angle.inputs[1])
    
    # cos(sun_angular_radius)
    cos_sun_radius = builder.math('COSINE', 3600, -100, 'Cos_Sun_Radius')
    builder.link(group_input.outputs['Sun_Angular_Radius'], cos_sun_radius.inputs[0])
    
    # Inside sun disk: cos_sun_angle > cos_sun_radius
    sun_mask = builder.math('GREATER_THAN', 3800, -150, 'Sun_Mask')
    builder.link(cos_sun_angle.outputs['Value'], sun_mask.inputs[0])
    builder.link(cos_sun_radius.outputs[0], sun_mask.inputs[1])
    
    # Multiply by Add_Sun_Disk toggle
    sun_mask_toggled = builder.math('MULTIPLY', 4000, -150, 'Sun_Mask_Toggle')
    builder.link(sun_mask.outputs[0], sun_mask_toggled.inputs[0])
    builder.link(group_input.outputs['Add_Sun_Disk'], sun_mask_toggled.inputs[1])
    
    # Sun solid angle = PI * sun_angular_radius^2
    sun_radius_sq = builder.math('POWER', 3600, 0, 'Radius_Sq', v1=2.0)
    builder.link(group_input.outputs['Sun_Angular_Radius'], sun_radius_sq.inputs[0])
    
    sun_solid_angle = builder.math('MULTIPLY', 3800, 0, 'Solid_Angle', v0=math.pi)
    builder.link(sun_radius_sq.outputs[0], sun_solid_angle.inputs[1])
    
    # Sun radiance = solar_irradiance / solid_angle
    sun_radiance = builder.vec_math('SCALE', 4000, 0, 'Sun_Radiance_Raw')
    builder.link(group_input.outputs['Solar_Irradiance'], sun_radiance.inputs[0])
    
    # Divide by solid angle (multiply by 1/solid_angle)
    inv_solid_angle = builder.math('DIVIDE', 3900, 50, 'Inv_Solid_Angle', v0=1.0)
    builder.link(sun_solid_angle.outputs[0], inv_solid_angle.inputs[1])
    builder.link(inv_solid_angle.outputs[0], sun_radiance.inputs['Scale'])
    
    # Attenuate sun by transmittance
    sun_attenuated = builder.vec_math('MULTIPLY', 4200, 0, 'Sun_Attenuated')
    builder.link(sun_radiance.outputs[0], sun_attenuated.inputs[0])
    builder.link(tex_transmittance.outputs['Color'], sun_attenuated.inputs[1])
    
    # Apply sun intensity
    sun_with_intensity = builder.vec_math('SCALE', 4400, 0, 'Sun_Intensity')
    builder.link(sun_attenuated.outputs[0], sun_with_intensity.inputs[0])
    builder.link(group_input.outputs['Sun_Intensity'], sun_with_intensity.inputs['Scale'])
    
    # Mask sun contribution
    sun_masked = builder.vec_math('SCALE', 4600, 0, 'Sun_Masked')
    builder.link(sun_with_intensity.outputs[0], sun_masked.inputs[0])
    builder.link(sun_mask_toggled.outputs[0], sun_masked.inputs['Scale'])
    
    # Add sun to radiance
    radiance_with_sun = builder.vec_math('ADD', 4800, -250, 'Radiance_With_Sun')
    builder.link(radiance_scaled.outputs[0], radiance_with_sun.inputs[0])
    builder.link(sun_masked.outputs[0], radiance_with_sun.inputs[1])
    
    # Apply exposure for Sky output
    sky_exposed = builder.vec_math('SCALE', 5000, -250, 'Sky_Exposed')
    builder.link(radiance_with_sun.outputs[0], sky_exposed.inputs[0])
    builder.link(group_input.outputs['Exposure'], sky_exposed.inputs['Scale'])
    
    # =========================================================================
    # OUTPUTS
    # =========================================================================
    
    # Sky = radiance × exposure
    builder.link(sky_exposed.outputs[0], group_output.inputs['Sky'])
    
    # Transmittance from LUT
    builder.link(tex_transmittance.outputs['Color'], group_output.inputs['Transmittance'])
    
    # Inscatter = radiance with sun (without exposure, for compositing)
    builder.link(radiance_with_sun.outputs[0], group_output.inputs['Inscatter'])
    
    print(f"Helios: Created sky node group '{SKY_NODE_GROUP_NAME}' (VERSION {SKY_NODE_VERSION})")
    return group


def get_or_create_sky_node_group(lut_dir=None):
    """Get existing sky node group or create a new one."""
    if SKY_NODE_GROUP_NAME in bpy.data.node_groups:
        return bpy.data.node_groups[SKY_NODE_GROUP_NAME]
    return create_sky_node_group(lut_dir)


# =============================================================================
# REGISTRATION
# =============================================================================

def register():
    pass


def unregister():
    pass
