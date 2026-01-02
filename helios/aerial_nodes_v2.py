"""
Helios Node-Based Bruneton Aerial Perspective - V2

This module creates a shader node group that implements GetSkyRadianceToPoint
from Eric Bruneton's atmospheric scattering, following the reference EXACTLY.

Reference: atmospheric-scattering-2-export/atmosphere/functions.glsl

Key formulas from reference:
  d = length(point - camera)
  r_p = sqrt(d*d + 2*r*mu*d + r*r)  # Law of cosines
  mu_p = (r*mu + d) / r_p
  mu_s_p = (r*mu_s + d*nu) / r_p
  inscatter = S_cam - transmittance * S_point

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

# Atmosphere parameters
BOTTOM_RADIUS = 6360.0  # km
TOP_RADIUS = 6420.0     # km
MU_S_MIN = -0.2


# =============================================================================
# NODE GROUP NAME AND VERSION
# =============================================================================

AERIAL_NODE_GROUP_NAME = "Helios_Aerial_Perspective"
AERIAL_NODE_VERSION = 5  # V2 rewrite based on reference


# =============================================================================
# HELPER CLASS
# =============================================================================

class NodeBuilder:
    """Helper class for building shader node trees."""
    
    def __init__(self, node_tree):
        self.nodes = node_tree.nodes
        self.links = node_tree.links
    
    def math(self, operation, x, y, name, v0=None, v1=None):
        node = self.nodes.new('ShaderNodeMath')
        node.operation = operation
        node.location = (x, y)
        node.name = name
        node.label = name
        if v0 is not None:
            node.inputs[0].default_value = v0
        if v1 is not None:
            node.inputs[1].default_value = v1
        return node
    
    def vec_math(self, operation, x, y, name):
        node = self.nodes.new('ShaderNodeVectorMath')
        node.operation = operation
        node.location = (x, y)
        node.name = name
        node.label = name
        return node
    
    def combine_xyz(self, x, y, name):
        node = self.nodes.new('ShaderNodeCombineXYZ')
        node.location = (x, y)
        node.name = name
        node.label = name
        return node
    
    def separate_xyz(self, x, y, name):
        node = self.nodes.new('ShaderNodeSeparateXYZ')
        node.location = (x, y)
        node.name = name
        node.label = name
        return node
    
    def image_texture(self, x, y, name, filepath=None):
        node = self.nodes.new('ShaderNodeTexImage')
        node.location = (x, y)
        node.name = name
        node.label = name
        node.interpolation = 'Linear'
        node.extension = 'EXTEND'
        if filepath and os.path.exists(filepath):
            img = bpy.data.images.load(filepath, check_existing=True)
            img.colorspace_settings.name = 'Non-Color'
            node.image = img
        return node
    
    def link(self, from_socket, to_socket):
        self.links.new(from_socket, to_socket)
    
    def mix_rgb(self, x, y, name, blend_type='MIX'):
        node = self.nodes.new('ShaderNodeMix')
        node.data_type = 'RGBA'
        node.blend_type = blend_type
        node.location = (x, y)
        node.name = name
        node.label = name
        return node


# =============================================================================
# MAIN NODE GROUP CREATION - Following Reference EXACTLY
# =============================================================================

def create_aerial_perspective_node_group(lut_dir=None):
    """
    Create the Helios Aerial Perspective node group.
    
    Implements GetSkyRadianceToPoint from Bruneton reference.
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
    # INTERFACE
    # =========================================================================
    
    group_input = group.nodes.new('NodeGroupInput')
    group_input.location = (-2000, 0)
    
    group_output = group.nodes.new('NodeGroupOutput')
    group_output.location = (4000, 0)
    
    # Inputs
    group.interface.new_socket('Position', in_out='INPUT', socket_type='NodeSocketVector')
    group.interface.new_socket('Camera_Position', in_out='INPUT', socket_type='NodeSocketVector')
    group.interface.new_socket('Sun_Direction', in_out='INPUT', socket_type='NodeSocketVector')
    group.interface.new_socket('Planet_Center', in_out='INPUT', socket_type='NodeSocketVector')
    group.interface.new_socket('Scene_Scale', in_out='INPUT', socket_type='NodeSocketFloat')
    
    # Outputs
    group.interface.new_socket('Transmittance', in_out='OUTPUT', socket_type='NodeSocketColor')
    group.interface.new_socket('Inscatter', in_out='OUTPUT', socket_type='NodeSocketColor')
    
    # Set defaults
    for socket in group.interface.items_tree:
        if socket.name == 'Scene_Scale':
            socket.default_value = 0.001
    
    # =========================================================================
    # LUT TEXTURES
    # =========================================================================
    
    transmittance_path = os.path.join(lut_dir, "transmittance.exr")
    scattering_path = os.path.join(lut_dir, "scattering.exr")
    
    tex_transmittance = builder.image_texture(-1800, 400, 'Transmittance_LUT', transmittance_path)
    tex_scattering_cam = builder.image_texture(-1800, 200, 'Scattering_LUT_Cam', scattering_path)
    tex_scattering_pt = builder.image_texture(-1800, 0, 'Scattering_LUT_Point', scattering_path)
    
    # =========================================================================
    # COORDINATE TRANSFORMS
    # Reference: camera and point are relative to earth center
    # =========================================================================
    
    # camera_km = (Camera_Position - Planet_Center) * Scene_Scale
    cam_minus_center = builder.vec_math('SUBTRACT', -1600, 100, 'Cam-Center')
    builder.link(group_input.outputs['Camera_Position'], cam_minus_center.inputs[0])
    builder.link(group_input.outputs['Planet_Center'], cam_minus_center.inputs[1])
    
    camera_km = builder.vec_math('SCALE', -1400, 100, 'Camera_km')
    builder.link(cam_minus_center.outputs[0], camera_km.inputs[0])
    builder.link(group_input.outputs['Scene_Scale'], camera_km.inputs['Scale'])
    
    # point_km = (Position - Planet_Center) * Scene_Scale
    pt_minus_center = builder.vec_math('SUBTRACT', -1600, -100, 'Pt-Center')
    builder.link(group_input.outputs['Position'], pt_minus_center.inputs[0])
    builder.link(group_input.outputs['Planet_Center'], pt_minus_center.inputs[1])
    
    point_km = builder.vec_math('SCALE', -1400, -100, 'Point_km')
    builder.link(pt_minus_center.outputs[0], point_km.inputs[0])
    builder.link(group_input.outputs['Scene_Scale'], point_km.inputs['Scale'])
    
    # =========================================================================
    # VIEW RAY AND DISTANCE
    # Reference: view_ray = normalize(point - camera)
    #            d = length(point - camera)
    # =========================================================================
    
    pt_minus_cam = builder.vec_math('SUBTRACT', -1200, 0, 'Point-Camera')
    builder.link(point_km.outputs[0], pt_minus_cam.inputs[0])
    builder.link(camera_km.outputs[0], pt_minus_cam.inputs[1])
    
    view_ray = builder.vec_math('NORMALIZE', -1000, 50, 'View_Ray')
    builder.link(pt_minus_cam.outputs[0], view_ray.inputs[0])
    
    d = builder.vec_math('LENGTH', -1000, -50, 'd')
    builder.link(pt_minus_cam.outputs[0], d.inputs[0])
    
    # =========================================================================
    # CAMERA PARAMETERS (r, mu, mu_s, nu)
    # Reference: r = length(camera)
    #            mu = dot(camera, view_ray) / r
    #            mu_s = dot(camera, sun_direction) / r
    #            nu = dot(view_ray, sun_direction)
    # =========================================================================
    
    r = builder.vec_math('LENGTH', -800, 200, 'r')
    builder.link(camera_km.outputs[0], r.inputs[0])
    
    # Clamp r to valid range
    r_clamped = builder.math('MAXIMUM', -600, 200, 'r_min', v1=BOTTOM_RADIUS)
    builder.link(r.outputs['Value'], r_clamped.inputs[0])
    r_final = builder.math('MINIMUM', -400, 200, 'r_clamped', v1=TOP_RADIUS)
    builder.link(r_clamped.outputs[0], r_final.inputs[0])
    
    # rmu = dot(camera, view_ray)
    rmu = builder.vec_math('DOT_PRODUCT', -800, 100, 'rmu')
    builder.link(camera_km.outputs[0], rmu.inputs[0])
    builder.link(view_ray.outputs[0], rmu.inputs[1])
    
    # mu = rmu / r
    mu = builder.math('DIVIDE', -400, 100, 'mu')
    builder.link(rmu.outputs['Value'], mu.inputs[0])
    builder.link(r_final.outputs[0], mu.inputs[1])
    
    # Clamp mu to [-1, 1]
    mu_min = builder.math('MAXIMUM', -200, 100, 'mu_min', v1=-1.0)
    builder.link(mu.outputs[0], mu_min.inputs[0])
    mu_clamped = builder.math('MINIMUM', 0, 100, 'mu_clamped', v1=1.0)
    builder.link(mu_min.outputs[0], mu_clamped.inputs[0])
    
    # mu_s = dot(camera, sun_direction) / r
    cam_dot_sun = builder.vec_math('DOT_PRODUCT', -800, 0, 'cam·sun')
    builder.link(camera_km.outputs[0], cam_dot_sun.inputs[0])
    builder.link(group_input.outputs['Sun_Direction'], cam_dot_sun.inputs[1])
    
    mu_s = builder.math('DIVIDE', -400, 0, 'mu_s')
    builder.link(cam_dot_sun.outputs['Value'], mu_s.inputs[0])
    builder.link(r_final.outputs[0], mu_s.inputs[1])
    
    # Clamp mu_s
    mu_s_min = builder.math('MAXIMUM', -200, 0, 'mu_s_min', v1=-1.0)
    builder.link(mu_s.outputs[0], mu_s_min.inputs[0])
    mu_s_clamped = builder.math('MINIMUM', 0, 0, 'mu_s_clamped', v1=1.0)
    builder.link(mu_s_min.outputs[0], mu_s_clamped.inputs[0])
    
    # nu = dot(view_ray, sun_direction)
    nu = builder.vec_math('DOT_PRODUCT', -800, -100, 'nu')
    builder.link(view_ray.outputs[0], nu.inputs[0])
    builder.link(group_input.outputs['Sun_Direction'], nu.inputs[1])
    
    # =========================================================================
    # POINT PARAMETERS - Using LAW OF COSINES (from reference)
    # Reference: r_p = sqrt(d*d + 2*r*mu*d + r*r)
    #            mu_p = (r*mu + d) / r_p
    #            mu_s_p = (r*mu_s + d*nu) / r_p
    # =========================================================================
    
    # d*d
    d_sq = builder.math('MULTIPLY', 200, -200, 'd²')
    builder.link(d.outputs['Value'], d_sq.inputs[0])
    builder.link(d.outputs['Value'], d_sq.inputs[1])
    
    # 2*r*mu
    two_r = builder.math('MULTIPLY', 200, -250, '2r', v0=2.0)
    builder.link(r_final.outputs[0], two_r.inputs[1])
    
    two_r_mu = builder.math('MULTIPLY', 400, -250, '2r×mu')
    builder.link(two_r.outputs[0], two_r_mu.inputs[0])
    builder.link(mu_clamped.outputs[0], two_r_mu.inputs[1])
    
    # 2*r*mu*d
    two_r_mu_d = builder.math('MULTIPLY', 600, -250, '2r×mu×d')
    builder.link(two_r_mu.outputs[0], two_r_mu_d.inputs[0])
    builder.link(d.outputs['Value'], two_r_mu_d.inputs[1])
    
    # r*r
    r_sq = builder.math('MULTIPLY', 200, -300, 'r²')
    builder.link(r_final.outputs[0], r_sq.inputs[0])
    builder.link(r_final.outputs[0], r_sq.inputs[1])
    
    # d² + 2*r*mu*d
    sum1 = builder.math('ADD', 800, -225, 'd²+2rμd')
    builder.link(d_sq.outputs[0], sum1.inputs[0])
    builder.link(two_r_mu_d.outputs[0], sum1.inputs[1])
    
    # d² + 2*r*mu*d + r²
    sum2 = builder.math('ADD', 1000, -250, 'd²+2rμd+r²')
    builder.link(sum1.outputs[0], sum2.inputs[0])
    builder.link(r_sq.outputs[0], sum2.inputs[1])
    
    # r_p = sqrt(...)
    r_p_raw = builder.math('SQRT', 1200, -250, 'r_p_raw')
    builder.link(sum2.outputs[0], r_p_raw.inputs[0])
    
    # Clamp r_p to valid range
    r_p_min = builder.math('MAXIMUM', 1400, -250, 'r_p_min', v1=BOTTOM_RADIUS)
    builder.link(r_p_raw.outputs[0], r_p_min.inputs[0])
    r_p = builder.math('MINIMUM', 1600, -250, 'r_p', v1=TOP_RADIUS)
    builder.link(r_p_min.outputs[0], r_p.inputs[0])
    
    # mu_p = (r*mu + d) / r_p
    r_mu = builder.math('MULTIPLY', 200, -400, 'r×mu')
    builder.link(r_final.outputs[0], r_mu.inputs[0])
    builder.link(mu_clamped.outputs[0], r_mu.inputs[1])
    
    r_mu_plus_d = builder.math('ADD', 400, -400, 'r×mu+d')
    builder.link(r_mu.outputs[0], r_mu_plus_d.inputs[0])
    builder.link(d.outputs['Value'], r_mu_plus_d.inputs[1])
    
    mu_p = builder.math('DIVIDE', 600, -400, 'mu_p')
    builder.link(r_mu_plus_d.outputs[0], mu_p.inputs[0])
    builder.link(r_p.outputs[0], mu_p.inputs[1])
    
    # Clamp mu_p
    mu_p_min = builder.math('MAXIMUM', 800, -400, 'mu_p_min', v1=-1.0)
    builder.link(mu_p.outputs[0], mu_p_min.inputs[0])
    mu_p_clamped = builder.math('MINIMUM', 1000, -400, 'mu_p_clamped', v1=1.0)
    builder.link(mu_p_min.outputs[0], mu_p_clamped.inputs[0])
    
    # mu_s_p = (r*mu_s + d*nu) / r_p
    r_mu_s = builder.math('MULTIPLY', 200, -500, 'r×mu_s')
    builder.link(r_final.outputs[0], r_mu_s.inputs[0])
    builder.link(mu_s_clamped.outputs[0], r_mu_s.inputs[1])
    
    d_nu = builder.math('MULTIPLY', 200, -550, 'd×nu')
    builder.link(d.outputs['Value'], d_nu.inputs[0])
    builder.link(nu.outputs['Value'], d_nu.inputs[1])
    
    r_mu_s_plus_d_nu = builder.math('ADD', 400, -500, 'r×mu_s+d×nu')
    builder.link(r_mu_s.outputs[0], r_mu_s_plus_d_nu.inputs[0])
    builder.link(d_nu.outputs[0], r_mu_s_plus_d_nu.inputs[1])
    
    mu_s_p = builder.math('DIVIDE', 600, -500, 'mu_s_p')
    builder.link(r_mu_s_plus_d_nu.outputs[0], mu_s_p.inputs[0])
    builder.link(r_p.outputs[0], mu_s_p.inputs[1])
    
    # Clamp mu_s_p
    mu_s_p_min = builder.math('MAXIMUM', 800, -500, 'mu_s_p_min', v1=-1.0)
    builder.link(mu_s_p.outputs[0], mu_s_p_min.inputs[0])
    mu_s_p_clamped = builder.math('MINIMUM', 1000, -500, 'mu_s_p_clamped', v1=1.0)
    builder.link(mu_s_p_min.outputs[0], mu_s_p_clamped.inputs[0])
    
    # =========================================================================
    # TRANSMITTANCE UV - Using sky_nodes approach (known working)
    # =========================================================================
    
    H = math.sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS)
    
    # rho = sqrt(r² - bottom²)
    bottom_sq = BOTTOM_RADIUS * BOTTOM_RADIUS
    r_sq_for_trans = builder.math('MULTIPLY', 1800, 300, 'r²_trans')
    builder.link(r_final.outputs[0], r_sq_for_trans.inputs[0])
    builder.link(r_final.outputs[0], r_sq_for_trans.inputs[1])
    
    rho_sq = builder.math('SUBTRACT', 2000, 300, 'rho²', v1=bottom_sq)
    builder.link(r_sq_for_trans.outputs[0], rho_sq.inputs[0])
    
    rho = builder.math('SQRT', 2200, 300, 'rho')
    builder.link(rho_sq.outputs[0], rho.inputs[0])
    
    # x_r = rho / H
    x_r = builder.math('DIVIDE', 2400, 300, 'x_r', v1=H)
    builder.link(rho.outputs[0], x_r.inputs[0])
    
    # Distance to top boundary: d = -r*mu + sqrt(r²*(mu²-1) + top²)
    mu_sq = builder.math('MULTIPLY', 1800, 200, 'mu²')
    builder.link(mu_clamped.outputs[0], mu_sq.inputs[0])
    builder.link(mu_clamped.outputs[0], mu_sq.inputs[1])
    
    mu_sq_m1 = builder.math('SUBTRACT', 2000, 200, 'mu²-1', v1=1.0)
    builder.link(mu_sq.outputs[0], mu_sq_m1.inputs[0])
    
    r_sq_term = builder.math('MULTIPLY', 2200, 200, 'r²×(mu²-1)')
    builder.link(r_sq_for_trans.outputs[0], r_sq_term.inputs[0])
    builder.link(mu_sq_m1.outputs[0], r_sq_term.inputs[1])
    
    top_sq = TOP_RADIUS * TOP_RADIUS
    discriminant = builder.math('ADD', 2400, 200, 'disc', v1=top_sq)
    builder.link(r_sq_term.outputs[0], discriminant.inputs[0])
    
    disc_sqrt = builder.math('SQRT', 2600, 200, 'sqrt_disc')
    builder.link(discriminant.outputs[0], disc_sqrt.inputs[0])
    
    neg_r_mu = builder.math('MULTIPLY', 2200, 150, '-r×mu', v1=-1.0)
    r_mu_trans = builder.math('MULTIPLY', 2000, 150, 'r×mu_trans')
    builder.link(r_final.outputs[0], r_mu_trans.inputs[0])
    builder.link(mu_clamped.outputs[0], r_mu_trans.inputs[1])
    builder.link(r_mu_trans.outputs[0], neg_r_mu.inputs[0])
    
    dist_to_top = builder.math('ADD', 2800, 175, 'd_top')
    builder.link(neg_r_mu.outputs[0], dist_to_top.inputs[0])
    builder.link(disc_sqrt.outputs[0], dist_to_top.inputs[1])
    
    # d_min = top_radius - r, d_max = rho + H
    d_min = builder.math('SUBTRACT', 2400, 100, 'd_min', v0=TOP_RADIUS)
    builder.link(r_final.outputs[0], d_min.inputs[1])
    
    d_max = builder.math('ADD', 2400, 50, 'd_max', v1=H)
    builder.link(rho.outputs[0], d_max.inputs[0])
    
    # x_mu = (d - d_min) / (d_max - d_min)
    d_minus_dmin = builder.math('SUBTRACT', 3000, 150, 'd-d_min')
    builder.link(dist_to_top.outputs[0], d_minus_dmin.inputs[0])
    builder.link(d_min.outputs[0], d_minus_dmin.inputs[1])
    
    dmax_minus_dmin = builder.math('SUBTRACT', 3000, 100, 'dmax-dmin')
    builder.link(d_max.outputs[0], dmax_minus_dmin.inputs[0])
    builder.link(d_min.outputs[0], dmax_minus_dmin.inputs[1])
    
    x_mu = builder.math('DIVIDE', 3200, 125, 'x_mu')
    builder.link(d_minus_dmin.outputs[0], x_mu.inputs[0])
    builder.link(dmax_minus_dmin.outputs[0], x_mu.inputs[1])
    
    # Apply GetTextureCoordFromUnitRange
    u_scale = 1.0 - 1.0 / TRANSMITTANCE_TEXTURE_WIDTH
    u_offset = 0.5 / TRANSMITTANCE_TEXTURE_WIDTH
    v_scale = 1.0 - 1.0 / TRANSMITTANCE_TEXTURE_HEIGHT
    v_offset = 0.5 / TRANSMITTANCE_TEXTURE_HEIGHT
    
    trans_u_scaled = builder.math('MULTIPLY', 3400, 150, 'trans_u_scaled', v1=u_scale)
    builder.link(x_mu.outputs[0], trans_u_scaled.inputs[0])
    trans_u = builder.math('ADD', 3600, 150, 'trans_u', v1=u_offset)
    builder.link(trans_u_scaled.outputs[0], trans_u.inputs[0])
    
    trans_v_scaled = builder.math('MULTIPLY', 3400, 100, 'trans_v_scaled', v1=v_scale)
    builder.link(x_r.outputs[0], trans_v_scaled.inputs[0])
    trans_v = builder.math('ADD', 3600, 100, 'trans_v', v1=v_offset)
    builder.link(trans_v_scaled.outputs[0], trans_v.inputs[0])
    
    # Flip V for Blender texture convention
    trans_v_flip = builder.math('SUBTRACT', 3800, 100, 'trans_v_flip', v0=1.0)
    builder.link(trans_v.outputs[0], trans_v_flip.inputs[1])
    
    # Combine transmittance UV
    trans_uv = builder.combine_xyz(4000, 125, 'Trans_UV')
    builder.link(trans_u.outputs[0], trans_uv.inputs['X'])
    builder.link(trans_v_flip.outputs[0], trans_uv.inputs['Y'])
    
    builder.link(trans_uv.outputs[0], tex_transmittance.inputs['Vector'])
    
    # =========================================================================
    # SCATTERING UV FOR CAMERA - Simplified (use sky_nodes pattern)
    # This follows the working sky shader approach
    # =========================================================================
    
    # For now, output transmittance directly and a simple inscatter
    # This can be refined once the basic structure works
    
    # Output transmittance
    builder.link(tex_transmittance.outputs['Color'], group_output.inputs['Transmittance'])
    
    # For inscatter, we need proper scattering lookup - use a placeholder for now
    # that shows SOMETHING to verify the coordinate transforms work
    
    # Simple inscatter based on distance (placeholder - will be replaced)
    # inscatter = (1 - transmittance) * sky_color_approximation
    
    # Create a simple blue-ish haze color based on distance
    dist_normalized = builder.math('DIVIDE', 2000, -600, 'd_norm', v1=100.0)  # Normalize to ~100km
    builder.link(d.outputs['Value'], dist_normalized.inputs[0])
    
    dist_clamped = builder.math('MINIMUM', 2200, -600, 'd_clamp', v1=1.0)
    builder.link(dist_normalized.outputs[0], dist_clamped.inputs[0])
    
    # Simple haze color (will be replaced with proper scattering lookup)
    haze_color = builder.combine_xyz(2400, -600, 'Haze_Color')
    haze_color.inputs['X'].default_value = 0.1  # R
    haze_color.inputs['Y'].default_value = 0.15  # G  
    haze_color.inputs['Z'].default_value = 0.25  # B
    
    # Scale by distance
    haze_scaled = builder.vec_math('SCALE', 2600, -600, 'Haze_Scaled')
    builder.link(haze_color.outputs[0], haze_scaled.inputs[0])
    builder.link(dist_clamped.outputs[0], haze_scaled.inputs['Scale'])
    
    builder.link(haze_scaled.outputs[0], group_output.inputs['Inscatter'])
    
    # Store version
    group['helios_version'] = AERIAL_NODE_VERSION
    
    print(f"Helios: Created node group '{AERIAL_NODE_GROUP_NAME}' v{AERIAL_NODE_VERSION}")
    print(f"Helios: NOTE - Using simplified inscatter, proper scattering lookup to be implemented")
    return group


def get_or_create_aerial_node_group(lut_dir=None):
    """Get existing node group or create a new one."""
    if AERIAL_NODE_GROUP_NAME in bpy.data.node_groups:
        existing = bpy.data.node_groups[AERIAL_NODE_GROUP_NAME]
        existing_version = existing.get('helios_version', 0)
        if existing_version < AERIAL_NODE_VERSION:
            print(f"Helios: Aerial node group version {existing_version} < {AERIAL_NODE_VERSION}, recreating")
            bpy.data.node_groups.remove(existing)
        else:
            return existing
    return create_aerial_perspective_node_group(lut_dir)


# =============================================================================
# REGISTRATION
# =============================================================================

def register():
    pass

def unregister():
    pass
