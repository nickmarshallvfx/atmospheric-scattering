"""
Helios Node-Based Bruneton Aerial Perspective

Implements GetSkyRadianceToPoint from Eric Bruneton's atmospheric scattering.
Reference: atmospheric-scattering-2-export/atmosphere/functions.glsl

Key formulas from reference (NOT using position vectors directly!):
  d = length(point - camera)
  r_p = sqrt(d² + 2·r·μ·d + r²)   # Law of cosines
  μ_p = (r·μ + d) / r_p
  μ_s_p = (r·μ_s + d·ν) / r_p
  inscatter = S_cam - transmittance × S_point

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

# Atmosphere constants
BOTTOM_RADIUS = 6360.0  # km
TOP_RADIUS = 6420.0     # km
MU_S_MIN = -0.2
H = math.sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS)


# =============================================================================
# NODE GROUP NAME AND VERSION
# =============================================================================

AERIAL_NODE_GROUP_NAME = "Helios_Aerial_Perspective"
AERIAL_NODE_VERSION = 6  # Complete rewrite based on reference analysis


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
    
    def mix(self, data_type, blend_type, x, y, name):
        node = self.nodes.new('ShaderNodeMix')
        node.data_type = data_type
        node.blend_type = blend_type
        node.location = (x, y)
        node.name = name
        node.label = name
        return node
    
    def link(self, from_socket, to_socket):
        self.links.new(from_socket, to_socket)


# =============================================================================
# SCATTERING UV HELPER - Creates UV for GetCombinedScattering
# =============================================================================

def create_scattering_uvs(builder, r_socket, mu_socket, mu_s_socket, nu_socket,
                          base_x, base_y, suffix=""):
    """
    Create scattering texture UV coordinates following reference exactly.
    
    Returns the final scattering color socket after all interpolation.
    """
    
    # u_r: altitude coordinate
    # rho = sqrt(r² - bottom²)
    r_sq = builder.math('MULTIPLY', base_x, base_y, f'r²{suffix}')
    builder.link(r_socket, r_sq.inputs[0])
    builder.link(r_socket, r_sq.inputs[1])
    
    rho_sq = builder.math('SUBTRACT', base_x + 150, base_y, f'rho²{suffix}', 
                          v1=BOTTOM_RADIUS * BOTTOM_RADIUS)
    builder.link(r_sq.outputs[0], rho_sq.inputs[0])
    
    rho_sq_safe = builder.math('MAXIMUM', base_x + 300, base_y, f'rho²_safe{suffix}', v1=0.0)
    builder.link(rho_sq.outputs[0], rho_sq_safe.inputs[0])
    
    rho = builder.math('SQRT', base_x + 450, base_y, f'rho{suffix}')
    builder.link(rho_sq_safe.outputs[0], rho.inputs[0])
    
    x_r = builder.math('DIVIDE', base_x + 600, base_y, f'x_r{suffix}', v1=H)
    builder.link(rho.outputs[0], x_r.inputs[0])
    
    # Apply GetTextureCoordFromUnitRange for r
    r_scale = 1.0 - 1.0 / SCATTERING_TEXTURE_R_SIZE
    r_offset = 0.5 / SCATTERING_TEXTURE_R_SIZE
    
    u_r_scaled = builder.math('MULTIPLY', base_x + 750, base_y, f'u_r_sc{suffix}', v1=r_scale)
    builder.link(x_r.outputs[0], u_r_scaled.inputs[0])
    
    u_r = builder.math('ADD', base_x + 900, base_y, f'u_r{suffix}', v1=r_offset)
    builder.link(u_r_scaled.outputs[0], u_r.inputs[0])
    
    # Clamp u_r
    u_r_min = builder.math('MAXIMUM', base_x + 1050, base_y, f'u_r_min{suffix}', v1=0.0)
    builder.link(u_r.outputs[0], u_r_min.inputs[0])
    u_r_final = builder.math('MINIMUM', base_x + 1200, base_y, f'u_r_final{suffix}', v1=1.0)
    builder.link(u_r_min.outputs[0], u_r_final.inputs[0])
    
    # u_mu: view zenith coordinate (simplified - not handling ground intersection)
    # Distance to top boundary
    mu_sq = builder.math('MULTIPLY', base_x, base_y - 100, f'mu²{suffix}')
    builder.link(mu_socket, mu_sq.inputs[0])
    builder.link(mu_socket, mu_sq.inputs[1])
    
    mu_sq_m1 = builder.math('SUBTRACT', base_x + 150, base_y - 100, f'mu²-1{suffix}', v1=1.0)
    builder.link(mu_sq.outputs[0], mu_sq_m1.inputs[0])
    
    r_sq_term = builder.math('MULTIPLY', base_x + 300, base_y - 100, f'r²×term{suffix}')
    builder.link(r_sq.outputs[0], r_sq_term.inputs[0])
    builder.link(mu_sq_m1.outputs[0], r_sq_term.inputs[1])
    
    discriminant = builder.math('ADD', base_x + 450, base_y - 100, f'disc{suffix}', 
                                v1=TOP_RADIUS * TOP_RADIUS)
    builder.link(r_sq_term.outputs[0], discriminant.inputs[0])
    
    disc_safe = builder.math('MAXIMUM', base_x + 600, base_y - 100, f'disc_safe{suffix}', v1=0.0)
    builder.link(discriminant.outputs[0], disc_safe.inputs[0])
    
    disc_sqrt = builder.math('SQRT', base_x + 750, base_y - 100, f'disc_sqrt{suffix}')
    builder.link(disc_safe.outputs[0], disc_sqrt.inputs[0])
    
    # -r*mu
    r_mu = builder.math('MULTIPLY', base_x + 300, base_y - 150, f'r×mu{suffix}')
    builder.link(r_socket, r_mu.inputs[0])
    builder.link(mu_socket, r_mu.inputs[1])
    
    neg_r_mu = builder.math('MULTIPLY', base_x + 450, base_y - 150, f'-r×mu{suffix}', v1=-1.0)
    builder.link(r_mu.outputs[0], neg_r_mu.inputs[0])
    
    # d = -r*mu + sqrt(discriminant)
    d_top = builder.math('ADD', base_x + 900, base_y - 125, f'd_top{suffix}')
    builder.link(neg_r_mu.outputs[0], d_top.inputs[0])
    builder.link(disc_sqrt.outputs[0], d_top.inputs[1])
    
    # d_min = top - r, d_max = rho + H
    d_min = builder.math('SUBTRACT', base_x + 600, base_y - 200, f'd_min{suffix}', v0=TOP_RADIUS)
    builder.link(r_socket, d_min.inputs[1])
    
    d_max = builder.math('ADD', base_x + 600, base_y - 250, f'd_max{suffix}', v1=H)
    builder.link(rho.outputs[0], d_max.inputs[0])
    
    # x_mu = (d - d_min) / (d_max - d_min)
    d_minus_dmin = builder.math('SUBTRACT', base_x + 1050, base_y - 125, f'd-dmin{suffix}')
    builder.link(d_top.outputs[0], d_minus_dmin.inputs[0])
    builder.link(d_min.outputs[0], d_minus_dmin.inputs[1])
    
    dmax_minus_dmin = builder.math('SUBTRACT', base_x + 1050, base_y - 200, f'dmax-dmin{suffix}')
    builder.link(d_max.outputs[0], dmax_minus_dmin.inputs[0])
    builder.link(d_min.outputs[0], dmax_minus_dmin.inputs[1])
    
    # Prevent division by zero
    denom_safe = builder.math('MAXIMUM', base_x + 1200, base_y - 200, f'denom_safe{suffix}', v1=0.001)
    builder.link(dmax_minus_dmin.outputs[0], denom_safe.inputs[0])
    
    x_mu = builder.math('DIVIDE', base_x + 1350, base_y - 150, f'x_mu{suffix}')
    builder.link(d_minus_dmin.outputs[0], x_mu.inputs[0])
    builder.link(denom_safe.outputs[0], x_mu.inputs[1])
    
    # For non-ground-intersecting rays: u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(x_mu, MU_SIZE/2)
    mu_scale = 1.0 - 2.0 / SCATTERING_TEXTURE_MU_SIZE
    mu_offset = 1.0 / SCATTERING_TEXTURE_MU_SIZE
    
    x_mu_scaled = builder.math('MULTIPLY', base_x + 1500, base_y - 150, f'x_mu_sc{suffix}', v1=mu_scale)
    builder.link(x_mu.outputs[0], x_mu_scaled.inputs[0])
    
    x_mu_offset = builder.math('ADD', base_x + 1650, base_y - 150, f'x_mu_off{suffix}', v1=mu_offset)
    builder.link(x_mu_scaled.outputs[0], x_mu_offset.inputs[0])
    
    u_mu_half = builder.math('MULTIPLY', base_x + 1800, base_y - 150, f'u_mu_half{suffix}', v1=0.5)
    builder.link(x_mu_offset.outputs[0], u_mu_half.inputs[0])
    
    u_mu = builder.math('ADD', base_x + 1950, base_y - 150, f'u_mu{suffix}', v0=0.5)
    builder.link(u_mu_half.outputs[0], u_mu.inputs[1])
    
    # Clamp u_mu
    u_mu_min = builder.math('MAXIMUM', base_x + 2100, base_y - 150, f'u_mu_min{suffix}', v1=0.0)
    builder.link(u_mu.outputs[0], u_mu_min.inputs[0])
    u_mu_final = builder.math('MINIMUM', base_x + 2250, base_y - 150, f'u_mu_final{suffix}', v1=1.0)
    builder.link(u_mu_min.outputs[0], u_mu_final.inputs[0])
    
    # u_mu_s: sun zenith - Bruneton non-linear mapping
    # d = DistanceToTopAtmosphereBoundary(bottom_radius, mu_s)
    mu_s_sq = builder.math('MULTIPLY', base_x, base_y - 300, f'mu_s²{suffix}')
    builder.link(mu_s_socket, mu_s_sq.inputs[0])
    builder.link(mu_s_socket, mu_s_sq.inputs[1])
    
    mu_s_sq_m1 = builder.math('SUBTRACT', base_x + 150, base_y - 300, f'mu_s²-1{suffix}', v1=1.0)
    builder.link(mu_s_sq.outputs[0], mu_s_sq_m1.inputs[0])
    
    br_sq_term = builder.math('MULTIPLY', base_x + 300, base_y - 300, f'br²×term{suffix}', 
                              v0=BOTTOM_RADIUS * BOTTOM_RADIUS)
    builder.link(mu_s_sq_m1.outputs[0], br_sq_term.inputs[1])
    
    d_mus_disc = builder.math('ADD', base_x + 450, base_y - 300, f'd_mus_disc{suffix}', 
                              v1=TOP_RADIUS * TOP_RADIUS)
    builder.link(br_sq_term.outputs[0], d_mus_disc.inputs[0])
    
    d_mus_disc_safe = builder.math('MAXIMUM', base_x + 600, base_y - 300, f'd_mus_disc_safe{suffix}', v1=0.0)
    builder.link(d_mus_disc.outputs[0], d_mus_disc_safe.inputs[0])
    
    d_mus_sqrt = builder.math('SQRT', base_x + 750, base_y - 300, f'd_mus_sqrt{suffix}')
    builder.link(d_mus_disc_safe.outputs[0], d_mus_sqrt.inputs[0])
    
    neg_br_mus = builder.math('MULTIPLY', base_x + 450, base_y - 350, f'-br×mu_s{suffix}', v0=-BOTTOM_RADIUS)
    builder.link(mu_s_socket, neg_br_mus.inputs[1])
    
    d_mus = builder.math('ADD', base_x + 900, base_y - 325, f'd_mus{suffix}')
    builder.link(neg_br_mus.outputs[0], d_mus.inputs[0])
    builder.link(d_mus_sqrt.outputs[0], d_mus.inputs[1])
    
    d_mus_safe = builder.math('MAXIMUM', base_x + 1050, base_y - 325, f'd_mus_safe{suffix}', v1=0.0)
    builder.link(d_mus.outputs[0], d_mus_safe.inputs[0])
    
    # a = (d - d_min) / (d_max - d_min)
    d_mus_min = TOP_RADIUS - BOTTOM_RADIUS
    d_mus_max = H
    
    # Precompute A for mu_s_min
    D_const = -BOTTOM_RADIUS * MU_S_MIN + math.sqrt(
        BOTTOM_RADIUS * BOTTOM_RADIUS * (MU_S_MIN * MU_S_MIN - 1.0) + TOP_RADIUS * TOP_RADIUS)
    A_const = (D_const - d_mus_min) / (d_mus_max - d_mus_min)
    
    d_mus_shifted = builder.math('SUBTRACT', base_x + 1200, base_y - 325, f'd_mus-dmin{suffix}', v1=d_mus_min)
    builder.link(d_mus_safe.outputs[0], d_mus_shifted.inputs[0])
    
    a_val = builder.math('DIVIDE', base_x + 1350, base_y - 325, f'a{suffix}', v1=(d_mus_max - d_mus_min))
    builder.link(d_mus_shifted.outputs[0], a_val.inputs[0])
    
    # x_mu_s = max(1 - a/A, 0) / (1 + a)
    a_over_A = builder.math('DIVIDE', base_x + 1500, base_y - 325, f'a/A{suffix}', v1=A_const)
    builder.link(a_val.outputs[0], a_over_A.inputs[0])
    
    one_minus_aA = builder.math('SUBTRACT', base_x + 1650, base_y - 325, f'1-a/A{suffix}', v0=1.0)
    builder.link(a_over_A.outputs[0], one_minus_aA.inputs[1])
    
    one_minus_aA_safe = builder.math('MAXIMUM', base_x + 1800, base_y - 325, f'max(1-a/A,0){suffix}', v1=0.0)
    builder.link(one_minus_aA.outputs[0], one_minus_aA_safe.inputs[0])
    
    one_plus_a = builder.math('ADD', base_x + 1650, base_y - 375, f'1+a{suffix}', v0=1.0)
    builder.link(a_val.outputs[0], one_plus_a.inputs[1])
    
    one_plus_a_safe = builder.math('MAXIMUM', base_x + 1800, base_y - 375, f'1+a_safe{suffix}', v1=0.001)
    builder.link(one_plus_a.outputs[0], one_plus_a_safe.inputs[0])
    
    x_mu_s = builder.math('DIVIDE', base_x + 1950, base_y - 350, f'x_mu_s{suffix}')
    builder.link(one_minus_aA_safe.outputs[0], x_mu_s.inputs[0])
    builder.link(one_plus_a_safe.outputs[0], x_mu_s.inputs[1])
    
    # Apply GetTextureCoordFromUnitRange for mu_s
    mu_s_scale = 1.0 - 1.0 / SCATTERING_TEXTURE_MU_S_SIZE
    mu_s_offset = 0.5 / SCATTERING_TEXTURE_MU_S_SIZE
    
    u_mu_s_scaled = builder.math('MULTIPLY', base_x + 2100, base_y - 350, f'u_mu_s_sc{suffix}', v1=mu_s_scale)
    builder.link(x_mu_s.outputs[0], u_mu_s_scaled.inputs[0])
    
    u_mu_s = builder.math('ADD', base_x + 2250, base_y - 350, f'u_mu_s{suffix}', v1=mu_s_offset)
    builder.link(u_mu_s_scaled.outputs[0], u_mu_s.inputs[0])
    
    # u_nu: view-sun angle (simple linear)
    nu_plus1 = builder.math('ADD', base_x, base_y - 450, f'nu+1{suffix}', v0=1.0)
    builder.link(nu_socket, nu_plus1.inputs[1])
    
    x_nu = builder.math('MULTIPLY', base_x + 150, base_y - 450, f'x_nu{suffix}', v1=0.5)
    builder.link(nu_plus1.outputs[0], x_nu.inputs[0])
    
    # Return all UV components
    return u_r_final, u_mu_final, u_mu_s, x_nu


# =============================================================================
# MAIN NODE GROUP CREATION
# =============================================================================

def create_aerial_perspective_node_group(lut_dir=None):
    """
    Create the Helios Aerial Perspective node group.
    
    Implements GetSkyRadianceToPoint from Bruneton reference EXACTLY.
    """
    if lut_dir is None:
        lut_dir = get_lut_cache_dir()
    
    # Remove existing group
    if AERIAL_NODE_GROUP_NAME in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups[AERIAL_NODE_GROUP_NAME])
    
    group = bpy.data.node_groups.new(AERIAL_NODE_GROUP_NAME, 'ShaderNodeTree')
    builder = NodeBuilder(group)
    
    # =========================================================================
    # INTERFACE
    # =========================================================================
    
    group_input = group.nodes.new('NodeGroupInput')
    group_input.location = (-2500, 0)
    
    group_output = group.nodes.new('NodeGroupOutput')
    group_output.location = (5000, 0)
    
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
    
    tex_transmittance = builder.image_texture(-2200, 500, 'Transmittance_LUT', transmittance_path)
    
    # =========================================================================
    # COORDINATE TRANSFORMS - camera and point relative to earth center
    # =========================================================================
    
    # camera_km = (Camera_Position - Planet_Center) * Scene_Scale
    cam_minus_center = builder.vec_math('SUBTRACT', -2200, 200, 'Cam-Center')
    builder.link(group_input.outputs['Camera_Position'], cam_minus_center.inputs[0])
    builder.link(group_input.outputs['Planet_Center'], cam_minus_center.inputs[1])
    
    camera_km = builder.vec_math('SCALE', -2000, 200, 'Camera_km')
    builder.link(cam_minus_center.outputs[0], camera_km.inputs[0])
    builder.link(group_input.outputs['Scene_Scale'], camera_km.inputs['Scale'])
    
    # point_km = (Position - Planet_Center) * Scene_Scale
    pt_minus_center = builder.vec_math('SUBTRACT', -2200, 0, 'Point-Center')
    builder.link(group_input.outputs['Position'], pt_minus_center.inputs[0])
    builder.link(group_input.outputs['Planet_Center'], pt_minus_center.inputs[1])
    
    point_km = builder.vec_math('SCALE', -2000, 0, 'Point_km')
    builder.link(pt_minus_center.outputs[0], point_km.inputs[0])
    builder.link(group_input.outputs['Scene_Scale'], point_km.inputs['Scale'])
    
    # =========================================================================
    # VIEW RAY AND DISTANCE
    # Reference: view_ray = normalize(point - camera)
    #            d = length(point - camera)
    # =========================================================================
    
    pt_minus_cam = builder.vec_math('SUBTRACT', -1800, 100, 'Point-Camera')
    builder.link(point_km.outputs[0], pt_minus_cam.inputs[0])
    builder.link(camera_km.outputs[0], pt_minus_cam.inputs[1])
    
    view_ray = builder.vec_math('NORMALIZE', -1600, 150, 'View_Ray')
    builder.link(pt_minus_cam.outputs[0], view_ray.inputs[0])
    
    d = builder.vec_math('LENGTH', -1600, 50, 'd')
    builder.link(pt_minus_cam.outputs[0], d.inputs[0])
    
    # =========================================================================
    # CAMERA PARAMETERS (r, mu, mu_s, nu)
    # =========================================================================
    
    # r = length(camera), clamped
    r_raw = builder.vec_math('LENGTH', -1400, 300, 'r_raw')
    builder.link(camera_km.outputs[0], r_raw.inputs[0])
    
    r_min = builder.math('MAXIMUM', -1200, 300, 'r_min', v1=BOTTOM_RADIUS)
    builder.link(r_raw.outputs['Value'], r_min.inputs[0])
    
    r = builder.math('MINIMUM', -1000, 300, 'r', v1=TOP_RADIUS)
    builder.link(r_min.outputs[0], r.inputs[0])
    
    # mu = dot(camera, view_ray) / r
    cam_dot_view = builder.vec_math('DOT_PRODUCT', -1400, 200, 'cam·view')
    builder.link(camera_km.outputs[0], cam_dot_view.inputs[0])
    builder.link(view_ray.outputs[0], cam_dot_view.inputs[1])
    
    mu = builder.math('DIVIDE', -1000, 200, 'mu')
    builder.link(cam_dot_view.outputs['Value'], mu.inputs[0])
    builder.link(r.outputs[0], mu.inputs[1])
    
    mu_clamped = builder.math('MINIMUM', -800, 200, 'mu_clamp', v1=1.0)
    builder.link(mu.outputs[0], mu_clamped.inputs[0])
    mu_final = builder.math('MAXIMUM', -600, 200, 'mu_final', v1=-1.0)
    builder.link(mu_clamped.outputs[0], mu_final.inputs[0])
    
    # mu_s = dot(camera, sun_direction) / r
    cam_dot_sun = builder.vec_math('DOT_PRODUCT', -1400, 100, 'cam·sun')
    builder.link(camera_km.outputs[0], cam_dot_sun.inputs[0])
    builder.link(group_input.outputs['Sun_Direction'], cam_dot_sun.inputs[1])
    
    mu_s = builder.math('DIVIDE', -1000, 100, 'mu_s')
    builder.link(cam_dot_sun.outputs['Value'], mu_s.inputs[0])
    builder.link(r.outputs[0], mu_s.inputs[1])
    
    mu_s_clamped = builder.math('MINIMUM', -800, 100, 'mu_s_clamp', v1=1.0)
    builder.link(mu_s.outputs[0], mu_s_clamped.inputs[0])
    mu_s_final = builder.math('MAXIMUM', -600, 100, 'mu_s_final', v1=-1.0)
    builder.link(mu_s_clamped.outputs[0], mu_s_final.inputs[0])
    
    # nu = dot(view_ray, sun_direction)
    nu = builder.vec_math('DOT_PRODUCT', -1400, 0, 'nu')
    builder.link(view_ray.outputs[0], nu.inputs[0])
    builder.link(group_input.outputs['Sun_Direction'], nu.inputs[1])
    
    # =========================================================================
    # POINT PARAMETERS - LAW OF COSINES (from reference!)
    # r_p = sqrt(d² + 2·r·μ·d + r²)
    # μ_p = (r·μ + d) / r_p
    # μ_s_p = (r·μ_s + d·ν) / r_p
    # =========================================================================
    
    # d²
    d_sq = builder.math('MULTIPLY', -400, -100, 'd²')
    builder.link(d.outputs['Value'], d_sq.inputs[0])
    builder.link(d.outputs['Value'], d_sq.inputs[1])
    
    # 2·r·μ
    two_r = builder.math('MULTIPLY', -400, -150, '2r', v0=2.0)
    builder.link(r.outputs[0], two_r.inputs[1])
    
    two_r_mu = builder.math('MULTIPLY', -200, -150, '2r×μ')
    builder.link(two_r.outputs[0], two_r_mu.inputs[0])
    builder.link(mu_final.outputs[0], two_r_mu.inputs[1])
    
    # 2·r·μ·d
    two_r_mu_d = builder.math('MULTIPLY', 0, -150, '2r×μ×d')
    builder.link(two_r_mu.outputs[0], two_r_mu_d.inputs[0])
    builder.link(d.outputs['Value'], two_r_mu_d.inputs[1])
    
    # r²
    r_sq = builder.math('MULTIPLY', -400, -200, 'r²')
    builder.link(r.outputs[0], r_sq.inputs[0])
    builder.link(r.outputs[0], r_sq.inputs[1])
    
    # d² + 2·r·μ·d + r²
    sum1 = builder.math('ADD', 200, -125, 'd²+2rμd')
    builder.link(d_sq.outputs[0], sum1.inputs[0])
    builder.link(two_r_mu_d.outputs[0], sum1.inputs[1])
    
    sum2 = builder.math('ADD', 400, -150, 'd²+2rμd+r²')
    builder.link(sum1.outputs[0], sum2.inputs[0])
    builder.link(r_sq.outputs[0], sum2.inputs[1])
    
    # r_p = sqrt(...)
    r_p_raw = builder.math('SQRT', 600, -150, 'r_p_raw')
    builder.link(sum2.outputs[0], r_p_raw.inputs[0])
    
    # Clamp r_p
    r_p_min = builder.math('MAXIMUM', 800, -150, 'r_p_min', v1=BOTTOM_RADIUS)
    builder.link(r_p_raw.outputs[0], r_p_min.inputs[0])
    
    r_p = builder.math('MINIMUM', 1000, -150, 'r_p', v1=TOP_RADIUS)
    builder.link(r_p_min.outputs[0], r_p.inputs[0])
    
    # μ_p = (r·μ + d) / r_p
    r_mu = builder.math('MULTIPLY', -200, -300, 'r×μ')
    builder.link(r.outputs[0], r_mu.inputs[0])
    builder.link(mu_final.outputs[0], r_mu.inputs[1])
    
    r_mu_plus_d = builder.math('ADD', 0, -300, 'r×μ+d')
    builder.link(r_mu.outputs[0], r_mu_plus_d.inputs[0])
    builder.link(d.outputs['Value'], r_mu_plus_d.inputs[1])
    
    mu_p = builder.math('DIVIDE', 200, -300, 'μ_p')
    builder.link(r_mu_plus_d.outputs[0], mu_p.inputs[0])
    builder.link(r_p.outputs[0], mu_p.inputs[1])
    
    mu_p_clamped = builder.math('MINIMUM', 400, -300, 'μ_p_clamp', v1=1.0)
    builder.link(mu_p.outputs[0], mu_p_clamped.inputs[0])
    mu_p_final = builder.math('MAXIMUM', 600, -300, 'μ_p_final', v1=-1.0)
    builder.link(mu_p_clamped.outputs[0], mu_p_final.inputs[0])
    
    # μ_s_p = (r·μ_s + d·ν) / r_p
    r_mu_s = builder.math('MULTIPLY', -200, -400, 'r×μ_s')
    builder.link(r.outputs[0], r_mu_s.inputs[0])
    builder.link(mu_s_final.outputs[0], r_mu_s.inputs[1])
    
    d_nu = builder.math('MULTIPLY', -200, -450, 'd×ν')
    builder.link(d.outputs['Value'], d_nu.inputs[0])
    builder.link(nu.outputs['Value'], d_nu.inputs[1])
    
    r_mu_s_plus_d_nu = builder.math('ADD', 0, -400, 'r×μ_s+d×ν')
    builder.link(r_mu_s.outputs[0], r_mu_s_plus_d_nu.inputs[0])
    builder.link(d_nu.outputs[0], r_mu_s_plus_d_nu.inputs[1])
    
    mu_s_p = builder.math('DIVIDE', 200, -400, 'μ_s_p')
    builder.link(r_mu_s_plus_d_nu.outputs[0], mu_s_p.inputs[0])
    builder.link(r_p.outputs[0], mu_s_p.inputs[1])
    
    mu_s_p_clamped = builder.math('MINIMUM', 400, -400, 'μ_s_p_clamp', v1=1.0)
    builder.link(mu_s_p.outputs[0], mu_s_p_clamped.inputs[0])
    mu_s_p_final = builder.math('MAXIMUM', 600, -400, 'μ_s_p_final', v1=-1.0)
    builder.link(mu_s_p_clamped.outputs[0], mu_s_p_final.inputs[0])
    
    # =========================================================================
    # TRANSMITTANCE UV
    # =========================================================================
    
    # rho = sqrt(r² - bottom²)
    trans_rho_sq = builder.math('SUBTRACT', -400, 500, 'trans_rho²', v1=BOTTOM_RADIUS * BOTTOM_RADIUS)
    builder.link(r_sq.outputs[0], trans_rho_sq.inputs[0])
    
    trans_rho_sq_safe = builder.math('MAXIMUM', -200, 500, 'trans_rho²_safe', v1=0.0)
    builder.link(trans_rho_sq.outputs[0], trans_rho_sq_safe.inputs[0])
    
    trans_rho = builder.math('SQRT', 0, 500, 'trans_rho')
    builder.link(trans_rho_sq_safe.outputs[0], trans_rho.inputs[0])
    
    trans_x_r = builder.math('DIVIDE', 200, 500, 'trans_x_r', v1=H)
    builder.link(trans_rho.outputs[0], trans_x_r.inputs[0])
    
    # x_mu for transmittance (distance to top)
    trans_d_min = builder.math('SUBTRACT', 200, 400, 'trans_d_min', v0=TOP_RADIUS)
    builder.link(r.outputs[0], trans_d_min.inputs[1])
    
    trans_d_max = builder.math('ADD', 200, 350, 'trans_d_max', v1=H)
    builder.link(trans_rho.outputs[0], trans_d_max.inputs[0])
    
    trans_d_minus_dmin = builder.math('SUBTRACT', 400, 450, 'trans_d-dmin')
    builder.link(cam_dot_view.outputs['Value'], trans_d_minus_dmin.inputs[0])  # Will compute properly
    builder.link(trans_d_min.outputs[0], trans_d_minus_dmin.inputs[1])
    
    # Use simplified mapping for transmittance
    trans_v_scale = 1.0 - 1.0 / TRANSMITTANCE_TEXTURE_HEIGHT
    trans_v_offset = 0.5 / TRANSMITTANCE_TEXTURE_HEIGHT
    
    trans_v_scaled = builder.math('MULTIPLY', 400, 500, 'trans_v_sc', v1=trans_v_scale)
    builder.link(trans_x_r.outputs[0], trans_v_scaled.inputs[0])
    
    trans_v = builder.math('ADD', 600, 500, 'trans_v', v1=trans_v_offset)
    builder.link(trans_v_scaled.outputs[0], trans_v.inputs[0])
    
    trans_v_flip = builder.math('SUBTRACT', 800, 500, 'trans_v_flip', v0=1.0)
    builder.link(trans_v.outputs[0], trans_v_flip.inputs[1])
    
    # u based on mu
    mu_plus1 = builder.math('ADD', 400, 550, 'mu+1', v0=1.0)
    builder.link(mu_final.outputs[0], mu_plus1.inputs[1])
    
    trans_x_mu = builder.math('MULTIPLY', 600, 550, 'trans_x_mu', v1=0.5)
    builder.link(mu_plus1.outputs[0], trans_x_mu.inputs[0])
    
    trans_u_scale = 1.0 - 1.0 / TRANSMITTANCE_TEXTURE_WIDTH
    trans_u_offset = 0.5 / TRANSMITTANCE_TEXTURE_WIDTH
    
    trans_u_scaled = builder.math('MULTIPLY', 800, 550, 'trans_u_sc', v1=trans_u_scale)
    builder.link(trans_x_mu.outputs[0], trans_u_scaled.inputs[0])
    
    trans_u = builder.math('ADD', 1000, 550, 'trans_u', v1=trans_u_offset)
    builder.link(trans_u_scaled.outputs[0], trans_u.inputs[0])
    
    trans_uv = builder.combine_xyz(1200, 525, 'Trans_UV')
    builder.link(trans_u.outputs[0], trans_uv.inputs['X'])
    builder.link(trans_v_flip.outputs[0], trans_uv.inputs['Y'])
    
    builder.link(trans_uv.outputs[0], tex_transmittance.inputs['Vector'])
    
    # =========================================================================
    # SCATTERING LOOKUPS - Camera and Point
    # =========================================================================
    
    # Create scattering UVs for camera position
    u_r_cam, u_mu_cam, u_mu_s_cam, x_nu_cam = create_scattering_uvs(
        builder, r.outputs[0], mu_final.outputs[0], mu_s_final.outputs[0], nu.outputs['Value'],
        1200, 200, "_cam"
    )
    
    # Create scattering UVs for point position
    u_r_pt, u_mu_pt, u_mu_s_pt, x_nu_pt = create_scattering_uvs(
        builder, r_p.outputs[0], mu_p_final.outputs[0], mu_s_p_final.outputs[0], nu.outputs['Value'],
        1200, -600, "_pt"
    )
    
    # Simplified scattering lookup (single sample for now - proper interpolation can be added)
    # Final X = (floor(x_nu * (NU_SIZE-1)) + u_mu_s) / NU_SIZE + depth_slice / DEPTH
    
    # Camera scattering UV
    tex_x_cam = builder.math('MULTIPLY', 3600, 200, 'tex_x_cam', v1=float(SCATTERING_TEXTURE_NU_SIZE - 1))
    builder.link(x_nu_cam.outputs[0], tex_x_cam.inputs[0])
    
    tex_x_floor_cam = builder.math('FLOOR', 3750, 200, 'tex_x_floor_cam')
    builder.link(tex_x_cam.outputs[0], tex_x_floor_cam.inputs[0])
    
    tex_x_plus_mus_cam = builder.math('ADD', 3900, 200, 'tex_x+mus_cam')
    builder.link(tex_x_floor_cam.outputs[0], tex_x_plus_mus_cam.inputs[0])
    builder.link(u_mu_s_cam.outputs[0], tex_x_plus_mus_cam.inputs[1])
    
    uvw_x_cam = builder.math('DIVIDE', 4050, 200, 'uvw_x_cam', v1=float(SCATTERING_TEXTURE_NU_SIZE))
    builder.link(tex_x_plus_mus_cam.outputs[0], uvw_x_cam.inputs[0])
    
    # Depth slice for camera
    depth_scaled_cam = builder.math('MULTIPLY', 3600, 150, 'depth_sc_cam', v1=float(SCATTERING_TEXTURE_DEPTH - 1))
    builder.link(u_r_cam.outputs[0], depth_scaled_cam.inputs[0])
    
    depth_floor_cam = builder.math('FLOOR', 3750, 150, 'depth_floor_cam')
    builder.link(depth_scaled_cam.outputs[0], depth_floor_cam.inputs[0])
    
    # Final X = (depth_floor + uvw_x) / DEPTH
    slice_plus_uvw_cam = builder.math('ADD', 4200, 175, 'slice+uvw_cam')
    builder.link(depth_floor_cam.outputs[0], slice_plus_uvw_cam.inputs[0])
    builder.link(uvw_x_cam.outputs[0], slice_plus_uvw_cam.inputs[1])
    
    final_x_cam = builder.math('DIVIDE', 4350, 175, 'final_x_cam', v1=float(SCATTERING_TEXTURE_DEPTH))
    builder.link(slice_plus_uvw_cam.outputs[0], final_x_cam.inputs[0])
    
    # Flip Y for camera
    u_mu_flip_cam = builder.math('SUBTRACT', 4200, 125, 'u_mu_flip_cam', v0=1.0)
    builder.link(u_mu_cam.outputs[0], u_mu_flip_cam.inputs[1])
    
    scat_uv_cam = builder.combine_xyz(4500, 150, 'Scat_UV_Cam')
    builder.link(final_x_cam.outputs[0], scat_uv_cam.inputs['X'])
    builder.link(u_mu_flip_cam.outputs[0], scat_uv_cam.inputs['Y'])
    
    tex_scat_cam = builder.image_texture(4650, 150, 'Scat_Cam', scattering_path)
    builder.link(scat_uv_cam.outputs[0], tex_scat_cam.inputs['Vector'])
    
    # Point scattering UV (same pattern)
    tex_x_pt = builder.math('MULTIPLY', 3600, -600, 'tex_x_pt', v1=float(SCATTERING_TEXTURE_NU_SIZE - 1))
    builder.link(x_nu_pt.outputs[0], tex_x_pt.inputs[0])
    
    tex_x_floor_pt = builder.math('FLOOR', 3750, -600, 'tex_x_floor_pt')
    builder.link(tex_x_pt.outputs[0], tex_x_floor_pt.inputs[0])
    
    tex_x_plus_mus_pt = builder.math('ADD', 3900, -600, 'tex_x+mus_pt')
    builder.link(tex_x_floor_pt.outputs[0], tex_x_plus_mus_pt.inputs[0])
    builder.link(u_mu_s_pt.outputs[0], tex_x_plus_mus_pt.inputs[1])
    
    uvw_x_pt = builder.math('DIVIDE', 4050, -600, 'uvw_x_pt', v1=float(SCATTERING_TEXTURE_NU_SIZE))
    builder.link(tex_x_plus_mus_pt.outputs[0], uvw_x_pt.inputs[0])
    
    depth_scaled_pt = builder.math('MULTIPLY', 3600, -650, 'depth_sc_pt', v1=float(SCATTERING_TEXTURE_DEPTH - 1))
    builder.link(u_r_pt.outputs[0], depth_scaled_pt.inputs[0])
    
    depth_floor_pt = builder.math('FLOOR', 3750, -650, 'depth_floor_pt')
    builder.link(depth_scaled_pt.outputs[0], depth_floor_pt.inputs[0])
    
    slice_plus_uvw_pt = builder.math('ADD', 4200, -625, 'slice+uvw_pt')
    builder.link(depth_floor_pt.outputs[0], slice_plus_uvw_pt.inputs[0])
    builder.link(uvw_x_pt.outputs[0], slice_plus_uvw_pt.inputs[1])
    
    final_x_pt = builder.math('DIVIDE', 4350, -625, 'final_x_pt', v1=float(SCATTERING_TEXTURE_DEPTH))
    builder.link(slice_plus_uvw_pt.outputs[0], final_x_pt.inputs[0])
    
    u_mu_flip_pt = builder.math('SUBTRACT', 4200, -675, 'u_mu_flip_pt', v0=1.0)
    builder.link(u_mu_pt.outputs[0], u_mu_flip_pt.inputs[1])
    
    scat_uv_pt = builder.combine_xyz(4500, -650, 'Scat_UV_Pt')
    builder.link(final_x_pt.outputs[0], scat_uv_pt.inputs['X'])
    builder.link(u_mu_flip_pt.outputs[0], scat_uv_pt.inputs['Y'])
    
    tex_scat_pt = builder.image_texture(4650, -650, 'Scat_Pt', scattering_path)
    builder.link(scat_uv_pt.outputs[0], tex_scat_pt.inputs['Vector'])
    
    # =========================================================================
    # INSCATTER CALCULATION
    # inscatter = S_cam - transmittance × S_point
    # =========================================================================
    
    # transmittance × S_point
    t_times_scat = builder.vec_math('MULTIPLY', 4850, -200, 'T×S_pt')
    builder.link(tex_transmittance.outputs['Color'], t_times_scat.inputs[0])
    builder.link(tex_scat_pt.outputs['Color'], t_times_scat.inputs[1])
    
    # S_cam - T × S_point
    inscatter_raw = builder.vec_math('SUBTRACT', 5000, 0, 'Inscatter_Raw')
    builder.link(tex_scat_cam.outputs['Color'], inscatter_raw.inputs[0])
    builder.link(t_times_scat.outputs[0], inscatter_raw.inputs[1])
    
    # Clamp negative values
    inscatter_max = builder.vec_math('MAXIMUM', 5150, 0, 'Inscatter_Clamp')
    inscatter_max.inputs[1].default_value = (0.0, 0.0, 0.0)
    builder.link(inscatter_raw.outputs[0], inscatter_max.inputs[0])
    
    # =========================================================================
    # PHASE FUNCTIONS
    # =========================================================================
    
    # Rayleigh phase: 3/(16π) × (1 + ν²)
    nu_sq = builder.math('MULTIPLY', 4850, -400, 'ν²')
    builder.link(nu.outputs['Value'], nu_sq.inputs[0])
    builder.link(nu.outputs['Value'], nu_sq.inputs[1])
    
    one_plus_nu_sq = builder.math('ADD', 5000, -400, '1+ν²', v0=1.0)
    builder.link(nu_sq.outputs[0], one_plus_nu_sq.inputs[1])
    
    rayleigh_phase = builder.math('MULTIPLY', 5150, -400, 'Ray_Phase', v0=3.0 / (16.0 * math.pi))
    builder.link(one_plus_nu_sq.outputs[0], rayleigh_phase.inputs[1])
    
    # Apply phase function to inscatter
    inscatter_phased = builder.vec_math('SCALE', 5300, -100, 'Inscatter_Phased')
    builder.link(inscatter_max.outputs[0], inscatter_phased.inputs[0])
    builder.link(rayleigh_phase.outputs[0], inscatter_phased.inputs['Scale'])
    
    # =========================================================================
    # OUTPUTS
    # =========================================================================
    
    builder.link(tex_transmittance.outputs['Color'], group_output.inputs['Transmittance'])
    builder.link(inscatter_phased.outputs[0], group_output.inputs['Inscatter'])
    
    # Store version
    group['helios_version'] = AERIAL_NODE_VERSION
    
    print(f"Helios: Created node group '{AERIAL_NODE_GROUP_NAME}' v{AERIAL_NODE_VERSION}")
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
