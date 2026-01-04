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
AERIAL_NODE_VERSION = 28  # DEBUG: Output raw inscatter (before phase) to diagnose saturated blue

# Minimum virtual camera altitude for atmospheric calculations (km)
# This prevents degenerate cases when camera is at planet surface
MIN_CAMERA_ALTITUDE = 0.5  # 500 meters minimum


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
# TRANSMITTANCE UV HELPER
# =============================================================================

def create_transmittance_uv(builder, r_socket, mu_socket, base_x, base_y, suffix=""):
    """
    Create transmittance texture UV coordinates.
    
    Reference: GetTransmittanceTextureUvFromRMu (functions.glsl lines 402-421)
    
    CRITICAL: x_mu is based on distance to top atmosphere boundary, NOT simple (mu+1)/2!
    
    x_r = rho / H  where rho = sqrt(r² - bottom²), H = sqrt(top² - bottom²)
    x_mu = (d - d_min) / (d_max - d_min)  where:
        d = DistanceToTopAtmosphereBoundary(r, mu) = -r*mu + sqrt(r²(μ²-1) + top²)
        d_min = top - r
        d_max = rho + H
    
    Returns: UV socket
    """
    # r² for multiple calculations
    r_sq = builder.math('MULTIPLY', base_x, base_y, f't_r²{suffix}')
    builder.link(r_socket, r_sq.inputs[0])
    builder.link(r_socket, r_sq.inputs[1])
    
    # rho = sqrt(r² - bottom²)
    rho_sq = builder.math('SUBTRACT', base_x + 150, base_y, f't_rho²{suffix}', 
                          v1=BOTTOM_RADIUS * BOTTOM_RADIUS)
    builder.link(r_sq.outputs[0], rho_sq.inputs[0])
    
    rho_sq_safe = builder.math('MAXIMUM', base_x + 300, base_y, f't_rho²_safe{suffix}', v1=0.0)
    builder.link(rho_sq.outputs[0], rho_sq_safe.inputs[0])
    
    rho = builder.math('SQRT', base_x + 450, base_y, f't_rho{suffix}')
    builder.link(rho_sq_safe.outputs[0], rho.inputs[0])
    
    # x_r = rho / H
    x_r = builder.math('DIVIDE', base_x + 600, base_y, f't_x_r{suffix}', v1=H)
    builder.link(rho.outputs[0], x_r.inputs[0])
    
    # Apply texture coord transform for V (altitude)
    v_scale = 1.0 - 1.0 / TRANSMITTANCE_TEXTURE_HEIGHT
    v_offset = 0.5 / TRANSMITTANCE_TEXTURE_HEIGHT
    
    v_scaled = builder.math('MULTIPLY', base_x + 750, base_y, f't_v_sc{suffix}', v1=v_scale)
    builder.link(x_r.outputs[0], v_scaled.inputs[0])
    
    v_coord = builder.math('ADD', base_x + 900, base_y, f't_v{suffix}', v1=v_offset)
    builder.link(v_scaled.outputs[0], v_coord.inputs[0])
    
    # Flip V for Blender texture convention
    v_flip = builder.math('SUBTRACT', base_x + 1050, base_y, f't_v_flip{suffix}', v0=1.0)
    builder.link(v_coord.outputs[0], v_flip.inputs[1])
    
    # =========================================================================
    # x_mu: Based on DISTANCE to top atmosphere boundary (NOT simple linear!)
    # Reference: d = -r*mu + sqrt(r²(μ²-1) + top²)
    # =========================================================================
    
    # μ², μ²-1
    mu_sq = builder.math('MULTIPLY', base_x, base_y - 50, f't_μ²{suffix}')
    builder.link(mu_socket, mu_sq.inputs[0])
    builder.link(mu_socket, mu_sq.inputs[1])
    
    mu_sq_m1 = builder.math('SUBTRACT', base_x + 150, base_y - 50, f't_μ²-1{suffix}', v1=1.0)
    builder.link(mu_sq.outputs[0], mu_sq_m1.inputs[0])
    
    # r²×(μ²-1)
    r_sq_term = builder.math('MULTIPLY', base_x + 300, base_y - 50, f't_r²×term{suffix}')
    builder.link(r_sq.outputs[0], r_sq_term.inputs[0])
    builder.link(mu_sq_m1.outputs[0], r_sq_term.inputs[1])
    
    # discriminant = r²×(μ²-1) + top²
    disc = builder.math('ADD', base_x + 450, base_y - 50, f't_disc{suffix}', 
                        v1=TOP_RADIUS * TOP_RADIUS)
    builder.link(r_sq_term.outputs[0], disc.inputs[0])
    
    disc_safe = builder.math('MAXIMUM', base_x + 600, base_y - 50, f't_disc_safe{suffix}', v1=0.0)
    builder.link(disc.outputs[0], disc_safe.inputs[0])
    
    disc_sqrt = builder.math('SQRT', base_x + 750, base_y - 50, f't_√disc{suffix}')
    builder.link(disc_safe.outputs[0], disc_sqrt.inputs[0])
    
    # -r*mu
    r_mu = builder.math('MULTIPLY', base_x + 300, base_y - 100, f't_r×μ{suffix}')
    builder.link(r_socket, r_mu.inputs[0])
    builder.link(mu_socket, r_mu.inputs[1])
    
    neg_r_mu = builder.math('MULTIPLY', base_x + 450, base_y - 100, f't_-r×μ{suffix}', v1=-1.0)
    builder.link(r_mu.outputs[0], neg_r_mu.inputs[0])
    
    # d = -r*mu + sqrt(disc)
    d = builder.math('ADD', base_x + 900, base_y - 75, f't_d{suffix}')
    builder.link(neg_r_mu.outputs[0], d.inputs[0])
    builder.link(disc_sqrt.outputs[0], d.inputs[1])
    
    # d_min = top - r
    d_min = builder.math('SUBTRACT', base_x + 600, base_y - 125, f't_d_min{suffix}', v0=TOP_RADIUS)
    builder.link(r_socket, d_min.inputs[1])
    
    # d_max = rho + H
    d_max = builder.math('ADD', base_x + 750, base_y - 125, f't_d_max{suffix}', v1=H)
    builder.link(rho.outputs[0], d_max.inputs[0])
    
    # x_mu = (d - d_min) / (d_max - d_min)
    d_minus_dmin = builder.math('SUBTRACT', base_x + 1050, base_y - 75, f't_d-dmin{suffix}')
    builder.link(d.outputs[0], d_minus_dmin.inputs[0])
    builder.link(d_min.outputs[0], d_minus_dmin.inputs[1])
    
    dmax_minus_dmin = builder.math('SUBTRACT', base_x + 1050, base_y - 125, f't_dmax-dmin{suffix}')
    builder.link(d_max.outputs[0], dmax_minus_dmin.inputs[0])
    builder.link(d_min.outputs[0], dmax_minus_dmin.inputs[1])
    
    denom_safe = builder.math('MAXIMUM', base_x + 1200, base_y - 125, f't_denom_safe{suffix}', v1=0.001)
    builder.link(dmax_minus_dmin.outputs[0], denom_safe.inputs[0])
    
    x_mu = builder.math('DIVIDE', base_x + 1350, base_y - 100, f't_x_mu{suffix}')
    builder.link(d_minus_dmin.outputs[0], x_mu.inputs[0])
    builder.link(denom_safe.outputs[0], x_mu.inputs[1])
    
    # Clamp x_mu to [0, 1]
    x_mu_clamped = builder.math('MINIMUM', base_x + 1500, base_y - 100, f't_x_mu_clamp{suffix}', v1=1.0)
    builder.link(x_mu.outputs[0], x_mu_clamped.inputs[0])
    x_mu_final = builder.math('MAXIMUM', base_x + 1650, base_y - 100, f't_x_mu_final{suffix}', v1=0.0)
    builder.link(x_mu_clamped.outputs[0], x_mu_final.inputs[0])
    
    # Apply texture coord transform for U
    u_scale = 1.0 - 1.0 / TRANSMITTANCE_TEXTURE_WIDTH
    u_offset = 0.5 / TRANSMITTANCE_TEXTURE_WIDTH
    
    u_scaled = builder.math('MULTIPLY', base_x + 1800, base_y - 100, f't_u_sc{suffix}', v1=u_scale)
    builder.link(x_mu_final.outputs[0], u_scaled.inputs[0])
    
    u_coord = builder.math('ADD', base_x + 1950, base_y - 100, f't_u{suffix}', v1=u_offset)
    builder.link(u_scaled.outputs[0], u_coord.inputs[0])
    
    # Combine UV with flipped V
    uv = builder.combine_xyz(base_x + 2100, base_y - 50, f'Trans_UV{suffix}')
    builder.link(u_coord.outputs[0], uv.inputs['X'])
    builder.link(v_flip.outputs[0], uv.inputs['Y'])
    
    return uv.outputs[0]


# =============================================================================
# SCATTERING TEXTURE SAMPLER - Full sampling with depth interpolation
# =============================================================================

def sample_scattering_texture(builder, r_socket, mu_socket, mu_s_socket, nu_socket,
                               scattering_path, base_x, base_y, suffix="",
                               ray_intersects_ground_socket=None, return_u_mu=False):
    """
    Sample scattering texture with proper depth slice interpolation.
    
    Args:
        ray_intersects_ground_socket: If provided, uses this to select u_mu formula.
                                      Must be same for camera and point lookups per reference.
        return_u_mu: If True, returns (color_socket, u_mu_socket, u_mu_ground_socket) for debugging.
    
    Returns the interpolated scattering color socket (or tuple if return_u_mu=True).
    """
    # First compute all UV components (now also returns u_mu_ground for debugging)
    u_r, u_mu, u_mu_s, x_nu, u_mu_ground_debug = _compute_scattering_uvwz(
        builder, r_socket, mu_socket, mu_s_socket, nu_socket,
        base_x, base_y, suffix, ray_intersects_ground_socket
    )
    
    # Store u_mu for potential debug output
    u_mu_for_debug = u_mu
    u_mu_ground_for_debug = u_mu_ground_debug
    
    # Compute nu slice index and fraction
    tex_coord_x = builder.math('MULTIPLY', base_x + 2400, base_y, f'tex_x{suffix}', 
                               v1=float(SCATTERING_TEXTURE_NU_SIZE - 1))
    builder.link(x_nu.outputs[0], tex_coord_x.inputs[0])
    
    tex_x_floor = builder.math('FLOOR', base_x + 2550, base_y, f'tex_x_floor{suffix}')
    builder.link(tex_coord_x.outputs[0], tex_x_floor.inputs[0])
    
    # uvw_x = (tex_x + u_mu_s) / NU_SIZE
    tex_x_plus_mus = builder.math('ADD', base_x + 2700, base_y, f'tex_x+mus{suffix}')
    builder.link(tex_x_floor.outputs[0], tex_x_plus_mus.inputs[0])
    builder.link(u_mu_s.outputs[0], tex_x_plus_mus.inputs[1])
    
    uvw_x = builder.math('DIVIDE', base_x + 2850, base_y, f'uvw_x{suffix}', 
                         v1=float(SCATTERING_TEXTURE_NU_SIZE))
    builder.link(tex_x_plus_mus.outputs[0], uvw_x.inputs[0])
    
    # Compute depth slice indices and fraction
    depth_scaled = builder.math('MULTIPLY', base_x + 2400, base_y - 100, f'depth_sc{suffix}', 
                                v1=float(SCATTERING_TEXTURE_DEPTH - 1))
    builder.link(u_r.outputs[0], depth_scaled.inputs[0])
    
    depth_floor = builder.math('FLOOR', base_x + 2550, base_y - 100, f'depth_floor{suffix}')
    builder.link(depth_scaled.outputs[0], depth_floor.inputs[0])
    
    depth_frac = builder.math('SUBTRACT', base_x + 2700, base_y - 100, f'depth_frac{suffix}')
    builder.link(depth_scaled.outputs[0], depth_frac.inputs[0])
    builder.link(depth_floor.outputs[0], depth_frac.inputs[1])
    
    # Depth ceil (clamped)
    depth_ceil = builder.math('ADD', base_x + 2550, base_y - 150, f'depth_ceil{suffix}', v1=1.0)
    builder.link(depth_floor.outputs[0], depth_ceil.inputs[0])
    
    depth_ceil_clamp = builder.math('MINIMUM', base_x + 2700, base_y - 150, f'depth_ceil_clamp{suffix}', 
                                    v1=float(SCATTERING_TEXTURE_DEPTH - 1))
    builder.link(depth_ceil.outputs[0], depth_ceil_clamp.inputs[0])
    
    # Compute final X for floor depth slice
    slice0_plus_uvw = builder.math('ADD', base_x + 3000, base_y, f'slice0+uvw{suffix}')
    builder.link(depth_floor.outputs[0], slice0_plus_uvw.inputs[0])
    builder.link(uvw_x.outputs[0], slice0_plus_uvw.inputs[1])
    
    final_x0 = builder.math('DIVIDE', base_x + 3150, base_y, f'final_x0{suffix}', 
                            v1=float(SCATTERING_TEXTURE_DEPTH))
    builder.link(slice0_plus_uvw.outputs[0], final_x0.inputs[0])
    
    # Compute final X for ceil depth slice
    slice1_plus_uvw = builder.math('ADD', base_x + 3000, base_y - 150, f'slice1+uvw{suffix}')
    builder.link(depth_ceil_clamp.outputs[0], slice1_plus_uvw.inputs[0])
    builder.link(uvw_x.outputs[0], slice1_plus_uvw.inputs[1])
    
    final_x1 = builder.math('DIVIDE', base_x + 3150, base_y - 150, f'final_x1{suffix}', 
                            v1=float(SCATTERING_TEXTURE_DEPTH))
    builder.link(slice1_plus_uvw.outputs[0], final_x1.inputs[0])
    
    # Y coordinate - flip for Blender texture convention (matches sky_nodes.py)
    u_mu_flip = builder.math('SUBTRACT', base_x + 3150, base_y - 50, f'u_mu_flip{suffix}', v0=1.0)
    builder.link(u_mu.outputs[0], u_mu_flip.inputs[1])
    
    # Sample depth slice 0
    uv0 = builder.combine_xyz(base_x + 3300, base_y, f'UV0{suffix}')
    builder.link(final_x0.outputs[0], uv0.inputs['X'])
    builder.link(u_mu_flip.outputs[0], uv0.inputs['Y'])
    
    tex0 = builder.image_texture(base_x + 3450, base_y, f'Scat0{suffix}', scattering_path)
    builder.link(uv0.outputs[0], tex0.inputs['Vector'])
    
    # Sample depth slice 1
    uv1 = builder.combine_xyz(base_x + 3300, base_y - 150, f'UV1{suffix}')
    builder.link(final_x1.outputs[0], uv1.inputs['X'])
    builder.link(u_mu_flip.outputs[0], uv1.inputs['Y'])
    
    tex1 = builder.image_texture(base_x + 3450, base_y - 150, f'Scat1{suffix}', scattering_path)
    builder.link(uv1.outputs[0], tex1.inputs['Vector'])
    
    # Interpolate between depth slices
    mix = builder.mix('RGBA', 'MIX', base_x + 3650, base_y - 75, f'DepthMix{suffix}')
    builder.link(depth_frac.outputs[0], mix.inputs['Factor'])
    builder.link(tex0.outputs['Color'], mix.inputs[6])  # A
    builder.link(tex1.outputs['Color'], mix.inputs[7])  # B
    
    if return_u_mu:
        return mix.outputs[2], u_mu_for_debug.outputs[0], u_mu_ground_for_debug  # Result + u_mu + u_mu_ground
    return mix.outputs[2]  # Result


def _compute_scattering_uvwz(builder, r_socket, mu_socket, mu_s_socket, nu_socket,
                              base_x, base_y, suffix="", ray_intersects_ground_socket=None):
    """
    Compute scattering texture UV coordinates (u_r, u_mu, u_mu_s, x_nu).
    
    Reference: GetScatteringTextureUvwzFromRMuMuSNu (functions.glsl lines 773-830)
    
    CRITICAL: Ground and non-ground rays use COMPLETELY DIFFERENT formulas for u_mu:
    - Ground: d = -r*mu - sqrt(disc), d_min = r - bottom, d_max = rho, u_mu in [0, 0.5]
    - Non-ground: d = -r*mu + sqrt(disc + H²), d_min = top - r, d_max = rho + H, u_mu in [0.5, 1.0]
    
    Args:
        ray_intersects_ground_socket: If provided, selects between ground/non-ground u_mu formula.
                                      Value 1.0 = ground intersecting, 0.0 = non-ground.
    
    Returns (u_r_node, u_mu_node, u_mu_s_node, x_nu_node).
    """
    
    # =========================================================================
    # u_r: altitude coordinate (same for both ground and non-ground)
    # rho = sqrt(r² - bottom²), u_r = GetTextureCoordFromUnitRange(rho/H, R_SIZE)
    # =========================================================================
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
    
    # =========================================================================
    # u_mu: view zenith coordinate - TWO COMPLETELY DIFFERENT FORMULAS
    # Reference: GetScatteringTextureUvwzFromRMuMuSNu lines 789-812
    # =========================================================================
    
    # Common: μ², μ²-1, r²×(μ²-1), r*mu
    mu_sq = builder.math('MULTIPLY', base_x, base_y - 100, f'μ²{suffix}')
    builder.link(mu_socket, mu_sq.inputs[0])
    builder.link(mu_socket, mu_sq.inputs[1])
    
    mu_sq_m1 = builder.math('SUBTRACT', base_x + 150, base_y - 100, f'μ²-1{suffix}', v1=1.0)
    builder.link(mu_sq.outputs[0], mu_sq_m1.inputs[0])
    
    r_sq_x_mu_term = builder.math('MULTIPLY', base_x + 300, base_y - 100, f'r²×(μ²-1){suffix}')
    builder.link(r_sq.outputs[0], r_sq_x_mu_term.inputs[0])
    builder.link(mu_sq_m1.outputs[0], r_sq_x_mu_term.inputs[1])
    
    r_mu = builder.math('MULTIPLY', base_x + 300, base_y - 150, f'r×μ{suffix}')
    builder.link(r_socket, r_mu.inputs[0])
    builder.link(mu_socket, r_mu.inputs[1])
    
    neg_r_mu = builder.math('MULTIPLY', base_x + 450, base_y - 150, f'-r×μ{suffix}', v1=-1.0)
    builder.link(r_mu.outputs[0], neg_r_mu.inputs[0])
    
    # -------------------------------------------------------------------------
    # NON-GROUND PATH: d = -r*mu + sqrt(r²(μ²-1) + top²), maps to [0.5, 1.0]
    # -------------------------------------------------------------------------
    disc_nonground = builder.math('ADD', base_x + 450, base_y - 100, f'disc_ng{suffix}', 
                                  v1=TOP_RADIUS * TOP_RADIUS)
    builder.link(r_sq_x_mu_term.outputs[0], disc_nonground.inputs[0])
    
    disc_ng_safe = builder.math('MAXIMUM', base_x + 600, base_y - 100, f'disc_ng_safe{suffix}', v1=0.0)
    builder.link(disc_nonground.outputs[0], disc_ng_safe.inputs[0])
    
    disc_ng_sqrt = builder.math('SQRT', base_x + 750, base_y - 100, f'√disc_ng{suffix}')
    builder.link(disc_ng_safe.outputs[0], disc_ng_sqrt.inputs[0])
    
    # d_nonground = -r*mu + sqrt(disc)  (ADDITION)
    d_ng = builder.math('ADD', base_x + 900, base_y - 100, f'd_ng{suffix}')
    builder.link(neg_r_mu.outputs[0], d_ng.inputs[0])
    builder.link(disc_ng_sqrt.outputs[0], d_ng.inputs[1])
    
    # d_min_ng = top - r, d_max_ng = rho + H
    d_min_ng = builder.math('SUBTRACT', base_x + 600, base_y - 50, f'd_min_ng{suffix}', v0=TOP_RADIUS)
    builder.link(r_socket, d_min_ng.inputs[1])
    
    d_max_ng = builder.math('ADD', base_x + 750, base_y - 50, f'd_max_ng{suffix}', v1=H)
    builder.link(rho.outputs[0], d_max_ng.inputs[0])
    
    # x_mu_ng = (d - d_min) / (d_max - d_min)
    d_minus_dmin_ng = builder.math('SUBTRACT', base_x + 1050, base_y - 75, f'd-dmin_ng{suffix}')
    builder.link(d_ng.outputs[0], d_minus_dmin_ng.inputs[0])
    builder.link(d_min_ng.outputs[0], d_minus_dmin_ng.inputs[1])
    
    dmax_minus_dmin_ng = builder.math('SUBTRACT', base_x + 1050, base_y - 25, f'dmax-dmin_ng{suffix}')
    builder.link(d_max_ng.outputs[0], dmax_minus_dmin_ng.inputs[0])
    builder.link(d_min_ng.outputs[0], dmax_minus_dmin_ng.inputs[1])
    
    denom_ng_safe = builder.math('MAXIMUM', base_x + 1200, base_y - 25, f'denom_ng_safe{suffix}', v1=0.001)
    builder.link(dmax_minus_dmin_ng.outputs[0], denom_ng_safe.inputs[0])
    
    x_mu_ng_raw = builder.math('DIVIDE', base_x + 1350, base_y - 50, f'x_mu_ng_raw{suffix}')
    builder.link(d_minus_dmin_ng.outputs[0], x_mu_ng_raw.inputs[0])
    builder.link(denom_ng_safe.outputs[0], x_mu_ng_raw.inputs[1])
    
    # Clamp x_mu_ng to [0, 1]
    x_mu_ng_clamped = builder.math('MINIMUM', base_x + 1500, base_y - 50, f'x_mu_ng_clamp{suffix}', v1=1.0)
    builder.link(x_mu_ng_raw.outputs[0], x_mu_ng_clamped.inputs[0])
    x_mu_ng = builder.math('MAXIMUM', base_x + 1650, base_y - 50, f'x_mu_ng{suffix}', v1=0.0)
    builder.link(x_mu_ng_clamped.outputs[0], x_mu_ng.inputs[0])
    
    # GetTextureCoordFromUnitRange for MU_SIZE/2, then u_mu = 0.5 + 0.5 * coord
    mu_scale = 1.0 - 2.0 / SCATTERING_TEXTURE_MU_SIZE
    mu_offset = 1.0 / SCATTERING_TEXTURE_MU_SIZE
    
    x_mu_ng_scaled = builder.math('MULTIPLY', base_x + 1800, base_y - 50, f'x_mu_ng_sc{suffix}', v1=mu_scale)
    builder.link(x_mu_ng.outputs[0], x_mu_ng_scaled.inputs[0])
    
    x_mu_ng_offset = builder.math('ADD', base_x + 1950, base_y - 50, f'x_mu_ng_off{suffix}', v1=mu_offset)
    builder.link(x_mu_ng_scaled.outputs[0], x_mu_ng_offset.inputs[0])
    
    # u_mu_nonground = 0.5 + 0.5 * coord  -> maps to [0.5, 1.0]
    u_mu_ng_half = builder.math('MULTIPLY', base_x + 2100, base_y - 50, f'u_mu_ng_half{suffix}', v1=0.5)
    builder.link(x_mu_ng_offset.outputs[0], u_mu_ng_half.inputs[0])
    
    u_mu_nonground = builder.math('ADD', base_x + 2250, base_y - 50, f'u_mu_ng{suffix}', v0=0.5)
    builder.link(u_mu_ng_half.outputs[0], u_mu_nonground.inputs[1])
    
    # -------------------------------------------------------------------------
    # GROUND PATH: d = -r*mu - sqrt(r²(μ²-1) + bottom²), maps to [0, 0.5]
    # Reference: lines 795-802 - uses SUBTRACTION and BOTTOM radius
    # -------------------------------------------------------------------------
    disc_ground = builder.math('ADD', base_x + 450, base_y - 200, f'disc_g{suffix}', 
                               v1=BOTTOM_RADIUS * BOTTOM_RADIUS)
    builder.link(r_sq_x_mu_term.outputs[0], disc_ground.inputs[0])
    
    disc_g_safe = builder.math('MAXIMUM', base_x + 600, base_y - 200, f'disc_g_safe{suffix}', v1=0.0)
    builder.link(disc_ground.outputs[0], disc_g_safe.inputs[0])
    
    disc_g_sqrt = builder.math('SQRT', base_x + 750, base_y - 200, f'√disc_g{suffix}')
    builder.link(disc_g_safe.outputs[0], disc_g_sqrt.inputs[0])
    
    # d_ground = -r*mu - sqrt(disc)  (SUBTRACTION!)
    d_g = builder.math('SUBTRACT', base_x + 900, base_y - 200, f'd_g{suffix}')
    builder.link(neg_r_mu.outputs[0], d_g.inputs[0])
    builder.link(disc_g_sqrt.outputs[0], d_g.inputs[1])
    
    # d_min_g = r - bottom, d_max_g = rho
    d_min_g = builder.math('SUBTRACT', base_x + 600, base_y - 250, f'd_min_g{suffix}', v1=BOTTOM_RADIUS)
    builder.link(r_socket, d_min_g.inputs[0])
    
    # d_max_g = rho (already computed)
    
    # x_mu_g = (d - d_min) / (d_max - d_min)
    d_minus_dmin_g = builder.math('SUBTRACT', base_x + 1050, base_y - 200, f'd-dmin_g{suffix}')
    builder.link(d_g.outputs[0], d_minus_dmin_g.inputs[0])
    builder.link(d_min_g.outputs[0], d_minus_dmin_g.inputs[1])
    
    dmax_minus_dmin_g = builder.math('SUBTRACT', base_x + 1050, base_y - 250, f'dmax-dmin_g{suffix}')
    builder.link(rho.outputs[0], dmax_minus_dmin_g.inputs[0])  # d_max_g = rho
    builder.link(d_min_g.outputs[0], dmax_minus_dmin_g.inputs[1])
    
    denom_g_safe = builder.math('MAXIMUM', base_x + 1200, base_y - 250, f'denom_g_safe{suffix}', v1=0.001)
    builder.link(dmax_minus_dmin_g.outputs[0], denom_g_safe.inputs[0])
    
    # Handle d_max == d_min case (reference: d_max == d_min ? 0.0 : ...)
    x_mu_g_raw = builder.math('DIVIDE', base_x + 1350, base_y - 225, f'x_mu_g_raw{suffix}')
    builder.link(d_minus_dmin_g.outputs[0], x_mu_g_raw.inputs[0])
    builder.link(denom_g_safe.outputs[0], x_mu_g_raw.inputs[1])
    
    # Clamp x_mu_g to [0, 1]
    x_mu_g_clamped = builder.math('MINIMUM', base_x + 1500, base_y - 225, f'x_mu_g_clamp{suffix}', v1=1.0)
    builder.link(x_mu_g_raw.outputs[0], x_mu_g_clamped.inputs[0])
    x_mu_g = builder.math('MAXIMUM', base_x + 1650, base_y - 225, f'x_mu_g{suffix}', v1=0.0)
    builder.link(x_mu_g_clamped.outputs[0], x_mu_g.inputs[0])
    
    # GetTextureCoordFromUnitRange for MU_SIZE/2, then u_mu = 0.5 - 0.5 * coord
    x_mu_g_scaled = builder.math('MULTIPLY', base_x + 1800, base_y - 225, f'x_mu_g_sc{suffix}', v1=mu_scale)
    builder.link(x_mu_g.outputs[0], x_mu_g_scaled.inputs[0])
    
    x_mu_g_offset = builder.math('ADD', base_x + 1950, base_y - 225, f'x_mu_g_off{suffix}', v1=mu_offset)
    builder.link(x_mu_g_scaled.outputs[0], x_mu_g_offset.inputs[0])
    
    # u_mu_ground = 0.5 - 0.5 * coord  -> maps to [0, 0.5]
    u_mu_g_half = builder.math('MULTIPLY', base_x + 2100, base_y - 225, f'u_mu_g_half{suffix}', v1=0.5)
    builder.link(x_mu_g_offset.outputs[0], u_mu_g_half.inputs[0])
    
    u_mu_ground = builder.math('SUBTRACT', base_x + 2250, base_y - 225, f'u_mu_g{suffix}', v0=0.5)
    builder.link(u_mu_g_half.outputs[0], u_mu_ground.inputs[1])
    
    # -------------------------------------------------------------------------
    # SELECT u_mu based on ray_intersects_ground
    # -------------------------------------------------------------------------
    if ray_intersects_ground_socket is not None:
        # Mix: factor=0 -> non-ground (A), factor=1 -> ground (B)
        u_mu_mix = builder.mix('FLOAT', 'MIX', base_x + 2400, base_y - 137, f'u_mu_sel{suffix}')
        builder.link(ray_intersects_ground_socket, u_mu_mix.inputs['Factor'])
        builder.link(u_mu_nonground.outputs[0], u_mu_mix.inputs[2])  # A (non-ground)
        builder.link(u_mu_ground.outputs[0], u_mu_mix.inputs[3])  # B (ground)
        u_mu_selected = u_mu_mix.outputs[0]
    else:
        # Default to non-ground formula
        u_mu_selected = u_mu_nonground.outputs[0]
    
    # Clamp u_mu to [0, 1]
    u_mu_min = builder.math('MAXIMUM', base_x + 2550, base_y - 137, f'u_mu_min{suffix}', v1=0.0)
    builder.link(u_mu_selected, u_mu_min.inputs[0])
    u_mu_final = builder.math('MINIMUM', base_x + 2700, base_y - 137, f'u_mu_final{suffix}', v1=1.0)
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
    
    # Return all UV components + u_mu_ground for debugging
    return u_r_final, u_mu_final, u_mu_s, x_nu, u_mu_ground.outputs[0]


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
    
    # r = length(camera), clamped to minimum altitude above ground
    # This prevents degenerate cases when camera is at planet surface
    r_raw = builder.vec_math('LENGTH', -1400, 300, 'r_raw')
    builder.link(camera_km.outputs[0], r_raw.inputs[0])
    
    # Enforce minimum altitude: r >= bottom_radius + MIN_CAMERA_ALTITUDE
    min_r = BOTTOM_RADIUS + MIN_CAMERA_ALTITUDE
    r_min = builder.math('MAXIMUM', -1200, 300, 'r_min', v1=min_r)
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
    # RAY INTERSECTS GROUND - compute FIRST, needed for transmittance and scattering
    # Reference: RayIntersectsGround(r, mu) = mu < 0 && r²(μ²-1) + bottom² >= 0
    # =========================================================================
    
    # Condition 1: mu < 0
    mu_negative = builder.math('LESS_THAN', 2000, 600, 'mu<0', v1=0.0)
    builder.link(mu_final.outputs[0], mu_negative.inputs[0])
    
    # Condition 2: r²(μ²-1) + bottom² >= 0
    r_sq_ground = builder.math('MULTIPLY', 2000, 550, 'r²_g')
    builder.link(r.outputs[0], r_sq_ground.inputs[0])
    builder.link(r.outputs[0], r_sq_ground.inputs[1])
    
    mu_sq_ground = builder.math('MULTIPLY', 2000, 500, 'μ²_g')
    builder.link(mu_final.outputs[0], mu_sq_ground.inputs[0])
    builder.link(mu_final.outputs[0], mu_sq_ground.inputs[1])
    
    mu_sq_m1_ground = builder.math('SUBTRACT', 2150, 500, 'μ²-1_g', v1=1.0)
    builder.link(mu_sq_ground.outputs[0], mu_sq_m1_ground.inputs[0])
    
    r_sq_term_ground = builder.math('MULTIPLY', 2300, 525, 'r²×(μ²-1)')
    builder.link(r_sq_ground.outputs[0], r_sq_term_ground.inputs[0])
    builder.link(mu_sq_m1_ground.outputs[0], r_sq_term_ground.inputs[1])
    
    discriminant_ground = builder.math('ADD', 2450, 525, 'disc_g', v1=BOTTOM_RADIUS * BOTTOM_RADIUS)
    builder.link(r_sq_term_ground.outputs[0], discriminant_ground.inputs[0])
    
    # disc >= 0
    disc_ge_zero = builder.math('GREATER_THAN', 2600, 525, 'disc>=0', v1=-0.0001)
    builder.link(discriminant_ground.outputs[0], disc_ge_zero.inputs[0])
    
    # Both conditions must be true: mu < 0 AND disc >= 0
    ray_intersects_ground = builder.math('MULTIPLY', 2750, 550, 'ray_hits_ground')
    builder.link(mu_negative.outputs[0], ray_intersects_ground.inputs[0])
    builder.link(disc_ge_zero.outputs[0], ray_intersects_ground.inputs[1])
    
    # =========================================================================
    # TRANSMITTANCE - Reference: GetTransmittance (functions.glsl lines 493-518)
    # 
    # CRITICAL: For ground-intersecting rays, uses NEGATED mu values and swaps order:
    #   Non-ground: T = T(r, mu) / T(r_d, mu_d)
    #   Ground:     T = T(r_d, -mu_d) / T(r, -mu)
    # =========================================================================
    
    # Compute negated mu values for ground case
    neg_mu = builder.math('MULTIPLY', -600, 600, '-μ', v1=-1.0)
    builder.link(mu_final.outputs[0], neg_mu.inputs[0])
    
    neg_mu_p = builder.math('MULTIPLY', -600, 400, '-μ_p', v1=-1.0)
    builder.link(mu_p_final.outputs[0], neg_mu_p.inputs[0])
    
    # --- NON-GROUND PATH: T(r, mu) / T(r_p, mu_p) ---
    trans_uv_cam_ng = create_transmittance_uv(
        builder, r.outputs[0], mu_final.outputs[0], 
        -400, 700, "_cam_ng"
    )
    trans_uv_pt_ng = create_transmittance_uv(
        builder, r_p.outputs[0], mu_p_final.outputs[0],
        -400, 500, "_pt_ng"
    )
    
    tex_trans_cam_ng = builder.image_texture(800, 700, 'T_cam_ng', transmittance_path)
    builder.link(trans_uv_cam_ng, tex_trans_cam_ng.inputs['Vector'])
    
    tex_trans_pt_ng = builder.image_texture(800, 500, 'T_pt_ng', transmittance_path)
    builder.link(trans_uv_pt_ng, tex_trans_pt_ng.inputs['Vector'])
    
    # T_ng = T_cam / T_pt (non-ground)
    sep_cam_ng = builder.nodes.new('ShaderNodeSeparateColor')
    sep_cam_ng.location = (950, 700)
    sep_cam_ng.name = 'Sep_T_cam_ng'
    builder.link(tex_trans_cam_ng.outputs['Color'], sep_cam_ng.inputs['Color'])
    
    sep_pt_ng = builder.nodes.new('ShaderNodeSeparateColor')
    sep_pt_ng.location = (950, 500)
    sep_pt_ng.name = 'Sep_T_pt_ng'
    builder.link(tex_trans_pt_ng.outputs['Color'], sep_pt_ng.inputs['Color'])
    
    # Safe division for non-ground (T_cam / T_pt)
    pt_ng_safe_r = builder.math('MAXIMUM', 1100, 550, 'T_pt_ng_safe_r', v1=0.0001)
    builder.link(sep_pt_ng.outputs['Red'], pt_ng_safe_r.inputs[0])
    pt_ng_safe_g = builder.math('MAXIMUM', 1100, 500, 'T_pt_ng_safe_g', v1=0.0001)
    builder.link(sep_pt_ng.outputs['Green'], pt_ng_safe_g.inputs[0])
    pt_ng_safe_b = builder.math('MAXIMUM', 1100, 450, 'T_pt_ng_safe_b', v1=0.0001)
    builder.link(sep_pt_ng.outputs['Blue'], pt_ng_safe_b.inputs[0])
    
    T_ng_r = builder.math('DIVIDE', 1250, 550, 'T_ng_r')
    builder.link(sep_cam_ng.outputs['Red'], T_ng_r.inputs[0])
    builder.link(pt_ng_safe_r.outputs[0], T_ng_r.inputs[1])
    T_ng_g = builder.math('DIVIDE', 1250, 500, 'T_ng_g')
    builder.link(sep_cam_ng.outputs['Green'], T_ng_g.inputs[0])
    builder.link(pt_ng_safe_g.outputs[0], T_ng_g.inputs[1])
    T_ng_b = builder.math('DIVIDE', 1250, 450, 'T_ng_b')
    builder.link(sep_cam_ng.outputs['Blue'], T_ng_b.inputs[0])
    builder.link(pt_ng_safe_b.outputs[0], T_ng_b.inputs[1])
    
    # --- GROUND PATH: T(r_p, -mu_p) / T(r, -mu) ---
    trans_uv_cam_g = create_transmittance_uv(
        builder, r.outputs[0], neg_mu.outputs[0], 
        -400, 300, "_cam_g"
    )
    trans_uv_pt_g = create_transmittance_uv(
        builder, r_p.outputs[0], neg_mu_p.outputs[0],
        -400, 100, "_pt_g"
    )
    
    tex_trans_cam_g = builder.image_texture(800, 300, 'T_cam_g', transmittance_path)
    builder.link(trans_uv_cam_g, tex_trans_cam_g.inputs['Vector'])
    
    tex_trans_pt_g = builder.image_texture(800, 100, 'T_pt_g', transmittance_path)
    builder.link(trans_uv_pt_g, tex_trans_pt_g.inputs['Vector'])
    
    # T_g = T_pt(-mu_p) / T_cam(-mu) (ground - SWAPPED!)
    sep_cam_g = builder.nodes.new('ShaderNodeSeparateColor')
    sep_cam_g.location = (950, 300)
    sep_cam_g.name = 'Sep_T_cam_g'
    builder.link(tex_trans_cam_g.outputs['Color'], sep_cam_g.inputs['Color'])
    
    sep_pt_g = builder.nodes.new('ShaderNodeSeparateColor')
    sep_pt_g.location = (950, 100)
    sep_pt_g.name = 'Sep_T_pt_g'
    builder.link(tex_trans_pt_g.outputs['Color'], sep_pt_g.inputs['Color'])
    
    # Safe division for ground (T_pt / T_cam - swapped!)
    cam_g_safe_r = builder.math('MAXIMUM', 1100, 350, 'T_cam_g_safe_r', v1=0.0001)
    builder.link(sep_cam_g.outputs['Red'], cam_g_safe_r.inputs[0])
    cam_g_safe_g = builder.math('MAXIMUM', 1100, 300, 'T_cam_g_safe_g', v1=0.0001)
    builder.link(sep_cam_g.outputs['Green'], cam_g_safe_g.inputs[0])
    cam_g_safe_b = builder.math('MAXIMUM', 1100, 250, 'T_cam_g_safe_b', v1=0.0001)
    builder.link(sep_cam_g.outputs['Blue'], cam_g_safe_b.inputs[0])
    
    T_g_r = builder.math('DIVIDE', 1250, 350, 'T_g_r')
    builder.link(sep_pt_g.outputs['Red'], T_g_r.inputs[0])
    builder.link(cam_g_safe_r.outputs[0], T_g_r.inputs[1])
    T_g_g = builder.math('DIVIDE', 1250, 300, 'T_g_g')
    builder.link(sep_pt_g.outputs['Green'], T_g_g.inputs[0])
    builder.link(cam_g_safe_g.outputs[0], T_g_g.inputs[1])
    T_g_b = builder.math('DIVIDE', 1250, 250, 'T_g_b')
    builder.link(sep_pt_g.outputs['Blue'], T_g_b.inputs[0])
    builder.link(cam_g_safe_b.outputs[0], T_g_b.inputs[1])
    
    # --- SELECT based on ray_intersects_ground ---
    T_sel_r = builder.mix('FLOAT', 'MIX', 1400, 500, 'T_sel_r')
    builder.link(ray_intersects_ground.outputs[0], T_sel_r.inputs['Factor'])
    builder.link(T_ng_r.outputs[0], T_sel_r.inputs[2])  # A = non-ground
    builder.link(T_g_r.outputs[0], T_sel_r.inputs[3])   # B = ground
    
    T_sel_g = builder.mix('FLOAT', 'MIX', 1400, 450, 'T_sel_g')
    builder.link(ray_intersects_ground.outputs[0], T_sel_g.inputs['Factor'])
    builder.link(T_ng_g.outputs[0], T_sel_g.inputs[2])
    builder.link(T_g_g.outputs[0], T_sel_g.inputs[3])
    
    T_sel_b = builder.mix('FLOAT', 'MIX', 1400, 400, 'T_sel_b')
    builder.link(ray_intersects_ground.outputs[0], T_sel_b.inputs['Factor'])
    builder.link(T_ng_b.outputs[0], T_sel_b.inputs[2])
    builder.link(T_g_b.outputs[0], T_sel_b.inputs[3])
    
    # Clamp to [0, 1]
    trans_r_clamp = builder.math('MINIMUM', 1550, 500, 'T_r_clamp', v1=1.0)
    builder.link(T_sel_r.outputs[0], trans_r_clamp.inputs[0])
    trans_g_clamp = builder.math('MINIMUM', 1550, 450, 'T_g_clamp', v1=1.0)
    builder.link(T_sel_g.outputs[0], trans_g_clamp.inputs[0])
    trans_b_clamp = builder.math('MINIMUM', 1550, 400, 'T_b_clamp', v1=1.0)
    builder.link(T_sel_b.outputs[0], trans_b_clamp.inputs[0])
    
    # Combine back to color
    transmittance_final = builder.combine_xyz(1700, 450, 'Transmittance_Final')
    builder.link(trans_r_clamp.outputs[0], transmittance_final.inputs['X'])
    builder.link(trans_g_clamp.outputs[0], transmittance_final.inputs['Y'])
    builder.link(trans_b_clamp.outputs[0], transmittance_final.inputs['Z'])
    
    # =========================================================================
    # SCATTERING LOOKUPS - Camera and Point (with depth interpolation)
    # Both use SAME ray_intersects_ground from camera params (per reference)
    # =========================================================================
    
    # Sample scattering at camera position with depth interpolation
    scat_cam_color = sample_scattering_texture(
        builder, r.outputs[0], mu_final.outputs[0], mu_s_final.outputs[0], nu.outputs['Value'],
        scattering_path, 1800, 200, "_cam", ray_intersects_ground.outputs[0]
    )
    
    # Sample scattering at point position with depth interpolation
    # CRITICAL: Use SAME ray_intersects_ground from camera, not from point!
    scat_pt_color = sample_scattering_texture(
        builder, r_p.outputs[0], mu_p_final.outputs[0], mu_s_p_final.outputs[0], nu.outputs['Value'],
        scattering_path, 1800, -400, "_pt", ray_intersects_ground.outputs[0]
    )
    
    # =========================================================================
    # INSCATTER CALCULATION
    # inscatter = S_cam - transmittance × S_point
    # =========================================================================
    
    # transmittance × S_point
    t_times_scat = builder.vec_math('MULTIPLY', 5500, -200, 'T×S_pt')
    builder.link(transmittance_final.outputs[0], t_times_scat.inputs[0])
    builder.link(scat_pt_color, t_times_scat.inputs[1])
    
    # S_cam - T × S_point
    inscatter_raw = builder.vec_math('SUBTRACT', 5650, 0, 'Inscatter_Raw')
    builder.link(scat_cam_color, inscatter_raw.inputs[0])
    builder.link(t_times_scat.outputs[0], inscatter_raw.inputs[1])
    
    # Clamp negative values
    inscatter_max = builder.vec_math('MAXIMUM', 5800, 0, 'Inscatter_Clamp')
    inscatter_max.inputs[1].default_value = (0.0, 0.0, 0.0)
    builder.link(inscatter_raw.outputs[0], inscatter_max.inputs[0])
    
    # =========================================================================
    # PHASE FUNCTIONS
    # =========================================================================
    
    # Rayleigh phase: 3/(16π) × (1 + ν²)
    nu_sq = builder.math('MULTIPLY', 5500, -400, 'ν²')
    builder.link(nu.outputs['Value'], nu_sq.inputs[0])
    builder.link(nu.outputs['Value'], nu_sq.inputs[1])
    
    one_plus_nu_sq = builder.math('ADD', 5650, -400, '1+ν²', v0=1.0)
    builder.link(nu_sq.outputs[0], one_plus_nu_sq.inputs[1])
    
    rayleigh_phase = builder.math('MULTIPLY', 5800, -400, 'Ray_Phase', v0=3.0 / (16.0 * math.pi))
    builder.link(one_plus_nu_sq.outputs[0], rayleigh_phase.inputs[1])
    
    # Apply phase function to inscatter
    inscatter_phased = builder.vec_math('SCALE', 5950, -100, 'Inscatter_Phased')
    builder.link(inscatter_max.outputs[0], inscatter_phased.inputs[0])
    builder.link(rayleigh_phase.outputs[0], inscatter_phased.inputs['Scale'])
    
    # DEBUG: Also output S_cam directly (like sky shader) to compare
    scat_cam_phased = builder.vec_math('SCALE', 5950, 100, 'S_cam_Phased')
    builder.link(scat_cam_color, scat_cam_phased.inputs[0])
    builder.link(rayleigh_phase.outputs[0], scat_cam_phased.inputs['Scale'])
    
    # DEBUG: Output S_point to check point scattering lookup
    scat_pt_phased = builder.vec_math('SCALE', 5950, 200, 'S_pt_Phased')
    builder.link(scat_pt_color, scat_pt_phased.inputs[0])
    builder.link(rayleigh_phase.outputs[0], scat_pt_phased.inputs['Scale'])
    
    # =========================================================================
    # OUTPUTS
    # =========================================================================
    
    builder.link(transmittance_final.outputs[0], group_output.inputs['Transmittance'])
    
    # DEBUG V28: Output RAW inscatter (before phase function) to see if saturation is from scattering or phase
    # This helps diagnose whether the saturated blue is in the LUT or the calculation
    builder.link(inscatter_max.outputs[0], group_output.inputs['Inscatter'])
    
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
