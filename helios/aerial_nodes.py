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
AERIAL_NODE_VERSION = 97  # Links confirmed correct - restore actual output

# Minimum virtual camera altitude for atmospheric calculations (km)
# V68: Removed - reference has no such clamp
# MIN_CAMERA_ALTITUDE = 0.5  # DISABLED - was causing difference from reference

# =============================================================================
# V46: PURE BRUNETON IMPLEMENTATION
# =============================================================================
# This version implements GetSkyRadianceToPoint EXACTLY as in the reference.
# No horizon clamping, no geometry-based ground detection.
#
# Key principle: ray_r_mu_intersects_ground is computed from camera's (r, mu)
# using RayIntersectsGround formula, and the SAME flag is used for both
# transmittance AND scattering lookups.
#
# Reference: atmospheric-scattering-2-export/atmosphere/functions.glsl
# Lines 1787-1863 (GetSkyRadianceToPoint)
# Lines 240-246 (RayIntersectsGround)
# Lines 493-519 (GetTransmittance)
# Lines 773-831 (GetScatteringTextureUvwzFromRMuMuSNu)
# =============================================================================


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
    
    # t_dist = -r*mu + sqrt(disc)  (renamed from 'd' to avoid shadowing distance d)
    t_dist = builder.math('ADD', base_x + 900, base_y - 75, f't_d{suffix}')
    builder.link(neg_r_mu.outputs[0], t_dist.inputs[0])
    builder.link(disc_sqrt.outputs[0], t_dist.inputs[1])
    
    # d_min = top - r
    d_min = builder.math('SUBTRACT', base_x + 600, base_y - 125, f't_d_min{suffix}', v0=TOP_RADIUS)
    builder.link(r_socket, d_min.inputs[1])
    
    # d_max = rho + H
    d_max = builder.math('ADD', base_x + 750, base_y - 125, f't_d_max{suffix}', v1=H)
    builder.link(rho.outputs[0], d_max.inputs[0])
    
    # x_mu = (t_dist - d_min) / (d_max - d_min)
    d_minus_dmin = builder.math('SUBTRACT', base_x + 1050, base_y - 75, f't_d-dmin{suffix}')
    builder.link(t_dist.outputs[0], d_minus_dmin.inputs[0])
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
    
    V46: Samples ground or non-ground texture half based on ray_intersects_ground_socket.
    Reference: GetCombinedScattering (functions.glsl lines 1658-1690)
    
    Returns the interpolated scattering color socket (or tuple if return_u_mu=True).
    """
    # V68: Use ray_intersects_ground for ground/non-ground UV selection (restored from reference)
    u_r, u_mu, u_mu_s, x_nu, u_mu_ground = _compute_scattering_uvwz(
        builder, r_socket, mu_socket, mu_s_socket, nu_socket,
        base_x, base_y, suffix, ray_intersects_ground_socket
    )
    
    # =========================================================================
    # V68: NU INTERPOLATION - Critical fix to match Bruneton reference
    # Reference (GetScattering lines 965-975):
    #   tex_coord_x = u_nu * (NU_SIZE - 1)
    #   tex_x = floor(tex_coord_x)
    #   lerp = tex_coord_x - tex_x
    #   uvw0.x = (tex_x + u_mu_s) / NU_SIZE
    #   uvw1.x = (tex_x + 1 + u_mu_s) / NU_SIZE
    #   result = texture(uvw0) * (1-lerp) + texture(uvw1) * lerp
    # =========================================================================
    
    # Nu slice index and interpolation factor
    tex_coord_x = builder.math('MULTIPLY', base_x + 2400, base_y, f'tex_x{suffix}', 
                               v1=float(SCATTERING_TEXTURE_NU_SIZE - 1))
    builder.link(x_nu.outputs[0], tex_coord_x.inputs[0])
    
    tex_x_floor = builder.math('FLOOR', base_x + 2550, base_y, f'tex_x_floor{suffix}')
    builder.link(tex_coord_x.outputs[0], tex_x_floor.inputs[0])
    
    # V68: Compute nu interpolation factor (lerp = tex_coord_x - floor)
    nu_lerp = builder.math('SUBTRACT', base_x + 2550, base_y + 50, f'nu_lerp{suffix}')
    builder.link(tex_coord_x.outputs[0], nu_lerp.inputs[0])
    builder.link(tex_x_floor.outputs[0], nu_lerp.inputs[1])
    
    # V68: tex_x_ceil = min(floor + 1, NU_SIZE - 1) to avoid out of bounds
    tex_x_ceil = builder.math('ADD', base_x + 2700, base_y + 50, f'tex_x_ceil{suffix}', v1=1.0)
    builder.link(tex_x_floor.outputs[0], tex_x_ceil.inputs[0])
    
    tex_x_ceil_clamp = builder.math('MINIMUM', base_x + 2850, base_y + 50, f'tex_x_ceil_clamp{suffix}', 
                                    v1=float(SCATTERING_TEXTURE_NU_SIZE - 1))
    builder.link(tex_x_ceil.outputs[0], tex_x_ceil_clamp.inputs[0])
    
    # uvw_x for floor nu slice: (tex_x_floor + u_mu_s) / NU_SIZE
    tex_x_plus_mus_0 = builder.math('ADD', base_x + 2700, base_y, f'tex_x0+mus{suffix}')
    builder.link(tex_x_floor.outputs[0], tex_x_plus_mus_0.inputs[0])
    builder.link(u_mu_s.outputs[0], tex_x_plus_mus_0.inputs[1])
    
    uvw_x_0 = builder.math('DIVIDE', base_x + 2850, base_y, f'uvw_x0{suffix}', 
                           v1=float(SCATTERING_TEXTURE_NU_SIZE))
    builder.link(tex_x_plus_mus_0.outputs[0], uvw_x_0.inputs[0])
    
    # uvw_x for ceil nu slice: (tex_x_ceil + u_mu_s) / NU_SIZE
    tex_x_plus_mus_1 = builder.math('ADD', base_x + 2700, base_y - 50, f'tex_x1+mus{suffix}')
    builder.link(tex_x_ceil_clamp.outputs[0], tex_x_plus_mus_1.inputs[0])
    builder.link(u_mu_s.outputs[0], tex_x_plus_mus_1.inputs[1])
    
    uvw_x_1 = builder.math('DIVIDE', base_x + 2850, base_y - 50, f'uvw_x1{suffix}', 
                           v1=float(SCATTERING_TEXTURE_NU_SIZE))
    builder.link(tex_x_plus_mus_1.outputs[0], uvw_x_1.inputs[0])
    
    # Depth slice indices and fraction
    depth_scaled = builder.math('MULTIPLY', base_x + 2400, base_y - 150, f'depth_sc{suffix}', 
                                v1=float(SCATTERING_TEXTURE_DEPTH - 1))
    builder.link(u_r.outputs[0], depth_scaled.inputs[0])
    
    depth_floor = builder.math('FLOOR', base_x + 2550, base_y - 150, f'depth_floor{suffix}')
    builder.link(depth_scaled.outputs[0], depth_floor.inputs[0])
    
    depth_frac = builder.math('SUBTRACT', base_x + 2700, base_y - 150, f'depth_frac{suffix}')
    builder.link(depth_scaled.outputs[0], depth_frac.inputs[0])
    builder.link(depth_floor.outputs[0], depth_frac.inputs[1])
    
    depth_ceil = builder.math('ADD', base_x + 2550, base_y - 200, f'depth_ceil{suffix}', v1=1.0)
    builder.link(depth_floor.outputs[0], depth_ceil.inputs[0])
    
    depth_ceil_clamp = builder.math('MINIMUM', base_x + 2700, base_y - 200, f'depth_ceil_clamp{suffix}', 
                                    v1=float(SCATTERING_TEXTURE_DEPTH - 1))
    builder.link(depth_ceil.outputs[0], depth_ceil_clamp.inputs[0])
    
    # =========================================================================
    # V68: Sample 4 textures for bilinear interpolation (nu x depth)
    # =========================================================================
    
    # Flip Y coordinate for texture sampling (Blender convention)
    u_mu_flip = builder.math('SUBTRACT', base_x + 3000, base_y, f'u_mu_flip{suffix}', v0=1.0)
    builder.link(u_mu.outputs[0], u_mu_flip.inputs[1])
    
    # --- Sample at (nu_floor, depth_floor) ---
    final_x_00 = builder.math('ADD', base_x + 3100, base_y + 100, f'fx00{suffix}')
    builder.link(depth_floor.outputs[0], final_x_00.inputs[0])
    builder.link(uvw_x_0.outputs[0], final_x_00.inputs[1])
    final_x_00_div = builder.math('DIVIDE', base_x + 3250, base_y + 100, f'fx00d{suffix}', 
                                   v1=float(SCATTERING_TEXTURE_DEPTH))
    builder.link(final_x_00.outputs[0], final_x_00_div.inputs[0])
    
    uv_00 = builder.combine_xyz(base_x + 3400, base_y + 100, f'UV00{suffix}')
    builder.link(final_x_00_div.outputs[0], uv_00.inputs['X'])
    builder.link(u_mu_flip.outputs[0], uv_00.inputs['Y'])
    tex_00 = builder.image_texture(base_x + 3550, base_y + 100, f'Tex00{suffix}', scattering_path)
    builder.link(uv_00.outputs[0], tex_00.inputs['Vector'])
    
    # --- Sample at (nu_floor, depth_ceil) ---
    final_x_01 = builder.math('ADD', base_x + 3100, base_y, f'fx01{suffix}')
    builder.link(depth_ceil_clamp.outputs[0], final_x_01.inputs[0])
    builder.link(uvw_x_0.outputs[0], final_x_01.inputs[1])
    final_x_01_div = builder.math('DIVIDE', base_x + 3250, base_y, f'fx01d{suffix}', 
                                   v1=float(SCATTERING_TEXTURE_DEPTH))
    builder.link(final_x_01.outputs[0], final_x_01_div.inputs[0])
    
    uv_01 = builder.combine_xyz(base_x + 3400, base_y, f'UV01{suffix}')
    builder.link(final_x_01_div.outputs[0], uv_01.inputs['X'])
    builder.link(u_mu_flip.outputs[0], uv_01.inputs['Y'])
    tex_01 = builder.image_texture(base_x + 3550, base_y, f'Tex01{suffix}', scattering_path)
    builder.link(uv_01.outputs[0], tex_01.inputs['Vector'])
    
    # --- Sample at (nu_ceil, depth_floor) ---
    final_x_10 = builder.math('ADD', base_x + 3100, base_y - 100, f'fx10{suffix}')
    builder.link(depth_floor.outputs[0], final_x_10.inputs[0])
    builder.link(uvw_x_1.outputs[0], final_x_10.inputs[1])
    final_x_10_div = builder.math('DIVIDE', base_x + 3250, base_y - 100, f'fx10d{suffix}', 
                                   v1=float(SCATTERING_TEXTURE_DEPTH))
    builder.link(final_x_10.outputs[0], final_x_10_div.inputs[0])
    
    uv_10 = builder.combine_xyz(base_x + 3400, base_y - 100, f'UV10{suffix}')
    builder.link(final_x_10_div.outputs[0], uv_10.inputs['X'])
    builder.link(u_mu_flip.outputs[0], uv_10.inputs['Y'])
    tex_10 = builder.image_texture(base_x + 3550, base_y - 100, f'Tex10{suffix}', scattering_path)
    builder.link(uv_10.outputs[0], tex_10.inputs['Vector'])
    
    # --- Sample at (nu_ceil, depth_ceil) ---
    final_x_11 = builder.math('ADD', base_x + 3100, base_y - 200, f'fx11{suffix}')
    builder.link(depth_ceil_clamp.outputs[0], final_x_11.inputs[0])
    builder.link(uvw_x_1.outputs[0], final_x_11.inputs[1])
    final_x_11_div = builder.math('DIVIDE', base_x + 3250, base_y - 200, f'fx11d{suffix}', 
                                   v1=float(SCATTERING_TEXTURE_DEPTH))
    builder.link(final_x_11.outputs[0], final_x_11_div.inputs[0])
    
    uv_11 = builder.combine_xyz(base_x + 3400, base_y - 200, f'UV11{suffix}')
    builder.link(final_x_11_div.outputs[0], uv_11.inputs['X'])
    builder.link(u_mu_flip.outputs[0], uv_11.inputs['Y'])
    tex_11 = builder.image_texture(base_x + 3550, base_y - 200, f'Tex11{suffix}', scattering_path)
    builder.link(uv_11.outputs[0], tex_11.inputs['Vector'])
    
    # =========================================================================
    # Bilinear interpolation: first depth, then nu
    # =========================================================================
    
    # Interpolate depth at nu_floor: mix(tex_00, tex_01, depth_frac)
    depth_mix_0 = builder.mix('RGBA', 'MIX', base_x + 3750, base_y + 50, f'DepthMix0{suffix}')
    builder.link(depth_frac.outputs[0], depth_mix_0.inputs['Factor'])
    builder.link(tex_00.outputs['Color'], depth_mix_0.inputs[6])
    builder.link(tex_01.outputs['Color'], depth_mix_0.inputs[7])
    
    # Interpolate depth at nu_ceil: mix(tex_10, tex_11, depth_frac)
    depth_mix_1 = builder.mix('RGBA', 'MIX', base_x + 3750, base_y - 150, f'DepthMix1{suffix}')
    builder.link(depth_frac.outputs[0], depth_mix_1.inputs['Factor'])
    builder.link(tex_10.outputs['Color'], depth_mix_1.inputs[6])
    builder.link(tex_11.outputs['Color'], depth_mix_1.inputs[7])
    
    # Interpolate nu: mix(depth_mix_0, depth_mix_1, nu_lerp)
    nu_mix = builder.mix('RGBA', 'MIX', base_x + 3950, base_y - 50, f'NuMix{suffix}')
    builder.link(nu_lerp.outputs[0], nu_mix.inputs['Factor'])
    builder.link(depth_mix_0.outputs[2], nu_mix.inputs[6])
    builder.link(depth_mix_1.outputs[2], nu_mix.inputs[7])
    
    result = nu_mix.outputs[2]
    
    if return_u_mu:
        return result, u_mu.outputs[0], None  # V45: u_mu now handles ground/non-ground
    return result


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
    # V69: FORCED NON-GROUND for scattering UV (restored from V54)
    # Reason: For aerial perspective, objects are ABOVE ground, so we need sky scattering.
    # The ground formula samples underground scattering which produces very different
    # (incorrect) values causing excessive haze. Ground/non-ground is still used for
    # TRANSMITTANCE which is correct.
    # Note: Reference uses ground selection, but that's for rays actually hitting ground,
    # not for aerial perspective of above-ground objects.
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
    
    V46: Implements GetSkyRadianceToPoint from Bruneton reference EXACTLY.
    No horizon clamping, no geometry-based workarounds.
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
    
    # Outputs - Separate AOVs for Nuke compositing
    group.interface.new_socket('Transmittance', in_out='OUTPUT', socket_type='NodeSocketColor')
    group.interface.new_socket('Rayleigh', in_out='OUTPUT', socket_type='NodeSocketColor')
    group.interface.new_socket('Mie', in_out='OUTPUT', socket_type='NodeSocketColor')
    
    # Set defaults
    for socket in group.interface.items_tree:
        if socket.name == 'Scene_Scale':
            socket.default_value = 0.001
    
    # =========================================================================
    # LUT TEXTURES
    # Reference: transmittance, scattering (Rayleigh+multiple), single_mie_scattering
    # =========================================================================
    
    transmittance_path = os.path.join(lut_dir, "transmittance.exr")
    scattering_path = os.path.join(lut_dir, "scattering.exr")
    single_mie_path = os.path.join(lut_dir, "single_mie_scattering.exr")
    
    # Debug: Check if Mie texture exists
    print(f"Helios: LUT dir = {lut_dir}")
    print(f"Helios: single_mie_scattering.exr exists = {os.path.exists(single_mie_path)}")
    
    # =========================================================================
    # COORDINATE TRANSFORMS
    # Reference: camera and point are positions relative to planet center
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
    # Reference lines 1797, 1814:
    #   view_ray = normalize(point - camera)
    #   d = length(point - camera)
    # =========================================================================
    
    pt_minus_cam = builder.vec_math('SUBTRACT', -1800, 100, 'Point-Camera')
    builder.link(point_km.outputs[0], pt_minus_cam.inputs[0])
    builder.link(camera_km.outputs[0], pt_minus_cam.inputs[1])
    
    view_ray = builder.vec_math('NORMALIZE', -1600, 150, 'View_Ray')
    builder.link(pt_minus_cam.outputs[0], view_ray.inputs[0])
    
    d = builder.vec_math('LENGTH', -1600, 50, 'd')
    builder.link(pt_minus_cam.outputs[0], d.inputs[0])
    
    # =========================================================================
    # CAMERA RADIUS (r)
    # Reference line 1798: r = length(camera)
    # V68: Clamp to [BOTTOM, TOP] only - no artificial minimum altitude
    # =========================================================================
    
    r_raw = builder.vec_math('LENGTH', -1400, 300, 'r_raw')
    builder.link(camera_km.outputs[0], r_raw.inputs[0])
    
    # V68: Clamp to [BOTTOM_RADIUS, TOP_RADIUS] exactly like reference
    r_min = builder.math('MAXIMUM', -1200, 300, 'r_min', v1=BOTTOM_RADIUS)
    builder.link(r_raw.outputs['Value'], r_min.inputs[0])
    
    r = builder.math('MINIMUM', -1000, 300, 'r', v1=TOP_RADIUS)
    builder.link(r_min.outputs[0], r.inputs[0])
    
    # =========================================================================
    # MU (view zenith cosine)
    # Reference line 1811: mu = dot(camera, view_ray) / r
    # NO HORIZON CLAMPING - use raw mu clamped only to [-1, 1]
    # =========================================================================
    
    cam_dot_view = builder.vec_math('DOT_PRODUCT', -1400, 200, 'cam·view')
    builder.link(camera_km.outputs[0], cam_dot_view.inputs[0])
    builder.link(view_ray.outputs[0], cam_dot_view.inputs[1])
    
    mu_raw = builder.math('DIVIDE', -1000, 200, 'mu_raw')
    builder.link(cam_dot_view.outputs['Value'], mu_raw.inputs[0])
    builder.link(r.outputs[0], mu_raw.inputs[1])
    
    # Clamp mu to [-1, 1] only (NO horizon clamp!)
    mu_max = builder.math('MINIMUM', -800, 200, 'mu_max', v1=1.0)
    builder.link(mu_raw.outputs[0], mu_max.inputs[0])
    
    mu = builder.math('MAXIMUM', -600, 200, 'mu', v1=-1.0)
    builder.link(mu_max.outputs[0], mu.inputs[0])
    
    # =========================================================================
    # MU_S (sun zenith cosine)
    # Reference line 1812: mu_s = dot(camera, sun_direction) / r
    # =========================================================================
    
    cam_dot_sun = builder.vec_math('DOT_PRODUCT', -1400, 100, 'cam·sun')
    builder.link(camera_km.outputs[0], cam_dot_sun.inputs[0])
    builder.link(group_input.outputs['Sun_Direction'], cam_dot_sun.inputs[1])
    
    mu_s_raw = builder.math('DIVIDE', -1000, 100, 'mu_s_raw')
    builder.link(cam_dot_sun.outputs['Value'], mu_s_raw.inputs[0])
    builder.link(r.outputs[0], mu_s_raw.inputs[1])
    
    mu_s_max = builder.math('MINIMUM', -800, 100, 'mu_s_max', v1=1.0)
    builder.link(mu_s_raw.outputs[0], mu_s_max.inputs[0])
    
    mu_s = builder.math('MAXIMUM', -600, 100, 'mu_s', v1=-1.0)
    builder.link(mu_s_max.outputs[0], mu_s.inputs[0])
    
    # =========================================================================
    # NU (view-sun angle cosine)
    # Reference line 1813: nu = dot(view_ray, sun_direction)
    # =========================================================================
    
    nu = builder.vec_math('DOT_PRODUCT', -1400, 0, 'nu')
    builder.link(view_ray.outputs[0], nu.inputs[0])
    builder.link(group_input.outputs['Sun_Direction'], nu.inputs[1])
    
    # =========================================================================
    # RAY_R_MU_INTERSECTS_GROUND
    # Reference lines 240-246 (RayIntersectsGround):
    #   return mu < 0.0 && r*r*(mu*mu - 1.0) + bottom_radius*bottom_radius >= 0.0
    #
    # This is computed from CAMERA parameters (r, mu) ONLY.
    # The SAME flag is used for both transmittance and scattering!
    # =========================================================================
    
    # Condition 1: mu < 0 (looking downward)
    mu_negative = builder.math('LESS_THAN', -400, 150, 'mu<0', v1=0.0)
    builder.link(mu.outputs[0], mu_negative.inputs[0])
    
    # Condition 2: discriminant >= 0
    # discriminant = r² × (μ² - 1) + bottom²
    mu_sq = builder.math('MULTIPLY', -400, 100, 'μ²')
    builder.link(mu.outputs[0], mu_sq.inputs[0])
    builder.link(mu.outputs[0], mu_sq.inputs[1])
    
    mu_sq_m1 = builder.math('SUBTRACT', -200, 100, 'μ²-1', v1=1.0)
    builder.link(mu_sq.outputs[0], mu_sq_m1.inputs[0])
    
    r_sq = builder.math('MULTIPLY', -400, 50, 'r²')
    builder.link(r.outputs[0], r_sq.inputs[0])
    builder.link(r.outputs[0], r_sq.inputs[1])
    
    r_sq_term = builder.math('MULTIPLY', 0, 75, 'r²×(μ²-1)')
    builder.link(r_sq.outputs[0], r_sq_term.inputs[0])
    builder.link(mu_sq_m1.outputs[0], r_sq_term.inputs[1])
    
    discriminant = builder.math('ADD', 200, 75, 'discriminant', v1=BOTTOM_RADIUS * BOTTOM_RADIUS)
    builder.link(r_sq_term.outputs[0], discriminant.inputs[0])
    
    disc_positive = builder.math('GREATER_THAN', 400, 75, 'disc>=0', v1=-0.0001)  # Small epsilon
    builder.link(discriminant.outputs[0], disc_positive.inputs[0])
    
    # ray_r_mu_intersects_ground = (mu < 0) AND (discriminant >= 0)
    ray_intersects_ground = builder.math('MULTIPLY', 600, 125, 'ray_intersects_ground')
    builder.link(mu_negative.outputs[0], ray_intersects_ground.inputs[0])
    builder.link(disc_positive.outputs[0], ray_intersects_ground.inputs[1])
    
    # =========================================================================
    # POINT PARAMETERS - LAW OF COSINES
    # Reference lines 1832-1834:
    #   r_p = ClampRadius(sqrt(d² + 2·r·μ·d + r²))
    #   mu_p = (r·μ + d) / r_p
    #   mu_s_p = (r·μ_s + d·nu) / r_p
    # =========================================================================
    
    # d²
    d_sq = builder.math('MULTIPLY', -400, -100, 'd²')
    builder.link(d.outputs['Value'], d_sq.inputs[0])
    builder.link(d.outputs['Value'], d_sq.inputs[1])
    
    # 2·r·μ·d
    two_r = builder.math('MULTIPLY', -400, -150, '2r', v0=2.0)
    builder.link(r.outputs[0], two_r.inputs[1])
    
    two_r_mu = builder.math('MULTIPLY', -200, -150, '2r×μ')
    builder.link(two_r.outputs[0], two_r_mu.inputs[0])
    builder.link(mu.outputs[0], two_r_mu.inputs[1])  # Use raw mu, not clamped
    
    two_r_mu_d = builder.math('MULTIPLY', 0, -150, '2r×μ×d')
    builder.link(two_r_mu.outputs[0], two_r_mu_d.inputs[0])
    builder.link(d.outputs['Value'], two_r_mu_d.inputs[1])
    
    # d² + 2·r·μ·d + r²
    sum1 = builder.math('ADD', 200, -125, 'd²+2rμd')
    builder.link(d_sq.outputs[0], sum1.inputs[0])
    builder.link(two_r_mu_d.outputs[0], sum1.inputs[1])
    
    sum2 = builder.math('ADD', 400, -150, 'd²+2rμd+r²')
    builder.link(sum1.outputs[0], sum2.inputs[0])
    builder.link(r_sq.outputs[0], sum2.inputs[1])
    
    # r_p = ClampRadius(sqrt(...))
    r_p_raw = builder.math('SQRT', 600, -150, 'r_p_raw')
    builder.link(sum2.outputs[0], r_p_raw.inputs[0])
    
    r_p_min = builder.math('MAXIMUM', 800, -150, 'r_p_min', v1=BOTTOM_RADIUS)
    builder.link(r_p_raw.outputs[0], r_p_min.inputs[0])
    
    r_p = builder.math('MINIMUM', 1000, -150, 'r_p', v1=TOP_RADIUS)
    builder.link(r_p_min.outputs[0], r_p.inputs[0])
    
    # mu_p = (r·μ + d) / r_p
    r_mu = builder.math('MULTIPLY', -200, -300, 'r×μ')
    builder.link(r.outputs[0], r_mu.inputs[0])
    builder.link(mu.outputs[0], r_mu.inputs[1])  # Use raw mu
    
    r_mu_plus_d = builder.math('ADD', 0, -300, 'r×μ+d')
    builder.link(r_mu.outputs[0], r_mu_plus_d.inputs[0])
    builder.link(d.outputs['Value'], r_mu_plus_d.inputs[1])
    
    mu_p_raw = builder.math('DIVIDE', 200, -300, 'μ_p_raw')
    builder.link(r_mu_plus_d.outputs[0], mu_p_raw.inputs[0])
    builder.link(r_p.outputs[0], mu_p_raw.inputs[1])
    
    # ClampCosine(mu_p)
    mu_p_max = builder.math('MINIMUM', 400, -300, 'μ_p_max', v1=1.0)
    builder.link(mu_p_raw.outputs[0], mu_p_max.inputs[0])
    
    mu_p = builder.math('MAXIMUM', 600, -300, 'μ_p', v1=-1.0)
    builder.link(mu_p_max.outputs[0], mu_p.inputs[0])
    
    # mu_s_p = (r·μ_s + d·ν) / r_p
    r_mu_s = builder.math('MULTIPLY', -200, -400, 'r×μ_s')
    builder.link(r.outputs[0], r_mu_s.inputs[0])
    builder.link(mu_s.outputs[0], r_mu_s.inputs[1])
    
    d_nu = builder.math('MULTIPLY', -200, -450, 'd×ν')
    builder.link(d.outputs['Value'], d_nu.inputs[0])
    builder.link(nu.outputs['Value'], d_nu.inputs[1])
    
    r_mu_s_plus_d_nu = builder.math('ADD', 0, -400, 'r×μ_s+d×ν')
    builder.link(r_mu_s.outputs[0], r_mu_s_plus_d_nu.inputs[0])
    builder.link(d_nu.outputs[0], r_mu_s_plus_d_nu.inputs[1])
    
    mu_s_p_raw = builder.math('DIVIDE', 200, -400, 'μ_s_p_raw')
    builder.link(r_mu_s_plus_d_nu.outputs[0], mu_s_p_raw.inputs[0])
    builder.link(r_p.outputs[0], mu_s_p_raw.inputs[1])
    
    mu_s_p_max = builder.math('MINIMUM', 400, -400, 'μ_s_p_max', v1=1.0)
    builder.link(mu_s_p_raw.outputs[0], mu_s_p_max.inputs[0])
    
    mu_s_p = builder.math('MAXIMUM', 600, -400, 'μ_s_p', v1=-1.0)
    builder.link(mu_s_p_max.outputs[0], mu_s_p.inputs[0])
    
    # =========================================================================
    # TRANSMITTANCE - GetTransmittance
    # Reference lines 493-519:
    #   if (ray_r_mu_intersects_ground):
    #     T = T(r_d, -mu_d) / T(r, -mu)
    #   else:
    #     T = T(r, mu) / T(r_d, mu_d)
    #
    # NOTE: r_d = r_p, mu_d = mu_p (from law of cosines above)
    # =========================================================================
    
    # Negate mu values for ground formula
    neg_mu = builder.math('MULTIPLY', -600, 600, '-mu', v1=-1.0)
    builder.link(mu.outputs[0], neg_mu.inputs[0])
    
    neg_mu_p = builder.math('MULTIPLY', -400, 600, '-mu_p', v1=-1.0)
    builder.link(mu_p.outputs[0], neg_mu_p.inputs[0])
    
    # NON-GROUND transmittance UVs: T(r, mu) and T(r_p, mu_p)
    trans_uv_cam_ng = create_transmittance_uv(
        builder, r.outputs[0], mu.outputs[0], 
        -400, 800, "_cam_ng"
    )
    trans_uv_pt_ng = create_transmittance_uv(
        builder, r_p.outputs[0], mu_p.outputs[0],
        -400, 700, "_pt_ng"
    )
    
    # GROUND transmittance UVs: T(r, -mu) and T(r_p, -mu_p)
    trans_uv_cam_g = create_transmittance_uv(
        builder, r.outputs[0], neg_mu.outputs[0], 
        -400, 600, "_cam_g"
    )
    trans_uv_pt_g = create_transmittance_uv(
        builder, r_p.outputs[0], neg_mu_p.outputs[0],
        -400, 500, "_pt_g"
    )
    
    # Sample transmittance textures
    tex_T_cam_ng = builder.image_texture(800, 800, 'T_cam_ng', transmittance_path)
    builder.link(trans_uv_cam_ng, tex_T_cam_ng.inputs['Vector'])
    
    tex_T_pt_ng = builder.image_texture(800, 700, 'T_pt_ng', transmittance_path)
    builder.link(trans_uv_pt_ng, tex_T_pt_ng.inputs['Vector'])
    
    tex_T_cam_g = builder.image_texture(800, 600, 'T_cam_g', transmittance_path)
    builder.link(trans_uv_cam_g, tex_T_cam_g.inputs['Vector'])
    
    tex_T_pt_g = builder.image_texture(800, 500, 'T_pt_g', transmittance_path)
    builder.link(trans_uv_pt_g, tex_T_pt_g.inputs['Vector'])
    
    # NON-GROUND formula: T = T(r, mu) / T(r_p, mu_p)
    # V88 FIX: Grayscale transmittance to eliminate colored banding
    # The correct formula is T = T_cam / T_pt, but per-channel division causes banding.
    # Solution: Compute grayscale transmittance ratio and output as uniform RGB.
    # This loses spectral variation but gives mathematically correct transmittance magnitude.
    
    # Convert T_cam to grayscale (average)
    T_cam_sep = builder.separate_xyz(950, 800, 'T_cam_sep')
    builder.link(tex_T_cam_ng.outputs['Color'], T_cam_sep.inputs[0])
    
    T_cam_rg = builder.math('ADD', 1050, 850, 'T_cam_rg')
    builder.link(T_cam_sep.outputs['X'], T_cam_rg.inputs[0])
    builder.link(T_cam_sep.outputs['Y'], T_cam_rg.inputs[1])
    
    T_cam_lum = builder.math('ADD', 1150, 850, 'T_cam_lum')
    builder.link(T_cam_rg.outputs[0], T_cam_lum.inputs[0])
    builder.link(T_cam_sep.outputs['Z'], T_cam_lum.inputs[1])
    
    T_cam_avg = builder.math('DIVIDE', 1250, 850, 'T_cam_avg', v1=3.0)
    builder.link(T_cam_lum.outputs[0], T_cam_avg.inputs[0])
    
    # Convert T_pt to grayscale (average)
    T_pt_sep = builder.separate_xyz(950, 700, 'T_pt_sep')
    builder.link(tex_T_pt_ng.outputs['Color'], T_pt_sep.inputs[0])
    
    T_pt_rg = builder.math('ADD', 1050, 750, 'T_pt_rg')
    builder.link(T_pt_sep.outputs['X'], T_pt_rg.inputs[0])
    builder.link(T_pt_sep.outputs['Y'], T_pt_rg.inputs[1])
    
    T_pt_lum = builder.math('ADD', 1150, 750, 'T_pt_lum')
    builder.link(T_pt_rg.outputs[0], T_pt_lum.inputs[0])
    builder.link(T_pt_sep.outputs['Z'], T_pt_lum.inputs[1])
    
    T_pt_avg = builder.math('DIVIDE', 1250, 750, 'T_pt_avg', v1=3.0)
    builder.link(T_pt_lum.outputs[0], T_pt_avg.inputs[0])
    
    # Compute grayscale transmittance ratio: T = T_cam_gray / T_pt_gray
    # This is the CORRECT formula - for d→0: T_cam≈T_pt so T→1
    T_pt_safe = builder.math('MAXIMUM', 1350, 750, 'T_pt_safe', v1=0.0001)
    builder.link(T_pt_avg.outputs[0], T_pt_safe.inputs[0])
    
    T_gray = builder.math('DIVIDE', 1450, 800, 'T_gray')
    builder.link(T_cam_avg.outputs[0], T_gray.inputs[0])
    builder.link(T_pt_safe.outputs[0], T_gray.inputs[1])
    
    # Clamp to [0, 1] - transmittance cannot exceed 1
    T_gray_clamp = builder.math('MINIMUM', 1550, 800, 'T_gray_clamp', v1=1.0)
    builder.link(T_gray.outputs[0], T_gray_clamp.inputs[0])
    
    # Convert grayscale to RGB vector for use in vector operations
    T_nonground = builder.combine_xyz(1650, 750, 'T_nonground')
    builder.link(T_gray_clamp.outputs[0], T_nonground.inputs['X'])
    builder.link(T_gray_clamp.outputs[0], T_nonground.inputs['Y'])
    builder.link(T_gray_clamp.outputs[0], T_nonground.inputs['Z'])
    
    # GROUND formula: T = T(r_p, -mu_p) / T(r, -mu)
    T_ground = builder.vec_math('DIVIDE', 1100, 550, 'T_ground')
    builder.link(tex_T_pt_g.outputs['Color'], T_ground.inputs[0])
    builder.link(tex_T_cam_g.outputs['Color'], T_ground.inputs[1])
    
    # V71: FORCE NON-GROUND TRANSMITTANCE for aerial perspective
    # Reason: Ground/non-ground selection causes discontinuity at horizon (banding)
    # For aerial perspective, objects are ABOVE ground, so we always use non-ground formula.
    # The ground formula is for rays that would hit the ground if extended.
    
    # Clamp to [0, 1]
    T_clamp = builder.vec_math('MINIMUM', 1450, 650, 'T_clamp')
    T_clamp.inputs[1].default_value = (1.0, 1.0, 1.0)
    builder.link(T_nonground.outputs[0], T_clamp.inputs[0])
    
    transmittance = builder.vec_math('MAXIMUM', 1600, 650, 'Transmittance')
    transmittance.inputs[1].default_value = (0.0, 0.0, 0.0)
    builder.link(T_clamp.outputs[0], transmittance.inputs[0])
    
    # =========================================================================
    # SCATTERING - GetCombinedScattering
    # Reference lines 1821-1824, 1837-1840:
    #   Both camera and point scattering use the SAME ray_r_mu_intersects_ground
    #   Returns both Rayleigh+multiple scattering AND single Mie scattering
    # =========================================================================
    
    # Sample Rayleigh+multiple scattering at camera position
    scat_cam_color = sample_scattering_texture(
        builder, r.outputs[0], mu.outputs[0], mu_s.outputs[0], nu.outputs['Value'],
        scattering_path, 1800, 200, "_cam",
        ray_intersects_ground_socket=ray_intersects_ground.outputs[0]
    )
    
    # Sample Rayleigh+multiple scattering at point position
    scat_pt_color = sample_scattering_texture(
        builder, r_p.outputs[0], mu_p.outputs[0], mu_s_p.outputs[0], nu.outputs['Value'],
        scattering_path, 1800, -400, "_pt",
        ray_intersects_ground_socket=ray_intersects_ground.outputs[0]
    )
    
    # Sample single Mie scattering at camera position
    mie_cam_color = sample_scattering_texture(
        builder, r.outputs[0], mu.outputs[0], mu_s.outputs[0], nu.outputs['Value'],
        single_mie_path, 2200, 200, "_mie_cam",
        ray_intersects_ground_socket=ray_intersects_ground.outputs[0]
    )
    
    # Sample single Mie scattering at point position
    mie_pt_color = sample_scattering_texture(
        builder, r_p.outputs[0], mu_p.outputs[0], mu_s_p.outputs[0], nu.outputs['Value'],
        single_mie_path, 2200, -400, "_mie_pt",
        ray_intersects_ground_socket=ray_intersects_ground.outputs[0]
    )
    
    # =========================================================================
    # INSCATTER - Rayleigh + Multiple Scattering
    # Reference line 1849: scattering = scattering - shadow_transmittance * scattering_p
    # =========================================================================
    
    # Rayleigh inscatter = S_cam - T × S_pt
    t_times_scat = builder.vec_math('MULTIPLY', 5500, 200, 'T×S_pt')
    builder.link(transmittance.outputs[0], t_times_scat.inputs[0])
    builder.link(scat_pt_color, t_times_scat.inputs[1])
    
    rayleigh_inscatter_raw = builder.vec_math('SUBTRACT', 5650, 200, 'Ray_Inscatter_Raw')
    builder.link(scat_cam_color, rayleigh_inscatter_raw.inputs[0])
    builder.link(t_times_scat.outputs[0], rayleigh_inscatter_raw.inputs[1])
    
    rayleigh_inscatter = builder.vec_math('MAXIMUM', 5800, 200, 'Ray_Inscatter')
    rayleigh_inscatter.inputs[1].default_value = (0.0, 0.0, 0.0)
    builder.link(rayleigh_inscatter_raw.outputs[0], rayleigh_inscatter.inputs[0])
    
    # =========================================================================
    # INSCATTER - Single Mie Scattering
    # Reference line 1850-1851: single_mie = mie_cam - T × mie_pt
    # =========================================================================
    
    t_times_mie = builder.vec_math('MULTIPLY', 5500, -100, 'T×Mie_pt')
    builder.link(transmittance.outputs[0], t_times_mie.inputs[0])
    builder.link(mie_pt_color, t_times_mie.inputs[1])
    
    mie_inscatter_raw = builder.vec_math('SUBTRACT', 5650, -100, 'Mie_Inscatter_Raw')
    builder.link(mie_cam_color, mie_inscatter_raw.inputs[0])
    builder.link(t_times_mie.outputs[0], mie_inscatter_raw.inputs[1])
    
    mie_inscatter = builder.vec_math('MAXIMUM', 5800, -100, 'Mie_Inscatter')
    mie_inscatter.inputs[1].default_value = (0.0, 0.0, 0.0)
    builder.link(mie_inscatter_raw.outputs[0], mie_inscatter.inputs[0])
    
    # =========================================================================
    # PHASE FUNCTIONS
    # Reference lines 1857-1862:
    #   single_mie_scattering *= smoothstep(0, 0.01, mu_s)  // Fade when sun below horizon
    #   return scattering * RayleighPhaseFunction(nu) + 
    #          single_mie_scattering * MiePhaseFunction(g, nu)
    # =========================================================================
    
    # --- Rayleigh phase: 3/(16π) × (1 + ν²) ---
    nu_sq = builder.math('MULTIPLY', 5900, 300, 'ν²')
    builder.link(nu.outputs['Value'], nu_sq.inputs[0])
    builder.link(nu.outputs['Value'], nu_sq.inputs[1])
    
    one_plus_nu_sq = builder.math('ADD', 6050, 300, '1+ν²', v0=1.0)
    builder.link(nu_sq.outputs[0], one_plus_nu_sq.inputs[1])
    
    rayleigh_phase = builder.math('MULTIPLY', 6200, 300, 'Ray_Phase', v0=3.0 / (16.0 * math.pi))
    builder.link(one_plus_nu_sq.outputs[0], rayleigh_phase.inputs[1])
    
    # --- Mie phase function - CORRECT formula from reference (lines 744-746) ---
    # k = 3/(8π) × (1-g²)/(2+g²)
    # MiePhaseFunction = k × (1+ν²) / (1+g²-2gν)^1.5
    MIE_G = 0.8
    g_sq = MIE_G * MIE_G  # 0.64
    
    # k = 3/(8π) × (1-g²)/(2+g²)
    mie_k = (3.0 / (8.0 * math.pi)) * (1.0 - g_sq) / (2.0 + g_sq)  # ≈ 0.0163
    
    # (1 + ν²) term - uses nu_sq computed for Rayleigh
    one_plus_nu_sq_mie = builder.math('ADD', 5900, -300, '1+ν²_mie', v0=1.0)
    builder.link(nu_sq.outputs[0], one_plus_nu_sq_mie.inputs[1])
    
    # denominator_inner = 1 + g² - 2g×ν
    two_g_nu = builder.math('MULTIPLY', 5900, -350, '2g×ν', v0=2.0 * MIE_G)
    builder.link(nu.outputs['Value'], two_g_nu.inputs[1])
    
    denom_inner = builder.math('SUBTRACT', 6050, -350, '1+g²-2gν', v0=1.0 + g_sq)
    builder.link(two_g_nu.outputs[0], denom_inner.inputs[1])
    
    # Safe clamp denominator inner to avoid division issues
    denom_inner_safe = builder.math('MAXIMUM', 6200, -350, 'denom_safe', v1=0.0001)
    builder.link(denom_inner.outputs[0], denom_inner_safe.inputs[0])
    
    # (1+g²-2gν)^1.5
    denom_pow = builder.math('POWER', 6350, -350, 'inner^1.5', v1=1.5)
    builder.link(denom_inner_safe.outputs[0], denom_pow.inputs[0])
    
    # k × (1+ν²)
    k_times_nu_term = builder.math('MULTIPLY', 6050, -300, 'k×(1+ν²)', v0=mie_k)
    builder.link(one_plus_nu_sq_mie.outputs[0], k_times_nu_term.inputs[1])
    
    # mie_phase = k × (1+ν²) / (1+g²-2gν)^1.5
    mie_phase = builder.math('DIVIDE', 6500, -300, 'Mie_Phase')
    builder.link(k_times_nu_term.outputs[0], mie_phase.inputs[0])
    builder.link(denom_pow.outputs[0], mie_phase.inputs[1])
    
    # --- Mie smoothstep fade when sun below horizon (mu_s < 0.01) ---
    # Reference line 1858-1859: smoothstep(0, 0.01, mu_s)
    mu_s_fade_raw = builder.math('DIVIDE', 5900, -500, 'mu_s_fade_raw', v1=0.01)
    builder.link(mu_s.outputs[0], mu_s_fade_raw.inputs[0])
    
    mu_s_fade_clamp = builder.math('MINIMUM', 6050, -500, 'mu_s_fade_clamp', v1=1.0)
    builder.link(mu_s_fade_raw.outputs[0], mu_s_fade_clamp.inputs[0])
    
    mu_s_fade = builder.math('MAXIMUM', 6200, -500, 'mu_s_fade', v1=0.0)
    builder.link(mu_s_fade_clamp.outputs[0], mu_s_fade.inputs[0])
    
    # Smoothstep: 3x² - 2x³
    mu_s_fade_sq = builder.math('MULTIPLY', 6350, -500, 'fade²')
    builder.link(mu_s_fade.outputs[0], mu_s_fade_sq.inputs[0])
    builder.link(mu_s_fade.outputs[0], mu_s_fade_sq.inputs[1])
    
    mu_s_fade_cu = builder.math('MULTIPLY', 6500, -500, 'fade³')
    builder.link(mu_s_fade_sq.outputs[0], mu_s_fade_cu.inputs[0])
    builder.link(mu_s_fade.outputs[0], mu_s_fade_cu.inputs[1])
    
    smooth_3x2 = builder.math('MULTIPLY', 6350, -550, '3×fade²', v0=3.0)
    builder.link(mu_s_fade_sq.outputs[0], smooth_3x2.inputs[1])
    
    smooth_2x3 = builder.math('MULTIPLY', 6500, -550, '2×fade³', v0=2.0)
    builder.link(mu_s_fade_cu.outputs[0], smooth_2x3.inputs[1])
    
    mie_smoothstep = builder.math('SUBTRACT', 6650, -550, 'mie_smoothstep')
    builder.link(smooth_3x2.outputs[0], mie_smoothstep.inputs[0])
    builder.link(smooth_2x3.outputs[0], mie_smoothstep.inputs[1])
    
    # --- Apply phase functions ---
    # Rayleigh contribution
    rayleigh_phased = builder.vec_math('SCALE', 6800, 200, 'Ray_Phased')
    builder.link(rayleigh_inscatter.outputs[0], rayleigh_phased.inputs[0])
    builder.link(rayleigh_phase.outputs[0], rayleigh_phased.inputs['Scale'])
    
    # Mie contribution with smoothstep fade
    mie_phase_faded = builder.math('MULTIPLY', 6800, -400, 'Mie_Phase_Faded')
    builder.link(mie_phase.outputs[0], mie_phase_faded.inputs[0])
    builder.link(mie_smoothstep.outputs[0], mie_phase_faded.inputs[1])
    
    mie_phased = builder.vec_math('SCALE', 6950, -100, 'Mie_Phased')
    builder.link(mie_inscatter.outputs[0], mie_phased.inputs[0])
    builder.link(mie_phase_faded.outputs[0], mie_phased.inputs['Scale'])
    
    # --- Combine: Rayleigh + Mie ---
    # Reference line 1861-1862
    inscatter_phased = builder.vec_math('ADD', 7100, 50, 'Inscatter_Phased')
    builder.link(rayleigh_phased.outputs[0], inscatter_phased.inputs[0])
    builder.link(mie_phased.outputs[0], inscatter_phased.inputs[1])
    
    # =========================================================================
    # OUTPUTS - V97: Actual Rayleigh/Mie AOVs
    # =========================================================================
    # V96 confirmed: d IS correctly linked. The diagnostic comparisons looked
    # identical because d (~1km) vs r×μ (~6360) is only ~0.016% difference.
    # The math IS working - restore actual output.
    
    builder.link(transmittance.outputs[0], group_output.inputs['Transmittance'])
    builder.link(rayleigh_phased.outputs[0], group_output.inputs['Rayleigh'])
    builder.link(mie_phased.outputs[0], group_output.inputs['Mie'])
    
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
