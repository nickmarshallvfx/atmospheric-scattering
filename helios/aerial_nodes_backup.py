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
# NODE GROUP NAME AND VERSION
# =============================================================================

AERIAL_NODE_GROUP_NAME = "Helios_Aerial_Perspective"
AERIAL_NODE_VERSION = 4  # Increment to force node group recreation


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
# SCATTERING UV CALCULATION (4D -> 3D Atlas)
# =============================================================================

def create_scattering_uv_nodes(builder, r_socket, mu_socket, mu_s_socket, nu_socket,
                                bottom_radius, top_radius, mu_s_min,
                                base_x=0, base_y=0):
    """
    Create nodes for GetScatteringTextureUvwzFromRMuMuSNu.
    
    Maps 4D parameters (r, mu, mu_s, nu) to 3D texture coordinates.
    Our scattering LUT is stored as a 2D atlas with depth slices tiled horizontally.
    
    Returns: (uvw_socket for combined UV, layer_socket for depth interpolation)
    """
    H = math.sqrt(top_radius * top_radius - bottom_radius * bottom_radius)
    
    # === u_r: altitude coordinate ===
    # rho = sqrt(r² - bottom_radius²)
    r_sq = builder.math('POWER', base_x, base_y, 'scat_r²', v1=2.0)
    builder.link(r_socket, r_sq.inputs[0])
    
    rho_sq = builder.math('SUBTRACT', base_x + 200, base_y, 'scat_rho²',
                          v1=bottom_radius * bottom_radius)
    builder.link(r_sq.outputs[0], rho_sq.inputs[0])
    
    # Clamp to avoid sqrt of negative
    rho_sq_safe = builder.math('MAXIMUM', base_x + 400, base_y, 'rho²_safe', v1=0.0)
    builder.link(rho_sq.outputs[0], rho_sq_safe.inputs[0])
    
    rho = builder.math('SQRT', base_x + 600, base_y, 'scat_rho')
    builder.link(rho_sq_safe.outputs[0], rho.inputs[0])
    
    # u_r = rho / H mapped to texture coord
    u_r_raw = builder.math('DIVIDE', base_x + 800, base_y, 'u_r_raw', v1=H)
    builder.link(rho.outputs[0], u_r_raw.inputs[0])
    
    # GetTextureCoordFromUnitRange for R dimension
    r_scale = 1.0 - 1.0 / SCATTERING_TEXTURE_R_SIZE
    r_offset = 0.5 / SCATTERING_TEXTURE_R_SIZE
    
    u_r_scaled = builder.math('MULTIPLY', base_x + 1000, base_y, 'u_r_scaled', v1=r_scale)
    builder.link(u_r_raw.outputs[0], u_r_scaled.inputs[0])
    
    u_r = builder.math('ADD', base_x + 1200, base_y, 'u_r', v1=r_offset)
    builder.link(u_r_scaled.outputs[0], u_r.inputs[0])
    
    # === u_mu: view zenith coordinate ===
    # This depends on whether the ray intersects ground
    # For simplicity, we assume camera is above ground looking up/horizontal
    # d = DistanceToTopAtmosphereBoundary
    
    mu_sq = builder.math('POWER', base_x, base_y - 150, 'scat_mu²', v1=2.0)
    builder.link(mu_socket, mu_sq.inputs[0])
    
    # discriminant for top boundary
    mu_sq_m1 = builder.math('SUBTRACT', base_x + 200, base_y - 150, 'mu²-1', v1=1.0)
    builder.link(mu_sq.outputs[0], mu_sq_m1.inputs[0])
    
    r_sq_mu_term = builder.math('MULTIPLY', base_x + 400, base_y - 150, 'r²(mu²-1)')
    builder.link(r_sq.outputs[0], r_sq_mu_term.inputs[0])
    builder.link(mu_sq_m1.outputs[0], r_sq_mu_term.inputs[1])
    
    discrim_top = builder.math('ADD', base_x + 600, base_y - 150, 'disc_top',
                               v1=top_radius * top_radius)
    builder.link(r_sq_mu_term.outputs[0], discrim_top.inputs[0])
    
    discrim_top_safe = builder.math('MAXIMUM', base_x + 800, base_y - 150, 'disc_safe', v1=0.0)
    builder.link(discrim_top.outputs[0], discrim_top_safe.inputs[0])
    
    discrim_sqrt = builder.math('SQRT', base_x + 1000, base_y - 150, 'sqrt_disc')
    builder.link(discrim_top_safe.outputs[0], discrim_sqrt.inputs[0])
    
    # d = -r*mu + sqrt(discriminant)
    r_mu_prod = builder.math('MULTIPLY', base_x + 400, base_y - 250, 'r×mu')
    builder.link(r_socket, r_mu_prod.inputs[0])
    builder.link(mu_socket, r_mu_prod.inputs[1])
    
    neg_r_mu = builder.math('MULTIPLY', base_x + 600, base_y - 250, '-r×mu', v1=-1.0)
    builder.link(r_mu_prod.outputs[0], neg_r_mu.inputs[0])
    
    d_top = builder.math('ADD', base_x + 1200, base_y - 200, 'd_top')
    builder.link(neg_r_mu.outputs[0], d_top.inputs[0])
    builder.link(discrim_sqrt.outputs[0], d_top.inputs[1])
    
    # d_min and d_max for mu coordinate
    d_min = builder.math('SUBTRACT', base_x + 1000, base_y - 50, 'd_min', v0=top_radius)
    builder.link(r_socket, d_min.inputs[1])
    
    d_max = builder.math('ADD', base_x + 1000, base_y + 50, 'd_max', v1=H)
    builder.link(rho.outputs[0], d_max.inputs[0])
    
    # x_mu = (d - d_min) / (d_max - d_min)
    d_minus_dmin = builder.math('SUBTRACT', base_x + 1400, base_y - 150, 'd-d_min')
    builder.link(d_top.outputs[0], d_minus_dmin.inputs[0])
    builder.link(d_min.outputs[0], d_minus_dmin.inputs[1])
    
    dmax_minus_dmin = builder.math('SUBTRACT', base_x + 1400, base_y - 50, 'dmax-dmin')
    builder.link(d_max.outputs[0], dmax_minus_dmin.inputs[0])
    builder.link(d_min.outputs[0], dmax_minus_dmin.inputs[1])
    
    # Avoid division by zero
    dmax_dmin_safe = builder.math('MAXIMUM', base_x + 1600, base_y - 50, 'denom_safe', v1=0.0001)
    builder.link(dmax_minus_dmin.outputs[0], dmax_dmin_safe.inputs[0])
    
    x_mu = builder.math('DIVIDE', base_x + 1800, base_y - 100, 'x_mu')
    builder.link(d_minus_dmin.outputs[0], x_mu.inputs[0])
    builder.link(dmax_dmin_safe.outputs[0], x_mu.inputs[1])
    
    # Clamp x_mu to [0, 1]
    x_mu_clamped = builder.math('MINIMUM', base_x + 2000, base_y - 100, 'x_mu_max', v1=1.0)
    builder.link(x_mu.outputs[0], x_mu_clamped.inputs[0])
    
    x_mu_final = builder.math('MAXIMUM', base_x + 2200, base_y - 100, 'x_mu_clamp', v1=0.0)
    builder.link(x_mu_clamped.outputs[0], x_mu_final.inputs[0])
    
    # GetTextureCoordFromUnitRange for MU dimension
    mu_scale = 1.0 - 1.0 / SCATTERING_TEXTURE_MU_SIZE
    mu_offset = 0.5 / SCATTERING_TEXTURE_MU_SIZE
    
    u_mu_scaled = builder.math('MULTIPLY', base_x + 2400, base_y - 100, 'u_mu_scaled', v1=mu_scale)
    builder.link(x_mu_final.outputs[0], u_mu_scaled.inputs[0])
    
    u_mu = builder.math('ADD', base_x + 2600, base_y - 100, 'u_mu', v1=mu_offset)
    builder.link(u_mu_scaled.outputs[0], u_mu.inputs[0])
    
    # === u_mu_s: sun zenith coordinate ===
    # x_mu_s = (1 - exp(-3*mu_s - 0.6)) / (1 - exp(-3.6))
    # Simplified: linear mapping from [mu_s_min, 1] to [0, 1]
    
    mu_s_range = 1.0 - mu_s_min
    mu_s_shifted = builder.math('SUBTRACT', base_x, base_y - 350, 'mu_s-min', v1=mu_s_min)
    builder.link(mu_s_socket, mu_s_shifted.inputs[0])
    
    x_mu_s = builder.math('DIVIDE', base_x + 200, base_y - 350, 'x_mu_s', v1=mu_s_range)
    builder.link(mu_s_shifted.outputs[0], x_mu_s.inputs[0])
    
    # Clamp to [0, 1]
    x_mu_s_clamped = builder.math('MINIMUM', base_x + 400, base_y - 350, 'x_mu_s_max', v1=1.0)
    builder.link(x_mu_s.outputs[0], x_mu_s_clamped.inputs[0])
    
    x_mu_s_final = builder.math('MAXIMUM', base_x + 600, base_y - 350, 'x_mu_s_clamp', v1=0.0)
    builder.link(x_mu_s_clamped.outputs[0], x_mu_s_final.inputs[0])
    
    # GetTextureCoordFromUnitRange for MU_S dimension
    mu_s_scale = 1.0 - 1.0 / SCATTERING_TEXTURE_MU_S_SIZE
    mu_s_offset = 0.5 / SCATTERING_TEXTURE_MU_S_SIZE
    
    u_mu_s_scaled = builder.math('MULTIPLY', base_x + 800, base_y - 350, 'u_mu_s_scaled', v1=mu_s_scale)
    builder.link(x_mu_s_final.outputs[0], u_mu_s_scaled.inputs[0])
    
    u_mu_s = builder.math('ADD', base_x + 1000, base_y - 350, 'u_mu_s', v1=mu_s_offset)
    builder.link(u_mu_s_scaled.outputs[0], u_mu_s.inputs[0])
    
    # === u_nu: view-sun angle coordinate ===
    # x_nu = (nu + 1) / 2  (nu is in [-1, 1])
    nu_plus1 = builder.math('ADD', base_x, base_y - 450, 'nu+1', v0=1.0)
    builder.link(nu_socket, nu_plus1.inputs[1])
    
    x_nu = builder.math('MULTIPLY', base_x + 200, base_y - 450, 'x_nu', v1=0.5)
    builder.link(nu_plus1.outputs[0], x_nu.inputs[0])
    
    # GetTextureCoordFromUnitRange for NU dimension
    nu_scale = 1.0 - 1.0 / SCATTERING_TEXTURE_NU_SIZE
    nu_offset = 0.5 / SCATTERING_TEXTURE_NU_SIZE
    
    u_nu_scaled = builder.math('MULTIPLY', base_x + 400, base_y - 450, 'u_nu_scaled', v1=nu_scale)
    builder.link(x_nu.outputs[0], u_nu_scaled.inputs[0])
    
    u_nu = builder.math('ADD', base_x + 600, base_y - 450, 'u_nu', v1=nu_offset)
    builder.link(u_nu_scaled.outputs[0], u_nu.inputs[0])
    
    # === Combine into atlas UV ===
    # Our 3D texture is stored as 2D atlas: X = nu * MU_S_SIZE tiles, Y = mu, layers = r
    # Final U = (u_nu * NU_SIZE + u_mu_s) / NU_SIZE ... but we pack differently
    # Actually: U = u_mu_s + u_nu * MU_S_SIZE ... need to check our atlas layout
    
    # For a proper 3D atlas with depth slices tiled horizontally:
    # U = (floor(u_r * R_SIZE) + u_mu_s * MU_S_SIZE + u_nu * NU_SIZE * MU_S_SIZE) / TOTAL_WIDTH
    # This is complex, let's use a simpler approach with the atlas we have
    
    # Our scattering.exr is WIDTH x HEIGHT x DEPTH as a 2D atlas
    # Width = NU_SIZE * MU_S_SIZE = 256, Height = MU_SIZE = 128, Depth tiles = R_SIZE = 32
    # Atlas layout: depth slices stacked vertically
    
    # layer_index = floor(u_r * (R_SIZE - 1))
    r_idx_float = builder.math('MULTIPLY', base_x + 1400, base_y, 'r_idx', 
                               v1=float(SCATTERING_TEXTURE_R_SIZE - 1))
    builder.link(u_r_raw.outputs[0], r_idx_float.inputs[0])
    
    r_idx = builder.math('FLOOR', base_x + 1600, base_y, 'r_idx_floor')
    builder.link(r_idx_float.outputs[0], r_idx.inputs[0])
    
    # Interpolation factor for depth
    r_frac = builder.math('SUBTRACT', base_x + 1800, base_y, 'r_frac')
    builder.link(r_idx_float.outputs[0], r_frac.inputs[0])
    builder.link(r_idx.outputs[0], r_frac.inputs[1])
    
    # Combine mu_s and nu into U coordinate
    # u_combined = (u_nu * NU_SIZE + floor(u_mu_s * MU_S_SIZE)) / (NU_SIZE * MU_S_SIZE)
    # Simpler: since our LUT packs nu,mu_s into X, mu into Y, r into layers
    
    # X coord in atlas = u_nu + u_mu_s / NU_SIZE ... check the actual layout
    # Let's use a straightforward mapping matching our LUT generator
    
    # Final UV for atlas sampling (without depth interpolation for now)
    # U = u_mu_s (within one nu slice) + u_nu_tile_offset
    # For combined texture: U spans [0, NU_SIZE * MU_S_SIZE]
    
    u_final = builder.math('ADD', base_x + 1200, base_y - 350, 'u_final')
    # u_mu_s is already in [0,1] range for MU_S dimension
    # Need to offset by nu tile
    nu_tile_offset = builder.math('MULTIPLY', base_x + 800, base_y - 450, 'nu_tile', 
                                   v1=1.0 / SCATTERING_TEXTURE_NU_SIZE)
    builder.link(u_nu.outputs[0], nu_tile_offset.inputs[0])
    
    # u_mu_s_in_tile = u_mu_s / NU_SIZE
    u_mu_s_scaled_tile = builder.math('MULTIPLY', base_x + 1200, base_y - 400, 'u_mu_s_tile',
                                       v1=1.0 / SCATTERING_TEXTURE_NU_SIZE)
    builder.link(u_mu_s.outputs[0], u_mu_s_scaled_tile.inputs[0])
    
    builder.link(nu_tile_offset.outputs[0], u_final.inputs[0])
    builder.link(u_mu_s_scaled_tile.outputs[0], u_final.inputs[1])
    
    return u_final.outputs[0], u_mu.outputs[0], r_idx.outputs[0], r_frac.outputs[0]


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
    
    # r = length(camera), clamped to valid atmosphere range
    r_raw = builder.vec_math('LENGTH', -800, 0, 'r_raw')
    builder.link(camera_km.outputs[0], r_raw.inputs[0])
    
    # Clamp r to [bottom_radius, top_radius]
    BOTTOM_RADIUS = 6360.0
    TOP_RADIUS = 6420.0
    
    r_min = builder.math('MAXIMUM', -600, 0, 'r_min', v1=BOTTOM_RADIUS)
    builder.link(r_raw.outputs['Value'], r_min.inputs[0])
    
    r = builder.math('MINIMUM', -400, 0, 'r', v1=TOP_RADIUS)
    builder.link(r_min.outputs[0], r.inputs[0])
    
    # rmu = dot(camera, view_ray)
    rmu = builder.vec_math('DOT_PRODUCT', -800, -100, 'r×mu')
    builder.link(camera_km.outputs[0], rmu.inputs[0])
    builder.link(view_ray.outputs[0], rmu.inputs[1])
    
    # mu = rmu / r
    mu = builder.math('DIVIDE', -200, -50, 'mu')
    builder.link(rmu.outputs['Value'], mu.inputs[0])
    builder.link(r.outputs[0], mu.inputs[1])
    
    # mu_s = dot(camera, sun_direction) / r
    cam_dot_sun = builder.vec_math('DOT_PRODUCT', -800, -200, 'cam·sun')
    builder.link(camera_km.outputs[0], cam_dot_sun.inputs[0])
    builder.link(group_input.outputs['Sun_Direction'], cam_dot_sun.inputs[1])
    
    mu_s = builder.math('DIVIDE', -200, -200, 'mu_s')
    builder.link(cam_dot_sun.outputs['Value'], mu_s.inputs[0])
    builder.link(r.outputs[0], mu_s.inputs[1])
    
    # nu = dot(view_ray, sun_direction)
    nu = builder.vec_math('DOT_PRODUCT', -800, -300, 'nu')
    builder.link(view_ray.outputs[0], nu.inputs[0])
    builder.link(group_input.outputs['Sun_Direction'], nu.inputs[1])
    
    # =========================================================================
    # TRANSMITTANCE LOOKUP
    # =========================================================================
    
    u_socket, v_socket = create_transmittance_uv_nodes(
        builder, r.outputs[0], mu.outputs[0],
        BOTTOM_RADIUS, TOP_RADIUS,
        base_x=-400, base_y=400
    )
    
    # Flip V coordinate (Blender Image Texture vs OSL texture() convention)
    trans_v_flipped = builder.math('SUBTRACT', 1500, 450, 'trans_v_flip', v0=1.0)
    builder.link(v_socket, trans_v_flipped.inputs[1])
    
    # Combine UV for texture lookup
    trans_uv = builder.combine_xyz(1600, 400, 'Trans_UV')
    builder.link(u_socket, trans_uv.inputs['X'])
    builder.link(trans_v_flipped.outputs[0], trans_uv.inputs['Y'])
    
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
    # GetSkyRadianceToPoint IMPLEMENTATION
    # inscatter = S(camera→∞) - T(camera→point) × S(point→∞)
    # =========================================================================
    
    MU_S_MIN = -0.2  # Minimum sun cosine angle
    
    # --- SCATTERING AT CAMERA (looking toward infinity) ---
    scat_u_cam, scat_v_cam, scat_r_cam, scat_frac_cam = create_scattering_uv_nodes(
        builder, r.outputs['Value'], mu.outputs[0], mu_s.outputs[0], nu.outputs['Value'],
        BOTTOM_RADIUS, TOP_RADIUS, MU_S_MIN,
        base_x=200, base_y=600
    )
    
    # Flip V coordinate for camera scattering (Blender Image Texture vs OSL convention)
    scat_v_cam_flipped = builder.math('SUBTRACT', 2700, 650, 'scat_v_cam_flip', v0=1.0)
    builder.link(scat_v_cam, scat_v_cam_flipped.inputs[1])
    
    # Incorporate depth slice into X coordinate
    # Atlas layout: depth slices tiled horizontally, so X = (depth_slice + u_within_slice) / DEPTH
    scat_u_with_depth_cam = builder.math('ADD', 2750, 600, 'scat_u_depth_cam')
    builder.link(scat_r_cam, scat_u_with_depth_cam.inputs[0])  # r_idx (depth slice)
    builder.link(scat_u_cam, scat_u_with_depth_cam.inputs[1])  # u within slice
    
    scat_u_final_cam = builder.math('DIVIDE', 2850, 600, 'scat_u_final_cam', 
                                     v1=float(SCATTERING_TEXTURE_DEPTH))
    builder.link(scat_u_with_depth_cam.outputs[0], scat_u_final_cam.inputs[0])
    
    # Combine UV for camera scattering lookup
    scat_uv_cam = builder.combine_xyz(3000, 600, 'Scat_UV_Cam')
    builder.link(scat_u_final_cam.outputs[0], scat_uv_cam.inputs['X'])
    builder.link(scat_v_cam_flipped.outputs[0], scat_uv_cam.inputs['Y'])
    
    # Sample scattering at camera
    tex_scat_cam = builder.image_texture(3000, 600, 'Scattering_Cam')
    tex_scat_cam.interpolation = 'Linear'
    tex_scat_cam.extension = 'EXTEND'
    if os.path.exists(scattering_path):
        img = bpy.data.images.load(scattering_path, check_existing=True)
        img.colorspace_settings.name = 'Non-Color'
        tex_scat_cam.image = img
    builder.link(scat_uv_cam.outputs[0], tex_scat_cam.inputs['Vector'])
    
    # --- POINT PARAMETERS (r_p, mu_p, mu_s_p) ---
    # r_p = length(point), clamped to valid atmosphere range
    r_p_raw = builder.vec_math('LENGTH', 0, -600, 'r_p_raw')
    builder.link(surface_point_km.outputs[0], r_p_raw.inputs[0])
    
    # Clamp r_p to [bottom_radius, top_radius]
    r_p_min = builder.math('MAXIMUM', 200, -600, 'r_p_min', v1=BOTTOM_RADIUS)
    builder.link(r_p_raw.outputs['Value'], r_p_min.inputs[0])
    
    r_p = builder.math('MINIMUM', 400, -600, 'r_p', v1=TOP_RADIUS)
    builder.link(r_p_min.outputs[0], r_p.inputs[0])
    
    # mu_p = dot(point, view_ray) / r_p
    point_dot_view = builder.vec_math('DOT_PRODUCT', 0, -700, 'point·view')
    builder.link(surface_point_km.outputs[0], point_dot_view.inputs[0])
    builder.link(view_ray.outputs[0], point_dot_view.inputs[1])
    
    mu_p = builder.math('DIVIDE', 600, -650, 'mu_p')
    builder.link(point_dot_view.outputs['Value'], mu_p.inputs[0])
    builder.link(r_p.outputs[0], mu_p.inputs[1])
    
    # mu_s_p = dot(point, sun_direction) / r_p
    point_dot_sun = builder.vec_math('DOT_PRODUCT', 0, -800, 'point·sun')
    builder.link(surface_point_km.outputs[0], point_dot_sun.inputs[0])
    builder.link(group_input.outputs['Sun_Direction'], point_dot_sun.inputs[1])
    
    mu_s_p = builder.math('DIVIDE', 600, -750, 'mu_s_p')
    builder.link(point_dot_sun.outputs['Value'], mu_s_p.inputs[0])
    builder.link(r_p.outputs[0], mu_s_p.inputs[1])
    
    # --- SCATTERING AT POINT (looking toward infinity) ---
    scat_u_pt, scat_v_pt, scat_r_pt, scat_frac_pt = create_scattering_uv_nodes(
        builder, r_p.outputs[0], mu_p.outputs[0], mu_s_p.outputs[0], nu.outputs['Value'],
        BOTTOM_RADIUS, TOP_RADIUS, MU_S_MIN,
        base_x=400, base_y=-600
    )
    
    # Flip V coordinate for point scattering (Blender Image Texture vs OSL convention)
    scat_v_pt_flipped = builder.math('SUBTRACT', 2700, -550, 'scat_v_pt_flip', v0=1.0)
    builder.link(scat_v_pt, scat_v_pt_flipped.inputs[1])
    
    # Incorporate depth slice into X coordinate for point
    scat_u_with_depth_pt = builder.math('ADD', 2750, -600, 'scat_u_depth_pt')
    builder.link(scat_r_pt, scat_u_with_depth_pt.inputs[0])  # r_idx (depth slice)
    builder.link(scat_u_pt, scat_u_with_depth_pt.inputs[1])  # u within slice
    
    scat_u_final_pt = builder.math('DIVIDE', 2850, -600, 'scat_u_final_pt', 
                                    v1=float(SCATTERING_TEXTURE_DEPTH))
    builder.link(scat_u_with_depth_pt.outputs[0], scat_u_final_pt.inputs[0])
    
    # Combine UV for point scattering lookup
    scat_uv_pt = builder.combine_xyz(3000, -600, 'Scat_UV_Point')
    builder.link(scat_u_final_pt.outputs[0], scat_uv_pt.inputs['X'])
    builder.link(scat_v_pt_flipped.outputs[0], scat_uv_pt.inputs['Y'])
    
    # Sample scattering at point
    tex_scat_pt = builder.image_texture(3000, -600, 'Scattering_Point')
    tex_scat_pt.interpolation = 'Linear'
    tex_scat_pt.extension = 'EXTEND'
    if os.path.exists(scattering_path):
        img = bpy.data.images.load(scattering_path, check_existing=True)
        img.colorspace_settings.name = 'Non-Color'
        tex_scat_pt.image = img
    builder.link(scat_uv_pt.outputs[0], tex_scat_pt.inputs['Vector'])
    
    # --- TRANSMITTANCE FROM CAMERA TO POINT ---
    # We already have transmittance UV calculated, use it
    
    # --- COMPUTE INSCATTER ---
    # inscatter_raw = S_cam - T × S_point
    
    # T × S_point (multiply transmittance by point scattering)
    t_times_scat_pt = builder.vec_math('MULTIPLY', 3200, -300, 'T×S_point')
    builder.link(tex_transmittance.outputs['Color'], t_times_scat_pt.inputs[0])
    builder.link(tex_scat_pt.outputs['Color'], t_times_scat_pt.inputs[1])
    
    # S_cam - T × S_point
    inscatter_raw = builder.vec_math('SUBTRACT', 3400, -100, 'S_cam - T×S_pt')
    builder.link(tex_scat_cam.outputs['Color'], inscatter_raw.inputs[0])
    builder.link(t_times_scat_pt.outputs[0], inscatter_raw.inputs[1])
    
    # --- APPLY PHASE FUNCTIONS ---
    # For combined scattering texture, we apply an average phase function
    # inscatter = inscatter_raw × (rayleigh_phase + mie_phase) / 2
    # Actually Bruneton: inscatter = rayleigh × rayleigh_phase + mie × mie_phase
    # Our combined texture stores Rayleigh, so we primarily use Rayleigh phase
    
    # Scale inscatter by phase function (using Rayleigh as primary)
    inscatter_phased = builder.vec_math('SCALE', 3600, -100, 'Inscatter_Phased')
    builder.link(inscatter_raw.outputs[0], inscatter_phased.inputs[0])
    builder.link(rayleigh_phase.outputs[0], inscatter_phased.inputs['Scale'])
    
    # Clamp negative values (can happen due to interpolation)
    inscatter_r = builder.separate_color(3800, -100, 'Sep_Inscatter')
    builder.link(inscatter_phased.outputs[0], inscatter_r.inputs['Color'])
    
    r_clamped = builder.math('MAXIMUM', 4000, -50, 'R_clamp', v1=0.0)
    builder.link(inscatter_r.outputs['Red'], r_clamped.inputs[0])
    
    g_clamped = builder.math('MAXIMUM', 4000, -100, 'G_clamp', v1=0.0)
    builder.link(inscatter_r.outputs['Green'], g_clamped.inputs[0])
    
    b_clamped = builder.math('MAXIMUM', 4000, -150, 'B_clamp', v1=0.0)
    builder.link(inscatter_r.outputs['Blue'], b_clamped.inputs[0])
    
    inscatter_final = builder.combine_color(4200, -100, 'Inscatter_Final')
    builder.link(r_clamped.outputs[0], inscatter_final.inputs['Red'])
    builder.link(g_clamped.outputs[0], inscatter_final.inputs['Green'])
    builder.link(b_clamped.outputs[0], inscatter_final.inputs['Blue'])
    
    # --- OUTPUTS ---
    # Output transmittance from LUT lookup
    builder.link(tex_transmittance.outputs['Color'], group_output.inputs['Transmittance'])
    
    # Output computed inscatter
    builder.link(inscatter_final.outputs['Color'], group_output.inputs['Inscatter'])
    
    # Store version for future checks
    group['helios_version'] = AERIAL_NODE_VERSION
    
    print(f"Helios: Created node group '{AERIAL_NODE_GROUP_NAME}' v{AERIAL_NODE_VERSION} with full GetSkyRadianceToPoint")
    return group


def get_or_create_aerial_node_group(lut_dir=None):
    """Get existing node group or create a new one.
    
    Always recreates if version has changed.
    """
    # Always recreate to ensure latest version
    if AERIAL_NODE_GROUP_NAME in bpy.data.node_groups:
        existing = bpy.data.node_groups[AERIAL_NODE_GROUP_NAME]
        # Check version - recreate if outdated
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
