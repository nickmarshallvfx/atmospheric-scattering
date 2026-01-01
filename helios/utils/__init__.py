"""
Helios Utilities
"""

import bpy
import os


def get_lut_cache_dir():
    """Get the directory where precomputed LUTs are stored."""
    blend_path = bpy.data.filepath
    if blend_path:
        cache_dir = os.path.join(os.path.dirname(blend_path), "helios_cache", "luts")
        if os.path.exists(cache_dir):
            return cache_dir
    
    config_dir = bpy.utils.user_resource('CONFIG')
    return os.path.join(config_dir, "helios_cache", "luts")


# Safe imports - these modules handle missing dependencies internally
try:
    from .gpu import AtmosphereTextures, get_atmosphere_textures
except ImportError:
    pass

try:
    from .exr import HeliosEXRWriter, read_helios_exr
except ImportError:
    pass
