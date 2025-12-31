"""
Helios GPU Utilities - Texture upload and management for Blender.
"""

import numpy as np
from typing import Optional, Tuple

try:
    import bpy
    import gpu
    from gpu.types import GPUTexture
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    GPUTexture = None


class AtmosphereTextures:
    """Manages GPU textures for atmosphere rendering."""
    
    def __init__(self):
        self.transmittance: Optional[GPUTexture] = None
        self.scattering: Optional[GPUTexture] = None
        self.irradiance: Optional[GPUTexture] = None
        self.single_mie: Optional[GPUTexture] = None
        self._is_valid = False
    
    @property
    def is_valid(self) -> bool:
        return self._is_valid and self.transmittance is not None
    
    def upload_from_arrays(
        self,
        transmittance: np.ndarray,
        scattering: np.ndarray,
        irradiance: np.ndarray,
        single_mie: Optional[np.ndarray] = None
    ) -> None:
        """
        Upload precomputed LUT arrays to GPU textures.
        
        Args:
            transmittance: 2D array (H, W, 3)
            scattering: 3D array (D, H, W, 4)
            irradiance: 2D array (H, W, 3)
            single_mie: Optional 3D array (D, H, W, 3)
        """
        if not HAS_GPU:
            raise RuntimeError("GPU module not available")
        
        # Free existing textures
        self.free()
        
        # Upload transmittance (2D texture)
        self.transmittance = self._create_2d_texture(transmittance)
        
        # Upload irradiance (2D texture)
        self.irradiance = self._create_2d_texture(irradiance)
        
        # Upload scattering (3D texture)
        self.scattering = self._create_3d_texture(scattering)
        
        # Upload single Mie if provided
        if single_mie is not None:
            self.single_mie = self._create_3d_texture(single_mie)
        
        self._is_valid = True
    
    def _create_2d_texture(self, data: np.ndarray) -> GPUTexture:
        """Create a 2D GPU texture from numpy array."""
        # Ensure float32 and contiguous
        data = np.ascontiguousarray(data, dtype=np.float32)
        
        height, width = data.shape[:2]
        channels = data.shape[2] if len(data.shape) > 2 else 1
        
        # Determine format
        if channels == 1:
            format_str = 'R32F'
        elif channels == 3:
            format_str = 'RGB32F'
        elif channels == 4:
            format_str = 'RGBA32F'
        else:
            raise ValueError(f"Unsupported channel count: {channels}")
        
        # Create texture
        # Note: Blender 4.0+ uses different GPU API
        # This is a simplified version - actual impl depends on Blender version
        texture = gpu.types.GPUTexture((width, height), format=format_str, data=gpu.types.Buffer('FLOAT', data.size, data.flatten()))
        
        return texture
    
    def _create_3d_texture(self, data: np.ndarray) -> GPUTexture:
        """Create a 3D GPU texture from numpy array."""
        # Ensure float32 and contiguous
        data = np.ascontiguousarray(data, dtype=np.float32)
        
        depth, height, width = data.shape[:3]
        channels = data.shape[3] if len(data.shape) > 3 else 1
        
        # For 3D textures, we may need to pack into 2D atlas
        # This depends on Blender's GPU capabilities
        # For now, return a placeholder
        
        # Note: Full 3D texture support requires checking Blender version
        # and potentially using shader storage buffers or texture atlases
        
        return None  # Placeholder - implement based on Blender version
    
    def free(self) -> None:
        """Free all GPU textures."""
        # In Blender, textures are garbage collected
        # but we clear references to allow collection
        self.transmittance = None
        self.scattering = None
        self.irradiance = None
        self.single_mie = None
        self._is_valid = False
    
    def bind_to_shader(self, shader, texture_slots: dict) -> None:
        """
        Bind textures to shader uniform slots.
        
        Args:
            shader: GPU shader program
            texture_slots: Dict mapping texture names to slot indices
        """
        if not self.is_valid:
            raise RuntimeError("Textures not initialized")
        
        # Bind each texture to its slot
        if 'transmittance' in texture_slots and self.transmittance:
            shader.uniform_sampler("transmittance_texture", self.transmittance)
        
        if 'irradiance' in texture_slots and self.irradiance:
            shader.uniform_sampler("irradiance_texture", self.irradiance)
        
        if 'scattering' in texture_slots and self.scattering:
            shader.uniform_sampler("scattering_texture", self.scattering)


def get_atmosphere_textures() -> AtmosphereTextures:
    """Get or create the global atmosphere textures instance."""
    # Store in bpy.app.driver_namespace to persist across reloads
    key = '_helios_atmosphere_textures'
    
    if not HAS_GPU:
        return AtmosphereTextures()
    
    if key not in bpy.app.driver_namespace:
        bpy.app.driver_namespace[key] = AtmosphereTextures()
    
    return bpy.app.driver_namespace[key]
