"""
Helios EXR Utilities - Multi-layer EXR export for Nuke.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import os

# Try to import OpenEXR - may not be available in all Blender builds
try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False


class HeliosEXRWriter:
    """
    Writes multi-layer EXR files with Helios AOV channels.
    
    Channel naming convention for Nuke compatibility:
    - helios.sky.R, helios.sky.G, helios.sky.B
    - helios.transmittance.R, helios.transmittance.G, helios.transmittance.B
    - helios.inscatter.R, helios.inscatter.G, helios.inscatter.B
    """
    
    CHANNEL_PREFIX = "helios"
    
    def __init__(self, width: int, height: int, half_precision: bool = False):
        """
        Initialize EXR writer.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            half_precision: Use 16-bit float (True) or 32-bit float (False)
        """
        self.width = width
        self.height = height
        self.half_precision = half_precision
        self.layers: Dict[str, np.ndarray] = {}
    
    def add_layer(self, name: str, data: np.ndarray) -> None:
        """
        Add a layer to the EXR.
        
        Args:
            name: Layer name (e.g., "sky", "transmittance", "inscatter")
            data: Image data as (height, width, 3) array
        """
        if data.shape != (self.height, self.width, 3):
            raise ValueError(f"Layer data must be shape ({self.height}, {self.width}, 3), "
                           f"got {data.shape}")
        
        self.layers[name] = np.asarray(data, dtype=np.float32)
    
    def write(self, filepath: str) -> None:
        """
        Write the EXR file.
        
        Args:
            filepath: Output file path
        """
        if not HAS_OPENEXR:
            raise RuntimeError("OpenEXR module not available. "
                             "Install with: pip install OpenEXR")
        
        if not self.layers:
            raise ValueError("No layers added to EXR")
        
        # Set up pixel type
        if self.half_precision:
            pixel_type = Imath.PixelType(Imath.PixelType.HALF)
        else:
            pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        
        # Build channel dictionary
        header = OpenEXR.Header(self.width, self.height)
        channels = {}
        channel_data = {}
        
        for layer_name, data in self.layers.items():
            for i, channel in enumerate(['R', 'G', 'B']):
                full_name = f"{self.CHANNEL_PREFIX}.{layer_name}.{channel}"
                channels[full_name] = Imath.Channel(pixel_type)
                
                # Extract channel and convert to bytes
                channel_array = data[:, :, i].astype(np.float32)
                channel_data[full_name] = channel_array.tobytes()
        
        header['channels'] = channels
        
        # Write file
        exr_file = OpenEXR.OutputFile(filepath, header)
        exr_file.writePixels(channel_data)
        exr_file.close()
    
    @classmethod
    def write_atmosphere_passes(
        cls,
        filepath: str,
        sky: np.ndarray,
        transmittance: np.ndarray,
        inscatter: np.ndarray,
        half_precision: bool = False
    ) -> None:
        """
        Convenience method to write standard atmosphere AOVs.
        
        Args:
            filepath: Output file path
            sky: Sky radiance (H, W, 3)
            transmittance: Atmospheric transmittance (H, W, 3)
            inscatter: Atmospheric inscattering (H, W, 3)
            half_precision: Use half float precision
        """
        height, width = sky.shape[:2]
        writer = cls(width, height, half_precision)
        
        writer.add_layer("sky", sky)
        writer.add_layer("transmittance", transmittance)
        writer.add_layer("inscatter", inscatter)
        
        writer.write(filepath)


def read_helios_exr(filepath: str) -> Dict[str, np.ndarray]:
    """
    Read Helios AOV layers from an EXR file.
    
    Args:
        filepath: Path to EXR file
        
    Returns:
        Dictionary mapping layer names to (H, W, 3) arrays
    """
    if not HAS_OPENEXR:
        raise RuntimeError("OpenEXR module not available")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"EXR file not found: {filepath}")
    
    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()
    
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # Find Helios channels
    channels = header['channels']
    helios_layers = {}
    
    prefix = HeliosEXRWriter.CHANNEL_PREFIX + "."
    
    for channel_name in channels:
        if channel_name.startswith(prefix):
            parts = channel_name[len(prefix):].split('.')
            if len(parts) == 2:
                layer_name, component = parts
                if layer_name not in helios_layers:
                    helios_layers[layer_name] = {}
                helios_layers[layer_name][component] = channel_name
    
    # Read layers
    result = {}
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    
    for layer_name, components in helios_layers.items():
        if 'R' in components and 'G' in components and 'B' in components:
            r_str = exr_file.channel(components['R'], pt)
            g_str = exr_file.channel(components['G'], pt)
            b_str = exr_file.channel(components['B'], pt)
            
            r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
            g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
            b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)
            
            result[layer_name] = np.stack([r, g, b], axis=2)
    
    exr_file.close()
    return result
