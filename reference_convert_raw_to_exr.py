"""
Convert raw float texture exports from Bruneton reference app to EXR files.

Usage: python convert_raw_to_exr.py [input_dir] [output_dir]

Reads .raw files with format:
- 2D: [width, height, channels] as int32, then float32 data
- 3D: [width, height, depth, channels] as int32, then float32 data (exported as tiled 2D)
"""

import numpy as np
import struct
import os
import sys

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("Warning: OpenEXR not available, will save as .npy instead")


def read_raw_2d(filepath):
    """Read a 2D raw texture file."""
    with open(filepath, 'rb') as f:
        dims = struct.unpack('3i', f.read(12))
        width, height, channels = dims
        data = np.frombuffer(f.read(), dtype=np.float32)
        data = data.reshape((height, width, channels))
    return data, (width, height, channels)


def read_raw_3d(filepath):
    """Read a 3D raw texture file."""
    with open(filepath, 'rb') as f:
        dims = struct.unpack('4i', f.read(16))
        width, height, depth, channels = dims
        data = np.frombuffer(f.read(), dtype=np.float32)
        data = data.reshape((depth, height, width, channels))
    return data, (width, height, depth, channels)


def save_exr_2d(data, filepath):
    """Save 2D RGBA data to EXR."""
    if not HAS_OPENEXR:
        np.save(filepath.replace('.exr', '.npy'), data)
        return
    
    height, width, channels = data.shape
    
    header = OpenEXR.Header(width, height)
    header['compression'] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    
    # Convert to channel dict
    channel_names = ['R', 'G', 'B', 'A'][:channels]
    channel_data = {}
    for i, name in enumerate(channel_names):
        channel_data[name] = data[:, :, i].astype(np.float32).tobytes()
    
    out = OpenEXR.OutputFile(filepath, header)
    out.writePixels(channel_data)
    out.close()


def save_exr_3d_tiled(data, filepath):
    """Save 3D texture as tiled 2D EXR (depth slices arranged horizontally)."""
    depth, height, width, channels = data.shape
    
    # Tile depth slices horizontally
    tiled_width = width * depth
    tiled = np.zeros((height, tiled_width, channels), dtype=np.float32)
    
    for z in range(depth):
        tiled[:, z*width:(z+1)*width, :] = data[z, :, :, :]
    
    if not HAS_OPENEXR:
        np.save(filepath.replace('.exr', '.npy'), tiled)
        return
    
    header = OpenEXR.Header(tiled_width, height)
    header['compression'] = Imath.Compression(Imath.Compression.ZIP_COMPRESSION)
    
    channel_names = ['R', 'G', 'B', 'A'][:channels]
    channel_data = {}
    for i, name in enumerate(channel_names):
        channel_data[name] = tiled[:, :, i].astype(np.float32).tobytes()
    
    out = OpenEXR.OutputFile(filepath, header)
    out.writePixels(channel_data)
    out.close()


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else './lut_export'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './lut_export'
    
    print(f"Converting raw textures from {input_dir} to {output_dir}")
    
    # Transmittance (2D)
    trans_path = os.path.join(input_dir, 'transmittance.raw')
    if os.path.exists(trans_path):
        data, dims = read_raw_2d(trans_path)
        print(f"  transmittance: {dims}, range [{data.min():.6f}, {data.max():.6f}]")
        save_exr_2d(data, os.path.join(output_dir, 'transmittance.exr'))
        print(f"    -> transmittance.exr")
    
    # Scattering (3D -> tiled 2D)
    scat_path = os.path.join(input_dir, 'scattering.raw')
    if os.path.exists(scat_path):
        data, dims = read_raw_3d(scat_path)
        print(f"  scattering: {dims}, range [{data.min():.6f}, {data.max():.6f}]")
        save_exr_3d_tiled(data, os.path.join(output_dir, 'scattering.exr'))
        print(f"    -> scattering.exr (tiled: {dims[0]*dims[2]}x{dims[1]})")
    
    # Irradiance (2D)
    irr_path = os.path.join(input_dir, 'irradiance.raw')
    if os.path.exists(irr_path):
        data, dims = read_raw_2d(irr_path)
        print(f"  irradiance: {dims}, range [{data.min():.6f}, {data.max():.6f}]")
        save_exr_2d(data, os.path.join(output_dir, 'irradiance.exr'))
        print(f"    -> irradiance.exr")
    
    # Single Mie Scattering (3D -> tiled 2D, optional)
    mie_path = os.path.join(input_dir, 'single_mie_scattering.raw')
    if os.path.exists(mie_path):
        data, dims = read_raw_3d(mie_path)
        print(f"  single_mie_scattering: {dims}, range [{data.min():.6f}, {data.max():.6f}]")
        save_exr_3d_tiled(data, os.path.join(output_dir, 'single_mie_scattering.exr'))
        print(f"    -> single_mie_scattering.exr")
    
    print("Done!")


if __name__ == '__main__':
    main()
