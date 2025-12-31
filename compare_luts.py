"""
LUT Comparison Script - Compares reference LUTs to generated LUTs.

Supports .npy (NumPy), .exr (OpenEXR), and .raw formats.

Usage:
    python compare_luts.py <reference_dir> <generated_dir>

Example:
    python compare_luts.py ../atmospheric-scattering-2-export/lut_export ./helios_cache/luts
"""

import os
import sys
import numpy as np

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("Note: OpenEXR not available")

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False


def load_npy(filepath):
    """Load NumPy .npy file."""
    return np.load(filepath)


def load_file(filepath):
    """Load LUT file (supports .npy, .exr)."""
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.npy':
        return load_npy(filepath)
    elif ext == '.exr':
        return load_exr(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


def load_exr(filepath):
    """Load EXR file and return numpy array."""
    if HAS_OPENEXR:
        exr_file = OpenEXR.InputFile(filepath)
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        
        # Get channel names
        channels = list(header['channels'].keys())
        
        # Read channels
        float_type = Imath.PixelType(Imath.PixelType.FLOAT)
        data = {}
        for ch in channels:
            raw = exr_file.channel(ch, float_type)
            data[ch] = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
        
        # Combine into RGB(A) array
        if 'R' in data and 'G' in data and 'B' in data:
            if 'A' in data:
                result = np.stack([data['R'], data['G'], data['B'], data['A']], axis=-1)
            else:
                result = np.stack([data['R'], data['G'], data['B']], axis=-1)
        else:
            # Single channel or other format
            result = list(data.values())[0]
        
        return result
    else:
        return imageio.imread(filepath)


def compare_arrays(name, ref, gen):
    """Compare two arrays and print statistics."""
    print(f"\n{'='*60}")
    print(f"Comparing: {name}")
    print(f"{'='*60}")
    
    # Shape check
    if ref.shape != gen.shape:
        print(f"  ERROR: Shape mismatch! Reference: {ref.shape}, Generated: {gen.shape}")
        return
    
    print(f"  Shape: {ref.shape}")
    
    # Basic statistics
    print(f"\n  Reference stats:")
    print(f"    Min: {ref.min():.6f}, Max: {ref.max():.6f}, Mean: {ref.mean():.6f}")
    
    print(f"\n  Generated stats:")
    print(f"    Min: {gen.min():.6f}, Max: {gen.max():.6f}, Mean: {gen.mean():.6f}")
    
    # Difference analysis
    diff = np.abs(ref - gen)
    
    print(f"\n  Absolute Difference:")
    print(f"    Min: {diff.min():.6f}, Max: {diff.max():.6f}, Mean: {diff.mean():.6f}")
    print(f"    Std: {diff.std():.6f}")
    
    # Check for NaN/Inf
    ref_nan = np.sum(np.isnan(ref))
    gen_nan = np.sum(np.isnan(gen))
    ref_inf = np.sum(np.isinf(ref))
    gen_inf = np.sum(np.isinf(gen))
    
    if ref_nan > 0 or gen_nan > 0:
        print(f"\n  WARNING: NaN values! Reference: {ref_nan}, Generated: {gen_nan}")
    if ref_inf > 0 or gen_inf > 0:
        print(f"\n  WARNING: Inf values! Reference: {ref_inf}, Generated: {gen_inf}")
    
    # Relative error (where reference is non-zero)
    mask = np.abs(ref) > 1e-6
    if np.any(mask):
        rel_diff = np.abs((ref[mask] - gen[mask]) / ref[mask])
        print(f"\n  Relative Error (where ref > 1e-6):")
        print(f"    Mean: {rel_diff.mean()*100:.2f}%, Max: {rel_diff.max()*100:.2f}%")
    
    # Find location of max difference
    max_diff_idx = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"\n  Max difference location: {max_diff_idx}")
    print(f"    Reference value: {ref[max_diff_idx]:.6f}")
    print(f"    Generated value: {gen[max_diff_idx]:.6f}")
    
    # Per-channel analysis if RGB
    if len(ref.shape) == 3 and ref.shape[2] >= 3:
        print(f"\n  Per-channel mean absolute error:")
        for i, ch in enumerate(['R', 'G', 'B', 'A'][:ref.shape[2]]):
            ch_diff = np.abs(ref[:,:,i] - gen[:,:,i]).mean()
            print(f"    {ch}: {ch_diff:.6f}")
    
    # Sample specific pixels
    print(f"\n  Sample pixel values (center of image):")
    h, w = ref.shape[:2]
    cy, cx = h // 2, w // 2
    print(f"    Center ({cx}, {cy}):")
    print(f"      Reference: {ref[cy, cx]}")
    print(f"      Generated: {gen[cy, cx]}")
    
    # Corner pixels
    print(f"    Top-left (0, 0):")
    print(f"      Reference: {ref[0, 0]}")
    print(f"      Generated: {gen[0, 0]}")
    
    print(f"    Bottom-right ({w-1}, {h-1}):")
    print(f"      Reference: {ref[-1, -1]}")
    print(f"      Generated: {gen[-1, -1]}")
    
    # Exact match count
    exact_matches = np.sum(ref == gen)
    total = ref.size
    print(f"\n  Exact matches: {exact_matches}/{total} ({100*exact_matches/total:.2f}%)")
    
    # Close matches (within tolerance)
    for tol in [1e-6, 1e-4, 1e-2, 0.1]:
        close = np.sum(diff < tol)
        print(f"  Within {tol}: {close}/{total} ({100*close/total:.2f}%)")


def find_file(directory, basename):
    """Find a file with any supported extension."""
    for ext in ['.npy', '.exr']:
        path = os.path.join(directory, basename + ext)
        if os.path.exists(path):
            return path
    return None


def reshape_for_comparison(ref, gen, name):
    """Reshape arrays if needed for comparison.
    
    Reference .npy files may have different shape conventions than our EXR files.
    - Transmittance: Should be 2D (height, width, 3)
    - Scattering: 3D LUT stored as tiled 2D (height, width, 4) 
    - Irradiance: 2D (height, width, 3)
    """
    print(f"  Reference shape: {ref.shape}")
    print(f"  Generated shape: {gen.shape}")
    
    # If shapes match, no reshaping needed
    if ref.shape == gen.shape:
        return ref, gen
    
    # Handle transmittance: reference might be (H, W, 3), ours (H, W, 4)
    if 'transmittance' in name.lower():
        if len(ref.shape) == 3 and len(gen.shape) == 3:
            # Compare only RGB channels
            if ref.shape[2] == 3 and gen.shape[2] == 4:
                gen = gen[:, :, :3]
            elif ref.shape[2] == 4 and gen.shape[2] == 3:
                ref = ref[:, :, :3]
    
    # Handle scattering: 3D vs tiled 2D
    if 'scattering' in name.lower():
        # Reference scattering.npy is typically (depth, height, width, channels)
        # Our EXR is tiled 2D (height, width*depth, channels)
        if len(ref.shape) == 4 and len(gen.shape) == 3:
            # Reshape reference from 3D to tiled 2D
            d, h, w, c = ref.shape
            ref_tiled = np.zeros((h, w * d, c), dtype=ref.dtype)
            for layer in range(d):
                ref_tiled[:, layer * w:(layer + 1) * w, :] = ref[layer]
            ref = ref_tiled
            print(f"  Reshaped reference to tiled 2D: {ref.shape}")
    
    # Handle irradiance
    if 'irradiance' in name.lower():
        if len(ref.shape) == 3 and len(gen.shape) == 3:
            # Compare only RGB channels
            min_ch = min(ref.shape[2], gen.shape[2])
            ref = ref[:, :, :min_ch]
            gen = gen[:, :, :min_ch]
    
    return ref, gen


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nUsage: python compare_luts.py <reference_dir> <generated_dir>")
        print("\nExample:")
        print("  python compare_luts.py ../atmospheric-scattering-2-export/lut_export ./helios_cache/luts")
        sys.exit(1)
    
    ref_dir = sys.argv[1]
    gen_dir = sys.argv[2]
    
    print(f"Reference directory: {ref_dir}")
    print(f"Generated directory: {gen_dir}")
    
    # LUT names to compare (without extension)
    luts_to_compare = [
        'transmittance',
        'scattering',
        'irradiance',
    ]
    
    for name in luts_to_compare:
        ref_path = find_file(ref_dir, name)
        gen_path = find_file(gen_dir, name)
        
        if not ref_path:
            print(f"\nSkipping {name}: Reference not found in {ref_dir}")
            continue
        if not gen_path:
            print(f"\nSkipping {name}: Generated not found in {gen_dir}")
            continue
        
        print(f"\nLoading {name}...")
        print(f"  Reference: {ref_path}")
        print(f"  Generated: {gen_path}")
        
        try:
            ref_data = load_file(ref_path)
            gen_data = load_file(gen_path)
            
            # Reshape if needed
            ref_data, gen_data = reshape_for_comparison(ref_data, gen_data, name)
            
            compare_arrays(name, ref_data, gen_data)
        except Exception as e:
            print(f"\nError comparing {name}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
