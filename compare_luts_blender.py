"""
LUT Comparison Script for Blender - Run from Blender's Text Editor or scripting console.

Compares reference EXRs to generated EXRs using Blender's image loading.
"""

import bpy
import numpy as np


def load_exr_as_array(filepath):
    """Load EXR using Blender and return as numpy array."""
    import os
    import time
    
    # Check file exists and show modification time
    if os.path.exists(filepath):
        mtime = os.path.getmtime(filepath)
        mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        fsize = os.path.getsize(filepath)
        print(f"    File: {os.path.basename(filepath)}, modified: {mtime_str}, size: {fsize} bytes")
    
    # Create a temporary image
    img = bpy.data.images.load(filepath, check_existing=False)
    
    # Get dimensions
    width = img.size[0]
    height = img.size[1]
    channels = img.channels
    
    # Get pixel data as flat array
    pixels = np.array(img.pixels[:])
    
    # Reshape to (height, width, channels)
    pixels = pixels.reshape(height, width, channels)
    
    # Remove the temporary image
    bpy.data.images.remove(img)
    
    return pixels


def compare_arrays(name, ref, gen):
    """Compare two arrays and print statistics."""
    print(f"\n{'='*60}")
    print(f"Comparing: {name}")
    print(f"{'='*60}")
    
    print(f"  Reference shape: {ref.shape}")
    print(f"  Generated shape: {gen.shape}")
    
    # Handle shape differences
    if ref.shape != gen.shape:
        # Try to compare common channels
        if len(ref.shape) == 3 and len(gen.shape) == 3:
            min_ch = min(ref.shape[2], gen.shape[2])
            ref = ref[:, :, :min_ch]
            gen = gen[:, :, :min_ch]
            print(f"  Comparing first {min_ch} channels only")
        
        if ref.shape != gen.shape:
            print(f"  ERROR: Shape mismatch after adjustment!")
            return
    
    # Basic statistics
    print(f"\n  Reference: min={ref.min():.6f}, max={ref.max():.6f}, mean={ref.mean():.6f}")
    print(f"  Generated: min={gen.min():.6f}, max={gen.max():.6f}, mean={gen.mean():.6f}")
    
    # Difference analysis
    diff = np.abs(ref - gen)
    
    print(f"\n  Absolute Difference:")
    print(f"    Min: {diff.min():.6f}, Max: {diff.max():.6f}, Mean: {diff.mean():.6f}")
    
    # Check for NaN/Inf
    ref_nan = np.sum(np.isnan(ref))
    gen_nan = np.sum(np.isnan(gen))
    if ref_nan > 0 or gen_nan > 0:
        print(f"\n  WARNING: NaN values! Reference: {ref_nan}, Generated: {gen_nan}")
    
    # Relative error
    mask = np.abs(ref) > 1e-6
    if np.any(mask):
        rel_diff = np.abs((ref[mask] - gen[mask]) / ref[mask])
        print(f"\n  Relative Error (where ref > 1e-6):")
        print(f"    Mean: {rel_diff.mean()*100:.2f}%, Max: {rel_diff.max()*100:.2f}%")
    
    # Per-channel analysis
    if len(ref.shape) == 3 and ref.shape[2] >= 3:
        print(f"\n  Per-channel mean absolute error:")
        for i, ch in enumerate(['R', 'G', 'B', 'A'][:ref.shape[2]]):
            ch_diff = np.abs(ref[:,:,i] - gen[:,:,i]).mean()
            print(f"    {ch}: {ch_diff:.6f}")
    
    # Sample pixels
    h, w = ref.shape[:2]
    cy, cx = h // 2, w // 2
    print(f"\n  Center pixel ({cx}, {cy}):")
    print(f"    Reference: {ref[cy, cx]}")
    print(f"    Generated: {gen[cy, cx]}")
    
    # Match percentages
    total = ref.size
    for tol in [1e-6, 1e-4, 1e-2, 0.1]:
        close = np.sum(diff < tol)
        print(f"  Within {tol}: {100*close/total:.1f}%")


def compare_luts(ref_dir, gen_dir):
    """Compare all LUTs in two directories."""
    import os
    
    print(f"Reference: {ref_dir}")
    print(f"Generated: {gen_dir}")
    
    for name in ['transmittance', 'scattering', 'irradiance']:
        ref_path = os.path.join(ref_dir, f'{name}.exr')
        gen_path = os.path.join(gen_dir, f'{name}.exr')
        
        if not os.path.exists(ref_path):
            print(f"\nSkipping {name}: Reference not found")
            continue
        if not os.path.exists(gen_path):
            print(f"\nSkipping {name}: Generated not found")
            continue
        
        try:
            ref_data = load_exr_as_array(ref_path)
            gen_data = load_exr_as_array(gen_path)
            compare_arrays(name, ref_data, gen_data)
        except Exception as e:
            print(f"\nError comparing {name}: {e}")
            import traceback
            traceback.print_exc()


def compare_npy_to_exr(npy_path, exr_path, name):
    """Compare reference .npy file to generated .exr file."""
    import os
    
    print(f"\n{'='*60}")
    print(f"Comparing {name}: .npy (reference) vs .exr (generated)")
    print(f"{'='*60}")
    
    # Load reference .npy
    ref = np.load(npy_path)
    print(f"  Reference .npy shape: {ref.shape}")
    print(f"  Reference: min={ref.min():.6f}, max={ref.max():.6f}, mean={ref.mean():.6f}")
    
    # Load generated .exr
    gen = load_exr_as_array(exr_path)
    print(f"  Generated .exr shape: {gen.shape}")
    print(f"  Generated: min={gen.min():.6f}, max={gen.max():.6f}, mean={gen.mean():.6f}")
    
    # For transmittance: ref is (H, W, 3), gen is (H, W, 4)
    if 'transmittance' in name.lower():
        if len(ref.shape) == 3 and ref.shape[2] == 3:
            gen_rgb = gen[:, :, :3]
            print(f"\n  Comparing RGB channels only:")
            diff = np.abs(ref - gen_rgb)
            print(f"    Absolute Difference: min={diff.min():.6f}, max={diff.max():.6f}, mean={diff.mean():.6f}")
            
            # Sample pixels
            h, w = ref.shape[:2]
            print(f"\n  Sample pixels:")
            for y, x in [(0, 0), (h//2, w//2), (h-1, w-1)]:
                print(f"    ({x}, {y}): ref={ref[y,x]}, gen={gen_rgb[y,x]}")


def compare_scattering_npy_to_exr(npy_path, exr_path):
    """Compare scattering .npy (3D tiled) to generated .exr."""
    print(f"\n{'='*60}")
    print(f"Comparing SCATTERING: .npy (reference) vs .exr (generated)")
    print(f"{'='*60}")
    
    # Load reference .npy - shape is (depth, height, width, channels) or similar
    ref = np.load(npy_path)
    print(f"  Reference .npy shape: {ref.shape}")
    print(f"  Reference: min={ref.min():.6f}, max={ref.max():.6f}, mean={ref.mean():.6f}")
    
    # Load generated .exr - already tiled 2D
    gen = load_exr_as_array(exr_path)
    print(f"  Generated .exr shape: {gen.shape}")
    
    # Detailed inf analysis
    has_inf = np.any(np.isinf(gen))
    has_nan = np.any(np.isnan(gen))
    print(f"  Generated has inf: {has_inf}, has nan: {has_nan}")
    
    if has_inf:
        inf_mask = np.isinf(gen)
        inf_count = np.sum(inf_mask)
        inf_locations = np.where(inf_mask)
        print(f"  INF COUNT: {inf_count} out of {gen.size} ({100*inf_count/gen.size:.2f}%)")
        if len(inf_locations[0]) > 0:
            # Show first few inf locations
            for i in range(min(5, len(inf_locations[0]))):
                y, x, c = inf_locations[0][i], inf_locations[1][i], inf_locations[2][i]
                print(f"    inf at pixel ({x}, {y}, ch={c})")
        
        # Check if inf is only in alpha channel
        rgb_has_inf = np.any(np.isinf(gen[:,:,:3]))
        alpha_has_inf = np.any(np.isinf(gen[:,:,3])) if gen.shape[2] > 3 else False
        print(f"  RGB has inf: {rgb_has_inf}, Alpha has inf: {alpha_has_inf}")
    
    # Finite values only stats
    finite_gen = gen[np.isfinite(gen)]
    if len(finite_gen) > 0:
        print(f"  Generated (finite only): min={finite_gen.min():.6f}, max={finite_gen.max():.6f}, mean={finite_gen.mean():.6f}")
    
    # If ref is 2D tiled (same as gen), compare directly
    if len(ref.shape) == 3 and ref.shape == gen.shape:
        diff = np.abs(ref[:,:,:3] - gen[:,:,:3])
        print(f"\n  RGB Absolute Difference:")
        print(f"    Min: {diff.min():.6f}, Max: {diff.max():.6f}, Mean: {diff.mean():.6f}")
    else:
        print(f"\n  Shape mismatch - cannot do direct comparison")
        print(f"  Reference needs reshaping from {ref.shape} to match {gen.shape}")


# ============================================================
# RUN THIS IN BLENDER:
# ============================================================
if __name__ == "__main__":
    import os
    
    # TRUE reference from standalone Bruneton app
    NPY_DIR = r"c:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-2-export\lut_export"
    GEN_DIR = r"c:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-3\helios_cache\luts"
    
    # Compare transmittance
    npy_path = os.path.join(NPY_DIR, "transmittance.npy")
    exr_path = os.path.join(GEN_DIR, "transmittance.exr")
    if os.path.exists(npy_path) and os.path.exists(exr_path):
        compare_npy_to_exr(npy_path, exr_path, "transmittance")
    
    # Compare scattering against TRUE reference
    npy_path = os.path.join(NPY_DIR, "scattering.npy")
    exr_path = os.path.join(GEN_DIR, "scattering.exr")
    if os.path.exists(npy_path) and os.path.exists(exr_path):
        compare_scattering_npy_to_exr(npy_path, exr_path)
    
    # Compare irradiance against TRUE reference (RGB only, exclude alpha)
    npy_path = os.path.join(NPY_DIR, "irradiance.npy")
    exr_path = os.path.join(GEN_DIR, "irradiance.exr")
    if os.path.exists(npy_path) and os.path.exists(exr_path):
        ref = np.load(npy_path)
        gen = load_exr_as_array(exr_path)
        # Compare only RGB (first 3 channels) to exclude alpha=1.0
        print(f"\n{'='*60}")
        print(f"Comparing irradiance: .npy (reference) vs .exr (generated)")
        print(f"{'='*60}")
        print(f"  Reference .npy shape: {ref.shape}")
        print(f"  Generated .exr shape: {gen.shape}")
        ref_rgb = ref[:, :, :3]
        gen_rgb = gen[:, :, :3]
        print(f"  Reference (RGB): min={ref_rgb.min():.6f}, max={ref_rgb.max():.6f}, mean={ref_rgb.mean():.6f}")
        print(f"  Generated (RGB): min={gen_rgb.min():.6f}, max={gen_rgb.max():.6f}, mean={gen_rgb.mean():.6f}")
        
        # Check for Y-flip: compare corners
        h, w = ref_rgb.shape[:2]
        print(f"\n  Pixel comparison (checking for Y-flip):")
        print(f"    Top-left (0,0):     ref={ref_rgb[0,0]}, gen={gen_rgb[0,0]}")
        print(f"    Top-right (0,{w-1}):   ref={ref_rgb[0,-1]}, gen={gen_rgb[0,-1]}")
        print(f"    Bottom-left ({h-1},0): ref={ref_rgb[-1,0]}, gen={gen_rgb[-1,0]}")
        print(f"    Bottom-right ({h-1},{w-1}): ref={ref_rgb[-1,-1]}, gen={gen_rgb[-1,-1]}")
        
        # Try flipped comparison
        gen_flipped = np.flipud(gen_rgb)
        diff_normal = np.abs(ref_rgb - gen_rgb).mean()
        diff_flipped = np.abs(ref_rgb - gen_flipped).mean()
        print(f"\n  Y-flip test:")
        print(f"    Normal diff mean: {diff_normal:.6f}")
        print(f"    Flipped diff mean: {diff_flipped:.6f}")
        print(f"    Better match: {'FLIPPED' if diff_flipped < diff_normal else 'NORMAL'}")
        
        if ref_rgb.shape == gen_rgb.shape:
            diff = np.abs(ref_rgb - gen_rgb)
            print(f"\n  RGB Absolute Difference:")
            print(f"    Min: {diff.min():.6f}, Max: {diff.max():.6f}, Mean: {diff.mean():.6f}")
