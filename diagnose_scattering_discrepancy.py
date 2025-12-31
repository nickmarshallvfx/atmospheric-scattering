"""
Diagnostic script to identify the exact source of scattering/irradiance discrepancy.

This script performs systematic comparisons between our generated LUTs and the reference
to pinpoint where values diverge.

Run in Blender Python console - see IRRADIANCE_DEBUG_ANALYSIS.md for instructions.
"""

import numpy as np
import os

# Paths - Use original Bruneton reference (.npy files)
REF_DIR = r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-2-export\lut_export"
REF_IS_NPY = True  # Reference uses .npy format
GEN_DIR = r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-3\helios_cache\luts"

# Atmospheric parameters (must match what we use)
BOTTOM_RADIUS = 6360000.0
TOP_RADIUS = 6420000.0
H = np.sqrt(TOP_RADIUS**2 - BOTTOM_RADIUS**2)  # ~79982 m

# Texture sizes
SCATTERING_TEXTURE_R_SIZE = 32
SCATTERING_TEXTURE_MU_SIZE = 128
SCATTERING_TEXTURE_MU_S_SIZE = 32
SCATTERING_TEXTURE_NU_SIZE = 8
SCATTERING_TEXTURE_WIDTH = 256  # NU_SIZE * MU_S_SIZE
SCATTERING_TEXTURE_HEIGHT = 128  # MU_SIZE

IRRADIANCE_TEXTURE_WIDTH = 64
IRRADIANCE_TEXTURE_HEIGHT = 16


def load_reference_scattering():
    """Load reference scattering (NPY or EXR)."""
    if REF_IS_NPY:
        path = os.path.join(REF_DIR, "scattering.npy")
        if os.path.exists(path):
            return np.load(path)
        print(f"Reference scattering not found at {path}")
        return None
    else:
        import bpy
        path = os.path.join(REF_DIR, "scattering.exr")
        if not os.path.exists(path):
            print(f"Reference scattering not found at {path}")
            return None
        img = bpy.data.images.load(path)
        pixels = np.array(img.pixels[:]).reshape(img.size[1], img.size[0], img.channels)
        pixels = np.flipud(pixels)
        bpy.data.images.remove(img)
        return pixels


def load_generated_scattering():
    """Load our generated scattering EXR."""
    import bpy
    path = os.path.join(GEN_DIR, "scattering.exr")
    if not os.path.exists(path):
        print(f"Generated scattering not found at {path}")
        return None
    
    # Load via Blender with proper colorspace
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = 'Non-Color'  # Prevent colorspace conversion
    pixels = np.array(img.pixels[:]).reshape(img.size[1], img.size[0], img.channels)
    
    # Debug: check for inf/nan immediately after load
    inf_count = np.sum(np.isinf(pixels))
    nan_count = np.sum(np.isnan(pixels))
    if inf_count > 0 or nan_count > 0:
        print(f"  [DEBUG] Loaded EXR has {inf_count} inf and {nan_count} nan values")
        # Find where the inf values are
        inf_mask = np.isinf(pixels)
        if inf_count > 0:
            inf_locs = np.where(inf_mask)
            print(f"  [DEBUG] First inf at y={inf_locs[0][0]}, x={inf_locs[1][0]}, c={inf_locs[2][0]}")
            # Check finite max
            finite_pixels = pixels[~np.isinf(pixels) & ~np.isnan(pixels)]
            print(f"  [DEBUG] Finite pixels: min={finite_pixels.min():.6f}, max={finite_pixels.max():.6f}")
    
    pixels = np.flipud(pixels)  # Blender stores bottom-to-top
    bpy.data.images.remove(img)
    return pixels


def analyze_scattering_structure():
    """Analyze the structure and content of both scattering textures."""
    print("=" * 80)
    print("SCATTERING TEXTURE STRUCTURE ANALYSIS")
    print("=" * 80)
    
    ref = load_reference_scattering()
    gen = load_generated_scattering()
    
    if ref is None or gen is None:
        return
    
    print(f"\nReference shape: {ref.shape}")
    print(f"Generated shape: {gen.shape}")
    
    # Both should be tiled 2D: (128, 8192, 4)
    if ref.shape != gen.shape:
        print("WARNING: Shapes don't match!")
    
    print(f"\nReference: min={ref.min():.6f}, max={ref.max():.6f}, mean={ref.mean():.6f}")
    print(f"Generated: min={gen.min():.6f}, max={gen.max():.6f}, mean={gen.mean():.6f}")
    print(f"Ratio (gen/ref) of max: {gen.max() / ref.max():.4f}")
    
    # Analyze by R-layer (32 layers, each 256 wide)
    print("\n" + "-" * 60)
    print("ANALYSIS BY R-LAYER (altitude)")
    print("-" * 60)
    
    layer_width = SCATTERING_TEXTURE_WIDTH  # 256
    
    for layer in [0, 1, 8, 16, 24, 31]:  # Sample layers
        x_start = layer * layer_width
        x_end = (layer + 1) * layer_width
        
        ref_layer = ref[:, x_start:x_end, :3]
        gen_layer = gen[:, x_start:x_end, :3]
        
        ref_max = ref_layer.max()
        gen_max = gen_layer.max()
        ratio = gen_max / ref_max if ref_max > 1e-10 else float('inf')
        
        # Compute r for this layer
        x_r = (layer + 0.5) / SCATTERING_TEXTURE_R_SIZE
        rho = H * x_r
        r = np.sqrt(rho**2 + BOTTOM_RADIUS**2)
        altitude_km = (r - BOTTOM_RADIUS) / 1000
        
        print(f"Layer {layer:2d} (alt={altitude_km:6.1f}km): ref_max={ref_max:.6f}, gen_max={gen_max:.6f}, ratio={ratio:.4f}")
    
    # Analyze specific pixel locations
    print("\n" + "-" * 60)
    print("SPECIFIC PIXEL COMPARISON")
    print("-" * 60)
    
    # Test points: (layer, y, x_within_layer) -> (r, mu, mu_s/nu combined)
    test_points = [
        (0, 64, 128),   # Ground, mu=0 (horizon), middle of mu_s/nu range
        (0, 96, 128),   # Ground, mu>0 (upward), middle
        (0, 127, 128),  # Ground, mu=1 (zenith), middle
        (16, 64, 128),  # Mid-atmosphere
        (31, 64, 128),  # Top of atmosphere
        (0, 96, 0),     # Ground, upward, nu=-1, mu_s=low
        (0, 96, 255),   # Ground, upward, nu=1, mu_s=high
    ]
    
    for layer, y, x_local in test_points:
        x_global = layer * layer_width + x_local
        
        if x_global >= ref.shape[1] or y >= ref.shape[0]:
            continue
            
        ref_val = ref[y, x_global, :3]
        gen_val = gen[y, x_global, :3]
        
        # Decode nu_idx and mu_s_idx from x_local
        nu_idx = x_local // SCATTERING_TEXTURE_MU_S_SIZE
        mu_s_idx = x_local % SCATTERING_TEXTURE_MU_S_SIZE
        
        print(f"  Layer={layer}, y={y}, nu_idx={nu_idx}, mu_s_idx={mu_s_idx}:")
        print(f"    ref=[{ref_val[0]:.6f}, {ref_val[1]:.6f}, {ref_val[2]:.6f}]")
        print(f"    gen=[{gen_val[0]:.6f}, {gen_val[1]:.6f}, {gen_val[2]:.6f}]")
        if ref_val.max() > 1e-10:
            ratios = gen_val / np.maximum(ref_val, 1e-10)
            print(f"    ratio=[{ratios[0]:.4f}, {ratios[1]:.4f}, {ratios[2]:.4f}]")


def analyze_single_mie_scattering():
    """Check if we have separate Mie scattering and what it contains."""
    print("\n" + "=" * 80)
    print("SINGLE MIE SCATTERING ANALYSIS")
    print("=" * 80)
    
    import bpy
    
    # Check if reference has single_mie
    if REF_IS_NPY:
        # Bruneton reference doesn't have separate single_mie - it's combined in scattering
        print("Bruneton reference doesn't export separate single_mie_scattering")
        ref_mie = None
    else:
        ref_mie_path = os.path.join(REF_DIR, "single_mie_scattering.exr")
        if os.path.exists(ref_mie_path):
            img = bpy.data.images.load(ref_mie_path)
            ref_mie = np.array(img.pixels[:]).reshape(img.size[1], img.size[0], img.channels)
            ref_mie = np.flipud(ref_mie)
            bpy.data.images.remove(img)
            print(f"Reference single_mie shape: {ref_mie.shape}")
            print(f"Reference single_mie: min={ref_mie.min():.6f}, max={ref_mie.max():.6f}")
        else:
            print("No reference single_mie_scattering.exr found")
            ref_mie = None
    
    # Our single_mie
    gen_mie_path = os.path.join(GEN_DIR, "single_mie_scattering.exr")
    if os.path.exists(gen_mie_path):
        img = bpy.data.images.load(gen_mie_path)
        gen_mie = np.array(img.pixels[:]).reshape(img.size[1], img.size[0], img.channels)
        gen_mie = np.flipud(gen_mie)
        bpy.data.images.remove(img)
        print(f"Generated single_mie shape: {gen_mie.shape}")
        print(f"Generated single_mie: min={gen_mie.min():.6f}, max={gen_mie.max():.6f}")
    else:
        print("No generated single_mie_scattering.exr found")
        gen_mie = None
    
    if ref_mie is not None and gen_mie is not None:
        ratio = gen_mie.max() / ref_mie.max() if ref_mie.max() > 1e-10 else float('inf')
        print(f"Mie max ratio (gen/ref): {ratio:.4f}")


def analyze_what_reference_scattering_contains():
    """
    Determine if reference scattering.npy contains:
    - Rayleigh only
    - Mie only  
    - Combined (Rayleigh + Mie * factor)
    """
    print("\n" + "=" * 80)
    print("REFERENCE SCATTERING CONTENT ANALYSIS")
    print("=" * 80)
    
    ref_scatter = load_reference_scattering()
    
    if REF_IS_NPY:
        print("Bruneton reference combines Rayleigh+Mie in scattering texture")
        print("(No separate Mie export available for analysis)")
        if ref_scatter is not None:
            print(f"Reference scattering max: {ref_scatter.max():.6f}")
        return
    
    import bpy
    ref_mie_path = os.path.join(REF_DIR, "single_mie_scattering.exr")
    if not os.path.exists(ref_mie_path):
        print("Cannot determine - no separate Mie reference")
        return
    
    img = bpy.data.images.load(ref_mie_path)
    ref_mie = np.array(img.pixels[:]).reshape(img.size[1], img.size[0], img.channels)
    ref_mie = np.flipud(ref_mie)
    bpy.data.images.remove(img)
    
    print(f"Reference scattering max: {ref_scatter.max():.6f}")
    print(f"Reference single_mie max: {ref_mie.max():.6f}")
    
    # If scattering = rayleigh_only, then scattering should be less than scattering + mie
    # If scattering = combined, then scattering ≈ rayleigh + mie * factor
    
    # Pick a point with significant values
    layer = 0
    y = 96
    x_local = 128
    x_global = layer * 256 + x_local
    
    ref_s = ref_scatter[y, x_global, :3]
    ref_m = ref_mie[y, x_global, :3]
    
    print(f"\nAt test point (layer=0, y=96, x=128):")
    print(f"  scattering = [{ref_s[0]:.6f}, {ref_s[1]:.6f}, {ref_s[2]:.6f}]")
    print(f"  single_mie = [{ref_m[0]:.6f}, {ref_m[1]:.6f}, {ref_m[2]:.6f}]")
    
    # Compute what Rayleigh-only would be: scattering - mie * (mie_scat/mie_ext)
    # mie_scattering/mie_extinction ≈ 4e-6 / 4.44e-6 ≈ 0.9
    mie_factor = 4e-6 / 4.44e-6
    implied_rayleigh = ref_s - ref_m * mie_factor
    print(f"  implied_rayleigh (if combined) = [{implied_rayleigh[0]:.6f}, {implied_rayleigh[1]:.6f}, {implied_rayleigh[2]:.6f}]")
    
    # Check ratio
    if ref_m.max() > 1e-10:
        excess = ref_s - implied_rayleigh
        print(f"  mie contribution in scattering = [{excess[0]:.6f}, {excess[1]:.6f}, {excess[2]:.6f}]")
        
        if np.allclose(ref_s, implied_rayleigh + ref_m * mie_factor, rtol=0.1):
            print("\n  CONCLUSION: Reference scattering appears to be COMBINED (Rayleigh + Mie*factor)")
        elif np.allclose(ref_s, implied_rayleigh, rtol=0.1):
            print("\n  CONCLUSION: Reference scattering appears to be RAYLEIGH ONLY")
        else:
            print("\n  CONCLUSION: Unable to determine - values don't match expected patterns")


def verify_our_mie_combination():
    """Check if our V35 Mie combination is actually being applied."""
    print("\n" + "=" * 80)
    print("VERIFYING OUR MIE COMBINATION (V35)")
    print("=" * 80)
    
    gen = load_generated_scattering()
    if gen is None:
        return
    
    import bpy
    gen_mie_path = os.path.join(GEN_DIR, "single_mie_scattering.exr")
    if not os.path.exists(gen_mie_path):
        print("No single_mie_scattering.exr found")
        return
    
    img = bpy.data.images.load(gen_mie_path)
    gen_mie = np.array(img.pixels[:]).reshape(img.size[1], img.size[0], img.channels)
    gen_mie = np.flipud(gen_mie)
    bpy.data.images.remove(img)
    
    # At a test point, check if scattering ≈ something + mie * factor
    layer = 0
    y = 96
    x_local = 128
    x_global = layer * 256 + x_local
    
    gen_s = gen[y, x_global, :3]
    gen_m = gen_mie[y, x_global, :3]
    
    mie_factor = 4e-6 / 4.44e-6
    expected_mie_contribution = gen_m * mie_factor
    
    print(f"At test point (layer=0, y=96, x=128):")
    print(f"  our scattering = [{gen_s[0]:.6f}, {gen_s[1]:.6f}, {gen_s[2]:.6f}]")
    print(f"  our single_mie = [{gen_m[0]:.6f}, {gen_m[1]:.6f}, {gen_m[2]:.6f}]")
    print(f"  mie_factor = {mie_factor:.6f}")
    print(f"  expected_mie_contrib = [{expected_mie_contribution[0]:.6f}, {expected_mie_contribution[1]:.6f}, {expected_mie_contribution[2]:.6f}]")
    
    # If V35 is working, scattering should be > mie_contribution
    # The Rayleigh component would be scattering - mie_contribution
    implied_rayleigh = gen_s - expected_mie_contribution
    print(f"  implied_rayleigh = [{implied_rayleigh[0]:.6f}, {implied_rayleigh[1]:.6f}, {implied_rayleigh[2]:.6f}]")
    
    if np.all(implied_rayleigh >= 0):
        print("\n  Mie combination appears to be applied (scattering > mie contribution)")
    else:
        print("\n  WARNING: Negative implied Rayleigh - Mie combination may have issues")


def run_full_diagnosis():
    """Run all diagnostic tests."""
    print("\n" + "=" * 80)
    print("FULL SCATTERING DISCREPANCY DIAGNOSIS")
    print("=" * 80)
    print(f"Reference dir: {REF_DIR}")
    print(f"Generated dir: {GEN_DIR}")
    
    analyze_scattering_structure()
    analyze_single_mie_scattering()
    analyze_what_reference_scattering_contains()
    verify_our_mie_combination()
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS COMPLETE")
    print("=" * 80)


# Run
run_full_diagnosis()
