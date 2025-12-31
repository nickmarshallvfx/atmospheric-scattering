"""
Full pixel-by-pixel analysis of irradiance LUT: reference vs generated.
Run this in Blender's Python console after precomputation.

Irradiance texture layout:
- Width (64): mu_s dimension (sun cosine angle, -1 to 1)
- Height (16): r dimension (altitude, ground to top of atmosphere)

So:
- x=0 corresponds to mu_s=-1 (sun at nadir, night)
- x=63 corresponds to mu_s=1 (sun at zenith)
- y=0 corresponds to r=bottom_radius (ground level)
- y=15 corresponds to r=top_radius (top of atmosphere)
"""

import numpy as np
import os

# Paths
ref_dir = r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-2-export\lut_export"
gen_dir = r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-3\helios_cache\luts"

ref_path = os.path.join(ref_dir, "irradiance.npy")
gen_path = os.path.join(gen_dir, "irradiance.exr")

print("=" * 80)
print("FULL IRRADIANCE PIXEL ANALYSIS")
print("=" * 80)

# Load reference
ref = np.load(ref_path)
print(f"Reference shape: {ref.shape}")  # (16, 64, 4)

# Load generated EXR
import bpy
img = bpy.data.images.load(gen_path)
w, h = img.size
pixels = np.array(img.pixels[:]).reshape(h, w, 4)
# Flip vertically (EXR convention)
gen = np.flipud(pixels)
bpy.data.images.remove(img)
print(f"Generated shape: {gen.shape}")

# Extract RGB only
ref_rgb = ref[:, :, :3]
gen_rgb = gen[:, :, :3]

print(f"\nReference RGB: min={ref_rgb.min():.6f}, max={ref_rgb.max():.6f}, mean={ref_rgb.mean():.6f}")
print(f"Generated RGB: min={gen_rgb.min():.6f}, max={gen_rgb.max():.6f}, mean={gen_rgb.mean():.6f}")

# Analyze by row (each row = different altitude r)
print("\n" + "=" * 80)
print("ROW-BY-ROW ANALYSIS (each row = different altitude)")
print("y=0: ground level, y=15: top of atmosphere")
print("=" * 80)

for y in range(16):
    ref_row = ref_rgb[y, :, :]
    gen_row = gen_rgb[y, :, :]
    
    ref_max = ref_row.max()
    gen_max = gen_row.max()
    ref_mean = ref_row.mean()
    gen_mean = gen_row.mean()
    
    ratio_max = gen_max / ref_max if ref_max > 0 else float('inf')
    ratio_mean = gen_mean / ref_mean if ref_mean > 0 else float('inf')
    
    print(f"Row {y:2d}: ref_max={ref_max:.6f}, gen_max={gen_max:.6f} (ratio={ratio_max:.2f}) | "
          f"ref_mean={ref_mean:.6f}, gen_mean={gen_mean:.6f} (ratio={ratio_mean:.2f})")

# Analyze by column (each column = different sun angle mu_s)
print("\n" + "=" * 80)
print("COLUMN-BY-COLUMN ANALYSIS (each column = different sun angle)")
print("x=0: mu_s=-1 (night), x=32: mu_s~0 (horizon), x=63: mu_s=1 (zenith)")
print("=" * 80)

# Sample a few key columns
key_cols = [0, 8, 16, 24, 32, 40, 48, 56, 63]
for x in key_cols:
    ref_col = ref_rgb[:, x, :]
    gen_col = gen_rgb[:, x, :]
    
    # Compute mu_s for this column
    # x_mu_s = GetUnitRangeFromTextureCoord(x/64, 64) = ((x+0.5)/64 - 0.5/64) / (1 - 1/64)
    # Simplified: x_mu_s ≈ x / 63 for the unit range
    x_mu_s = x / 63.0
    mu_s = 2.0 * x_mu_s - 1.0
    
    ref_max = ref_col.max()
    gen_max = gen_col.max()
    ref_sum = ref_col.sum()
    gen_sum = gen_col.sum()
    
    print(f"Col {x:2d} (mu_s={mu_s:+.3f}): ref_max={ref_max:.6f}, gen_max={gen_max:.6f} | "
          f"ref_sum={ref_sum:.6f}, gen_sum={gen_sum:.6f}")

# Detailed pixel dump for row 0 (ground level - should have highest irradiance)
print("\n" + "=" * 80)
print("ROW 0 DETAILED (ground level, all 64 mu_s values)")
print("Format: x, mu_s, ref_B, gen_B, diff, ratio")
print("=" * 80)

for x in range(64):
    x_mu_s = x / 63.0
    mu_s = 2.0 * x_mu_s - 1.0
    
    ref_b = ref_rgb[0, x, 2]  # Blue channel (highest values typically)
    gen_b = gen_rgb[0, x, 2]
    
    diff = gen_b - ref_b
    ratio = gen_b / ref_b if ref_b > 1e-6 else float('inf') if gen_b > 1e-6 else 1.0
    
    if x % 4 == 0 or ref_b > 0.01 or gen_b > 0.01:  # Print every 4th or significant values
        print(f"x={x:2d} mu_s={mu_s:+.3f}: ref={ref_b:.6f}, gen={gen_b:.6f}, diff={diff:+.6f}, ratio={ratio:.2f}")

# Find where reference has zero but we have non-zero
print("\n" + "=" * 80)
print("PIXELS WHERE REF=0 BUT GEN>0 (spurious energy)")
print("=" * 80)

threshold = 1e-6
count = 0
for y in range(16):
    for x in range(64):
        ref_max_ch = ref_rgb[y, x, :].max()
        gen_max_ch = gen_rgb[y, x, :].max()
        
        if ref_max_ch < threshold and gen_max_ch > threshold:
            count += 1
            if count <= 20:  # First 20 only
                x_mu_s = x / 63.0
                mu_s = 2.0 * x_mu_s - 1.0
                print(f"  ({y:2d}, {x:2d}) mu_s={mu_s:+.3f}: ref={ref_rgb[y,x,:]}, gen={gen_rgb[y,x,:]}")

print(f"Total spurious pixels: {count} out of {16*64} ({100*count/(16*64):.1f}%)")

# Find where reference has high values but we're much lower
print("\n" + "=" * 80)
print("PIXELS WHERE GEN << REF (missing energy)")
print("=" * 80)

missing_threshold = 0.5  # gen < 50% of ref
count = 0
for y in range(16):
    for x in range(64):
        ref_max_ch = ref_rgb[y, x, :].max()
        gen_max_ch = gen_rgb[y, x, :].max()
        
        if ref_max_ch > 0.01 and gen_max_ch < ref_max_ch * missing_threshold:
            count += 1
            if count <= 20:
                x_mu_s = x / 63.0
                mu_s = 2.0 * x_mu_s - 1.0
                ratio = gen_max_ch / ref_max_ch
                print(f"  ({y:2d}, {x:2d}) mu_s={mu_s:+.3f}: ref_max={ref_max_ch:.4f}, gen_max={gen_max_ch:.4f}, ratio={ratio:.2f}")

print(f"Total missing energy pixels: {count}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
