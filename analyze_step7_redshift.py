"""Analyze Step 7 output for red shift pattern."""
import OpenImageIO as oiio
import numpy as np

path = r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-4\tests\step7.1_combinedFunctions_test_v001_03.exr"

# Also check Composite.Combined (subimage 0) which might show different values
inp = oiio.ImageInput.open(path)
spec = inp.spec()
print(f"Subimage 0: {spec.width}x{spec.height}, channels: {list(spec.channelnames)}")
comp_pixels = np.array(inp.read_image(format='float')).reshape(spec.height, spec.width, spec.nchannels)
inp.close()

comp_r = comp_pixels[:, :, 0]
comp_g = comp_pixels[:, :, 1]
comp_b = comp_pixels[:, :, 2]

print("\n=== Composite.Combined (subimage 0) ===")
print(f"  R: min={comp_r.min():.4f}, max={comp_r.max():.4f}, mean={comp_r.mean():.4f}")
print(f"  G: min={comp_g.min():.4f}, max={comp_g.max():.4f}, mean={comp_g.mean():.4f}")
print(f"  B: min={comp_b.min():.4f}, max={comp_b.max():.4f}, mean={comp_b.mean():.4f}")

# Check for red-shifted pixels in composite
comp_geom = (comp_r > 0.01) | (comp_g > 0.01) | (comp_b > 0.01)
if comp_geom.any():
    print(f"\nGeometry pixels: {comp_geom.sum()}")
    print(f"  Mean: R={comp_r[comp_geom].mean():.4f}, G={comp_g[comp_geom].mean():.4f}, B={comp_b[comp_geom].mean():.4f}")
    
    # Find max red pixels
    red_dom = comp_geom & (comp_r > comp_g) & (comp_r > comp_b)
    if red_dom.any():
        print(f"\nRed-dominant pixels: {red_dom.sum()}")
        max_r_idx = np.unravel_index(np.argmax(comp_r), comp_r.shape)
        print(f"  Max R at ({max_r_idx[1]},{max_r_idx[0]}): R={comp_r[max_r_idx]:.4f}, G={comp_g[max_r_idx]:.4f}, B={comp_b[max_r_idx]:.4f}")

print("\n" + "=" * 70)

inp = oiio.ImageInput.open(path)
inp.seek_subimage(1, 0)  # ViewLayer.Combined
spec = inp.spec()
w, h = spec.width, spec.height
pixels = np.array(inp.read_image(format='float')).reshape(h, w, spec.nchannels)
inp.close()

r = pixels[:, :, 0]
g = pixels[:, :, 1]
b = pixels[:, :, 2]

print(f"Image: {w}x{h}")
print("=" * 70)

# Check horizontal scan at multiple Y levels
print("\n--- Horizontal scan at multiple Y levels ---")
for cy in [h//4, h//2, 3*h//4]:
    print(f"\n  Y={cy}:")
    for x in range(0, w, 120):
        val_r, val_g, val_b = r[cy, x], g[cy, x], b[cy, x]
        total = val_r + val_g + val_b
        if total > 0.001:  # Only show non-black pixels
            ratio_rg = val_r / max(val_g, 0.0001)
            print(f"    x={x:3d}: R={val_r:.4f}, G={val_g:.4f}, B={val_b:.4f}  R/G={ratio_rg:.2f}")

# Check left vs right halves
print("\n--- Left vs Right comparison ---")
left_mask = np.zeros((h, w), dtype=bool)
left_mask[:, :w//2] = True
right_mask = np.zeros((h, w), dtype=bool)
right_mask[:, w//2:] = True
geom_mask = (r > 0.001) | (g > 0.001) | (b > 0.001)

left_geom = geom_mask & left_mask
right_geom = geom_mask & right_mask

if left_geom.any():
    print(f"Left half ({left_geom.sum()} pixels):")
    print(f"  R: mean={r[left_geom].mean():.4f}, G: mean={g[left_geom].mean():.4f}, B: mean={b[left_geom].mean():.4f}")
    print(f"  R/G ratio: {r[left_geom].mean() / max(g[left_geom].mean(), 0.0001):.3f}")

if right_geom.any():
    print(f"Right half ({right_geom.sum()} pixels):")
    print(f"  R: mean={r[right_geom].mean():.4f}, G: mean={g[right_geom].mean():.4f}, B: mean={b[right_geom].mean():.4f}")
    print(f"  R/G ratio: {r[right_geom].mean() / max(g[right_geom].mean(), 0.0001):.3f}")

# Find pixels with strong red shift (R > 2*G)
print("\n--- Pixels with strong red shift (R > 2*G) ---")
red_shift_mask = geom_mask & (r > 2 * g) & (r > 0.01)
if red_shift_mask.any():
    print(f"  Count: {red_shift_mask.sum()} pixels")
    ys, xs = np.where(red_shift_mask)
    print(f"  X range: {xs.min()} to {xs.max()}")
    print(f"  Y range: {ys.min()} to {ys.max()}")
    print(f"  Sample pixels:")
    for i in range(min(10, len(ys))):
        y, x = ys[i], xs[i]
        print(f"    ({x},{y}): R={r[y,x]:.4f}, G={g[y,x]:.4f}, B={b[y,x]:.4f}")
else:
    print("  No strong red shift detected")

# Check if this correlates with nu (view-sun angle)
print("\n--- Analysis: Is red shift correlated with view direction? ---")
print("  Sun direction from Step 7: (0.557, 0.663, 0.500)")
print("  Left side of image = looking away from sun (nu < 0)")
print("  Right side of image = looking toward sun (nu > 0)")
print("  Mie phase function peaks when looking toward sun")
print("  If red shift is on LEFT (away from sun), it's NOT Mie phase")
