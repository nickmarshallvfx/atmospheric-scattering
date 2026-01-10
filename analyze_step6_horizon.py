"""Detailed analysis of Step 6 transmittance around horizon line."""
import OpenImageIO as oiio
import numpy as np

path = r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-4\tests\step6.1_brunetonTransmittance_test_v001_04.exr"

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

# Find horizon line by scanning for transition
print("\n--- Vertical scan at center X (looking for horizon) ---")
cx = w // 2
for y in range(0, h, 10):
    val_r, val_g, val_b = r[y, cx], g[y, cx], b[y, cx]
    # Check if this is geometry (non-zero) vs sky (zero)
    is_geom = val_r > 0.01 or val_g > 0.01 or val_b > 0.01
    marker = "<<< GEOMETRY" if is_geom else ""
    print(f"  y={y:3d}: R={val_r:.4f}, G={val_g:.4f}, B={val_b:.4f} {marker}")

# Find the exact transition zone
print("\n--- Finding transition zone (first geometry from top) ---")
first_geom_y = None
for y in range(h):
    if r[y, cx] > 0.01 or g[y, cx] > 0.01 or b[y, cx] > 0.01:
        first_geom_y = y
        break

if first_geom_y:
    print(f"First geometry at y={first_geom_y}")
    print("\n--- Detailed scan around transition ---")
    for y in range(max(0, first_geom_y - 5), min(h, first_geom_y + 20)):
        val_r, val_g, val_b = r[y, cx], g[y, cx], b[y, cx]
        print(f"  y={y:3d}: R={val_r:.4f}, G={val_g:.4f}, B={val_b:.4f}")

# Check for color collapse (where B << R)
print("\n--- Looking for blue channel collapse ---")
geom_mask = (r > 0.01) | (g > 0.01) | (b > 0.01)
if geom_mask.any():
    geom_r = r[geom_mask]
    geom_g = g[geom_mask]
    geom_b = b[geom_mask]
    
    print(f"Geometry pixels: {geom_mask.sum()}")
    print(f"  R: min={geom_r.min():.4f}, max={geom_r.max():.4f}, mean={geom_r.mean():.4f}")
    print(f"  G: min={geom_g.min():.4f}, max={geom_g.max():.4f}, mean={geom_g.mean():.4f}")
    print(f"  B: min={geom_b.min():.4f}, max={geom_b.max():.4f}, mean={geom_b.mean():.4f}")
    
    # Find pixels where B < 0.5 * R (color collapse)
    collapse_mask = geom_mask & (b < 0.5 * r) & (r > 0.1)
    if collapse_mask.any():
        print(f"\n  Color collapse pixels (B < 0.5*R): {collapse_mask.sum()}")
        # Find where these are
        collapse_y, collapse_x = np.where(collapse_mask)
        print(f"  Y range: {collapse_y.min()} to {collapse_y.max()}")
        print(f"  Sample collapse pixels:")
        for i in range(min(10, len(collapse_y))):
            y, x = collapse_y[i], collapse_x[i]
            print(f"    ({x},{y}): R={r[y,x]:.4f}, G={g[y,x]:.4f}, B={b[y,x]:.4f}")
    else:
        print("\n  No color collapse detected (B >= 0.5*R everywhere)")
