"""Analyze Step 2.4 inscatter render for discontinuity."""
import OpenImageIO as oiio
import numpy as np

# Load render
inp = oiio.ImageInput.open(r'C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-4\tests\step2.4_fullInscatter_test_v001_01.exr')
inp.seek_subimage(1, 0)  # View layer
spec = inp.spec()
pixels = inp.read_image(format=oiio.FLOAT).reshape(spec.height, spec.width, spec.nchannels)
inp.close()

r, g, b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
luminance = 0.2126*r + 0.7152*g + 0.0722*b

print(f"Image: {spec.width}x{spec.height}")
print(f"\nOverall stats:")
valid = luminance > 0.0001
print(f"  R: {r[valid].mean():.6f}, G: {g[valid].mean():.6f}, B: {b[valid].mean():.6f}")
print(f"  B/R ratio: {b[valid].mean()/r[valid].mean():.2f}")

print(f"\nRow-by-row analysis (looking for discontinuity):")
for row in range(0, spec.height, 50):
    row_lum = luminance[row, :]
    row_valid = row_lum > 0.0001
    if row_valid.any():
        mean_lum = row_lum[row_valid].mean()
        mean_r = r[row, row_valid].mean()
        mean_b = b[row, row_valid].mean()
        br_ratio = mean_b / mean_r if mean_r > 0 else 0
        print(f"  Row {row:3d}: lum={mean_lum:.4f}, R={mean_r:.4f}, B={mean_b:.4f}, B/R={br_ratio:.2f}")

print(f"\nLooking for sharp luminance changes between adjacent rows:")
prev_lum = 0
for row in range(spec.height):
    row_lum = luminance[row, :]
    row_valid = row_lum > 0.0001
    if row_valid.any():
        mean_lum = row_lum[row_valid].mean()
        if prev_lum > 0 and abs(mean_lum - prev_lum) / prev_lum > 0.3:
            print(f"  DISCONTINUITY at row {row}: {prev_lum:.4f} -> {mean_lum:.4f} ({(mean_lum/prev_lum - 1)*100:+.1f}%)")
        prev_lum = mean_lum
