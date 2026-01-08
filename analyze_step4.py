import OpenImageIO as oiio
import numpy as np

path = r'C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-4\tests\step5.1_fullScatteringImplementation_test_v001_01.exr'

inp = oiio.ImageInput.open(path)
if not inp:
    print(f"ERROR: {oiio.geterror()}")
    exit(1)

print(f"File: {path}")
print("=" * 60)

# Find ViewLayer.Combined channels
subimage = 0
target_channels = {}

while True:
    spec = inp.spec(subimage, 0)
    if spec.width == 0:
        break
    
    channels = list(spec.channelnames)
    print(f"\nSubimage {subimage}: {spec.width}x{spec.height}")
    print(f"  Channels: {channels}")
    
    # Check for ViewLayer.Combined
    for ch in channels:
        if 'ViewLayer.Combined' in ch or 'View Layer.Combined' in ch:
            target_channels[ch] = subimage
    
    subimage += 1
    if not inp.seek_subimage(subimage, 0):
        break

inp.close()

# Read ViewLayer.Combined if found
if target_channels:
    print(f"\n=== Reading ViewLayer.Combined ===")
    inp = oiio.ImageInput.open(path)
    
    # Find which subimage has ViewLayer
    for ch, sub in target_channels.items():
        inp.seek_subimage(sub, 0)
        break
    
    spec = inp.spec()
    w, h = spec.width, spec.height
    pixels = np.array(inp.read_image(format='float')).reshape(h, w, spec.nchannels)
    channels = list(spec.channelnames)
    
    print(f"Resolution: {w}x{h}, Channels: {spec.nchannels}")
    print(f"Channel names: {channels}")
    
    # Find RGB indices
    r_idx = next((i for i, c in enumerate(channels) if c.endswith('.R')), 0)
    g_idx = next((i for i, c in enumerate(channels) if c.endswith('.G')), 1)
    b_idx = next((i for i, c in enumerate(channels) if c.endswith('.B')), 2)
    
    print(f"\nUsing indices: R={r_idx}, G={g_idx}, B={b_idx}")
    
    r = pixels[:, :, r_idx]
    g = pixels[:, :, g_idx]
    b = pixels[:, :, b_idx]
    
    print(f"\nPixel value ranges:")
    print(f"  R: min={r.min():.4f}, max={r.max():.4f}, mean={r.mean():.4f}")
    print(f"  G: min={g.min():.4f}, max={g.max():.4f}, mean={g.mean():.4f}")
    print(f"  B: min={b.min():.4f}, max={b.max():.4f}, mean={b.mean():.4f}")
    
    cy, cx = h // 2, w // 2
    print(f"\nCenter pixel ({cx},{cy}): R={r[cy,cx]:.4f}, G={g[cy,cx]:.4f}, B={b[cy,cx]:.4f}")
    
    print("\nSample at various Y positions (center X):")
    for y in [50, h//4, h//2, 3*h//4, h-50]:
        if y < h:
            print(f"  y={y}: R={r[y,cx]:.4f}, G={g[y,cx]:.4f}, B={b[y,cx]:.4f}")
    
    inp.close()
else:
    print("\nNo ViewLayer.Combined channels found")
