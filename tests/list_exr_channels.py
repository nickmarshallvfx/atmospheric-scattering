"""List all channels in an EXR file across all subimages."""
import sys
import OpenImageIO as oiio

exr_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-4\tests\base_render_reference_noAtmos_v001_01.exr"

inp = oiio.ImageInput.open(exr_path)
if not inp:
    print(f"ERROR: {oiio.geterror()}")
    sys.exit(1)

print(f"File: {exr_path}")
print("="*60)

subimage = 0
all_channels = []

while True:
    spec = inp.spec(subimage, 0)
    if spec.width == 0:
        break
    
    channels = list(spec.channelnames)
    all_channels.extend(channels)
    
    print(f"\nSubimage {subimage}: {spec.width}x{spec.height}")
    print(f"  Channels ({len(channels)}):")
    for ch in channels:
        print(f"    {ch}")
    
    subimage += 1
    if not inp.seek_subimage(subimage, 0):
        break

inp.close()

print(f"\n{'='*60}")
print(f"Total: {len(all_channels)} channels across {subimage} subimage(s)")
