import OpenImageIO as oiio
import numpy as np

filepath = r'C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-4\tests\step9.1_combinedNodeRebuild_test_v001_10.exr'
inp = oiio.ImageInput.open(filepath)

# List all subimages
print("Scanning subimages...")
subimg = 0
target_subimg = None
while True:
    spec = inp.spec()
    channels = [spec.channelnames[i] for i in range(spec.nchannels)]
    print(f"Subimage {subimg}: {channels[:4]}...")
    if any('ViewLayer.Combined' in c for c in channels):
        target_subimg = subimg
        print(f"  ^ Found ViewLayer.Combined!")
    if not inp.seek_subimage(subimg + 1, 0):
        break
    subimg += 1

# Read the ViewLayer.Combined subimage
if target_subimg is not None:
    inp.seek_subimage(target_subimg, 0)
    spec = inp.spec()
    pixels = inp.read_image('float')
    arr = np.array(pixels).reshape(spec.height, spec.width, spec.nchannels)
    
    # Find R channel index
    r_idx = None
    for i, name in enumerate([spec.channelnames[j] for j in range(spec.nchannels)]):
        if 'ViewLayer.Combined.R' in name:
            r_idx = i
            break
    
    if r_idx is not None:
        print(f'\nAnalyzing ViewLayer.Combined.R (index {r_idx}):')
        r_channel = arr[:,:,r_idx]
        print(f'  Range: {r_channel.min():.6f} to {r_channel.max():.6f}')
        print(f'  Center: {r_channel[spec.height//2, spec.width//2]:.6f}')
        
        # Sample various positions
        print(f'\nSamples at different y positions (x=center):')
        for y in [50, 150, 250, 350, 450]:
            if y < spec.height:
                print(f'  y={y}: {r_channel[y, spec.width//2]:.6f}')
        
        # Check discontinuities
        print(f'\nDiscontinuity check (row y=450):')
        if 450 < spec.height:
            row = r_channel[450, :]
            diffs = np.abs(np.diff(row))
            if len(diffs) > 0:
                max_idx = np.argmax(diffs)
                print(f'  Max diff: {diffs.max():.6f} at x={max_idx}')
        
        # For mu_p: convert back from [0,1] to [-1,1]
        print(f'\nConverting to mu_p (from remapped [0,1] to [-1,1]):')
        print(f'  Min mu_p: {r_channel.min() * 2 - 1:.4f}')
        print(f'  Max mu_p: {r_channel.max() * 2 - 1:.4f}')
        print(f'  Center mu_p: {(r_channel[spec.height//2, spec.width//2] * 2 - 1):.4f}')
        
        # Check ground vs sky
        print(f'\nGround (bottom row) mu_p:')
        if spec.height > 10:
            ground_vals = r_channel[-10, :] * 2 - 1
            print(f'  Range: {ground_vals.min():.4f} to {ground_vals.max():.4f}')
            print(f'  Mean: {ground_vals.mean():.4f}')
else:
    print("ViewLayer.Combined not found!")
