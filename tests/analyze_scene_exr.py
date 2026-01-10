"""
Analyze Multi-Layer EXR to Extract Scene Geometry Reference

This script reads a multi-layer EXR with position data and extracts:
- Camera position (inferred from position data)
- Distance ranges (min, max, percentiles)
- Object depth distribution
- Scene bounds

Outputs a TEST_SCENE_REFERENCE.md file for validation context.

Usage:
    python analyze_scene_exr.py <path_to_exr>
    
Expected EXR layers:
    - Position (or P, or ViewLayer.Position) - world space XYZ
"""

import sys
import os
import numpy as np
from datetime import datetime

try:
    import OpenImageIO as oiio
except ImportError:
    print("ERROR: OpenImageIO not found. Install with: pip install OpenImageIO")
    print("Or run this script in Blender's Python environment.")
    sys.exit(1)


def find_position_layer(spec):
    """Find the position layer name in the EXR subimage."""
    position_names = [
        'ViewLayer.Position', 'View Layer.Position',
        'RenderLayer.Position',
        'Position', 'P', 
        'position', 'pos'
    ]
    
    channels = list(spec.channelnames)
    
    # Look for position channels (need .X, .Y, .Z or .R, .G, .B)
    for name in position_names:
        if f"{name}.X" in channels or f"{name}.R" in channels:
            return name
    
    # Try to find any channel with Position in name
    for ch in channels:
        if 'position' in ch.lower() or '.P.' in ch:
            base = ch.rsplit('.', 1)[0]
            return base
    
    return None


def read_position_data(exr_path):
    """Read position data from EXR file, scanning all subimages."""
    inp = oiio.ImageInput.open(exr_path)
    if not inp:
        print(f"ERROR: Could not open {exr_path}")
        print(f"OIIO Error: {oiio.geterror()}")
        return None, None, None
    
    # Scan all subimages to find position layer
    subimage = 0
    pos_layer = None
    pos_subimage = None
    all_channels = []
    
    while True:
        spec = inp.spec(subimage, 0)
        if spec.width == 0:
            break
        
        channels = list(spec.channelnames)
        all_channels.extend([(subimage, ch) for ch in channels])
        
        # Check this subimage for position layer
        layer = find_position_layer(spec)
        if layer:
            pos_layer = layer
            pos_subimage = subimage
            print(f"Found position layer '{pos_layer}' in subimage {subimage}")
            break
        
        subimage += 1
        if not inp.seek_subimage(subimage, 0):
            break
    
    if pos_layer is None:
        print("ERROR: Could not find Position layer in any subimage")
        print("Available channels:")
        for si, ch in all_channels:
            print(f"  [{si}] {ch}")
        inp.close()
        return None, None, None
    
    # Seek to the correct subimage
    inp.seek_subimage(pos_subimage, 0)
    spec = inp.spec()
    print(f"Image: {spec.width}x{spec.height}")
    print(f"Channels in subimage {pos_subimage}: {spec.nchannels}")
    
    # Read the position subimage
    pixels = inp.read_image(format=oiio.FLOAT)
    inp.close()
    
    if pixels is None:
        print("ERROR: Failed to read image data")
        return None, None, None
    
    # Reshape to (height, width, channels)
    pixels = pixels.reshape(spec.height, spec.width, spec.nchannels)
    
    # Find position channel indices within this subimage
    channels = list(spec.channelnames)
    try:
        # Try .X, .Y, .Z first
        x_idx = channels.index(f"{pos_layer}.X")
        y_idx = channels.index(f"{pos_layer}.Y")
        z_idx = channels.index(f"{pos_layer}.Z")
    except ValueError:
        try:
            # Try .R, .G, .B
            x_idx = channels.index(f"{pos_layer}.R")
            y_idx = channels.index(f"{pos_layer}.G")
            z_idx = channels.index(f"{pos_layer}.B")
        except ValueError:
            print(f"ERROR: Could not find XYZ or RGB channels for {pos_layer}")
            return None, None, None
    
    # Extract position channels
    pos_x = pixels[:, :, x_idx]
    pos_y = pixels[:, :, y_idx]
    pos_z = pixels[:, :, z_idx]
    
    return pos_x, pos_y, pos_z


def analyze_scene(pos_x, pos_y, pos_z, camera_pos=None):
    """Analyze scene geometry from position data."""
    
    # Flatten and filter out sky (typically very large or inf values)
    x_flat = pos_x.flatten()
    y_flat = pos_y.flatten()
    z_flat = pos_z.flatten()
    
    # Filter out invalid/sky pixels (position magnitude > 100km is probably sky)
    magnitude = np.sqrt(x_flat**2 + y_flat**2 + z_flat**2)
    valid_mask = (magnitude < 100000) & (magnitude > 0) & np.isfinite(magnitude)
    
    x_valid = x_flat[valid_mask]
    y_valid = y_flat[valid_mask]
    z_valid = z_flat[valid_mask]
    
    valid_count = np.sum(valid_mask)
    total_count = len(x_flat)
    sky_percent = (1 - valid_count / total_count) * 100
    
    print(f"\nValid geometry pixels: {valid_count:,} / {total_count:,} ({100-sky_percent:.1f}%)")
    print(f"Sky pixels: {sky_percent:.1f}%")
    
    if valid_count == 0:
        print("ERROR: No valid geometry found in position data")
        return None
    
    # Scene bounds
    bounds = {
        'x_min': float(np.min(x_valid)),
        'x_max': float(np.max(x_valid)),
        'y_min': float(np.min(y_valid)),
        'y_max': float(np.max(y_valid)),
        'z_min': float(np.min(z_valid)),
        'z_max': float(np.max(z_valid)),
    }
    
    # Estimate camera position (if not provided)
    # Camera is likely where rays originate - approximate from position distribution
    if camera_pos is None:
        # Use the closest points to estimate camera region
        # This is a rough heuristic - works best if there's geometry near camera
        dist_from_origin = np.sqrt(x_valid**2 + y_valid**2 + z_valid**2)
        near_mask = dist_from_origin < np.percentile(dist_from_origin, 5)
        
        if np.sum(near_mask) > 100:
            # Estimate camera as behind the nearest points
            # This is imperfect but gives a reasonable estimate
            cam_x = float(np.median(x_valid[near_mask]))
            cam_y = float(np.median(y_valid[near_mask]))
            cam_z = float(np.percentile(z_valid, 95))  # Camera usually above scene
        else:
            cam_x, cam_y, cam_z = 0, 0, float(np.percentile(z_valid, 95))
        
        camera_pos = (cam_x, cam_y, cam_z)
        print(f"Estimated camera position: ({cam_x:.1f}, {cam_y:.1f}, {cam_z:.1f})")
    
    # Calculate distances from camera
    distances = np.sqrt(
        (x_valid - camera_pos[0])**2 + 
        (y_valid - camera_pos[1])**2 + 
        (z_valid - camera_pos[2])**2
    )
    
    # Distance statistics
    dist_stats = {
        'min': float(np.min(distances)),
        'max': float(np.max(distances)),
        'mean': float(np.mean(distances)),
        'median': float(np.median(distances)),
        'p5': float(np.percentile(distances, 5)),
        'p25': float(np.percentile(distances, 25)),
        'p75': float(np.percentile(distances, 75)),
        'p95': float(np.percentile(distances, 95)),
    }
    
    # Height (Z) statistics
    height_stats = {
        'min': float(np.min(z_valid)),
        'max': float(np.max(z_valid)),
        'mean': float(np.mean(z_valid)),
        'ground_level': float(np.percentile(z_valid, 5)),  # Approximate ground
    }
    
    return {
        'bounds': bounds,
        'camera_pos': camera_pos,
        'camera_altitude': camera_pos[2] - height_stats['ground_level'],
        'dist_stats': dist_stats,
        'height_stats': height_stats,
        'sky_percent': sky_percent,
        'valid_pixels': valid_count,
        'total_pixels': total_count,
        'image_width': pos_x.shape[1],
        'image_height': pos_x.shape[0],
    }


def generate_reference_doc(analysis, exr_path, output_path):
    """Generate markdown reference document."""
    
    doc = f"""# Test Scene Geometry Reference

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Source EXR**: `{os.path.basename(exr_path)}`
**Resolution**: {analysis['image_width']}x{analysis['image_height']}

---

## Camera

| Parameter | Value |
|-----------|-------|
| Position (estimated) | ({analysis['camera_pos'][0]:.1f}, {analysis['camera_pos'][1]:.1f}, {analysis['camera_pos'][2]:.1f}) meters |
| Altitude above ground | {analysis['camera_altitude']:.1f} meters |

---

## Scene Bounds

| Axis | Min | Max | Range |
|------|-----|-----|-------|
| X | {analysis['bounds']['x_min']:.1f}m | {analysis['bounds']['x_max']:.1f}m | {analysis['bounds']['x_max'] - analysis['bounds']['x_min']:.1f}m |
| Y | {analysis['bounds']['y_min']:.1f}m | {analysis['bounds']['y_max']:.1f}m | {analysis['bounds']['y_max'] - analysis['bounds']['y_min']:.1f}m |
| Z | {analysis['bounds']['z_min']:.1f}m | {analysis['bounds']['z_max']:.1f}m | {analysis['bounds']['z_max'] - analysis['bounds']['z_min']:.1f}m |

---

## Distance from Camera

| Statistic | Value | Value (km) |
|-----------|-------|------------|
| Minimum | {analysis['dist_stats']['min']:.1f}m | {analysis['dist_stats']['min']/1000:.3f}km |
| 5th percentile | {analysis['dist_stats']['p5']:.1f}m | {analysis['dist_stats']['p5']/1000:.3f}km |
| 25th percentile | {analysis['dist_stats']['p25']:.1f}m | {analysis['dist_stats']['p25']/1000:.3f}km |
| Median | {analysis['dist_stats']['median']:.1f}m | {analysis['dist_stats']['median']/1000:.3f}km |
| Mean | {analysis['dist_stats']['mean']:.1f}m | {analysis['dist_stats']['mean']/1000:.3f}km |
| 75th percentile | {analysis['dist_stats']['p75']:.1f}m | {analysis['dist_stats']['p75']/1000:.3f}km |
| 95th percentile | {analysis['dist_stats']['p95']:.1f}m | {analysis['dist_stats']['p95']/1000:.3f}km |
| Maximum | {analysis['dist_stats']['max']:.1f}m | {analysis['dist_stats']['max']/1000:.3f}km |

---

## Coverage

| Type | Pixels | Percentage |
|------|--------|------------|
| Geometry | {analysis['valid_pixels']:,} | {100 - analysis['sky_percent']:.1f}% |
| Sky | {analysis['total_pixels'] - analysis['valid_pixels']:,} | {analysis['sky_percent']:.1f}% |

---

## Expected Aerial Perspective Behavior

Based on distance statistics:

### Near Objects (< {analysis['dist_stats']['p25']:.0f}m)
- **Transmittance**: ~0.99+ (almost no absorption)
- **Inscatter**: Minimal
- **Expected appearance**: Nearly original color

### Mid-range Objects ({analysis['dist_stats']['p25']:.0f}m - {analysis['dist_stats']['p75']:.0f}m)
- **Transmittance**: ~0.95-0.99
- **Inscatter**: Moderate
- **Expected appearance**: Slight haze, reduced contrast

### Far Objects (> {analysis['dist_stats']['p75']:.0f}m)
- **Transmittance**: ~0.85-0.95
- **Inscatter**: Significant
- **Expected appearance**: Visible haze, blending toward sky color

### Very Far Objects (> {analysis['dist_stats']['p95']:.0f}m)
- **Transmittance**: < 0.85
- **Inscatter**: Heavy
- **Expected appearance**: Strong haze, significantly blended with sky

---

## Validation Reference

When validating Step 1.1 (Distance), output `distance / {analysis['dist_stats']['max']/1000:.1f}km`:
- Near geometry should appear **dark** (low distance value)
- Far geometry should appear **bright** (high distance value)
- Sky pixels should be **black** (filtered out or clamped)

"""
    
    with open(output_path, 'w') as f:
        f.write(doc)
    
    print(f"\nReference document saved to: {output_path}")
    return doc


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_scene_exr.py <path_to_exr> [camera_x,camera_y,camera_z]")
        print("\nExample:")
        print("  python analyze_scene_exr.py render.exr")
        print("  python analyze_scene_exr.py render.exr 0,0,500")
        sys.exit(1)
    
    exr_path = sys.argv[1]
    
    # Optional camera position override
    camera_pos = None
    if len(sys.argv) > 2:
        try:
            parts = sys.argv[2].split(',')
            camera_pos = (float(parts[0]), float(parts[1]), float(parts[2]))
            print(f"Using provided camera position: {camera_pos}")
        except:
            print("Warning: Could not parse camera position, will estimate")
    
    if not os.path.exists(exr_path):
        print(f"ERROR: File not found: {exr_path}")
        sys.exit(1)
    
    print(f"Analyzing: {exr_path}")
    print("="*50)
    
    # Read position data
    pos_x, pos_y, pos_z = read_position_data(exr_path)
    if pos_x is None:
        sys.exit(1)
    
    # Analyze scene
    analysis = analyze_scene(pos_x, pos_y, pos_z, camera_pos)
    if analysis is None:
        sys.exit(1)
    
    # Print summary
    print("\n" + "="*50)
    print("SCENE SUMMARY")
    print("="*50)
    print(f"Camera altitude: {analysis['camera_altitude']:.1f}m")
    print(f"Distance range: {analysis['dist_stats']['min']:.1f}m - {analysis['dist_stats']['max']:.1f}m")
    print(f"Median distance: {analysis['dist_stats']['median']:.1f}m ({analysis['dist_stats']['median']/1000:.2f}km)")
    print(f"Sky coverage: {analysis['sky_percent']:.1f}%")
    
    # Generate reference document
    output_dir = os.path.dirname(os.path.abspath(exr_path))
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "TEST_SCENE_REFERENCE.md"
    )
    
    generate_reference_doc(analysis, exr_path, output_path)
    
    print("\n" + "="*50)
    print("Analysis complete!")
    print("="*50)


if __name__ == "__main__":
    main()
