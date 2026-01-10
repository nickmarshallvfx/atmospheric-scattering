"""
Validate Step 1.1: Distance Calculation

Compares rendered distance values against computed distance from Position AOV.

Usage:
    python validate_step_1_1.py <distance_render.exr> <position_reference.exr>
    
Or with just distance render (computes expected from same file if it has Position):
    python validate_step_1_1.py <distance_render.exr>
"""

import sys
import os
import numpy as np

try:
    import OpenImageIO as oiio
except ImportError:
    print("ERROR: OpenImageIO not found")
    sys.exit(1)


# Test parameters
MAX_DISTANCE_KM = 10.645  # Must match the value used in the test node group
CAMERA_POS = np.array([37.069, -44.786, 6.0])  # Actual camera position from Blender


def read_layer(exr_path, layer_prefix, channel_suffix=['R', 'G', 'B']):
    """Read a specific layer from an EXR file."""
    inp = oiio.ImageInput.open(exr_path)
    if not inp:
        return None, None
    
    subimage = 0
    while True:
        spec = inp.spec(subimage, 0)
        if spec.width == 0:
            break
        
        channels = list(spec.channelnames)
        
        # Check if this subimage has our layer
        target_channels = [f"{layer_prefix}.{s}" for s in channel_suffix]
        if all(ch in channels for ch in target_channels):
            inp.seek_subimage(subimage, 0)
            pixels = inp.read_image(format=oiio.FLOAT)
            pixels = pixels.reshape(spec.height, spec.width, spec.nchannels)
            
            # Extract channels
            indices = [channels.index(ch) for ch in target_channels]
            data = np.stack([pixels[:, :, i] for i in indices], axis=-1)
            
            inp.close()
            return data, spec
        
        subimage += 1
        if not inp.seek_subimage(subimage, 0):
            break
    
    inp.close()
    return None, None


def compute_distance_from_position(pos_data, camera_pos):
    """Compute distance from camera for each pixel."""
    # pos_data is (H, W, 3) in meters
    # camera_pos is (3,) in meters
    
    diff = pos_data - camera_pos.reshape(1, 1, 3)
    distance_m = np.sqrt(np.sum(diff**2, axis=-1))
    distance_km = distance_m * 0.001
    
    return distance_km


def validate_distance(distance_render_path, position_reference_path=None):
    """
    Validate rendered distance against computed distance from position.
    """
    
    print(f"Validating Step 1.1: Distance Calculation")
    print("="*60)
    
    # Read rendered distance (should be in Combined layer as grayscale)
    rendered, spec = read_layer(distance_render_path, "ViewLayer.Combined")
    if rendered is None:
        rendered, spec = read_layer(distance_render_path, "Composite.Combined")
    if rendered is None:
        print("ERROR: Could not find Combined layer in distance render")
        return False
    
    print(f"Rendered image: {spec.width}x{spec.height}")
    
    # The rendered value is distance_km / MAX_DISTANCE_KM, clamped to [0,1]
    # So rendered distance in km = rendered_value * MAX_DISTANCE_KM
    rendered_distance_km = rendered[:, :, 0] * MAX_DISTANCE_KM  # Use R channel
    
    # Read position data
    pos_path = position_reference_path or distance_render_path
    pos_data, pos_spec = read_layer(pos_path, "ViewLayer.Position", ['X', 'Y', 'Z'])
    if pos_data is None:
        print(f"ERROR: Could not find Position layer in {pos_path}")
        print("Cannot validate without position reference.")
        return False
    
    print(f"Position data: {pos_spec.width}x{pos_spec.height}")
    
    # Compute expected distance
    expected_distance_km = compute_distance_from_position(pos_data, CAMERA_POS)
    
    # Filter out sky pixels (very large or inf position values)
    valid_mask = (expected_distance_km < 100) & (expected_distance_km > 0) & np.isfinite(expected_distance_km)
    
    # Also filter where rendered is 0 (sky)
    valid_mask &= (rendered[:, :, 0] > 0.001)
    
    n_valid = np.sum(valid_mask)
    print(f"Valid pixels for comparison: {n_valid:,}")
    
    if n_valid == 0:
        print("ERROR: No valid pixels to compare")
        return False
    
    # Compare
    rendered_valid = rendered_distance_km[valid_mask]
    expected_valid = expected_distance_km[valid_mask]
    
    # Clamp expected to max (since our shader clamps)
    expected_valid_clamped = np.clip(expected_valid, 0, MAX_DISTANCE_KM)
    
    # Calculate error
    abs_error = np.abs(rendered_valid - expected_valid_clamped)
    rel_error = abs_error / np.maximum(expected_valid_clamped, 0.001)
    
    print("\n" + "-"*60)
    print("RESULTS")
    print("-"*60)
    
    print(f"\nRendered distance range: {rendered_valid.min():.3f} - {rendered_valid.max():.3f} km")
    print(f"Expected distance range: {expected_valid.min():.3f} - {expected_valid.max():.3f} km")
    
    print(f"\nAbsolute error:")
    print(f"  Mean: {np.mean(abs_error):.4f} km")
    print(f"  Max:  {np.max(abs_error):.4f} km")
    print(f"  Std:  {np.std(abs_error):.4f} km")
    
    print(f"\nRelative error:")
    print(f"  Mean: {np.mean(rel_error)*100:.2f}%")
    print(f"  Max:  {np.max(rel_error)*100:.2f}%")
    print(f"  Median: {np.median(rel_error)*100:.2f}%")
    
    # Check percentiles
    print(f"\nError percentiles:")
    for p in [50, 90, 95, 99]:
        print(f"  {p}th: {np.percentile(abs_error, p):.4f} km ({np.percentile(rel_error, p)*100:.2f}%)")
    
    # Pass/fail criteria
    mean_error_threshold = 0.05  # 50 meters
    max_error_threshold = 0.5   # 500 meters (allowing for edge cases)
    
    passed = np.mean(abs_error) < mean_error_threshold
    
    print("\n" + "="*60)
    if passed:
        print("PASS: STEP 1.1 VALIDATED - Distance calculation is correct!")
    else:
        print("FAIL: STEP 1.1 - Distance calculation has errors")
        print(f"   Mean error {np.mean(abs_error):.4f} km > threshold {mean_error_threshold} km")
    print("="*60)
    
    # Additional diagnostics
    print("\nDiagnostics:")
    print(f"  Camera position used: {CAMERA_POS}")
    print(f"  Max distance for normalization: {MAX_DISTANCE_KM} km")
    
    # Check for systematic offset
    offset = np.mean(rendered_valid - expected_valid_clamped)
    print(f"  Systematic offset (rendered - expected): {offset:.4f} km")
    
    # Sample some specific pixels for debugging
    print("\nSample pixel comparisons (10 random):")
    indices = np.where(valid_mask)
    sample_idx = np.random.choice(len(indices[0]), min(10, len(indices[0])), replace=False)
    for i in sample_idx:
        y, x = indices[0][i], indices[1][i]
        print(f"  [{x},{y}]: rendered={rendered_valid[i]:.3f}km, expected={expected_valid_clamped[i]:.3f}km, diff={abs_error[i]:.3f}km")
    
    return passed


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_step_1_1.py <distance_render.exr> [position_reference.exr]")
        sys.exit(1)
    
    distance_path = sys.argv[1]
    position_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(distance_path):
        print(f"ERROR: File not found: {distance_path}")
        sys.exit(1)
    
    success = validate_distance(distance_path, position_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
