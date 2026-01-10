"""
Validate Step 1.2: r, mu Calculation

Compares rendered r and mu values against computed values from Position AOV.

Usage:
    python validate_step_1_2.py <r_mu_render.exr>
"""

import sys
import os
import numpy as np

try:
    import OpenImageIO as oiio
except ImportError:
    print("ERROR: OpenImageIO not found")
    sys.exit(1)


# Constants
BOTTOM_RADIUS = 6360.0  # km
PLANET_CENTER = np.array([0.0, 0.0, -6360.0 * 1000])  # meters
CAMERA_POS = np.array([37.069, -44.786, 6.0])  # meters


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
        
        target_channels = [f"{layer_prefix}.{s}" for s in channel_suffix]
        if all(ch in channels for ch in target_channels):
            inp.seek_subimage(subimage, 0)
            pixels = inp.read_image(format=oiio.FLOAT)
            pixels = pixels.reshape(spec.height, spec.width, spec.nchannels)
            
            indices = [channels.index(ch) for ch in target_channels]
            data = np.stack([pixels[:, :, i] for i in indices], axis=-1)
            
            inp.close()
            return data, spec
        
        subimage += 1
        if not inp.seek_subimage(subimage, 0):
            break
    
    inp.close()
    return None, None


def compute_r_mu_from_position(pos_data, camera_pos, planet_center):
    """
    Compute r and mu from position data.
    
    r = length(camera - planet_center) in km
    mu = dot(view_dir, up_at_camera)
    
    where:
        view_dir = normalize(world_pos - camera)
        up_at_camera = normalize(camera - planet_center)
    """
    # Camera relative to planet center (in meters)
    cam_rel_m = camera_pos - planet_center
    
    # r = distance from planet center to camera (km)
    r = np.linalg.norm(cam_rel_m) * 0.001
    
    # Up direction at camera
    up_at_cam = cam_rel_m / np.linalg.norm(cam_rel_m)
    
    # View direction for each pixel
    view_vec = pos_data - camera_pos.reshape(1, 1, 3)
    view_dist = np.linalg.norm(view_vec, axis=-1, keepdims=True)
    view_dir = view_vec / np.maximum(view_dist, 1e-6)
    
    # mu = dot(view_dir, up_at_cam)
    mu = np.sum(view_dir * up_at_cam.reshape(1, 1, 3), axis=-1)
    
    return r, mu


def validate_r_mu(render_path):
    """Validate rendered r and mu against computed values."""
    
    print(f"Validating Step 1.2: r, mu Calculation")
    print("="*60)
    
    # Read rendered values (R = r/6400, G = (mu+1)/2)
    rendered, spec = read_layer(render_path, "ViewLayer.Combined")
    if rendered is None:
        rendered, spec = read_layer(render_path, "Composite.Combined")
    if rendered is None:
        print("ERROR: Could not find Combined layer")
        return False
    
    print(f"Rendered image: {spec.width}x{spec.height}")
    
    # Decode rendered values
    rendered_r = rendered[:, :, 0] * 6400.0  # R channel * 6400 = r in km
    rendered_mu = rendered[:, :, 1] * 2.0 - 1.0  # G channel * 2 - 1 = mu
    
    # Read position data
    pos_data, pos_spec = read_layer(render_path, "ViewLayer.Position", ['X', 'Y', 'Z'])
    if pos_data is None:
        print("ERROR: Could not find Position layer")
        return False
    
    # Compute expected values
    expected_r, expected_mu = compute_r_mu_from_position(pos_data, CAMERA_POS, PLANET_CENTER)
    
    # Valid pixels (exclude sky)
    valid_mask = np.isfinite(expected_mu) & (np.linalg.norm(pos_data, axis=-1) < 1e10)
    valid_mask &= (rendered[:, :, 0] > 0.001)  # Has geometry
    
    n_valid = np.sum(valid_mask)
    print(f"Valid pixels: {n_valid:,}")
    
    if n_valid == 0:
        print("ERROR: No valid pixels")
        return False
    
    # --- Validate r ---
    print("\n" + "-"*60)
    print("R VALUE (distance from planet center)")
    print("-"*60)
    
    # r should be constant (camera position doesn't change per pixel)
    print(f"Expected r: {expected_r:.6f} km")
    print(f"Rendered r range: {rendered_r[valid_mask].min():.6f} - {rendered_r[valid_mask].max():.6f} km")
    print(f"Rendered r mean: {rendered_r[valid_mask].mean():.6f} km")
    
    r_error = np.abs(rendered_r[valid_mask] - expected_r)
    print(f"r error - Mean: {r_error.mean():.6f} km, Max: {r_error.max():.6f} km")
    
    r_passed = r_error.mean() < 0.01  # Within 10 meters
    
    # --- Validate mu ---
    print("\n" + "-"*60)
    print("MU VALUE (cosine of view zenith angle)")
    print("-"*60)
    
    rendered_mu_valid = rendered_mu[valid_mask]
    expected_mu_valid = expected_mu[valid_mask]
    
    print(f"Expected mu range: {expected_mu_valid.min():.4f} - {expected_mu_valid.max():.4f}")
    print(f"Rendered mu range: {rendered_mu_valid.min():.4f} - {rendered_mu_valid.max():.4f}")
    
    mu_error = np.abs(rendered_mu_valid - expected_mu_valid)
    print(f"mu error - Mean: {mu_error.mean():.6f}, Max: {mu_error.max():.6f}")
    print(f"mu error - Median: {np.median(mu_error):.6f}")
    
    # Error percentiles
    print(f"\nmu error percentiles:")
    for p in [50, 90, 95, 99]:
        print(f"  {p}th: {np.percentile(mu_error, p):.6f}")
    
    mu_passed = mu_error.mean() < 0.01  # mu error < 0.01
    
    # --- Sample comparisons ---
    print("\n" + "-"*60)
    print("Sample pixel comparisons:")
    print("-"*60)
    indices = np.where(valid_mask)
    sample_idx = np.random.choice(len(indices[0]), min(10, len(indices[0])), replace=False)
    for i in sample_idx:
        y, x = indices[0][i], indices[1][i]
        print(f"  [{x},{y}]: r={rendered_r[y,x]:.3f}km (exp={expected_r:.3f}), mu={rendered_mu[y,x]:.4f} (exp={expected_mu[y,x]:.4f})")
    
    # --- Final verdict ---
    print("\n" + "="*60)
    passed = r_passed and mu_passed
    if passed:
        print("PASS: STEP 1.2 VALIDATED - r, mu calculation is correct!")
    else:
        print("FAIL: STEP 1.2 - Errors detected")
        if not r_passed:
            print("  - r calculation has errors")
        if not mu_passed:
            print("  - mu calculation has errors")
    print("="*60)
    
    return passed


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_step_1_2.py <r_mu_render.exr>")
        sys.exit(1)
    
    render_path = sys.argv[1]
    if not os.path.exists(render_path):
        print(f"ERROR: File not found: {render_path}")
        sys.exit(1)
    
    success = validate_r_mu(render_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
