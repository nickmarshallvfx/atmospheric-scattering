"""
Validation script for aerial perspective inscatter values.
Computes expected inscatter using Bruneton reference formulas and compares to Blender output.

Run this in Blender Python console after rendering with aerial perspective.
"""

import math

# Constants matching our implementation
BOTTOM_RADIUS = 6360.0  # km
TOP_RADIUS = 6420.0     # km
H = math.sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS)

# Texture sizes
TRANSMITTANCE_WIDTH = 256
TRANSMITTANCE_HEIGHT = 64
SCATTERING_R_SIZE = 32
SCATTERING_MU_SIZE = 128
SCATTERING_MU_S_SIZE = 32
SCATTERING_NU_SIZE = 8


def clamp(x, min_val, max_val):
    return max(min_val, min(max_val, x))


def safe_sqrt(x):
    return math.sqrt(max(0.0, x))


def get_texture_coord_from_unit_range(x, texture_size):
    """Convert unit range [0,1] to texture coordinate with half-pixel offset."""
    return 0.5 / texture_size + x * (1.0 - 1.0 / texture_size)


def get_unit_range_from_texture_coord(u, texture_size):
    """Convert texture coordinate to unit range [0,1]."""
    return (u - 0.5 / texture_size) / (1.0 - 1.0 / texture_size)


def ray_intersects_ground(r, mu):
    """Check if ray from altitude r in direction mu intersects ground."""
    if mu >= 0:
        return False
    discriminant = r * r * (mu * mu - 1.0) + BOTTOM_RADIUS * BOTTOM_RADIUS
    return discriminant >= 0


def get_transmittance_uv(r, mu):
    """
    Compute UV coordinates for transmittance texture lookup.
    Reference: GetTransmittanceTextureUvFromRMu (functions.glsl)
    """
    # rho = distance to horizon
    rho = safe_sqrt(r * r - BOTTOM_RADIUS * BOTTOM_RADIUS)
    
    # x_r = rho / H
    x_r = rho / H
    
    # Distance to top of atmosphere
    # d = -r*mu + sqrt(r²μ² - r² + top²) = -r*mu + sqrt(disc)
    disc = r * r * (mu * mu - 1.0) + TOP_RADIUS * TOP_RADIUS
    d = -r * mu + safe_sqrt(disc)
    
    # d_min = top - r (minimum distance when looking straight up)
    d_min = TOP_RADIUS - r
    
    # d_max = rho + H (maximum distance when looking at horizon)
    d_max = rho + H
    
    # x_mu = (d - d_min) / (d_max - d_min)
    if d_max > d_min:
        x_mu = (d - d_min) / (d_max - d_min)
    else:
        x_mu = 0.0
    
    x_mu = clamp(x_mu, 0.0, 1.0)
    x_r = clamp(x_r, 0.0, 1.0)
    
    u = get_texture_coord_from_unit_range(x_mu, TRANSMITTANCE_WIDTH)
    v = get_texture_coord_from_unit_range(x_r, TRANSMITTANCE_HEIGHT)
    
    return u, v


def get_transmittance_value(r, mu, transmittance_data):
    """
    Sample transmittance texture.
    transmittance_data should be numpy array of shape (height, width, 3)
    """
    u, v = get_transmittance_uv(r, mu)
    
    # Convert to pixel coordinates
    x = u * (transmittance_data.shape[1] - 1)
    y = (1.0 - v) * (transmittance_data.shape[0] - 1)  # Flip V
    
    # Bilinear interpolation
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, transmittance_data.shape[1] - 1), min(y0 + 1, transmittance_data.shape[0] - 1)
    
    fx, fy = x - x0, y - y0
    
    v00 = transmittance_data[y0, x0]
    v10 = transmittance_data[y0, x1]
    v01 = transmittance_data[y1, x0]
    v11 = transmittance_data[y1, x1]
    
    return (v00 * (1-fx) * (1-fy) + v10 * fx * (1-fy) + 
            v01 * (1-fx) * fy + v11 * fx * fy)


def compute_transmittance_between_points(r, mu, d, ray_hits_ground, transmittance_data):
    """
    Compute transmittance between camera at (r, mu) and point at distance d.
    Reference: GetTransmittance (functions.glsl lines 493-518)
    """
    # Compute r_p and mu_p at the point
    r_p = clamp(safe_sqrt(d*d + 2.0*r*mu*d + r*r), BOTTOM_RADIUS, TOP_RADIUS)
    mu_p = (r * mu + d) / r_p
    mu_p = clamp(mu_p, -1.0, 1.0)
    
    if ray_hits_ground:
        # Ground formula: T(r_p, -mu_p) / T(r, -mu)
        T_pt = get_transmittance_value(r_p, -mu_p, transmittance_data)
        T_cam = get_transmittance_value(r, -mu, transmittance_data)
        # Avoid division by zero
        T_cam = np.maximum(T_cam, 1e-6)
        T = T_pt / T_cam
    else:
        # Non-ground formula: T(r, mu) / T(r_p, mu_p)
        T_cam = get_transmittance_value(r, mu, transmittance_data)
        T_pt = get_transmittance_value(r_p, mu_p, transmittance_data)
        # Avoid division by zero
        T_pt = np.maximum(T_pt, 1e-6)
        T = T_cam / T_pt
    
    return np.minimum(T, 1.0), r_p, mu_p


def print_test_case(name, camera_alt_m, point_dist_km, mu, sun_elevation_deg=45.0):
    """
    Print expected values for a test case.
    
    Args:
        camera_alt_m: Camera altitude above ground in meters
        point_dist_km: Distance from camera to point in km
        mu: cos(view_zenith) - 1.0 = straight up, -1.0 = straight down, 0 = horizontal
        sun_elevation_deg: Sun elevation angle in degrees
    """
    print(f"\n{'='*60}")
    print(f"TEST CASE: {name}")
    print(f"{'='*60}")
    
    # Camera position
    r = BOTTOM_RADIUS + camera_alt_m / 1000.0  # Convert m to km
    d = point_dist_km
    
    # Compute point parameters
    r_p_sq = d*d + 2.0*r*mu*d + r*r
    r_p = clamp(safe_sqrt(r_p_sq), BOTTOM_RADIUS, TOP_RADIUS)
    mu_p = (r * mu + d) / r_p
    mu_p = clamp(mu_p, -1.0, 1.0)
    
    # Check ray intersects ground
    ray_hits = ray_intersects_ground(r, mu)
    
    # Transmittance UVs
    uv_cam_ng = get_transmittance_uv(r, mu)
    uv_pt_ng = get_transmittance_uv(r_p, mu_p)
    uv_cam_g = get_transmittance_uv(r, -mu)
    uv_pt_g = get_transmittance_uv(r_p, -mu_p)
    
    # Mu horizon
    rho = safe_sqrt(r * r - BOTTOM_RADIUS * BOTTOM_RADIUS)
    mu_horizon = -rho / r if r > 0 else 0
    
    print(f"\nInputs:")
    print(f"  Camera altitude: {camera_alt_m} m ({r:.4f} km from center)")
    print(f"  Point distance:  {point_dist_km} km")
    print(f"  View mu (cos):   {mu:.6f}")
    print(f"  View angle:      {math.degrees(math.acos(clamp(mu, -1, 1))):.2f}° from zenith")
    
    print(f"\nDerived values:")
    print(f"  r_p:             {r_p:.6f} km")
    print(f"  mu_p:            {mu_p:.6f}")
    print(f"  mu_horizon:      {mu_horizon:.6f}")
    print(f"  ray_intersects:  {ray_hits}")
    
    print(f"\nTransmittance UVs:")
    print(f"  Non-ground cam:  ({uv_cam_ng[0]:.6f}, {uv_cam_ng[1]:.6f})")
    print(f"  Non-ground pt:   ({uv_pt_ng[0]:.6f}, {uv_pt_ng[1]:.6f})")
    print(f"  Ground cam:      ({uv_cam_g[0]:.6f}, {uv_cam_g[1]:.6f})")
    print(f"  Ground pt:       ({uv_pt_g[0]:.6f}, {uv_pt_g[1]:.6f})")
    
    print(f"\nExpected formula: {'GROUND' if ray_hits else 'NON-GROUND'}")


def main():
    """Run validation test cases."""
    print("AERIAL PERSPECTIVE INSCATTER VALIDATION")
    print("=" * 60)
    print("These test cases show expected intermediate values.")
    print("Compare with Blender node outputs to validate implementation.")
    
    # Test case 1: Camera 500m up, looking horizontally at point 5km away
    print_test_case(
        "Horizontal view, 5km distance",
        camera_alt_m=500,
        point_dist_km=5.0,
        mu=0.0  # Horizontal
    )
    
    # Test case 2: Camera 500m up, looking slightly down at ground 2km away
    # mu = cos(angle from zenith), so looking down = mu < 0
    print_test_case(
        "Looking down at ground, 2km",
        camera_alt_m=500,
        point_dist_km=2.0,
        mu=-0.1  # Slightly below horizontal
    )
    
    # Test case 3: Camera 500m up, looking at distant building 10km away, slightly below horizon
    print_test_case(
        "Distant building, 10km",
        camera_alt_m=500,
        point_dist_km=10.0,
        mu=-0.01  # Just below horizontal
    )
    
    # Test case 4: Camera 500m up, looking up at tall building 1km away
    print_test_case(
        "Looking up at building, 1km",
        camera_alt_m=500,
        point_dist_km=1.0,
        mu=0.2  # Above horizontal
    )
    
    # Test case 5: Near the mu_horizon boundary
    # At 500m altitude, mu_horizon ≈ -sqrt(r² - bottom²) / r
    r_test = BOTTOM_RADIUS + 0.5
    rho_test = safe_sqrt(r_test * r_test - BOTTOM_RADIUS * BOTTOM_RADIUS)
    mu_horizon_test = -rho_test / r_test
    
    print_test_case(
        f"AT horizon boundary (mu_h={mu_horizon_test:.6f})",
        camera_alt_m=500,
        point_dist_km=5.0,
        mu=mu_horizon_test
    )
    
    print_test_case(
        "Just ABOVE horizon",
        camera_alt_m=500,
        point_dist_km=5.0,
        mu=mu_horizon_test + 0.001
    )
    
    print_test_case(
        "Just BELOW horizon",
        camera_alt_m=500,
        point_dist_km=5.0,
        mu=mu_horizon_test - 0.001
    )


if __name__ == "__main__":
    main()
