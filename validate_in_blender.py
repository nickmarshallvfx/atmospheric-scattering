"""
Run this script in Blender to validate aerial perspective inscatter values.
Samples actual LUT textures and computes expected inscatter.

Usage: Run in Blender's scripting tab or Python console.
"""

import bpy
import math
import os

# Constants
BOTTOM_RADIUS = 6360.0
TOP_RADIUS = 6420.0
H = math.sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def safe_sqrt(x):
    return math.sqrt(max(0.0, x))

def get_texture_coord(x, size):
    return 0.5 / size + x * (1.0 - 1.0 / size)

def ray_intersects_ground(r, mu):
    if mu >= 0:
        return False
    disc = r * r * (mu * mu - 1.0) + BOTTOM_RADIUS * BOTTOM_RADIUS
    return disc >= 0

def get_transmittance_uv(r, mu):
    """Compute UV for transmittance texture."""
    rho = safe_sqrt(r * r - BOTTOM_RADIUS * BOTTOM_RADIUS)
    x_r = rho / H
    
    disc = r * r * (mu * mu - 1.0) + TOP_RADIUS * TOP_RADIUS
    d = -r * mu + safe_sqrt(disc)
    d_min = TOP_RADIUS - r
    d_max = rho + H
    
    if d_max > d_min:
        x_mu = (d - d_min) / (d_max - d_min)
    else:
        x_mu = 0.0
    
    x_mu = clamp(x_mu, 0.0, 1.0)
    x_r = clamp(x_r, 0.0, 1.0)
    
    u = get_texture_coord(x_mu, 256)
    v = get_texture_coord(x_r, 64)
    
    return u, 1.0 - v  # Flip V for image coordinates

def sample_image(image, u, v):
    """Sample image at UV coordinates using bilinear interpolation."""
    width = image.size[0]
    height = image.size[1]
    
    x = u * (width - 1)
    y = v * (height - 1)
    
    x0 = int(x)
    y0 = int(y)
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    
    fx = x - x0
    fy = y - y0
    
    pixels = image.pixels[:]
    channels = 4 if len(pixels) == width * height * 4 else 3
    
    def get_pixel(px, py):
        idx = (py * width + px) * channels
        return pixels[idx:idx+3]
    
    p00 = get_pixel(x0, y0)
    p10 = get_pixel(x1, y0)
    p01 = get_pixel(x0, y1)
    p11 = get_pixel(x1, y1)
    
    result = []
    for i in range(3):
        val = (p00[i] * (1-fx) * (1-fy) + 
               p10[i] * fx * (1-fy) + 
               p01[i] * (1-fx) * fy + 
               p11[i] * fx * fy)
        result.append(val)
    
    return result

def validate_transmittance():
    """Validate transmittance values at test points."""
    
    # Find transmittance texture - use hardcoded path
    lut_dir = "C:/Users/space/Documents/mattepaint/dev/atmospheric-scattering-3/helios_cache/luts"
    if not os.path.exists(lut_dir):
        # Try relative to blend file
        lut_dir = os.path.join(os.path.dirname(bpy.data.filepath), "luts")
    
    trans_path = os.path.join(lut_dir, "transmittance.exr")
    if not os.path.exists(trans_path):
        print(f"ERROR: Cannot find transmittance.exr at {trans_path}")
        print("Please ensure LUTs are generated.")
        return
    
    # Load image
    if "transmittance_validate" in bpy.data.images:
        img = bpy.data.images["transmittance_validate"]
    else:
        img = bpy.data.images.load(trans_path)
        img.name = "transmittance_validate"
    
    print("\n" + "=" * 60)
    print("TRANSMITTANCE VALIDATION")
    print("=" * 60)
    
    # Test cases: (r, mu, description)
    test_cases = [
        (6360.5, 1.0, "Zenith (straight up)"),
        (6360.5, 0.0, "Horizontal"),
        (6360.5, -0.012539, "At horizon boundary"),
        (6360.5, -0.011539, "Just above horizon"),
        (6360.5, -0.013539, "Just below horizon"),
        (6360.5, -0.1, "Looking down"),
    ]
    
    for r, mu, desc in test_cases:
        u, v = get_transmittance_uv(r, mu)
        T = sample_image(img, u, v)
        ray_hits = ray_intersects_ground(r, mu)
        
        print(f"\n{desc}:")
        print(f"  r={r:.4f}, mu={mu:.6f}")
        print(f"  UV=({u:.6f}, {v:.6f})")
        print(f"  T=({T[0]:.6f}, {T[1]:.6f}, {T[2]:.6f})")
        print(f"  ray_intersects_ground={ray_hits}")
        
        # Also sample with negated mu (for ground formula)
        u_neg, v_neg = get_transmittance_uv(r, -mu)
        T_neg = sample_image(img, u_neg, v_neg)
        print(f"  T(-mu)=({T_neg[0]:.6f}, {T_neg[1]:.6f}, {T_neg[2]:.6f})")

def validate_transmittance_ratio():
    """Validate transmittance ratio between camera and point."""
    
    lut_dir = "C:/Users/space/Documents/mattepaint/dev/atmospheric-scattering-3/helios_cache/luts"
    trans_path = os.path.join(lut_dir, "transmittance.exr")
    
    if not os.path.exists(trans_path):
        print(f"ERROR: Cannot find transmittance.exr")
        return
    
    if "transmittance_validate" in bpy.data.images:
        img = bpy.data.images["transmittance_validate"]
    else:
        img = bpy.data.images.load(trans_path)
        img.name = "transmittance_validate"
    
    print("\n" + "=" * 60)
    print("TRANSMITTANCE RATIO VALIDATION (Camera to Point)")
    print("=" * 60)
    
    # Camera at 500m altitude
    r = 6360.5
    
    # Test cases: (d, mu, description)
    test_cases = [
        (5.0, 0.0, "Horizontal, 5km"),
        (5.0, -0.011539, "Just above horizon, 5km"),
        (5.0, -0.012539, "At horizon, 5km"),
        (5.0, -0.013539, "Just below horizon, 5km"),
        (2.0, -0.1, "Looking down, 2km"),
    ]
    
    for d, mu, desc in test_cases:
        # Compute point parameters
        r_p = clamp(safe_sqrt(d*d + 2*r*mu*d + r*r), BOTTOM_RADIUS, TOP_RADIUS)
        mu_p = clamp((r*mu + d) / r_p, -1.0, 1.0)
        
        ray_hits = ray_intersects_ground(r, mu)
        
        print(f"\n{desc}:")
        print(f"  Camera: r={r:.4f}, mu={mu:.6f}")
        print(f"  Point:  r_p={r_p:.6f}, mu_p={mu_p:.6f}")
        print(f"  ray_intersects_ground={ray_hits}")
        
        # Non-ground formula: T(r, mu) / T(r_p, mu_p)
        u_cam, v_cam = get_transmittance_uv(r, mu)
        u_pt, v_pt = get_transmittance_uv(r_p, mu_p)
        T_cam = sample_image(img, u_cam, v_cam)
        T_pt = sample_image(img, u_pt, v_pt)
        
        T_ng = [T_cam[i] / max(T_pt[i], 1e-6) for i in range(3)]
        T_ng = [min(t, 1.0) for t in T_ng]
        
        print(f"  Non-ground: T_cam/T_pt = ({T_ng[0]:.6f}, {T_ng[1]:.6f}, {T_ng[2]:.6f})")
        
        # Ground formula: T(r_p, -mu_p) / T(r, -mu)
        u_cam_g, v_cam_g = get_transmittance_uv(r, -mu)
        u_pt_g, v_pt_g = get_transmittance_uv(r_p, -mu_p)
        T_cam_g = sample_image(img, u_cam_g, v_cam_g)
        T_pt_g = sample_image(img, u_pt_g, v_pt_g)
        
        T_g = [T_pt_g[i] / max(T_cam_g[i], 1e-6) for i in range(3)]
        T_g = [min(t, 1.0) for t in T_g]
        
        print(f"  Ground:     T_pt(-mu_p)/T_cam(-mu) = ({T_g[0]:.6f}, {T_g[1]:.6f}, {T_g[2]:.6f})")
        
        # Which one should be used?
        expected = "GROUND" if ray_hits else "NON-GROUND"
        print(f"  Expected formula: {expected}")
        
        # Show the difference at boundary
        diff = [abs(T_ng[i] - T_g[i]) for i in range(3)]
        print(f"  Difference: ({diff[0]:.6f}, {diff[1]:.6f}, {diff[2]:.6f})")


def get_scattering_u_mu(r, mu, ray_hits_ground):
    """
    Compute u_mu for scattering texture.
    Non-ground maps to [0.5, 1.0], ground maps to [0, 0.5].
    """
    rho = safe_sqrt(r * r - BOTTOM_RADIUS * BOTTOM_RADIUS)
    
    MU_SIZE = 128
    mu_scale = 1.0 - 2.0 / MU_SIZE
    mu_offset = 1.0 / MU_SIZE
    
    if not ray_hits_ground:
        # Non-ground: d = -r*mu + sqrt(r²(μ²-1) + top²)
        disc = r * r * (mu * mu - 1.0) + TOP_RADIUS * TOP_RADIUS
        d = -r * mu + safe_sqrt(disc)
        d_min = TOP_RADIUS - r
        d_max = rho + H
        
        if d_max > d_min:
            x_mu = (d - d_min) / (d_max - d_min)
        else:
            x_mu = 0.0
        x_mu = clamp(x_mu, 0.0, 1.0)
        
        coord = x_mu * mu_scale + mu_offset
        u_mu = 0.5 + 0.5 * coord  # Maps to [0.5, 1.0]
    else:
        # Ground: d = -r*mu - sqrt(r²(μ²-1) + bottom²)
        disc = r * r * (mu * mu - 1.0) + BOTTOM_RADIUS * BOTTOM_RADIUS
        d = -r * mu - safe_sqrt(disc)
        d_min = r - BOTTOM_RADIUS
        d_max = rho
        
        if d_max > d_min:
            x_mu = (d - d_min) / (d_max - d_min)
        else:
            x_mu = 0.0
        x_mu = clamp(x_mu, 0.0, 1.0)
        
        coord = x_mu * mu_scale + mu_offset
        u_mu = 0.5 - 0.5 * coord  # Maps to [0, 0.5]
    
    return clamp(u_mu, 0.0, 1.0)


def validate_scattering_u_mu():
    """Validate scattering u_mu at horizon boundary."""
    
    print("\n" + "=" * 60)
    print("SCATTERING U_MU VALIDATION (Horizon Boundary)")
    print("=" * 60)
    
    r = 6360.5  # 500m altitude
    
    # Compute mu_horizon
    rho = safe_sqrt(r * r - BOTTOM_RADIUS * BOTTOM_RADIUS)
    mu_horizon = -rho / r
    
    print(f"\nCamera altitude: 500m (r={r})")
    print(f"mu_horizon = {mu_horizon:.6f}")
    print(f"\nScattering u_mu values around horizon:")
    print("-" * 50)
    
    test_mus = [
        (0.0, "Horizontal"),
        (mu_horizon + 0.01, "Above horizon +0.01"),
        (mu_horizon + 0.001, "Above horizon +0.001"),
        (mu_horizon + 0.0001, "Above horizon +0.0001"),
        (mu_horizon, "AT horizon (exact)"),
        (mu_horizon - 0.0001, "Below horizon -0.0001"),
        (mu_horizon - 0.001, "Below horizon -0.001"),
        (mu_horizon - 0.01, "Below horizon -0.01"),
        (-0.1, "Looking down"),
    ]
    
    for mu, desc in test_mus:
        ray_hits = ray_intersects_ground(r, mu)
        u_mu = get_scattering_u_mu(r, mu, ray_hits)
        
        # Also compute what the OTHER formula would give
        u_mu_alt = get_scattering_u_mu(r, mu, not ray_hits)
        
        print(f"\n{desc} (mu={mu:.6f}):")
        print(f"  ray_intersects_ground = {ray_hits}")
        print(f"  u_mu (correct formula) = {u_mu:.6f}")
        print(f"  u_mu (other formula)   = {u_mu_alt:.6f}")
        print(f"  JUMP if switched       = {abs(u_mu - u_mu_alt):.6f}")


def validate_inscatter_continuity():
    """
    Check if INSCATTER result is continuous at the horizon boundary.
    
    The key insight: it's not about texture values matching at u_mu edges.
    It's about the INSCATTER RESULT being continuous when the formula switches.
    
    inscatter = S_cam - T * S_pt
    
    At horizon, when formula switches:
    - inscatter_nonground = S_ng(cam) - T_ng * S_ng(pt)
    - inscatter_ground = S_g(cam) - T_g * S_g(pt)
    
    These should give similar results at the exact horizon.
    """
    
    print("\n" + "=" * 60)
    print("INSCATTER CONTINUITY AT HORIZON")
    print("=" * 60)
    
    # Test parameters
    r = 6360.5  # Camera at 500m
    d = 5.0     # Point 5km away
    
    # Compute mu_horizon
    rho = safe_sqrt(r * r - BOTTOM_RADIUS * BOTTOM_RADIUS)
    mu_horizon = -rho / r
    
    print(f"\nTest setup:")
    print(f"  Camera altitude: 500m (r={r})")
    print(f"  Point distance: {d} km")
    print(f"  mu_horizon: {mu_horizon:.6f}")
    
    # Load textures
    lut_dir = "C:/Users/space/Documents/mattepaint/dev/atmospheric-scattering-3/helios_cache/luts"
    trans_path = os.path.join(lut_dir, "transmittance.exr")
    
    if not os.path.exists(trans_path):
        print(f"ERROR: Cannot find transmittance.exr")
        return
    
    if "transmittance_validate" in bpy.data.images:
        trans_img = bpy.data.images["transmittance_validate"]
    else:
        trans_img = bpy.data.images.load(trans_path)
        trans_img.name = "transmittance_validate"
    
    print("\n" + "-" * 60)
    print("Transmittance ratio comparison at horizon:")
    print("-" * 60)
    
    # Test just above and below horizon
    test_cases = [
        (mu_horizon + 0.001, "Just above horizon"),
        (mu_horizon, "AT horizon (exact)"),
        (mu_horizon - 0.001, "Just below horizon"),
    ]
    
    for mu, desc in test_cases:
        # Compute point parameters
        r_p = clamp(safe_sqrt(d*d + 2*r*mu*d + r*r), BOTTOM_RADIUS, TOP_RADIUS)
        mu_p = clamp((r*mu + d) / r_p, -1.0, 1.0)
        
        ray_hits = ray_intersects_ground(r, mu)
        
        # Compute transmittance using BOTH formulas
        # Non-ground: T = T(r, mu) / T(r_p, mu_p)
        u_cam_ng, v_cam_ng = get_transmittance_uv(r, mu)
        u_pt_ng, v_pt_ng = get_transmittance_uv(r_p, mu_p)
        T_cam_ng = sample_image(trans_img, u_cam_ng, v_cam_ng)
        T_pt_ng = sample_image(trans_img, u_pt_ng, v_pt_ng)
        T_ng = [T_cam_ng[i] / max(T_pt_ng[i], 1e-6) for i in range(3)]
        T_ng = [min(t, 1.0) for t in T_ng]
        
        # Ground: T = T(r_p, -mu_p) / T(r, -mu)
        u_cam_g, v_cam_g = get_transmittance_uv(r, -mu)
        u_pt_g, v_pt_g = get_transmittance_uv(r_p, -mu_p)
        T_cam_g = sample_image(trans_img, u_cam_g, v_cam_g)
        T_pt_g = sample_image(trans_img, u_pt_g, v_pt_g)
        T_g = [T_pt_g[i] / max(T_cam_g[i], 1e-6) for i in range(3)]
        T_g = [min(t, 1.0) for t in T_g]
        
        # Show results
        print(f"\n{desc} (mu={mu:.6f}):")
        print(f"  ray_intersects_ground = {ray_hits}")
        print(f"  T (non-ground): ({T_ng[0]:.4f}, {T_ng[1]:.4f}, {T_ng[2]:.4f})")
        print(f"  T (ground):     ({T_g[0]:.4f}, {T_g[1]:.4f}, {T_g[2]:.4f})")
        
        T_diff = [abs(T_ng[i] - T_g[i]) for i in range(3)]
        print(f"  T difference:   ({T_diff[0]:.4f}, {T_diff[1]:.4f}, {T_diff[2]:.4f})")
        
        # Which formula gives the correct result?
        if ray_hits:
            T_correct = T_g
            T_other = T_ng
        else:
            T_correct = T_ng
            T_other = T_g
        
        print(f"  Correct T:      ({T_correct[0]:.4f}, {T_correct[1]:.4f}, {T_correct[2]:.4f})")
    
    print("\n" + "-" * 60)
    print("CONCLUSION:")
    print("-" * 60)
    print("If T values jump significantly at the horizon boundary,")
    print("and our flat scene has geometry on BOTH sides of that boundary,")
    print("we'll see a visible discontinuity.")
    print("\nThe Bruneton reference works because scene geometry follows")
    print("the planet curvature, so no object spans the horizon boundary.")


if __name__ == "__main__":
    validate_transmittance()
    validate_transmittance_ratio()
    validate_scattering_u_mu()
    validate_inscatter_continuity()
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)
