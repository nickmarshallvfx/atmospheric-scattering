"""
Step 6: Deep dive into transmittance LUT sampling issues

The LUT-based transmittance broke the render (nearly black output).
This script will investigate:
1. What values are actually in the transmittance LUT
2. What UV coordinates we're computing
3. Whether the ratio method T = T_cam / T_pt is correct
4. What values we should expect for typical viewing scenarios
"""

import numpy as np
import math

# Try to load the transmittance LUT
try:
    import OpenImageIO as oiio
    HAS_OIIO = True
except ImportError:
    HAS_OIIO = False
    print("WARNING: OpenImageIO not available")

# Constants from Bruneton
BOTTOM_RADIUS = 6360.0  # km
TOP_RADIUS = 6420.0     # km
H = math.sqrt(TOP_RADIUS**2 - BOTTOM_RADIUS**2)  # ~773.8 km
TRANSMITTANCE_WIDTH = 256
TRANSMITTANCE_HEIGHT = 64

print("=" * 70)
print("TRANSMITTANCE LUT DEEP DIVE")
print("=" * 70)
print(f"\nConstants:")
print(f"  BOTTOM_RADIUS = {BOTTOM_RADIUS} km")
print(f"  TOP_RADIUS = {TOP_RADIUS} km")
print(f"  H = sqrt(top² - bottom²) = {H:.4f} km")
print(f"  LUT size: {TRANSMITTANCE_WIDTH} x {TRANSMITTANCE_HEIGHT}")

# Load the transmittance LUT
lut_path = r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-3\helios_cache\luts\transmittance.exr"

if HAS_OIIO:
    inp = oiio.ImageInput.open(lut_path)
    if inp:
        spec = inp.spec()
        pixels = np.array(inp.read_image(format='float')).reshape(spec.height, spec.width, spec.nchannels)
        inp.close()
        
        print(f"\n--- Transmittance LUT Analysis ---")
        print(f"  Shape: {pixels.shape}")
        print(f"  R range: [{pixels[:,:,0].min():.6f}, {pixels[:,:,0].max():.6f}]")
        print(f"  G range: [{pixels[:,:,1].min():.6f}, {pixels[:,:,1].max():.6f}]")
        print(f"  B range: [{pixels[:,:,2].min():.6f}, {pixels[:,:,2].max():.6f}]")
        
        # Sample at different UV positions
        print(f"\n--- Sample Values at Key UV Positions ---")
        test_uvs = [
            (0.0, 0.0, "bottom-left (low alt, looking down?)"),
            (0.5, 0.0, "middle-left"),
            (1.0, 0.0, "top-left"),
            (0.0, 0.5, "bottom-middle"),
            (0.5, 0.5, "center"),
            (1.0, 0.5, "top-middle"),
            (0.0, 1.0, "bottom-right (high alt?)"),
            (0.5, 1.0, "middle-right"),
            (1.0, 1.0, "top-right"),
        ]
        
        for u, v, desc in test_uvs:
            px = int(u * (TRANSMITTANCE_WIDTH - 1))
            py = int(v * (TRANSMITTANCE_HEIGHT - 1))
            val = pixels[py, px, :3]
            print(f"  UV({u:.1f}, {v:.1f}) -> pixel({px:3d}, {py:2d}): RGB=({val[0]:.4f}, {val[1]:.4f}, {val[2]:.4f}) - {desc}")
    else:
        print(f"ERROR: Could not open {lut_path}")

# Now let's analyze the UV parameterization
print("\n" + "=" * 70)
print("UV PARAMETERIZATION ANALYSIS")
print("=" * 70)

def get_transmittance_uv(r, mu):
    """
    Bruneton's GetTransmittanceTextureUvFromRMu function.
    
    r = distance from planet center (km)
    mu = cos(zenith angle) - where zenith is "up" from surface
         mu = 1 means looking straight up
         mu = 0 means looking horizontally
         mu = -1 means looking straight down
    """
    # rho = distance from planet center to horizon (at altitude r)
    rho = math.sqrt(max(r*r - BOTTOM_RADIUS*BOTTOM_RADIUS, 0))
    
    # H = max possible rho (at top of atmosphere)
    # Already computed above
    
    # d = distance to atmosphere boundary along view direction
    # For mu >= 0 (looking up): d = distance to top of atmosphere
    # d = -r*mu + sqrt(r²*mu² - r² + top²) = -r*mu + sqrt(r²*(mu²-1) + top²)
    
    discriminant = r*r*(mu*mu - 1) + TOP_RADIUS*TOP_RADIUS
    if discriminant < 0:
        d = 0  # Ray doesn't hit atmosphere (shouldn't happen for valid inputs)
    else:
        d = -r*mu + math.sqrt(discriminant)
    
    # d_min = minimum distance (when looking straight up, mu=1)
    d_min = TOP_RADIUS - r
    
    # d_max = maximum distance (when looking at horizon)
    d_max = rho + H
    
    # x_mu = normalized distance parameter [0, 1]
    if d_max - d_min > 0.001:
        x_mu = (d - d_min) / (d_max - d_min)
    else:
        x_mu = 0.0
    
    x_mu = max(0, min(1, x_mu))  # clamp
    
    # x_r = normalized altitude parameter [0, 1]
    x_r = rho / H
    
    # Convert to texture coordinates with half-pixel offset
    u = 0.5 / TRANSMITTANCE_WIDTH + x_mu * (1 - 1/TRANSMITTANCE_WIDTH)
    v = 0.5 / TRANSMITTANCE_HEIGHT + x_r * (1 - 1/TRANSMITTANCE_HEIGHT)
    
    return u, v, d, d_min, d_max, x_mu, x_r, rho

# Test scenarios
print("\n--- Test Scenarios ---")

# Camera at 1m above ground (typical Blender scene)
cam_alt_m = 1.0  # 1 meter
cam_alt_km = cam_alt_m * 0.001  # Convert to km
r_cam = BOTTOM_RADIUS + cam_alt_km

print(f"\nScenario 1: Camera at {cam_alt_m}m altitude")
print(f"  r_cam = {r_cam:.6f} km")

# Looking horizontally (mu = 0)
mu = 0.0
u, v, d, d_min, d_max, x_mu, x_r, rho = get_transmittance_uv(r_cam, mu)
print(f"\n  Looking horizontal (mu=0):")
print(f"    rho = {rho:.6f} km")
print(f"    d = {d:.4f} km, d_min = {d_min:.4f} km, d_max = {d_max:.4f} km")
print(f"    x_mu = {x_mu:.6f}, x_r = {x_r:.10f}")
print(f"    UV = ({u:.6f}, {v:.6f})")

# Looking slightly down (mu = -0.1, typical for ground objects)
mu = -0.1
u, v, d, d_min, d_max, x_mu, x_r, rho = get_transmittance_uv(r_cam, mu)
print(f"\n  Looking slightly down (mu=-0.1):")
print(f"    d = {d:.4f} km")
print(f"    x_mu = {x_mu:.6f}")
print(f"    UV = ({u:.6f}, {v:.6f})")

# Looking up (mu = 0.5)
mu = 0.5
u, v, d, d_min, d_max, x_mu, x_r, rho = get_transmittance_uv(r_cam, mu)
print(f"\n  Looking up (mu=0.5):")
print(f"    d = {d:.4f} km")
print(f"    x_mu = {x_mu:.6f}")
print(f"    UV = ({u:.6f}, {v:.6f})")

# Now the KEY question: What about a point on the ground?
print("\n" + "=" * 70)
print("THE RATIO METHOD: T(cam->pt) = T(cam->top) / T(pt->top)")
print("=" * 70)

# For an object at distance 1km, at ground level
d_to_object = 1.0  # km
r_point = BOTTOM_RADIUS  # Object at ground level

# mu_p at the point (looking back along ray towards top of atmosphere)
# This is complex because we need the ray direction at the point
# mu_p = (r_cam * mu + d) / r_p

print(f"\nObject at {d_to_object} km distance, at ground level")
print(f"  r_point = {r_point:.4f} km (exactly at surface)")

# The issue: when r_point = BOTTOM_RADIUS exactly, rho = 0
rho_pt = math.sqrt(max(r_point*r_point - BOTTOM_RADIUS*BOTTOM_RADIUS, 0))
print(f"  rho_pt = {rho_pt:.10f} km")

# This means x_r = 0, so v = 0.5/64 ≈ 0.0078
x_r_pt = rho_pt / H
v_pt = 0.5 / TRANSMITTANCE_HEIGHT + x_r_pt * (1 - 1/TRANSMITTANCE_HEIGHT)
print(f"  x_r_pt = {x_r_pt:.10f}")
print(f"  v_pt = {v_pt:.10f}")

print("\n--- POTENTIAL ISSUE IDENTIFIED ---")
print("When sampling transmittance at points near ground level:")
print("  - rho approaches 0")
print("  - V coordinate approaches the bottom edge of the LUT")
print("  - This might be sampling invalid/extreme values")

# Let's check what's at the bottom edge of the LUT
if HAS_OIIO and 'pixels' in dir():
    print("\n--- Values at bottom edge of LUT (v ~ 0) ---")
    for u_test in [0.0, 0.25, 0.5, 0.75, 1.0]:
        px = int(u_test * (TRANSMITTANCE_WIDTH - 1))
        val = pixels[0, px, :3]  # Row 0 = bottom of texture
        print(f"  UV({u_test:.2f}, 0.0) -> RGB=({val[0]:.6f}, {val[1]:.6f}, {val[2]:.6f})")
    
    print("\n--- Values at v = 0.5 (mid-altitude) ---")
    py = TRANSMITTANCE_HEIGHT // 2
    for u_test in [0.0, 0.25, 0.5, 0.75, 1.0]:
        px = int(u_test * (TRANSMITTANCE_WIDTH - 1))
        val = pixels[py, px, :3]
        print(f"  UV({u_test:.2f}, 0.5) -> RGB=({val[0]:.6f}, {val[1]:.6f}, {val[2]:.6f})")

print("\n" + "=" * 70)
print("BRUNETON'S GetTransmittance LOGIC")
print("=" * 70)

def ray_intersects_ground(r, mu):
    """Check if ray from (r, mu) intersects the ground."""
    # Ray intersects ground if discriminant r²(mu²-1) + bottom² < 0
    # AND mu < 0 (looking downward)
    if mu >= 0:
        return False
    discriminant = r*r*(mu*mu - 1) + BOTTOM_RADIUS*BOTTOM_RADIUS
    return discriminant >= 0

def get_transmittance_bruneton(r, mu, d):
    """
    Compute transmittance from point at (r) looking in direction (mu) 
    to a point at distance d.
    
    This follows Bruneton's GetTransmittance exactly.
    """
    # Compute r_d and mu_d at the target point
    r_d_sq = d*d + 2*r*mu*d + r*r
    r_d = math.sqrt(max(r_d_sq, BOTTOM_RADIUS*BOTTOM_RADIUS))
    r_d = max(BOTTOM_RADIUS, min(TOP_RADIUS, r_d))  # ClampRadius
    
    mu_d = (r*mu + d) / r_d
    mu_d = max(-1, min(1, mu_d))  # ClampCosine
    
    intersects_ground = ray_intersects_ground(r, mu)
    
    if intersects_ground:
        # T = T(r_d, -mu_d) / T(r, -mu)
        u1, v1, _, _, _, _, _, _ = get_transmittance_uv(r_d, -mu_d)
        u2, v2, _, _, _, _, _, _ = get_transmittance_uv(r, -mu)
        return u1, v1, u2, v2, "ground", r_d, mu_d
    else:
        # T = T(r, mu) / T(r_d, mu_d)
        u1, v1, _, _, _, _, _, _ = get_transmittance_uv(r, mu)
        u2, v2, _, _, _, _, _, _ = get_transmittance_uv(r_d, mu_d)
        return u1, v1, u2, v2, "sky", r_d, mu_d

# Test the actual scenario
print("\nCamera at 1m, looking at ground objects:")

r_cam = BOTTOM_RADIUS + 0.001  # 1m altitude in km
scenarios = [
    (0.001, "Object 1m away"),
    (0.01, "Object 10m away"),
    (0.1, "Object 100m away"),
    (1.0, "Object 1km away"),
    (10.0, "Object 10km away"),
]

for d_km, desc in scenarios:
    # Looking slightly down to hit ground
    # mu = -altitude / distance (approximately for small angles)
    mu = -0.001 / d_km if d_km > 0.001 else -0.1
    mu = max(-1, min(0, mu))
    
    u1, v1, u2, v2, case, r_d, mu_d = get_transmittance_bruneton(r_cam, mu, d_km)
    intersects = ray_intersects_ground(r_cam, mu)
    
    print(f"\n  {desc} (d={d_km}km, mu={mu:.4f}):")
    print(f"    Intersects ground: {intersects} -> case: {case}")
    print(f"    r_d = {r_d:.6f}, mu_d = {mu_d:.6f}")
    print(f"    UV1 (numerator): ({u1:.4f}, {v1:.4f})")
    print(f"    UV2 (denominator): ({u2:.4f}, {v2:.4f})")
    
    # Sample the LUT if available
    if HAS_OIIO and 'pixels' in dir():
        px1 = int(min(u1, 0.999) * (TRANSMITTANCE_WIDTH - 1))
        py1 = int(min(v1, 0.999) * (TRANSMITTANCE_HEIGHT - 1))
        px2 = int(min(u2, 0.999) * (TRANSMITTANCE_WIDTH - 1))
        py2 = int(min(v2, 0.999) * (TRANSMITTANCE_HEIGHT - 1))
        
        t1 = pixels[py1, px1, :3]
        t2 = pixels[py2, px2, :3]
        
        # Compute ratio (with safe division)
        t2_safe = np.maximum(t2, 0.001)
        t_ratio = np.minimum(t1 / t2_safe, 1.0)
        
        print(f"    T_numerator: ({t1[0]:.4f}, {t1[1]:.4f}, {t1[2]:.4f})")
        print(f"    T_denominator: ({t2[0]:.4f}, {t2[1]:.4f}, {t2[2]:.4f})")
        print(f"    T_result: ({t_ratio[0]:.4f}, {t_ratio[1]:.4f}, {t_ratio[2]:.4f})")

print("\n" + "=" * 70)
print("KEY INSIGHT")
print("=" * 70)
print("""
For ground-looking rays (mu < 0), Bruneton uses NEGATED mu values:
  T = T(r_d, -mu_d) / T(r, -mu)

This flips the lookup to use the "looking up" portion of the LUT,
which has higher transmittance values and avoids the near-zero
values at the bottom-right of the LUT.

Our failed implementation likely used:
  T = T(r, mu) / T(r_d, mu_d)  [WRONG for ground rays]

Which samples the low-transmittance region and causes division issues.
""")
