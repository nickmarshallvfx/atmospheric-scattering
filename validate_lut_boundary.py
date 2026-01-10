"""
LUT Boundary Continuity Validation

This script checks if the scattering texture has matching values at the 
u_mu = 0.5 boundary (where ground and non-ground halves meet).

If the values are NOT continuous at this boundary, blending will never work
and we need a fundamentally different approach.

Run this in Blender's Python console.
"""

import bpy
import os
import math

# Constants
BOTTOM_RADIUS = 6360.0
TOP_RADIUS = 6420.0
H = math.sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS)

SCATTERING_TEXTURE_R_SIZE = 32
SCATTERING_TEXTURE_MU_SIZE = 128
SCATTERING_TEXTURE_MU_S_SIZE = 32
SCATTERING_TEXTURE_NU_SIZE = 8

LUT_DIR = r"c:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-3\helios_cache\luts"

def load_exr_image(filepath):
    """Load an EXR image and return the Blender image object."""
    name = os.path.basename(filepath)
    if name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[name])
    img = bpy.data.images.load(filepath)
    return img

def sample_image(img, u, v):
    """Sample an image at UV coordinates (0-1 range)."""
    width, height = img.size
    x = int(u * (width - 1))
    y = int(v * (height - 1))
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    
    pixels = img.pixels[:]
    channels = len(pixels) // (width * height)
    idx = (y * width + x) * channels
    
    if channels >= 3:
        return (pixels[idx], pixels[idx + 1], pixels[idx + 2])
    return (pixels[idx], 0, 0)

def validate_boundary_continuity():
    """Check if scattering texture is continuous at u_mu = 0.5 boundary."""
    
    scattering_path = os.path.join(LUT_DIR, "scattering.exr")
    if not os.path.exists(scattering_path):
        print(f"ERROR: Scattering texture not found at {scattering_path}")
        return
    
    img = load_exr_image(scattering_path)
    print(f"Loaded scattering texture: {img.size[0]}x{img.size[1]}")
    
    # The scattering texture is 3D stored as 2D with horizontal tiling:
    # Width = NU_SIZE * MU_S_SIZE * R_SIZE = 8 * 32 * 32 = 8192
    # Height = MU_SIZE = 128
    #
    # u_mu maps to Y coordinate (0 = ground, 1 = non-ground top)
    # The boundary at u_mu = 0.5 is where ground and non-ground halves meet
    
    print("\n" + "="*70)
    print("SCATTERING TEXTURE BOUNDARY CONTINUITY TEST")
    print("="*70)
    print("\nTesting if values at u_mu=0.5-epsilon match u_mu=0.5+epsilon")
    print("(If they don't match, blending between texture halves won't work)")
    print()
    
    # Sample at various u_r, u_mu_s, u_nu values
    test_cases = [
        (0.1, 0.5, 0.5),  # Low altitude
        (0.5, 0.5, 0.5),  # Mid altitude
        (0.9, 0.5, 0.5),  # High altitude
        (0.5, 0.2, 0.5),  # Different sun angle
        (0.5, 0.8, 0.5),  # Different sun angle
    ]
    
    epsilon = 0.01  # How close to the boundary to sample
    
    all_continuous = True
    
    for u_r, u_mu_s, u_nu in test_cases:
        # Compute texture coordinates for just below and just above boundary
        # Y coordinate is u_mu (flipped in Blender typically, but let's check both)
        
        # For the scattering texture, X encodes (depth_slice, nu, mu_s)
        # Let's sample at a fixed depth slice and nu
        depth_slice = int(u_r * (SCATTERING_TEXTURE_R_SIZE - 1))
        nu_slice = int(u_nu * (SCATTERING_TEXTURE_NU_SIZE - 1))
        mu_s_coord = u_mu_s
        
        # X = (depth_slice + (nu_slice + mu_s_coord) / NU_SIZE) / R_SIZE
        tex_x = (depth_slice + (nu_slice + mu_s_coord) / SCATTERING_TEXTURE_NU_SIZE) / SCATTERING_TEXTURE_R_SIZE
        
        # Sample just below boundary (ground side)
        u_mu_below = 0.5 - epsilon
        # Sample just above boundary (non-ground side)  
        u_mu_above = 0.5 + epsilon
        
        # Y coordinate (may need to flip)
        # Try both flipped and non-flipped
        for flip in [False, True]:
            y_below = (1.0 - u_mu_below) if flip else u_mu_below
            y_above = (1.0 - u_mu_above) if flip else u_mu_above
            
            val_below = sample_image(img, tex_x, y_below)
            val_above = sample_image(img, tex_x, y_above)
            
            diff = (
                abs(val_below[0] - val_above[0]),
                abs(val_below[1] - val_above[1]),
                abs(val_below[2] - val_above[2])
            )
            max_diff = max(diff)
            
            if flip:
                flip_str = "(Y flipped)"
            else:
                flip_str = "(Y normal)"
            
            print(f"u_r={u_r:.1f}, u_mu_s={u_mu_s:.1f}, u_nu={u_nu:.1f} {flip_str}:")
            print(f"  Below boundary (u_mu={u_mu_below:.2f}): ({val_below[0]:.6f}, {val_below[1]:.6f}, {val_below[2]:.6f})")
            print(f"  Above boundary (u_mu={u_mu_above:.2f}): ({val_above[0]:.6f}, {val_above[1]:.6f}, {val_above[2]:.6f})")
            print(f"  Difference: ({diff[0]:.6f}, {diff[1]:.6f}, {diff[2]:.6f}) Max: {max_diff:.6f}")
            
            if max_diff > 0.01:
                print(f"  *** DISCONTINUOUS ***")
                all_continuous = False
            else:
                print(f"  (continuous)")
            print()
    
    print("="*70)
    if all_continuous:
        print("RESULT: Texture appears CONTINUOUS at boundary - blending may work")
    else:
        print("RESULT: Texture is DISCONTINUOUS at boundary - blending WILL NOT work")
        print("        Need fundamentally different approach for flat scenes")
    print("="*70)

# Run the validation
validate_boundary_continuity()
