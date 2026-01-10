# =============================================================================
# STEP 2.4b: FULL INSCATTER WITH LUT TRANSMITTANCE
# V136 - January 9, 2026
# =============================================================================
# This file is a STUB - the full implementation needs to be added to
# aerial_test_steps.py due to its length. See STEP_2_4B_IMPLEMENTATION_SPEC.md
# =============================================================================

"""
Step 2.4b builds everything from scratch:
1. Geometry (d, mu, view_dir) - shared
2. Sun parameters (mu_s, nu)
3. Point parameters (r_p, mu_p, mu_s_p) - shared by scattering AND transmittance
4. Scattering at camera (S_cam)
5. Scattering at point (S_pt)
6. LUT Transmittance with ground handling and horizon fallback
7. Inscatter = S_cam - T × S_pt
8. Phase functions
9. AOVs

Key insight: r_p/mu_p are the same as r_d/mu_d - share them!
"""

# Implementation will be added directly to aerial_test_steps.py
# This file serves as documentation placeholder.
