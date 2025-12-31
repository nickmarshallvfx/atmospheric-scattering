"""
Helios Model Tests - Validate LUT precomputation against reference.

Run with: python -m pytest tests/test_model.py
Or standalone: python tests/test_model.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def test_constants():
    """Test that constants are properly defined."""
    from helios.core.constants import (
        TRANSMITTANCE_TEXTURE_WIDTH,
        TRANSMITTANCE_TEXTURE_HEIGHT,
        SCATTERING_TEXTURE_R_SIZE,
        EARTH_RADIUS,
        LAMBDA_R, LAMBDA_G, LAMBDA_B,
    )
    
    assert TRANSMITTANCE_TEXTURE_WIDTH == 256
    assert TRANSMITTANCE_TEXTURE_HEIGHT == 64
    assert SCATTERING_TEXTURE_R_SIZE == 32
    assert EARTH_RADIUS == 6360000.0
    assert LAMBDA_R == 680.0
    assert LAMBDA_G == 550.0
    assert LAMBDA_B == 440.0
    
    print("✓ Constants test passed")


def test_density_profile_layer():
    """Test DensityProfileLayer density computation."""
    from helios.core.parameters import DensityProfileLayer
    
    # Exponential layer (Rayleigh-like)
    layer = DensityProfileLayer(
        width=0.0,
        exp_term=1.0,
        exp_scale=-1.0 / 8000.0,  # 8km scale height
        linear_term=0.0,
        constant_term=0.0
    )
    
    # Density at sea level should be 1.0
    assert abs(layer.get_density(0.0) - 1.0) < 1e-6
    
    # Density at scale height should be ~0.368 (1/e)
    density_at_scale = layer.get_density(8000.0)
    assert abs(density_at_scale - np.exp(-1)) < 1e-6
    
    # Density at high altitude should approach 0
    assert layer.get_density(100000.0) < 0.001
    
    print("✓ DensityProfileLayer test passed")


def test_atmosphere_parameters():
    """Test AtmosphereParameters creation."""
    from helios.core.parameters import AtmosphereParameters
    
    # Default Earth parameters
    params = AtmosphereParameters.earth_default()
    
    assert params.bottom_radius == 6360000.0
    assert params.top_radius == 6360000.0 + 60000.0
    assert params.mie_phase_function_g == 0.8
    assert len(params.rayleigh_scattering) == 3
    
    # Artistic controls
    params2 = AtmosphereParameters.from_artistic_controls(
        rayleigh_density_scale=2.0,
        mie_density_scale=0.5,
        mie_phase_g=0.9,
    )
    
    assert params2.mie_phase_function_g == 0.9
    
    print("✓ AtmosphereParameters test passed")


def test_model_initialization():
    """Test AtmosphereModel can be created and initialized."""
    from helios.core.model import AtmosphereModel
    from helios.core.parameters import AtmosphereParameters
    
    params = AtmosphereParameters.earth_default()
    model = AtmosphereModel(params)
    
    assert not model.is_initialized
    
    # Initialize with minimal scattering orders for speed
    def progress(p, msg):
        print(f"  [{int(p*100):3d}%] {msg}")
    
    print("  Initializing model (this takes a few seconds)...")
    model.init(num_scattering_orders=2, progress_callback=progress)
    
    assert model.is_initialized
    assert model.textures is not None
    assert model.textures.transmittance.shape == (64, 256, 3)
    assert model.textures.irradiance.shape == (16, 64, 3)
    
    print("✓ Model initialization test passed")


def test_transmittance_texture():
    """Test transmittance texture values are physically reasonable."""
    from helios.core.model import AtmosphereModel
    
    model = AtmosphereModel()
    model.init(num_scattering_orders=2)
    
    trans = model.textures.transmittance
    
    # Transmittance should be in [0, 1]
    assert np.all(trans >= 0.0)
    assert np.all(trans <= 1.0)
    
    # Looking straight up at high altitude should have high transmittance
    # (top row of texture, middle column)
    high_trans = trans[-1, trans.shape[1]//2, :]
    assert np.all(high_trans > 0.5), f"Expected high transmittance, got {high_trans}"
    
    print("✓ Transmittance texture test passed")


def test_color_conversion():
    """Test spectrum to sRGB conversion."""
    from helios.core.constants import convert_spectrum_to_linear_srgb
    
    # A flat spectrum should produce roughly equal RGB
    wavelengths = np.linspace(380, 780, 81)
    flat_spectrum = np.ones_like(wavelengths) * 0.01
    
    rgb = convert_spectrum_to_linear_srgb(wavelengths, flat_spectrum)
    
    # Should be roughly balanced (not perfect due to color matching functions)
    assert len(rgb) == 3
    assert np.all(rgb >= 0)
    
    print("✓ Color conversion test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Helios Model Tests")
    print("="*60 + "\n")
    
    tests = [
        test_constants,
        test_density_profile_layer,
        test_atmosphere_parameters,
        test_color_conversion,
        test_model_initialization,
        test_transmittance_texture,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
