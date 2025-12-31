"""
Helios Core - Atmospheric scattering model implementation.
"""

# Safe imports - defer heavy imports until actually needed
CUPY_AVAILABLE = False
BLENDER_GPU_AVAILABLE = False

try:
    from .constants import *
    from .parameters import AtmosphereParameters, DensityProfileLayer, RenderingParameters
    from .model import AtmosphereModel
    
    # Try CuPy backend
    try:
        from .model_gpu import GPUAtmosphereModel, CUPY_AVAILABLE
    except ImportError:
        pass
    
    # Try Blender GPU backend - USE V2 (fresh implementation)
    try:
        from .gpu_precompute_v2 import BlenderGPUAtmosphereModel, GPUPrecompute
        BLENDER_GPU_AVAILABLE = True
        print("[Helios] Using GPU precompute V2 (fresh baseline)")
    except ImportError:
        # Fallback to v1 if v2 fails
        try:
            from .gpu_precompute import BlenderGPUAtmosphereModel, GPUPrecompute
            BLENDER_GPU_AVAILABLE = True
        except ImportError:
            pass
        
except ImportError as e:
    print(f"Helios Core: Import error (numpy may not be available): {e}")
