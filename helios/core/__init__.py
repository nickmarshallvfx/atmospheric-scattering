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
    
    # Try Blender GPU backend
    try:
        from .gpu_precompute import BlenderGPUAtmosphereModel, GPUPrecompute
        BLENDER_GPU_AVAILABLE = True
    except ImportError:
        pass
        
except ImportError as e:
    print(f"Helios Core: Import error (numpy may not be available): {e}")
