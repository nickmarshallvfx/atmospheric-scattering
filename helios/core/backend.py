"""
GPU/CPU Backend Abstraction for Helios Precomputation.

Provides a unified interface that uses CuPy (GPU) when available,
falling back to NumPy (CPU) otherwise.
"""

import numpy as np

# Try to import CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class ComputeBackend:
    """
    Backend abstraction for array operations.
    
    Automatically uses CuPy (GPU) if available and requested,
    otherwise falls back to NumPy (CPU).
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize compute backend.
        
        Args:
            use_gpu: If True, use GPU (CuPy) when available
        """
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.xp = cp if self.use_gpu else np
        
        if self.use_gpu:
            print(f"[Helios] Using GPU backend (CuPy) - Device: {cp.cuda.Device().name.decode()}")
        else:
            if use_gpu and not CUPY_AVAILABLE:
                print("[Helios] CuPy not available, using CPU backend (NumPy)")
            else:
                print("[Helios] Using CPU backend (NumPy)")
    
    @property
    def name(self) -> str:
        """Get backend name."""
        return "CuPy (GPU)" if self.use_gpu else "NumPy (CPU)"
    
    def zeros(self, shape, dtype=np.float32):
        """Create zero-filled array."""
        return self.xp.zeros(shape, dtype=dtype)
    
    def ones(self, shape, dtype=np.float32):
        """Create one-filled array."""
        return self.xp.ones(shape, dtype=dtype)
    
    def arange(self, *args, **kwargs):
        """Create range array."""
        return self.xp.arange(*args, **kwargs)
    
    def meshgrid(self, *args, **kwargs):
        """Create meshgrid."""
        return self.xp.meshgrid(*args, **kwargs)
    
    def sqrt(self, x):
        """Element-wise square root."""
        return self.xp.sqrt(x)
    
    def exp(self, x):
        """Element-wise exponential."""
        return self.xp.exp(x)
    
    def clip(self, x, a_min, a_max):
        """Clip values to range."""
        return self.xp.clip(x, a_min, a_max)
    
    def maximum(self, x1, x2):
        """Element-wise maximum."""
        return self.xp.maximum(x1, x2)
    
    def minimum(self, x1, x2):
        """Element-wise minimum."""
        return self.xp.minimum(x1, x2)
    
    def where(self, condition, x, y):
        """Conditional selection."""
        return self.xp.where(condition, x, y)
    
    def asarray(self, x, dtype=None):
        """Convert to array."""
        if dtype is not None:
            return self.xp.asarray(x, dtype=dtype)
        return self.xp.asarray(x)
    
    def broadcast_arrays(self, *args):
        """Broadcast arrays to common shape."""
        return self.xp.broadcast_arrays(*args)
    
    def sum(self, x, axis=None):
        """Sum array elements."""
        return self.xp.sum(x, axis=axis)
    
    def abs(self, x):
        """Absolute value."""
        return self.xp.abs(x)
    
    def floor(self, x):
        """Floor."""
        return self.xp.floor(x)
    
    def sin(self, x):
        """Sine."""
        return self.xp.sin(x)
    
    def cos(self, x):
        """Cosine."""
        return self.xp.cos(x)
    
    def dot(self, a, b):
        """Dot product."""
        return self.xp.dot(a, b)
    
    def stack(self, arrays, axis=0):
        """Stack arrays."""
        return self.xp.stack(arrays, axis=axis)
    
    def concatenate(self, arrays, axis=0):
        """Concatenate arrays."""
        return self.xp.concatenate(arrays, axis=axis)
    
    def to_numpy(self, x):
        """Convert array to NumPy (for output/saving)."""
        if self.use_gpu:
            return cp.asnumpy(x)
        return np.asarray(x)
    
    def from_numpy(self, x):
        """Convert NumPy array to backend array."""
        if self.use_gpu:
            return cp.asarray(x)
        return x
    
    def synchronize(self):
        """Synchronize GPU (no-op for CPU)."""
        if self.use_gpu:
            cp.cuda.Stream.null.synchronize()


# Global backend instance (can be changed at runtime)
_backend = None


def get_backend(use_gpu: bool = True) -> ComputeBackend:
    """Get or create the compute backend."""
    global _backend
    if _backend is None:
        _backend = ComputeBackend(use_gpu=use_gpu)
    return _backend


def set_backend(use_gpu: bool = True) -> ComputeBackend:
    """Set the compute backend."""
    global _backend
    _backend = ComputeBackend(use_gpu=use_gpu)
    return _backend


def is_gpu_available() -> bool:
    """Check if GPU (CuPy) is available."""
    return CUPY_AVAILABLE
