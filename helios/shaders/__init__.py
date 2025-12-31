"""
Helios Shaders - GLSL and OSL shaders for atmospheric rendering.
"""

import os

SHADER_DIR = os.path.dirname(__file__)


def get_shader_path(name: str) -> str:
    """Get full path to a shader file."""
    return os.path.join(SHADER_DIR, name)


def load_shader(name: str) -> str:
    """Load shader source code as string."""
    path = get_shader_path(name)
    with open(path, 'r') as f:
        return f.read()
