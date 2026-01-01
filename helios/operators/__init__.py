"""
Helios Operators
"""

import bpy
from .precompute import HELIOS_OT_precompute_luts
from .export import HELIOS_OT_export_exr
from . import aerial_ops
from .. import render_passes

# Import preset operators from panels (they're defined there for organization)
from ..panels.atmosphere_panel import (
    HELIOS_OT_preset_earth,
    HELIOS_OT_preset_mars,
    HELIOS_OT_preset_titan,
)

# Import world operators
from ..world import (
    HELIOS_OT_create_world,
    HELIOS_OT_update_world,
)

classes = (
    HELIOS_OT_precompute_luts,
    HELIOS_OT_export_exr,
    HELIOS_OT_preset_earth,
    HELIOS_OT_preset_mars,
    HELIOS_OT_preset_titan,
    HELIOS_OT_create_world,
    HELIOS_OT_update_world,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    aerial_ops.register()
    render_passes.register()


def unregister():
    render_passes.unregister()
    aerial_ops.unregister()
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
