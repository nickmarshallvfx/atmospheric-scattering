"""
Helios UI Panels
"""

import bpy
from .atmosphere_panel import (
    HELIOS_PT_main_panel,
    HELIOS_PT_sun_panel,
    HELIOS_PT_atmosphere_panel,
    HELIOS_PT_rendering_panel,
    HELIOS_PT_planet_panel,
)

classes = (
    HELIOS_PT_main_panel,
    HELIOS_PT_sun_panel,
    HELIOS_PT_atmosphere_panel,
    HELIOS_PT_rendering_panel,
    HELIOS_PT_planet_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
