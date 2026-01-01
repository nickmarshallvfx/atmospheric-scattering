"""
Helios Two-Pass Render System

Workaround for Blender bug where AOV Output nodes don't work with OSL enabled.
(See Blender issues #79942, #100282, and pending PR #134021)

Pass 1: OSL enabled - renders beauty with correct sky/atmosphere
Pass 2: OSL disabled - renders AOVs (transmittance, inscatter)

The two passes are combined in compositing or Nuke.
"""

import bpy
from bpy.types import Operator


class HELIOS_OT_render_beauty(Operator):
    """Render beauty pass with OSL enabled (correct sky/atmosphere)"""
    bl_idname = "helios.render_beauty"
    bl_label = "Render Beauty (OSL)"
    bl_description = "Render beauty pass with OSL enabled for correct atmospheric visuals"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        scene = context.scene
        
        # Ensure OSL is enabled for correct sky rendering
        scene.cycles.shading_system = True
        
        self.report({'INFO'}, "Helios: Rendering beauty pass (OSL enabled)")
        
        # Trigger render
        bpy.ops.render.render('INVOKE_DEFAULT')
        
        return {'FINISHED'}


class HELIOS_OT_render_aovs(Operator):
    """Render AOV pass with OSL disabled (AOVs work correctly)"""
    bl_idname = "helios.render_aovs"
    bl_label = "Render AOVs (No OSL)"
    bl_description = "Render AOV pass with OSL disabled so AOV Output nodes work"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        scene = context.scene
        
        # Disable OSL so AOVs work
        scene.cycles.shading_system = False
        
        self.report({'INFO'}, "Helios: Rendering AOV pass (OSL disabled)")
        
        # Trigger render
        bpy.ops.render.render('INVOKE_DEFAULT')
        
        return {'FINISHED'}


class HELIOS_OT_render_both_passes(Operator):
    """Render both beauty and AOV passes sequentially"""
    bl_idname = "helios.render_both_passes"
    bl_label = "Render Both Passes"
    bl_description = "Render beauty (OSL) then AOVs (no OSL) sequentially"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        scene = context.scene
        
        # Store original output path
        original_path = scene.render.filepath
        
        # Pass 1: Beauty with OSL
        scene.cycles.shading_system = True
        scene.render.filepath = original_path + "_beauty"
        self.report({'INFO'}, "Helios: Rendering Pass 1/2 - Beauty (OSL)")
        bpy.ops.render.render(write_still=True)
        
        # Pass 2: AOVs without OSL
        scene.cycles.shading_system = False
        scene.render.filepath = original_path + "_aovs"
        self.report({'INFO'}, "Helios: Rendering Pass 2/2 - AOVs (no OSL)")
        bpy.ops.render.render(write_still=True)
        
        # Restore original path and OSL setting
        scene.render.filepath = original_path
        scene.cycles.shading_system = True  # Restore for viewport
        
        self.report({'INFO'}, "Helios: Both passes complete")
        return {'FINISHED'}


class HELIOS_OT_toggle_osl(Operator):
    """Toggle OSL on/off for testing"""
    bl_idname = "helios.toggle_osl"
    bl_label = "Toggle OSL"
    bl_description = "Toggle Open Shading Language on/off"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        scene = context.scene
        scene.cycles.shading_system = not scene.cycles.shading_system
        
        status = "enabled" if scene.cycles.shading_system else "disabled"
        self.report({'INFO'}, f"Helios: OSL {status}")
        return {'FINISHED'}


# Registration
classes = [
    HELIOS_OT_render_beauty,
    HELIOS_OT_render_aovs,
    HELIOS_OT_render_both_passes,
    HELIOS_OT_toggle_osl,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
