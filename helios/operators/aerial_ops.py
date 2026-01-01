"""
Helios Aerial Perspective Operators

Operators for managing aerial perspective on scene materials.
"""

import bpy
from bpy.types import Operator


class HELIOS_OT_setup_aerial_aovs(Operator):
    """Set up AOVs for aerial perspective rendering"""
    bl_idname = "helios.setup_aerial_aovs"
    bl_label = "Setup Aerial AOVs"
    bl_description = "Create AOV outputs for transmittance and inscatter"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from .. import aerial
        
        if aerial.setup_aerial_aovs(context):
            self.report({'INFO'}, "Aerial perspective AOVs created")
            return {'FINISHED'}
        else:
            self.report({'ERROR'}, "Failed to create AOVs")
            return {'CANCELLED'}


class HELIOS_OT_add_aerial_to_selected(Operator):
    """Add aerial perspective to selected objects' materials"""
    bl_idname = "helios.add_aerial_to_selected"
    bl_label = "Add Aerial to Selected"
    bl_description = "Add aerial perspective shader to materials of selected objects"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from .. import aerial
        
        # First ensure AOVs are set up
        aerial.setup_aerial_aovs(context)
        
        count = 0
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                for slot in obj.material_slots:
                    if slot.material:
                        if aerial.add_aerial_to_material(slot.material, context):
                            count += 1
        
        # Force full scene update so shaders recompile with AOV nodes
        context.view_layer.update()
        
        if count > 0:
            self.report({'INFO'}, f"Added aerial perspective to {count} material(s)")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "No materials found on selected objects")
            return {'CANCELLED'}


class HELIOS_OT_add_aerial_to_all(Operator):
    """Add aerial perspective to all materials in the scene"""
    bl_idname = "helios.add_aerial_to_all"
    bl_label = "Add Aerial to All Materials"
    bl_description = "Add aerial perspective shader to all materials in the scene"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from .. import aerial
        
        # First ensure AOVs are set up
        aerial.setup_aerial_aovs(context)
        
        count = aerial.add_aerial_to_all_materials(context)
        
        # Force full scene update so shaders recompile with AOV nodes
        context.view_layer.update()
        
        if count > 0:
            self.report({'INFO'}, f"Added aerial perspective to {count} material(s)")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "No materials found")
            return {'CANCELLED'}


class HELIOS_OT_remove_aerial_from_selected(Operator):
    """Remove aerial perspective from selected objects' materials"""
    bl_idname = "helios.remove_aerial_from_selected"
    bl_label = "Remove Aerial from Selected"
    bl_description = "Remove aerial perspective shader from materials of selected objects"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from .. import aerial
        
        count = 0
        for obj in context.selected_objects:
            if obj.type == 'MESH':
                for slot in obj.material_slots:
                    if slot.material:
                        if aerial.remove_aerial_from_material(slot.material):
                            count += 1
        
        if count > 0:
            self.report({'INFO'}, f"Removed aerial perspective from {count} material(s)")
            return {'FINISHED'}
        else:
            self.report({'WARNING'}, "No materials found on selected objects")
            return {'CANCELLED'}


class HELIOS_OT_remove_aerial_from_all(Operator):
    """Remove aerial perspective from all materials"""
    bl_idname = "helios.remove_aerial_from_all"
    bl_label = "Remove Aerial from All"
    bl_description = "Remove aerial perspective shader from all materials"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        from .. import aerial
        
        count = aerial.remove_aerial_from_all_materials()
        
        # Also remove AOVs
        aerial.remove_aerial_aovs(context)
        
        self.report({'INFO'}, f"Removed aerial perspective from {count} material(s)")
        return {'FINISHED'}


class HELIOS_OT_update_aerial(Operator):
    """Update all aerial perspective nodes with current settings"""
    bl_idname = "helios.update_aerial"
    bl_label = "Update Aerial Perspective"
    bl_description = "Update all aerial perspective nodes with current Helios settings"
    bl_options = {'REGISTER'}
    
    def execute(self, context):
        from .. import aerial
        
        aerial.update_all_aerial_nodes(context)
        self.report({'INFO'}, "Updated aerial perspective nodes")
        return {'FINISHED'}


# Registration
classes = [
    HELIOS_OT_setup_aerial_aovs,
    HELIOS_OT_add_aerial_to_selected,
    HELIOS_OT_add_aerial_to_all,
    HELIOS_OT_remove_aerial_from_selected,
    HELIOS_OT_remove_aerial_from_all,
    HELIOS_OT_update_aerial,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
