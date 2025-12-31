"""
Helios Export Operator - Export atmospheric AOVs to multi-layer EXR.
"""

import bpy
from bpy.types import Operator
from bpy.props import StringProperty, BoolProperty
import os


class HELIOS_OT_export_exr(Operator):
    """Export atmospheric AOVs to multi-layer EXR for Nuke"""
    bl_idname = "helios.export_exr"
    bl_label = "Export Atmosphere EXR"
    bl_description = "Export Sky, Transmittance, and Inscattering AOVs to multi-layer EXR"
    bl_options = {'REGISTER'}
    
    filepath: StringProperty(
        name="File Path",
        description="Path to save the EXR file",
        subtype='FILE_PATH',
        default="//atmosphere.exr"
    )
    
    include_sky: BoolProperty(
        name="Include Sky",
        description="Include sky radiance pass",
        default=True
    )
    
    include_transmittance: BoolProperty(
        name="Include Transmittance",
        description="Include atmospheric transmittance pass",
        default=True
    )
    
    include_inscatter: BoolProperty(
        name="Include Inscattering",
        description="Include atmospheric inscattering pass",
        default=True
    )
    
    def execute(self, context):
        settings = context.scene.helios
        
        if not settings.luts_valid:
            self.report({'ERROR'}, "Atmosphere not precomputed. Run 'Precompute Atmosphere' first.")
            return {'CANCELLED'}
        
        # Resolve filepath
        filepath = bpy.path.abspath(self.filepath)
        if not filepath.lower().endswith('.exr'):
            filepath += '.exr'
        
        try:
            # For now, report that this feature requires render integration
            self.report({'WARNING'}, 
                "EXR export requires rendering. Use Blender's compositor with "
                "Helios AOV nodes, then render and save as EXR.")
            
            # TODO: Implement direct EXR export when render integration is complete
            # This would involve:
            # 1. Rendering atmosphere passes
            # 2. Combining into multi-layer EXR with proper channel naming:
            #    - helios.sky.R, helios.sky.G, helios.sky.B
            #    - helios.transmittance.R, helios.transmittance.G, helios.transmittance.B
            #    - helios.inscatter.R, helios.inscatter.G, helios.inscatter.B
            
        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {str(e)}")
            return {'CANCELLED'}
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
    
    def draw(self, context):
        layout = self.layout
        layout.prop(self, "include_sky")
        layout.prop(self, "include_transmittance")
        layout.prop(self, "include_inscatter")
