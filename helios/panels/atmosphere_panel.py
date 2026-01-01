"""
Helios UI Panels - Atmosphere control panels for Blender.
"""

import bpy
from bpy.types import Panel


class HELIOS_PT_main_panel(Panel):
    """Main Helios Atmosphere panel"""
    bl_label = "Helios Atmosphere"
    bl_idname = "HELIOS_PT_main_panel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "world"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        settings = context.scene.helios
        
        # Check if Helios world exists
        has_helios_world = (context.scene.world and 
                           context.scene.world.name == "Helios Atmosphere")
        
        # Create/Update World button
        col = layout.column(align=True)
        col.scale_y = 1.5
        if has_helios_world:
            col.operator("helios.update_world", text="Update Preview", icon='FILE_REFRESH')
        else:
            col.operator("helios.create_world", text="Create Atmosphere", icon='WORLD')
        
        layout.separator()
        
        # Status indicator
        box = layout.box()
        row = box.row()
        if has_helios_world:
            row.label(text="World: Active", icon='CHECKMARK')
        else:
            row.label(text="World: Not Created", icon='ERROR')
        
        row = box.row()
        if settings.luts_valid:
            row.label(text="LUTs: Ready for Final Render", icon='CHECKMARK')
        else:
            row.label(text="LUTs: Preview Mode (Approximate)", icon='INFO')
        
        # Precompute button (for final render quality)
        layout.separator()
        col = layout.column(align=True)
        col.label(text="Final Render:")
        row = col.row(align=True)
        row.operator("helios.precompute_luts", text="Precompute LUTs", icon='RENDER_STILL')
        
        # GPU toggle with checkbox
        row = col.row(align=True)
        row.prop(settings, "use_gpu", text="Use GPU Acceleration", icon='PREFERENCES')
        
        layout.separator()
        col = layout.column(align=True)
        col.operator("helios.export_exr", icon='RENDER_RESULT')


class HELIOS_PT_sun_panel(Panel):
    """Sun/Environment settings panel"""
    bl_label = "Sun & Environment"
    bl_idname = "HELIOS_PT_sun_panel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "world"
    bl_parent_id = "HELIOS_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        settings = context.scene.helios
        
        col = layout.column(align=True)
        col.prop(settings, "sun_elevation")
        col.prop(settings, "sun_heading")
        
        layout.separator()
        layout.prop(settings, "sun_intensity")


class HELIOS_PT_atmosphere_panel(Panel):
    """Atmospheric composition settings panel"""
    bl_label = "Atmosphere Composition"
    bl_idname = "HELIOS_PT_atmosphere_panel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "world"
    bl_parent_id = "HELIOS_PT_main_panel"
    
    def draw_header(self, context):
        layout = self.layout
        layout.label(text="", icon='WORLD')
    
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        settings = context.scene.helios
        
        # Info: changes update preview in real-time
        box = layout.box()
        box.label(text="Changes update preview instantly", icon='INFO')
        box.label(text="Precompute LUTs for final render quality")
        
        layout.separator()
        
        # Rayleigh (sky color)
        box = layout.box()
        box.label(text="Rayleigh Scattering (Sky Color)", icon='LIGHT_SUN')
        col = box.column(align=True)
        col.prop(settings, "rayleigh_density", text="Density")
        col.prop(settings, "rayleigh_scale_height", text="Scale Height")
        
        layout.separator()
        
        # Mie (haze/fog)
        box = layout.box()
        box.label(text="Mie Scattering (Haze/Fog)", icon='OUTLINER_OB_VOLUME')
        col = box.column(align=True)
        col.prop(settings, "mie_density", text="Density")
        col.prop(settings, "mie_scale_height", text="Scale Height")
        col.prop(settings, "mie_phase_g", text="Phase G")
        col.prop(settings, "mie_angstrom_beta", text="Angstrom Beta")
        
        layout.separator()
        
        # Ozone (sunset colors)
        box = layout.box()
        box.label(text="Ozone Layer (Sunset Colors)", icon='LIGHT_HEMI')
        col = box.column(align=True)
        col.prop(settings, "use_ozone", text="Enable Ozone")
        if settings.use_ozone:
            col.prop(settings, "ozone_density", text="Density")
        
        layout.separator()
        
        # Ground
        layout.prop(settings, "ground_albedo")
        
        # Info box
        if settings.mie_phase_g > 0.9:
            box = layout.box()
            box.alert = True
            box.label(text="High Phase G creates intense sun glare", icon='INFO')


class HELIOS_PT_rendering_panel(Panel):
    """Rendering options panel"""
    bl_label = "Rendering"
    bl_idname = "HELIOS_PT_rendering_panel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "world"
    bl_parent_id = "HELIOS_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        settings = context.scene.helios
        
        # Preview quality at the top - important for workflow
        box = layout.box()
        box.prop(settings, "preview_quality")
        if settings.preview_quality == 'PREVIEW':
            box.label(text="Preview: 2 orders (~3x faster)", icon='INFO')
        else:
            box.label(text="Final: 4 orders (full quality)", icon='CHECKMARK')
        
        layout.separator()
        
        col = layout.column(align=True)
        col.prop(settings, "exposure")
        col.prop(settings, "white_balance")
        
        layout.separator()
        
        col = layout.column(align=True)
        col.prop(settings, "luminance_mode")
        col.prop(settings, "render_mode")
        
        layout.separator()
        
        col = layout.column(align=True)
        col.prop(settings, "use_ozone")


class HELIOS_PT_planet_panel(Panel):
    """Planet parameters panel"""
    bl_label = "Planet"
    bl_idname = "HELIOS_PT_planet_panel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "world"
    bl_parent_id = "HELIOS_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        settings = context.scene.helios
        
        col = layout.column(align=True)
        col.prop(settings, "planet_radius", text="Radius (km)")
        col.prop(settings, "atmosphere_height", text="Atmosphere (km)")
        
        # Presets
        layout.separator()
        layout.label(text="Presets:")
        row = layout.row(align=True)
        row.operator("helios.preset_earth", text="Earth")
        row.operator("helios.preset_mars", text="Mars")
        row.operator("helios.preset_titan", text="Titan")


# Preset operators
class HELIOS_OT_preset_earth(bpy.types.Operator):
    """Set Earth atmosphere parameters"""
    bl_idname = "helios.preset_earth"
    bl_label = "Earth Preset"
    
    def execute(self, context):
        settings = context.scene.helios
        settings.planet_radius = 6360.0
        settings.atmosphere_height = 60.0
        settings.rayleigh_density = 1.0
        settings.mie_density = 1.0
        settings.mie_phase_g = 0.8
        settings.rayleigh_scale_height = 8000.0
        settings.mie_scale_height = 1200.0
        settings.ground_albedo = 0.1
        settings.use_ozone = True
        settings.luts_valid = False
        return {'FINISHED'}


class HELIOS_OT_preset_mars(bpy.types.Operator):
    """Set Mars atmosphere parameters"""
    bl_idname = "helios.preset_mars"
    bl_label = "Mars Preset"
    
    def execute(self, context):
        settings = context.scene.helios
        settings.planet_radius = 3390.0
        settings.atmosphere_height = 100.0
        settings.rayleigh_density = 0.02  # Very thin atmosphere
        settings.mie_density = 5.0  # Dusty
        settings.mie_phase_g = 0.76
        settings.rayleigh_scale_height = 11000.0
        settings.mie_scale_height = 2000.0
        settings.ground_albedo = 0.25  # Reddish surface
        settings.use_ozone = False
        settings.luts_valid = False
        return {'FINISHED'}


class HELIOS_OT_preset_titan(bpy.types.Operator):
    """Set Titan atmosphere parameters"""
    bl_idname = "helios.preset_titan"
    bl_label = "Titan Preset"
    
    def execute(self, context):
        settings = context.scene.helios
        settings.planet_radius = 2575.0
        settings.atmosphere_height = 600.0  # Very thick atmosphere
        settings.rayleigh_density = 4.0  # Dense nitrogen atmosphere
        settings.mie_density = 10.0  # Heavy haze
        settings.mie_phase_g = 0.85
        settings.rayleigh_scale_height = 20000.0
        settings.mie_scale_height = 5000.0
        settings.ground_albedo = 0.15
        settings.use_ozone = False
        settings.luts_valid = False
        return {'FINISHED'}


class HELIOS_PT_aerial_panel(Panel):
    """Aerial Perspective settings panel"""
    bl_label = "Aerial Perspective"
    bl_idname = "HELIOS_PT_aerial_panel"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "world"
    bl_parent_id = "HELIOS_PT_main_panel"
    
    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False
        
        # Info box
        box = layout.box()
        box.label(text="Atmospheric haze for scene objects", icon='OUTLINER_OB_VOLUME')
        box.label(text="Outputs AOVs: transmittance, inscatter")
        
        layout.separator()
        
        # Add to materials buttons
        col = layout.column(align=True)
        col.label(text="Add to Materials:")
        row = col.row(align=True)
        row.operator("helios.add_aerial_to_selected", text="Selected", icon='RESTRICT_SELECT_OFF')
        row.operator("helios.add_aerial_to_all", text="All", icon='WORLD')
        
        layout.separator()
        
        # Remove from materials buttons
        col = layout.column(align=True)
        col.label(text="Remove from Materials:")
        row = col.row(align=True)
        row.operator("helios.remove_aerial_from_selected", text="Selected", icon='X')
        row.operator("helios.remove_aerial_from_all", text="All", icon='TRASH')
        
        layout.separator()
        
        # Update button
        col = layout.column(align=True)
        col.operator("helios.update_aerial", text="Update All Nodes", icon='FILE_REFRESH')
        
        layout.separator()
        
        # Nuke workflow info
        box = layout.box()
        box.label(text="Nuke Compositing:", icon='INFO')
        box.label(text="Beauty × Transmittance + Inscatter")
        box.label(text="Then merge over Sky AOV")
