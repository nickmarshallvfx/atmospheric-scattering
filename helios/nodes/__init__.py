"""
Helios Custom Nodes - Blender compositor and shader nodes.
"""

import bpy
from bpy.types import NodeTree, Node, NodeSocket

# Custom socket types for atmosphere data
class HeliosAtmosphereSocket(NodeSocket):
    """Socket for passing atmosphere data between nodes"""
    bl_idname = 'HeliosAtmosphereSocket'
    bl_label = 'Atmosphere Socket'
    
    def draw(self, context, layout, node, text):
        layout.label(text=text)
    
    def draw_color(self, context, node):
        return (0.2, 0.6, 1.0, 1.0)  # Blue for atmosphere


class HeliosNodeTree(NodeTree):
    """Custom node tree for Helios atmosphere nodes"""
    bl_idname = 'HeliosNodeTree'
    bl_label = 'Helios Atmosphere'
    bl_icon = 'WORLD'


# Base class for Helios nodes
class HeliosNodeBase:
    """Base class for all Helios nodes"""
    
    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname in ('HeliosNodeTree', 'ShaderNodeTree', 'CompositorNodeTree')


# ============================================================================
# COMPOSITOR NODES
# ============================================================================

class HELIOS_NODE_atmosphere_input(Node, HeliosNodeBase):
    """Input node for atmosphere parameters"""
    bl_idname = 'HeliosAtmosphereInput'
    bl_label = 'Atmosphere Input'
    bl_icon = 'WORLD'
    
    def init(self, context):
        self.outputs.new('NodeSocketColor', 'Transmittance LUT')
        self.outputs.new('NodeSocketColor', 'Scattering LUT')
        self.outputs.new('NodeSocketColor', 'Irradiance LUT')
    
    def draw_buttons(self, context, layout):
        settings = context.scene.helios
        if settings.luts_valid:
            layout.label(text="LUTs Ready", icon='CHECKMARK')
        else:
            layout.label(text="Not Computed", icon='ERROR')
            layout.operator("helios.precompute_luts", text="Precompute")


class HELIOS_NODE_sky_radiance(Node, HeliosNodeBase):
    """Compute sky radiance from atmosphere"""
    bl_idname = 'HeliosSkyRadiance'
    bl_label = 'Sky Radiance'
    bl_icon = 'LIGHT_SUN'
    
    def init(self, context):
        # Inputs
        self.inputs.new('NodeSocketVector', 'View Direction')
        self.inputs.new('NodeSocketVector', 'Sun Direction')
        self.inputs.new('NodeSocketVector', 'Camera Position')
        
        # Outputs
        self.outputs.new('NodeSocketColor', 'Sky')
        self.outputs.new('NodeSocketColor', 'Transmittance')
    
    def draw_buttons(self, context, layout):
        layout.label(text="Sky radiance lookup")


class HELIOS_NODE_aerial_perspective(Node, HeliosNodeBase):
    """Apply aerial perspective to rendered image"""
    bl_idname = 'HeliosAerialPerspective'
    bl_label = 'Aerial Perspective'
    bl_icon = 'OUTLINER_OB_VOLUME'
    
    def init(self, context):
        # Inputs
        self.inputs.new('NodeSocketColor', 'Image')
        self.inputs.new('NodeSocketFloat', 'Depth')
        self.inputs.new('NodeSocketVector', 'Camera Position')
        self.inputs.new('NodeSocketVector', 'Sun Direction')
        
        # Outputs  
        self.outputs.new('NodeSocketColor', 'Image')
        self.outputs.new('NodeSocketColor', 'Transmittance')
        self.outputs.new('NodeSocketColor', 'Inscattering')
    
    def draw_buttons(self, context, layout):
        settings = context.scene.helios
        layout.prop(settings, "exposure", text="Exposure")


class HELIOS_NODE_aov_split(Node, HeliosNodeBase):
    """Split atmosphere into AOV passes for Nuke"""
    bl_idname = 'HeliosAOVSplit'
    bl_label = 'AOV Split'
    bl_icon = 'RENDERLAYERS'
    
    def init(self, context):
        # Input - combined atmosphere
        self.inputs.new('NodeSocketColor', 'Atmosphere')
        
        # Outputs - individual AOVs
        self.outputs.new('NodeSocketColor', 'helios.sky')
        self.outputs.new('NodeSocketColor', 'helios.transmittance')
        self.outputs.new('NodeSocketColor', 'helios.inscatter')
    
    def draw_buttons(self, context, layout):
        layout.label(text="Nuke-compatible AOVs")


# Registration
node_classes = (
    HeliosAtmosphereSocket,
    HeliosNodeTree,
    HELIOS_NODE_atmosphere_input,
    HELIOS_NODE_sky_radiance,
    HELIOS_NODE_aerial_perspective,
    HELIOS_NODE_aov_split,
)


def register():
    for cls in node_classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(node_classes):
        bpy.utils.unregister_class(cls)
