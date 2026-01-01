"""
Test: Create NEW material AFTER OSL is enabled.
Run this while OSL is already enabled.
"""

import bpy

def test_new_material_with_osl():
    print("\n" + "="*60)
    print("TEST: NEW material with OSL enabled")
    print("="*60)
    
    # Verify OSL is enabled
    osl_enabled = bpy.context.scene.cycles.shading_system
    print(f"OSL enabled: {osl_enabled}")
    
    if not osl_enabled:
        print("Enabling OSL...")
        bpy.context.scene.cycles.shading_system = True
    
    view_layer = bpy.context.view_layer
    aovs = view_layer.aovs
    
    AOV_NAME = "Test_OSL_AOV"
    
    # Remove and recreate AOV
    for existing in list(aovs):
        if existing.name == AOV_NAME:
            aovs.remove(existing)
    
    aov = aovs.add()
    aov.name = AOV_NAME
    aov.type = 'COLOR'
    print(f"Created AOV: {AOV_NAME}")
    
    # Create NEW material
    mat = bpy.data.materials.new(name="Test_OSL_Material")
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create Principled BSDF
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (0, 0)
    
    # Create Material Output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    
    # Connect
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    # Create Geometry + AOV
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.location = (0, -200)
    
    aov_node = nodes.new('ShaderNodeOutputAOV')
    aov_node.location = (300, -200)
    aov_node.aov_name = AOV_NAME
    
    links.new(geom.outputs['Position'], aov_node.inputs['Color'])
    
    # Force update
    mat.node_tree.update_tag()
    mat.update_tag()
    view_layer.update()
    
    # Apply to active object
    obj = bpy.context.active_object
    if obj and obj.type == 'MESH':
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat
        print(f"Applied new material to {obj.name}")
    
    print("\n" + "="*60)
    print("NOW: Render (F12) and check 'Test_OSL_AOV'")
    print("="*60)

test_new_material_with_osl()
