"""
Test AOV on EXISTING material (like the addon does).
Run this in Blender after selecting an object with a material.
"""

import bpy

def test_aov_on_existing():
    print("\n" + "="*60)
    print("TEST: AOV on EXISTING material")
    print("="*60)
    
    view_layer = bpy.context.view_layer
    aovs = view_layer.aovs
    
    AOV_NAME = "Test_Existing_AOV"
    
    # Remove and recreate AOV
    for existing in list(aovs):
        if existing.name == AOV_NAME:
            aovs.remove(existing)
            print(f"Removed old AOV '{AOV_NAME}'")
    
    aov = aovs.add()
    aov.name = AOV_NAME
    aov.type = 'COLOR'
    print(f"Created AOV '{AOV_NAME}'")
    print(f"All AOVs: {[a.name for a in aovs]}")
    
    # Get EXISTING material from active object
    obj = bpy.context.active_object
    if not obj or obj.type != 'MESH' or len(obj.data.materials) == 0:
        print("ERROR: Select an object with a material first!")
        return
    
    mat = obj.data.materials[0]
    print(f"Using existing material: '{mat.name}'")
    
    if not mat.use_nodes:
        mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # DON'T clear nodes - add to existing (like addon does)
    
    # Remove any old test nodes
    for node_name in ["Test_Geom", "Test_AOV"]:
        if node_name in nodes:
            nodes.remove(nodes[node_name])
    
    # Create Geometry node
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.name = "Test_Geom"
    geom.location = (600, 200)
    
    # Create AOV Output
    aov_node = nodes.new('ShaderNodeOutputAOV')
    aov_node.name = "Test_AOV"
    aov_node.location = (800, 200)
    aov_node.aov_name = AOV_NAME
    
    # Connect
    links.new(geom.outputs['Position'], aov_node.inputs['Color'])
    
    print(f"Added nodes to '{mat.name}'")
    print(f"AOV node aov_name = '{aov_node.aov_name}'")
    print(f"AOV input links: {[l.from_node.name for l in aov_node.inputs['Color'].links]}")
    
    print("\n" + "="*60)
    print("NOW: Render (F12) and check for 'Test_Existing_AOV'")
    print("="*60)

test_aov_on_existing()
