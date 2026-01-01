"""
Test AOV on EXISTING material with FORCED shader update.
"""

import bpy

def test_aov_force_update():
    print("\n" + "="*60)
    print("TEST: AOV on EXISTING material with FORCED update")
    print("="*60)
    
    view_layer = bpy.context.view_layer
    aovs = view_layer.aovs
    
    AOV_NAME = "Test_Force_AOV"
    
    # Remove and recreate AOV
    for existing in list(aovs):
        if existing.name == AOV_NAME:
            aovs.remove(existing)
    
    aov = aovs.add()
    aov.name = AOV_NAME
    aov.type = 'COLOR'
    print(f"Created AOV '{AOV_NAME}'")
    
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
    
    # === FORCE SHADER RECOMPILE ===
    print("Forcing shader recompile...")
    
    # Method 1: Update the node tree
    mat.node_tree.update_tag()
    
    # Method 2: Force depsgraph update
    bpy.context.view_layer.update()
    
    # Method 3: Tag material for update
    mat.update_tag()
    
    # Method 4: Force scene update
    bpy.context.scene.frame_set(bpy.context.scene.frame_current)
    
    print("Shader update forced")
    print(f"AOV node aov_name = '{aov_node.aov_name}'")
    
    # Verify node still exists after toggle
    if "Test_AOV" in mat.node_tree.nodes:
        print("AOV node still exists after update")
    else:
        print("WARNING: AOV node was lost!")
    
    print("\n" + "="*60)
    print("NOW: Render (F12) and check for 'Test_Force_AOV'")
    print("="*60)

test_aov_force_update()
