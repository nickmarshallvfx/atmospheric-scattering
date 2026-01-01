"""
Test: Isolate which atmosphere setting breaks AOVs.
Run BEFORE creating atmosphere in a fresh scene.
"""

import bpy

def test_each_setting():
    print("\n" + "="*60)
    print("TEST: Isolating atmosphere settings")
    print("="*60)
    
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    
    # Get material
    obj = bpy.context.active_object
    if not obj or obj.type != 'MESH' or len(obj.data.materials) == 0:
        print("ERROR: Select an object with a material!")
        return
    mat = obj.data.materials[0]
    
    # Create test AOV
    AOV_NAME = "Test_Isolate_AOV"
    aovs = view_layer.aovs
    for existing in list(aovs):
        if existing.name == AOV_NAME:
            aovs.remove(existing)
    aov = aovs.add()
    aov.name = AOV_NAME
    aov.type = 'COLOR'
    print(f"Created AOV: {AOV_NAME}")
    
    # Create nodes
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    for name in ["Test_Geom", "Test_AOV"]:
        if name in nodes:
            nodes.remove(nodes[name])
    
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.name = "Test_Geom"
    geom.location = (600, 200)
    
    aov_node = nodes.new('ShaderNodeOutputAOV')
    aov_node.name = "Test_AOV"
    aov_node.location = (800, 200)
    aov_node.aov_name = AOV_NAME
    
    links.new(geom.outputs['Position'], aov_node.inputs['Color'])
    
    # Force update
    mat.node_tree.update_tag()
    mat.update_tag()
    view_layer.update()
    
    print("AOV nodes created.")
    print("\n--- NOW TESTING EACH SETTING ---")
    print("Render (F12) after EACH step to see which breaks it:\n")
    
    print("Step 1: RENDER NOW to confirm AOV works")
    print("Step 2: Run: bpy.context.scene.cycles.shading_system = 'OSL'")
    print("        Then render to see if OSL breaks it")
    print("Step 3: Run: bpy.context.scene.render.film_transparent = True")
    print("        Then render to see if film_transparent breaks it")
    print("Step 4: Run: bpy.context.view_layer.use_pass_environment = True")
    print("        Then render to see if environment pass breaks it")
    
    print("\n" + "="*60)

test_each_setting()
