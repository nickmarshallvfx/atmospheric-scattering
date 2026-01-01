"""
Minimal AOV test script for Blender 5.0
Run this in the Blender scripting editor to test if AOVs work at all.

This creates a simple material with an AOV Output node connected to a Geometry node,
registers the AOV in the view layer, and prints diagnostic info.
"""

import bpy

def test_aov():
    print("\n" + "="*60)
    print("BLENDER 5.0 AOV TEST")
    print("="*60)
    
    # Get context
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    
    # Print Blender version
    print(f"Blender version: {bpy.app.version_string}")
    
    # 1. Create/get test AOV in view layer
    AOV_NAME = "Test_Position_AOV"
    
    print(f"\n1. Registering AOV '{AOV_NAME}' in view layer...")
    
    # Check existing AOVs
    print(f"   Existing AOVs: {[aov.name for aov in view_layer.aovs]}")
    
    # Remove existing test AOV if present
    for aov in view_layer.aovs:
        if aov.name == AOV_NAME:
            view_layer.aovs.remove(aov)
            print(f"   Removed existing AOV '{AOV_NAME}'")
            break
    
    # Add new AOV
    aov = view_layer.aovs.add()
    aov.name = AOV_NAME
    aov.type = 'COLOR'
    print(f"   Created AOV: name='{aov.name}', type='{aov.type}'")
    
    # Verify it was added
    print(f"   All AOVs now: {[a.name for a in view_layer.aovs]}")
    
    # 2. Create a test material
    print(f"\n2. Creating test material...")
    
    mat_name = "AOV_Test_Material"
    if mat_name in bpy.data.materials:
        mat = bpy.data.materials[mat_name]
        mat.node_tree.nodes.clear()
    else:
        mat = bpy.data.materials.new(name=mat_name)
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear existing nodes
    nodes.clear()
    
    # Create Principled BSDF
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    
    # Create Material Output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    # Create Geometry node
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.location = (0, -200)
    
    # Create AOV Output node
    aov_node = nodes.new('ShaderNodeOutputAOV')
    aov_node.location = (300, -200)
    aov_node.aov_name = AOV_NAME
    
    # Connect geometry position to AOV
    links.new(geom.outputs['Position'], aov_node.inputs['Color'])
    
    print(f"   Created material '{mat_name}'")
    print(f"   AOV Output node aov_name: '{aov_node.aov_name}'")
    print(f"   AOV Output node inputs: {[i.name for i in aov_node.inputs]}")
    print(f"   Connected: Geometry.Position -> AOV.Color")
    
    # 3. Apply material to active object
    print(f"\n3. Applying material to active object...")
    
    obj = bpy.context.active_object
    if obj and obj.type == 'MESH':
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        else:
            obj.data.materials[0] = mat
        print(f"   Applied '{mat_name}' to '{obj.name}'")
    else:
        print(f"   WARNING: No active mesh object! Select a mesh and re-run.")
    
    # 4. Check render settings
    print(f"\n4. Render settings:")
    print(f"   Engine: {scene.render.engine}")
    print(f"   Film transparent: {scene.render.film_transparent}")
    print(f"   Use nodes (compositor): {scene.use_nodes}")
    
    # 5. Instructions
    print(f"\n" + "="*60)
    print("NEXT STEPS:")
    print("="*60)
    print("1. Render the scene (F12)")
    print("2. In Image Editor, check if AOV appears in layer dropdown")
    print("3. Or in Compositor, check Render Layers node for AOV output")
    print(f"\nLooking for AOV named: '{AOV_NAME}'")
    print("It should show object world position as RGB color")
    print("="*60 + "\n")

# Run the test
test_aov()
