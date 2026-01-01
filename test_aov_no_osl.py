"""
Test: Use addon's AOV setup but WITHOUT OSL node.
Isolates if OSL node creation is breaking AOVs.
"""

import bpy
import sys
import os

# Add helios to path
addon_path = r"C:\Program Files\Blender Foundation\Blender 5.0\5.0\scripts\addons_core\helios"
if addon_path not in sys.path:
    sys.path.insert(0, os.path.dirname(addon_path))

from helios import aerial

def test_aov_no_osl():
    print("\n" + "="*60)
    print("TEST: Addon AOV setup WITHOUT OSL node")
    print("="*60)
    
    context = bpy.context
    
    # 1. Use addon's AOV setup function
    print("\n--- Step 1: aerial.setup_aerial_aovs ---")
    aerial.setup_aerial_aovs(context)
    
    # 2. Get material
    obj = context.active_object
    if not obj or obj.type != 'MESH' or len(obj.data.materials) == 0:
        print("ERROR: Select an object with a material!")
        return
    
    mat = obj.data.materials[0]
    print(f"\n--- Step 2: Manually create AOV nodes in '{mat.name}' (no OSL) ---")
    
    if not mat.use_nodes:
        mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Remove any existing test nodes
    for name in ["Helios_Geom", "Helios_AOV_Transmittance", "Helios_AOV_Inscatter"]:
        if name in nodes:
            nodes.remove(nodes[name])
    
    # Create Geometry node (same as addon)
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.name = "Helios_Geom"
    geom.location = (700, 0)
    
    # Create AOV nodes (same as addon)
    aov_trans = nodes.new('ShaderNodeOutputAOV')
    aov_trans.name = "Helios_AOV_Transmittance"
    aov_trans.location = (900, 50)
    aov_trans.aov_name = aerial.AERIAL_AOV_TRANSMITTANCE
    
    aov_inscatter = nodes.new('ShaderNodeOutputAOV')
    aov_inscatter.name = "Helios_AOV_Inscatter"
    aov_inscatter.location = (900, -100)
    aov_inscatter.aov_name = aerial.AERIAL_AOV_INSCATTER
    
    # Connect (same as addon)
    links.new(geom.outputs['Position'], aov_trans.inputs[0])
    links.new(geom.outputs['Normal'], aov_inscatter.inputs[0])
    
    print(f"Created nodes with aov_name: '{aov_trans.aov_name}', '{aov_inscatter.aov_name}'")
    
    # 3. Force update (same as addon)
    print("\n--- Step 3: Force update ---")
    mat.node_tree.update_tag()
    mat.update_tag()
    context.view_layer.update()
    
    print("\n" + "="*60)
    print("NOW: Render (F12) and check Helios_Transmittance / Helios_Inscatter")
    print("="*60)

test_aov_no_osl()
