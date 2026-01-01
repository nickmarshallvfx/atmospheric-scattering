"""
Test: Call the exact same addon functions as a standalone script.
This isolates whether the issue is with operator context.
"""

import bpy
import sys
import os

# Add helios to path
addon_path = r"C:\Program Files\Blender Foundation\Blender 5.0\5.0\scripts\addons_core\helios"
if addon_path not in sys.path:
    sys.path.insert(0, os.path.dirname(addon_path))

# Import addon modules
from helios import aerial

def test_addon_functions():
    print("\n" + "="*60)
    print("TEST: Calling addon functions directly")
    print("="*60)
    
    context = bpy.context
    
    # 1. Setup AOVs (like operator does)
    print("\n--- Step 1: setup_aerial_aovs ---")
    aerial.setup_aerial_aovs(context)
    
    # 2. Get ONE material from active object (simplified test)
    obj = context.active_object
    if not obj or obj.type != 'MESH' or len(obj.data.materials) == 0:
        print("ERROR: Select an object with a material!")
        return
    
    mat = obj.data.materials[0]
    print(f"\n--- Step 2: add_aerial_to_material('{mat.name}') ---")
    
    # 3. Call add_aerial_to_material
    result = aerial.add_aerial_to_material(mat, context)
    print(f"Result: {result}")
    
    # 4. Force view layer update (like operator does)
    print("\n--- Step 3: view_layer.update() ---")
    context.view_layer.update()
    
    # 5. Verify AOVs exist
    print("\n--- Verification ---")
    aovs = context.view_layer.aovs
    print(f"View layer AOVs: {[(a.name, a.type) for a in aovs]}")
    
    # Check material nodes
    if mat.use_nodes:
        nodes = mat.node_tree.nodes
        for node in nodes:
            if 'AOV' in node.name or 'Helios' in node.name:
                print(f"Node: {node.name} ({node.bl_idname})")
                if hasattr(node, 'aov_name'):
                    print(f"  aov_name: {node.aov_name}")
                for inp in node.inputs:
                    if inp.links:
                        print(f"  Input '{inp.name}' linked from: {[l.from_node.name for l in inp.links]}")
    
    print("\n" + "="*60)
    print("NOW: Render (F12) and check Helios_Transmittance / Helios_Inscatter")
    print("="*60)

test_addon_functions()
