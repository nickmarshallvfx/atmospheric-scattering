"""
Standardized Aerial Perspective Test Scene Generator

Creates a simple scene with objects at known distances for validating
the aerial perspective implementation step by step.

Run in Blender: 
    File > Open > (this script location)
    Then run from Text Editor

Or via command line:
    blender --python create_test_scene.py
"""

import bpy
import math

def clear_scene():
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def create_material(name, color):
    """Create a simple diffuse material."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear default nodes
    nodes.clear()
    
    # Create nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    
    diffuse = nodes.new('ShaderNodeBsdfDiffuse')
    diffuse.location = (0, 0)
    diffuse.inputs['Color'].default_value = (*color, 1.0)
    
    links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    
    return mat

def create_test_objects():
    """Create the standardized test objects."""
    
    # Materials
    white_mat = create_material("Test_White", (0.8, 0.8, 0.8))
    ground_mat = create_material("Test_Ground", (0.3, 0.35, 0.3))
    
    objects_created = []
    
    # Near cubes (500m distance)
    for name, loc in [("Near_L", (-353, 353, 50)), ("Near_R", (353, 353, 50))]:
        bpy.ops.mesh.primitive_cube_add(size=100, location=loc)
        obj = bpy.context.active_object
        obj.name = name
        obj.data.materials.append(white_mat)
        objects_created.append((name, loc, 500))
    
    # Mid spheres (5km distance)
    for name, loc in [("Mid_L", (-3536, 3536, 100)), ("Mid_R", (3536, 3536, 100))]:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=100, location=loc, segments=32, ring_count=16)
        obj = bpy.context.active_object
        obj.name = name
        obj.data.materials.append(white_mat)
        objects_created.append((name, loc, 5000))
    
    # Far spheres (20km distance)
    for name, loc in [("Far_L", (-14142, 14142, 250)), ("Far_R", (14142, 14142, 250))]:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=250, location=loc, segments=32, ring_count=16)
        obj = bpy.context.active_object
        obj.name = name
        obj.data.materials.append(white_mat)
        objects_created.append((name, loc, 20000))
    
    # Ground plane (50km x 50km)
    bpy.ops.mesh.primitive_plane_add(size=50000, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.data.materials.append(ground_mat)
    objects_created.append(("Ground", (0, 0, 0), 0))
    
    return objects_created

def setup_camera():
    """Set up the camera at 500m altitude looking north."""
    # Create camera
    bpy.ops.object.camera_add(location=(0, 0, 500))
    camera = bpy.context.active_object
    camera.name = "TestCamera"
    
    # Point camera north (toward +Y) with slight downward tilt
    # Rotation: X=80° (looking slightly down), Y=0°, Z=0° (facing +Y)
    camera.rotation_euler = (math.radians(80), 0, 0)
    
    # Camera settings
    camera.data.lens = 35  # 35mm focal length
    camera.data.sensor_width = 36  # Full frame
    
    # Set as active camera
    bpy.context.scene.camera = camera
    
    return camera

def setup_sun():
    """Set up the sun at 45° elevation from the east."""
    # Create sun light
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 1000))
    sun = bpy.context.active_object
    sun.name = "TestSun"
    
    # Sun from east at 45° elevation
    # Rotation: X=45° (elevation), Y=0°, Z=-90° (from east)
    sun.rotation_euler = (math.radians(45), 0, math.radians(-90))
    
    # Sun strength
    sun.data.energy = 5.0
    
    return sun

def setup_render_settings():
    """Configure render settings for testing."""
    scene = bpy.context.scene
    
    # Render settings
    scene.render.engine = 'CYCLES'
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    
    # Cycles settings
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    
    # Film
    scene.render.film_transparent = False
    
    # Color management
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'

def add_scene_info_text():
    """Add text object with scene information."""
    info_text = """AERIAL PERSPECTIVE TEST SCENE
==============================
Camera: (0, 0, 500m) - 500m altitude
Sun: 45° elevation, from East

Objects:
- Near cubes: 500m distance
- Mid spheres: 5km distance  
- Far spheres: 20km distance
- Ground: 50km plane

Expected with aerial perspective:
- Near: minimal haze
- Mid: visible haze
- Far: heavy haze, blending with sky
"""
    
    # Store in scene custom properties for reference
    bpy.context.scene["test_scene_info"] = info_text
    
    print(info_text)

def main():
    """Main function to create the test scene."""
    print("\n" + "="*50)
    print("Creating Aerial Perspective Test Scene")
    print("="*50 + "\n")
    
    # Clear existing
    clear_scene()
    
    # Create scene elements
    objects = create_test_objects()
    camera = setup_camera()
    sun = setup_sun()
    
    # Configure render
    setup_render_settings()
    
    # Add info
    add_scene_info_text()
    
    # Summary
    print("\nObjects created:")
    for name, loc, dist in objects:
        print(f"  {name}: location={loc}, distance={dist}m")
    
    print(f"\nCamera: {camera.location}")
    print(f"Sun elevation: 45°, azimuth: East")
    
    print("\n" + "="*50)
    print("Test scene created successfully!")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
