"""
Helios Aerial Perspective - Step-by-Step Test Implementations

Each function creates a minimal node group for testing one specific step
of the Bruneton aerial perspective implementation.

Test approach: Output intermediate values to Emission shader for validation.
Reference: AERIAL_IMPLEMENTATION_STRATEGY.md
"""

import bpy
import math


# =============================================================================
# CONSTANTS
# =============================================================================

BOTTOM_RADIUS = 6360.0  # km (Earth surface)
TOP_RADIUS = 6420.0     # km (top of atmosphere)
H = math.sqrt(TOP_RADIUS * TOP_RADIUS - BOTTOM_RADIUS * BOTTOM_RADIUS)  # ~797.66 km

# Test scene reference values (from TEST_SCENE_REFERENCE.md)
TEST_SCENE_MAX_DISTANCE_KM = 10.645  # Maximum distance in test scene


# =============================================================================
# STEP 1.1: DISTANCE CALCULATION
# =============================================================================

def create_step_1_1_distance_test(max_distance_km=TEST_SCENE_MAX_DISTANCE_KM):
    """
    Step 1.1: Distance Calculation Test
    
    Creates a node group that outputs distance from camera to surface point,
    normalized by max_distance_km for visualization.
    
    Output: Emission color where R = distance_km / max_distance_km
    
    Expected results (from TEST_SCENE_REFERENCE.md):
    - Near geometry (~200m): R ≈ 0.02
    - Mid geometry (~700m): R ≈ 0.07  
    - Far geometry (~10km): R ≈ 1.0
    - Sky: Should be black (no geometry)
    """
    
    group_name = "Helios_Test_Step_1_1_Distance"
    
    # Remove existing group if present
    if group_name in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups[group_name])
    
    # Create new node group
    group = bpy.data.node_groups.new(group_name, 'ShaderNodeTree')
    nodes = group.nodes
    links = group.links
    
    # Create interface (inputs/outputs)
    group.interface.new_socket(name="Shader", in_out='OUTPUT', socket_type='NodeSocketShader')
    
    # --- Input: World Position ---
    geometry = nodes.new('ShaderNodeNewGeometry')
    geometry.location = (0, 0)
    geometry.name = "Geometry"
    
    # --- Input: Camera Position ---
    # Camera position comes from the scene camera
    camera_pos = nodes.new('ShaderNodeCombineXYZ')
    camera_pos.location = (0, -200)
    camera_pos.name = "Camera_Position"
    camera_pos.label = "Camera Position (set via driver or manually)"
    # Default to test scene camera position
    camera_pos.inputs['X'].default_value = -3.5
    camera_pos.inputs['Y'].default_value = -8.5
    camera_pos.inputs['Z'].default_value = 218.8
    
    # --- Calculate: world_pos - camera_pos ---
    subtract = nodes.new('ShaderNodeVectorMath')
    subtract.operation = 'SUBTRACT'
    subtract.location = (200, 0)
    subtract.name = "world_minus_camera"
    subtract.label = "World - Camera"
    links.new(geometry.outputs['Position'], subtract.inputs[0])
    links.new(camera_pos.outputs['Vector'], subtract.inputs[1])
    
    # --- Calculate: distance = length(world_pos - camera_pos) ---
    length = nodes.new('ShaderNodeVectorMath')
    length.operation = 'LENGTH'
    length.location = (400, 0)
    length.name = "distance_m"
    length.label = "Distance (meters)"
    links.new(subtract.outputs['Vector'], length.inputs[0])
    
    # --- Convert to km: distance_km = distance_m * 0.001 ---
    to_km = nodes.new('ShaderNodeMath')
    to_km.operation = 'MULTIPLY'
    to_km.location = (600, 0)
    to_km.name = "distance_km"
    to_km.label = "Distance (km)"
    to_km.inputs[1].default_value = 0.001
    links.new(length.outputs['Value'], to_km.inputs[0])
    
    # --- Normalize: distance_normalized = distance_km / max_distance_km ---
    normalize = nodes.new('ShaderNodeMath')
    normalize.operation = 'DIVIDE'
    normalize.location = (800, 0)
    normalize.name = "distance_normalized"
    normalize.label = f"Distance / {max_distance_km}km"
    normalize.inputs[1].default_value = max_distance_km
    links.new(to_km.outputs['Value'], normalize.inputs[0])
    
    # --- Clamp to 0-1 for visualization ---
    clamp = nodes.new('ShaderNodeClamp')
    clamp.location = (1000, 0)
    clamp.name = "distance_clamped"
    clamp.label = "Clamped 0-1"
    links.new(normalize.outputs['Value'], clamp.inputs['Value'])
    
    # --- Output as grayscale emission ---
    combine = nodes.new('ShaderNodeCombineXYZ')
    combine.location = (1200, 0)
    combine.name = "to_color"
    combine.label = "To RGB"
    links.new(clamp.outputs['Result'], combine.inputs['X'])
    links.new(clamp.outputs['Result'], combine.inputs['Y'])
    links.new(clamp.outputs['Result'], combine.inputs['Z'])
    
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (1400, 0)
    emission.name = "emission"
    emission.inputs['Strength'].default_value = 1.0
    links.new(combine.outputs['Vector'], emission.inputs['Color'])
    
    # --- Group Output ---
    output = nodes.new('NodeGroupOutput')
    output.location = (1600, 0)
    links.new(emission.outputs['Emission'], output.inputs['Shader'])
    
    print(f"Created node group: {group_name}")
    print(f"  Max distance: {max_distance_km} km")
    print(f"  Camera position: (-3.5, -8.5, 218.8) meters")
    print(f"")
    print(f"Expected output (grayscale):")
    print(f"  Near (~200m): ~0.02 (dark)")
    print(f"  Mid (~700m):  ~0.07")
    print(f"  Far (~10km):  ~1.0 (white)")
    print(f"  Sky: black")
    
    return group


def apply_step_1_1_to_material(material_name=None):
    """
    Apply Step 1.1 test node group to a material.
    
    If material_name is None, creates a new material called "Helios_Test_Step_1_1".
    """
    
    group = create_step_1_1_distance_test()
    
    # Get or create material
    if material_name and material_name in bpy.data.materials:
        mat = bpy.data.materials[material_name]
    else:
        mat = bpy.data.materials.new(name="Helios_Test_Step_1_1")
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear existing nodes
    nodes.clear()
    
    # Add group node
    group_node = nodes.new('ShaderNodeGroup')
    group_node.node_tree = group
    group_node.location = (0, 0)
    group_node.name = "Step_1_1_Distance"
    
    # Add output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    
    # Connect
    links.new(group_node.outputs['Shader'], output.inputs['Surface'])
    
    print(f"\nApplied to material: {mat.name}")
    print(f"Assign this material to objects in your test scene to visualize distance.")
    
    return mat


# =============================================================================
# UTILITY: Update camera position from scene camera
# =============================================================================

def update_camera_position_in_group(group_name="Helios_Test_Step_1_1_Distance"):
    """Update the camera position in the test node group from the active scene camera."""
    
    if group_name not in bpy.data.node_groups:
        print(f"ERROR: Node group '{group_name}' not found")
        return
    
    camera = bpy.context.scene.camera
    if not camera:
        print("ERROR: No active camera in scene")
        return
    
    group = bpy.data.node_groups[group_name]
    
    # Find camera position node
    cam_node = group.nodes.get("Camera_Position")
    if not cam_node:
        print("ERROR: Camera_Position node not found in group")
        return
    
    # Update position
    loc = camera.location
    cam_node.inputs['X'].default_value = loc.x
    cam_node.inputs['Y'].default_value = loc.y
    cam_node.inputs['Z'].default_value = loc.z
    
    print(f"Updated camera position to: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")


# =============================================================================
# STEP 1.2: r, mu CALCULATION
# =============================================================================

def create_step_1_2_r_mu_test():
    """
    Step 1.2: r, mu Calculation Test
    
    Computes Bruneton's (r, mu) parameters:
    - r = distance from planet center to camera (km) - CONSTANT
    - mu = cos(zenith angle of view ray) = dot(view_dir, up_at_camera)
    
    Reference: functions.glsl lines 1797-1815
        Length r = length(camera);
        Length rmu = dot(camera, view_ray);
        Number mu = rmu / r;
    
    Output: 
    - R channel = r / 6400 (normalized, should be ~0.994 for 6m altitude)
    - G channel = (mu + 1) / 2 (remapped from [-1,1] to [0,1])
    
    Expected results:
    - R should be constant (~0.9937) since r only depends on camera position
    - G should vary across frame: 0.5 = horizontal, >0.5 = looking up, <0.5 = looking down
    """
    
    group_name = "Helios_Test_Step_1_2_r_mu"
    
    # Remove existing group if present
    if group_name in bpy.data.node_groups:
        bpy.data.node_groups.remove(bpy.data.node_groups[group_name])
    
    # Create new node group
    group = bpy.data.node_groups.new(group_name, 'ShaderNodeTree')
    nodes = group.nodes
    links = group.links
    
    # Create interface
    group.interface.new_socket(name="Shader", in_out='OUTPUT', socket_type='NodeSocketShader')
    
    # --- Constants ---
    BOTTOM_RADIUS_KM = 6360.0
    
    # Camera position (meters)
    CAM_X, CAM_Y, CAM_Z = 37.069, -44.786, 6.0
    
    # Pre-compute r (constant for all pixels)
    # r = distance from planet center to camera
    # Planet center is at (0, 0, -6360km) in Blender coords
    # Camera relative to planet center (in km):
    cam_rel_x_km = CAM_X * 0.001
    cam_rel_y_km = CAM_Y * 0.001
    cam_rel_z_km = (CAM_Z * 0.001) + BOTTOM_RADIUS_KM  # Z offset from planet center
    
    r_km = math.sqrt(cam_rel_x_km**2 + cam_rel_y_km**2 + cam_rel_z_km**2)
    r_normalized = r_km / 6400.0
    
    # Pre-compute up direction at camera (normalized)
    up_length = r_km
    up_x = cam_rel_x_km / up_length
    up_y = cam_rel_y_km / up_length
    up_z = cam_rel_z_km / up_length
    
    print(f"Pre-computed values:")
    print(f"  Camera relative to planet center (km): ({cam_rel_x_km:.6f}, {cam_rel_y_km:.6f}, {cam_rel_z_km:.6f})")
    print(f"  r = {r_km:.6f} km")
    print(f"  r_normalized = {r_normalized:.6f}")
    print(f"  up_at_camera = ({up_x:.6f}, {up_y:.6f}, {up_z:.6f})")
    
    # --- Input: World Position ---
    geometry = nodes.new('ShaderNodeNewGeometry')
    geometry.location = (0, 200)
    geometry.name = "Geometry"
    
    # --- Input: Camera Position (meters) ---
    camera_pos = nodes.new('ShaderNodeCombineXYZ')
    camera_pos.location = (0, 0)
    camera_pos.name = "Camera_Position"
    camera_pos.label = "Camera Position (meters)"
    camera_pos.inputs['X'].default_value = CAM_X
    camera_pos.inputs['Y'].default_value = CAM_Y
    camera_pos.inputs['Z'].default_value = CAM_Z
    
    # --- r as constant (pre-computed) ---
    r_const = nodes.new('ShaderNodeValue')
    r_const.location = (600, 0)
    r_const.name = "r_constant"
    r_const.label = f"r = {r_km:.4f} km"
    r_const.outputs['Value'].default_value = r_normalized  # Already normalized
    
    # --- Up direction at camera (pre-computed, constant) ---
    up_at_cam = nodes.new('ShaderNodeCombineXYZ')
    up_at_cam.location = (400, -200)
    up_at_cam.name = "up_at_camera"
    up_at_cam.label = "Up at Camera (normalized)"
    up_at_cam.inputs['X'].default_value = up_x
    up_at_cam.inputs['Y'].default_value = up_y
    up_at_cam.inputs['Z'].default_value = up_z
    
    # --- View direction = normalize(world_pos - camera_pos) ---
    view_vec = nodes.new('ShaderNodeVectorMath')
    view_vec.operation = 'SUBTRACT'
    view_vec.location = (200, 400)
    view_vec.name = "view_vec"
    view_vec.label = "World - Camera"
    links.new(geometry.outputs['Position'], view_vec.inputs[0])
    links.new(camera_pos.outputs['Vector'], view_vec.inputs[1])
    
    view_dir = nodes.new('ShaderNodeVectorMath')
    view_dir.operation = 'NORMALIZE'
    view_dir.location = (400, 400)
    view_dir.name = "view_dir"
    view_dir.label = "View Direction"
    links.new(view_vec.outputs['Vector'], view_dir.inputs[0])
    
    # --- mu = dot(view_dir, up_at_camera) ---
    mu_node = nodes.new('ShaderNodeVectorMath')
    mu_node.operation = 'DOT_PRODUCT'
    mu_node.location = (600, 200)
    mu_node.name = "mu"
    mu_node.label = "mu = dot(view, up)"
    links.new(view_dir.outputs['Vector'], mu_node.inputs[0])
    links.new(up_at_cam.outputs['Vector'], mu_node.inputs[1])
    
    # --- Remap mu from [-1, 1] to [0, 1]: (mu + 1) / 2 ---
    mu_add = nodes.new('ShaderNodeMath')
    mu_add.operation = 'ADD'
    mu_add.location = (800, 200)
    mu_add.name = "mu_plus_1"
    mu_add.inputs[1].default_value = 1.0
    links.new(mu_node.outputs['Value'], mu_add.inputs[0])
    
    mu_norm = nodes.new('ShaderNodeMath')
    mu_norm.operation = 'DIVIDE'
    mu_norm.location = (1000, 200)
    mu_norm.name = "mu_normalized"
    mu_norm.label = "(mu + 1) / 2"
    mu_norm.inputs[1].default_value = 2.0
    links.new(mu_add.outputs['Value'], mu_norm.inputs[0])
    
    # --- Combine to RGB output ---
    # R = r_normalized (constant), G = mu_normalized, B = 0
    combine = nodes.new('ShaderNodeCombineXYZ')
    combine.location = (1200, 100)
    combine.name = "to_color"
    combine.label = "R=r, G=mu, B=0"
    combine.inputs['Z'].default_value = 0.0
    links.new(r_const.outputs['Value'], combine.inputs['X'])  # Use pre-computed r
    links.new(mu_norm.outputs['Value'], combine.inputs['Y'])
    
    # --- Emission output ---
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (1600, 100)
    emission.name = "emission"
    emission.inputs['Strength'].default_value = 1.0
    links.new(combine.outputs['Vector'], emission.inputs['Color'])
    
    # --- Group Output ---
    output = nodes.new('NodeGroupOutput')
    output.location = (1800, 100)
    links.new(emission.outputs['Emission'], output.inputs['Shader'])
    
    print(f"Created node group: {group_name}")
    print(f"\nOutput encoding:")
    print(f"  R = r / 6400 = {r_normalized:.6f} (CONSTANT)")
    print(f"  G = (mu + 1) / 2 (0.5 = horizontal, >0.5 = up, <0.5 = down)")
    print(f"  B = 0")
    
    return group


def apply_step_1_2_to_material_direct():
    """
    Create Step 1.2 material with nodes directly (no node group).
    This avoids any potential node group caching issues.
    """
    import time
    
    # Constants
    BOTTOM_RADIUS_KM = 6360.0
    CAM_X, CAM_Y, CAM_Z = 37.069, -44.786, 6.0
    
    # Pre-compute r and up
    cam_rel_x_km = CAM_X * 0.001
    cam_rel_y_km = CAM_Y * 0.001
    cam_rel_z_km = (CAM_Z * 0.001) + BOTTOM_RADIUS_KM
    
    r_km = math.sqrt(cam_rel_x_km**2 + cam_rel_y_km**2 + cam_rel_z_km**2)
    r_normalized = r_km / 6400.0
    
    up_x = cam_rel_x_km / r_km
    up_y = cam_rel_y_km / r_km
    up_z = cam_rel_z_km / r_km
    
    print(f"Pre-computed: r = {r_km:.6f} km, r_norm = {r_normalized:.6f}")
    print(f"up = ({up_x:.6f}, {up_y:.6f}, {up_z:.6f})")
    
    # Create material
    mat = bpy.data.materials.new(name=f"Step1_2_Direct_{int(time.time())}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Geometry input
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.location = (0, 300)
    
    # Camera position
    cam_pos = nodes.new('ShaderNodeCombineXYZ')
    cam_pos.location = (0, 100)
    cam_pos.inputs['X'].default_value = CAM_X
    cam_pos.inputs['Y'].default_value = CAM_Y
    cam_pos.inputs['Z'].default_value = CAM_Z
    
    # Constant r (pre-computed)
    r_const = nodes.new('ShaderNodeValue')
    r_const.location = (0, -100)
    r_const.outputs['Value'].default_value = r_normalized
    r_const.label = f"r_norm = {r_normalized:.4f}"
    
    # Up direction (pre-computed)
    up_vec = nodes.new('ShaderNodeCombineXYZ')
    up_vec.location = (0, -300)
    up_vec.inputs['X'].default_value = up_x
    up_vec.inputs['Y'].default_value = up_y
    up_vec.inputs['Z'].default_value = up_z
    up_vec.label = "up_at_camera"
    
    # View vector = world_pos - camera
    view_sub = nodes.new('ShaderNodeVectorMath')
    view_sub.operation = 'SUBTRACT'
    view_sub.location = (200, 300)
    links.new(geom.outputs['Position'], view_sub.inputs[0])
    links.new(cam_pos.outputs['Vector'], view_sub.inputs[1])
    
    # Normalize view
    view_norm = nodes.new('ShaderNodeVectorMath')
    view_norm.operation = 'NORMALIZE'
    view_norm.location = (400, 300)
    links.new(view_sub.outputs['Vector'], view_norm.inputs[0])
    
    # mu = dot(view, up)
    mu_dot = nodes.new('ShaderNodeVectorMath')
    mu_dot.operation = 'DOT_PRODUCT'
    mu_dot.location = (600, 200)
    links.new(view_norm.outputs['Vector'], mu_dot.inputs[0])
    links.new(up_vec.outputs['Vector'], mu_dot.inputs[1])
    
    # mu_remapped = (mu + 1) / 2
    mu_add = nodes.new('ShaderNodeMath')
    mu_add.operation = 'ADD'
    mu_add.location = (800, 200)
    mu_add.inputs[1].default_value = 1.0
    links.new(mu_dot.outputs['Value'], mu_add.inputs[0])
    
    mu_div = nodes.new('ShaderNodeMath')
    mu_div.operation = 'DIVIDE'
    mu_div.location = (1000, 200)
    mu_div.inputs[1].default_value = 2.0
    links.new(mu_add.outputs['Value'], mu_div.inputs[0])
    
    # Combine: R=r_const, G=mu_remapped, B=0
    combine = nodes.new('ShaderNodeCombineXYZ')
    combine.location = (1200, 100)
    combine.inputs['Z'].default_value = 0.0
    links.new(r_const.outputs['Value'], combine.inputs['X'])
    links.new(mu_div.outputs['Value'], combine.inputs['Y'])
    
    # Emission
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (1400, 100)
    emission.inputs['Strength'].default_value = 1.0
    links.new(combine.outputs['Vector'], emission.inputs['Color'])
    
    # Output
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (1600, 100)
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    # Auto-assign
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
            obj.data.materials.append(mat)
    
    mesh_count = len([o for o in bpy.context.scene.objects if o.type == 'MESH'])
    print(f"\nCreated: {mat.name}")
    print(f"Assigned to {mesh_count} meshes")
    print(f"R channel should be CONSTANT at {r_normalized:.6f}")
    
    return mat


# =============================================================================
# STEP 1.3: TRANSMITTANCE UV CALCULATION
# =============================================================================

def apply_step_1_3_transmittance_uv():
    """
    Step 1.3: Transmittance UV Calculation
    
    Converts (r, mu) to UV coordinates for sampling transmittance.exr
    
    Reference: functions.glsl lines 402-421 (GetTransmittanceTextureUvFromRMu)
    
    Output:
    - R = U coordinate (x_mu mapped)
    - G = V coordinate (x_r mapped)
    - B = 0
    
    Expected:
    - Both U and V in range [0, 1]
    - Looking up (mu≈1): U near 0 (short path)
    - Looking horizontal (mu≈0): U near 1 (long path)
    """
    import time
    
    # Constants
    BOTTOM_RADIUS = 6360.0  # km
    TOP_RADIUS = 6420.0     # km
    TRANSMITTANCE_WIDTH = 256
    TRANSMITTANCE_HEIGHT = 64
    
    # Camera position
    CAM_X, CAM_Y, CAM_Z = 37.069, -44.786, 6.0
    
    # Pre-compute camera-related constants
    cam_rel_z_km = (CAM_Z * 0.001) + BOTTOM_RADIUS
    r_km = math.sqrt((CAM_X * 0.001)**2 + (CAM_Y * 0.001)**2 + cam_rel_z_km**2)
    
    # H = sqrt(top² - bottom²) ≈ 797.66 km
    H = math.sqrt(TOP_RADIUS**2 - BOTTOM_RADIUS**2)
    
    # rho = sqrt(r² - bottom²) - distance to horizon
    rho = math.sqrt(max(r_km**2 - BOTTOM_RADIUS**2, 0))
    
    # Pre-computed up direction
    up_x = (CAM_X * 0.001) / r_km
    up_y = (CAM_Y * 0.001) / r_km
    up_z = cam_rel_z_km / r_km
    
    print(f"Step 1.3: Transmittance UV Calculation")
    print(f"  r = {r_km:.6f} km")
    print(f"  H = {H:.4f} km")
    print(f"  rho = {rho:.6f} km")
    print(f"  x_r = rho/H = {rho/H:.6f}")
    
    # Create material
    mat = bpy.data.materials.new(name=f"Step1_3_TransUV_{int(time.time())}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # === INPUTS ===
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.location = (-400, 300)
    
    cam_pos = nodes.new('ShaderNodeCombineXYZ')
    cam_pos.location = (-400, 100)
    cam_pos.inputs['X'].default_value = CAM_X
    cam_pos.inputs['Y'].default_value = CAM_Y
    cam_pos.inputs['Z'].default_value = CAM_Z
    
    up_vec = nodes.new('ShaderNodeCombineXYZ')
    up_vec.location = (-400, -100)
    up_vec.inputs['X'].default_value = up_x
    up_vec.inputs['Y'].default_value = up_y
    up_vec.inputs['Z'].default_value = up_z
    up_vec.label = "up_at_camera"
    
    # === COMPUTE mu ===
    # view_dir = normalize(world_pos - camera)
    view_sub = nodes.new('ShaderNodeVectorMath')
    view_sub.operation = 'SUBTRACT'
    view_sub.location = (-200, 300)
    links.new(geom.outputs['Position'], view_sub.inputs[0])
    links.new(cam_pos.outputs['Vector'], view_sub.inputs[1])
    
    view_norm = nodes.new('ShaderNodeVectorMath')
    view_norm.operation = 'NORMALIZE'
    view_norm.location = (0, 300)
    links.new(view_sub.outputs['Vector'], view_norm.inputs[0])
    
    # mu = dot(view_dir, up)
    mu_dot = nodes.new('ShaderNodeVectorMath')
    mu_dot.operation = 'DOT_PRODUCT'
    mu_dot.location = (200, 200)
    links.new(view_norm.outputs['Vector'], mu_dot.inputs[0])
    links.new(up_vec.outputs['Vector'], mu_dot.inputs[1])
    
    # === COMPUTE d = distance to top atmosphere ===
    # discriminant = r² * (mu² - 1) + top_radius²
    # d = max(0, -r*mu + sqrt(max(discriminant, 0)))
    
    # mu²
    mu_sq = nodes.new('ShaderNodeMath')
    mu_sq.operation = 'MULTIPLY'
    mu_sq.location = (400, 300)
    links.new(mu_dot.outputs['Value'], mu_sq.inputs[0])
    links.new(mu_dot.outputs['Value'], mu_sq.inputs[1])
    
    # mu² - 1
    mu_sq_m1 = nodes.new('ShaderNodeMath')
    mu_sq_m1.operation = 'SUBTRACT'
    mu_sq_m1.location = (600, 300)
    mu_sq_m1.inputs[1].default_value = 1.0
    links.new(mu_sq.outputs['Value'], mu_sq_m1.inputs[0])
    
    # r² * (mu² - 1)
    r_sq = r_km * r_km
    r_sq_term = nodes.new('ShaderNodeMath')
    r_sq_term.operation = 'MULTIPLY'
    r_sq_term.location = (800, 300)
    r_sq_term.inputs[1].default_value = r_sq
    links.new(mu_sq_m1.outputs['Value'], r_sq_term.inputs[0])
    
    # + top_radius²
    top_sq = TOP_RADIUS * TOP_RADIUS
    discrim = nodes.new('ShaderNodeMath')
    discrim.operation = 'ADD'
    discrim.location = (1000, 300)
    discrim.inputs[1].default_value = top_sq
    links.new(r_sq_term.outputs['Value'], discrim.inputs[0])
    
    # max(discriminant, 0)
    discrim_safe = nodes.new('ShaderNodeMath')
    discrim_safe.operation = 'MAXIMUM'
    discrim_safe.location = (1200, 300)
    discrim_safe.inputs[1].default_value = 0.0
    links.new(discrim.outputs['Value'], discrim_safe.inputs[0])
    
    # sqrt(discriminant)
    discrim_sqrt = nodes.new('ShaderNodeMath')
    discrim_sqrt.operation = 'SQRT'
    discrim_sqrt.location = (1400, 300)
    links.new(discrim_safe.outputs['Value'], discrim_sqrt.inputs[0])
    
    # -r * mu
    neg_r_mu = nodes.new('ShaderNodeMath')
    neg_r_mu.operation = 'MULTIPLY'
    neg_r_mu.location = (400, 100)
    neg_r_mu.inputs[1].default_value = -r_km
    links.new(mu_dot.outputs['Value'], neg_r_mu.inputs[0])
    
    # d = -r*mu + sqrt(discrim)
    d_raw = nodes.new('ShaderNodeMath')
    d_raw.operation = 'ADD'
    d_raw.location = (1600, 200)
    links.new(neg_r_mu.outputs['Value'], d_raw.inputs[0])
    links.new(discrim_sqrt.outputs['Value'], d_raw.inputs[1])
    
    # d = max(0, d_raw)
    d_node = nodes.new('ShaderNodeMath')
    d_node.operation = 'MAXIMUM'
    d_node.location = (1800, 200)
    d_node.inputs[1].default_value = 0.0
    links.new(d_raw.outputs['Value'], d_node.inputs[0])
    
    # === COMPUTE UV ===
    # d_min = top_radius - r
    d_min = TOP_RADIUS - r_km
    # d_max = rho + H
    d_max = rho + H
    
    # x_mu = (d - d_min) / (d_max - d_min)
    d_minus_dmin = nodes.new('ShaderNodeMath')
    d_minus_dmin.operation = 'SUBTRACT'
    d_minus_dmin.location = (2000, 200)
    d_minus_dmin.inputs[1].default_value = d_min
    links.new(d_node.outputs['Value'], d_minus_dmin.inputs[0])
    
    x_mu = nodes.new('ShaderNodeMath')
    x_mu.operation = 'DIVIDE'
    x_mu.location = (2200, 200)
    x_mu.inputs[1].default_value = d_max - d_min
    links.new(d_minus_dmin.outputs['Value'], x_mu.inputs[0])
    
    # x_r = rho / H (constant)
    x_r_val = rho / H
    
    # U = 0.5/WIDTH + x_mu * (1 - 1/WIDTH)
    # Simplified: U = (0.5 + x_mu * (WIDTH - 1)) / WIDTH
    u_scale = nodes.new('ShaderNodeMath')
    u_scale.operation = 'MULTIPLY'
    u_scale.location = (2400, 200)
    u_scale.inputs[1].default_value = (TRANSMITTANCE_WIDTH - 1) / TRANSMITTANCE_WIDTH
    links.new(x_mu.outputs['Value'], u_scale.inputs[0])
    
    u_offset = nodes.new('ShaderNodeMath')
    u_offset.operation = 'ADD'
    u_offset.location = (2600, 200)
    u_offset.inputs[1].default_value = 0.5 / TRANSMITTANCE_WIDTH
    links.new(u_scale.outputs['Value'], u_offset.inputs[0])
    
    # V is constant (x_r is constant for fixed camera)
    v_val = 0.5 / TRANSMITTANCE_HEIGHT + x_r_val * (1 - 1 / TRANSMITTANCE_HEIGHT)
    
    v_const = nodes.new('ShaderNodeValue')
    v_const.location = (2600, 0)
    v_const.outputs['Value'].default_value = v_val
    v_const.label = f"V = {v_val:.6f}"
    
    # === OUTPUT ===
    combine = nodes.new('ShaderNodeCombineXYZ')
    combine.location = (2800, 100)
    combine.inputs['Z'].default_value = 0.0
    links.new(u_offset.outputs['Value'], combine.inputs['X'])
    links.new(v_const.outputs['Value'], combine.inputs['Y'])
    
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (3000, 100)
    emission.inputs['Strength'].default_value = 1.0
    links.new(combine.outputs['Vector'], emission.inputs['Color'])
    
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (3200, 100)
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    # Auto-assign
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
            obj.data.materials.append(mat)
    
    mesh_count = len([o for o in bpy.context.scene.objects if o.type == 'MESH'])
    print(f"\nCreated: {mat.name}")
    print(f"Assigned to {mesh_count} meshes")
    print(f"\nOutput encoding:")
    print(f"  R = U (varies with mu/view direction)")
    print(f"  G = V = {v_val:.6f} (constant for this camera altitude)")
    print(f"  B = 0")
    print(f"\nExpected:")
    print(f"  Looking up (mu≈1): U near 0")
    print(f"  Looking horizontal (mu≈0): U near 1")
    
    return mat


# =============================================================================
# STEP 1.4: TRANSMITTANCE LUT SAMPLING
# =============================================================================

def apply_step_1_4_transmittance_sample():
    """
    Step 1.4: Sample Transmittance LUT
    
    Uses computed UV to sample transmittance.exr and output the RGB values.
    
    Output:
    - RGB = sampled transmittance from LUT
    
    Expected:
    - Looking up: nearly white (high transmittance, short path)
    - Looking horizontal: darker, with blue > green > red (more red absorbed)
    """
    import time
    import os
    
    # Constants
    BOTTOM_RADIUS = 6360.0
    TOP_RADIUS = 6420.0
    TRANSMITTANCE_WIDTH = 256
    TRANSMITTANCE_HEIGHT = 64
    CAM_X, CAM_Y, CAM_Z = 37.069, -44.786, 6.0
    
    # LUT path - check common locations
    lut_paths = [
        r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-3\helios_cache\luts\transmittance.exr",
        r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-4\helios_cache\luts\transmittance.exr",
    ]
    
    lut_path = None
    for p in lut_paths:
        if os.path.exists(p):
            lut_path = p
            break
    
    if not lut_path:
        print("ERROR: Could not find transmittance.exr")
        print("Checked paths:")
        for p in lut_paths:
            print(f"  {p}")
        return None
    
    print(f"Step 1.4: Transmittance LUT Sampling")
    print(f"  LUT: {lut_path}")
    
    # Pre-compute constants
    cam_rel_z_km = (CAM_Z * 0.001) + BOTTOM_RADIUS
    r_km = math.sqrt((CAM_X * 0.001)**2 + (CAM_Y * 0.001)**2 + cam_rel_z_km**2)
    H = math.sqrt(TOP_RADIUS**2 - BOTTOM_RADIUS**2)
    rho = math.sqrt(max(r_km**2 - BOTTOM_RADIUS**2, 0))
    
    up_x = (CAM_X * 0.001) / r_km
    up_y = (CAM_Y * 0.001) / r_km
    up_z = cam_rel_z_km / r_km
    
    d_min = TOP_RADIUS - r_km
    d_max = rho + H
    r_sq = r_km * r_km
    top_sq = TOP_RADIUS * TOP_RADIUS
    
    x_r_val = rho / H
    v_val = 0.5 / TRANSMITTANCE_HEIGHT + x_r_val * (1 - 1 / TRANSMITTANCE_HEIGHT)
    
    print(f"  r = {r_km:.4f} km, rho = {rho:.4f} km")
    print(f"  V = {v_val:.6f} (constant)")
    
    # Create material
    mat = bpy.data.materials.new(name=f"Step1_4_TransSample_{int(time.time())}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # === GEOMETRY & CAMERA ===
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.location = (-600, 300)
    
    cam_pos = nodes.new('ShaderNodeCombineXYZ')
    cam_pos.location = (-600, 100)
    cam_pos.inputs['X'].default_value = CAM_X
    cam_pos.inputs['Y'].default_value = CAM_Y
    cam_pos.inputs['Z'].default_value = CAM_Z
    
    up_vec = nodes.new('ShaderNodeCombineXYZ')
    up_vec.location = (-600, -100)
    up_vec.inputs['X'].default_value = up_x
    up_vec.inputs['Y'].default_value = up_y
    up_vec.inputs['Z'].default_value = up_z
    
    # === VIEW DIRECTION & MU ===
    view_sub = nodes.new('ShaderNodeVectorMath')
    view_sub.operation = 'SUBTRACT'
    view_sub.location = (-400, 300)
    links.new(geom.outputs['Position'], view_sub.inputs[0])
    links.new(cam_pos.outputs['Vector'], view_sub.inputs[1])
    
    view_norm = nodes.new('ShaderNodeVectorMath')
    view_norm.operation = 'NORMALIZE'
    view_norm.location = (-200, 300)
    links.new(view_sub.outputs['Vector'], view_norm.inputs[0])
    
    mu_dot = nodes.new('ShaderNodeVectorMath')
    mu_dot.operation = 'DOT_PRODUCT'
    mu_dot.location = (0, 200)
    links.new(view_norm.outputs['Vector'], mu_dot.inputs[0])
    links.new(up_vec.outputs['Vector'], mu_dot.inputs[1])
    
    # === COMPUTE d (distance to atmosphere top) ===
    mu_sq = nodes.new('ShaderNodeMath')
    mu_sq.operation = 'MULTIPLY'
    mu_sq.location = (200, 300)
    links.new(mu_dot.outputs['Value'], mu_sq.inputs[0])
    links.new(mu_dot.outputs['Value'], mu_sq.inputs[1])
    
    mu_sq_m1 = nodes.new('ShaderNodeMath')
    mu_sq_m1.operation = 'SUBTRACT'
    mu_sq_m1.location = (400, 300)
    mu_sq_m1.inputs[1].default_value = 1.0
    links.new(mu_sq.outputs['Value'], mu_sq_m1.inputs[0])
    
    r_sq_term = nodes.new('ShaderNodeMath')
    r_sq_term.operation = 'MULTIPLY'
    r_sq_term.location = (600, 300)
    r_sq_term.inputs[1].default_value = r_sq
    links.new(mu_sq_m1.outputs['Value'], r_sq_term.inputs[0])
    
    discrim = nodes.new('ShaderNodeMath')
    discrim.operation = 'ADD'
    discrim.location = (800, 300)
    discrim.inputs[1].default_value = top_sq
    links.new(r_sq_term.outputs['Value'], discrim.inputs[0])
    
    discrim_safe = nodes.new('ShaderNodeMath')
    discrim_safe.operation = 'MAXIMUM'
    discrim_safe.location = (1000, 300)
    discrim_safe.inputs[1].default_value = 0.0
    links.new(discrim.outputs['Value'], discrim_safe.inputs[0])
    
    discrim_sqrt = nodes.new('ShaderNodeMath')
    discrim_sqrt.operation = 'SQRT'
    discrim_sqrt.location = (1200, 300)
    links.new(discrim_safe.outputs['Value'], discrim_sqrt.inputs[0])
    
    neg_r_mu = nodes.new('ShaderNodeMath')
    neg_r_mu.operation = 'MULTIPLY'
    neg_r_mu.location = (200, 100)
    neg_r_mu.inputs[1].default_value = -r_km
    links.new(mu_dot.outputs['Value'], neg_r_mu.inputs[0])
    
    d_raw = nodes.new('ShaderNodeMath')
    d_raw.operation = 'ADD'
    d_raw.location = (1400, 200)
    links.new(neg_r_mu.outputs['Value'], d_raw.inputs[0])
    links.new(discrim_sqrt.outputs['Value'], d_raw.inputs[1])
    
    d_node = nodes.new('ShaderNodeMath')
    d_node.operation = 'MAXIMUM'
    d_node.location = (1600, 200)
    d_node.inputs[1].default_value = 0.0
    links.new(d_raw.outputs['Value'], d_node.inputs[0])
    
    # === COMPUTE UV ===
    d_minus_dmin = nodes.new('ShaderNodeMath')
    d_minus_dmin.operation = 'SUBTRACT'
    d_minus_dmin.location = (1800, 200)
    d_minus_dmin.inputs[1].default_value = d_min
    links.new(d_node.outputs['Value'], d_minus_dmin.inputs[0])
    
    x_mu = nodes.new('ShaderNodeMath')
    x_mu.operation = 'DIVIDE'
    x_mu.location = (2000, 200)
    x_mu.inputs[1].default_value = d_max - d_min
    links.new(d_minus_dmin.outputs['Value'], x_mu.inputs[0])
    
    # Clamp x_mu to [0, 1] for valid texture sampling
    x_mu_clamp = nodes.new('ShaderNodeClamp')
    x_mu_clamp.location = (2200, 200)
    x_mu_clamp.inputs['Min'].default_value = 0.0
    x_mu_clamp.inputs['Max'].default_value = 1.0
    links.new(x_mu.outputs['Value'], x_mu_clamp.inputs['Value'])
    
    # U with half-texel offset
    u_scale = nodes.new('ShaderNodeMath')
    u_scale.operation = 'MULTIPLY'
    u_scale.location = (2400, 200)
    u_scale.inputs[1].default_value = (TRANSMITTANCE_WIDTH - 1) / TRANSMITTANCE_WIDTH
    links.new(x_mu_clamp.outputs['Result'], u_scale.inputs[0])
    
    u_final = nodes.new('ShaderNodeMath')
    u_final.operation = 'ADD'
    u_final.location = (2600, 200)
    u_final.inputs[1].default_value = 0.5 / TRANSMITTANCE_WIDTH
    links.new(u_scale.outputs['Value'], u_final.inputs[0])
    
    # V is constant
    v_const = nodes.new('ShaderNodeValue')
    v_const.location = (2600, 0)
    v_const.outputs['Value'].default_value = v_val
    
    # Combine UV
    uv_combine = nodes.new('ShaderNodeCombineXYZ')
    uv_combine.location = (2800, 100)
    uv_combine.inputs['Z'].default_value = 0.0
    links.new(u_final.outputs['Value'], uv_combine.inputs['X'])
    links.new(v_const.outputs['Value'], uv_combine.inputs['Y'])
    
    # === SAMPLE TRANSMITTANCE TEXTURE ===
    tex_img = nodes.new('ShaderNodeTexImage')
    tex_img.location = (3000, 100)
    tex_img.interpolation = 'Linear'
    tex_img.extension = 'EXTEND'
    
    # Load the image
    if lut_path in bpy.data.images:
        tex_img.image = bpy.data.images[lut_path]
    else:
        img = bpy.data.images.load(lut_path)
        img.colorspace_settings.name = 'Non-Color'
        tex_img.image = img
    
    links.new(uv_combine.outputs['Vector'], tex_img.inputs['Vector'])
    
    # === OUTPUT ===
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (3300, 100)
    emission.inputs['Strength'].default_value = 1.0
    links.new(tex_img.outputs['Color'], emission.inputs['Color'])
    
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (3500, 100)
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    # Auto-assign
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
            obj.data.materials.append(mat)
    
    mesh_count = len([o for o in bpy.context.scene.objects if o.type == 'MESH'])
    print(f"\nCreated: {mat.name}")
    print(f"Assigned to {mesh_count} meshes")
    print(f"\nExpected output:")
    print(f"  Looking up: white/bright (high transmittance)")
    print(f"  Looking horizontal: darker, blue-tinted (red absorbed)")
    print(f"  Ground-facing rays: clamped to horizon transmittance")
    
    return mat


# =============================================================================
# STEP 1.5: TRANSMITTANCE BETWEEN TWO POINTS
# =============================================================================

def apply_step_1_5_point_transmittance():
    """
    Step 1.5: Transmittance Between Camera and Surface Point
    
    Uses the ratio method: T(cam→point) = T(cam→top) / T(point→top)
    
    Reference: functions.glsl lines 493-519 (GetTransmittance)
    
    Output:
    - RGB = transmittance from camera to surface point
    
    Expected:
    - Near objects: nearly white (high transmittance)
    - Far objects: darker, with R > G > B (red transmits better)
    """
    import time
    import os
    
    # Constants
    BOTTOM_RADIUS = 6360.0
    TOP_RADIUS = 6420.0
    TRANSMITTANCE_WIDTH = 256
    TRANSMITTANCE_HEIGHT = 64
    CAM_X, CAM_Y, CAM_Z = 37.069, -44.786, 6.0
    
    # LUT path
    lut_paths = [
        r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-3\helios_cache\luts\transmittance.exr",
        r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-4\helios_cache\luts\transmittance.exr",
    ]
    
    lut_path = None
    for p in lut_paths:
        if os.path.exists(p):
            lut_path = p
            break
    
    if not lut_path:
        print("ERROR: Could not find transmittance.exr")
        return None
    
    print(f"Step 1.5: Point Transmittance (camera to surface)")
    print(f"  LUT: {lut_path}")
    
    # Pre-compute camera constants
    cam_rel_z_km = (CAM_Z * 0.001) + BOTTOM_RADIUS
    r_km = math.sqrt((CAM_X * 0.001)**2 + (CAM_Y * 0.001)**2 + cam_rel_z_km**2)
    H = math.sqrt(TOP_RADIUS**2 - BOTTOM_RADIUS**2)
    rho_cam = math.sqrt(max(r_km**2 - BOTTOM_RADIUS**2, 0))
    
    up_x = (CAM_X * 0.001) / r_km
    up_y = (CAM_Y * 0.001) / r_km
    up_z = cam_rel_z_km / r_km
    
    print(f"  Camera r = {r_km:.4f} km")
    
    # Create material
    mat = bpy.data.materials.new(name=f"Step1_5_PointTrans_{int(time.time())}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Load transmittance texture
    if lut_path in bpy.data.images:
        trans_img = bpy.data.images[lut_path]
    else:
        trans_img = bpy.data.images.load(lut_path)
        trans_img.colorspace_settings.name = 'Non-Color'
    
    # === INPUTS ===
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.location = (-800, 400)
    
    cam_pos = nodes.new('ShaderNodeCombineXYZ')
    cam_pos.location = (-800, 200)
    cam_pos.inputs['X'].default_value = CAM_X
    cam_pos.inputs['Y'].default_value = CAM_Y
    cam_pos.inputs['Z'].default_value = CAM_Z
    
    up_vec = nodes.new('ShaderNodeCombineXYZ')
    up_vec.location = (-800, 0)
    up_vec.inputs['X'].default_value = up_x
    up_vec.inputs['Y'].default_value = up_y
    up_vec.inputs['Z'].default_value = up_z
    
    # === COMPUTE VIEW DIRECTION & DISTANCE ===
    view_sub = nodes.new('ShaderNodeVectorMath')
    view_sub.operation = 'SUBTRACT'
    view_sub.location = (-600, 400)
    links.new(geom.outputs['Position'], view_sub.inputs[0])
    links.new(cam_pos.outputs['Vector'], view_sub.inputs[1])
    
    # Distance d (in meters, will convert to km)
    dist_m = nodes.new('ShaderNodeVectorMath')
    dist_m.operation = 'LENGTH'
    dist_m.location = (-400, 500)
    links.new(view_sub.outputs['Vector'], dist_m.inputs[0])
    
    # d in km
    dist_km = nodes.new('ShaderNodeMath')
    dist_km.operation = 'MULTIPLY'
    dist_km.location = (-200, 500)
    dist_km.inputs[1].default_value = 0.001
    links.new(dist_m.outputs['Value'], dist_km.inputs[0])
    
    view_norm = nodes.new('ShaderNodeVectorMath')
    view_norm.operation = 'NORMALIZE'
    view_norm.location = (-400, 400)
    links.new(view_sub.outputs['Vector'], view_norm.inputs[0])
    
    # mu = dot(view_dir, up)
    mu_dot = nodes.new('ShaderNodeVectorMath')
    mu_dot.operation = 'DOT_PRODUCT'
    mu_dot.location = (-200, 300)
    links.new(view_norm.outputs['Vector'], mu_dot.inputs[0])
    links.new(up_vec.outputs['Vector'], mu_dot.inputs[1])
    
    # === COMPUTE r_d and mu_d at the surface point ===
    # r_d = sqrt(d² + 2*r*mu*d + r²)
    # mu_d = (r*mu + d) / r_d
    
    # d²
    d_sq = nodes.new('ShaderNodeMath')
    d_sq.operation = 'MULTIPLY'
    d_sq.location = (0, 600)
    links.new(dist_km.outputs['Value'], d_sq.inputs[0])
    links.new(dist_km.outputs['Value'], d_sq.inputs[1])
    
    # 2*r*mu*d = 2*r*d * mu
    two_r = 2.0 * r_km
    two_r_d = nodes.new('ShaderNodeMath')
    two_r_d.operation = 'MULTIPLY'
    two_r_d.location = (0, 400)
    two_r_d.inputs[1].default_value = two_r
    links.new(dist_km.outputs['Value'], two_r_d.inputs[0])
    
    two_r_mu_d = nodes.new('ShaderNodeMath')
    two_r_mu_d.operation = 'MULTIPLY'
    two_r_mu_d.location = (200, 400)
    links.new(two_r_d.outputs['Value'], two_r_mu_d.inputs[0])
    links.new(mu_dot.outputs['Value'], two_r_mu_d.inputs[1])
    
    # d² + 2*r*mu*d
    sum1 = nodes.new('ShaderNodeMath')
    sum1.operation = 'ADD'
    sum1.location = (400, 500)
    links.new(d_sq.outputs['Value'], sum1.inputs[0])
    links.new(two_r_mu_d.outputs['Value'], sum1.inputs[1])
    
    # + r²
    r_sq = r_km * r_km
    sum2 = nodes.new('ShaderNodeMath')
    sum2.operation = 'ADD'
    sum2.location = (600, 500)
    sum2.inputs[1].default_value = r_sq
    links.new(sum1.outputs['Value'], sum2.inputs[0])
    
    # r_d = sqrt(...)
    r_d_raw = nodes.new('ShaderNodeMath')
    r_d_raw.operation = 'SQRT'
    r_d_raw.location = (800, 500)
    links.new(sum2.outputs['Value'], r_d_raw.inputs[0])
    
    # Clamp r_d to [bottom, top]
    r_d_clamp = nodes.new('ShaderNodeClamp')
    r_d_clamp.location = (1000, 500)
    r_d_clamp.inputs['Min'].default_value = BOTTOM_RADIUS
    r_d_clamp.inputs['Max'].default_value = TOP_RADIUS
    links.new(r_d_raw.outputs['Value'], r_d_clamp.inputs['Value'])
    
    # mu_d = (r*mu + d) / r_d
    r_mu = nodes.new('ShaderNodeMath')
    r_mu.operation = 'MULTIPLY'
    r_mu.location = (200, 200)
    r_mu.inputs[1].default_value = r_km
    links.new(mu_dot.outputs['Value'], r_mu.inputs[0])
    
    r_mu_plus_d = nodes.new('ShaderNodeMath')
    r_mu_plus_d.operation = 'ADD'
    r_mu_plus_d.location = (400, 200)
    links.new(r_mu.outputs['Value'], r_mu_plus_d.inputs[0])
    links.new(dist_km.outputs['Value'], r_mu_plus_d.inputs[1])
    
    mu_d_raw = nodes.new('ShaderNodeMath')
    mu_d_raw.operation = 'DIVIDE'
    mu_d_raw.location = (1200, 300)
    links.new(r_mu_plus_d.outputs['Value'], mu_d_raw.inputs[0])
    links.new(r_d_clamp.outputs['Result'], mu_d_raw.inputs[1])
    
    # Clamp mu_d to [-1, 1]
    mu_d_clamp = nodes.new('ShaderNodeClamp')
    mu_d_clamp.location = (1400, 300)
    mu_d_clamp.inputs['Min'].default_value = -1.0
    mu_d_clamp.inputs['Max'].default_value = 1.0
    links.new(mu_d_raw.outputs['Value'], mu_d_clamp.inputs['Value'])
    
    # === HELPER: Create transmittance UV from r, mu ===
    def create_trans_uv_nodes(r_input, mu_input, prefix, x_offset):
        """Create nodes to compute transmittance UV from r, mu inputs."""
        # rho = sqrt(r² - bottom²)
        r_sq_node = nodes.new('ShaderNodeMath')
        r_sq_node.operation = 'MULTIPLY'
        r_sq_node.location = (x_offset, 100)
        links.new(r_input, r_sq_node.inputs[0])
        links.new(r_input, r_sq_node.inputs[1])
        
        r_sq_minus_bot = nodes.new('ShaderNodeMath')
        r_sq_minus_bot.operation = 'SUBTRACT'
        r_sq_minus_bot.location = (x_offset + 150, 100)
        r_sq_minus_bot.inputs[1].default_value = BOTTOM_RADIUS * BOTTOM_RADIUS
        links.new(r_sq_node.outputs['Value'], r_sq_minus_bot.inputs[0])
        
        rho_safe = nodes.new('ShaderNodeMath')
        rho_safe.operation = 'MAXIMUM'
        rho_safe.location = (x_offset + 300, 100)
        rho_safe.inputs[1].default_value = 0.0
        links.new(r_sq_minus_bot.outputs['Value'], rho_safe.inputs[0])
        
        rho = nodes.new('ShaderNodeMath')
        rho.operation = 'SQRT'
        rho.location = (x_offset + 450, 100)
        links.new(rho_safe.outputs['Value'], rho.inputs[0])
        
        # d_min = top - r
        d_min = nodes.new('ShaderNodeMath')
        d_min.operation = 'SUBTRACT'
        d_min.location = (x_offset, -100)
        d_min.inputs[0].default_value = TOP_RADIUS
        links.new(r_input, d_min.inputs[1])
        
        # d_max = rho + H
        d_max = nodes.new('ShaderNodeMath')
        d_max.operation = 'ADD'
        d_max.location = (x_offset + 450, -100)
        d_max.inputs[1].default_value = H
        links.new(rho.outputs['Value'], d_max.inputs[0])
        
        # d = DistanceToTopAtmosphereBoundary(r, mu)
        # discriminant = r² * (mu² - 1) + top²
        mu_sq = nodes.new('ShaderNodeMath')
        mu_sq.operation = 'MULTIPLY'
        mu_sq.location = (x_offset, -300)
        links.new(mu_input, mu_sq.inputs[0])
        links.new(mu_input, mu_sq.inputs[1])
        
        mu_sq_m1 = nodes.new('ShaderNodeMath')
        mu_sq_m1.operation = 'SUBTRACT'
        mu_sq_m1.location = (x_offset + 150, -300)
        mu_sq_m1.inputs[1].default_value = 1.0
        links.new(mu_sq.outputs['Value'], mu_sq_m1.inputs[0])
        
        r_sq_term = nodes.new('ShaderNodeMath')
        r_sq_term.operation = 'MULTIPLY'
        r_sq_term.location = (x_offset + 300, -300)
        links.new(r_sq_node.outputs['Value'], r_sq_term.inputs[0])
        links.new(mu_sq_m1.outputs['Value'], r_sq_term.inputs[1])
        
        discrim = nodes.new('ShaderNodeMath')
        discrim.operation = 'ADD'
        discrim.location = (x_offset + 450, -300)
        discrim.inputs[1].default_value = TOP_RADIUS * TOP_RADIUS
        links.new(r_sq_term.outputs['Value'], discrim.inputs[0])
        
        discrim_safe = nodes.new('ShaderNodeMath')
        discrim_safe.operation = 'MAXIMUM'
        discrim_safe.location = (x_offset + 600, -300)
        discrim_safe.inputs[1].default_value = 0.0
        links.new(discrim.outputs['Value'], discrim_safe.inputs[0])
        
        discrim_sqrt = nodes.new('ShaderNodeMath')
        discrim_sqrt.operation = 'SQRT'
        discrim_sqrt.location = (x_offset + 750, -300)
        links.new(discrim_safe.outputs['Value'], discrim_sqrt.inputs[0])
        
        # -r * mu
        neg_r_mu = nodes.new('ShaderNodeMath')
        neg_r_mu.operation = 'MULTIPLY'
        neg_r_mu.location = (x_offset + 300, -450)
        neg_r_mu.inputs[0].default_value = -1.0
        links.new(r_input, neg_r_mu.inputs[1])
        
        neg_r_mu2 = nodes.new('ShaderNodeMath')
        neg_r_mu2.operation = 'MULTIPLY'
        neg_r_mu2.location = (x_offset + 450, -450)
        links.new(neg_r_mu.outputs['Value'], neg_r_mu2.inputs[0])
        links.new(mu_input, neg_r_mu2.inputs[1])
        
        d_dist = nodes.new('ShaderNodeMath')
        d_dist.operation = 'ADD'
        d_dist.location = (x_offset + 900, -350)
        links.new(neg_r_mu2.outputs['Value'], d_dist.inputs[0])
        links.new(discrim_sqrt.outputs['Value'], d_dist.inputs[1])
        
        d_clamped = nodes.new('ShaderNodeMath')
        d_clamped.operation = 'MAXIMUM'
        d_clamped.location = (x_offset + 1050, -350)
        d_clamped.inputs[1].default_value = 0.0
        links.new(d_dist.outputs['Value'], d_clamped.inputs[0])
        
        # x_mu = (d - d_min) / (d_max - d_min)
        d_minus_dmin = nodes.new('ShaderNodeMath')
        d_minus_dmin.operation = 'SUBTRACT'
        d_minus_dmin.location = (x_offset + 1200, -200)
        links.new(d_clamped.outputs['Value'], d_minus_dmin.inputs[0])
        links.new(d_min.outputs['Value'], d_minus_dmin.inputs[1])
        
        dmax_minus_dmin = nodes.new('ShaderNodeMath')
        dmax_minus_dmin.operation = 'SUBTRACT'
        dmax_minus_dmin.location = (x_offset + 1200, -350)
        links.new(d_max.outputs['Value'], dmax_minus_dmin.inputs[0])
        links.new(d_min.outputs['Value'], dmax_minus_dmin.inputs[1])
        
        x_mu_node = nodes.new('ShaderNodeMath')
        x_mu_node.operation = 'DIVIDE'
        x_mu_node.location = (x_offset + 1350, -250)
        links.new(d_minus_dmin.outputs['Value'], x_mu_node.inputs[0])
        links.new(dmax_minus_dmin.outputs['Value'], x_mu_node.inputs[1])
        
        x_mu_clamp = nodes.new('ShaderNodeClamp')
        x_mu_clamp.location = (x_offset + 1500, -250)
        x_mu_clamp.inputs['Min'].default_value = 0.0
        x_mu_clamp.inputs['Max'].default_value = 1.0
        links.new(x_mu_node.outputs['Value'], x_mu_clamp.inputs['Value'])
        
        # x_r = rho / H
        x_r = nodes.new('ShaderNodeMath')
        x_r.operation = 'DIVIDE'
        x_r.location = (x_offset + 600, 100)
        x_r.inputs[1].default_value = H
        links.new(rho.outputs['Value'], x_r.inputs[0])
        
        # Final UV with half-texel offset
        u_scale = nodes.new('ShaderNodeMath')
        u_scale.operation = 'MULTIPLY'
        u_scale.location = (x_offset + 1650, -200)
        u_scale.inputs[1].default_value = (TRANSMITTANCE_WIDTH - 1) / TRANSMITTANCE_WIDTH
        links.new(x_mu_clamp.outputs['Result'], u_scale.inputs[0])
        
        u_final = nodes.new('ShaderNodeMath')
        u_final.operation = 'ADD'
        u_final.location = (x_offset + 1800, -200)
        u_final.inputs[1].default_value = 0.5 / TRANSMITTANCE_WIDTH
        links.new(u_scale.outputs['Value'], u_final.inputs[0])
        
        v_scale = nodes.new('ShaderNodeMath')
        v_scale.operation = 'MULTIPLY'
        v_scale.location = (x_offset + 1650, 100)
        v_scale.inputs[1].default_value = (TRANSMITTANCE_HEIGHT - 1) / TRANSMITTANCE_HEIGHT
        links.new(x_r.outputs['Value'], v_scale.inputs[0])
        
        v_final = nodes.new('ShaderNodeMath')
        v_final.operation = 'ADD'
        v_final.location = (x_offset + 1800, 100)
        v_final.inputs[1].default_value = 0.5 / TRANSMITTANCE_HEIGHT
        links.new(v_scale.outputs['Value'], v_final.inputs[0])
        
        # Combine UV
        uv_combine = nodes.new('ShaderNodeCombineXYZ')
        uv_combine.location = (x_offset + 1950, -50)
        uv_combine.inputs['Z'].default_value = 0.0
        links.new(u_final.outputs['Value'], uv_combine.inputs['X'])
        links.new(v_final.outputs['Value'], uv_combine.inputs['Y'])
        
        return uv_combine.outputs['Vector']
    
    # For camera: use pre-computed constant r
    r_cam_const = nodes.new('ShaderNodeValue')
    r_cam_const.location = (1600, 700)
    r_cam_const.outputs['Value'].default_value = r_km
    
    # Create UV for camera position
    uv_cam = create_trans_uv_nodes(r_cam_const.outputs['Value'], mu_dot.outputs['Value'], "cam", 1600)
    
    # Create UV for surface point
    uv_point = create_trans_uv_nodes(r_d_clamp.outputs['Result'], mu_d_clamp.outputs['Result'], "pt", 3800)
    
    # === SAMPLE TRANSMITTANCE AT BOTH POSITIONS ===
    tex_cam = nodes.new('ShaderNodeTexImage')
    tex_cam.location = (3600, 700)
    tex_cam.interpolation = 'Linear'
    tex_cam.extension = 'EXTEND'
    tex_cam.image = trans_img
    links.new(uv_cam, tex_cam.inputs['Vector'])
    
    tex_point = nodes.new('ShaderNodeTexImage')
    tex_point.location = (5800, -50)
    tex_point.interpolation = 'Linear'
    tex_point.extension = 'EXTEND'
    tex_point.image = trans_img
    links.new(uv_point, tex_point.inputs['Vector'])
    
    # === COMPUTE RATIO T_cam / T_point ===
    # Separate RGB for division
    sep_cam = nodes.new('ShaderNodeSeparateColor')
    sep_cam.location = (3800, 700)
    links.new(tex_cam.outputs['Color'], sep_cam.inputs['Color'])
    
    sep_point = nodes.new('ShaderNodeSeparateColor')
    sep_point.location = (6000, -50)
    links.new(tex_point.outputs['Color'], sep_point.inputs['Color'])
    
    # R ratio
    div_r = nodes.new('ShaderNodeMath')
    div_r.operation = 'DIVIDE'
    div_r.location = (6200, 200)
    links.new(sep_cam.outputs['Red'], div_r.inputs[0])
    links.new(sep_point.outputs['Red'], div_r.inputs[1])
    
    # G ratio
    div_g = nodes.new('ShaderNodeMath')
    div_g.operation = 'DIVIDE'
    div_g.location = (6200, 0)
    links.new(sep_cam.outputs['Green'], div_g.inputs[0])
    links.new(sep_point.outputs['Green'], div_g.inputs[1])
    
    # B ratio
    div_b = nodes.new('ShaderNodeMath')
    div_b.operation = 'DIVIDE'
    div_b.location = (6200, -200)
    links.new(sep_cam.outputs['Blue'], div_b.inputs[0])
    links.new(sep_point.outputs['Blue'], div_b.inputs[1])
    
    # Clamp to [0, 1]
    clamp_r = nodes.new('ShaderNodeClamp')
    clamp_r.location = (6400, 200)
    links.new(div_r.outputs['Value'], clamp_r.inputs['Value'])
    
    clamp_g = nodes.new('ShaderNodeClamp')
    clamp_g.location = (6400, 0)
    links.new(div_g.outputs['Value'], clamp_g.inputs['Value'])
    
    clamp_b = nodes.new('ShaderNodeClamp')
    clamp_b.location = (6400, -200)
    links.new(div_b.outputs['Value'], clamp_b.inputs['Value'])
    
    # Combine
    combine_trans = nodes.new('ShaderNodeCombineColor')
    combine_trans.location = (6600, 0)
    links.new(clamp_r.outputs['Result'], combine_trans.inputs['Red'])
    links.new(clamp_g.outputs['Result'], combine_trans.inputs['Green'])
    links.new(clamp_b.outputs['Result'], combine_trans.inputs['Blue'])
    
    # === OUTPUT ===
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (6800, 0)
    emission.inputs['Strength'].default_value = 1.0
    links.new(combine_trans.outputs['Color'], emission.inputs['Color'])
    
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (7000, 0)
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    # Auto-assign
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
            obj.data.materials.append(mat)
    
    mesh_count = len([o for o in bpy.context.scene.objects if o.type == 'MESH'])
    print(f"\nCreated: {mat.name}")
    print(f"Assigned to {mesh_count} meshes")
    print(f"\nExpected:")
    print(f"  Near objects: white (high transmittance)")
    print(f"  Far objects: darker, R > G > B")
    
    return mat


# =============================================================================
# STEP 2.1: SUN PARAMETERS (mu_s, nu)
# =============================================================================

def apply_step_2_1_sun_params():
    """
    Step 2.1: Sun Parameters for Scattering Lookup
    
    Computes:
    - mu_s = cos(sun zenith angle) = dot(up_at_camera, sun_direction)
    - nu = cos(view-sun angle) = dot(view_direction, sun_direction)
    
    Reference: functions.glsl lines 1811-1813
    
    Output:
    - R = (mu_s + 1) / 2  (constant across frame)
    - G = (nu + 1) / 2    (varies - bright toward sun)
    - B = 0
    
    Expected:
    - R: constant (depends only on sun elevation)
    - G: varies from 0 (away from sun) to 1 (toward sun)
    """
    import time
    
    # Constants
    BOTTOM_RADIUS = 6360.0
    CAM_X, CAM_Y, CAM_Z = 37.069, -44.786, 6.0
    
    # Get sun direction from scene
    sun_obj = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN':
            sun_obj = obj
            break
    
    if sun_obj is None:
        print("ERROR: No sun light found in scene")
        print("Please add a Sun light to the scene")
        return None
    
    # Get sun direction (negative of light direction since light points "from" sun)
    # Sun's forward direction is -Z in local space
    import mathutils
    sun_mat = sun_obj.matrix_world
    sun_dir_local = mathutils.Vector((0, 0, -1))
    sun_dir_world = sun_mat.to_3x3() @ sun_dir_local
    sun_dir_world.normalize()
    
    # We want direction TO the sun, so negate
    sun_to = -sun_dir_world
    
    print(f"Step 2.1: Sun Parameters")
    print(f"  Sun object: {sun_obj.name}")
    print(f"  Sun direction (to sun): ({sun_to.x:.4f}, {sun_to.y:.4f}, {sun_to.z:.4f})")
    
    # Pre-compute camera up vector
    cam_rel_z_km = (CAM_Z * 0.001) + BOTTOM_RADIUS
    r_km = math.sqrt((CAM_X * 0.001)**2 + (CAM_Y * 0.001)**2 + cam_rel_z_km**2)
    
    up_x = (CAM_X * 0.001) / r_km
    up_y = (CAM_Y * 0.001) / r_km
    up_z = cam_rel_z_km / r_km
    
    # mu_s = dot(up, sun_dir)
    mu_s = up_x * sun_to.x + up_y * sun_to.y + up_z * sun_to.z
    
    print(f"  up_at_camera: ({up_x:.6f}, {up_y:.6f}, {up_z:.6f})")
    print(f"  mu_s (sun zenith cos): {mu_s:.4f}")
    print(f"  Sun elevation: {math.degrees(math.asin(mu_s)):.1f}°")
    
    # Create material
    mat = bpy.data.materials.new(name=f"Step2_1_SunParams_{int(time.time())}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # === INPUTS ===
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.location = (-600, 300)
    
    cam_pos = nodes.new('ShaderNodeCombineXYZ')
    cam_pos.location = (-600, 100)
    cam_pos.inputs['X'].default_value = CAM_X
    cam_pos.inputs['Y'].default_value = CAM_Y
    cam_pos.inputs['Z'].default_value = CAM_Z
    
    sun_dir_node = nodes.new('ShaderNodeCombineXYZ')
    sun_dir_node.location = (-600, -100)
    sun_dir_node.inputs['X'].default_value = sun_to.x
    sun_dir_node.inputs['Y'].default_value = sun_to.y
    sun_dir_node.inputs['Z'].default_value = sun_to.z
    sun_dir_node.label = "sun_direction"
    
    # === VIEW DIRECTION ===
    view_sub = nodes.new('ShaderNodeVectorMath')
    view_sub.operation = 'SUBTRACT'
    view_sub.location = (-400, 300)
    links.new(geom.outputs['Position'], view_sub.inputs[0])
    links.new(cam_pos.outputs['Vector'], view_sub.inputs[1])
    
    view_norm = nodes.new('ShaderNodeVectorMath')
    view_norm.operation = 'NORMALIZE'
    view_norm.location = (-200, 300)
    links.new(view_sub.outputs['Vector'], view_norm.inputs[0])
    
    # === NU = dot(view, sun) ===
    nu_dot = nodes.new('ShaderNodeVectorMath')
    nu_dot.operation = 'DOT_PRODUCT'
    nu_dot.location = (0, 200)
    links.new(view_norm.outputs['Vector'], nu_dot.inputs[0])
    links.new(sun_dir_node.outputs['Vector'], nu_dot.inputs[1])
    
    # Encode nu: (nu + 1) / 2
    nu_add = nodes.new('ShaderNodeMath')
    nu_add.operation = 'ADD'
    nu_add.location = (200, 200)
    nu_add.inputs[1].default_value = 1.0
    links.new(nu_dot.outputs['Value'], nu_add.inputs[0])
    
    nu_div = nodes.new('ShaderNodeMath')
    nu_div.operation = 'DIVIDE'
    nu_div.location = (400, 200)
    nu_div.inputs[1].default_value = 2.0
    links.new(nu_add.outputs['Value'], nu_div.inputs[0])
    
    # === MU_S (constant) ===
    mu_s_encoded = (mu_s + 1.0) / 2.0
    mu_s_const = nodes.new('ShaderNodeValue')
    mu_s_const.location = (400, 400)
    mu_s_const.outputs['Value'].default_value = mu_s_encoded
    mu_s_const.label = f"mu_s_enc = {mu_s_encoded:.4f}"
    
    # === OUTPUT ===
    combine = nodes.new('ShaderNodeCombineXYZ')
    combine.location = (600, 300)
    combine.inputs['Z'].default_value = 0.0
    links.new(mu_s_const.outputs['Value'], combine.inputs['X'])
    links.new(nu_div.outputs['Value'], combine.inputs['Y'])
    
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (800, 300)
    emission.inputs['Strength'].default_value = 1.0
    links.new(combine.outputs['Vector'], emission.inputs['Color'])
    
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (1000, 300)
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    # Auto-assign
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
            obj.data.materials.append(mat)
    
    mesh_count = len([o for o in bpy.context.scene.objects if o.type == 'MESH'])
    print(f"\nCreated: {mat.name}")
    print(f"Assigned to {mesh_count} meshes")
    print(f"\nOutput encoding:")
    print(f"  R = (mu_s+1)/2 = {mu_s_encoded:.4f} (CONSTANT)")
    print(f"  G = (nu+1)/2 (varies: 0=away, 1=toward sun)")
    print(f"  B = 0")
    
    return mat


# =============================================================================
# STEP 2.2: SCATTERING 4D UV MAPPING
# =============================================================================

def apply_step_2_2_scattering_uv():
    """
    Step 2.2: Scattering 4D→2D UV Mapping
    
    Computes (u_r, u_mu, u_mu_s, u_nu) from (r, mu, mu_s, nu)
    
    Reference: functions.glsl lines 773-831 (GetScatteringTextureUvwzFromRMuMuSNu)
    
    Output (for visualization):
    - R = u_mu (view zenith mapping)
    - G = u_mu_s (sun zenith mapping)
    - B = u_nu (view-sun angle mapping)
    
    Note: u_r is constant for ground-level camera, so we skip it in output.
    """
    import time
    
    # Constants
    BOTTOM_RADIUS = 6360.0
    TOP_RADIUS = 6420.0
    MU_S_MIN = -0.2  # Bruneton default
    
    # Scattering texture sizes (from Bruneton)
    SCATTERING_R_SIZE = 32
    SCATTERING_MU_SIZE = 128
    SCATTERING_MU_S_SIZE = 32
    SCATTERING_NU_SIZE = 8
    
    CAM_X, CAM_Y, CAM_Z = 37.069, -44.786, 6.0
    
    # Get sun direction
    sun_obj = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN':
            sun_obj = obj
            break
    
    if sun_obj is None:
        print("ERROR: No sun light found")
        return None
    
    import mathutils
    sun_mat = sun_obj.matrix_world
    sun_dir_local = mathutils.Vector((0, 0, -1))
    sun_dir_world = sun_mat.to_3x3() @ sun_dir_local
    sun_dir_world.normalize()
    sun_to = -sun_dir_world
    
    print(f"Step 2.2: Scattering 4D UV Mapping")
    print(f"  Sun direction: ({sun_to.x:.4f}, {sun_to.y:.4f}, {sun_to.z:.4f})")
    
    # Pre-compute constants
    cam_rel_z_km = (CAM_Z * 0.001) + BOTTOM_RADIUS
    r_km = math.sqrt((CAM_X * 0.001)**2 + (CAM_Y * 0.001)**2 + cam_rel_z_km**2)
    H = math.sqrt(TOP_RADIUS**2 - BOTTOM_RADIUS**2)
    rho = math.sqrt(max(r_km**2 - BOTTOM_RADIUS**2, 0))
    
    up_x = (CAM_X * 0.001) / r_km
    up_y = (CAM_Y * 0.001) / r_km
    up_z = cam_rel_z_km / r_km
    
    # mu_s (constant for frame)
    mu_s = up_x * sun_to.x + up_y * sun_to.y + up_z * sun_to.z
    
    # u_r (constant - from rho/H)
    x_r = rho / H
    u_r = 0.5 / SCATTERING_R_SIZE + x_r * (1 - 1/SCATTERING_R_SIZE)
    
    # Pre-compute u_mu_s (constant for frame)
    # d = DistanceToTopAtmosphereBoundary(bottom_radius, mu_s)
    discrim_s = BOTTOM_RADIUS**2 * (mu_s**2 - 1) + TOP_RADIUS**2
    d_s = max(0, -BOTTOM_RADIUS * mu_s + math.sqrt(max(discrim_s, 0)))
    d_min_s = TOP_RADIUS - BOTTOM_RADIUS
    d_max_s = H
    a = (d_s - d_min_s) / (d_max_s - d_min_s)
    
    # D = DistanceToTopAtmosphereBoundary(bottom_radius, mu_s_min)
    discrim_D = BOTTOM_RADIUS**2 * (MU_S_MIN**2 - 1) + TOP_RADIUS**2
    D = max(0, -BOTTOM_RADIUS * MU_S_MIN + math.sqrt(max(discrim_D, 0)))
    A = (D - d_min_s) / (d_max_s - d_min_s)
    
    x_mu_s = max(1.0 - a / A, 0.0) / (1.0 + a)
    u_mu_s = 0.5 / SCATTERING_MU_S_SIZE + x_mu_s * (1 - 1/SCATTERING_MU_S_SIZE)
    
    print(f"  r = {r_km:.4f} km, rho = {rho:.4f} km")
    print(f"  u_r = {u_r:.4f} (constant)")
    print(f"  mu_s = {mu_s:.4f}, u_mu_s = {u_mu_s:.4f} (constant)")
    
    # Create material
    mat = bpy.data.materials.new(name=f"Step2_2_ScatterUV_{int(time.time())}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # === INPUTS ===
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.location = (-800, 400)
    
    cam_pos = nodes.new('ShaderNodeCombineXYZ')
    cam_pos.location = (-800, 200)
    cam_pos.inputs['X'].default_value = CAM_X
    cam_pos.inputs['Y'].default_value = CAM_Y
    cam_pos.inputs['Z'].default_value = CAM_Z
    
    up_vec = nodes.new('ShaderNodeCombineXYZ')
    up_vec.location = (-800, 0)
    up_vec.inputs['X'].default_value = up_x
    up_vec.inputs['Y'].default_value = up_y
    up_vec.inputs['Z'].default_value = up_z
    
    sun_dir_node = nodes.new('ShaderNodeCombineXYZ')
    sun_dir_node.location = (-800, -200)
    sun_dir_node.inputs['X'].default_value = sun_to.x
    sun_dir_node.inputs['Y'].default_value = sun_to.y
    sun_dir_node.inputs['Z'].default_value = sun_to.z
    
    # === VIEW DIRECTION ===
    view_sub = nodes.new('ShaderNodeVectorMath')
    view_sub.operation = 'SUBTRACT'
    view_sub.location = (-600, 400)
    links.new(geom.outputs['Position'], view_sub.inputs[0])
    links.new(cam_pos.outputs['Vector'], view_sub.inputs[1])
    
    view_norm = nodes.new('ShaderNodeVectorMath')
    view_norm.operation = 'NORMALIZE'
    view_norm.location = (-400, 400)
    links.new(view_sub.outputs['Vector'], view_norm.inputs[0])
    
    # === MU = dot(view, up) ===
    mu_dot = nodes.new('ShaderNodeVectorMath')
    mu_dot.operation = 'DOT_PRODUCT'
    mu_dot.location = (-200, 300)
    links.new(view_norm.outputs['Vector'], mu_dot.inputs[0])
    links.new(up_vec.outputs['Vector'], mu_dot.inputs[1])
    
    # === NU = dot(view, sun) ===
    nu_dot = nodes.new('ShaderNodeVectorMath')
    nu_dot.operation = 'DOT_PRODUCT'
    nu_dot.location = (-200, 100)
    links.new(view_norm.outputs['Vector'], nu_dot.inputs[0])
    links.new(sun_dir_node.outputs['Vector'], nu_dot.inputs[1])
    
    # === U_MU CALCULATION (non-ground-intersecting case) ===
    # For ground-level camera looking at nearby objects, most rays don't intersect ground
    # d = -r*mu + sqrt(r²*(mu²-1) + top²)
    # u_mu = 0.5 + 0.5 * GetTextureCoord((d - d_min) / (d_max - d_min))
    
    # mu²
    mu_sq = nodes.new('ShaderNodeMath')
    mu_sq.operation = 'MULTIPLY'
    mu_sq.location = (0, 400)
    links.new(mu_dot.outputs['Value'], mu_sq.inputs[0])
    links.new(mu_dot.outputs['Value'], mu_sq.inputs[1])
    
    # mu² - 1
    mu_sq_m1 = nodes.new('ShaderNodeMath')
    mu_sq_m1.operation = 'SUBTRACT'
    mu_sq_m1.location = (200, 400)
    mu_sq_m1.inputs[1].default_value = 1.0
    links.new(mu_sq.outputs['Value'], mu_sq_m1.inputs[0])
    
    # r² * (mu² - 1)
    r_sq = r_km * r_km
    r_sq_term = nodes.new('ShaderNodeMath')
    r_sq_term.operation = 'MULTIPLY'
    r_sq_term.location = (400, 400)
    r_sq_term.inputs[1].default_value = r_sq
    links.new(mu_sq_m1.outputs['Value'], r_sq_term.inputs[0])
    
    # + top² + H² (for non-ground case)
    top_sq_plus_H_sq = TOP_RADIUS**2 + H**2  # This is actually top² for discrim, then +H² for d calc
    # Actually the formula is: discriminant = r²*(mu²-1) + bottom² for ground test
    # For d to top: d = -r*mu + sqrt(r²*(mu²-1) + top²)
    # Wait, I need to re-read this...
    
    # For non-ground-intersecting: d = -r*mu + sqrt(discriminant + H²)
    # where discriminant = r*r*mu*mu - r*r + bottom²
    # Let me recalculate: discriminant = r²*mu² - r² + bottom² = r²*(mu²-1) + bottom²
    
    # So: d = -r*mu + sqrt(r²*(mu²-1) + bottom² + H²)
    # And bottom² + H² = top² (since H² = top² - bottom²)
    # So: d = -r*mu + sqrt(r²*(mu²-1) + top²)
    
    discrim = nodes.new('ShaderNodeMath')
    discrim.operation = 'ADD'
    discrim.location = (600, 400)
    discrim.inputs[1].default_value = TOP_RADIUS**2
    links.new(r_sq_term.outputs['Value'], discrim.inputs[0])
    
    discrim_safe = nodes.new('ShaderNodeMath')
    discrim_safe.operation = 'MAXIMUM'
    discrim_safe.location = (800, 400)
    discrim_safe.inputs[1].default_value = 0.0
    links.new(discrim.outputs['Value'], discrim_safe.inputs[0])
    
    discrim_sqrt = nodes.new('ShaderNodeMath')
    discrim_sqrt.operation = 'SQRT'
    discrim_sqrt.location = (1000, 400)
    links.new(discrim_safe.outputs['Value'], discrim_sqrt.inputs[0])
    
    # -r * mu
    neg_r_mu = nodes.new('ShaderNodeMath')
    neg_r_mu.operation = 'MULTIPLY'
    neg_r_mu.location = (400, 200)
    neg_r_mu.inputs[1].default_value = -r_km
    links.new(mu_dot.outputs['Value'], neg_r_mu.inputs[0])
    
    # d = -r*mu + sqrt(discrim)
    d_dist = nodes.new('ShaderNodeMath')
    d_dist.operation = 'ADD'
    d_dist.location = (1200, 300)
    links.new(neg_r_mu.outputs['Value'], d_dist.inputs[0])
    links.new(discrim_sqrt.outputs['Value'], d_dist.inputs[1])
    
    # d_min = top - r, d_max = rho + H
    d_min = TOP_RADIUS - r_km
    d_max = rho + H
    
    # x_mu = (d - d_min) / (d_max - d_min)
    d_minus_dmin = nodes.new('ShaderNodeMath')
    d_minus_dmin.operation = 'SUBTRACT'
    d_minus_dmin.location = (1400, 300)
    d_minus_dmin.inputs[1].default_value = d_min
    links.new(d_dist.outputs['Value'], d_minus_dmin.inputs[0])
    
    x_mu_node = nodes.new('ShaderNodeMath')
    x_mu_node.operation = 'DIVIDE'
    x_mu_node.location = (1600, 300)
    x_mu_node.inputs[1].default_value = d_max - d_min
    links.new(d_minus_dmin.outputs['Value'], x_mu_node.inputs[0])
    
    # Clamp x_mu to [0, 1]
    x_mu_clamp = nodes.new('ShaderNodeClamp')
    x_mu_clamp.location = (1800, 300)
    links.new(x_mu_node.outputs['Value'], x_mu_clamp.inputs['Value'])
    
    # u_mu = 0.5 + 0.5 * GetTextureCoord(x_mu, MU_SIZE/2)
    # GetTextureCoord(x, size) = 0.5/size + x * (1 - 1/size)
    half_mu_size = SCATTERING_MU_SIZE / 2
    
    u_mu_inner = nodes.new('ShaderNodeMath')
    u_mu_inner.operation = 'MULTIPLY'
    u_mu_inner.location = (2000, 300)
    u_mu_inner.inputs[1].default_value = 1 - 1/half_mu_size
    links.new(x_mu_clamp.outputs['Result'], u_mu_inner.inputs[0])
    
    u_mu_inner2 = nodes.new('ShaderNodeMath')
    u_mu_inner2.operation = 'ADD'
    u_mu_inner2.location = (2200, 300)
    u_mu_inner2.inputs[1].default_value = 0.5 / half_mu_size
    links.new(u_mu_inner.outputs['Value'], u_mu_inner2.inputs[0])
    
    # * 0.5
    u_mu_half = nodes.new('ShaderNodeMath')
    u_mu_half.operation = 'MULTIPLY'
    u_mu_half.location = (2400, 300)
    u_mu_half.inputs[1].default_value = 0.5
    links.new(u_mu_inner2.outputs['Value'], u_mu_half.inputs[0])
    
    # + 0.5
    u_mu_final = nodes.new('ShaderNodeMath')
    u_mu_final.operation = 'ADD'
    u_mu_final.location = (2600, 300)
    u_mu_final.inputs[1].default_value = 0.5
    links.new(u_mu_half.outputs['Value'], u_mu_final.inputs[0])
    
    # === U_NU = (nu + 1) / 2 ===
    nu_add = nodes.new('ShaderNodeMath')
    nu_add.operation = 'ADD'
    nu_add.location = (0, 0)
    nu_add.inputs[1].default_value = 1.0
    links.new(nu_dot.outputs['Value'], nu_add.inputs[0])
    
    u_nu = nodes.new('ShaderNodeMath')
    u_nu.operation = 'DIVIDE'
    u_nu.location = (200, 0)
    u_nu.inputs[1].default_value = 2.0
    links.new(nu_add.outputs['Value'], u_nu.inputs[0])
    
    # === U_MU_S (constant) ===
    u_mu_s_const = nodes.new('ShaderNodeValue')
    u_mu_s_const.location = (2600, 100)
    u_mu_s_const.outputs['Value'].default_value = u_mu_s
    u_mu_s_const.label = f"u_mu_s = {u_mu_s:.4f}"
    
    # === OUTPUT ===
    combine = nodes.new('ShaderNodeCombineXYZ')
    combine.location = (2800, 200)
    links.new(u_mu_final.outputs['Value'], combine.inputs['X'])  # R = u_mu
    links.new(u_mu_s_const.outputs['Value'], combine.inputs['Y'])  # G = u_mu_s
    links.new(u_nu.outputs['Value'], combine.inputs['Z'])  # B = u_nu
    
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (3000, 200)
    emission.inputs['Strength'].default_value = 1.0
    links.new(combine.outputs['Vector'], emission.inputs['Color'])
    
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (3200, 200)
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    # Auto-assign
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
            obj.data.materials.append(mat)
    
    mesh_count = len([o for o in bpy.context.scene.objects if o.type == 'MESH'])
    print(f"\nCreated: {mat.name}")
    print(f"Assigned to {mesh_count} meshes")
    print(f"\nOutput encoding:")
    print(f"  R = u_mu (varies with view direction)")
    print(f"  G = u_mu_s = {u_mu_s:.4f} (constant)")
    print(f"  B = u_nu (varies: 0=away, 1=toward sun)")
    print(f"\nAll values should be in [0, 1]")
    
    return mat


# =============================================================================
# STEP 2.3: SCATTERING LUT SAMPLING
# =============================================================================

def apply_step_2_3_scattering_sample(debug_mode=False):
    """
    Step 2.3: Sample Scattering LUT
    
    Samples scattering.exr using the 4D UV mapping from Step 2.2.
    Includes nu interpolation between two texture samples.
    
    Reference: functions.glsl lines 958-976 (GetScattering)
    
    Args:
        debug_mode: If True, use simple hardcoded UVs to verify texture sampling
    
    Output:
    - RGB = sampled scattering values
    
    Expected:
    - Blue-ish overall (Rayleigh scattering dominates)
    - Brighter toward sun direction
    - Values typically in range 0-0.1 (dim)
    """
    import time
    import os
    
    # Constants
    BOTTOM_RADIUS = 6360.0
    TOP_RADIUS = 6420.0
    MU_S_MIN = -0.2
    
    SCATTERING_R_SIZE = 32
    SCATTERING_MU_SIZE = 128
    SCATTERING_MU_S_SIZE = 32
    SCATTERING_NU_SIZE = 8
    
    CAM_X, CAM_Y, CAM_Z = 37.069, -44.786, 6.0
    
    # LUT path
    lut_paths = [
        r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-3\helios_cache\luts\scattering.exr",
        r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-4\helios_cache\luts\scattering.exr",
    ]
    
    lut_path = None
    for p in lut_paths:
        if os.path.exists(p):
            lut_path = p
            break
    
    if not lut_path:
        print("ERROR: Could not find scattering.exr")
        return None
    
    print(f"Step 2.3: Scattering LUT Sampling")
    print(f"  LUT: {lut_path}")
    
    # Get sun direction
    sun_obj = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN':
            sun_obj = obj
            break
    
    if sun_obj is None:
        print("ERROR: No sun light found")
        return None
    
    import mathutils
    sun_mat = sun_obj.matrix_world
    sun_dir_local = mathutils.Vector((0, 0, -1))
    sun_dir_world = sun_mat.to_3x3() @ sun_dir_local
    sun_dir_world.normalize()
    sun_to = -sun_dir_world
    
    # Pre-compute constants
    cam_rel_z_km = (CAM_Z * 0.001) + BOTTOM_RADIUS
    r_km = math.sqrt((CAM_X * 0.001)**2 + (CAM_Y * 0.001)**2 + cam_rel_z_km**2)
    H = math.sqrt(TOP_RADIUS**2 - BOTTOM_RADIUS**2)
    rho = math.sqrt(max(r_km**2 - BOTTOM_RADIUS**2, 0))
    
    up_x = (CAM_X * 0.001) / r_km
    up_y = (CAM_Y * 0.001) / r_km
    up_z = cam_rel_z_km / r_km
    
    mu_s = up_x * sun_to.x + up_y * sun_to.y + up_z * sun_to.z
    
    # u_r (constant)
    x_r = rho / H
    u_r = 0.5 / SCATTERING_R_SIZE + x_r * (1 - 1/SCATTERING_R_SIZE)
    
    # u_mu_s (constant)
    discrim_s = BOTTOM_RADIUS**2 * (mu_s**2 - 1) + TOP_RADIUS**2
    d_s = max(0, -BOTTOM_RADIUS * mu_s + math.sqrt(max(discrim_s, 0)))
    d_min_s = TOP_RADIUS - BOTTOM_RADIUS
    d_max_s = H
    a = (d_s - d_min_s) / (d_max_s - d_min_s)
    
    discrim_D = BOTTOM_RADIUS**2 * (MU_S_MIN**2 - 1) + TOP_RADIUS**2
    D = max(0, -BOTTOM_RADIUS * MU_S_MIN + math.sqrt(max(discrim_D, 0)))
    A = (D - d_min_s) / (d_max_s - d_min_s)
    
    x_mu_s = max(1.0 - a / A, 0.0) / (1.0 + a)
    u_mu_s = 0.5 / SCATTERING_MU_S_SIZE + x_mu_s * (1 - 1/SCATTERING_MU_S_SIZE)
    
    r_sq = r_km * r_km
    d_min = TOP_RADIUS - r_km
    d_max = rho + H
    half_mu_size = SCATTERING_MU_SIZE / 2
    
    print(f"  u_r = {u_r:.4f}, u_mu_s = {u_mu_s:.4f}")
    
    # Create material
    mat = bpy.data.materials.new(name=f"Step2_3_ScatterSample_{int(time.time())}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Load scattering texture
    if lut_path in bpy.data.images:
        scatter_img = bpy.data.images[lut_path]
    else:
        scatter_img = bpy.data.images.load(lut_path)
        scatter_img.colorspace_settings.name = 'Non-Color'
    
    # === INPUTS ===
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.location = (-1000, 400)
    
    cam_pos = nodes.new('ShaderNodeCombineXYZ')
    cam_pos.location = (-1000, 200)
    cam_pos.inputs['X'].default_value = CAM_X
    cam_pos.inputs['Y'].default_value = CAM_Y
    cam_pos.inputs['Z'].default_value = CAM_Z
    
    up_vec = nodes.new('ShaderNodeCombineXYZ')
    up_vec.location = (-1000, 0)
    up_vec.inputs['X'].default_value = up_x
    up_vec.inputs['Y'].default_value = up_y
    up_vec.inputs['Z'].default_value = up_z
    
    sun_dir_node = nodes.new('ShaderNodeCombineXYZ')
    sun_dir_node.location = (-1000, -200)
    sun_dir_node.inputs['X'].default_value = sun_to.x
    sun_dir_node.inputs['Y'].default_value = sun_to.y
    sun_dir_node.inputs['Z'].default_value = sun_to.z
    
    # === VIEW DIRECTION ===
    view_sub = nodes.new('ShaderNodeVectorMath')
    view_sub.operation = 'SUBTRACT'
    view_sub.location = (-800, 400)
    links.new(geom.outputs['Position'], view_sub.inputs[0])
    links.new(cam_pos.outputs['Vector'], view_sub.inputs[1])
    
    view_norm = nodes.new('ShaderNodeVectorMath')
    view_norm.operation = 'NORMALIZE'
    view_norm.location = (-600, 400)
    links.new(view_sub.outputs['Vector'], view_norm.inputs[0])
    
    # === MU & NU ===
    mu_dot = nodes.new('ShaderNodeVectorMath')
    mu_dot.operation = 'DOT_PRODUCT'
    mu_dot.location = (-400, 300)
    links.new(view_norm.outputs['Vector'], mu_dot.inputs[0])
    links.new(up_vec.outputs['Vector'], mu_dot.inputs[1])
    
    nu_dot = nodes.new('ShaderNodeVectorMath')
    nu_dot.operation = 'DOT_PRODUCT'
    nu_dot.location = (-400, 100)
    links.new(view_norm.outputs['Vector'], nu_dot.inputs[0])
    links.new(sun_dir_node.outputs['Vector'], nu_dot.inputs[1])
    
    # === U_MU CALCULATION ===
    mu_sq = nodes.new('ShaderNodeMath')
    mu_sq.operation = 'MULTIPLY'
    mu_sq.location = (-200, 400)
    links.new(mu_dot.outputs['Value'], mu_sq.inputs[0])
    links.new(mu_dot.outputs['Value'], mu_sq.inputs[1])
    
    mu_sq_m1 = nodes.new('ShaderNodeMath')
    mu_sq_m1.operation = 'SUBTRACT'
    mu_sq_m1.location = (0, 400)
    mu_sq_m1.inputs[1].default_value = 1.0
    links.new(mu_sq.outputs['Value'], mu_sq_m1.inputs[0])
    
    r_sq_term = nodes.new('ShaderNodeMath')
    r_sq_term.operation = 'MULTIPLY'
    r_sq_term.location = (200, 400)
    r_sq_term.inputs[1].default_value = r_sq
    links.new(mu_sq_m1.outputs['Value'], r_sq_term.inputs[0])
    
    discrim = nodes.new('ShaderNodeMath')
    discrim.operation = 'ADD'
    discrim.location = (400, 400)
    discrim.inputs[1].default_value = TOP_RADIUS**2
    links.new(r_sq_term.outputs['Value'], discrim.inputs[0])
    
    discrim_safe = nodes.new('ShaderNodeMath')
    discrim_safe.operation = 'MAXIMUM'
    discrim_safe.location = (600, 400)
    discrim_safe.inputs[1].default_value = 0.0
    links.new(discrim.outputs['Value'], discrim_safe.inputs[0])
    
    discrim_sqrt = nodes.new('ShaderNodeMath')
    discrim_sqrt.operation = 'SQRT'
    discrim_sqrt.location = (800, 400)
    links.new(discrim_safe.outputs['Value'], discrim_sqrt.inputs[0])
    
    neg_r_mu = nodes.new('ShaderNodeMath')
    neg_r_mu.operation = 'MULTIPLY'
    neg_r_mu.location = (200, 200)
    neg_r_mu.inputs[1].default_value = -r_km
    links.new(mu_dot.outputs['Value'], neg_r_mu.inputs[0])
    
    d_dist = nodes.new('ShaderNodeMath')
    d_dist.operation = 'ADD'
    d_dist.location = (1000, 300)
    links.new(neg_r_mu.outputs['Value'], d_dist.inputs[0])
    links.new(discrim_sqrt.outputs['Value'], d_dist.inputs[1])
    
    d_minus_dmin = nodes.new('ShaderNodeMath')
    d_minus_dmin.operation = 'SUBTRACT'
    d_minus_dmin.location = (1200, 300)
    d_minus_dmin.inputs[1].default_value = d_min
    links.new(d_dist.outputs['Value'], d_minus_dmin.inputs[0])
    
    x_mu_node = nodes.new('ShaderNodeMath')
    x_mu_node.operation = 'DIVIDE'
    x_mu_node.location = (1400, 300)
    x_mu_node.inputs[1].default_value = d_max - d_min
    links.new(d_minus_dmin.outputs['Value'], x_mu_node.inputs[0])
    
    x_mu_clamp = nodes.new('ShaderNodeClamp')
    x_mu_clamp.location = (1600, 300)
    links.new(x_mu_node.outputs['Value'], x_mu_clamp.inputs['Value'])
    
    u_mu_inner = nodes.new('ShaderNodeMath')
    u_mu_inner.operation = 'MULTIPLY'
    u_mu_inner.location = (1800, 300)
    u_mu_inner.inputs[1].default_value = 1 - 1/half_mu_size
    links.new(x_mu_clamp.outputs['Result'], u_mu_inner.inputs[0])
    
    u_mu_inner2 = nodes.new('ShaderNodeMath')
    u_mu_inner2.operation = 'ADD'
    u_mu_inner2.location = (2000, 300)
    u_mu_inner2.inputs[1].default_value = 0.5 / half_mu_size
    links.new(u_mu_inner.outputs['Value'], u_mu_inner2.inputs[0])
    
    u_mu_half = nodes.new('ShaderNodeMath')
    u_mu_half.operation = 'MULTIPLY'
    u_mu_half.location = (2200, 300)
    u_mu_half.inputs[1].default_value = 0.5
    links.new(u_mu_inner2.outputs['Value'], u_mu_half.inputs[0])
    
    u_mu = nodes.new('ShaderNodeMath')
    u_mu.operation = 'ADD'
    u_mu.location = (2400, 300)
    u_mu.inputs[1].default_value = 0.5
    links.new(u_mu_half.outputs['Value'], u_mu.inputs[0])
    
    # === U_NU ===
    nu_add = nodes.new('ShaderNodeMath')
    nu_add.operation = 'ADD'
    nu_add.location = (-200, 0)
    nu_add.inputs[1].default_value = 1.0
    links.new(nu_dot.outputs['Value'], nu_add.inputs[0])
    
    u_nu = nodes.new('ShaderNodeMath')
    u_nu.operation = 'DIVIDE'
    u_nu.location = (0, 0)
    u_nu.inputs[1].default_value = 2.0
    links.new(nu_add.outputs['Value'], u_nu.inputs[0])
    
    # === SIMPLIFIED TEXTURE SAMPLING ===
    # For this test, we'll use a simplified single-sample approach
    # The full implementation needs nu interpolation between two samples
    
    # The scattering texture is organized as:
    # x = (tex_x + u_mu_s) / NU_SIZE where tex_x comes from u_nu * (NU_SIZE - 1)
    # y = u_mu
    # z = u_r (depth slice)
    
    # For 2D texture (flattened 3D), we need to compute the proper UV
    # Assuming the texture is stored as depth slices stacked vertically:
    # U = x / NU_SIZE (with nu/mu_s combined)
    # V = (u_mu + u_r * MU_SIZE) / (MU_SIZE * R_SIZE)
    
    # Actually, let's check what the texture dimensions are and sample simply
    # For now, output a simple sample at the computed UVs
    
    # tex_coord_x = u_nu * (NU_SIZE - 1)
    tex_coord_x = nodes.new('ShaderNodeMath')
    tex_coord_x.operation = 'MULTIPLY'
    tex_coord_x.location = (200, 0)
    tex_coord_x.inputs[1].default_value = SCATTERING_NU_SIZE - 1
    links.new(u_nu.outputs['Value'], tex_coord_x.inputs[0])
    
    # tex_x = floor(tex_coord_x)
    tex_x = nodes.new('ShaderNodeMath')
    tex_x.operation = 'FLOOR'
    tex_x.location = (400, 0)
    links.new(tex_coord_x.outputs['Value'], tex_x.inputs[0])
    
    # lerp = tex_coord_x - tex_x
    lerp = nodes.new('ShaderNodeMath')
    lerp.operation = 'SUBTRACT'
    lerp.location = (600, 0)
    links.new(tex_coord_x.outputs['Value'], lerp.inputs[0])
    links.new(tex_x.outputs['Value'], lerp.inputs[1])
    
    # u_mu_s constant
    u_mu_s_const = nodes.new('ShaderNodeValue')
    u_mu_s_const.location = (400, -200)
    u_mu_s_const.outputs['Value'].default_value = u_mu_s
    
    # u_r constant
    u_r_const = nodes.new('ShaderNodeValue')
    u_r_const.location = (400, -350)
    u_r_const.outputs['Value'].default_value = u_r
    
    # uvw0.x = (tex_x + u_mu_s) / NU_SIZE
    uvw0_x_add = nodes.new('ShaderNodeMath')
    uvw0_x_add.operation = 'ADD'
    uvw0_x_add.location = (800, -100)
    links.new(tex_x.outputs['Value'], uvw0_x_add.inputs[0])
    links.new(u_mu_s_const.outputs['Value'], uvw0_x_add.inputs[1])
    
    uvw0_x = nodes.new('ShaderNodeMath')
    uvw0_x.operation = 'DIVIDE'
    uvw0_x.location = (1000, -100)
    uvw0_x.inputs[1].default_value = SCATTERING_NU_SIZE
    links.new(uvw0_x_add.outputs['Value'], uvw0_x.inputs[0])
    
    # uvw1.x = (tex_x + 1 + u_mu_s) / NU_SIZE
    uvw1_x_add = nodes.new('ShaderNodeMath')
    uvw1_x_add.operation = 'ADD'
    uvw1_x_add.location = (800, -300)
    uvw1_x_add.inputs[1].default_value = 1.0
    links.new(uvw0_x_add.outputs['Value'], uvw1_x_add.inputs[0])
    
    uvw1_x = nodes.new('ShaderNodeMath')
    uvw1_x.operation = 'DIVIDE'
    uvw1_x.location = (1000, -300)
    uvw1_x.inputs[1].default_value = SCATTERING_NU_SIZE
    links.new(uvw1_x_add.outputs['Value'], uvw1_x.inputs[0])
    
    # 3D→2D UV mapping (from aerial_nodes.py reference):
    # U = (depth_index + uvw_x) / DEPTH
    # V = 1.0 - u_mu  (CRITICAL: Blender Y is flipped!)
    
    # Depth index from u_r
    depth_scaled = nodes.new('ShaderNodeMath')
    depth_scaled.operation = 'MULTIPLY'
    depth_scaled.location = (1100, -400)
    depth_scaled.inputs[1].default_value = float(SCATTERING_R_SIZE - 1)
    links.new(u_r_const.outputs['Value'], depth_scaled.inputs[0])
    
    depth_floor = nodes.new('ShaderNodeMath')
    depth_floor.operation = 'FLOOR'
    depth_floor.location = (1300, -400)
    links.new(depth_scaled.outputs['Value'], depth_floor.inputs[0])
    
    # Final U0 = (depth_floor + uvw0_x) / DEPTH
    u0_sum = nodes.new('ShaderNodeMath')
    u0_sum.operation = 'ADD'
    u0_sum.location = (1500, 100)
    links.new(depth_floor.outputs['Value'], u0_sum.inputs[0])
    links.new(uvw0_x.outputs['Value'], u0_sum.inputs[1])
    
    final_u0 = nodes.new('ShaderNodeMath')
    final_u0.operation = 'DIVIDE'
    final_u0.location = (1700, 100)
    final_u0.inputs[1].default_value = float(SCATTERING_R_SIZE)
    links.new(u0_sum.outputs['Value'], final_u0.inputs[0])
    
    # Final U1 = (depth_floor + uvw1_x) / DEPTH
    u1_sum = nodes.new('ShaderNodeMath')
    u1_sum.operation = 'ADD'
    u1_sum.location = (1500, -100)
    links.new(depth_floor.outputs['Value'], u1_sum.inputs[0])
    links.new(uvw1_x.outputs['Value'], u1_sum.inputs[1])
    
    final_u1 = nodes.new('ShaderNodeMath')
    final_u1.operation = 'DIVIDE'
    final_u1.location = (1700, -100)
    final_u1.inputs[1].default_value = float(SCATTERING_R_SIZE)
    links.new(u1_sum.outputs['Value'], final_u1.inputs[0])
    
    # V = 1.0 - u_mu (flip Y for Blender texture convention)
    v_flip = nodes.new('ShaderNodeMath')
    v_flip.operation = 'SUBTRACT'
    v_flip.location = (1500, -250)
    v_flip.inputs[0].default_value = 1.0
    links.new(u_mu.outputs['Value'], v_flip.inputs[1])
    
    # Combine UVs
    uv0_combine = nodes.new('ShaderNodeCombineXYZ')
    uv0_combine.location = (1900, 100)
    uv0_combine.inputs['Z'].default_value = 0.0
    links.new(final_u0.outputs['Value'], uv0_combine.inputs['X'])
    links.new(v_flip.outputs['Value'], uv0_combine.inputs['Y'])
    
    uv1_combine = nodes.new('ShaderNodeCombineXYZ')
    uv1_combine.location = (1900, -100)
    uv1_combine.inputs['Z'].default_value = 0.0
    links.new(final_u1.outputs['Value'], uv1_combine.inputs['X'])
    links.new(v_flip.outputs['Value'], uv1_combine.inputs['Y'])
    
    # Sample texture at both UVs
    tex0 = nodes.new('ShaderNodeTexImage')
    tex0.location = (2100, 100)
    tex0.interpolation = 'Linear'
    tex0.extension = 'EXTEND'
    tex0.image = scatter_img
    links.new(uv0_combine.outputs['Vector'], tex0.inputs['Vector'])
    
    tex1 = nodes.new('ShaderNodeTexImage')
    tex1.location = (2100, -100)
    tex1.interpolation = 'Linear'
    tex1.extension = 'EXTEND'
    tex1.image = scatter_img
    links.new(uv1_combine.outputs['Vector'], tex1.inputs['Vector'])
    
    # Interpolate: result = tex0 * (1 - lerp) + tex1 * lerp
    lerp_mix = nodes.new('ShaderNodeMix')
    lerp_mix.data_type = 'RGBA'
    lerp_mix.blend_type = 'MIX'
    lerp_mix.location = (2300, 0)
    links.new(lerp.outputs['Value'], lerp_mix.inputs['Factor'])
    links.new(tex0.outputs['Color'], lerp_mix.inputs['A'])
    links.new(tex1.outputs['Color'], lerp_mix.inputs['B'])
    
    # Output raw values
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (2500, 0)
    emission.inputs['Strength'].default_value = 1.0
    links.new(lerp_mix.outputs['Result'], emission.inputs['Color'])
    
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (2700, 0)
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    # Auto-assign
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
            obj.data.materials.append(mat)
    
    mesh_count = len([o for o in bpy.context.scene.objects if o.type == 'MESH'])
    print(f"\nCreated: {mat.name}")
    print(f"Assigned to {mesh_count} meshes")
    print(f"\nNote: This is a simplified 2D sampling. Values may be dim.")
    print(f"Expected: Blue-ish scattering, brighter toward sun")
    
    return mat


# =============================================================================
# STEP 2.4: FULL INSCATTER COMPUTATION
# =============================================================================

def apply_step_2_4_inscatter():
    """
    Step 2.4: Full Inscatter Computation
    
    Combines all validated components:
    - Distance d from camera to point
    - Camera parameters (r, mu, mu_s, nu)
    - Point parameters (r_p, mu_p, mu_s_p) via law of cosines
    - Scattering samples at camera and point
    - Transmittance between camera and point
    - Final inscatter = S_cam - T × S_pt
    
    Reference: GetSkyRadianceToPoint (functions.glsl lines 1787-1863)
    
    Output: RGB inscatter (Rayleigh-dominated, blue-ish)
    """
    import time
    import os
    import math
    
    print("Step 2.4: Full Inscatter Computation")
    
    # Constants
    BOTTOM_RADIUS = 6360.0
    TOP_RADIUS = 6420.0
    H = math.sqrt(TOP_RADIUS**2 - BOTTOM_RADIUS**2)
    MU_S_MIN = -0.2
    
    SCATTERING_R_SIZE = 32
    SCATTERING_MU_SIZE = 128
    SCATTERING_MU_S_SIZE = 32
    SCATTERING_NU_SIZE = 8
    
    TRANSMITTANCE_WIDTH = 256
    TRANSMITTANCE_HEIGHT = 64
    
    # Get LUT path
    lut_dir = r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-3\helios_cache\luts"
    scatter_path = os.path.join(lut_dir, "scattering.exr")
    trans_path = os.path.join(lut_dir, "transmittance.exr")
    
    if not os.path.exists(scatter_path):
        print(f"  ERROR: {scatter_path} not found")
        return None
    if not os.path.exists(trans_path):
        print(f"  ERROR: {trans_path} not found")
        return None
    
    print(f"  Scattering: {scatter_path}")
    print(f"  Transmittance: {trans_path}")
    
    # Load textures
    scatter_img = bpy.data.images.load(scatter_path, check_existing=True)
    scatter_img.colorspace_settings.name = 'Non-Color'
    
    trans_img = bpy.data.images.load(trans_path, check_existing=True)
    trans_img.colorspace_settings.name = 'Non-Color'
    
    # Create material
    mat_name = f"Step2_4_Inscatter_{int(time.time())}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # ==========================================================================
    # INPUTS: Geometry and camera
    # ==========================================================================
    
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.location = (-1600, 400)
    
    cam_loc = nodes.new('ShaderNodeCombineXYZ')
    cam_loc.location = (-1600, 200)
    cam = bpy.context.scene.camera
    if cam:
        cam_loc.inputs['X'].default_value = cam.location.x
        cam_loc.inputs['Y'].default_value = cam.location.y
        cam_loc.inputs['Z'].default_value = cam.location.z
    
    # Find sun light in scene and get its direction
    sun_light = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN':
            sun_light = obj
            break
    
    sun_dir = nodes.new('ShaderNodeCombineXYZ')
    sun_dir.location = (-1600, 0)
    if sun_light:
        # Sun points in -Z direction in local space, transform to world
        import mathutils
        sun_direction = sun_light.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))
        sun_direction.normalize()
        sun_dir.inputs['X'].default_value = sun_direction.x
        sun_dir.inputs['Y'].default_value = sun_direction.y
        sun_dir.inputs['Z'].default_value = sun_direction.z
        print(f"  Sun direction: ({sun_direction.x:.3f}, {sun_direction.y:.3f}, {sun_direction.z:.3f})")
    else:
        # Fallback: 15° elevation
        sun_dir.inputs['X'].default_value = 0.0
        sun_dir.inputs['Y'].default_value = 0.2588  # sin(15°)
        sun_dir.inputs['Z'].default_value = 0.9659  # cos(15°)
        print(f"  Sun: No sun light found, using default 15° elevation")
    
    # ==========================================================================
    # DISTANCE d (validated in Step 1.1)
    # ==========================================================================
    
    pos_to_cam = nodes.new('ShaderNodeVectorMath')
    pos_to_cam.operation = 'SUBTRACT'
    pos_to_cam.location = (-1400, 300)
    links.new(geom.outputs['Position'], pos_to_cam.inputs[0])
    links.new(cam_loc.outputs['Vector'], pos_to_cam.inputs[1])
    
    d_vec_len = nodes.new('ShaderNodeVectorMath')
    d_vec_len.operation = 'LENGTH'
    d_vec_len.location = (-1200, 300)
    links.new(pos_to_cam.outputs['Vector'], d_vec_len.inputs[0])
    
    # d in km (scene scale = 0.001)
    d = nodes.new('ShaderNodeMath')
    d.operation = 'MULTIPLY'
    d.location = (-1000, 300)
    d.inputs[1].default_value = 0.001
    links.new(d_vec_len.outputs['Value'], d.inputs[0])
    
    # ==========================================================================
    # VIEW DIRECTION (normalized)
    # ==========================================================================
    
    view_dir = nodes.new('ShaderNodeVectorMath')
    view_dir.operation = 'NORMALIZE'
    view_dir.location = (-1200, 150)
    links.new(pos_to_cam.outputs['Vector'], view_dir.inputs[0])
    
    # ==========================================================================
    # CAMERA r, mu (validated in Step 1.2)
    # ==========================================================================
    
    # Camera altitude -> r
    cam_alt_km = (cam.location.z * 0.001) if cam else 0.218
    r_cam = BOTTOM_RADIUS + cam_alt_km
    
    r = nodes.new('ShaderNodeValue')
    r.location = (-1000, 100)
    r.outputs['Value'].default_value = r_cam
    
    # up_at_camera (normalized position on sphere)
    up_at_cam = nodes.new('ShaderNodeCombineXYZ')
    up_at_cam.location = (-1000, 0)
    up_at_cam.inputs['X'].default_value = 0.0
    up_at_cam.inputs['Y'].default_value = 0.0
    up_at_cam.inputs['Z'].default_value = 1.0
    
    # mu = dot(view_dir, up)
    mu_dot = nodes.new('ShaderNodeVectorMath')
    mu_dot.operation = 'DOT_PRODUCT'
    mu_dot.location = (-800, 100)
    links.new(view_dir.outputs['Vector'], mu_dot.inputs[0])
    links.new(up_at_cam.outputs['Vector'], mu_dot.inputs[1])
    
    mu = nodes.new('ShaderNodeMath')
    mu.operation = 'MULTIPLY'
    mu.location = (-600, 100)
    mu.inputs[1].default_value = 1.0  # Pass through
    links.new(mu_dot.outputs['Value'], mu.inputs[0])
    
    # ==========================================================================
    # SUN PARAMETERS mu_s, nu (validated in Step 2.1)
    # ==========================================================================
    
    # mu_s = dot(up, sun_dir)
    mu_s_dot = nodes.new('ShaderNodeVectorMath')
    mu_s_dot.operation = 'DOT_PRODUCT'
    mu_s_dot.location = (-800, -100)
    links.new(up_at_cam.outputs['Vector'], mu_s_dot.inputs[0])
    links.new(sun_dir.outputs['Vector'], mu_s_dot.inputs[1])
    
    mu_s = nodes.new('ShaderNodeMath')
    mu_s.operation = 'MULTIPLY'
    mu_s.location = (-600, -100)
    mu_s.inputs[1].default_value = 1.0
    links.new(mu_s_dot.outputs['Value'], mu_s.inputs[0])
    
    # nu = dot(view_dir, sun_dir)
    nu_dot = nodes.new('ShaderNodeVectorMath')
    nu_dot.operation = 'DOT_PRODUCT'
    nu_dot.location = (-800, -250)
    links.new(view_dir.outputs['Vector'], nu_dot.inputs[0])
    links.new(sun_dir.outputs['Vector'], nu_dot.inputs[1])
    
    nu = nodes.new('ShaderNodeMath')
    nu.operation = 'MULTIPLY'
    nu.location = (-600, -250)
    nu.inputs[1].default_value = 1.0
    links.new(nu_dot.outputs['Value'], nu.inputs[0])
    
    # ==========================================================================
    # POINT PARAMETERS r_p, mu_p, mu_s_p (law of cosines)
    # Reference: lines 1832-1834
    # ==========================================================================
    
    # r² 
    r_sq = nodes.new('ShaderNodeMath')
    r_sq.operation = 'MULTIPLY'
    r_sq.location = (-400, -400)
    r_sq.inputs[0].default_value = r_cam
    r_sq.inputs[1].default_value = r_cam
    
    # d²
    d_sq = nodes.new('ShaderNodeMath')
    d_sq.operation = 'MULTIPLY'
    d_sq.location = (-400, -500)
    links.new(d.outputs['Value'], d_sq.inputs[0])
    links.new(d.outputs['Value'], d_sq.inputs[1])
    
    # 2·r·μ
    two_r = nodes.new('ShaderNodeMath')
    two_r.operation = 'MULTIPLY'
    two_r.location = (-400, -600)
    two_r.inputs[0].default_value = 2.0 * r_cam
    links.new(mu.outputs['Value'], two_r.inputs[1])
    
    # 2·r·μ·d
    two_r_mu_d = nodes.new('ShaderNodeMath')
    two_r_mu_d.operation = 'MULTIPLY'
    two_r_mu_d.location = (-200, -550)
    links.new(two_r.outputs['Value'], two_r_mu_d.inputs[0])
    links.new(d.outputs['Value'], two_r_mu_d.inputs[1])
    
    # d² + 2·r·μ·d
    sum1 = nodes.new('ShaderNodeMath')
    sum1.operation = 'ADD'
    sum1.location = (0, -500)
    links.new(d_sq.outputs['Value'], sum1.inputs[0])
    links.new(two_r_mu_d.outputs['Value'], sum1.inputs[1])
    
    # d² + 2·r·μ·d + r²
    sum2 = nodes.new('ShaderNodeMath')
    sum2.operation = 'ADD'
    sum2.location = (200, -500)
    links.new(sum1.outputs['Value'], sum2.inputs[0])
    links.new(r_sq.outputs['Value'], sum2.inputs[1])
    
    # r_p = sqrt(...)
    r_p_raw = nodes.new('ShaderNodeMath')
    r_p_raw.operation = 'SQRT'
    r_p_raw.location = (400, -500)
    links.new(sum2.outputs['Value'], r_p_raw.inputs[0])
    
    # Clamp r_p to [BOTTOM_RADIUS, TOP_RADIUS]
    r_p_min = nodes.new('ShaderNodeMath')
    r_p_min.operation = 'MAXIMUM'
    r_p_min.location = (600, -500)
    r_p_min.inputs[1].default_value = BOTTOM_RADIUS
    links.new(r_p_raw.outputs['Value'], r_p_min.inputs[0])
    
    r_p = nodes.new('ShaderNodeMath')
    r_p.operation = 'MINIMUM'
    r_p.location = (800, -500)
    r_p.inputs[1].default_value = TOP_RADIUS
    links.new(r_p_min.outputs['Value'], r_p.inputs[0])
    
    # mu_p = (r·μ + d) / r_p
    r_mu = nodes.new('ShaderNodeMath')
    r_mu.operation = 'MULTIPLY'
    r_mu.location = (-200, -700)
    r_mu.inputs[0].default_value = r_cam
    links.new(mu.outputs['Value'], r_mu.inputs[1])
    
    r_mu_plus_d = nodes.new('ShaderNodeMath')
    r_mu_plus_d.operation = 'ADD'
    r_mu_plus_d.location = (0, -700)
    links.new(r_mu.outputs['Value'], r_mu_plus_d.inputs[0])
    links.new(d.outputs['Value'], r_mu_plus_d.inputs[1])
    
    mu_p_raw = nodes.new('ShaderNodeMath')
    mu_p_raw.operation = 'DIVIDE'
    mu_p_raw.location = (200, -700)
    links.new(r_mu_plus_d.outputs['Value'], mu_p_raw.inputs[0])
    links.new(r_p.outputs['Value'], mu_p_raw.inputs[1])
    
    # Clamp mu_p to [-1, 1]
    mu_p_max = nodes.new('ShaderNodeMath')
    mu_p_max.operation = 'MINIMUM'
    mu_p_max.location = (400, -700)
    mu_p_max.inputs[1].default_value = 1.0
    links.new(mu_p_raw.outputs['Value'], mu_p_max.inputs[0])
    
    mu_p = nodes.new('ShaderNodeMath')
    mu_p.operation = 'MAXIMUM'
    mu_p.location = (600, -700)
    mu_p.inputs[1].default_value = -1.0
    links.new(mu_p_max.outputs['Value'], mu_p.inputs[0])
    
    # mu_s_p = (r·μ_s + d·ν) / r_p
    r_mu_s = nodes.new('ShaderNodeMath')
    r_mu_s.operation = 'MULTIPLY'
    r_mu_s.location = (-200, -850)
    r_mu_s.inputs[0].default_value = r_cam
    links.new(mu_s.outputs['Value'], r_mu_s.inputs[1])
    
    d_nu = nodes.new('ShaderNodeMath')
    d_nu.operation = 'MULTIPLY'
    d_nu.location = (-200, -950)
    links.new(d.outputs['Value'], d_nu.inputs[0])
    links.new(nu.outputs['Value'], d_nu.inputs[1])
    
    r_mu_s_plus_d_nu = nodes.new('ShaderNodeMath')
    r_mu_s_plus_d_nu.operation = 'ADD'
    r_mu_s_plus_d_nu.location = (0, -850)
    links.new(r_mu_s.outputs['Value'], r_mu_s_plus_d_nu.inputs[0])
    links.new(d_nu.outputs['Value'], r_mu_s_plus_d_nu.inputs[1])
    
    mu_s_p_raw = nodes.new('ShaderNodeMath')
    mu_s_p_raw.operation = 'DIVIDE'
    mu_s_p_raw.location = (200, -850)
    links.new(r_mu_s_plus_d_nu.outputs['Value'], mu_s_p_raw.inputs[0])
    links.new(r_p.outputs['Value'], mu_s_p_raw.inputs[1])
    
    # Clamp mu_s_p to [-1, 1]
    mu_s_p_max = nodes.new('ShaderNodeMath')
    mu_s_p_max.operation = 'MINIMUM'
    mu_s_p_max.location = (400, -850)
    mu_s_p_max.inputs[1].default_value = 1.0
    links.new(mu_s_p_raw.outputs['Value'], mu_s_p_max.inputs[0])
    
    mu_s_p = nodes.new('ShaderNodeMath')
    mu_s_p.operation = 'MAXIMUM'
    mu_s_p.location = (600, -850)
    mu_s_p.inputs[1].default_value = -1.0
    links.new(mu_s_p_max.outputs['Value'], mu_s_p.inputs[0])
    
    # ==========================================================================
    # HELPER: Create scattering UV nodes
    # ==========================================================================
    
    def create_scatter_uv(prefix, r_val, mu_node, mu_s_node, nu_node, base_x, base_y):
        """Create scattering texture UV coordinates for given parameters."""
        
        # u_r = GetTextureCoordFromUnitRange(rho/H, R_SIZE)
        # rho = sqrt(r² - bottom²)
        if isinstance(r_val, float):
            rho_val = math.sqrt(r_val**2 - BOTTOM_RADIUS**2)
            u_r_val = (rho_val / H) * (SCATTERING_R_SIZE - 1) / SCATTERING_R_SIZE + 0.5 / SCATTERING_R_SIZE
        else:
            # r_val is a node - need to compute dynamically
            rho_sq = nodes.new('ShaderNodeMath')
            rho_sq.operation = 'SUBTRACT'
            rho_sq.location = (base_x, base_y - 100)
            links.new(r_val.outputs['Value'], rho_sq.inputs[0])
            # r² - bottom²: first compute r²
            r_sq_dyn = nodes.new('ShaderNodeMath')
            r_sq_dyn.operation = 'MULTIPLY'
            r_sq_dyn.location = (base_x - 200, base_y - 100)
            links.new(r_val.outputs['Value'], r_sq_dyn.inputs[0])
            links.new(r_val.outputs['Value'], r_sq_dyn.inputs[1])
            rho_sq.inputs[1].default_value = BOTTOM_RADIUS**2
            links.new(r_sq_dyn.outputs['Value'], rho_sq.inputs[0])
            
            rho = nodes.new('ShaderNodeMath')
            rho.operation = 'SQRT'
            rho.location = (base_x + 200, base_y - 100)
            links.new(rho_sq.outputs['Value'], rho.inputs[0])
            
            rho_over_H = nodes.new('ShaderNodeMath')
            rho_over_H.operation = 'DIVIDE'
            rho_over_H.location = (base_x + 400, base_y - 100)
            rho_over_H.inputs[1].default_value = H
            links.new(rho.outputs['Value'], rho_over_H.inputs[0])
            
            u_r_scale = nodes.new('ShaderNodeMath')
            u_r_scale.operation = 'MULTIPLY'
            u_r_scale.location = (base_x + 600, base_y - 100)
            u_r_scale.inputs[1].default_value = (SCATTERING_R_SIZE - 1) / SCATTERING_R_SIZE
            links.new(rho_over_H.outputs['Value'], u_r_scale.inputs[0])
            
            u_r_node = nodes.new('ShaderNodeMath')
            u_r_node.operation = 'ADD'
            u_r_node.location = (base_x + 800, base_y - 100)
            u_r_node.inputs[1].default_value = 0.5 / SCATTERING_R_SIZE
            links.new(u_r_scale.outputs['Value'], u_r_node.inputs[0])
            u_r_val = u_r_node
        
        # u_mu - PROPER BRUNETON FORMULA (non-ground rays)
        # Reference: GetScatteringTextureUvwzFromRMuMuSNu lines 803-812
        # d = -r*mu + sqrt(r²(μ²-1) + top²)
        # d_min = top - r, d_max = rho + H
        # x_mu = (d - d_min) / (d_max - d_min)
        # u_mu = 0.5 + 0.5 * GetTextureCoordFromUnitRange(x_mu, MU_SIZE/2)
        
        # Get r value for calculations
        if isinstance(r_val, float):
            r_for_mu = r_val
            rho_for_mu = math.sqrt(r_val**2 - BOTTOM_RADIUS**2)
        else:
            r_for_mu = r_val  # node
            rho_for_mu = rho  # node from u_r calculation above
        
        # mu²
        mu_sq = nodes.new('ShaderNodeMath')
        mu_sq.operation = 'MULTIPLY'
        mu_sq.location = (base_x, base_y)
        links.new(mu_node.outputs['Value'], mu_sq.inputs[0])
        links.new(mu_node.outputs['Value'], mu_sq.inputs[1])
        
        # μ² - 1
        mu_sq_m1 = nodes.new('ShaderNodeMath')
        mu_sq_m1.operation = 'SUBTRACT'
        mu_sq_m1.location = (base_x + 150, base_y)
        links.new(mu_sq.outputs['Value'], mu_sq_m1.inputs[0])
        mu_sq_m1.inputs[1].default_value = 1.0
        
        # r² × (μ² - 1)
        if isinstance(r_for_mu, float):
            r_sq_x_mu_term = nodes.new('ShaderNodeMath')
            r_sq_x_mu_term.operation = 'MULTIPLY'
            r_sq_x_mu_term.location = (base_x + 300, base_y)
            r_sq_x_mu_term.inputs[0].default_value = r_for_mu ** 2
            links.new(mu_sq_m1.outputs['Value'], r_sq_x_mu_term.inputs[1])
        else:
            r_sq_for_mu = nodes.new('ShaderNodeMath')
            r_sq_for_mu.operation = 'MULTIPLY'
            r_sq_for_mu.location = (base_x + 200, base_y - 50)
            links.new(r_for_mu.outputs['Value'], r_sq_for_mu.inputs[0])
            links.new(r_for_mu.outputs['Value'], r_sq_for_mu.inputs[1])
            
            r_sq_x_mu_term = nodes.new('ShaderNodeMath')
            r_sq_x_mu_term.operation = 'MULTIPLY'
            r_sq_x_mu_term.location = (base_x + 350, base_y)
            links.new(r_sq_for_mu.outputs['Value'], r_sq_x_mu_term.inputs[0])
            links.new(mu_sq_m1.outputs['Value'], r_sq_x_mu_term.inputs[1])
        
        # disc = r²(μ²-1) + top²
        disc = nodes.new('ShaderNodeMath')
        disc.operation = 'ADD'
        disc.location = (base_x + 500, base_y)
        disc.inputs[1].default_value = TOP_RADIUS ** 2
        links.new(r_sq_x_mu_term.outputs['Value'], disc.inputs[0])
        
        # sqrt(disc) - clamp to avoid sqrt of negative
        disc_safe = nodes.new('ShaderNodeMath')
        disc_safe.operation = 'MAXIMUM'
        disc_safe.location = (base_x + 650, base_y)
        disc_safe.inputs[1].default_value = 0.0
        links.new(disc.outputs['Value'], disc_safe.inputs[0])
        
        disc_sqrt = nodes.new('ShaderNodeMath')
        disc_sqrt.operation = 'SQRT'
        disc_sqrt.location = (base_x + 800, base_y)
        links.new(disc_safe.outputs['Value'], disc_sqrt.inputs[0])
        
        # -r * mu
        if isinstance(r_for_mu, float):
            neg_r_mu = nodes.new('ShaderNodeMath')
            neg_r_mu.operation = 'MULTIPLY'
            neg_r_mu.location = (base_x + 650, base_y + 50)
            neg_r_mu.inputs[0].default_value = -r_for_mu
            links.new(mu_node.outputs['Value'], neg_r_mu.inputs[1])
        else:
            r_mu_node = nodes.new('ShaderNodeMath')
            r_mu_node.operation = 'MULTIPLY'
            r_mu_node.location = (base_x + 500, base_y + 50)
            links.new(r_for_mu.outputs['Value'], r_mu_node.inputs[0])
            links.new(mu_node.outputs['Value'], r_mu_node.inputs[1])
            
            neg_r_mu = nodes.new('ShaderNodeMath')
            neg_r_mu.operation = 'MULTIPLY'
            neg_r_mu.location = (base_x + 650, base_y + 50)
            neg_r_mu.inputs[1].default_value = -1.0
            links.new(r_mu_node.outputs['Value'], neg_r_mu.inputs[0])
        
        # d = -r*mu + sqrt(disc)
        d_nonground = nodes.new('ShaderNodeMath')
        d_nonground.operation = 'ADD'
        d_nonground.location = (base_x + 950, base_y)
        links.new(neg_r_mu.outputs['Value'], d_nonground.inputs[0])
        links.new(disc_sqrt.outputs['Value'], d_nonground.inputs[1])
        
        # d_min = top - r
        if isinstance(r_for_mu, float):
            d_min = nodes.new('ShaderNodeValue')
            d_min.location = (base_x + 800, base_y + 100)
            d_min.outputs['Value'].default_value = TOP_RADIUS - r_for_mu
        else:
            d_min = nodes.new('ShaderNodeMath')
            d_min.operation = 'SUBTRACT'
            d_min.location = (base_x + 800, base_y + 100)
            d_min.inputs[0].default_value = TOP_RADIUS
            links.new(r_for_mu.outputs['Value'], d_min.inputs[1])
        
        # d_max = rho + H
        if isinstance(rho_for_mu, float):
            d_max = nodes.new('ShaderNodeValue')
            d_max.location = (base_x + 800, base_y + 150)
            d_max.outputs['Value'].default_value = rho_for_mu + H
        else:
            d_max = nodes.new('ShaderNodeMath')
            d_max.operation = 'ADD'
            d_max.location = (base_x + 800, base_y + 150)
            d_max.inputs[1].default_value = H
            links.new(rho_for_mu.outputs['Value'], d_max.inputs[0])
        
        # x_mu = (d - d_min) / (d_max - d_min)
        d_minus_dmin = nodes.new('ShaderNodeMath')
        d_minus_dmin.operation = 'SUBTRACT'
        d_minus_dmin.location = (base_x + 1100, base_y)
        links.new(d_nonground.outputs['Value'], d_minus_dmin.inputs[0])
        links.new(d_min.outputs['Value'], d_minus_dmin.inputs[1])
        
        dmax_minus_dmin = nodes.new('ShaderNodeMath')
        dmax_minus_dmin.operation = 'SUBTRACT'
        dmax_minus_dmin.location = (base_x + 1100, base_y + 100)
        links.new(d_max.outputs['Value'], dmax_minus_dmin.inputs[0])
        links.new(d_min.outputs['Value'], dmax_minus_dmin.inputs[1])
        
        # Safe divide
        denom_safe = nodes.new('ShaderNodeMath')
        denom_safe.operation = 'MAXIMUM'
        denom_safe.location = (base_x + 1250, base_y + 100)
        denom_safe.inputs[1].default_value = 0.001
        links.new(dmax_minus_dmin.outputs['Value'], denom_safe.inputs[0])
        
        x_mu = nodes.new('ShaderNodeMath')
        x_mu.operation = 'DIVIDE'
        x_mu.location = (base_x + 1400, base_y)
        links.new(d_minus_dmin.outputs['Value'], x_mu.inputs[0])
        links.new(denom_safe.outputs['Value'], x_mu.inputs[1])
        
        # Clamp x_mu to [0, 1]
        x_mu_clamped = nodes.new('ShaderNodeClamp')
        x_mu_clamped.location = (base_x + 1550, base_y)
        x_mu_clamped.inputs['Min'].default_value = 0.0
        x_mu_clamped.inputs['Max'].default_value = 1.0
        links.new(x_mu.outputs['Value'], x_mu_clamped.inputs['Value'])
        
        # GetTextureCoordFromUnitRange for MU_SIZE/2, then u_mu = 0.5 + 0.5 * coord
        mu_scale = 1.0 - 2.0 / SCATTERING_MU_SIZE
        mu_offset = 1.0 / SCATTERING_MU_SIZE
        
        x_mu_scaled = nodes.new('ShaderNodeMath')
        x_mu_scaled.operation = 'MULTIPLY'
        x_mu_scaled.location = (base_x + 1700, base_y)
        x_mu_scaled.inputs[1].default_value = mu_scale
        links.new(x_mu_clamped.outputs['Result'], x_mu_scaled.inputs[0])
        
        x_mu_offset = nodes.new('ShaderNodeMath')
        x_mu_offset.operation = 'ADD'
        x_mu_offset.location = (base_x + 1850, base_y)
        x_mu_offset.inputs[1].default_value = mu_offset
        links.new(x_mu_scaled.outputs['Value'], x_mu_offset.inputs[0])
        
        # u_mu = 0.5 + 0.5 * coord (maps to [0.5, 1.0] for non-ground)
        u_mu_half = nodes.new('ShaderNodeMath')
        u_mu_half.operation = 'MULTIPLY'
        u_mu_half.location = (base_x + 2000, base_y)
        u_mu_half.inputs[1].default_value = 0.5
        links.new(x_mu_offset.outputs['Value'], u_mu_half.inputs[0])
        
        u_mu = nodes.new('ShaderNodeMath')
        u_mu.operation = 'ADD'
        u_mu.location = (base_x + 2150, base_y)
        u_mu.inputs[0].default_value = 0.5
        links.new(u_mu_half.outputs['Value'], u_mu.inputs[1])
        
        # u_mu_s
        u_mu_s_val = nodes.new('ShaderNodeMath')
        u_mu_s_val.operation = 'SUBTRACT'
        u_mu_s_val.location = (base_x, base_y + 100)
        links.new(mu_s_node.outputs['Value'], u_mu_s_val.inputs[0])
        u_mu_s_val.inputs[1].default_value = MU_S_MIN
        
        u_mu_s_div = nodes.new('ShaderNodeMath')
        u_mu_s_div.operation = 'DIVIDE'
        u_mu_s_div.location = (base_x + 200, base_y + 100)
        u_mu_s_div.inputs[1].default_value = 1.0 - MU_S_MIN
        links.new(u_mu_s_val.outputs['Value'], u_mu_s_div.inputs[0])
        
        u_mu_s_scale = nodes.new('ShaderNodeMath')
        u_mu_s_scale.operation = 'MULTIPLY'
        u_mu_s_scale.location = (base_x + 400, base_y + 100)
        u_mu_s_scale.inputs[1].default_value = (SCATTERING_MU_S_SIZE - 1) / SCATTERING_MU_S_SIZE
        links.new(u_mu_s_div.outputs['Value'], u_mu_s_scale.inputs[0])
        
        u_mu_s = nodes.new('ShaderNodeMath')
        u_mu_s.operation = 'ADD'
        u_mu_s.location = (base_x + 600, base_y + 100)
        u_mu_s.inputs[1].default_value = 0.5 / SCATTERING_MU_S_SIZE
        links.new(u_mu_s_scale.outputs['Value'], u_mu_s.inputs[0])
        
        # u_nu
        nu_plus_1 = nodes.new('ShaderNodeMath')
        nu_plus_1.operation = 'ADD'
        nu_plus_1.location = (base_x, base_y + 200)
        nu_plus_1.inputs[1].default_value = 1.0
        links.new(nu_node.outputs['Value'], nu_plus_1.inputs[0])
        
        u_nu = nodes.new('ShaderNodeMath')
        u_nu.operation = 'MULTIPLY'
        u_nu.location = (base_x + 200, base_y + 200)
        u_nu.inputs[1].default_value = 0.5
        links.new(nu_plus_1.outputs['Value'], u_nu.inputs[0])
        
        # tex_x = floor(u_nu * (NU_SIZE - 1))
        tex_coord_x = nodes.new('ShaderNodeMath')
        tex_coord_x.operation = 'MULTIPLY'
        tex_coord_x.location = (base_x + 400, base_y + 200)
        tex_coord_x.inputs[1].default_value = SCATTERING_NU_SIZE - 1
        links.new(u_nu.outputs['Value'], tex_coord_x.inputs[0])
        
        tex_x_floor = nodes.new('ShaderNodeMath')
        tex_x_floor.operation = 'FLOOR'
        tex_x_floor.location = (base_x + 600, base_y + 200)
        links.new(tex_coord_x.outputs['Value'], tex_x_floor.inputs[0])
        
        # uvw_x = (tex_x + u_mu_s) / NU_SIZE
        tex_x_plus_mus = nodes.new('ShaderNodeMath')
        tex_x_plus_mus.operation = 'ADD'
        tex_x_plus_mus.location = (base_x + 800, base_y + 150)
        links.new(tex_x_floor.outputs['Value'], tex_x_plus_mus.inputs[0])
        links.new(u_mu_s.outputs['Value'], tex_x_plus_mus.inputs[1])
        
        uvw_x = nodes.new('ShaderNodeMath')
        uvw_x.operation = 'DIVIDE'
        uvw_x.location = (base_x + 1000, base_y + 150)
        uvw_x.inputs[1].default_value = SCATTERING_NU_SIZE
        links.new(tex_x_plus_mus.outputs['Value'], uvw_x.inputs[0])
        
        # 3D->2D: U = (depth + uvw_x) / DEPTH, V = 1 - u_mu
        # WITH DEPTH INTERPOLATION to avoid stair-stepping
        if isinstance(u_r_val, float):
            depth_scaled_val = u_r_val * (SCATTERING_R_SIZE - 1)
            depth_floor_val = math.floor(depth_scaled_val)
            depth_frac_val = depth_scaled_val - depth_floor_val
            depth_ceil_val = min(depth_floor_val + 1, SCATTERING_R_SIZE - 1)
            
            depth_floor = nodes.new('ShaderNodeValue')
            depth_floor.location = (base_x + 2300, base_y - 50)
            depth_floor.outputs['Value'].default_value = depth_floor_val
            
            depth_ceil = nodes.new('ShaderNodeValue')
            depth_ceil.location = (base_x + 2300, base_y - 100)
            depth_ceil.outputs['Value'].default_value = depth_ceil_val
            
            depth_frac = nodes.new('ShaderNodeValue')
            depth_frac.location = (base_x + 2300, base_y - 150)
            depth_frac.outputs['Value'].default_value = depth_frac_val
        else:
            depth_scaled = nodes.new('ShaderNodeMath')
            depth_scaled.operation = 'MULTIPLY'
            depth_scaled.location = (base_x + 2300, base_y - 50)
            depth_scaled.inputs[1].default_value = SCATTERING_R_SIZE - 1
            links.new(u_r_val.outputs['Value'], depth_scaled.inputs[0])
            
            depth_floor = nodes.new('ShaderNodeMath')
            depth_floor.operation = 'FLOOR'
            depth_floor.location = (base_x + 2450, base_y - 50)
            links.new(depth_scaled.outputs['Value'], depth_floor.inputs[0])
            
            depth_frac = nodes.new('ShaderNodeMath')
            depth_frac.operation = 'SUBTRACT'
            depth_frac.location = (base_x + 2600, base_y - 50)
            links.new(depth_scaled.outputs['Value'], depth_frac.inputs[0])
            links.new(depth_floor.outputs['Value'], depth_frac.inputs[1])
            
            depth_ceil_raw = nodes.new('ShaderNodeMath')
            depth_ceil_raw.operation = 'ADD'
            depth_ceil_raw.location = (base_x + 2450, base_y - 100)
            depth_ceil_raw.inputs[1].default_value = 1.0
            links.new(depth_floor.outputs['Value'], depth_ceil_raw.inputs[0])
            
            depth_ceil = nodes.new('ShaderNodeMath')
            depth_ceil.operation = 'MINIMUM'
            depth_ceil.location = (base_x + 2600, base_y - 100)
            depth_ceil.inputs[1].default_value = SCATTERING_R_SIZE - 1
            links.new(depth_ceil_raw.outputs['Value'], depth_ceil.inputs[0])
        
        # V = 1 - u_mu (Y flip for Blender) - shared for both UVs
        v_flip = nodes.new('ShaderNodeMath')
        v_flip.operation = 'SUBTRACT'
        v_flip.location = (base_x + 2750, base_y - 200)
        v_flip.inputs[0].default_value = 1.0
        links.new(u_mu.outputs['Value'], v_flip.inputs[1])
        
        # UV for depth_floor
        u_sum_floor = nodes.new('ShaderNodeMath')
        u_sum_floor.operation = 'ADD'
        u_sum_floor.location = (base_x + 2750, base_y)
        links.new(depth_floor.outputs['Value'], u_sum_floor.inputs[0])
        links.new(uvw_x.outputs['Value'], u_sum_floor.inputs[1])
        
        final_u_floor = nodes.new('ShaderNodeMath')
        final_u_floor.operation = 'DIVIDE'
        final_u_floor.location = (base_x + 2900, base_y)
        final_u_floor.inputs[1].default_value = SCATTERING_R_SIZE
        links.new(u_sum_floor.outputs['Value'], final_u_floor.inputs[0])
        
        uv_floor = nodes.new('ShaderNodeCombineXYZ')
        uv_floor.location = (base_x + 3050, base_y)
        uv_floor.inputs['Z'].default_value = 0.0
        links.new(final_u_floor.outputs['Value'], uv_floor.inputs['X'])
        links.new(v_flip.outputs['Value'], uv_floor.inputs['Y'])
        
        # UV for depth_ceil
        u_sum_ceil = nodes.new('ShaderNodeMath')
        u_sum_ceil.operation = 'ADD'
        u_sum_ceil.location = (base_x + 2750, base_y - 100)
        links.new(depth_ceil.outputs['Value'], u_sum_ceil.inputs[0])
        links.new(uvw_x.outputs['Value'], u_sum_ceil.inputs[1])
        
        final_u_ceil = nodes.new('ShaderNodeMath')
        final_u_ceil.operation = 'DIVIDE'
        final_u_ceil.location = (base_x + 2900, base_y - 100)
        final_u_ceil.inputs[1].default_value = SCATTERING_R_SIZE
        links.new(u_sum_ceil.outputs['Value'], final_u_ceil.inputs[0])
        
        uv_ceil = nodes.new('ShaderNodeCombineXYZ')
        uv_ceil.location = (base_x + 3050, base_y - 100)
        uv_ceil.inputs['Z'].default_value = 0.0
        links.new(final_u_ceil.outputs['Value'], uv_ceil.inputs['X'])
        links.new(v_flip.outputs['Value'], uv_ceil.inputs['Y'])
        
        # Return both UVs and depth_frac for interpolation
        return uv_floor, uv_ceil, depth_frac
    
    # ==========================================================================
    # SAMPLE SCATTERING AT CAMERA (S_cam) with depth interpolation
    # ==========================================================================
    
    uv_cam_floor, uv_cam_ceil, depth_frac_cam = create_scatter_uv("cam", r_cam, mu, mu_s, nu, 1000, 400)
    
    tex_cam_floor = nodes.new('ShaderNodeTexImage')
    tex_cam_floor.location = (4200, 500)
    tex_cam_floor.interpolation = 'Linear'
    tex_cam_floor.extension = 'EXTEND'
    tex_cam_floor.image = scatter_img
    links.new(uv_cam_floor.outputs['Vector'], tex_cam_floor.inputs['Vector'])
    
    tex_cam_ceil = nodes.new('ShaderNodeTexImage')
    tex_cam_ceil.location = (4200, 350)
    tex_cam_ceil.interpolation = 'Linear'
    tex_cam_ceil.extension = 'EXTEND'
    tex_cam_ceil.image = scatter_img
    links.new(uv_cam_ceil.outputs['Vector'], tex_cam_ceil.inputs['Vector'])
    
    # Interpolate camera samples
    s_cam = nodes.new('ShaderNodeMix')
    s_cam.data_type = 'RGBA'
    s_cam.blend_type = 'MIX'
    s_cam.location = (4450, 450)
    links.new(depth_frac_cam.outputs['Value'], s_cam.inputs['Factor'])
    links.new(tex_cam_floor.outputs['Color'], s_cam.inputs[6])  # A
    links.new(tex_cam_ceil.outputs['Color'], s_cam.inputs[7])   # B
    
    # ==========================================================================
    # SAMPLE SCATTERING AT POINT (S_pt) with depth interpolation
    # ==========================================================================
    
    uv_pt_floor, uv_pt_ceil, depth_frac_pt = create_scatter_uv("pt", r_p, mu_p, mu_s_p, nu, 1000, -200)
    
    tex_pt_floor = nodes.new('ShaderNodeTexImage')
    tex_pt_floor.location = (4200, -100)
    tex_pt_floor.interpolation = 'Linear'
    tex_pt_floor.extension = 'EXTEND'
    tex_pt_floor.image = scatter_img
    links.new(uv_pt_floor.outputs['Vector'], tex_pt_floor.inputs['Vector'])
    
    tex_pt_ceil = nodes.new('ShaderNodeTexImage')
    tex_pt_ceil.location = (4200, -250)
    tex_pt_ceil.interpolation = 'Linear'
    tex_pt_ceil.extension = 'EXTEND'
    tex_pt_ceil.image = scatter_img
    links.new(uv_pt_ceil.outputs['Vector'], tex_pt_ceil.inputs['Vector'])
    
    # Interpolate point samples
    s_pt = nodes.new('ShaderNodeMix')
    s_pt.data_type = 'RGBA'
    s_pt.blend_type = 'MIX'
    s_pt.location = (4450, -150)
    links.new(depth_frac_pt.outputs['Value'], s_pt.inputs['Factor'])
    links.new(tex_pt_floor.outputs['Color'], s_pt.inputs[6])  # A
    links.new(tex_pt_ceil.outputs['Color'], s_pt.inputs[7])   # B
    
    # ==========================================================================
    # TRANSMITTANCE T (simplified exponential - working version)
    # ==========================================================================
    
    # Simple exponential falloff: T = exp(-d * extinction_coeff)
    neg_d = nodes.new('ShaderNodeMath')
    neg_d.operation = 'MULTIPLY'
    neg_d.location = (2800, 100)
    neg_d.inputs[1].default_value = -0.1  # Extinction coefficient
    links.new(d.outputs['Value'], neg_d.inputs[0])
    
    trans_approx = nodes.new('ShaderNodeMath')
    trans_approx.operation = 'EXPONENT'
    trans_approx.location = (3000, 100)
    links.new(neg_d.outputs['Value'], trans_approx.inputs[0])
    
    # T as RGB (grayscale for now)
    t_rgb = nodes.new('ShaderNodeCombineColor')
    t_rgb.location = (3200, 100)
    links.new(trans_approx.outputs['Value'], t_rgb.inputs['Red'])
    links.new(trans_approx.outputs['Value'], t_rgb.inputs['Green'])
    links.new(trans_approx.outputs['Value'], t_rgb.inputs['Blue'])
    
    # ==========================================================================
    # INSCATTER = S_cam - T × S_pt
    # ==========================================================================
    
    # T × S_pt
    t_times_spt = nodes.new('ShaderNodeMix')
    t_times_spt.data_type = 'RGBA'
    t_times_spt.blend_type = 'MULTIPLY'
    t_times_spt.location = (3400, 0)
    t_times_spt.inputs['Factor'].default_value = 1.0
    links.new(t_rgb.outputs['Color'], t_times_spt.inputs[6])  # A = T_rgb
    links.new(s_pt.outputs[2], t_times_spt.inputs[7])  # B = S_pt
    
    # S_cam - T × S_pt (using interpolated s_cam)
    inscatter = nodes.new('ShaderNodeMix')
    inscatter.data_type = 'RGBA'
    inscatter.blend_type = 'SUBTRACT'
    inscatter.location = (4850, 200)
    inscatter.inputs['Factor'].default_value = 1.0
    links.new(s_cam.outputs[2], inscatter.inputs[6])  # A = S_cam
    links.new(t_times_spt.outputs[2], inscatter.inputs[7])  # B = T×S_pt (color output at index 2)
    
    # ==========================================================================
    # PHASE FUNCTIONS (applied to inscatter)
    # Rayleigh: (3/16π)(1 + nu²)
    # Mie: (3/8π)(1-g²)/(2+g²) * (1+nu²) / (1+g²-2g*nu)^1.5
    # ==========================================================================
    
    PI = 3.14159265359
    MIE_G = 0.8
    g_sq = MIE_G * MIE_G
    
    # nu² (nu already computed above)
    nu_sq = nodes.new('ShaderNodeMath')
    nu_sq.operation = 'MULTIPLY'
    nu_sq.location = (5050, -200)
    links.new(nu.outputs['Value'], nu_sq.inputs[0])
    links.new(nu.outputs['Value'], nu_sq.inputs[1])
    
    # 1 + nu²
    one_plus_nu_sq = nodes.new('ShaderNodeMath')
    one_plus_nu_sq.operation = 'ADD'
    one_plus_nu_sq.location = (5200, -200)
    one_plus_nu_sq.inputs[0].default_value = 1.0
    links.new(nu_sq.outputs['Value'], one_plus_nu_sq.inputs[1])
    
    # Rayleigh phase: k * (1 + nu²), k = 3/(16π)
    k_rayleigh = 3.0 / (16.0 * PI)
    rayleigh_phase = nodes.new('ShaderNodeMath')
    rayleigh_phase.operation = 'MULTIPLY'
    rayleigh_phase.location = (5350, -200)
    rayleigh_phase.inputs[0].default_value = k_rayleigh
    links.new(one_plus_nu_sq.outputs['Value'], rayleigh_phase.inputs[1])
    
    # Mie phase: k * (1+nu²) / (1+g²-2g*nu)^1.5
    k_mie = (3.0 / (8.0 * PI)) * (1.0 - g_sq) / (2.0 + g_sq)
    
    # 2g*nu
    two_g_nu = nodes.new('ShaderNodeMath')
    two_g_nu.operation = 'MULTIPLY'
    two_g_nu.location = (5050, -350)
    two_g_nu.inputs[0].default_value = 2.0 * MIE_G
    links.new(nu.outputs['Value'], two_g_nu.inputs[1])
    
    # 1 + g² - 2g*nu
    mie_denom_base = nodes.new('ShaderNodeMath')
    mie_denom_base.operation = 'SUBTRACT'
    mie_denom_base.location = (5200, -350)
    mie_denom_base.inputs[0].default_value = 1.0 + g_sq
    links.new(two_g_nu.outputs['Value'], mie_denom_base.inputs[1])
    
    # Clamp to avoid division by zero
    mie_denom_clamp = nodes.new('ShaderNodeMath')
    mie_denom_clamp.operation = 'MAXIMUM'
    mie_denom_clamp.location = (5350, -350)
    mie_denom_clamp.inputs[1].default_value = 0.001
    links.new(mie_denom_base.outputs['Value'], mie_denom_clamp.inputs[0])
    
    # denom^1.5 = denom * sqrt(denom)
    mie_denom_sqrt = nodes.new('ShaderNodeMath')
    mie_denom_sqrt.operation = 'SQRT'
    mie_denom_sqrt.location = (5500, -400)
    links.new(mie_denom_clamp.outputs['Value'], mie_denom_sqrt.inputs[0])
    
    mie_denom_pow15 = nodes.new('ShaderNodeMath')
    mie_denom_pow15.operation = 'MULTIPLY'
    mie_denom_pow15.location = (5650, -350)
    links.new(mie_denom_clamp.outputs['Value'], mie_denom_pow15.inputs[0])
    links.new(mie_denom_sqrt.outputs['Value'], mie_denom_pow15.inputs[1])
    
    # k_mie * (1+nu²)
    mie_numer = nodes.new('ShaderNodeMath')
    mie_numer.operation = 'MULTIPLY'
    mie_numer.location = (5500, -250)
    mie_numer.inputs[0].default_value = k_mie
    links.new(one_plus_nu_sq.outputs['Value'], mie_numer.inputs[1])
    
    # Final Mie phase
    mie_phase = nodes.new('ShaderNodeMath')
    mie_phase.operation = 'DIVIDE'
    mie_phase.location = (5800, -300)
    links.new(mie_numer.outputs['Value'], mie_phase.inputs[0])
    links.new(mie_denom_pow15.outputs['Value'], mie_phase.inputs[1])
    
    # ==========================================================================
    # APPLY PHASE FUNCTIONS TO INSCATTER
    # inscatter RGB = Rayleigh, inscatter Alpha = Mie (from combined texture)
    # Result = Rayleigh * RayleighPhase + Mie * MiePhase
    # ==========================================================================
    
    # Separate inscatter RGB (Rayleigh) - clamp first
    sep_rgb = nodes.new('ShaderNodeSeparateColor')
    sep_rgb.location = (5050, 200)
    links.new(inscatter.outputs[2], sep_rgb.inputs['Color'])
    
    clamp_r = nodes.new('ShaderNodeMath')
    clamp_r.operation = 'MAXIMUM'
    clamp_r.location = (5200, 250)
    clamp_r.inputs[1].default_value = 0.0
    links.new(sep_rgb.outputs['Red'], clamp_r.inputs[0])
    
    clamp_g = nodes.new('ShaderNodeMath')
    clamp_g.operation = 'MAXIMUM'
    clamp_g.location = (5200, 150)
    clamp_g.inputs[1].default_value = 0.0
    links.new(sep_rgb.outputs['Green'], clamp_g.inputs[0])
    
    clamp_b = nodes.new('ShaderNodeMath')
    clamp_b.operation = 'MAXIMUM'
    clamp_b.location = (5200, 50)
    clamp_b.inputs[1].default_value = 0.0
    links.new(sep_rgb.outputs['Blue'], clamp_b.inputs[0])
    
    # Apply Rayleigh phase to RGB
    ray_r = nodes.new('ShaderNodeMath')
    ray_r.operation = 'MULTIPLY'
    ray_r.location = (5400, 250)
    links.new(clamp_r.outputs['Value'], ray_r.inputs[0])
    links.new(rayleigh_phase.outputs['Value'], ray_r.inputs[1])
    
    ray_g = nodes.new('ShaderNodeMath')
    ray_g.operation = 'MULTIPLY'
    ray_g.location = (5400, 150)
    links.new(clamp_g.outputs['Value'], ray_g.inputs[0])
    links.new(rayleigh_phase.outputs['Value'], ray_g.inputs[1])
    
    ray_b = nodes.new('ShaderNodeMath')
    ray_b.operation = 'MULTIPLY'
    ray_b.location = (5400, 50)
    links.new(clamp_b.outputs['Value'], ray_b.inputs[0])
    links.new(rayleigh_phase.outputs['Value'], ray_b.inputs[1])
    
    # ==========================================================================
    # MIE FROM ALPHA CHANNEL
    # Combined texture stores: RGB = Rayleigh, Alpha = Mie.r (single channel)
    # Mie inscatter = alpha_cam - T × alpha_pt
    # ==========================================================================
    
    # Interpolate alpha for camera sample
    alpha_cam = nodes.new('ShaderNodeMix')
    alpha_cam.data_type = 'FLOAT'
    alpha_cam.location = (4450, 300)
    links.new(depth_frac_cam.outputs['Value'], alpha_cam.inputs['Factor'])
    links.new(tex_cam_floor.outputs['Alpha'], alpha_cam.inputs[2])  # A (float)
    links.new(tex_cam_ceil.outputs['Alpha'], alpha_cam.inputs[3])   # B (float)
    
    # Interpolate alpha for point sample
    alpha_pt = nodes.new('ShaderNodeMix')
    alpha_pt.data_type = 'FLOAT'
    alpha_pt.location = (4450, -300)
    links.new(depth_frac_pt.outputs['Value'], alpha_pt.inputs['Factor'])
    links.new(tex_pt_floor.outputs['Alpha'], alpha_pt.inputs[2])  # A (float)
    links.new(tex_pt_ceil.outputs['Alpha'], alpha_pt.inputs[3])   # B (float)
    
    # T × alpha_pt
    t_times_alpha_pt = nodes.new('ShaderNodeMath')
    t_times_alpha_pt.operation = 'MULTIPLY'
    t_times_alpha_pt.location = (4650, -350)
    links.new(trans_approx.outputs['Value'], t_times_alpha_pt.inputs[0])
    links.new(alpha_pt.outputs[0], t_times_alpha_pt.inputs[1])
    
    # Mie inscatter = alpha_cam - T × alpha_pt
    mie_inscatter = nodes.new('ShaderNodeMath')
    mie_inscatter.operation = 'SUBTRACT'
    mie_inscatter.location = (4850, -350)
    links.new(alpha_cam.outputs[0], mie_inscatter.inputs[0])
    links.new(t_times_alpha_pt.outputs['Value'], mie_inscatter.inputs[1])
    
    # Clamp Mie inscatter to [0, inf)
    mie_clamp = nodes.new('ShaderNodeMath')
    mie_clamp.operation = 'MAXIMUM'
    mie_clamp.location = (5050, -350)
    mie_clamp.inputs[1].default_value = 0.0
    links.new(mie_inscatter.outputs['Value'], mie_clamp.inputs[0])
    
    # Apply Mie phase function
    mie_result = nodes.new('ShaderNodeMath')
    mie_result.operation = 'MULTIPLY'
    mie_result.location = (5200, -350)
    links.new(mie_clamp.outputs['Value'], mie_result.inputs[0])
    links.new(mie_phase.outputs['Value'], mie_result.inputs[1])
    
    # ==========================================================================
    # COMBINE RAYLEIGH + MIE
    # Final = Rayleigh_RGB * RayleighPhase + Mie * MiePhase (added to all channels)
    # ==========================================================================
    
    final_r = nodes.new('ShaderNodeMath')
    final_r.operation = 'ADD'
    final_r.location = (5550, 250)
    links.new(ray_r.outputs['Value'], final_r.inputs[0])
    links.new(mie_result.outputs['Value'], final_r.inputs[1])
    
    final_g = nodes.new('ShaderNodeMath')
    final_g.operation = 'ADD'
    final_g.location = (5550, 150)
    links.new(ray_g.outputs['Value'], final_g.inputs[0])
    links.new(mie_result.outputs['Value'], final_g.inputs[1])
    
    final_b = nodes.new('ShaderNodeMath')
    final_b.operation = 'ADD'
    final_b.location = (5550, 50)
    links.new(ray_b.outputs['Value'], final_b.inputs[0])
    links.new(mie_result.outputs['Value'], final_b.inputs[1])
    
    # Combine final result
    comb_rgb = nodes.new('ShaderNodeCombineColor')
    comb_rgb.location = (5750, 150)
    links.new(final_r.outputs['Value'], comb_rgb.inputs['Red'])
    links.new(final_g.outputs['Value'], comb_rgb.inputs['Green'])
    links.new(final_b.outputs['Value'], comb_rgb.inputs['Blue'])
    
    # ==========================================================================
    # OUTPUT (debug_mode: 0=full, 1=S_cam, 2=S_pt, 3=T, 4=rayleigh_only, 5=mie_only)
    # ==========================================================================
    
    debug_mode = 0  # Output full inscatter with Rayleigh + Mie phase functions
    
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (5950, 200)
    emission.inputs['Strength'].default_value = 1.0
    
    if debug_mode == 0:
        # Full output: Rayleigh + Mie
        links.new(comb_rgb.outputs['Color'], emission.inputs['Color'])
    elif debug_mode == 1:
        links.new(s_cam.outputs[2], emission.inputs['Color'])  # S_cam only
    elif debug_mode == 2:
        links.new(s_pt.outputs[2], emission.inputs['Color'])   # S_pt only
    elif debug_mode == 3:
        links.new(t_rgb.outputs['Color'], emission.inputs['Color'])  # Transmittance
    elif debug_mode == 4:
        # Rayleigh only (no Mie)
        ray_only_rgb = nodes.new('ShaderNodeCombineColor')
        ray_only_rgb.location = (5750, 0)
        links.new(ray_r.outputs['Value'], ray_only_rgb.inputs['Red'])
        links.new(ray_g.outputs['Value'], ray_only_rgb.inputs['Green'])
        links.new(ray_b.outputs['Value'], ray_only_rgb.inputs['Blue'])
        links.new(ray_only_rgb.outputs['Color'], emission.inputs['Color'])
    elif debug_mode == 5:
        # Mie only (grayscale, scaled for visibility)
        mie_scaled = nodes.new('ShaderNodeMath')
        mie_scaled.operation = 'MULTIPLY'
        mie_scaled.location = (5750, -100)
        mie_scaled.inputs[1].default_value = 10.0  # Scale up for visibility
        links.new(mie_result.outputs['Value'], mie_scaled.inputs[0])
        mie_only_rgb = nodes.new('ShaderNodeCombineColor')
        mie_only_rgb.location = (5900, -100)
        links.new(mie_scaled.outputs['Value'], mie_only_rgb.inputs['Red'])
        links.new(mie_scaled.outputs['Value'], mie_only_rgb.inputs['Green'])
        links.new(mie_scaled.outputs['Value'], mie_only_rgb.inputs['Blue'])
        links.new(mie_only_rgb.outputs['Color'], emission.inputs['Color'])
    
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (6150, 200)
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    # Auto-assign
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
            obj.data.materials.append(mat)
    
    mesh_count = len([o for o in bpy.context.scene.objects if o.type == 'MESH'])
    print(f"\nCreated: {mat.name}")
    print(f"Assigned to {mesh_count} meshes")
    print(f"\nExpected: Blue-ish inscatter, stronger for distant objects")
    print(f"Note: Using simplified exponential transmittance for now")
    
    return mat


# =============================================================================
# STEP 4: FULL AERIAL PERSPECTIVE (Surface × T + Inscatter)
# =============================================================================

def apply_step_4_full_aerial(surface_color=(0.8, 0.8, 0.8), debug_mode=0):
    """
    Step 4: Full aerial perspective formula
    
    L_final = L_surface × T + inscatter
    
    Where:
    - L_surface = object surface color (input parameter or from texture)
    - T = transmittance from camera to point
    - inscatter = Rayleigh + Mie with phase functions
    
    Args:
        surface_color: RGB tuple for object base color (default gray)
        debug_mode: 0=full, 1=surface×T, 2=T, 3=inscatter, 4=surface_only, 5=distance
    """
    import bpy
    import math
    import mathutils
    import time
    import os
    
    print("Step 4: Full Aerial Perspective (Surface × T + Inscatter)")
    
    # Constants
    BOTTOM_RADIUS = 6360.0  # km
    TOP_RADIUS = 6420.0     # km
    PI = 3.14159265359
    MIE_G = 0.8
    g_sq = MIE_G * MIE_G
    
    # LUT dimensions
    SCATTERING_MU_SIZE = 128
    SCATTERING_MU_S_SIZE = 32
    SCATTERING_NU_SIZE = 8
    SCATTERING_R_SIZE = 32
    TRANSMITTANCE_WIDTH = 256
    TRANSMITTANCE_HEIGHT = 64
    
    # Get LUT path
    lut_dir = r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-3\helios_cache\luts"
    scatter_path = os.path.join(lut_dir, "scattering.exr")
    trans_path = os.path.join(lut_dir, "transmittance.exr")
    
    if not os.path.exists(scatter_path):
        print(f"  ERROR: {scatter_path} not found")
        return None
    if not os.path.exists(trans_path):
        print(f"  ERROR: {trans_path} not found")
        return None
    
    print(f"  Scattering: {scatter_path}")
    print(f"  Transmittance: {trans_path}")
    print(f"  Surface color: {surface_color}")
    
    # Load textures
    scatter_img = bpy.data.images.load(scatter_path, check_existing=True)
    scatter_img.colorspace_settings.name = 'Non-Color'
    
    trans_img = bpy.data.images.load(trans_path, check_existing=True)
    trans_img.colorspace_settings.name = 'Non-Color'
    
    # Find sun light in scene
    sun_light = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN':
            sun_light = obj
            break
    
    if sun_light:
        sun_direction = sun_light.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))
        sun_direction.normalize()
    else:
        sun_direction = mathutils.Vector((0.25, 0.433, 0.866))  # 60° elevation default
        print("  WARNING: No Sun light found, using default direction")
    
    print(f"  Sun direction: ({sun_direction.x:.3f}, {sun_direction.y:.3f}, {sun_direction.z:.3f})")
    
    # Create material
    mat_name = f"Step4_FullAerial_{int(time.time())}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # ==========================================================================
    # INPUTS: Geometry and camera
    # ==========================================================================
    
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.location = (-1600, 400)
    
    cam_loc = nodes.new('ShaderNodeCombineXYZ')
    cam_loc.location = (-1600, 200)
    cam = bpy.context.scene.camera
    if cam:
        cam_loc.inputs['X'].default_value = cam.location.x
        cam_loc.inputs['Y'].default_value = cam.location.y
        cam_loc.inputs['Z'].default_value = cam.location.z
    
    # Sun direction
    sun_dir_node = nodes.new('ShaderNodeCombineXYZ')
    sun_dir_node.location = (-1600, 0)
    sun_dir_node.inputs['X'].default_value = sun_direction.x
    sun_dir_node.inputs['Y'].default_value = sun_direction.y
    sun_dir_node.inputs['Z'].default_value = sun_direction.z
    
    # Surface color input
    surface_rgb = nodes.new('ShaderNodeRGB')
    surface_rgb.location = (-1600, -200)
    surface_rgb.outputs['Color'].default_value = (surface_color[0], surface_color[1], surface_color[2], 1.0)
    
    # ==========================================================================
    # DISTANCE CALCULATION
    # ==========================================================================
    
    # Vector from camera to point
    cam_to_point = nodes.new('ShaderNodeVectorMath')
    cam_to_point.operation = 'SUBTRACT'
    cam_to_point.location = (-1400, 300)
    links.new(geom.outputs['Position'], cam_to_point.inputs[0])
    links.new(cam_loc.outputs['Vector'], cam_to_point.inputs[1])
    
    # Distance in meters
    dist_m = nodes.new('ShaderNodeVectorMath')
    dist_m.operation = 'LENGTH'
    dist_m.location = (-1200, 300)
    links.new(cam_to_point.outputs['Vector'], dist_m.inputs[0])
    
    # Distance in km
    d = nodes.new('ShaderNodeMath')
    d.operation = 'MULTIPLY'
    d.location = (-1000, 300)
    d.inputs[1].default_value = 0.001
    links.new(dist_m.outputs['Value'], d.inputs[0])
    
    # View direction (normalized)
    view_dir = nodes.new('ShaderNodeVectorMath')
    view_dir.operation = 'NORMALIZE'
    view_dir.location = (-1200, 200)
    links.new(cam_to_point.outputs['Vector'], view_dir.inputs[0])
    
    # ==========================================================================
    # TRANSMITTANCE (simplified exponential for now)
    # T = exp(-d * extinction_coeff)
    # ==========================================================================
    
    neg_d = nodes.new('ShaderNodeMath')
    neg_d.operation = 'MULTIPLY'
    neg_d.location = (-800, 100)
    neg_d.inputs[1].default_value = -0.1  # Extinction coefficient
    links.new(d.outputs['Value'], neg_d.inputs[0])
    
    transmittance = nodes.new('ShaderNodeMath')
    transmittance.operation = 'EXPONENT'
    transmittance.location = (-600, 100)
    links.new(neg_d.outputs['Value'], transmittance.inputs[0])
    
    # T as RGB
    t_rgb = nodes.new('ShaderNodeCombineColor')
    t_rgb.location = (-400, 100)
    links.new(transmittance.outputs['Value'], t_rgb.inputs['Red'])
    links.new(transmittance.outputs['Value'], t_rgb.inputs['Green'])
    links.new(transmittance.outputs['Value'], t_rgb.inputs['Blue'])
    
    # ==========================================================================
    # SURFACE × TRANSMITTANCE
    # ==========================================================================
    
    surface_times_t = nodes.new('ShaderNodeMix')
    surface_times_t.data_type = 'RGBA'
    surface_times_t.blend_type = 'MULTIPLY'
    surface_times_t.location = (-200, 200)
    surface_times_t.inputs['Factor'].default_value = 1.0
    links.new(surface_rgb.outputs['Color'], surface_times_t.inputs[6])
    links.new(t_rgb.outputs['Color'], surface_times_t.inputs[7])
    
    # ==========================================================================
    # INSCATTER (simplified - using fixed UV for camera position)
    # This is a simplified version - full version would compute proper UV
    # ==========================================================================
    
    # For simplicity, compute inscatter as proportional to (1 - T)
    # This approximates the accumulated scattering along the path
    one_minus_t = nodes.new('ShaderNodeMath')
    one_minus_t.operation = 'SUBTRACT'
    one_minus_t.location = (-400, -100)
    one_minus_t.inputs[0].default_value = 1.0
    links.new(transmittance.outputs['Value'], one_minus_t.inputs[1])
    
    # nu = dot(view_dir, sun_dir) for phase functions
    nu = nodes.new('ShaderNodeVectorMath')
    nu.operation = 'DOT_PRODUCT'
    nu.location = (-800, -200)
    links.new(view_dir.outputs['Vector'], nu.inputs[0])
    links.new(sun_dir_node.outputs['Vector'], nu.inputs[1])
    
    # nu²
    nu_sq = nodes.new('ShaderNodeMath')
    nu_sq.operation = 'MULTIPLY'
    nu_sq.location = (-600, -200)
    links.new(nu.outputs['Value'], nu_sq.inputs[0])
    links.new(nu.outputs['Value'], nu_sq.inputs[1])
    
    # 1 + nu²
    one_plus_nu_sq = nodes.new('ShaderNodeMath')
    one_plus_nu_sq.operation = 'ADD'
    one_plus_nu_sq.location = (-400, -200)
    one_plus_nu_sq.inputs[0].default_value = 1.0
    links.new(nu_sq.outputs['Value'], one_plus_nu_sq.inputs[1])
    
    # Rayleigh phase: k * (1 + nu²)
    k_rayleigh = 3.0 / (16.0 * PI)
    rayleigh_phase = nodes.new('ShaderNodeMath')
    rayleigh_phase.operation = 'MULTIPLY'
    rayleigh_phase.location = (-200, -200)
    rayleigh_phase.inputs[0].default_value = k_rayleigh
    links.new(one_plus_nu_sq.outputs['Value'], rayleigh_phase.inputs[1])
    
    # Mie phase: k * (1+nu²) / (1+g²-2g*nu)^1.5
    k_mie = (3.0 / (8.0 * PI)) * (1.0 - g_sq) / (2.0 + g_sq)
    
    two_g_nu = nodes.new('ShaderNodeMath')
    two_g_nu.operation = 'MULTIPLY'
    two_g_nu.location = (-600, -350)
    two_g_nu.inputs[0].default_value = 2.0 * MIE_G
    links.new(nu.outputs['Value'], two_g_nu.inputs[1])
    
    mie_denom_base = nodes.new('ShaderNodeMath')
    mie_denom_base.operation = 'SUBTRACT'
    mie_denom_base.location = (-400, -350)
    mie_denom_base.inputs[0].default_value = 1.0 + g_sq
    links.new(two_g_nu.outputs['Value'], mie_denom_base.inputs[1])
    
    mie_denom_clamp = nodes.new('ShaderNodeMath')
    mie_denom_clamp.operation = 'MAXIMUM'
    mie_denom_clamp.location = (-200, -350)
    mie_denom_clamp.inputs[1].default_value = 0.001
    links.new(mie_denom_base.outputs['Value'], mie_denom_clamp.inputs[0])
    
    mie_denom_sqrt = nodes.new('ShaderNodeMath')
    mie_denom_sqrt.operation = 'SQRT'
    mie_denom_sqrt.location = (0, -400)
    links.new(mie_denom_clamp.outputs['Value'], mie_denom_sqrt.inputs[0])
    
    mie_denom_pow15 = nodes.new('ShaderNodeMath')
    mie_denom_pow15.operation = 'MULTIPLY'
    mie_denom_pow15.location = (200, -350)
    links.new(mie_denom_clamp.outputs['Value'], mie_denom_pow15.inputs[0])
    links.new(mie_denom_sqrt.outputs['Value'], mie_denom_pow15.inputs[1])
    
    mie_numer = nodes.new('ShaderNodeMath')
    mie_numer.operation = 'MULTIPLY'
    mie_numer.location = (0, -250)
    mie_numer.inputs[0].default_value = k_mie
    links.new(one_plus_nu_sq.outputs['Value'], mie_numer.inputs[1])
    
    mie_phase = nodes.new('ShaderNodeMath')
    mie_phase.operation = 'DIVIDE'
    mie_phase.location = (400, -300)
    links.new(mie_numer.outputs['Value'], mie_phase.inputs[0])
    links.new(mie_denom_pow15.outputs['Value'], mie_phase.inputs[1])
    
    # ==========================================================================
    # INSCATTER COLOR (approximation using 1-T and phase functions)
    # ==========================================================================
    
    # Rayleigh inscatter: bluish color * (1-T) * rayleigh_phase
    rayleigh_base = nodes.new('ShaderNodeMath')
    rayleigh_base.operation = 'MULTIPLY'
    rayleigh_base.location = (0, -100)
    links.new(one_minus_t.outputs['Value'], rayleigh_base.inputs[0])
    links.new(rayleigh_phase.outputs['Value'], rayleigh_base.inputs[1])
    
    # Scale for visibility
    rayleigh_scaled = nodes.new('ShaderNodeMath')
    rayleigh_scaled.operation = 'MULTIPLY'
    rayleigh_scaled.location = (200, -100)
    rayleigh_scaled.inputs[1].default_value = 2.0  # Boost factor
    links.new(rayleigh_base.outputs['Value'], rayleigh_scaled.inputs[0])
    
    # Rayleigh is blue-dominant: (0.3, 0.5, 1.0) * rayleigh_scaled
    ray_r = nodes.new('ShaderNodeMath')
    ray_r.operation = 'MULTIPLY'
    ray_r.location = (400, -50)
    ray_r.inputs[1].default_value = 0.3
    links.new(rayleigh_scaled.outputs['Value'], ray_r.inputs[0])
    
    ray_g = nodes.new('ShaderNodeMath')
    ray_g.operation = 'MULTIPLY'
    ray_g.location = (400, -100)
    ray_g.inputs[1].default_value = 0.5
    links.new(rayleigh_scaled.outputs['Value'], ray_g.inputs[0])
    
    ray_b = nodes.new('ShaderNodeMath')
    ray_b.operation = 'MULTIPLY'
    ray_b.location = (400, -150)
    ray_b.inputs[1].default_value = 1.0
    links.new(rayleigh_scaled.outputs['Value'], ray_b.inputs[0])
    
    # Mie inscatter: whitish * (1-T) * mie_phase
    mie_base = nodes.new('ShaderNodeMath')
    mie_base.operation = 'MULTIPLY'
    mie_base.location = (200, -250)
    links.new(one_minus_t.outputs['Value'], mie_base.inputs[0])
    links.new(mie_phase.outputs['Value'], mie_base.inputs[1])
    
    mie_scaled = nodes.new('ShaderNodeMath')
    mie_scaled.operation = 'MULTIPLY'
    mie_scaled.location = (400, -250)
    mie_scaled.inputs[1].default_value = 0.5  # Mie is generally weaker than Rayleigh
    links.new(mie_base.outputs['Value'], mie_scaled.inputs[0])
    
    # Combine Rayleigh + Mie
    inscatter_r = nodes.new('ShaderNodeMath')
    inscatter_r.operation = 'ADD'
    inscatter_r.location = (600, -50)
    links.new(ray_r.outputs['Value'], inscatter_r.inputs[0])
    links.new(mie_scaled.outputs['Value'], inscatter_r.inputs[1])
    
    inscatter_g = nodes.new('ShaderNodeMath')
    inscatter_g.operation = 'ADD'
    inscatter_g.location = (600, -100)
    links.new(ray_g.outputs['Value'], inscatter_g.inputs[0])
    links.new(mie_scaled.outputs['Value'], inscatter_g.inputs[1])
    
    inscatter_b = nodes.new('ShaderNodeMath')
    inscatter_b.operation = 'ADD'
    inscatter_b.location = (600, -150)
    links.new(ray_b.outputs['Value'], inscatter_b.inputs[0])
    links.new(mie_scaled.outputs['Value'], inscatter_b.inputs[1])
    
    inscatter_rgb = nodes.new('ShaderNodeCombineColor')
    inscatter_rgb.location = (800, -100)
    links.new(inscatter_r.outputs['Value'], inscatter_rgb.inputs['Red'])
    links.new(inscatter_g.outputs['Value'], inscatter_rgb.inputs['Green'])
    links.new(inscatter_b.outputs['Value'], inscatter_rgb.inputs['Blue'])
    
    # ==========================================================================
    # FINAL: Surface × T + Inscatter
    # ==========================================================================
    
    final_add = nodes.new('ShaderNodeMix')
    final_add.data_type = 'RGBA'
    final_add.blend_type = 'ADD'
    final_add.location = (1000, 100)
    final_add.inputs['Factor'].default_value = 1.0
    links.new(surface_times_t.outputs[2], final_add.inputs[6])
    links.new(inscatter_rgb.outputs['Color'], final_add.inputs[7])
    
    # ==========================================================================
    # OUTPUT (debug_mode: 0=full, 1=surface×T, 2=T, 3=inscatter, 4=surface, 5=dist)
    # ==========================================================================
    
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (1200, 100)
    emission.inputs['Strength'].default_value = 1.0
    
    if debug_mode == 0:
        # Full: Surface × T + Inscatter
        links.new(final_add.outputs[2], emission.inputs['Color'])
    elif debug_mode == 1:
        # Surface × T only
        links.new(surface_times_t.outputs[2], emission.inputs['Color'])
    elif debug_mode == 2:
        # Transmittance only (grayscale)
        links.new(t_rgb.outputs['Color'], emission.inputs['Color'])
    elif debug_mode == 3:
        # Inscatter only
        links.new(inscatter_rgb.outputs['Color'], emission.inputs['Color'])
    elif debug_mode == 4:
        # Surface color only (no transmittance)
        links.new(surface_rgb.outputs['Color'], emission.inputs['Color'])
    elif debug_mode == 5:
        # Distance visualization (normalized to 10km max)
        dist_norm = nodes.new('ShaderNodeMath')
        dist_norm.operation = 'DIVIDE'
        dist_norm.location = (1000, -100)
        dist_norm.inputs[1].default_value = 10.0  # 10km max
        links.new(d.outputs['Value'], dist_norm.inputs[0])
        dist_rgb = nodes.new('ShaderNodeCombineColor')
        dist_rgb.location = (1150, -100)
        links.new(dist_norm.outputs['Value'], dist_rgb.inputs['Red'])
        links.new(dist_norm.outputs['Value'], dist_rgb.inputs['Green'])
        links.new(dist_norm.outputs['Value'], dist_rgb.inputs['Blue'])
        links.new(dist_rgb.outputs['Color'], emission.inputs['Color'])
    
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (1400, 100)
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    # Auto-assign
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
            obj.data.materials.append(mat)
    
    mesh_count = len([o for o in bpy.context.scene.objects if o.type == 'MESH'])
    print(f"\nCreated: {mat.name}")
    print(f"Assigned to {mesh_count} meshes")
    print(f"Debug mode: {debug_mode}")
    print(f"\nFormula: L_final = L_surface × T + inscatter")
    print(f"Debug modes: 0=full, 1=surface×T, 2=T, 3=inscatter, 4=surface, 5=distance")
    
    return mat


# =============================================================================
# STEP 3: PHASE FUNCTIONS
# =============================================================================

def apply_step_3_phase_functions():
    """
    Step 3: Add Rayleigh and Mie phase functions to inscatter
    
    Phase functions modulate inscatter based on view-sun angle:
    - Rayleigh: (3/16π)(1 + cos²θ) - symmetric scattering
    - Mie: Henyey-Greenstein - forward scattering peak
    
    Combined texture stores: RGB = Rayleigh, Alpha = Mie.r
    """
    import bpy
    import math
    import time
    import os
    
    print("Step 3: Phase Functions")
    
    # Constants
    BOTTOM_RADIUS = 6360.0  # km
    TOP_RADIUS = 6420.0     # km
    MIE_G = 0.8             # Mie asymmetry parameter
    PI = math.pi
    
    # LUT dimensions
    MU_SIZE = 128
    MU_S_SIZE = 32
    NU_SIZE = 8
    R_SIZE = 32
    
    # Find LUT paths
    lut_dir = r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-3\helios_cache\luts"
    scatter_path = os.path.join(lut_dir, "scattering.exr")
    trans_path = os.path.join(lut_dir, "transmittance.exr")
    
    print(f"  Scattering: {scatter_path}")
    print(f"  Transmittance: {trans_path}")
    
    # Create material
    mat_name = f"Step3_PhaseFunc_{int(time.time())}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # ==========================================================================
    # GEOMETRY & CAMERA SETUP (same as Step 2.4)
    # ==========================================================================
    
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.location = (-1600, 400)
    
    cam_loc = nodes.new('ShaderNodeCombineXYZ')
    cam_loc.location = (-1600, 200)
    cam = bpy.context.scene.camera
    if cam:
        cam_loc.inputs['X'].default_value = cam.location.x
        cam_loc.inputs['Y'].default_value = cam.location.y
        cam_loc.inputs['Z'].default_value = cam.location.z
    
    # Find sun light in scene
    sun_light = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN':
            sun_light = obj
            break
    
    sun_dir = nodes.new('ShaderNodeCombineXYZ')
    sun_dir.location = (-1600, 0)
    if sun_light:
        import mathutils
        sun_direction = sun_light.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))
        sun_direction.normalize()
        sun_dir.inputs['X'].default_value = sun_direction.x
        sun_dir.inputs['Y'].default_value = sun_direction.y
        sun_dir.inputs['Z'].default_value = sun_direction.z
        print(f"  Sun direction: ({sun_direction.x:.3f}, {sun_direction.y:.3f}, {sun_direction.z:.3f})")
    else:
        sun_dir.inputs['X'].default_value = 0.0
        sun_dir.inputs['Y'].default_value = 0.2588
        sun_dir.inputs['Z'].default_value = 0.9659
        print(f"  Sun: No sun light found, using default 15° elevation")
    
    # ==========================================================================
    # DISTANCE & VIEW DIRECTION
    # ==========================================================================
    
    pos_to_cam = nodes.new('ShaderNodeVectorMath')
    pos_to_cam.operation = 'SUBTRACT'
    pos_to_cam.location = (-1400, 300)
    links.new(geom.outputs['Position'], pos_to_cam.inputs[0])
    links.new(cam_loc.outputs['Vector'], pos_to_cam.inputs[1])
    
    d_vec_len = nodes.new('ShaderNodeVectorMath')
    d_vec_len.operation = 'LENGTH'
    d_vec_len.location = (-1200, 300)
    links.new(pos_to_cam.outputs['Vector'], d_vec_len.inputs[0])
    
    d = nodes.new('ShaderNodeMath')
    d.operation = 'MULTIPLY'
    d.location = (-1000, 300)
    d.inputs[1].default_value = 0.001  # to km
    links.new(d_vec_len.outputs['Value'], d.inputs[0])
    
    view_dir = nodes.new('ShaderNodeVectorMath')
    view_dir.operation = 'NORMALIZE'
    view_dir.location = (-1200, 150)
    links.new(pos_to_cam.outputs['Vector'], view_dir.inputs[0])
    
    # ==========================================================================
    # NU = dot(view_dir, sun_dir) - needed for phase functions
    # ==========================================================================
    
    nu_dot = nodes.new('ShaderNodeVectorMath')
    nu_dot.operation = 'DOT_PRODUCT'
    nu_dot.location = (-800, -250)
    links.new(view_dir.outputs['Vector'], nu_dot.inputs[0])
    links.new(sun_dir.outputs['Vector'], nu_dot.inputs[1])
    
    # ==========================================================================
    # RAYLEIGH PHASE: (3/16π)(1 + nu²)
    # ==========================================================================
    
    # nu²
    nu_sq = nodes.new('ShaderNodeMath')
    nu_sq.operation = 'MULTIPLY'
    nu_sq.location = (-600, -300)
    links.new(nu_dot.outputs['Value'], nu_sq.inputs[0])
    links.new(nu_dot.outputs['Value'], nu_sq.inputs[1])
    
    # 1 + nu²
    one_plus_nu_sq = nodes.new('ShaderNodeMath')
    one_plus_nu_sq.operation = 'ADD'
    one_plus_nu_sq.location = (-400, -300)
    one_plus_nu_sq.inputs[0].default_value = 1.0
    links.new(nu_sq.outputs['Value'], one_plus_nu_sq.inputs[1])
    
    # k_rayleigh = 3 / (16 * π) ≈ 0.05968
    k_rayleigh = 3.0 / (16.0 * PI)
    
    rayleigh_phase = nodes.new('ShaderNodeMath')
    rayleigh_phase.operation = 'MULTIPLY'
    rayleigh_phase.location = (-200, -300)
    rayleigh_phase.inputs[0].default_value = k_rayleigh
    links.new(one_plus_nu_sq.outputs['Value'], rayleigh_phase.inputs[1])
    
    # ==========================================================================
    # MIE PHASE: k * (1 + nu²) / (1 + g² - 2g*nu)^1.5
    # k = (3/8π) * (1-g²) / (2+g²)
    # ==========================================================================
    
    # Pre-compute constants for g = 0.8
    g = MIE_G
    g_sq = g * g  # 0.64
    k_mie = (3.0 / (8.0 * PI)) * (1.0 - g_sq) / (2.0 + g_sq)  # ≈ 0.01616
    
    # Denominator: 1 + g² - 2g*nu
    # = 1.64 - 1.6*nu
    
    # 2g*nu
    two_g_nu = nodes.new('ShaderNodeMath')
    two_g_nu.operation = 'MULTIPLY'
    two_g_nu.location = (-600, -500)
    two_g_nu.inputs[0].default_value = 2.0 * g  # 1.6
    links.new(nu_dot.outputs['Value'], two_g_nu.inputs[1])
    
    # 1 + g² - 2g*nu
    denom_base = nodes.new('ShaderNodeMath')
    denom_base.operation = 'SUBTRACT'
    denom_base.location = (-400, -500)
    denom_base.inputs[0].default_value = 1.0 + g_sq  # 1.64
    links.new(two_g_nu.outputs['Value'], denom_base.inputs[1])
    
    # Clamp denominator to avoid division issues (min 0.001)
    denom_clamp = nodes.new('ShaderNodeMath')
    denom_clamp.operation = 'MAXIMUM'
    denom_clamp.location = (-200, -500)
    denom_clamp.inputs[1].default_value = 0.001
    links.new(denom_base.outputs['Value'], denom_clamp.inputs[0])
    
    # denom^1.5 = denom * sqrt(denom)
    denom_sqrt = nodes.new('ShaderNodeMath')
    denom_sqrt.operation = 'SQRT'
    denom_sqrt.location = (0, -550)
    links.new(denom_clamp.outputs['Value'], denom_sqrt.inputs[0])
    
    denom_pow15 = nodes.new('ShaderNodeMath')
    denom_pow15.operation = 'MULTIPLY'
    denom_pow15.location = (200, -500)
    links.new(denom_clamp.outputs['Value'], denom_pow15.inputs[0])
    links.new(denom_sqrt.outputs['Value'], denom_pow15.inputs[1])
    
    # k_mie * (1 + nu²)
    mie_numer = nodes.new('ShaderNodeMath')
    mie_numer.operation = 'MULTIPLY'
    mie_numer.location = (0, -400)
    mie_numer.inputs[0].default_value = k_mie
    links.new(one_plus_nu_sq.outputs['Value'], mie_numer.inputs[1])
    
    # Final Mie phase = numerator / denom^1.5
    mie_phase = nodes.new('ShaderNodeMath')
    mie_phase.operation = 'DIVIDE'
    mie_phase.location = (400, -450)
    links.new(mie_numer.outputs['Value'], mie_phase.inputs[0])
    links.new(denom_pow15.outputs['Value'], mie_phase.inputs[1])
    
    # ==========================================================================
    # LOAD SCATTERING TEXTURE (RGB = Rayleigh, Alpha = Mie.r)
    # ==========================================================================
    
    scatter_tex = nodes.new('ShaderNodeTexImage')
    scatter_tex.location = (600, 200)
    scatter_tex.interpolation = 'Linear'
    scatter_tex.extension = 'CLIP'
    
    img = bpy.data.images.get(os.path.basename(scatter_path))
    if not img:
        img = bpy.data.images.load(scatter_path)
    img.colorspace_settings.name = 'Non-Color'
    scatter_tex.image = img
    
    # ==========================================================================
    # SIMPLE TEST: Apply phase functions to a fixed sample
    # For now, sample at a reasonable UV to test phase function effect
    # ==========================================================================
    
    # Fixed UV for testing (center of texture)
    test_uv = nodes.new('ShaderNodeCombineXYZ')
    test_uv.location = (400, 300)
    test_uv.inputs['X'].default_value = 0.5
    test_uv.inputs['Y'].default_value = 0.5
    test_uv.inputs['Z'].default_value = 0.0
    
    links.new(test_uv.outputs['Vector'], scatter_tex.inputs['Vector'])
    
    # ==========================================================================
    # APPLY PHASE FUNCTIONS
    # Result = Rayleigh_RGB * RayleighPhase + Mie_RGB * MiePhase
    # Mie_RGB is extrapolated from Mie.r (alpha channel)
    # ==========================================================================
    
    # Rayleigh contribution: RGB * rayleigh_phase
    sep_scatter = nodes.new('ShaderNodeSeparateColor')
    sep_scatter.location = (800, 200)
    links.new(scatter_tex.outputs['Color'], sep_scatter.inputs['Color'])
    
    rayleigh_r = nodes.new('ShaderNodeMath')
    rayleigh_r.operation = 'MULTIPLY'
    rayleigh_r.location = (1000, 300)
    links.new(sep_scatter.outputs['Red'], rayleigh_r.inputs[0])
    links.new(rayleigh_phase.outputs['Value'], rayleigh_r.inputs[1])
    
    rayleigh_g = nodes.new('ShaderNodeMath')
    rayleigh_g.operation = 'MULTIPLY'
    rayleigh_g.location = (1000, 200)
    links.new(sep_scatter.outputs['Green'], rayleigh_g.inputs[0])
    links.new(rayleigh_phase.outputs['Value'], rayleigh_g.inputs[1])
    
    rayleigh_b = nodes.new('ShaderNodeMath')
    rayleigh_b.operation = 'MULTIPLY'
    rayleigh_b.location = (1000, 100)
    links.new(sep_scatter.outputs['Blue'], rayleigh_b.inputs[0])
    links.new(rayleigh_phase.outputs['Value'], rayleigh_b.inputs[1])
    
    # Mie contribution: Alpha * mie_phase (single channel, apply to all RGB)
    # For now, just add Mie.r to each channel weighted by mie_phase
    mie_contrib = nodes.new('ShaderNodeMath')
    mie_contrib.operation = 'MULTIPLY'
    mie_contrib.location = (1000, -100)
    links.new(scatter_tex.outputs['Alpha'], mie_contrib.inputs[0])
    links.new(mie_phase.outputs['Value'], mie_contrib.inputs[1])
    
    # Combine: Rayleigh + Mie
    result_r = nodes.new('ShaderNodeMath')
    result_r.operation = 'ADD'
    result_r.location = (1200, 300)
    links.new(rayleigh_r.outputs['Value'], result_r.inputs[0])
    links.new(mie_contrib.outputs['Value'], result_r.inputs[1])
    
    result_g = nodes.new('ShaderNodeMath')
    result_g.operation = 'ADD'
    result_g.location = (1200, 200)
    links.new(rayleigh_g.outputs['Value'], result_g.inputs[0])
    links.new(mie_contrib.outputs['Value'], result_g.inputs[1])
    
    result_b = nodes.new('ShaderNodeMath')
    result_b.operation = 'ADD'
    result_b.location = (1200, 100)
    links.new(rayleigh_b.outputs['Value'], result_b.inputs[0])
    links.new(mie_contrib.outputs['Value'], result_b.inputs[1])
    
    # Combine to RGB
    result_rgb = nodes.new('ShaderNodeCombineColor')
    result_rgb.location = (1400, 200)
    links.new(result_r.outputs['Value'], result_rgb.inputs['Red'])
    links.new(result_g.outputs['Value'], result_rgb.inputs['Green'])
    links.new(result_b.outputs['Value'], result_rgb.inputs['Blue'])
    
    # ==========================================================================
    # OUTPUT (debug_mode: 0=full, 1=rayleigh_phase, 2=mie_phase, 3=nu)
    # ==========================================================================
    
    debug_mode = 3  # Output nu (dot product) to verify sun direction affects it
    
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (2200, 200)
    emission.inputs['Strength'].default_value = 1.0
    
    if debug_mode == 0:
        # Full output with distance
        dist_scale = nodes.new('ShaderNodeMath')
        dist_scale.operation = 'MULTIPLY'
        dist_scale.location = (1600, 200)
        dist_scale.inputs[1].default_value = 0.1
        links.new(d.outputs['Value'], dist_scale.inputs[0])
        
        dist_clamp = nodes.new('ShaderNodeMath')
        dist_clamp.operation = 'MINIMUM'
        dist_clamp.location = (1800, 200)
        dist_clamp.inputs[1].default_value = 1.0
        links.new(dist_scale.outputs['Value'], dist_clamp.inputs[0])
        
        final_mix = nodes.new('ShaderNodeMix')
        final_mix.data_type = 'RGBA'
        final_mix.location = (2000, 200)
        links.new(dist_clamp.outputs['Value'], final_mix.inputs['Factor'])
        final_mix.inputs[6].default_value = (0, 0, 0, 1)
        links.new(result_rgb.outputs['Color'], final_mix.inputs[7])
        links.new(final_mix.outputs[2], emission.inputs['Color'])
    elif debug_mode == 1:
        # Rayleigh phase only (grayscale)
        phase_rgb = nodes.new('ShaderNodeCombineColor')
        phase_rgb.location = (1600, 200)
        links.new(rayleigh_phase.outputs['Value'], phase_rgb.inputs['Red'])
        links.new(rayleigh_phase.outputs['Value'], phase_rgb.inputs['Green'])
        links.new(rayleigh_phase.outputs['Value'], phase_rgb.inputs['Blue'])
        links.new(phase_rgb.outputs['Color'], emission.inputs['Color'])
    elif debug_mode == 2:
        # Mie phase only (grayscale, scaled up for visibility)
        mie_scaled = nodes.new('ShaderNodeMath')
        mie_scaled.operation = 'MULTIPLY'
        mie_scaled.location = (1600, -400)
        mie_scaled.inputs[1].default_value = 10.0  # Scale up for visibility
        links.new(mie_phase.outputs['Value'], mie_scaled.inputs[0])
        
        phase_rgb = nodes.new('ShaderNodeCombineColor')
        phase_rgb.location = (1800, 200)
        links.new(mie_scaled.outputs['Value'], phase_rgb.inputs['Red'])
        links.new(mie_scaled.outputs['Value'], phase_rgb.inputs['Green'])
        links.new(mie_scaled.outputs['Value'], phase_rgb.inputs['Blue'])
        links.new(phase_rgb.outputs['Color'], emission.inputs['Color'])
    elif debug_mode == 3:
        # Nu (view-sun dot product) - should vary with sun direction
        # Remap from [-1,1] to [0,1] for visualization
        nu_remap = nodes.new('ShaderNodeMath')
        nu_remap.operation = 'MULTIPLY_ADD'
        nu_remap.location = (1600, 200)
        nu_remap.inputs[1].default_value = 0.5  # multiply by 0.5
        nu_remap.inputs[2].default_value = 0.5  # add 0.5
        links.new(nu_dot.outputs['Value'], nu_remap.inputs[0])
        
        nu_rgb = nodes.new('ShaderNodeCombineColor')
        nu_rgb.location = (1800, 200)
        links.new(nu_remap.outputs['Value'], nu_rgb.inputs['Red'])
        links.new(nu_remap.outputs['Value'], nu_rgb.inputs['Green'])
        links.new(nu_remap.outputs['Value'], nu_rgb.inputs['Blue'])
        links.new(nu_rgb.outputs['Color'], emission.inputs['Color'])
    
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (2400, 200)
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    # Auto-assign
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
            obj.data.materials.append(mat)
    
    mesh_count = len([o for o in bpy.context.scene.objects if o.type == 'MESH'])
    print(f"\nCreated: {mat.name}")
    print(f"Assigned to {mesh_count} meshes")
    print(f"\nPhase functions:")
    print(f"  Rayleigh: (3/16π)(1 + nu²), k = {k_rayleigh:.5f}")
    print(f"  Mie: HG with g = {g}, k = {k_mie:.5f}")
    print(f"\nExpected: Brighter toward sun, blue Rayleigh + white Mie haze")
    
    return mat


# =============================================================================
# STEP 3 FULL: INSCATTER WITH PHASE FUNCTIONS
# =============================================================================

def apply_step_3_full():
    """
    Step 3 Full: Complete inscatter with phase functions
    
    Combines:
    - Full inscatter computation from Step 2.4 (S_cam - T×S_pt)
    - Rayleigh phase function on RGB channels
    - Mie phase function on alpha channel (single_mie_scattering)
    """
    import bpy
    import math
    import time
    import os
    
    print("Step 3 Full: Inscatter with Phase Functions")
    
    # Constants
    BOTTOM_RADIUS = 6360.0
    TOP_RADIUS = 6420.0
    H = TOP_RADIUS - BOTTOM_RADIUS
    MIE_G = 0.8
    PI = math.pi
    
    # LUT dimensions
    MU_SIZE = 128
    MU_S_SIZE = 32
    NU_SIZE = 8
    R_SIZE = 32
    
    # LUT paths
    lut_dir = r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-3\helios_cache\luts"
    scatter_path = os.path.join(lut_dir, "scattering.exr")
    
    print(f"  Scattering: {scatter_path}")
    
    # Create material
    mat_name = f"Step3_Full_{int(time.time())}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # ==========================================================================
    # GEOMETRY & CAMERA
    # ==========================================================================
    
    geom = nodes.new('ShaderNodeNewGeometry')
    geom.location = (-1800, 400)
    
    cam_loc = nodes.new('ShaderNodeCombineXYZ')
    cam_loc.location = (-1800, 200)
    cam = bpy.context.scene.camera
    cam_alt_km = 0.218
    if cam:
        cam_loc.inputs['X'].default_value = cam.location.x
        cam_loc.inputs['Y'].default_value = cam.location.y
        cam_loc.inputs['Z'].default_value = cam.location.z
        cam_alt_km = cam.location.z * 0.001
    
    r_cam = BOTTOM_RADIUS + cam_alt_km
    
    # Sun direction from scene
    sun_light = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN':
            sun_light = obj
            break
    
    sun_dir = nodes.new('ShaderNodeCombineXYZ')
    sun_dir.location = (-1800, 0)
    if sun_light:
        import mathutils
        sun_direction = sun_light.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))
        sun_direction.normalize()
        sun_dir.inputs['X'].default_value = sun_direction.x
        sun_dir.inputs['Y'].default_value = sun_direction.y
        sun_dir.inputs['Z'].default_value = sun_direction.z
        print(f"  Sun direction: ({sun_direction.x:.3f}, {sun_direction.y:.3f}, {sun_direction.z:.3f})")
    else:
        sun_dir.inputs['X'].default_value = 0.0
        sun_dir.inputs['Y'].default_value = 0.2588
        sun_dir.inputs['Z'].default_value = 0.9659
        print(f"  Sun: default 15° elevation")
    
    # ==========================================================================
    # DISTANCE & VIEW DIRECTION
    # ==========================================================================
    
    pos_to_cam = nodes.new('ShaderNodeVectorMath')
    pos_to_cam.operation = 'SUBTRACT'
    pos_to_cam.location = (-1600, 300)
    links.new(geom.outputs['Position'], pos_to_cam.inputs[0])
    links.new(cam_loc.outputs['Vector'], pos_to_cam.inputs[1])
    
    d_vec_len = nodes.new('ShaderNodeVectorMath')
    d_vec_len.operation = 'LENGTH'
    d_vec_len.location = (-1400, 300)
    links.new(pos_to_cam.outputs['Vector'], d_vec_len.inputs[0])
    
    d = nodes.new('ShaderNodeMath')
    d.operation = 'MULTIPLY'
    d.location = (-1200, 300)
    d.inputs[1].default_value = 0.001
    links.new(d_vec_len.outputs['Value'], d.inputs[0])
    
    view_dir = nodes.new('ShaderNodeVectorMath')
    view_dir.operation = 'NORMALIZE'
    view_dir.location = (-1400, 150)
    links.new(pos_to_cam.outputs['Vector'], view_dir.inputs[0])
    
    # up vector
    up_vec = nodes.new('ShaderNodeCombineXYZ')
    up_vec.location = (-1800, -100)
    up_vec.inputs['X'].default_value = 0.0
    up_vec.inputs['Y'].default_value = 0.0
    up_vec.inputs['Z'].default_value = 1.0
    
    # mu = dot(view, up)
    mu_dot = nodes.new('ShaderNodeVectorMath')
    mu_dot.operation = 'DOT_PRODUCT'
    mu_dot.location = (-1200, 100)
    links.new(view_dir.outputs['Vector'], mu_dot.inputs[0])
    links.new(up_vec.outputs['Vector'], mu_dot.inputs[1])
    
    # mu_s = dot(up, sun)
    mu_s_dot = nodes.new('ShaderNodeVectorMath')
    mu_s_dot.operation = 'DOT_PRODUCT'
    mu_s_dot.location = (-1200, -100)
    links.new(up_vec.outputs['Vector'], mu_s_dot.inputs[0])
    links.new(sun_dir.outputs['Vector'], mu_s_dot.inputs[1])
    
    # nu = dot(view, sun)
    nu_dot = nodes.new('ShaderNodeVectorMath')
    nu_dot.operation = 'DOT_PRODUCT'
    nu_dot.location = (-1200, -250)
    links.new(view_dir.outputs['Vector'], nu_dot.inputs[0])
    links.new(sun_dir.outputs['Vector'], nu_dot.inputs[1])
    
    # ==========================================================================
    # PHASE FUNCTIONS
    # ==========================================================================
    
    # Rayleigh: (3/16π)(1 + nu²)
    nu_sq = nodes.new('ShaderNodeMath')
    nu_sq.operation = 'MULTIPLY'
    nu_sq.location = (-1000, -300)
    links.new(nu_dot.outputs['Value'], nu_sq.inputs[0])
    links.new(nu_dot.outputs['Value'], nu_sq.inputs[1])
    
    one_plus_nu_sq = nodes.new('ShaderNodeMath')
    one_plus_nu_sq.operation = 'ADD'
    one_plus_nu_sq.location = (-800, -300)
    one_plus_nu_sq.inputs[0].default_value = 1.0
    links.new(nu_sq.outputs['Value'], one_plus_nu_sq.inputs[1])
    
    k_rayleigh = 3.0 / (16.0 * PI)
    rayleigh_phase = nodes.new('ShaderNodeMath')
    rayleigh_phase.operation = 'MULTIPLY'
    rayleigh_phase.location = (-600, -300)
    rayleigh_phase.inputs[0].default_value = k_rayleigh
    links.new(one_plus_nu_sq.outputs['Value'], rayleigh_phase.inputs[1])
    
    # Mie: k * (1 + nu²) / (1 + g² - 2g*nu)^1.5
    g = MIE_G
    g_sq = g * g
    k_mie = (3.0 / (8.0 * PI)) * (1.0 - g_sq) / (2.0 + g_sq)
    
    two_g_nu = nodes.new('ShaderNodeMath')
    two_g_nu.operation = 'MULTIPLY'
    two_g_nu.location = (-1000, -500)
    two_g_nu.inputs[0].default_value = 2.0 * g
    links.new(nu_dot.outputs['Value'], two_g_nu.inputs[1])
    
    denom_base = nodes.new('ShaderNodeMath')
    denom_base.operation = 'SUBTRACT'
    denom_base.location = (-800, -500)
    denom_base.inputs[0].default_value = 1.0 + g_sq
    links.new(two_g_nu.outputs['Value'], denom_base.inputs[1])
    
    denom_clamp = nodes.new('ShaderNodeMath')
    denom_clamp.operation = 'MAXIMUM'
    denom_clamp.location = (-600, -500)
    denom_clamp.inputs[1].default_value = 0.001
    links.new(denom_base.outputs['Value'], denom_clamp.inputs[0])
    
    denom_sqrt = nodes.new('ShaderNodeMath')
    denom_sqrt.operation = 'SQRT'
    denom_sqrt.location = (-400, -550)
    links.new(denom_clamp.outputs['Value'], denom_sqrt.inputs[0])
    
    denom_pow15 = nodes.new('ShaderNodeMath')
    denom_pow15.operation = 'MULTIPLY'
    denom_pow15.location = (-200, -500)
    links.new(denom_clamp.outputs['Value'], denom_pow15.inputs[0])
    links.new(denom_sqrt.outputs['Value'], denom_pow15.inputs[1])
    
    mie_numer = nodes.new('ShaderNodeMath')
    mie_numer.operation = 'MULTIPLY'
    mie_numer.location = (-400, -400)
    mie_numer.inputs[0].default_value = k_mie
    links.new(one_plus_nu_sq.outputs['Value'], mie_numer.inputs[1])
    
    mie_phase = nodes.new('ShaderNodeMath')
    mie_phase.operation = 'DIVIDE'
    mie_phase.location = (0, -450)
    links.new(mie_numer.outputs['Value'], mie_phase.inputs[0])
    links.new(denom_pow15.outputs['Value'], mie_phase.inputs[1])
    
    # ==========================================================================
    # LOAD SCATTERING TEXTURE
    # ==========================================================================
    
    scatter_tex = nodes.new('ShaderNodeTexImage')
    scatter_tex.location = (200, 400)
    scatter_tex.interpolation = 'Linear'
    scatter_tex.extension = 'CLIP'
    
    img = bpy.data.images.get(os.path.basename(scatter_path))
    if not img:
        img = bpy.data.images.load(scatter_path)
    img.colorspace_settings.name = 'Non-Color'
    scatter_tex.image = img
    
    # ==========================================================================
    # SIMPLIFIED SCATTERING UV (for camera position)
    # Using a simplified approach for now - fixed ground-level sample
    # ==========================================================================
    
    # u_mu_s = (1 - exp(-3*mu_s - 0.6)) / (1 - exp(-3.6))
    mu_s_scaled = nodes.new('ShaderNodeMath')
    mu_s_scaled.operation = 'MULTIPLY'
    mu_s_scaled.location = (0, 200)
    mu_s_scaled.inputs[0].default_value = -3.0
    links.new(mu_s_dot.outputs['Value'], mu_s_scaled.inputs[1])
    
    mu_s_offset = nodes.new('ShaderNodeMath')
    mu_s_offset.operation = 'ADD'
    mu_s_offset.location = (200, 200)
    mu_s_offset.inputs[1].default_value = -0.6
    links.new(mu_s_scaled.outputs['Value'], mu_s_offset.inputs[0])
    
    mu_s_exp = nodes.new('ShaderNodeMath')
    mu_s_exp.operation = 'EXPONENT'
    mu_s_exp.location = (400, 200)
    links.new(mu_s_offset.outputs['Value'], mu_s_exp.inputs[0])
    
    u_mu_s_num = nodes.new('ShaderNodeMath')
    u_mu_s_num.operation = 'SUBTRACT'
    u_mu_s_num.location = (600, 200)
    u_mu_s_num.inputs[0].default_value = 1.0
    links.new(mu_s_exp.outputs['Value'], u_mu_s_num.inputs[1])
    
    u_mu_s = nodes.new('ShaderNodeMath')
    u_mu_s.operation = 'DIVIDE'
    u_mu_s.location = (800, 200)
    u_mu_s.inputs[1].default_value = 1.0 - math.exp(-3.6)
    links.new(u_mu_s_num.outputs['Value'], u_mu_s.inputs[0])
    
    # u_nu = (nu + 1) / 2
    u_nu = nodes.new('ShaderNodeMath')
    u_nu.operation = 'MULTIPLY_ADD'
    u_nu.location = (800, 0)
    u_nu.inputs[1].default_value = 0.5
    u_nu.inputs[2].default_value = 0.5
    links.new(nu_dot.outputs['Value'], u_nu.inputs[0])
    
    # u_mu simplified: (1 + mu) / 2
    u_mu = nodes.new('ShaderNodeMath')
    u_mu.operation = 'MULTIPLY_ADD'
    u_mu.location = (800, -150)
    u_mu.inputs[1].default_value = 0.5
    u_mu.inputs[2].default_value = 0.5
    links.new(mu_dot.outputs['Value'], u_mu.inputs[0])
    
    # Combine into UV (simplified - using depth 0 for ground level)
    # X = (u_nu + depth * NU_SIZE) / (NU_SIZE * R_SIZE) + u_mu_s / MU_S_SIZE
    # Simplified: X = u_mu_s (using first slice)
    # Y = 1 - u_mu (Y-flip for Blender)
    
    scatter_uv = nodes.new('ShaderNodeCombineXYZ')
    scatter_uv.location = (1000, 100)
    links.new(u_mu_s.outputs['Value'], scatter_uv.inputs['X'])
    
    y_flip = nodes.new('ShaderNodeMath')
    y_flip.operation = 'SUBTRACT'
    y_flip.location = (1000, -100)
    y_flip.inputs[0].default_value = 1.0
    links.new(u_mu.outputs['Value'], y_flip.inputs[1])
    links.new(y_flip.outputs['Value'], scatter_uv.inputs['Y'])
    
    links.new(scatter_uv.outputs['Vector'], scatter_tex.inputs['Vector'])
    
    # ==========================================================================
    # APPLY PHASE FUNCTIONS TO SCATTERING
    # Result = Rayleigh_RGB * RayleighPhase + Mie_alpha * MiePhase
    # ==========================================================================
    
    # Separate Rayleigh (RGB) and Mie (Alpha)
    sep_scatter = nodes.new('ShaderNodeSeparateColor')
    sep_scatter.location = (1200, 300)
    links.new(scatter_tex.outputs['Color'], sep_scatter.inputs['Color'])
    
    # Rayleigh * phase
    ray_r = nodes.new('ShaderNodeMath')
    ray_r.operation = 'MULTIPLY'
    ray_r.location = (1400, 350)
    links.new(sep_scatter.outputs['Red'], ray_r.inputs[0])
    links.new(rayleigh_phase.outputs['Value'], ray_r.inputs[1])
    
    ray_g = nodes.new('ShaderNodeMath')
    ray_g.operation = 'MULTIPLY'
    ray_g.location = (1400, 250)
    links.new(sep_scatter.outputs['Green'], ray_g.inputs[0])
    links.new(rayleigh_phase.outputs['Value'], ray_g.inputs[1])
    
    ray_b = nodes.new('ShaderNodeMath')
    ray_b.operation = 'MULTIPLY'
    ray_b.location = (1400, 150)
    links.new(sep_scatter.outputs['Blue'], ray_b.inputs[0])
    links.new(rayleigh_phase.outputs['Value'], ray_b.inputs[1])
    
    # Mie * phase (single channel from alpha)
    mie_contrib = nodes.new('ShaderNodeMath')
    mie_contrib.operation = 'MULTIPLY'
    mie_contrib.location = (1400, -50)
    links.new(scatter_tex.outputs['Alpha'], mie_contrib.inputs[0])
    links.new(mie_phase.outputs['Value'], mie_contrib.inputs[1])
    
    # Combine: Rayleigh + Mie
    final_r = nodes.new('ShaderNodeMath')
    final_r.operation = 'ADD'
    final_r.location = (1600, 350)
    links.new(ray_r.outputs['Value'], final_r.inputs[0])
    links.new(mie_contrib.outputs['Value'], final_r.inputs[1])
    
    final_g = nodes.new('ShaderNodeMath')
    final_g.operation = 'ADD'
    final_g.location = (1600, 250)
    links.new(ray_g.outputs['Value'], final_g.inputs[0])
    links.new(mie_contrib.outputs['Value'], final_g.inputs[1])
    
    final_b = nodes.new('ShaderNodeMath')
    final_b.operation = 'ADD'
    final_b.location = (1600, 150)
    links.new(ray_b.outputs['Value'], final_b.inputs[0])
    links.new(mie_contrib.outputs['Value'], final_b.inputs[1])
    
    final_rgb = nodes.new('ShaderNodeCombineColor')
    final_rgb.location = (1800, 250)
    links.new(final_r.outputs['Value'], final_rgb.inputs['Red'])
    links.new(final_g.outputs['Value'], final_rgb.inputs['Green'])
    links.new(final_b.outputs['Value'], final_rgb.inputs['Blue'])
    
    # ==========================================================================
    # DISTANCE MODULATION (aerial perspective increases with distance)
    # ==========================================================================
    
    # Simple distance factor: 1 - exp(-d * scale)
    d_scaled = nodes.new('ShaderNodeMath')
    d_scaled.operation = 'MULTIPLY'
    d_scaled.location = (1800, -100)
    d_scaled.inputs[1].default_value = -0.5  # negative for exp decay
    links.new(d.outputs['Value'], d_scaled.inputs[0])
    
    d_exp = nodes.new('ShaderNodeMath')
    d_exp.operation = 'EXPONENT'
    d_exp.location = (2000, -100)
    links.new(d_scaled.outputs['Value'], d_exp.inputs[0])
    
    d_factor = nodes.new('ShaderNodeMath')
    d_factor.operation = 'SUBTRACT'
    d_factor.location = (2200, -100)
    d_factor.inputs[0].default_value = 1.0
    links.new(d_exp.outputs['Value'], d_factor.inputs[1])
    
    # Scale inscatter by distance factor
    inscatter_scaled = nodes.new('ShaderNodeMix')
    inscatter_scaled.data_type = 'RGBA'
    inscatter_scaled.location = (2000, 250)
    links.new(d_factor.outputs['Value'], inscatter_scaled.inputs['Factor'])
    inscatter_scaled.inputs[6].default_value = (0, 0, 0, 1)  # Near = black
    links.new(final_rgb.outputs['Color'], inscatter_scaled.inputs[7])  # Far = scatter
    
    # ==========================================================================
    # OUTPUT (debug_mode: 0=full, 1=scatter_raw, 2=UV, 3=distance, 4=phase)
    # ==========================================================================
    
    debug_mode = 2  # Output UV coordinates
    
    emission = nodes.new('ShaderNodeEmission')
    emission.location = (2400, 250)
    emission.inputs['Strength'].default_value = 1.0
    
    if debug_mode == 0:
        # Full output
        links.new(inscatter_scaled.outputs[2], emission.inputs['Color'])
    elif debug_mode == 1:
        # Raw scattering texture (before phase functions)
        links.new(scatter_tex.outputs['Color'], emission.inputs['Color'])
    elif debug_mode == 2:
        # UV coordinates visualization
        links.new(scatter_uv.outputs['Vector'], emission.inputs['Color'])
    elif debug_mode == 3:
        # Distance factor
        dist_rgb = nodes.new('ShaderNodeCombineColor')
        dist_rgb.location = (2200, 100)
        links.new(d_factor.outputs['Value'], dist_rgb.inputs['Red'])
        links.new(d_factor.outputs['Value'], dist_rgb.inputs['Green'])
        links.new(d_factor.outputs['Value'], dist_rgb.inputs['Blue'])
        links.new(dist_rgb.outputs['Color'], emission.inputs['Color'])
    elif debug_mode == 4:
        # Phase functions (Rayleigh scaled up)
        phase_rgb = nodes.new('ShaderNodeCombineColor')
        phase_rgb.location = (2200, 100)
        ray_scaled = nodes.new('ShaderNodeMath')
        ray_scaled.operation = 'MULTIPLY'
        ray_scaled.location = (2000, 100)
        ray_scaled.inputs[1].default_value = 10.0
        links.new(rayleigh_phase.outputs['Value'], ray_scaled.inputs[0])
        links.new(ray_scaled.outputs['Value'], phase_rgb.inputs['Red'])
        links.new(ray_scaled.outputs['Value'], phase_rgb.inputs['Green'])
        links.new(ray_scaled.outputs['Value'], phase_rgb.inputs['Blue'])
        links.new(phase_rgb.outputs['Color'], emission.inputs['Color'])
    
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (2600, 250)
    links.new(emission.outputs['Emission'], output.inputs['Surface'])
    
    # Auto-assign
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            obj.data.materials.clear()
            obj.data.materials.append(mat)
    
    mesh_count = len([o for o in bpy.context.scene.objects if o.type == 'MESH'])
    print(f"\nCreated: {mat.name}")
    print(f"Assigned to {mesh_count} meshes")
    print(f"\nPhase functions applied:")
    print(f"  Rayleigh k = {k_rayleigh:.5f}")
    print(f"  Mie k = {k_mie:.5f}, g = {g}")
    print(f"\nExpected: Blue inscatter, brighter toward sun, distance falloff")
    
    return mat


# =============================================================================
# STEP 5: FULL BRUNETON AERIAL PERSPECTIVE (LUT-based Transmittance)
# =============================================================================

def apply_step_5_full_bruneton(surface_color=(0.8, 0.8, 0.8), debug_mode=0):
    """
    Step 5: Full Bruneton Aerial Perspective with LUT-based Transmittance
    
    Uses:
    - LUT-based Rayleigh scattering (scattering.exr RGB)
    - LUT-based Mie scattering (scattering.exr Alpha)
    - LUT-based Transmittance (transmittance.exr) - ratio method
    - Proper phase functions
    
    Formula: L_final = L_surface × T + inscatter
    
    Args:
        surface_color: RGB tuple for object base color
        debug_mode: 0=full, 1=surface×T, 2=T_only, 3=inscatter, 4=rayleigh, 5=mie
    """
    import bpy
    import math
    import mathutils
    import time
    import os
    
    print("=" * 60)
    print("Step 5: Full Bruneton Aerial Perspective (LUT-based)")
    print("=" * 60)
    
    # Constants
    BOTTOM_RADIUS = 6360.0
    TOP_RADIUS = 6420.0
    H = math.sqrt(TOP_RADIUS**2 - BOTTOM_RADIUS**2)
    MU_S_MIN = -0.2
    SCATTERING_R_SIZE = 32
    SCATTERING_MU_SIZE = 128
    SCATTERING_MU_S_SIZE = 32
    SCATTERING_NU_SIZE = 8
    TRANSMITTANCE_WIDTH = 256
    TRANSMITTANCE_HEIGHT = 64
    PI = 3.14159265359
    MIE_G = 0.8
    g_sq = MIE_G * MIE_G
    
    # Load LUTs
    lut_dir = r"C:\Users\space\Documents\mattepaint\dev\atmospheric-scattering-3\helios_cache\luts"
    scatter_path = os.path.join(lut_dir, "scattering.exr")
    trans_path = os.path.join(lut_dir, "transmittance.exr")
    
    if not os.path.exists(scatter_path) or not os.path.exists(trans_path):
        print(f"  ERROR: LUTs not found in {lut_dir}")
        return None
    
    print(f"  Scattering: {scatter_path}")
    print(f"  Transmittance: {trans_path}")
    
    scatter_img = bpy.data.images.load(scatter_path, check_existing=True)
    scatter_img.colorspace_settings.name = 'Non-Color'
    trans_img = bpy.data.images.load(trans_path, check_existing=True)
    trans_img.colorspace_settings.name = 'Non-Color'
    
    # Get scene data
    cam = bpy.context.scene.camera
    if not cam:
        print("  ERROR: No camera")
        return None
    
    sun_light = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT' and obj.data.type == 'SUN':
            sun_light = obj
            break
    
    if sun_light:
        sun_direction = sun_light.matrix_world.to_quaternion() @ mathutils.Vector((0, 0, -1))
        sun_direction.normalize()
    else:
        sun_direction = mathutils.Vector((0.0, 0.2588, 0.9659))
    print(f"  Sun: ({sun_direction.x:.3f}, {sun_direction.y:.3f}, {sun_direction.z:.3f})")
    
    cam_alt_km = cam.location.z * 0.001
    r_cam = BOTTOM_RADIUS + cam_alt_km
    print(f"  Camera: alt={cam_alt_km:.3f}km, r={r_cam:.4f}km")
    
    # This function reuses Step 2.4's scattering UV and adds proper transmittance
    # For now, call Step 2.4 as base and note that full implementation is WIP
    print("\n  NOTE: Step 5 builds on Step 2.4 with LUT-based transmittance")
    print("  Full implementation requires proper transmittance UV sampling")
    print("  Currently using Step 2.4 as reference implementation")
    
    # Call the existing inscatter function which has working scattering
    mat = apply_step_2_4_inscatter()
    
    if mat:
        # Rename to indicate Step 5
        mat.name = f"Step5_Bruneton_{int(time.time())}"
        print(f"\n  Created: {mat.name}")
        print(f"  Debug mode: {debug_mode}")
        print(f"\n  TODO: Integrate LUT-based transmittance into this material")
    
    return mat


# =============================================================================
# MAIN - Run when script is executed in Blender
# =============================================================================

if __name__ == "__main__":
    pass
