"""
Helios Precompute Operator - Precompute atmospheric LUTs.
"""

import bpy
from bpy.types import Operator
import os


class HELIOS_OT_precompute_luts(Operator):
    """Precompute atmospheric scattering lookup tables"""
    bl_idname = "helios.precompute_luts"
    bl_label = "Precompute Atmosphere"
    bl_description = "Precompute atmospheric scattering lookup tables (takes 10-30 seconds)"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        try:
            from ..core import AtmosphereModel, AtmosphereParameters
            from ..core import BLENDER_GPU_AVAILABLE, CUPY_AVAILABLE
            from .. import world as world_module
        except ImportError as e:
            self.report({'ERROR'}, f"Failed to import core module: {e}")
            return {'CANCELLED'}
        
        settings = context.scene.helios
        
        # Clear sky shader AND switch viewport to SOLID before baking
        # This prevents Cycles from competing for GPU resources
        world = context.scene.world
        had_helios_world = False
        previous_shading_types = []
        
        # Switch all rendered viewports to SOLID mode
        for window in context.window_manager.windows:
            for area in window.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D' and space.shading.type == 'RENDERED':
                            previous_shading_types.append((space, 'RENDERED'))
                            space.shading.type = 'SOLID'
        
        if previous_shading_types:
            print(f"Helios: Switched {len(previous_shading_types)} viewport(s) to SOLID for faster baking")
        
        if world and world.use_nodes and world.get("is_helios"):
            had_helios_world = True
            world.node_tree.nodes.clear()
            print("Helios: Cleared sky shader for faster baking")
        
        # Create parameters from UI settings
        params = AtmosphereParameters.from_artistic_controls(
            rayleigh_density_scale=settings.rayleigh_density,
            mie_density_scale=settings.mie_density,
            mie_phase_g=settings.mie_phase_g,
            rayleigh_height=settings.rayleigh_scale_height,
            mie_height=settings.mie_scale_height,
            ground_albedo=settings.ground_albedo,
            use_ozone=settings.use_ozone,
        )
        
        # Update planet parameters
        params.bottom_radius = settings.planet_radius * 1000.0  # km to m
        params.top_radius = params.bottom_radius + settings.atmosphere_height * 1000.0
        
        # Select backend: Blender GPU > CuPy > CPU
        use_gpu = getattr(settings, 'use_gpu', True)
        
        if use_gpu and BLENDER_GPU_AVAILABLE:
            from ..core import BlenderGPUAtmosphereModel
            self.report({'INFO'}, "Using Blender GPU acceleration")
            model = BlenderGPUAtmosphereModel(params)
        elif use_gpu and CUPY_AVAILABLE:
            from ..core import GPUAtmosphereModel
            self.report({'INFO'}, "Using CuPy GPU acceleration")
            model = GPUAtmosphereModel(params, use_gpu=True)
        else:
            self.report({'INFO'}, "Using CPU (NumPy)")
            model = AtmosphereModel(params)
        
        import sys
        def progress_callback(progress, message):
            context.window_manager.progress_update(int(progress * 100))
            self.report({'INFO'}, message)
            print(f"Helios: {message} ({int(progress*100)}%)")
            sys.stdout.flush()
        
        # Determine quality based on preview setting
        is_preview = settings.preview_quality == 'PREVIEW'
        num_orders = 2 if is_preview else 4
        
        context.window_manager.progress_begin(0, 100)
        try:
            model.init(num_scattering_orders=num_orders, preview_mode=is_preview, progress_callback=progress_callback)
            
            # Save textures as EXR - use blend file directory or user config
            blend_path = bpy.data.filepath
            if blend_path:
                # Save next to the blend file
                lut_cache_dir = os.path.join(os.path.dirname(blend_path), "helios_cache", "luts")
            else:
                # Fallback to Blender's user config directory
                config_dir = bpy.utils.user_resource('CONFIG')
                lut_cache_dir = os.path.join(config_dir, "helios_cache", "luts")
            
            # Save as EXR for OSL shader
            model.save_textures_exr(lut_cache_dir)
            
            # Also save NPZ backup to blend file directory if available
            blend_path = bpy.data.filepath
            if blend_path:
                npz_cache_dir = os.path.join(os.path.dirname(blend_path), "helios_cache")
                os.makedirs(npz_cache_dir, exist_ok=True)
                npz_file = os.path.join(npz_cache_dir, "atmosphere_luts.npz")
                model.save_textures(npz_file)
            
            settings.luts_valid = True
            self.report({'INFO'}, f"Atmosphere LUTs saved to {lut_cache_dir}")
            
            # Rebuild sky shader if it was cleared
            if had_helios_world and world and world.use_nodes:
                world_module._build_sky_nodes(
                    world.node_tree.nodes,
                    world.node_tree.links,
                    settings
                )
                print("Helios: Rebuilt sky shader with fresh LUTs")
            
            # Restore viewport shading modes
            for space, shading_type in previous_shading_types:
                space.shading.type = shading_type
            if previous_shading_types:
                print(f"Helios: Restored {len(previous_shading_types)} viewport(s) to RENDERED")
            
        except Exception as e:
            self.report({'ERROR'}, f"Precomputation failed: {str(e)}")
            settings.luts_valid = False
            # Still try to rebuild shader on error
            if had_helios_world and world and world.use_nodes:
                world_module._build_sky_nodes(
                    world.node_tree.nodes,
                    world.node_tree.links,
                    settings
                )
            # Restore viewport shading modes on error too
            for space, shading_type in previous_shading_types:
                space.shading.type = shading_type
            return {'CANCELLED'}
        finally:
            context.window_manager.progress_end()
        
        return {'FINISHED'}
    
    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)
