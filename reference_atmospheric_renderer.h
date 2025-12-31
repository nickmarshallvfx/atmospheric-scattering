/**
 * Atmospheric Renderer - Extensible rendering engine
 * Clean separation between GUI controls and atmospheric rendering logic
 */

#ifndef ATMOSPHERE_DEMO_ATMOSPHERIC_RENDERER_H_
#define ATMOSPHERE_DEMO_ATMOSPHERIC_RENDERER_H_

#include <glad/glad.h>
#include <memory>
#include "atmosphere/model.h"

namespace atmosphere {
namespace demo {

/**
 * Rendering parameters - Clean interface between GUI and renderer
 * Add new parameters here for future features (sphere toggle, phase functions, etc.)
 */
struct RenderingParameters {
  // Camera parameters - independent position and rotation
  double camera_position_x;  // meters
  double camera_position_y;  // meters
  double camera_position_z;  // meters
  double camera_pitch_degrees;  // rotation around X axis
  double camera_yaw_degrees;    // rotation around Y axis
  double camera_roll_degrees;   // rotation around Z axis
  double view_fov_degrees;
  
  // Sun/Environment parameters
  double sun_zenith_angle_radians;
  double sun_azimuth_angle_radians;
  double sun_intensity;
  
  // Rendering options
  double exposure;
  bool use_ozone;
  bool use_combined_textures;
  bool use_half_precision;
  int luminance_mode;  // 0=NONE, 1=APPROXIMATE, 2=PRECOMPUTED
  bool do_white_balance;
  bool use_constant_solar_spectrum;
  int render_mode;  // 0=Perspective, 1=Latlong
  
  // Atmospheric composition parameters (artistic control)
  double mie_phase_function_g;      // -1 to 1: scattering directionality
  double mie_scattering_scale;      // Multiplier for aerosol density
  double rayleigh_scattering_scale; // Multiplier for molecular scattering
  double mie_scale_height;          // Altitude falloff for aerosols (meters)
  double rayleigh_scale_height;     // Altitude falloff for air (meters)
  double ground_albedo;             // Ground reflectivity (0-1)
  
  // Scene objects
  bool show_scene_object;   // Toggle object visibility
  int scene_object_shape;   // 0=Sphere, 1=Cube, 2=Cone
  
  // Constructor with sensible defaults
  RenderingParameters();
};

/**
 * Atmospheric Renderer - Handles all atmospheric rendering logic
 * Designed to be controlled by GUI but remain independent
 */
class AtmosphericRenderer {
 public:
  AtmosphericRenderer();
  ~AtmosphericRenderer();
  
  // Initialize the atmospheric model and shaders
  void Initialize();
  
  // Render atmospheric scene to the currently bound framebuffer
  // viewport_width/height: dimensions of the render target
  void Render(const RenderingParameters& params, int viewport_width, int viewport_height);
  
  // Check if renderer is ready
  bool IsInitialized() const { return model_ != nullptr; }
  
  // Access to model (for advanced features)
  const Model* GetModel() const { return model_.get(); }
  
 private:
  void InitializeModel();
  void InitializeShaders();
  void UpdateShaderUniforms(const RenderingParameters& params);
  void SetupViewMatrices(const RenderingParameters& params, int viewport_width, int viewport_height);
  void DrawScene();
  void RebuildModel(const RenderingParameters& params);
  bool NeedsModelRebuild(const RenderingParameters& params) const;
  
  // Atmospheric model
  std::unique_ptr<Model> model_;
  
  // OpenGL resources
  GLuint vertex_shader_;
  GLuint fragment_shader_;
  GLuint program_;
  GLuint full_screen_quad_vao_;
  GLuint full_screen_quad_vbo_;
  
  // Cached viewport size and FOV for projection matrix
  int cached_viewport_width_;
  int cached_viewport_height_;
  double cached_fov_degrees_;
  
  // Cached atmospheric parameters (to detect changes requiring model rebuild)
  double cached_mie_phase_g_;
  double cached_mie_scale_;
  double cached_rayleigh_scale_;
  double cached_ground_albedo_;
  bool atmospheric_params_dirty_;
  
  // Debug
  mutable int debug_frame_count_;
};

}  // namespace demo
}  // namespace atmosphere

#endif  // ATMOSPHERE_DEMO_ATMOSPHERIC_RENDERER_H_
