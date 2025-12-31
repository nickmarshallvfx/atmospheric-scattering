/**
 * Atmospheric Renderer Implementation
 * Extracted rendering logic with clean parameter interface
 */

#include "atmosphere/demo/atmospheric_renderer.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

namespace atmosphere {
namespace demo {

namespace {

constexpr double kPi = 3.1415926;
constexpr double kSunAngularRadius = 0.00935 / 2.0;
constexpr double kSunSolidAngle = kPi * kSunAngularRadius * kSunAngularRadius;
constexpr double kLengthUnitInMeters = 1000.0;

const char kVertexShader[] = R"(
    #version 330
    uniform mat4 model_from_view;
    uniform mat4 view_from_clip;
    layout(location = 0) in vec4 vertex;
    out vec3 view_ray;
    void main() {
      view_ray =
          (model_from_view * vec4((view_from_clip * vertex).xyz, 0.0)).xyz;
      gl_Position = vertex;
    })";

#include "atmosphere/demo/demo.glsl.inc"

}  // anonymous namespace

// ============================================================================
// RenderingParameters implementation
// ============================================================================

RenderingParameters::RenderingParameters() 
    : camera_position_x(0.0),
      camera_position_y(1000.0),
      camera_position_z(9000.0),
      camera_pitch_degrees(90.0),
      camera_yaw_degrees(180.0),
      camera_roll_degrees(0.0),
      view_fov_degrees(50.0),
      sun_zenith_angle_radians(1.3),
      sun_azimuth_angle_radians(2.9),
      sun_intensity(1.0),
      exposure(10.0),
      use_ozone(true),
      use_combined_textures(true),
      use_half_precision(true),
      luminance_mode(0),  // NONE
      do_white_balance(false),
      use_constant_solar_spectrum(false),
      render_mode(0),  // Perspective
      mie_phase_function_g(0.8),            // Default from kMiePhaseFunctionG
      mie_scattering_scale(1.0),            // 1.0 = default density
      rayleigh_scattering_scale(1.0),       // 1.0 = default density
      mie_scale_height(1200.0),             // Default from kMieScaleHeight
      rayleigh_scale_height(8000.0),        // Default from kRayleighScaleHeight
      ground_albedo(0.1),                   // Default from kGroundAlbedo
      show_scene_object(true),              // Show object by default
      scene_object_shape(0)                 // 0 = Sphere
{
}

// ============================================================================
// AtmosphericRenderer implementation
// ============================================================================

AtmosphericRenderer::AtmosphericRenderer()
    : model_(nullptr),
      vertex_shader_(0),
      fragment_shader_(0),
      program_(0),
      full_screen_quad_vao_(0),
      full_screen_quad_vbo_(0),
      cached_viewport_width_(0),
      cached_viewport_height_(0),
      cached_fov_degrees_(0.0),
      cached_mie_phase_g_(0.8),
      cached_mie_scale_(1.0),
      cached_rayleigh_scale_(1.0),
      cached_ground_albedo_(0.1),
      atmospheric_params_dirty_(false),
      debug_frame_count_(0) {
}

AtmosphericRenderer::~AtmosphericRenderer() {
  if (program_ != 0) glDeleteProgram(program_);
  if (fragment_shader_ != 0) glDeleteShader(fragment_shader_);
  if (vertex_shader_ != 0) glDeleteShader(vertex_shader_);
  if (full_screen_quad_vbo_ != 0) glDeleteBuffers(1, &full_screen_quad_vbo_);
  if (full_screen_quad_vao_ != 0) glDeleteVertexArrays(1, &full_screen_quad_vao_);
}

void AtmosphericRenderer::Initialize() {
  // Create full-screen quad for rendering
  glGenVertexArrays(1, &full_screen_quad_vao_);
  glBindVertexArray(full_screen_quad_vao_);
  glGenBuffers(1, &full_screen_quad_vbo_);
  glBindBuffer(GL_ARRAY_BUFFER, full_screen_quad_vbo_);
  
  const GLfloat vertices[] = {
    -1.0, -1.0, 0.0, 1.0,
    +1.0, -1.0, 0.0, 1.0,
    -1.0, +1.0, 0.0, 1.0,
    +1.0, +1.0, 0.0, 1.0,
  };
  
  glBufferData(GL_ARRAY_BUFFER, sizeof vertices, vertices, GL_STATIC_DRAW);
  constexpr GLuint kAttribIndex = 0;
  constexpr int kCoordsPerVertex = 4;
  glVertexAttribPointer(kAttribIndex, kCoordsPerVertex, GL_FLOAT, false, 0, 0);
  glEnableVertexAttribArray(kAttribIndex);
  glBindVertexArray(0);
  
  // Initialize with default parameters to set up the model
  RenderingParameters default_params;
  InitializeModel();
  InitializeShaders();
}

void AtmosphericRenderer::InitializeModel() {
  // Solar spectral irradiance data
  constexpr int kLambdaMin = 360;
  constexpr int kLambdaMax = 830;
  constexpr double kSolarIrradiance[48] = {
    1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887, 1.61253,
    1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298,
    1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533,
    1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482,
    1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758, 1.2367, 1.2082,
    1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992
  };
  
  // Ozone cross-section data
  constexpr double kOzoneCrossSection[48] = {
    1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27,
    8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26,
    1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25,
    4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25,
    2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26,
    6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26,
    2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27
  };
  
  constexpr double kDobsonUnit = 2.687e20;
  constexpr double kMaxOzoneNumberDensity = 300.0 * kDobsonUnit / 15000.0;
  constexpr double kConstantSolarIrradiance = 1.5;
  constexpr double kBottomRadius = 6360000.0;
  constexpr double kTopRadius = 6420000.0;
  constexpr double kRayleigh = 1.24062e-6;
  constexpr double kRayleighScaleHeight = 8000.0;
  constexpr double kMieScaleHeight = 1200.0;
  constexpr double kMieAngstromAlpha = 0.0;
  constexpr double kMieAngstromBeta = 5.328e-3;
  constexpr double kMieSingleScatteringAlbedo = 0.9;
  constexpr double kMiePhaseFunctionG = 0.8;
  constexpr double kGroundAlbedo = 0.1;
  
  // Use default parameters for initialization (will be updated per render)
  RenderingParameters default_params;
  const double max_sun_zenith_angle =
      (default_params.use_half_precision ? 102.0 : 120.0) / 180.0 * kPi;
  
  DensityProfileLayer rayleigh_layer(0.0, 1.0, -1.0 / kRayleighScaleHeight, 0.0, 0.0);
  DensityProfileLayer mie_layer(0.0, 1.0, -1.0 / kMieScaleHeight, 0.0, 0.0);
  
  std::vector<DensityProfileLayer> ozone_density;
  ozone_density.push_back(
      DensityProfileLayer(25000.0, 0.0, 0.0, 1.0 / 15000.0, -2.0 / 3.0));
  ozone_density.push_back(
      DensityProfileLayer(0.0, 0.0, 0.0, -1.0 / 15000.0, 8.0 / 3.0));
  
  std::vector<double> wavelengths;
  std::vector<double> solar_irradiance;
  std::vector<double> rayleigh_scattering;
  std::vector<double> mie_scattering;
  std::vector<double> mie_extinction;
  std::vector<double> absorption_extinction;
  std::vector<double> ground_albedo;
  
  for (int l = kLambdaMin; l <= kLambdaMax; l += 10) {
    double lambda = static_cast<double>(l) * 1e-3;  // micro-meters
    double mie = kMieAngstromBeta / kMieScaleHeight * pow(lambda, -kMieAngstromAlpha);
    
    wavelengths.push_back(l);
    if (default_params.use_constant_solar_spectrum) {
      solar_irradiance.push_back(kConstantSolarIrradiance);
    } else {
      solar_irradiance.push_back(kSolarIrradiance[(l - kLambdaMin) / 10]);
    }
    rayleigh_scattering.push_back(kRayleigh * pow(lambda, -4));
    mie_scattering.push_back(mie * kMieSingleScatteringAlbedo);
    mie_extinction.push_back(mie);
    absorption_extinction.push_back(default_params.use_ozone ?
        kMaxOzoneNumberDensity * kOzoneCrossSection[(l - kLambdaMin) / 10] : 0.0);
    ground_albedo.push_back(kGroundAlbedo);
  }
  
  model_.reset(new Model(
      wavelengths, solar_irradiance, kSunAngularRadius,
      kBottomRadius, kTopRadius, {rayleigh_layer}, rayleigh_scattering,
      {mie_layer}, mie_scattering, mie_extinction, kMiePhaseFunctionG,
      ozone_density, absorption_extinction, ground_albedo, max_sun_zenith_angle,
      kLengthUnitInMeters, default_params.luminance_mode == 2 ? 15 : 3,
      default_params.use_combined_textures, default_params.use_half_precision));
  
  model_->Init();
}

void AtmosphericRenderer::InitializeShaders() {
  // Create vertex shader
  vertex_shader_ = glCreateShader(GL_VERTEX_SHADER);
  const char* const vertex_shader_source = kVertexShader;
  glShaderSource(vertex_shader_, 1, &vertex_shader_source, NULL);
  glCompileShader(vertex_shader_);
  
  // Check vertex shader compilation
  GLint success;
  glGetShaderiv(vertex_shader_, GL_COMPILE_STATUS, &success);
  if (!success) {
    GLint log_length = 0;
    glGetShaderiv(vertex_shader_, GL_INFO_LOG_LENGTH, &log_length);
    if (log_length > 0) {
      std::vector<char> log(log_length);
      glGetShaderInfoLog(vertex_shader_, log_length, NULL, log.data());
      std::cerr << "INIT: Vertex shader compilation error:\n" << log.data() << std::endl;
    }
    throw std::runtime_error("Vertex shader compilation failed during initialization");
  }
  std::cerr << "INIT: Vertex shader compiled successfully." << std::endl;
  
  // Create fragment shader with appropriate defines
  RenderingParameters default_params;
  const std::string fragment_shader_str =
      "#version 330\n" +
      std::string(default_params.luminance_mode != 0 ? "#define USE_LUMINANCE\n" : "") +
      "const float kLengthUnitInMeters = " +
      std::to_string(kLengthUnitInMeters) + ";\n" +
      demo_glsl;
  const char* fragment_shader_source = fragment_shader_str.c_str();
  fragment_shader_ = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader_, 1, &fragment_shader_source, NULL);
  glCompileShader(fragment_shader_);
  
  // Check fragment shader compilation
  glGetShaderiv(fragment_shader_, GL_COMPILE_STATUS, &success);
  if (!success) {
    GLint log_length = 0;
    glGetShaderiv(fragment_shader_, GL_INFO_LOG_LENGTH, &log_length);
    if (log_length > 0) {
      std::vector<char> log(log_length);
      glGetShaderInfoLog(fragment_shader_, log_length, NULL, log.data());
      std::cerr << "INIT: Fragment shader compilation error:\n" << log.data() << std::endl;
    }
    throw std::runtime_error("Fragment shader compilation failed during initialization");
  }
  std::cerr << "INIT: Fragment shader compiled successfully." << std::endl;
  
  // Link program
  program_ = glCreateProgram();
  glAttachShader(program_, vertex_shader_);
  glAttachShader(program_, fragment_shader_);
  glAttachShader(program_, model_->shader());
  glLinkProgram(program_);
  
  // Check program linking
  glGetProgramiv(program_, GL_LINK_STATUS, &success);
  if (!success) {
    GLint log_length = 0;
    glGetProgramiv(program_, GL_INFO_LOG_LENGTH, &log_length);
    if (log_length > 0) {
      std::vector<char> log(log_length);
      glGetProgramInfoLog(program_, log_length, NULL, log.data());
      std::cerr << "INIT: Program linking error:\n" << log.data() << std::endl;
    }
    throw std::runtime_error("Program linking failed during initialization");
  }
  std::cerr << "INIT: Program linked successfully (ID: " << program_ << ")." << std::endl;
  
  glDetachShader(program_, vertex_shader_);
  glDetachShader(program_, fragment_shader_);
  glDetachShader(program_, model_->shader());
  
  // Set uniforms that don't change per frame
  glUseProgram(program_);
  model_->SetProgramUniforms(program_, 0, 1, 2, 3);
  
  // Earth center position
  constexpr double kBottomRadius = 6360000.0;
  glUniform3f(glGetUniformLocation(program_, "earth_center"),
      0.0, 0.0, -kBottomRadius / kLengthUnitInMeters);
  
  // Sun size
  glUniform2f(glGetUniformLocation(program_, "sun_size"),
      tan(kSunAngularRadius), cos(kSunAngularRadius));
}

void AtmosphericRenderer::Render(const RenderingParameters& params, 
                                  int viewport_width, int viewport_height) {
  if (!model_) {
    std::cerr << "ERROR: Renderer not initialized (model is null)" << std::endl;
    throw std::runtime_error("Renderer not initialized");
  }
  
  if (program_ == 0) {
    std::cerr << "ERROR: Shader program is invalid (0)" << std::endl;
    throw std::runtime_error("Shader program not initialized");
  }
  
  // Check if atmospheric parameters have changed and rebuild model if needed
  if (NeedsModelRebuild(params)) {
    std::cerr << "Atmospheric parameters changed, triggering rebuild..." << std::endl;
    RebuildModel(params);
    
    // Verify model and program are still valid after rebuild
    if (!model_) {
      std::cerr << "CRITICAL: Model is null after rebuild!" << std::endl;
      throw std::runtime_error("Model became null after rebuild");
    }
    if (program_ == 0) {
      std::cerr << "CRITICAL: Program is 0 after rebuild!" << std::endl;
      throw std::runtime_error("Program became invalid after rebuild");
    }
    std::cerr << "Model and program validated after rebuild. Continuing render..." << std::endl;
  }
  
  SetupViewMatrices(params, viewport_width, viewport_height);
  UpdateShaderUniforms(params);
  DrawScene();
  
  // Debug logging every 60 frames
  debug_frame_count_++;
  if (debug_frame_count_ % 60 == 0) {
    std::cerr << "Rendering frame " << debug_frame_count_ 
              << " (model=" << (model_ ? "valid" : "null") 
              << ", program=" << program_ << ")" << std::endl;
  }
}

bool AtmosphericRenderer::NeedsModelRebuild(const RenderingParameters& params) const {
  const double epsilon = 1e-6;
  return (std::abs(params.mie_phase_function_g - cached_mie_phase_g_) > epsilon ||
          std::abs(params.mie_scattering_scale - cached_mie_scale_) > epsilon ||
          std::abs(params.rayleigh_scattering_scale - cached_rayleigh_scale_) > epsilon ||
          std::abs(params.ground_albedo - cached_ground_albedo_) > epsilon);
}

void AtmosphericRenderer::RebuildModel(const RenderingParameters& params) {
  std::cerr << "\n=== MODEL REBUILD STARTED ===" << std::endl;
  std::cerr << "Phase G: " << params.mie_phase_function_g << std::endl;
  std::cerr << "Mie scale: " << params.mie_scattering_scale << std::endl;
  std::cerr << "Rayleigh scale: " << params.rayleigh_scattering_scale << std::endl;
  std::cerr << "Ground albedo: " << params.ground_albedo << std::endl;
  
  // Validate parameters before rebuilding
  if (params.mie_scattering_scale <= 0.0 || params.rayleigh_scattering_scale <= 0.0) {
    std::cerr << "ERROR: Invalid parameters (negative or zero scales)" << std::endl;
    return;
  }
  
  std::unique_ptr<Model> new_model;
  
  try {
    std::cerr << "Creating new model..." << std::endl;
    // Solar spectral irradiance data
    constexpr int kLambdaMin = 360;
    constexpr int kLambdaMax = 830;
  constexpr double kSolarIrradiance[48] = {
    1.11776, 1.14259, 1.01249, 1.14716, 1.72765, 1.73054, 1.6887, 1.61253,
    1.91198, 2.03474, 2.02042, 2.02212, 1.93377, 1.95809, 1.91686, 1.8298,
    1.8685, 1.8931, 1.85149, 1.8504, 1.8341, 1.8345, 1.8147, 1.78158, 1.7533,
    1.6965, 1.68194, 1.64654, 1.6048, 1.52143, 1.55622, 1.5113, 1.474, 1.4482,
    1.41018, 1.36775, 1.34188, 1.31429, 1.28303, 1.26758, 1.2367, 1.2082,
    1.18737, 1.14683, 1.12362, 1.1058, 1.07124, 1.04992
  };
  constexpr double kOzoneCrossSection[48] = {
    1.18e-27, 2.182e-28, 2.818e-28, 6.636e-28, 1.527e-27, 2.763e-27, 5.52e-27,
    8.451e-27, 1.582e-26, 2.316e-26, 3.669e-26, 4.924e-26, 7.752e-26, 9.016e-26,
    1.48e-25, 1.602e-25, 2.139e-25, 2.755e-25, 3.091e-25, 3.5e-25, 4.266e-25,
    4.672e-25, 4.398e-25, 4.701e-25, 5.019e-25, 4.305e-25, 3.74e-25, 3.215e-25,
    2.662e-25, 2.238e-25, 1.852e-25, 1.473e-25, 1.209e-25, 9.423e-26, 7.455e-26,
    6.566e-26, 5.105e-26, 4.15e-26, 4.228e-26, 3.237e-26, 2.451e-26, 2.801e-26,
    2.534e-26, 1.624e-26, 1.465e-26, 2.078e-26, 1.383e-26, 7.105e-27
  };
  
  constexpr double kDobsonUnit = 2.687e20;
  constexpr double kMaxOzoneNumberDensity = 300.0 * kDobsonUnit / 15000.0;
  constexpr double kConstantSolarIrradiance = 1.5;
  constexpr double kBottomRadius = 6360000.0;
  constexpr double kTopRadius = 6420000.0;
  constexpr double kRayleigh = 1.24062e-6;
  constexpr double kMieAngstromAlpha = 0.0;
  constexpr double kMieAngstromBeta = 5.328e-3;
  constexpr double kMieSingleScatteringAlbedo = 0.9;
  
  // Use user-specified parameters from sliders
  const double mie_phase_g = params.mie_phase_function_g;
  const double rayleigh_scale_height = params.rayleigh_scale_height;
  const double mie_scale_height = params.mie_scale_height;
  const double ground_albedo_value = params.ground_albedo;
  
  const double max_sun_zenith_angle =
      (params.use_half_precision ? 102.0 : 120.0) / 180.0 * kPi;
  
  DensityProfileLayer rayleigh_layer(0.0, 1.0, -1.0 / rayleigh_scale_height, 0.0, 0.0);
  DensityProfileLayer mie_layer(0.0, 1.0, -1.0 / mie_scale_height, 0.0, 0.0);
  
  std::vector<DensityProfileLayer> ozone_density;
  ozone_density.push_back(
      DensityProfileLayer(25000.0, 0.0, 0.0, 1.0 / 15000.0, -2.0 / 3.0));
  ozone_density.push_back(
      DensityProfileLayer(0.0, 0.0, 0.0, -1.0 / 15000.0, 8.0 / 3.0));
  
  std::vector<double> wavelengths;
  std::vector<double> solar_irradiance;
  std::vector<double> rayleigh_scattering;
  std::vector<double> mie_scattering;
  std::vector<double> mie_extinction;
  std::vector<double> absorption_extinction;
  std::vector<double> ground_albedo;
  
  for (int l = kLambdaMin; l <= kLambdaMax; l += 10) {
    double lambda = static_cast<double>(l) * 1e-3;
    double mie = kMieAngstromBeta / mie_scale_height * pow(lambda, -kMieAngstromAlpha);
    
    wavelengths.push_back(l);
    if (params.use_constant_solar_spectrum) {
      solar_irradiance.push_back(kConstantSolarIrradiance);
    } else {
      solar_irradiance.push_back(kSolarIrradiance[(l - kLambdaMin) / 10]);
    }
    // Apply user scaling factors
    rayleigh_scattering.push_back(kRayleigh * pow(lambda, -4) * params.rayleigh_scattering_scale);
    mie_scattering.push_back(mie * kMieSingleScatteringAlbedo * params.mie_scattering_scale);
    mie_extinction.push_back(mie * params.mie_scattering_scale);
    absorption_extinction.push_back(params.use_ozone ?
        kMaxOzoneNumberDensity * kOzoneCrossSection[(l - kLambdaMin) / 10] : 0.0);
    ground_albedo.push_back(ground_albedo_value);
  }
  
    // Create new model with updated parameters
    new_model.reset(new Model(
        wavelengths, solar_irradiance, kSunAngularRadius,
        kBottomRadius, kTopRadius, {rayleigh_layer}, rayleigh_scattering,
        {mie_layer}, mie_scattering, mie_extinction, mie_phase_g,
        ozone_density, absorption_extinction, ground_albedo, max_sun_zenith_angle,
        kLengthUnitInMeters, params.luminance_mode == 2 ? 15 : 3,
        params.use_combined_textures, params.use_half_precision));
    
    std::cerr << "Initializing new model (this takes 2-5 seconds)..." << std::endl;
    new_model->Init();
    std::cerr << "Model initialization complete." << std::endl;
    
    // Log old model info before swap
    std::cerr << "OLD model shader ID: " << (model_ ? model_->shader() : 0) << std::endl;
    std::cerr << "NEW model shader ID: " << new_model->shader() << std::endl;
    
    // Store old shader IDs
    GLuint old_fragment_shader = fragment_shader_;
    GLuint old_program = program_;
    std::cerr << "OLD program ID: " << old_program << std::endl;
    
    // Create new fragment shader and program with new model (don't swap model yet)
    const std::string fragment_shader_str =
        "#version 330\n" +
        std::string(params.luminance_mode != 0 ? "#define USE_LUMINANCE\n" : "") +
        "const float kLengthUnitInMeters = " +
        std::to_string(kLengthUnitInMeters) + ";\n" +
        demo_glsl;
    const char* fragment_shader_source = fragment_shader_str.c_str();
    fragment_shader_ = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader_, 1, &fragment_shader_source, NULL);
    glCompileShader(fragment_shader_);
    
    // Check fragment shader compilation
    GLint success;
    glGetShaderiv(fragment_shader_, GL_COMPILE_STATUS, &success);
    if (!success) {
      // Get error log
      GLint log_length = 0;
      glGetShaderiv(fragment_shader_, GL_INFO_LOG_LENGTH, &log_length);
      if (log_length > 0) {
        std::vector<char> log(log_length);
        glGetShaderInfoLog(fragment_shader_, log_length, NULL, log.data());
        std::cerr << "Fragment shader compilation error:\n" << log.data() << std::endl;
      }
      // Compilation failed, restore old shaders
      glDeleteShader(fragment_shader_);
      fragment_shader_ = old_fragment_shader;
      program_ = old_program;
      throw std::runtime_error("Fragment shader compilation failed during model rebuild");
    }
    std::cerr << "Fragment shader compiled successfully." << std::endl;
    
    // Link program with new model's shader
    program_ = glCreateProgram();
    std::cerr << "NEW program ID created: " << program_ << std::endl;
    glAttachShader(program_, vertex_shader_);
    glAttachShader(program_, fragment_shader_);
    glAttachShader(program_, new_model->shader());
    glLinkProgram(program_);
    
    // Check program linking
    std::cerr << "Linking shader program..." << std::endl;
    glGetProgramiv(program_, GL_LINK_STATUS, &success);
    if (!success) {
      // Get error log
      GLint log_length = 0;
      glGetProgramiv(program_, GL_INFO_LOG_LENGTH, &log_length);
      if (log_length > 0) {
        std::vector<char> log(log_length);
        glGetProgramInfoLog(program_, log_length, NULL, log.data());
        std::cerr << "Program linking error:\n" << log.data() << std::endl;
      }
      // Linking failed, restore old shaders
      glDeleteProgram(program_);
      glDeleteShader(fragment_shader_);
      fragment_shader_ = old_fragment_shader;
      program_ = old_program;
      throw std::runtime_error("Shader program linking failed during model rebuild");
    }
    std::cerr << "Program linked successfully." << std::endl;
    
    glDetachShader(program_, vertex_shader_);
    glDetachShader(program_, fragment_shader_);
    glDetachShader(program_, new_model->shader());
    
    // Set uniforms that don't change per frame
    glUseProgram(program_);
    new_model->SetProgramUniforms(program_, 0, 1, 2, 3);
    
    // Earth center position (using constant from earlier in function)
    glUniform3f(glGetUniformLocation(program_, "earth_center"),
        0.0, 0.0, -kBottomRadius / kLengthUnitInMeters);
    
    // Sun size
    glUniform2f(glGetUniformLocation(program_, "sun_size"),
        tan(kSunAngularRadius), cos(kSunAngularRadius));
    
    std::cerr << "Setting up uniforms..." << std::endl;
    
    // Success! Everything worked, now swap in the new model
    std::cerr << "About to swap model..." << std::endl;
    model_ = std::move(new_model);
    std::cerr << "Model swapped successfully." << std::endl;
    
    // Verify the swap worked
    if (!model_) {
      std::cerr << "CRITICAL ERROR: model_ is null after swap!" << std::endl;
      throw std::runtime_error("Model swap failed");
    }
    std::cerr << "Model verified non-null after swap." << std::endl;
    std::cerr << "Current model shader ID after swap: " << model_->shader() << std::endl;
    
    // CRITICAL: Rebind textures immediately after swap to ensure they're set
    std::cerr << "Rebinding textures with new model..." << std::endl;
    glUseProgram(program_);
    model_->SetProgramUniforms(program_, 0, 1, 2, 3);
    
    // Verify what textures are actually bound after SetProgramUniforms
    GLint tex0, tex1, tex2;
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &tex0);
    glActiveTexture(GL_TEXTURE1);
    glGetIntegerv(GL_TEXTURE_BINDING_3D, &tex1);
    glActiveTexture(GL_TEXTURE2);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &tex2);
    
    // Check if these are actually valid texture objects
    GLboolean tex0_valid = glIsTexture(tex0);
    GLboolean tex1_valid = glIsTexture(tex1);
    GLboolean tex2_valid = glIsTexture(tex2);
    
    std::cerr << "Immediately after rebind - Bound textures: Unit0=" << tex0 << " (valid=" << (tex0_valid ? "yes" : "NO") 
              << "), Unit1=" << tex1 << " (valid=" << (tex1_valid ? "yes" : "NO")
              << "), Unit2=" << tex2 << " (valid=" << (tex2_valid ? "yes" : "NO") << ")" << std::endl;
    std::cerr << "Textures rebound after model swap." << std::endl;
    
    // Delete old shaders now that new ones are working
    if (old_fragment_shader != 0) {
      glDeleteShader(old_fragment_shader);
      std::cerr << "Old fragment shader deleted." << std::endl;
    }
    if (old_program != 0) {
      glDeleteProgram(old_program);
      std::cerr << "Old program deleted." << std::endl;
    }
    
    // Update cached values
    cached_mie_phase_g_ = params.mie_phase_function_g;
    cached_mie_scale_ = params.mie_scattering_scale;
    cached_rayleigh_scale_ = params.rayleigh_scattering_scale;
    cached_ground_albedo_ = params.ground_albedo;
    
    // CRITICAL: Invalidate cached viewport/FOV to force view matrix recalculation
    cached_viewport_width_ = 0;
    cached_viewport_height_ = 0;
    cached_fov_degrees_ = 0.0;
    std::cerr << "Cached viewport/FOV invalidated to force matrix recalculation." << std::endl;
    
    std::cerr << "=== MODEL REBUILD COMPLETE ===\n" << std::endl;
  } catch (const std::exception& e) {
    // Rebuild failed, keep old model and update cached values
    // so we don't keep trying to rebuild with bad params
    std::cerr << "ERROR: Rebuild failed - " << e.what() << std::endl;
    std::cerr << "Keeping old model active." << std::endl;
    cached_mie_phase_g_ = params.mie_phase_function_g;
    cached_mie_scale_ = params.mie_scattering_scale;
    cached_rayleigh_scale_ = params.rayleigh_scattering_scale;
    cached_ground_albedo_ = params.ground_albedo;
    std::cerr << "=== MODEL REBUILD FAILED ===\n" << std::endl;
  }
}

void AtmosphericRenderer::SetupViewMatrices(const RenderingParameters& params,
                                             int viewport_width, int viewport_height) {
  glUseProgram(program_);
  
  // Always set viewport size uniform (used by latlong mode)
  glUniform2f(glGetUniformLocation(program_, "viewport_size"), 
              static_cast<float>(viewport_width), 
              static_cast<float>(viewport_height));
  
  // Update view_from_clip matrix if viewport size or FOV changed
  // (Only needed for perspective mode, but set anyway for simplicity)
  if (viewport_width != cached_viewport_width_ || 
      viewport_height != cached_viewport_height_ ||
      params.view_fov_degrees != cached_fov_degrees_) {
    cached_viewport_width_ = viewport_width;
    cached_viewport_height_ = viewport_height;
    cached_fov_degrees_ = params.view_fov_degrees;
    
    if (params.render_mode == 0) {
      // Perspective mode: compute view_from_clip matrix
      double aspect_ratio = static_cast<double>(viewport_width) / viewport_height;
      double fov_rad = params.view_fov_degrees * kPi / 180.0;
      double tan_fov = tan(fov_rad / 2.0);
      
      float view_from_clip[16] = {
        static_cast<float>(tan_fov * aspect_ratio), 0.0, 0.0, 0.0,
        0.0, static_cast<float>(tan_fov), 0.0, 0.0,
        0.0, 0.0, 0.0, -1.0,
        0.0, 0.0, 1.0, 1.0
      };
      
      glUniformMatrix4fv(glGetUniformLocation(program_, "view_from_clip"),
          1, true, view_from_clip);
    }
    // Latlong mode doesn't need view_from_clip matrix
  }
}

void AtmosphericRenderer::UpdateShaderUniforms(const RenderingParameters& params) {
  if (program_ == 0) {
    std::cerr << "ERROR in UpdateShaderUniforms: program_ is 0!" << std::endl;
    return;
  }
  
  if (!model_) {
    std::cerr << "ERROR in UpdateShaderUniforms: model_ is null!" << std::endl;
    return;
  }
  
  glUseProgram(program_);
  GLenum err = glGetError();
  if (err != GL_NO_ERROR) {
    std::cerr << "OpenGL error in UpdateShaderUniforms after glUseProgram: " << err << std::endl;
  }
  
  // CRITICAL: Rebind model's texture uniforms every frame
  // The model contains precomputed atmospheric scattering textures that must be bound
  model_->SetProgramUniforms(program_, 0, 1, 2, 3);
  err = glGetError();
  if (err != GL_NO_ERROR) {
    std::cerr << "OpenGL error after SetProgramUniforms: " << err << std::endl;
  }
  
  // Verify texture bindings on every 60th frame
  static int texture_check_counter = 0;
  texture_check_counter++;
  if (texture_check_counter % 60 == 0) {
    GLint tex0, tex1, tex2;
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &tex0);
    glActiveTexture(GL_TEXTURE1);
    glGetIntegerv(GL_TEXTURE_BINDING_3D, &tex1);
    glActiveTexture(GL_TEXTURE2);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &tex2);
    std::cerr << "Bound textures: Unit0=" << tex0 << ", Unit1=" << tex1 << ", Unit2=" << tex2 << std::endl;
  }
  
  // Convert GUI coordinate system to renderer's coordinate system
  // GUI: X=right, Y=up, Z=forward
  // Renderer: X=right, Y=forward, Z=up (standard OpenGL with Z up for Earth)
  double pos_x = params.camera_position_x;  // right/left stays X
  double pos_y = params.camera_position_z;  // forward/back becomes Y
  double pos_z = params.camera_position_y;  // up/down becomes Z
  
  // Build camera rotation matrix from Euler angles
  // Coordinate system: Renderer uses Z-up, with camera naturally looking down +Y axis
  // Rotation order: Yaw (around Z) -> Pitch (around local X) -> Roll (around local Y/view)
  const double kPi = 3.14159265358979323846;
  
  // Convert to radians - pitch is negated for intuitive up=positive, down=negative
  double pitch_rad = -params.camera_pitch_degrees * kPi / 180.0;
  double yaw_rad = params.camera_yaw_degrees * kPi / 180.0;
  double roll_rad = params.camera_roll_degrees * kPi / 180.0;
  
  // Precompute sin/cos
  double cp = cos(pitch_rad), sp = sin(pitch_rad);
  double cy = cos(yaw_rad), sy = sin(yaw_rad);
  double cr = cos(roll_rad), sr = sin(roll_rad);
  
  // YXZ Euler rotation matrix (guarantees orthonormality)
  // Right vector (X axis in camera space)
  float ux[3] = {
    static_cast<float>(cy * cr - sy * sp * sr),
    static_cast<float>(cp * sr),
    static_cast<float>(-sy * cr - cy * sp * sr)
  };
  
  // Forward vector (Y axis in camera space - view direction)
  float uy[3] = {
    static_cast<float>(sy * sp * cr + cy * sr),
    static_cast<float>(cp * cr),
    static_cast<float>(cy * sp * cr - sy * sr)
  };
  
  // Up vector (Z axis in camera space)
  float uz[3] = {
    static_cast<float>(sy * cp),
    static_cast<float>(-sp),
    static_cast<float>(cy * cp)
  };
  
  // Camera position in render units
  float cam_x = static_cast<float>(pos_x / kLengthUnitInMeters);
  float cam_y = static_cast<float>(pos_y / kLengthUnitInMeters);
  float cam_z = static_cast<float>(pos_z / kLengthUnitInMeters);
  
  // Model-from-view transform matrix (camera orientation + position)
  float model_from_view[16] = {
    ux[0], uy[0], uz[0], cam_x,
    ux[1], uy[1], uz[1], cam_y,
    ux[2], uy[2], uz[2], cam_z,
    0.0, 0.0, 0.0, 1.0
  };
  
  glUniformMatrix4fv(glGetUniformLocation(program_, "model_from_view"),
      1, true, model_from_view);
  
  glUniform3f(glGetUniformLocation(program_, "camera"), cam_x, cam_y, cam_z);
  
  // Sun direction
  glUniform3f(glGetUniformLocation(program_, "sun_direction"),
      cos(params.sun_azimuth_angle_radians) * sin(params.sun_zenith_angle_radians),
      sin(params.sun_azimuth_angle_radians) * sin(params.sun_zenith_angle_radians),
      cos(params.sun_zenith_angle_radians));
  
  // Exposure (adjust for luminance mode)
  float exposure = params.luminance_mode != 0 ? params.exposure * 1e-5 : params.exposure;
  glUniform1f(glGetUniformLocation(program_, "exposure"), exposure);
  
  // Debug: log critical uniforms every 120 frames
  static int uniform_log_counter = 0;
  uniform_log_counter++;
  if (uniform_log_counter % 120 == 0 || uniform_log_counter == 1) {
    std::cerr << "Uniforms - Camera: (" << pos_x << ", " << pos_y << ", " << pos_z 
              << "), Sun zenith: " << params.sun_zenith_angle_radians
              << ", Exposure: " << exposure << std::endl;
  }
  
  // White balance (if enabled, would need to recompute - for now using default)
  glUniform3f(glGetUniformLocation(program_, "white_point"), 1.0, 1.0, 1.0);
  
  // Scene object parameters
  glUniform1i(glGetUniformLocation(program_, "show_scene_object"), params.show_scene_object ? 1 : 0);
  glUniform1i(glGetUniformLocation(program_, "scene_object_shape"), params.scene_object_shape);
  
  // Render mode
  glUniform1i(glGetUniformLocation(program_, "render_mode"), params.render_mode);
}

void AtmosphericRenderer::DrawScene() {
  if (program_ == 0) {
    std::cerr << "ERROR in DrawScene: program_ is 0!" << std::endl;
    return;
  }
  
  // Check framebuffer status
  GLenum fb_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  if (fb_status != GL_FRAMEBUFFER_COMPLETE) {
    std::cerr << "Framebuffer not complete! Status: " << fb_status << std::endl;
  }
  
  glUseProgram(program_);
  GLenum err = glGetError();
  if (err != GL_NO_ERROR) {
    std::cerr << "OpenGL error after glUseProgram: " << err << std::endl;
  }
  
  // Verify program is actually in use
  GLint current_program = 0;
  glGetIntegerv(GL_CURRENT_PROGRAM, &current_program);
  if (current_program != (GLint)program_) {
    std::cerr << "ERROR: Program not active! Expected " << program_ << ", got " << current_program << std::endl;
  }
  
  glBindVertexArray(full_screen_quad_vao_);
  err = glGetError();
  if (err != GL_NO_ERROR) {
    std::cerr << "OpenGL error after glBindVertexArray: " << err << std::endl;
  }
  
  glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
  err = glGetError();
  if (err != GL_NO_ERROR) {
    std::cerr << "OpenGL error after glDrawArrays: " << err << std::endl;
  }
  
  // Sample center pixel every 60 frames to verify output
  static int pixel_check_counter = 0;
  pixel_check_counter++;
  if (pixel_check_counter % 60 == 0) {
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    int center_x = viewport[2] / 2;
    int center_y = viewport[3] / 2;
    float pixel[4] = {0, 0, 0, 0};
    glReadPixels(center_x, center_y, 1, 1, GL_RGBA, GL_FLOAT, pixel);
    std::cerr << "Center pixel RGB: (" << pixel[0] << ", " << pixel[1] << ", " << pixel[2] << ")" << std::endl;
  }
  
  glBindVertexArray(0);
}

}  // namespace demo
}  // namespace atmosphere
