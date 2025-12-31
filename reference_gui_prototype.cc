/**
 * Beautiful Modern GUI Prototype Implementation
 * Focus on design aesthetics with MattePaint branding
 */

#include "atmosphere/demo/gui_prototype.h"

#include <glad/glad.h>
#include <GL/freeglut.h>
#include <imgui.h>
#include <backends/imgui_impl_glut.h>
#include <backends/imgui_impl_opengl3.h>

#include <map>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace atmosphere {
namespace demo {

namespace {
static std::map<int, GUIPrototype*> INSTANCES;
}

GUIPrototype::GUIPrototype(int width, int height) :
    show_demo_window_(false),
    camera_pos_x_(0.0f),
    camera_pos_y_(1000.0f),
    camera_pos_z_(9000.0f),
    camera_rot_pitch_(90.0f),
    camera_rot_yaw_(180.0f),
    camera_rot_roll_(0.0f),
    camera_fov_(50.0f),
    sun_zenith_(75.0f),
    sun_azimuth_(165.0f),
    sun_intensity_(1.0f),
    exposure_(10.0f),
    use_ozone_(true),
    use_combined_textures_(true),
    luminance_mode_(0),
    white_balance_(false),
    render_mode_(0),
    mie_phase_g_(0.8f),
    mie_density_(1.0f),
    rayleigh_density_(1.0f),
    mie_height_(1200.0f),
    rayleigh_height_(8000.0f),
    ground_albedo_(0.1f),
    show_scene_object_(true),
    scene_object_shape_(0),
    last_committed_mie_phase_g_(0.8f),
    last_committed_mie_density_(1.0f),
    last_committed_rayleigh_density_(1.0f),
    last_committed_ground_albedo_(0.1f),
    renderer_(nullptr),
    framebuffer_(0),
    viewport_texture_(0),
    depth_renderbuffer_(0),
    viewport_width_(1280),
    viewport_height_(720) {
  
  glutInitWindowSize(width, height);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
  window_id_ = glutCreateWindow("Atmospheric Scattering | Professional Edition");
  INSTANCES[window_id_] = this;
  
  if (!gladLoadGL()) {
    throw std::runtime_error("GLAD initialization failed");
  }

  glutDisplayFunc([]() { INSTANCES[glutGetWindow()]->HandleRedisplayEvent(); });
  glutReshapeFunc([](int w, int h) { INSTANCES[glutGetWindow()]->HandleReshapeEvent(w, h); });
  glutKeyboardFunc([](unsigned char key, int x, int y) {
    if (key == 27) glutLeaveMainLoop(); // ESC
  });

  // Initialize atmospheric renderer
  renderer_.reset(new AtmosphericRenderer());
  renderer_->Initialize();
  
  // Setup framebuffer for rendering
  SetupFramebuffer(viewport_width_, viewport_height_);

  InitImGui();
  SetupMattePaintStyle();
}

GUIPrototype::~GUIPrototype() {
  CleanupFramebuffer();
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGLUT_Shutdown();
  ImGui::DestroyContext();
  INSTANCES.erase(window_id_);
}

void GUIPrototype::InitImGui() {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.IniFilename = nullptr;
  
  // Configure bold modern font
  ImFontConfig font_config;
  font_config.OversampleH = 4;  // Maximum oversampling for bold appearance
  font_config.OversampleV = 4;  // Maximum vertical oversampling
  font_config.PixelSnapH = false;  // Smoother rendering for bold look
  font_config.RasterizerMultiply = 1.3f;  // Increase font weight/thickness
  io.Fonts->AddFontDefault(&font_config);
  io.FontGlobalScale = 1.2f;  // Larger for bold appearance
  
  ImGui_ImplGLUT_Init();
  ImGui_ImplGLUT_InstallFuncs();
  ImGui_ImplOpenGL3_Init("#version 330");
}

void GUIPrototype::SetupMattePaintStyle() {
  ImGuiStyle& style = ImGui::GetStyle();
  ImVec4* colors = style.Colors;
  
  // MattePaint.com Exact Color Scheme - Deep Black Theme
  // MattePaint Orange: #ec871b (RGB: 236, 135, 27)
  const ImVec4 mattepaint_orange = ImVec4(0.925f, 0.529f, 0.106f, 1.00f);
  const ImVec4 true_black = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);
  const ImVec4 near_black = ImVec4(0.02f, 0.02f, 0.02f, 1.00f);
  const ImVec4 dark_gray = ImVec4(0.05f, 0.05f, 0.05f, 1.00f);
  const ImVec4 frame_gray = ImVec4(0.08f, 0.08f, 0.08f, 1.00f);
  
  colors[ImGuiCol_Text]                   = ImVec4(0.95f, 0.95f, 0.97f, 1.00f);
  colors[ImGuiCol_TextDisabled]           = ImVec4(0.50f, 0.50f, 0.52f, 1.00f);
  colors[ImGuiCol_WindowBg]               = near_black;
  colors[ImGuiCol_ChildBg]                = dark_gray;
  colors[ImGuiCol_PopupBg]                = near_black;
  colors[ImGuiCol_Border]                 = ImVec4(0.15f, 0.15f, 0.15f, 0.60f);
  colors[ImGuiCol_BorderShadow]           = true_black;
  colors[ImGuiCol_FrameBg]                = frame_gray;
  colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.12f + mattepaint_orange.x * 0.05f, 0.12f + mattepaint_orange.y * 0.05f, 0.12f + mattepaint_orange.z * 0.05f, 1.00f);
  colors[ImGuiCol_FrameBgActive]          = ImVec4(0.15f + mattepaint_orange.x * 0.08f, 0.15f + mattepaint_orange.y * 0.08f, 0.15f + mattepaint_orange.z * 0.08f, 1.00f);
  colors[ImGuiCol_TitleBg]                = true_black;
  colors[ImGuiCol_TitleBgActive]          = near_black;
  colors[ImGuiCol_TitleBgCollapsed]       = true_black;
  colors[ImGuiCol_MenuBarBg]              = near_black;
  colors[ImGuiCol_ScrollbarBg]            = near_black;
  colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.20f, 0.20f, 0.20f, 1.00f);
  colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.30f, 0.30f, 0.30f, 1.00f);
  colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
  colors[ImGuiCol_CheckMark]              = mattepaint_orange;
  colors[ImGuiCol_SliderGrab]             = mattepaint_orange;
  colors[ImGuiCol_SliderGrabActive]       = ImVec4(1.00f, 0.60f, 0.20f, 1.00f);
  colors[ImGuiCol_Button]                 = mattepaint_orange;
  colors[ImGuiCol_ButtonHovered]          = ImVec4(0.941f, 0.620f, 0.278f, 1.00f);  // #f09e47
  colors[ImGuiCol_ButtonActive]           = ImVec4(1.00f, 0.60f, 0.20f, 1.00f);
  colors[ImGuiCol_Header]                 = ImVec4(mattepaint_orange.x, mattepaint_orange.y, mattepaint_orange.z, 0.70f);
  colors[ImGuiCol_HeaderHovered]          = mattepaint_orange;
  colors[ImGuiCol_HeaderActive]           = ImVec4(1.00f, 0.60f, 0.20f, 1.00f);
  colors[ImGuiCol_Separator]              = ImVec4(0.15f, 0.15f, 0.15f, 0.60f);
  colors[ImGuiCol_SeparatorHovered]       = ImVec4(mattepaint_orange.x, mattepaint_orange.y, mattepaint_orange.z, 0.78f);
  colors[ImGuiCol_SeparatorActive]        = mattepaint_orange;
  colors[ImGuiCol_ResizeGrip]             = ImVec4(mattepaint_orange.x, mattepaint_orange.y, mattepaint_orange.z, 0.30f);
  colors[ImGuiCol_ResizeGripHovered]      = ImVec4(mattepaint_orange.x, mattepaint_orange.y, mattepaint_orange.z, 0.67f);
  colors[ImGuiCol_ResizeGripActive]       = mattepaint_orange;
  colors[ImGuiCol_Tab]                    = dark_gray;
  colors[ImGuiCol_TabHovered]             = ImVec4(mattepaint_orange.x, mattepaint_orange.y, mattepaint_orange.z, 0.80f);
  colors[ImGuiCol_TabActive]              = ImVec4(mattepaint_orange.x, mattepaint_orange.y, mattepaint_orange.z, 0.65f);
  colors[ImGuiCol_TabUnfocused]           = near_black;
  colors[ImGuiCol_TabUnfocusedActive]     = dark_gray;
  colors[ImGuiCol_PlotLines]              = mattepaint_orange;
  colors[ImGuiCol_PlotLinesHovered]       = ImVec4(1.00f, 0.52f, 0.31f, 1.00f);
  colors[ImGuiCol_PlotHistogram]          = mattepaint_orange;
  colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(1.00f, 0.52f, 0.31f, 1.00f);
  colors[ImGuiCol_TextSelectedBg]         = ImVec4(mattepaint_orange.x, mattepaint_orange.y, mattepaint_orange.z, 0.35f);
  colors[ImGuiCol_DragDropTarget]         = mattepaint_orange;
  colors[ImGuiCol_NavHighlight]           = mattepaint_orange;
  
  // Polished modern styling
  style.WindowRounding = 8.0f;
  style.ChildRounding = 6.0f;
  style.FrameRounding = 5.0f;
  style.PopupRounding = 6.0f;
  style.ScrollbarRounding = 9.0f;
  style.GrabRounding = 5.0f;
  style.TabRounding = 6.0f;
  style.WindowPadding = ImVec2(14, 14);
  style.FramePadding = ImVec2(8, 6);
  style.ItemSpacing = ImVec2(12, 10);
  style.ItemInnerSpacing = ImVec2(10, 8);
  style.IndentSpacing = 24.0f;
  style.ScrollbarSize = 18.0f;
  style.GrabMinSize = 14.0f;
  style.WindowBorderSize = 1.0f;
  style.ChildBorderSize = 1.0f;
  style.PopupBorderSize = 1.0f;
  style.FrameBorderSize = 0.0f;
  style.TabBorderSize = 0.0f;
  style.ButtonTextAlign = ImVec2(0.5f, 0.5f);  // Center align button text
  style.WindowTitleAlign = ImVec2(0.5f, 0.5f);
}

void GUIPrototype::HandleRedisplayEvent() {
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGLUT_NewFrame();
  ImGui::NewFrame();
  
  RenderUI();
  
  ImGui::Render();
  glViewport(0, 0, glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);  // True black to match MattePaint.com
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  
  glutSwapBuffers();
  glutPostRedisplay();
}

void GUIPrototype::HandleReshapeEvent(int w, int h) {
  glViewport(0, 0, w, h);
}

void GUIPrototype::RenderUI() {
  // Get display size
  ImGuiIO& io = ImGui::GetIO();
  ImVec2 display_size = io.DisplaySize;
  
  // Menu bar
  ImGui::SetNextWindowPos(ImVec2(0, 0));
  ImGui::SetNextWindowSize(ImVec2(display_size.x, 40));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(20, 10));
  ImGui::Begin("MenuBar", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar);
  ImGui::PopStyleVar(2);
  RenderMenuBar();
  ImGui::End();
  
  // Left panel (controls)
  float left_width = 320.0f;
  ImGui::SetNextWindowPos(ImVec2(0, 40));
  ImGui::SetNextWindowSize(ImVec2(left_width, display_size.y - 40));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
  ImGui::PopStyleVar();
  RenderCameraPanel();
  ImGui::Separator();
  ImGui::Spacing();
  RenderEnvironmentPanel();
  ImGui::Separator();
  ImGui::Spacing();
  RenderAtmospherePanel();
  
  // Reset All button at bottom
  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();
  const ImVec4 mattepaint_orange = ImVec4(0.925f, 0.529f, 0.106f, 1.00f);
  ImGui::PushStyleColor(ImGuiCol_Button, mattepaint_orange);
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.941f, 0.620f, 0.278f, 1.00f));
  if (ImGui::Button("RESET ALL", ImVec2(-1, 40))) {
    ResetAllParameters();
  }
  ImGui::PopStyleColor(2);
  
  ImGui::End();
  
  // Right panel (settings & presets)
  float right_width = 280.0f;
  ImGui::SetNextWindowPos(ImVec2(display_size.x - right_width, 40));
  ImGui::SetNextWindowSize(ImVec2(right_width, display_size.y - 40));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
  ImGui::PopStyleVar();
  RenderRenderingPanel();
  ImGui::Separator();
  ImGui::Spacing();
  RenderPresetsPanel();
  ImGui::End();
  
  // Center viewport
  ImGui::SetNextWindowPos(ImVec2(left_width, 40));
  ImGui::SetNextWindowSize(ImVec2(display_size.x - left_width - right_width, display_size.y - 40));
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
  ImGui::Begin("Viewport", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
  ImGui::PopStyleVar(2);
  RenderViewportPanel();
  ImGui::End();
}

void GUIPrototype::RenderMenuBar() {
  const ImVec4 mattepaint_orange = ImVec4(0.925f, 0.529f, 0.106f, 1.00f);
  
  // Left side branding
  ImGui::PushStyleColor(ImGuiCol_Text, mattepaint_orange);
  ImGui::Text("●");
  ImGui::PopStyleColor();
  ImGui::SameLine();
  ImGui::Text("ATMOSPHERIC SCATTERING");
  
  // Calculate right side text width and position properly
  const char* right_text = "Professional Edition | MattePaint";
  float text_width = ImGui::CalcTextSize(right_text).x;
  ImGui::SameLine(ImGui::GetWindowWidth() - text_width - 30);  // 30px padding from edge
  ImGui::TextDisabled("Professional Edition");
  ImGui::SameLine();
  ImGui::PushStyleColor(ImGuiCol_Text, mattepaint_orange);
  ImGui::Text("| MattePaint");
  ImGui::PopStyleColor();
}

void GUIPrototype::RenderViewportPanel() {
  ImVec2 size = ImGui::GetContentRegionAvail();
  if (size.x > 0 && size.y > 0) {
    // Adjust size for latlong mode (needs 2:1 aspect ratio)
    int target_width = static_cast<int>(size.x);
    int target_height = static_cast<int>(size.y);
    
    if (render_mode_ == 1) {
      // Latlong mode: force 2:1 aspect ratio (wide)
      target_height = target_width / 2;
      // Clamp to available space
      if (target_height > static_cast<int>(size.y)) {
        target_height = static_cast<int>(size.y);
        target_width = target_height * 2;
      }
    }
    
    // Resize framebuffer if viewport size changed significantly
    if (std::abs(target_width - viewport_width_) > 10 || 
        std::abs(target_height - viewport_height_) > 10) {
      SetupFramebuffer(target_width, target_height);
    }
    
    // Render atmospheric scene to framebuffer
    if (renderer_ && renderer_->IsInitialized()) {
      // Bind framebuffer
      glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
      glViewport(0, 0, viewport_width_, viewport_height_);
      
      // Clear to black
      glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      
      // Get rendering parameters from GUI
      RenderingParameters params = GetRenderingParameters();
      
      // Render atmospheric scene
      renderer_->Render(params, viewport_width_, viewport_height_);
      
      // Unbind framebuffer
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    
    // Display rendered texture in ImGui viewport
    // For latlong mode, center the 2:1 image in the available space
    ImVec2 display_size = ImVec2(viewport_width_, viewport_height_);
    if (render_mode_ == 1) {
      // Center latlong image
      float offset_x = (size.x - display_size.x) * 0.5f;
      float offset_y = (size.y - display_size.y) * 0.5f;
      if (offset_x > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset_x);
      if (offset_y > 0) ImGui::SetCursorPosY(ImGui::GetCursorPosY() + offset_y);
    }
    ImGui::Image((void*)(intptr_t)viewport_texture_, display_size, ImVec2(0,1), ImVec2(1,0));
    
    // Handle mouse interactions directly in the viewport
    if (ImGui::IsItemHovered()) {
      ImGuiIO& io = ImGui::GetIO();
      
      // Check for mouse drag
      if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0.0f)) {
        ImVec2 drag_delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 0.0f);
        ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
        
        constexpr float kRotationScale = 0.2f;  // Degrees per pixel
        constexpr float kSunScale = 0.2f;       // Degrees per pixel
        
        if (io.KeyCtrl) {
          // Ctrl + Drag: Move sun
          sun_zenith_ -= drag_delta.y * kSunScale;
          sun_zenith_ = std::max(0.0f, std::min(180.0f, sun_zenith_));
          sun_azimuth_ += drag_delta.x * kSunScale;
          
          // Wrap azimuth to 0-360 range
          while (sun_azimuth_ < 0.0f) sun_azimuth_ += 360.0f;
          while (sun_azimuth_ >= 360.0f) sun_azimuth_ -= 360.0f;
        } else {
          // Drag: Rotate camera (no clamping for free rotation)
          camera_rot_pitch_ += drag_delta.y * kRotationScale;
          camera_rot_yaw_ += drag_delta.x * kRotationScale;
        }
      }
      
      // Handle mouse wheel
      if (io.MouseWheel != 0.0f) {
        const double kPi = 3.14159265358979323846;
        float pitch_rad = -camera_rot_pitch_ * kPi / 180.0f;
        float yaw_rad = camera_rot_yaw_ * kPi / 180.0f;
        
        // Calculate forward direction vector (in GUI coordinate system)
        float forward_x = sin(yaw_rad) * cos(pitch_rad);
        float forward_y = sin(pitch_rad);
        float forward_z = cos(yaw_rad) * cos(pitch_rad);
        
        // Calculate zoom step based on current distance from origin
        float distance = sqrt(camera_pos_x_ * camera_pos_x_ + 
                             camera_pos_y_ * camera_pos_y_ + 
                             camera_pos_z_ * camera_pos_z_);
        float zoom_step = std::max(100.0f, distance * 0.05f);  // 5% of distance or min 100m
        
        if (io.MouseWheel > 0) {
          // Zoom in: move forward
          camera_pos_x_ += forward_x * zoom_step;
          camera_pos_y_ += forward_y * zoom_step;
          camera_pos_z_ += forward_z * zoom_step;
        } else {
          // Zoom out: move backward
          camera_pos_x_ -= forward_x * zoom_step;
          camera_pos_y_ -= forward_y * zoom_step;
          camera_pos_z_ -= forward_z * zoom_step;
        }
      }
    }
  }
}

void GUIPrototype::RenderCameraPanel() {
  const ImVec4 mattepaint_orange = ImVec4(0.925f, 0.529f, 0.106f, 1.00f);
  ImGui::PushStyleColor(ImGuiCol_Text, mattepaint_orange);
  ImGui::Text("CAMERA CONTROLS");
  ImGui::PopStyleColor();
  ImGui::Separator();
  ImGui::Spacing();
  
  // Position section
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.85f, 0.87f, 1.00f));
  ImGui::Text("Position");
  ImGui::PopStyleColor();
  
  ImGui::Text("X"); ImGui::SameLine(45); ImGui::PushItemWidth(-100);
  ImGui::InputFloat("##pos_x", &camera_pos_x_, 0.0f, 0.0f, "%.1f m");
  ImGui::PopItemWidth(); ImGui::SameLine();
  if (ImGui::Button("Reset##px", ImVec2(60, 0))) camera_pos_x_ = 0.0f;
  
  ImGui::Text("Y"); ImGui::SameLine(45); ImGui::PushItemWidth(-100);
  ImGui::InputFloat("##pos_y", &camera_pos_y_, 0.0f, 0.0f, "%.1f m");
  ImGui::PopItemWidth(); ImGui::SameLine();
  if (ImGui::Button("Reset##py", ImVec2(60, 0))) camera_pos_y_ = 1000.0f;
  
  ImGui::Text("Z"); ImGui::SameLine(45); ImGui::PushItemWidth(-100);
  ImGui::InputFloat("##pos_z", &camera_pos_z_, 0.0f, 0.0f, "%.1f m");
  ImGui::PopItemWidth(); ImGui::SameLine();
  if (ImGui::Button("Reset##pz", ImVec2(60, 0))) camera_pos_z_ = 9000.0f;
  
  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();
  
  // Rotation section
  ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.85f, 0.87f, 1.00f));
  ImGui::Text("Rotation");
  ImGui::PopStyleColor();
  
  ImGui::Text("X"); ImGui::SameLine(45); ImGui::PushItemWidth(-100);
  ImGui::InputFloat("##rot_pitch", &camera_rot_pitch_, 0.0f, 0.0f, "%.1f°");
  ImGui::PopItemWidth(); ImGui::SameLine();
  if (ImGui::Button("Reset##rp", ImVec2(60, 0))) camera_rot_pitch_ = 90.0f;
  
  ImGui::Text("Y"); ImGui::SameLine(45); ImGui::PushItemWidth(-100);
  ImGui::InputFloat("##rot_yaw", &camera_rot_yaw_, 0.0f, 0.0f, "%.1f°");
  ImGui::PopItemWidth(); ImGui::SameLine();
  if (ImGui::Button("Reset##ry", ImVec2(60, 0))) camera_rot_yaw_ = 180.0f;
  
  ImGui::Text("Z"); ImGui::SameLine(45); ImGui::PushItemWidth(-100);
  ImGui::InputFloat("##rot_roll", &camera_rot_roll_, 0.0f, 0.0f, "%.1f°");
  ImGui::PopItemWidth(); ImGui::SameLine();
  if (ImGui::Button("Reset##rr", ImVec2(60, 0))) camera_rot_roll_ = 0.0f;
  
  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();
  
  ImGui::Text("FOV"); ImGui::SameLine(45); ImGui::PushItemWidth(-100);
  ImGui::InputFloat("##fov", &camera_fov_, 0.0f, 0.0f, "%.1f°");
  ImGui::PopItemWidth(); ImGui::SameLine();
  if (ImGui::Button("Reset##fov", ImVec2(60, 0))) camera_fov_ = 50.0f;
}

void GUIPrototype::RenderEnvironmentPanel() {
  const ImVec4 mattepaint_orange = ImVec4(0.925f, 0.529f, 0.106f, 1.00f);
  ImGui::PushStyleColor(ImGuiCol_Text, mattepaint_orange);
  ImGui::Text("SUN & ATMOSPHERE");
  ImGui::PopStyleColor();
  ImGui::Separator();
  ImGui::Spacing();
  
  ImGui::Text("Zenith"); ImGui::SameLine(90); ImGui::PushItemWidth(-100);
  ImGui::SliderFloat("##sunzenith", &sun_zenith_, 0.0f, 180.0f, "%.1f°");
  ImGui::PopItemWidth(); ImGui::SameLine();
  if (ImGui::Button("Reset##sunzen", ImVec2(60, 0))) sun_zenith_ = 75.0f;
  
  ImGui::Text("Azimuth"); ImGui::SameLine(90); ImGui::PushItemWidth(-100);
  ImGui::SliderFloat("##sunazimuth", &sun_azimuth_, -180.0f, 180.0f, "%.1f°");
  ImGui::PopItemWidth(); ImGui::SameLine();
  if (ImGui::Button("Reset##sunazi", ImVec2(60, 0))) sun_azimuth_ = 165.0f;
  
  ImGui::Text("Intensity"); ImGui::SameLine(90); ImGui::PushItemWidth(-100);
  ImGui::SliderFloat("##sunintensity", &sun_intensity_, 0.0f, 2.0f, "%.2f");
  ImGui::PopItemWidth(); ImGui::SameLine();
  if (ImGui::Button("Reset##sunint", ImVec2(60, 0))) sun_intensity_ = 1.0f;
}

void GUIPrototype::RenderAtmospherePanel() {
  const ImVec4 mattepaint_orange = ImVec4(0.925f, 0.529f, 0.106f, 1.00f);
  ImGui::PushStyleColor(ImGuiCol_Text, mattepaint_orange);
  ImGui::Text("ATMOSPHERE COMPOSITION");
  ImGui::PopStyleColor();
  ImGui::Separator();
  ImGui::Spacing();
  
  // Phase Function G - controls scattering direction
  ImGui::Text("Phase G"); ImGui::SameLine(90); ImGui::PushItemWidth(-100);
  ImGui::SliderFloat("##phaseg", &mie_phase_g_, -1.0f, 1.0f, "%.2f");
  if (ImGui::IsItemActive()) {
    ImGui::SetTooltip("Release to update atmosphere");
  }
  if (ImGui::IsItemDeactivatedAfterEdit()) {
    last_committed_mie_phase_g_ = mie_phase_g_;
  }
  ImGui::PopItemWidth(); ImGui::SameLine();
  if (ImGui::Button("Reset##phaseg", ImVec2(60, 0))) {
    mie_phase_g_ = 0.8f;
    last_committed_mie_phase_g_ = 0.8f;
  }
  
  // Mie (aerosol) density
  ImGui::Text("Aerosol"); ImGui::SameLine(90); ImGui::PushItemWidth(-100);
  ImGui::SliderFloat("##mieden", &mie_density_, 0.0f, 3.0f, "%.2f");
  if (ImGui::IsItemActive()) {
    ImGui::SetTooltip("Release to update atmosphere");
  }
  if (ImGui::IsItemDeactivatedAfterEdit()) {
    last_committed_mie_density_ = mie_density_;
  }
  ImGui::PopItemWidth(); ImGui::SameLine();
  if (ImGui::Button("Reset##mieden", ImVec2(60, 0))) {
    mie_density_ = 1.0f;
    last_committed_mie_density_ = 1.0f;
  }
  
  // Rayleigh (molecular) density
  ImGui::Text("Air"); ImGui::SameLine(90); ImGui::PushItemWidth(-100);
  ImGui::SliderFloat("##rayden", &rayleigh_density_, 0.0f, 3.0f, "%.2f");
  if (ImGui::IsItemActive()) {
    ImGui::SetTooltip("Release to update atmosphere");
  }
  if (ImGui::IsItemDeactivatedAfterEdit()) {
    last_committed_rayleigh_density_ = rayleigh_density_;
  }
  ImGui::PopItemWidth(); ImGui::SameLine();
  if (ImGui::Button("Reset##rayden", ImVec2(60, 0))) {
    rayleigh_density_ = 1.0f;
    last_committed_rayleigh_density_ = 1.0f;
  }
  
  // Ground albedo
  ImGui::Text("Ground"); ImGui::SameLine(90); ImGui::PushItemWidth(-100);
  ImGui::SliderFloat("##groundalb", &ground_albedo_, 0.0f, 1.0f, "%.2f");
  if (ImGui::IsItemActive()) {
    ImGui::SetTooltip("Release to update atmosphere");
  }
  if (ImGui::IsItemDeactivatedAfterEdit()) {
    last_committed_ground_albedo_ = ground_albedo_;
  }
  ImGui::PopItemWidth(); ImGui::SameLine();
  if (ImGui::Button("Reset##groundalb", ImVec2(60, 0))) {
    ground_albedo_ = 0.1f;
    last_committed_ground_albedo_ = 0.1f;
  }
}

void GUIPrototype::RenderRenderingPanel() {
  const ImVec4 mattepaint_orange = ImVec4(0.925f, 0.529f, 0.106f, 1.00f);
  ImGui::PushStyleColor(ImGuiCol_Text, mattepaint_orange);
  ImGui::Text("RENDERING SETTINGS");
  ImGui::PopStyleColor();
  ImGui::Separator();
  ImGui::Spacing();
  
  ImGui::Text("Exposure"); ImGui::SameLine(90); ImGui::PushItemWidth(-100);
  ImGui::SliderFloat("##exposure", &exposure_, 0.1f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
  ImGui::PopItemWidth(); ImGui::SameLine();
  if (ImGui::Button("Reset##exp", ImVec2(60, 0))) exposure_ = 10.0f;
  
  ImGui::Separator();
  ImGui::Spacing();
  
  ImGui::Checkbox("Use Ozone Layer", &use_ozone_);
  ImGui::SameLine(ImGui::GetWindowWidth() - 70);
  if (ImGui::Button("Reset##ozo", ImVec2(60, 0))) use_ozone_ = true;
  
  ImGui::Checkbox("Combined Textures", &use_combined_textures_);
  ImGui::SameLine(ImGui::GetWindowWidth() - 70);
  if (ImGui::Button("Reset##tex", ImVec2(60, 0))) use_combined_textures_ = true;
  
  ImGui::Checkbox("White Balance", &white_balance_);
  ImGui::SameLine(ImGui::GetWindowWidth() - 70);
  if (ImGui::Button("Reset##wb", ImVec2(60, 0))) white_balance_ = false;
  
  ImGui::Spacing();
  ImGui::Text("Luminance Mode");
  ImGui::PushItemWidth(-70);
  const char* luminance_items[] = { "Off", "Approximate", "Precomputed" };
  ImGui::Combo("##luminance", &luminance_mode_, luminance_items, 3);
  ImGui::PopItemWidth();
  ImGui::SameLine();
  if (ImGui::Button("Reset##lum", ImVec2(60, 0))) luminance_mode_ = 0;
  
  ImGui::Spacing();
  ImGui::Text("Render Mode");
  ImGui::PushItemWidth(-70);
  const char* render_mode_items[] = { "Perspective", "Latlong 360" };
  ImGui::Combo("##rendermode", &render_mode_, render_mode_items, 2);
  ImGui::PopItemWidth();
  ImGui::SameLine();
  if (ImGui::Button("Reset##rm", ImVec2(60, 0))) render_mode_ = 0;
  
  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();
  
  // Scene Objects section
  ImGui::PushStyleColor(ImGuiCol_Text, mattepaint_orange);
  ImGui::Text("SCENE OBJECTS");
  ImGui::PopStyleColor();
  ImGui::Spacing();
  
  ImGui::Checkbox("Show Object", &show_scene_object_);
  ImGui::SameLine(ImGui::GetWindowWidth() - 70);
  if (ImGui::Button("Reset##obj", ImVec2(60, 0))) show_scene_object_ = true;
  
  ImGui::Spacing();
  ImGui::Text("Object Shape");
  ImGui::PushItemWidth(-70);
  const char* shape_items[] = { "Sphere", "Cube", "Cone" };
  ImGui::Combo("##shape", &scene_object_shape_, shape_items, 3);
  ImGui::PopItemWidth();
  ImGui::SameLine();
  if (ImGui::Button("Reset##shp", ImVec2(60, 0))) scene_object_shape_ = 0;
}

void GUIPrototype::RenderPresetsPanel() {
  const ImVec4 mattepaint_orange = ImVec4(0.925f, 0.529f, 0.106f, 1.00f);
  ImGui::PushStyleColor(ImGuiCol_Text, mattepaint_orange);
  ImGui::Text("VIEW PRESETS");
  ImGui::PopStyleColor();
  ImGui::Separator();
  ImGui::Spacing();
  
  if (ImGui::Button("Sunrise", ImVec2(-1, 40))) {
    sun_zenith_ = 85.0f;
  }
  if (ImGui::Button("Noon", ImVec2(-1, 40))) {
    sun_zenith_ = 20.0f;
  }
  if (ImGui::Button("Sunset", ImVec2(-1, 40))) {
    sun_zenith_ = 85.0f;
  }
  if (ImGui::Button("Night", ImVec2(-1, 40))) {
    sun_zenith_ = 120.0f;
  }
  
  ImGui::Spacing();
  ImGui::Separator();
  ImGui::Spacing();
  
  ImGui::Text("Camera Presets");
  if (ImGui::Button("Ground Level", ImVec2(-1, 35))) {
    camera_pos_x_ = 0.0f; camera_pos_y_ = 1000.0f; camera_pos_z_ = 9000.0f;
    camera_rot_pitch_ = 90.0f; camera_rot_yaw_ = 180.0f; camera_rot_roll_ = 0.0f;
  }
  if (ImGui::Button("Low Orbit", ImVec2(-1, 35))) {
    camera_pos_x_ = 0.0f; camera_pos_y_ = 1000.0f; camera_pos_z_ = 2700000.0f;
    camera_rot_pitch_ = 0.0f; camera_rot_yaw_ = 0.0f; camera_rot_roll_ = 0.0f;
  }
  if (ImGui::Button("High Orbit", ImVec2(-1, 35))) {
    camera_pos_x_ = 0.0f; camera_pos_y_ = 1000.0f; camera_pos_z_ = 12000000.0f;
    camera_rot_pitch_ = 0.0f; camera_rot_yaw_ = 0.0f; camera_rot_roll_ = 0.0f;
  }
}

void GUIPrototype::Run() {
  glutMainLoop();
}

RenderingParameters GUIPrototype::GetRenderingParameters() const {
  RenderingParameters params;
  
  // Convert GUI units to rendering units
  const double kPi = 3.1415926;
  
  // Pass camera position directly (GUI uses standard XYZ: X=right, Y=up, Z=forward)
  // Renderer will handle coordinate system mapping internally
  params.camera_position_x = camera_pos_x_;  // meters
  params.camera_position_y = camera_pos_y_;  // meters
  params.camera_position_z = camera_pos_z_;  // meters
  
  // Pass camera rotation directly
  params.camera_pitch_degrees = camera_rot_pitch_;
  params.camera_yaw_degrees = camera_rot_yaw_;
  params.camera_roll_degrees = camera_rot_roll_;
  
  params.view_fov_degrees = camera_fov_;
  
  params.sun_zenith_angle_radians = sun_zenith_ * kPi / 180.0;
  params.sun_azimuth_angle_radians = sun_azimuth_ * kPi / 180.0;
  params.sun_intensity = sun_intensity_;
  
  params.exposure = exposure_;
  params.use_ozone = use_ozone_;
  params.use_combined_textures = use_combined_textures_;
  params.use_half_precision = true;  // Fixed for now
  params.luminance_mode = luminance_mode_;
  params.do_white_balance = white_balance_;
  params.use_constant_solar_spectrum = false;  // Fixed for now
  params.render_mode = render_mode_;
  
  // Atmospheric composition (use last committed values to avoid rebuilds during drag)
  params.mie_phase_function_g = last_committed_mie_phase_g_;
  params.mie_scattering_scale = last_committed_mie_density_;
  params.rayleigh_scattering_scale = last_committed_rayleigh_density_;
  params.mie_scale_height = mie_height_;
  params.rayleigh_scale_height = rayleigh_height_;
  params.ground_albedo = last_committed_ground_albedo_;
  
  // Scene objects
  params.show_scene_object = show_scene_object_;
  params.scene_object_shape = scene_object_shape_;
  
  return params;
}

void GUIPrototype::SetupFramebuffer(int width, int height) {
  // Clean up existing framebuffer if any
  CleanupFramebuffer();
  
  // Create texture to render to
  glGenTextures(1, &viewport_texture_);
  glBindTexture(GL_TEXTURE_2D, viewport_texture_);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  
  // Create depth renderbuffer
  glGenRenderbuffers(1, &depth_renderbuffer_);
  glBindRenderbuffer(GL_RENDERBUFFER, depth_renderbuffer_);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
  
  // Create framebuffer
  glGenFramebuffers(1, &framebuffer_);
  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, viewport_texture_, 0);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_renderbuffer_);
  
  // Check framebuffer completeness
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    throw std::runtime_error("Framebuffer is not complete");
  }
  
  // Unbind
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindRenderbuffer(GL_RENDERBUFFER, 0);
  
  viewport_width_ = width;
  viewport_height_ = height;
}

void GUIPrototype::CleanupFramebuffer() {
  if (framebuffer_) glDeleteFramebuffers(1, &framebuffer_);
  if (viewport_texture_) glDeleteTextures(1, &viewport_texture_);
  if (depth_renderbuffer_) glDeleteRenderbuffers(1, &depth_renderbuffer_);
  
  framebuffer_ = 0;
  viewport_texture_ = 0;
  depth_renderbuffer_ = 0;
}

void GUIPrototype::ResetCameraParameters() {
  camera_pos_x_ = 0.0f;
  camera_pos_y_ = 1000.0f;
  camera_pos_z_ = 9000.0f;
  camera_rot_pitch_ = 90.0f;
  camera_rot_yaw_ = 180.0f;
  camera_rot_roll_ = 0.0f;
  camera_fov_ = 50.0f;
}

void GUIPrototype::ResetEnvironmentParameters() {
  sun_zenith_ = 75.0f;
  sun_azimuth_ = 165.0f;
  sun_intensity_ = 1.0f;
}

void GUIPrototype::ResetAtmosphereParameters() {
  mie_phase_g_ = 0.8f;
  mie_density_ = 1.0f;
  rayleigh_density_ = 1.0f;
  mie_height_ = 1200.0f;
  rayleigh_height_ = 8000.0f;
  ground_albedo_ = 0.1f;
}

void GUIPrototype::ResetRenderingParameters() {
  exposure_ = 10.0f;
  use_ozone_ = true;
  use_combined_textures_ = true;
  luminance_mode_ = 0;
  white_balance_ = false;
}

void GUIPrototype::ResetAllParameters() {
  ResetCameraParameters();
  ResetEnvironmentParameters();
  ResetAtmosphereParameters();
  ResetRenderingParameters();
}

}  // namespace demo
}  // namespace atmosphere
