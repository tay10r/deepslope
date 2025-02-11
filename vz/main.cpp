#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#define GLFW_INCLUDE_NONE 1
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <implot.h>

#include <glad/glad.h>

#include <iostream>

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <stdint.h>

namespace {

namespace py = pybind11;

constexpr int max_samples{ 8192 };

template<typename Scalar>
static auto
clamp(Scalar x, Scalar min, Scalar max) -> Scalar
{
  return std::max(std::min(x, max), min);
}

class window final
{
public:
  window()
    : ui_thread_(&window::run_thread, this)
  {
  }

  window(const window&) = delete;

  void log_test_results(int h, int w, const float* data)
  {
    std::vector<unsigned char> copy(w * h * 4, 0);

    for (auto i = 0; i < (w * h); i++) {
      auto* dst = &copy[i * 4];
      const auto src = clamp(static_cast<int>(data[i] * 255.0F), 0, 255);
      dst[0] = src;
      dst[1] = src;
      dst[2] = src;
      dst[3] = 255;
    }

    {
      std::lock_guard<std::mutex> lock(test_results_lock_);
      test_results_ = std::move(copy);
      test_results_width_ = w;
      test_results_height_ = h;
      test_results_dirty_ = true;
    }
  }

  void log_loss(const std::string& name, const float epoch, const float loss)
  {
    std::lock_guard<std::mutex> lock(loss_lock_);

    auto it = loss_.find(name);
    if (it == loss_.end()) {
      it = loss_.emplace(name, std::make_pair(std::vector<float>{}, std::vector<float>{})).first;
    }

    it->second.first.emplace_back(epoch);
    it->second.second.emplace_back(loss);

    if (it->second.first.size() > max_samples) {
      it->second.first.erase(it->second.first.begin());
      it->second.second.erase(it->second.second.begin());
    }
  }

  void log_grad(const std::string& name, const float value)
  {
    std::lock_guard<std::mutex> lock(grad_lock_);

    auto it = grad_indices_.find(name);
    if (it == grad_indices_.end()) {
      it = grad_indices_.emplace(name, grad_values_.size()).first;
      grad_values_.emplace_back(0.0F);
      grad_names_.emplace_back(name);
      grad_ticks_.emplace_back(static_cast<double>(grad_ticks_.size()));
    }

    grad_values_[it->second] = value;
  }

  void close()
  {
    close_request_.store(true);

    if (ui_thread_.joinable()) {
      ui_thread_.join();
    }
  }

  [[nodiscard]] auto is_open() const -> bool { return is_open_.load(); }

protected:
  void log_error(const char* what) { std::cerr << "[vz] [error]: " << what << std::endl; }

  void run_thread()
  {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
    glfwWindowHint(GLFW_MAXIMIZED, 1);

    auto* monitor = glfwGetPrimaryMonitor();

    const auto* mode = glfwGetVideoMode(monitor);

    window_ = glfwCreateWindow(mode->width, mode->height, "DeepSlope Visualizer", nullptr, nullptr);
    if (!window_) {
      is_open_.store(false);
      log_error("Failed to create GLFW window.");
      return;
    }

    glfwMakeContextCurrent(window_);

    gladLoadGLES2Loader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress));

    glClearColor(0, 0, 0, 1);

    glGenTextures(1, &test_results_texture_);
    glBindTexture(GL_TEXTURE_2D, test_results_texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window_, true);
    ImGui_ImplOpenGL3_Init("#version 100");

    ImPlot::CreateContext();

    auto& style = ImGui::GetStyle();
    style.WindowBorderSize = 0;

    auto& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    while (!close_request_.load()) {

      if (glfwWindowShouldClose(window_)) {
        break;
      }

      if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
        break;
      }

      check_test_results();

      int w = 0;
      int h = 0;
      glfwGetFramebufferSize(window_, &w, &h);

      glViewport(0, 0, w, h);

      glClear(GL_COLOR_BUFFER_BIT);

      ImGui_ImplOpenGL3_NewFrame();
      ImGui_ImplGlfw_NewFrame();
      ImGui::NewFrame();

      ImGui::DockSpaceOverViewport();

      if (ImGui::Begin("gradient")) {
        render_grad();
      }
      ImGui::End();

      if (ImGui::Begin("loss")) {
        render_loss();
      }
      ImGui::End();

      if (ImGui::Begin("sample")) {
        render_sample();
      }
      ImGui::End();

      ImGui::Render();

      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

      glfwSwapBuffers(window_);
    }

    ImPlot::DestroyContext();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glDeleteTextures(1, &test_results_texture_);

    glfwDestroyWindow(window_);

    is_open_.store(false);
  }

  void render_sample()
  {
    if (!ImPlot::BeginPlot("##sample", ImVec2(-1, -1), ImPlotFlags_Crosshairs | ImPlotFlags_NoFrame)) {
      return;
    }

    ImPlot::PlotImage("##test_result",
                      reinterpret_cast<ImTextureID>(static_cast<uint64_t>(test_results_texture_)),
                      // reinterpret_cast<ImTextureID>(&test_results_texture_),
                      ImPlotPoint(0, 0),
                      ImPlotPoint(1, 1));

    ImPlot::EndPlot();
  }

  void render_loss()
  {
    if (!ImPlot::BeginPlot("##loss", ImVec2(-1, -1), ImPlotFlags_Crosshairs | ImPlotFlags_NoFrame)) {
      return;
    }

    ImPlot::SetupAxes("Epoch", "Loss", ImPlotAxisFlags_AutoFit, 0);

    ImPlot::SetupAxisLimits(ImAxis_Y1, -2, 2, ImPlotCond_Once);

    {
      std::lock_guard<std::mutex> lock(loss_lock_);

      for (const auto& entry : loss_) {
        auto& xy = entry.second;
        ImPlot::PlotLine(entry.first.c_str(), xy.first.data(), xy.second.data(), xy.second.size());
      }
    }

    ImPlot::EndPlot();
  }

  void render_grad()
  {
    if (!ImPlot::BeginPlot("##grad", ImVec2(-1, -1), ImPlotFlags_NoFrame | ImPlotFlags_Crosshairs)) {
      return;
    }

    {
      std::lock_guard<std::mutex> lock(grad_lock_);

      std::vector<const char*> ptrs;
      for (const auto& entry : grad_names_) {
        ptrs.emplace_back(entry.c_str());
      }

      ImPlot::SetupAxisTicks(ImAxis_Y1, grad_ticks_.data(), grad_ticks_.size(), ptrs.data());

      ImPlot::PlotBars("##Grad", grad_values_.data(), grad_values_.size(), 0.67, 0.0, ImPlotBarsFlags_Horizontal);
    }

    ImPlot::EndPlot();
  }

  void check_test_results()
  {
    auto dirty{ false };
    int w{};
    int h{};
    std::vector<unsigned char> data;

    {
      std::lock_guard<std::mutex> lock(test_results_lock_);
      if (test_results_dirty_) {
        dirty = true;
        w = test_results_width_;
        h = test_results_height_;
        data = std::move(test_results_);
        test_results_dirty_ = false;
      }
    }

    if (!dirty) {
      return;
    }

    glBindTexture(GL_TEXTURE_2D, test_results_texture_);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
  }

private:
  std::atomic<bool> close_request_{ false };

  std::atomic<bool> is_open_{ true };

  std::thread ui_thread_;

  std::mutex test_results_lock_;

  std::vector<unsigned char> test_results_;

  int test_results_width_{};

  int test_results_height_{};

  bool test_results_dirty_{ false };

  // note: this is only owned by the UI thread.
  GLuint test_results_texture_{};

  std::mutex loss_lock_;

  std::map<std::string, std::pair<std::vector<float>, std::vector<float>>> loss_;

  std::mutex grad_lock_;

  std::map<std::string, int> grad_indices_;

  std::vector<std::string> grad_names_;

  std::vector<double> grad_ticks_;

  std::vector<float> grad_values_;

  GLFWwindow* window_{};
};

} // namespace

PYBIND11_MODULE(vz, m)
{
  m.def("init", []() {
    if (glfwInit() != GLFW_TRUE) {
      throw std::runtime_error("Failed to initialize GLFW.");
    }
  });

  m.def("teardown", []() { glfwTerminate(); });

  m.def("poll_events", []() { glfwPollEvents(); });

  m.def("wait_events", []() { glfwWaitEvents(); });

  py::class_<window>(m, "Window")
    .def(py::init<>())
    .def("close", &window::close)
    .def("is_open", &window::is_open)
    .def("log_test_results",
         [](window& self, const py::array_t<float, py::array::forcecast | py::array::c_style>& img) {
           if (img.ndim() != 2) {
             throw std::runtime_error("Test results must be 2 dimensional.");
           }
           self.log_test_results(img.shape(0), img.shape(1), img.data(0));
         })
    .def("log_loss", &window::log_loss)
    .def("log_grad", &window::log_grad);
}
