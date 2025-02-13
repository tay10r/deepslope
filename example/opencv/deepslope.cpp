#include "deepslope.h"

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <math.h>
#include <stddef.h>

namespace deepslope {

namespace {

class diffusion_model_base : public diffusion_model
{
public:
  void add_observer(observer* o) override { observers_.emplace(o); }

  void remove_observer(observer* o) override
  {
    auto it = observers_.find(o);
    if (it != observers_.end()) {
      observers_.erase(it);
    }
  }

protected:
  void notify_error(const char* what)
  {
    for (auto* o : observers_) {
      o->on_error(what);
    }
  }

  void notify_step(const int step, const float* terrain, const float* predicted_noise)
  {
    for (auto* o : observers_) {
      o->on_step(step, terrain, predicted_noise);
    }
  }

  void notify_setup()
  {
    for (auto* o : observers_) {
      o->on_setup(*this);
    }
  }

private:
  std::set<observer*> observers_;
};

class null_diffusion_model final : public diffusion_model_base
{
public:
  null_diffusion_model(const bool return_value, const bool emit_error)
    : return_value_(return_value)
    , emit_error_(emit_error)
  {
  }

  auto setup(const char*, const char*) -> bool override
  {
    if (emit_error_) {
      notify_error("Cannot setup because diffusion model is a placeholder.");
    }
    return return_value_;
  }

  auto denoise(const float*, float*, int, float, int) -> bool override
  {
    if (emit_error_) {
      notify_error("Cannot denoise because diffusion model is a placeholder.");
    }
    return return_value_;
  }

  auto num_steps() const -> int override { return 0; }

  auto input_size() const -> int override { return 0; }

private:
  bool return_value_{ false };

  bool emit_error_{ true };
};

class diffusion_model_impl final : public diffusion_model_base
{
public:
  auto setup(const char* onnx_model_path, const char* yaml_config_path) -> bool override
  {
    try {
      net_ = cv::dnn::readNetFromONNX(onnx_model_path);
    } catch (const cv::Exception& e) {
      notify_error(e.what());
      return false;
    }

    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

    cv::FileStorage config_file;

    try {
      config_file.open(yaml_config_path, cv::FileStorage::READ);
    } catch (const cv::Exception& e) {
      notify_error(e.what());
    }

    double beta_min{};
    double beta_max{};

    config_file["beta_min"] >> beta_min;
    config_file["beta_max"] >> beta_max;
    config_file["t"] >> steps_;
    config_file["input_size"] >> input_size_;

    if (steps_ <= 1) {
      std::ostringstream error;
      error << "There must be at least 2 steps in the noise schedule, but there are " << steps_;
      notify_error(error.str().c_str());
      return false;
    }

    beta_.resize(steps_);
    for (size_t i = 0; i < beta_.size(); i++) {
      // linear interpolation
      // TODO : try cosine or quadratic schedule (or make them configurable)
      const auto k = static_cast<float>(i) / static_cast<float>(beta_.size() - 1);
      beta_[i] = static_cast<float>(beta_min + (beta_max - beta_min) * k);
    }

    alpha_.resize(steps_);
    alpha_bar_.resize(steps_);
    float cumulative_prod{ 1.0F };
    for (size_t i = 0; i < alpha_bar_.size(); i++) {
      const auto alpha = 1.0F - beta_[i];
      cumulative_prod *= alpha;
      alpha_[i] = alpha;
      alpha_bar_[i] = cumulative_prod;
    }

    notify_setup();

    return true;
  }

  auto denoise(const float* input,
               float* output,
               const int size,
               const float noise_level,
               const int optional_pitch = -1) -> bool override
  {
    if (size != input_size_) {
      std::ostringstream error;
      error << "Input size is " << size << " but model expects " << input_size_;
      notify_error(error.str().c_str());
      return false;
    }

    // We convert the "noise level" to the index of the step in the noise schedule.
    // The input noise level is not exactly how much of the noise sample is blended into the input.
    // The actual blend ratio is based on the phase in the noise schedule, which the step index specifies.
    const int init_step{ static_cast<int>(std::max(std::min(noise_level, 1.0F), 0.0F) * (steps_ - 1)) };

    cv::Mat original(cv::Size(size, size), CV_32FC1, cv::Scalar());
    for (size_t i = 0; i < (size * size); i++) {
      original.at<float>(i) = input[i];
    }

    cv::Mat epsilon(cv::Size(size, size), CV_32FC1);

    cv::randn(epsilon, 0.0F, 1.0F);

    cv::Mat input_mat(cv::Size(size, size), CV_32FC1);

    forward_diffusion(original, epsilon, init_step, input_mat);

    for (auto i = init_step - 1; i >= 0; i--) {

      auto input_blob = cv::dnn::blobFromImage(input_mat);

      net_.setInput(input_blob);

      auto predicted_noise = net_.forward();

      predicted_noise = predicted_noise.reshape(1, input_size_);

      input_mat = reverse_diffusion(input_mat, predicted_noise, i);

      notify_step(
        i, reinterpret_cast<const float*>(input_mat.data), reinterpret_cast<const float*>(predicted_noise.data));
    }

    for (size_t i = 0; i < (size * size); i++) {
      output[i] = input_mat.at<float>(i);
    }

    return true;
  }

  auto num_steps() const -> int override { return steps_; }

protected:
  void forward_diffusion(const cv::Mat& x0, const cv::Mat& epsilon, const size_t step, cv::Mat& x_t)
  {
    const auto a = alpha_bar_.at(step);
    x_t = x0 * sqrtf(a) + sqrtf(1.0F - a) * epsilon;
  }

  auto reverse_diffusion(const cv::Mat& x_t, const cv::Mat& predicted_epsilon, const int step) const -> cv::Mat
  {
    cv::Mat result = x_t - sqrtf(1.0F - alpha_bar_.at(step)) * predicted_epsilon / sqrtf(alpha_.at(step));

    if (step > 0) {
      cv::Mat added_noise(cv::Size(input_size_, input_size_), CV_32FC1);
      cv::randn(added_noise, 0, 1);
      result = result + sqrtf(beta_.at(step)) * added_noise;
    }

    return result;
  }

  auto input_size() const -> int override { return input_size_; }

private:
  cv::dnn::Net net_;

  double noise_{};

  int steps_{};

  int input_size_{};

  std::vector<float> beta_;

  std::vector<float> alpha_;

  std::vector<float> alpha_bar_;
};

} // namespace

auto
diffusion_model::create() -> std::unique_ptr<diffusion_model>
{
  return std::make_unique<diffusion_model_impl>();
}

auto
diffusion_model::create_null(const bool return_value, const bool emit_error) -> std::unique_ptr<diffusion_model>
{
  return std::make_unique<null_diffusion_model>(return_value, emit_error);
}

} // namespace deepslope
