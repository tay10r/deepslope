/// @file example.cpp
///
/// @brief This file demonstrates how to use the API.

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <stdint.h>
#include <stdlib.h>

#include <opencv2/opencv.hpp>

#include "deepslope.h"

void
save_image(const std::string& filename, const float* data, const int w, const int h)
{
  cv::Mat mat(cv::Size(w, h), CV_8UC1);
  for (auto y = 0; y < h; y++) {
    for (auto x = 0; x < w; x++) {
      const auto i = y * w + x;
      const auto v = static_cast<int>(data[i] * 255);
      mat.at<uint8_t>(y * w + x) = static_cast<uint8_t>(std::max(std::min(v, 255), 0));
    }
  }
  cv::imwrite(filename, mat);
}

class example_observer final : public deepslope::observer
{
public:
  void set_terrain_size(const int terrain_size) { terrain_size_ = terrain_size; }

  void on_error(const char* what) override { std::cerr << "ERROR: " << what << std::endl; }

  void on_step(const int step, const float* terrain, const float* predicted_noise) override
  {
    std::cout << "step " << step << " complete." << std::endl;

    {
      std::ostringstream path_stream;
      path_stream << "terrain_" << std::setw(5) << std::setfill('0') << step << ".png";
      save_image(path_stream.str(), terrain, terrain_size_, terrain_size_);
    }

    {
      std::ostringstream path_stream;
      path_stream << "noise_" << std::setw(5) << std::setfill('0') << step << ".png";
      save_image(path_stream.str(), predicted_noise, terrain_size_, terrain_size_);
    }
  }

  void on_setup(const deepslope::diffusion_model& model) override { terrain_size_ = model.input_size(); }

private:
  int terrain_size_{};
};

template<typename T>
void
convert(const cv::Mat& mat, float* out, const float scale)
{
  const auto dim_size = mat.size.p[0];

  for (auto i = 0; i < (dim_size * dim_size); i++) {
    out[i] = static_cast<float>(mat.at<T>(i)) * scale;
  }
}

auto
main() -> int
{
  const char* input_path{ PROJECT_SOURCE_DIR "/example_input_64.png" };
  const char* output_path{ "output.png" };

  const char* model_path{ "terrain-diffuse-64.onnx" };
  const char* config_path{ "terrain-diffuse-64.json" };

  float noise_level{ 1.0F };
  // float noise_level{ 0.5F };

  auto model = deepslope::diffusion_model::create();

  example_observer observer;

  model->add_observer(&observer);

  if (!model->setup(model_path, config_path)) {
    return EXIT_FAILURE;
  }

  cv::Mat input;

  try {
    input = cv::imread(input_path, cv::IMREAD_GRAYSCALE);
  } catch (const cv::Exception& e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  const auto dim_size = input.size.p[0];
  if (dim_size != input.size.p[1]) {
    std::cerr << "Dimensions of input must be equal." << std::endl;
    return EXIT_FAILURE;
  }

  std::vector<float> input_data(dim_size * dim_size, 0.0F);

  switch (input.type()) {
    case CV_8UC1:
      std::cout << "Using 8-bit integer type." << std::endl;
      convert<uint8_t>(input, input_data.data(), 1.0F / 255.0F);
      break;
    case CV_16UC1:
      std::cout << "Using 16-bit integer type." << std::endl;
      convert<uint16_t>(input, input_data.data(), 1.0F / 65535.0F);
      break;
    case CV_32FC1:
      std::cout << "Using 32-bit float type." << std::endl;
      convert<float>(input, input_data.data(), 1.0F);
      break;
    default:
      std::cerr << "Unsupported pixel type: " << cv::typeToString(input.type()) << std::endl;
      return EXIT_FAILURE;
  }

  std::vector<float> output_data(dim_size * dim_size, 0.0F);

  if (!model->denoise(input_data.data(), output_data.data(), dim_size, noise_level)) {
    return EXIT_FAILURE;
  }

  cv::Mat output(dim_size, dim_size, CV_8UC1);

  for (auto i = 0; i < (dim_size * dim_size); i++) {
    const auto val = std::min(std::max(static_cast<int>(output_data[i] * 255), 0), 255);
    output.data[i] = static_cast<uint8_t>(val);
  }

  cv::imwrite("output.png", output);

  return EXIT_SUCCESS;
}
