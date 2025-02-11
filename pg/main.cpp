#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <random>

#include "deps/FastNoiseLite.h"

namespace {

namespace py = pybind11;

class generator final
{
public:
  generator(const int seed)
    : rng_(seed)
    , seed_(seed)
  {
    //
  }

  auto generate(int h, int w) -> py::array_t<float>
  {
    py::array_t<float> result(std::vector<int>{ h, w });

    switch (select_option(/*num_options=*/2)) {
      case 0:
        generate_brush(h, w, result.mutable_data(0));
        break;
      case 1:
        generate_noise(h, w, result.mutable_data(0));
        break;
    }

    return result;
  }

protected:
  void generate_brush(int h, int w, float* data)
  {
    const auto x_scale{ 1.0F / w };
    const auto y_scale{ 1.0F / h };

    std::uniform_real_distribution<float> height_dist(min_height_, max_height_);
    const auto max_height = height_dist(rng_);

    std::uniform_real_distribution<float> uv_dist(0, 1.0F);
    const auto u_center = uv_dist(rng_);
    const auto v_center = uv_dist(rng_);

    std::uniform_real_distribution<float> r_dist(0.001F, 0.5F);
    const auto r = r_dist(rng_);

#pragma omp parallel for

    for (auto y = 0; y < h; y++) {
      for (auto x = 0; x < w; x++) {
        const auto u = (x + 0.5F) * x_scale;
        const auto v = (y + 0.5F) * y_scale;
        const auto du = u - u_center;
        const auto dv = v - v_center;
        const auto h = max_height / (1.0F + (du * du + dv * dv) / r);
        data[y * w + x] = h;
      }
    }
  }

  void generate_noise(int h, int w, float* data)
  {
    std::uniform_real_distribution<float> height_dist(min_height_, max_height_);
    const auto max_height = height_dist(rng_);

    std::uniform_real_distribution<float> freq_dist(min_frequency_, max_frequency_);
    const auto freq = freq_dist(rng_);

    std::uniform_int_distribution<int> octave_dist(min_octaves_, max_octaves_);
    const auto octaves = octave_dist(rng_);

    std::uniform_real_distribution<float> offset_dist(-1000.0F, 1000.0F);
    const auto u_offset = offset_dist(rng_);
    const auto v_offset = offset_dist(rng_);

    std::uniform_real_distribution<float> cutoff_dist(-max_height, max_height);
    const auto cutoff = cutoff_dist(rng_);

    FastNoiseLite noise(seed_);
    noise.SetFrequency(freq);
    noise.SetFractalOctaves(octaves);
    noise.SetFractalType(FastNoiseLite::FractalType_FBm);

    const auto x_scale{ 1.0F / w };
    const auto y_scale{ 1.0F / h };

#pragma omp parallel for

    for (auto y = 0; y < h; y++) {
      for (auto x = 0; x < w; x++) {
        const auto u = (x + 0.5F) * x_scale + u_offset;
        const auto v = (y + 0.5F) * y_scale + v_offset;
        const auto h = noise.GetNoise(u, v) * max_height;
        data[y * w + x] = (h < cutoff) ? 0.0F : (h - cutoff);
      }
    }
  }

  auto select_option(int num_options) -> int
  {
    std::uniform_int_distribution<int> dist(0, num_options - 1);
    return dist(rng_);
  }

  auto dice_roll(float p = 0.5F) -> bool
  {
    std::uniform_real_distribution<float> dist(0, 1);
    return dist(rng_) < p;
  }

private:
  std::mt19937 rng_;

  int seed_{ 0 };

  float min_height_{ 10.0F };

  float max_height_{ 100.0F };

  float min_frequency_{ 1.0F };

  float max_frequency_{ 2.0F };

  int min_octaves_{ 1 };

  int max_octaves_{ 8 };
};

} // namespace

PYBIND11_MODULE(pg, m)
{
  py::class_<generator>(m, "Generator")
    .def(py::init<int>(), py::arg("seed"))
    .def("generate", &generator::generate, py::arg("h"), py::arg("w"));
}
