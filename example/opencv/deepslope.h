#pragma once

#include <memory>

namespace deepslope {

class diffusion_model;

/// @brief This is an optional interface class to allow callers to monitor things going on in the model.
///        Examples include detailed error conditions and denoising progress.
class observer
{
public:
  virtual ~observer() = default;

  /// @brief Called when an error occurs in the model.
  ///
  /// @param what An error message describing what happened.
  virtual void on_error(const char* what) = 0;

  /// @brief Called when a step in the diffusion process is completed.
  ///
  /// @param step The index of the step that was completed.
  ///
  /// @param terrain The current state of the denoised terrain.
  ///
  /// @param predicted_noise The noise predicted from the network.
  virtual void on_step(int step, const float* terrain, const float* predicted_noise) = 0;

  /// @brief Called when the diffusion model is successfully setup.
  virtual void on_setup(const diffusion_model&) = 0;
};

/// @brief This is the interface to the diffusion model.
///
/// @details This class is an abstract base class to allow for dependency injection as well as to avoid exposing
///          to much of the implementation and included header files.
class diffusion_model
{
public:
  /// @brief Creates a new diffusion model instance.
  ///
  /// @details After creating an instance with this function, you'll need to call @ref diffusion_model::setup before
  ///          denoising the terrain.
  [[nodiscard]] static auto create() -> std::unique_ptr<diffusion_model>;

  /// @brief This creates a null diffusion model instance.
  ///
  /// @details The null instance allows there to be a place holder for a diffusion model, when needed.
  ///          When using the instance creating by this class, all function calls will return the boolean value passed
  ///          to this function.
  ///
  /// @param return_value The return value to give back to the caller any time a function is called on the null object.
  ///
  /// @param emit_error Whether to pass an error message to the observer classes.
  [[nodiscard]] static auto create_null(bool return_value = false, bool emit_error = true)
    -> std::unique_ptr<diffusion_model>;

  virtual ~diffusion_model() = default;

  /// @brief Adds an observer to the model.
  ///
  /// @param o The observer to add.
  ///
  /// @note The model does not take ownership of this pointer.
  virtual void add_observer(observer* o) = 0;

  /// @brief Removes an observer from the model.
  ///
  /// @param o A pointer to the observer to remove.
  virtual void remove_observer(observer* o) = 0;

  /// @brief Initializes the model.
  ///
  /// @param onnx_model_path The path to the ONNX file containing the model.
  ///
  /// @param yaml_config_path The path to the corresponding YAML file that contains the noise schedule.
  ///
  /// @return True on success, false on failure.
  [[nodiscard]] virtual auto setup(const char* onnx_model_path, const char* yaml_config_path) -> bool = 0;

  /// @brief Takes a terrain height map as an input, adds a given amount of noise to it, and then denoises it.
  ///
  /// @details The amount of noise to add depends on how much detail the caller would like to add to the terrain.
  ///          Once the noise is added, the terrain gets progressively refined until the noise is removed and smooth
  ///          detail is left over.
  ///
  /// @param input A pointer to the input terrain.
  ///
  /// @param output A pointer to the buffer that is assigned the results of the diffusion process.
  ///
  /// @param size The size of the input terrain. The diffusion models expect a certain terrain size. If the terrain is
  ///             not the size that the model is expecting, this function call will fail.
  ///
  /// @param noise_level The amount of noise to add to the terrain. This value can range from 0 to 1, where a value of
  ///                    one means to completely replace the terrain with noise.
  ///
  /// @param optional_input_pitch The number of bytes per row in the input terrain. If this value is left as negative
  ///                             one (the default value), the pitch will be computed automatically and the terrain is
  ///                             assumed to be densely packed.
  ///
  /// @return True on success, false on failure.
  [[nodiscard]] virtual auto denoise(const float* input,
                                     float* output,
                                     int size,
                                     float noise_level,
                                     int optional_input_pitch = -1) -> bool = 0;

  /// @brief Gets the number of steps in the model's diffusion process.
  ///
  /// @return The number of steps in the diffusion process.
  [[nodiscard]] virtual auto num_steps() const -> int = 0;

  /// @brief Gets the input size of the model.
  ///
  /// @return The width and height of the terrain input.
  [[nodiscard]] virtual auto input_size() const -> int = 0;
};

} // namespace deepslope
