#ifndef MOCAP_POSE_MODEL_HPP
#define MOCAP_POSE_MODEL_HPP

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

#include "error.hpp"
#include "preprocess.hpp"
#include "session.hpp"

namespace mocap {

struct Keypoint {
  float x;            // source frame pixels
  float y;
  float confidence;
};

struct PoseResult {
  uint8_t camera_id;
  std::array<Keypoint, NUM_KEYPOINTS> keypoints;
};

// RTMW pose estimation over a frameset, batched across cameras.
//
// Staging and inference are separate calls on purpose. The source surfaces are
// only read during staging, so the caller can hand the frameset back before
// the network runs, which is the difference between holding a decoder surface
// for a preprocessing kernel and holding it for the whole pipeline.
class PoseModel {
public:
  static Result<PoseModel> load(const std::filesystem::path& engine_path,
                                uint32_t frame_width,
                                uint32_t frame_height);

  PoseModel(PoseModel&& other) noexcept;
  PoseModel& operator=(PoseModel&& other) noexcept;
  PoseModel(const PoseModel&) = delete;
  PoseModel& operator=(const PoseModel&) = delete;
  ~PoseModel();

  // Preprocesses every frame in the set into the network input and returns
  // once the GPU has finished reading them. The frameset may be released as
  // soon as this returns.
  Result<void> stage(const Frameset& set);

  // Runs the network on what stage() left, and decodes keypoints into source
  // frame coordinates. The span is valid until the next call.
  Result<std::span<const PoseResult>> infer();

private:
  struct State;
  explicit PoseModel(std::unique_ptr<State> state);

  std::unique_ptr<State> m_state;
};

} // namespace mocap

#endif // MOCAP_POSE_MODEL_HPP
