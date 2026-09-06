#include <chrono>
#include <cstdint>
#include <expected>
#include <optional>
#include <print>
#include <span>
#include <thread>

#include "pose_model.hpp"
#include "session.hpp"
#include "triangulate.hpp"

namespace {

constexpr const char* CONFIG_PATH = "cams.toml";
constexpr const char* ENGINE_PATH = "../models/rtmw-x.plan";

constexpr std::chrono::seconds RUN_FOR{10};
constexpr std::chrono::microseconds POLL_INTERVAL{500};

// Acquire a frameset, stage it, hand it straight back, then run the network.
//
// The release sits between staging and inference on purpose. Staging is the
// only step that reads the decoder surfaces, so holding the frameset across
// inference would pin an NVDEC surface per camera for the whole pipeline, and
// the surface pool is what back pressures into the cameras when it runs out.
void run_inference(mocap::Session& session, mocap::PoseModel& model,
                   mocap::Triangulator& triangulator) {
  uint64_t framesets = 0;
  uint64_t poses = 0;
  uint64_t solved = 0;

  std::chrono::steady_clock::time_point deadline =
    std::chrono::steady_clock::now() + RUN_FOR;

  while (std::chrono::steady_clock::now() < deadline) {
    std::optional<mocap::Frameset> set = session.try_acquire_frameset();
    if (!set) {
      std::this_thread::sleep_for(POLL_INTERVAL);
      continue;
    }

    const uint64_t timestamp = set->timestamp;

    mocap::Result<void> staged = model.stage(*set);
    session.release_frameset(*set);

    if (!staged) {
      std::println(stderr, "stage: {}: {}",
                   staged.error().detail, staged.error().ec.message());
      continue;
    }

    mocap::Result<std::span<const mocap::PoseResult>> results = model.infer();
    if (!results) {
      std::println(stderr, "infer: {}: {}",
                   results.error().detail, results.error().ec.message());
      continue;
    }

    // Stopping here on purpose: the 3D pose is in the reference camera's
    // frame, in board square units, for one capture timestamp. Nothing
    // downstream consumes it yet.
    const mocap::Pose3d& pose = triangulator.triangulate(*results);
    (void)timestamp;

    for (const mocap::Keypoint3d& keypoint : pose.keypoints)
      solved += keypoint.views >= 2 ? 1 : 0;

    framesets += 1;
    poses += results->size();
  }

  std::println("{} framesets, {} poses, {} joints triangulated",
               framesets, poses, solved);
}

} // namespace

int main() {
  mocap::Result<mocap::Session> session = mocap::Session::start(CONFIG_PATH);
  if (!session) {
    std::println(stderr, "session: {}: {}",
                 session.error().detail, session.error().ec.message());
    return 1;
  }

  const mocap::Config& conf = session->config();

  mocap::Result<mocap::PoseModel> model = mocap::PoseModel::load(
    ENGINE_PATH, conf.stream.frame_width, conf.stream.frame_height);
  if (!model) {
    std::println(stderr, "model: {}: {}",
                 model.error().detail, model.error().ec.message());
    return 1;
  }

  mocap::Result<mocap::Triangulator> triangulator =
    mocap::Triangulator::load(conf.cameras);
  if (!triangulator) {
    std::println(stderr, "triangulator: {}: {}",
                 triangulator.error().detail, triangulator.error().ec.message());
    return 1;
  }

  std::println("running inference for {}s", RUN_FOR.count());
  run_inference(*session, *model, *triangulator);

  return 0;
}
