#include <chrono>
#include <cstdint>
#include <print>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "copy_gray_to_host.hpp"
#include "calibration_params.hpp"
#include "session.hpp"
#include "stereo_calibration.hpp"

namespace {

constexpr const char* CONFIG_PATH = "cams.toml";
constexpr int BOARD_WIDTH = 9;
constexpr int BOARD_HEIGHT = 6;
constexpr float SQUARE_SIZE = 25.0f;   // mm

constexpr std::chrono::microseconds POLL_INTERVAL{500};
constexpr int ESC_KEY = 27;

// stereo extrinsics are solved on top of each camera's intrinsics, so every
// camera must have been through lens calibration first
std::optional<std::vector<mocap::calibration_params>>
load_intrinsics(const std::vector<mocap::Camera>& cameras) {
  std::vector<mocap::calibration_params> intrinsics(cameras.size());

  for (size_t i = 0; i < cameras.size(); i += 1) {
    std::string filename = cameras[i].name + "_calibration.yaml";
    if (!mocap::load_calibration_params(filename, intrinsics[i])) {
      std::println(stderr, "missing {}, run lens_calibration first", filename);
      return std::nullopt;
    }
  }

  return intrinsics;
}

// every camera gets its own window so the operator can see where the board is
// while moving it between pairs. returns false if the operator aborted.
bool show(const cv::Mat& image, const std::string& window) {
  cv::imshow(window, image);
  return cv::waitKey(1) != ESC_KEY;
}

void print_progress(const std::vector<mocap::PairProgress>& pairs,
                    const std::vector<mocap::Camera>& cameras) {
  for (const mocap::PairProgress& pair : pairs)
    std::print("{}+{} {}/{}   ",
               cameras[pair.cam1].name, cameras[pair.cam2].name,
               pair.views, mocap::MIN_FRAMES + 1);

  std::print("\r");
  std::fflush(stdout);
}

// runs until check_status is satisfied: every pair is either untouched or has
// enough shared views, and at least one has enough. a pair left at zero is the
// operator not having covered it, not something the tool can wait out.
void collect_shared_views(mocap::Session& session,
                          mocap::StereoCalibration& stereo,
                          const std::vector<mocap::Camera>& cameras) {
  std::vector<cv::Mat> frames(cameras.size());
  bool running = true;

  while (running && !stereo.check_status()) {
    std::optional<mocap::Frameset> set = session.try_acquire_frameset();
    if (!set) {
      std::this_thread::sleep_for(POLL_INTERVAL);
      continue;
    }

    // try_frames indexes by camera and would read an empty Mat for any camera
    // that missed the window, so only complete sets are fed to it
    const bool complete = set->frames.size() == cameras.size();

    bool converted = complete;
    for (const mocap::FrameView& view : set->frames) {
      if (!mocap::copy_gray_to_host(view, frames[view.camera_id])) {
        converted = false;
        continue;
      }

      if (!show(frames[view.camera_id], cameras[view.camera_id].name))
        running = false;
    }

    session.release_frameset(*set);

    if (!converted)
      continue;

    stereo.try_frames(frames.data());
    print_progress(stereo.progress(), cameras);
  }

  std::println("");
}

} // namespace

int main() {
  mocap::Result<mocap::Session> session = mocap::Session::start(CONFIG_PATH);
  if (!session) {
    std::println(stderr, "session: {}: {}",
                 session.error().detail, session.error().ec.message());
    return 1;
  }

  // the session outlives this scope, so the config it owns can be read
  // directly rather than copied out
  const mocap::Config& conf = session->config();

  std::optional<std::vector<mocap::calibration_params>> intrinsics =
    load_intrinsics(conf.cameras);
  if (!intrinsics)
    return 1;

  mocap::StereoCalibration stereo(
    intrinsics->data(),
    static_cast<int>(conf.cameras.size()),
    static_cast<int>(conf.stream.frame_width),
    static_cast<int>(conf.stream.frame_height),
    BOARD_WIDTH, BOARD_HEIGHT, SQUARE_SIZE
  );

  std::println("hold the {}x{} board where two cameras can see it at once",
               BOARD_WIDTH, BOARD_HEIGHT);
  std::println("every pair needs more than {} shared views, and finishes when",
               mocap::MIN_FRAMES);
  std::println("no pair is left part way\n");

  collect_shared_views(*session, stereo, conf.cameras);

  if (!stereo.check_status()) {
    std::println(stderr, "not enough shared views of the board, nothing saved");
    return 1;
  }

  stereo.calibrate();
  stereo.save_params(conf.cameras);
  std::println("saved extrinsics");

  return 0;
}
