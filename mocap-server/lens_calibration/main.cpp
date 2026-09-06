#include <chrono>
#include <cstdint>
#include <expected>
#include <optional>
#include <print>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "copy_gray_to_host.hpp"
#include "lens_calibration.hpp"
#include "session.hpp"

namespace {

constexpr const char* CONFIG_PATH = "cams.toml";
constexpr int BOARD_WIDTH = 9;
constexpr int BOARD_HEIGHT = 6;
constexpr float SQUARE_SIZE = 25.0f;   // mm

// calibrateCamera is expensive, so a camera that has enough frames is retried
// only every few new ones rather than after each
constexpr int RECALIBRATE_EVERY = 5;
constexpr std::chrono::microseconds POLL_INTERVAL{500};
constexpr int ESC_KEY = 27;

struct CameraProgress {
  int accepted = 0;
  int last_attempt = 0;
  bool calibrated = false;
  double error = 0.0;
};

bool all_calibrated(const std::vector<CameraProgress>& progress) {
  for (const CameraProgress& camera : progress)
    if (!camera.calibrated)
      return false;

  return true;
}

// enough frames to solve at all, and enough new ones since the last solve to
// plausibly change the answer
bool due_for_calibration(const CameraProgress& camera) {
  return !camera.calibrated
      && camera.accepted >= mocap::MIN_FRAMES
      && camera.accepted - camera.last_attempt >= RECALIBRATE_EVERY;
}

void attempt_calibration(mocap::LensCalibration& calibrator,
                         CameraProgress& camera,
                         const std::string& name) {
  camera.last_attempt = camera.accepted;
  camera.error = calibrator.calibrate();

  if (!calibrator.check_status()) {
    std::println("[{}] {} frames, reprojection error {:.4f} px, still above {:.1f}",
                 name, camera.accepted, camera.error, mocap::MIN_ERR);
    return;
  }

  camera.calibrated = true;
  std::println("[{}] calibrated: {} frames, reprojection error {:.4f} px",
               name, camera.accepted, camera.error);
}

// runs until every camera meets MIN_ERR, or the operator presses escape
// every camera gets its own window so the operator can see what each one sees
// while positioning the board. returns false if the operator aborted.
bool show(const cv::Mat& image, const std::string& window) {
  cv::imshow(window, image);
  return cv::waitKey(1) != ESC_KEY;
}

void collect_until_calibrated(mocap::Session& session,
                              std::vector<mocap::LensCalibration>& calibrators,
                              std::vector<CameraProgress>& progress,
                              const std::vector<mocap::Camera>& cameras) {
  cv::Mat gray;
  cv::Mat preview;
  bool running = true;

  while (running && !all_calibrated(progress)) {
    std::optional<mocap::Frameset> set = session.try_acquire_frameset();
    if (!set) {
      std::this_thread::sleep_for(POLL_INTERVAL);
      continue;
    }

    for (const mocap::FrameView& view : set->frames) {
      CameraProgress& camera = progress[view.camera_id];
      if (camera.calibrated)
        continue;

      if (!mocap::copy_gray_to_host(view, gray))
        continue;

      const bool found = calibrators[view.camera_id].try_frame(gray);

      cv::cvtColor(gray, preview, cv::COLOR_GRAY2BGR);
      if (found)
        calibrators[view.camera_id].draw_corners(preview);

      if (!show(preview, cameras[view.camera_id].name))
        running = false;

      if (!found)
        continue;

      camera.accepted += 1;

      if (due_for_calibration(camera))
        attempt_calibration(calibrators[view.camera_id], camera, cameras[view.camera_id].name);
    }

    session.release_frameset(*set);
  }
}

void save_all(std::vector<mocap::LensCalibration>& calibrators,
              const std::vector<CameraProgress>& progress,
              const std::vector<mocap::Camera>& cameras) {
  for (size_t i = 0; i < cameras.size(); i += 1) {
    if (!progress[i].calibrated) {
      std::println("{}: not calibrated, {} frames accepted, nothing saved",
                   cameras[i].name, progress[i].accepted);
      continue;
    }

    std::string filename = cameras[i].name + "_calibration.yaml";
    if (!calibrators[i].save_params(filename)) {
      std::println(stderr, "{}: could not write {}", cameras[i].name, filename);
      continue;
    }

    std::println("{}: {:.4f} px from {} frames -> {}",
                 cameras[i].name, progress[i].error, progress[i].accepted, filename);
  }
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

  std::vector<mocap::LensCalibration> calibrators;
  std::vector<CameraProgress> progress(conf.cameras.size());
  calibrators.reserve(conf.cameras.size());
  for (size_t i = 0; i < conf.cameras.size(); i += 1)
    calibrators.emplace_back(
      static_cast<int>(conf.stream.frame_width),
      static_cast<int>(conf.stream.frame_height),
      BOARD_WIDTH, BOARD_HEIGHT, SQUARE_SIZE
    );

  std::println("hold the {}x{} board in view of each camera",
               BOARD_WIDTH, BOARD_HEIGHT);
  std::println("each needs at least {} views and under {:.1f} px error",
               mocap::MIN_FRAMES, mocap::MIN_ERR);
  std::println("escape to stop early\n");

  collect_until_calibrated(*session, calibrators, progress, conf.cameras);

  cv::destroyAllWindows();

  std::println("");
  save_all(calibrators, progress, conf.cameras);

  return 0;
}
