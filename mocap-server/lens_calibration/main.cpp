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

// The loop runs faster than anyone can move a board, so without this it banks
// ten copies of one pose in under a second. Ten views of the same pose leave
// focal length and board distance trading off against each other freely, which
// solves to a low reprojection error and physically wrong intrinsics. Only
// tilting the board between views breaks that tie.
constexpr std::chrono::milliseconds ACCEPT_COOLDOWN{1500};
constexpr std::chrono::microseconds POLL_INTERVAL{500};
constexpr int ESC_KEY = 27;

struct CameraProgress {
  int accepted = 0;
  int last_attempt = 0;
  bool calibrated = false;
  double error = 0.0;

  // default constructs to the epoch, so the first view is never held back
  std::chrono::steady_clock::time_point last_accepted{};
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

  const mocap::CalibrationStatus status = calibrator.status();
  if (status != mocap::CalibrationStatus::Ok) {
    std::println("[{}] {} frames, reprojection error {:.4f} px, rejected: {}",
                 name, camera.accepted, camera.error, mocap::describe(status));
    return;
  }

  camera.calibrated = true;
  std::println("[{}] calibrated: {} frames, reprojection error {:.4f} px",
               name, camera.accepted, camera.error);
}

// how many views each camera has banked, so the operator can tell a camera
// that is not seeing the board from a tool that is not running
void print_progress(const std::vector<CameraProgress>& progress,
                    const std::vector<mocap::Camera>& cameras) {
  for (size_t i = 0; i < cameras.size(); i += 1)
    std::print("{} {}/{}{}   ", cameras[i].name, progress[i].accepted,
               mocap::MIN_FRAMES, progress[i].calibrated ? " done" : "");

  std::print("\r");
  std::fflush(stdout);
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
  bool pending_solve = false;

  while (running && !all_calibrated(progress)) {
    std::optional<mocap::Frameset> set = session.try_acquire_frameset();
    if (!set) {
      std::this_thread::sleep_for(POLL_INTERVAL);
      continue;
    }

    for (const mocap::FrameView& view : set->frames) {
      CameraProgress& camera = progress[view.camera_id];

      if (!mocap::copy_gray_to_host(view, gray))
        continue;

      // A camera that is done, or waiting out its cooldown, still gets drawn.
      // Its window is the only way to see where the board is while walking it
      // around, and a window that stops updating reads as the tool having hung.
      // Skipping the search is also what keeps the preview responsive, since
      // findChessboardCorners is the expensive part of this loop.
      const std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now();
      const bool searching = !camera.calibrated
                          && now - camera.last_accepted >= ACCEPT_COOLDOWN;
      const bool found = searching && calibrators[view.camera_id].try_frame(gray);

      cv::cvtColor(gray, preview, cv::COLOR_GRAY2BGR);
      if (found)
        calibrators[view.camera_id].draw_corners(preview);

      if (!show(preview, cameras[view.camera_id].name))
        running = false;

      if (!found)
        continue;

      camera.accepted += 1;
      camera.last_accepted = now;
      pending_solve = pending_solve || due_for_calibration(camera);
    }

    // The frameset is handed back before solving, because calibrateCamera
    // takes seconds and holding a decoder surface across it back pressures
    // into the cameras for no reason.
    session.release_frameset(*set);

    print_progress(progress, cameras);

    if (!pending_solve)
      continue;

    pending_solve = false;
    for (size_t i = 0; i < cameras.size(); i += 1) {
      if (!due_for_calibration(progress[i]))
        continue;

      // the solve blocks this thread, so the windows stop repainting for its
      // duration. saying so beats looking hung.
      std::println("\n[{}] solving on {} views", cameras[i].name, progress[i].accepted);
      attempt_calibration(calibrators[i], progress[i], cameras[i].name);
    }
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
  std::println("move the board between views: tilt it, and cover the corners of");
  std::println("the frame, not just the middle");
  std::println("escape to stop early\n");

  collect_until_calibrated(*session, calibrators, progress, conf.cameras);

  // the windows go away here, so say why before they do
  std::println("\ndone collecting, closing previews");
  cv::destroyAllWindows();

  std::println("");
  save_all(calibrators, progress, conf.cameras);

  return 0;
}
