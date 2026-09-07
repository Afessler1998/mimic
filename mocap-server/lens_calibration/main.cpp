#include <algorithm>
#include <chrono>
#include <optional>
#include <print>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "copy_gray_to_host.hpp"
#include "lens_calibration.hpp"
#include "preview.hpp"
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

std::vector<mocap::CameraCalibrationContext> make_camera_contexts(const mocap::Config& conf) {
  std::vector<mocap::CameraCalibrationContext> contexts;
  contexts.reserve(conf.cameras.size());

  for (const mocap::Camera& camera : conf.cameras)
    contexts.push_back(mocap::CameraCalibrationContext{
      camera,
      mocap::LensCalibration(
        static_cast<int>(conf.stream.frame_width),
        static_cast<int>(conf.stream.frame_height),
        BOARD_WIDTH, BOARD_HEIGHT, SQUARE_SIZE
      ),
      mocap::CameraProgress{}
    });

  return contexts;
}

bool all_calibrated(const std::vector<mocap::CameraCalibrationContext>& contexts) {
  for (const mocap::CameraCalibrationContext& context : contexts)
    if (!context.progress.calibrated)
      return false;

  return true;
}

// enough frames to solve at all, and enough new ones since the last solve to
// plausibly change the answer
bool is_due_for_calibration(const mocap::CameraCalibrationContext& context) {
  return !context.progress.calibrated
      && context.progress.accepted >= mocap::MIN_LENS_FRAMES
      && context.progress.accepted - context.progress.last_attempt >= RECALIBRATE_EVERY;
}

// A high reprojection error means something different depending on how it is
// distributed: a median near the RMS is a floor under every view, where a
// median well below it is a handful of bad ones dragging the average. The index
// of the worst says whether it was an early view or a recent one.
void report_view_errors(const cv::Mat& view_errors, const std::string& name) {
  if (view_errors.empty())
    return;

  std::vector<double> sorted;
  sorted.reserve(view_errors.total());

  size_t worst = 0;
  for (size_t i = 0; i < view_errors.total(); i += 1) {
    const double error = view_errors.at<double>(static_cast<int>(i));
    sorted.push_back(error);
    if (error > view_errors.at<double>(static_cast<int>(worst)))
      worst = i;
  }

  std::sort(sorted.begin(), sorted.end());
  std::println("[{}] per view error: median {:.3f}, worst {:.3f} (view {}), best {:.3f}",
               name, sorted[sorted.size() / 2], sorted.back(), worst, sorted.front());
}

void solve_for_camera_intrinsics(mocap::CameraCalibrationContext& context) {
  const std::string& name = context.camera.name;

  std::println("\n[{}] solving on {} views", name, context.progress.accepted);

  context.progress.last_attempt = context.progress.accepted;
  context.progress.error = context.calibrator.calibrate();

  report_view_errors(context.calibrator.per_view_errors(), name);

  if (context.calibrator.discarded() > 0)
    std::println("[{}] dropped {} outlier view(s) and solved again",
                 name, context.calibrator.discarded());

  const mocap::CalibrationStatus status = context.calibrator.status();
  if (status != mocap::CalibrationStatus::Ok) {
    std::println("[{}] {} frames, reprojection error {:.4f} px, rejected: {}",
                 name, context.progress.accepted, context.progress.error,
                 mocap::describe(status));
    return;
  }

  context.progress.calibrated = true;
  std::println("[{}] calibrated: {} frames, reprojection error {:.4f} px",
               name, context.progress.accepted, context.progress.error);
}

void solve_due_cameras(std::vector<mocap::CameraCalibrationContext>& contexts) {
  for (mocap::CameraCalibrationContext& context : contexts)
    if (is_due_for_calibration(context))
      solve_for_camera_intrinsics(context);
}

// try_frame is the expensive call in this loop and it is slowest when it finds
// nothing, so a camera that is finished is never searched again. Its frames
// still reach the preview, because a pane that stops updating reads as the tool
// having hung.
void consider_frame_for_calibration(mocap::CameraCalibrationContext& context, cv::Mat& gray) {
  if (context.progress.calibrated)
    return;

  context.progress.last_result = context.calibrator.try_bank_view(gray);
  if (context.progress.last_result != mocap::FrameResult::Accepted)
    return;

  context.progress.accepted += 1;
  std::println("\n[{}] view {}: sharpness {:.0f}, corner spacing {:.1f} px",
               context.camera.name, context.progress.accepted,
               context.calibrator.last_sharpness(), context.calibrator.last_spacing());
}

void collect_until_calibrated(mocap::Session& session,
                              std::vector<mocap::CameraCalibrationContext>& contexts,
                              std::vector<cv::Mat>& frames,
                              const std::vector<mocap::PreviewPane>& panes) {
  std::vector<bool> has_new_frame(contexts.size());

  while (!all_calibrated(contexts)) {
    std::optional<mocap::Frameset> set = session.try_acquire_frameset();
    if (!set) {
      std::this_thread::sleep_for(POLL_INTERVAL);
      continue;
    }

    // The frameset is handed back as soon as its surfaces have been copied, so
    // nothing slow runs while a decoder surface per camera is still held: that
    // is what back pressures into the cameras.
    std::fill(has_new_frame.begin(), has_new_frame.end(), false);
    for (const mocap::FrameView& view : set->frames)
      has_new_frame[view.camera_id] = mocap::copy_gray_to_host(view, frames[view.camera_id]);

    session.release_frameset(*set);

    for (mocap::CameraCalibrationContext& context : contexts)
      if (has_new_frame[context.camera.id])
        consider_frame_for_calibration(context, frames[context.camera.id]);

    if (!show_previews(panes))
      return;

    solve_due_cameras(contexts);
  }
}

void save_all(const std::vector<mocap::CameraCalibrationContext>& contexts) {
  for (const mocap::CameraCalibrationContext& context : contexts) {
    if (!context.progress.calibrated) {
      std::println("{}: not calibrated, {} frames accepted, nothing saved",
                   context.camera.name, context.progress.accepted);
      continue;
    }

    const std::string filename = context.camera.name + "_calibration.yaml";
    if (!context.calibrator.save_params(filename)) {
      std::println(stderr, "{}: could not write {}", context.camera.name, filename);
      continue;
    }

    std::println("{}: {:.4f} px from {} frames -> {}", context.camera.name,
                 context.progress.error, context.progress.accepted, filename);
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

  std::vector<mocap::CameraCalibrationContext> contexts = make_camera_contexts(conf);
  std::vector<cv::Mat> frames(contexts.size());
  const std::vector<mocap::PreviewPane> panes = make_preview_panes(contexts, frames);

  collect_until_calibrated(*session, contexts, frames, panes);

  cv::destroyAllWindows();
  save_all(contexts);

  return 0;
}
