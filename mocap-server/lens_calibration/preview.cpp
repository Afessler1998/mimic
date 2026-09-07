#include <format>

#include "preview.hpp"

namespace mocap {

namespace {

// The cameras are mounted on their side, so the panes are portrait: the frame
// is rotated upright for display only, and everything the solve reads stays in
// source coordinates.
constexpr int PANE_WIDTH = 360;
constexpr int PANE_HEIGHT = 640;
constexpr int ESC_KEY = 27;

// The coverage grid, shaded over the camera's own view. This is the same data
// the acceptance test reads, so what the operator is being told to do and what
// the tool will accept cannot drift apart: a red cell is one that will take a
// view, a green cell is one that will not.
void draw_coverage(cv::Mat& bgr, const std::vector<int>& coverage) {
  cv::Mat shade = bgr.clone();

  const int cell_width = bgr.cols / COVERAGE_COLS;
  const int cell_height = bgr.rows / COVERAGE_ROWS;

  for (int row = 0; row < COVERAGE_ROWS; row += 1) {
    for (int col = 0; col < COVERAGE_COLS; col += 1) {
      const cv::Rect cell(col * cell_width, row * cell_height,
                          cell_width, cell_height);
      const bool covered =
        coverage[row * COVERAGE_COLS + col] >= VIEWS_PER_CELL;

      cv::rectangle(shade, cell,
                    covered ? cv::Scalar(0, 160, 0) : cv::Scalar(0, 0, 160),
                    cv::FILLED);
      cv::rectangle(bgr, cell, cv::Scalar(60, 60, 60), 1);
    }
  }

  cv::addWeighted(shade, 0.25, bgr, 0.75, 0.0, bgr);
}

// The tilt buckets, as a row of bars under the label. Same contract as the
// coverage grid: this is the data the acceptance test reads, so a bar that is
// not full is an angle the tool will still take a view at.
void draw_tilt(cv::Mat& bgr, const std::vector<int>& tilt) {
  constexpr int BAR_HEIGHT = 18;
  constexpr int MARGIN = 12;

  const int bar_width = (bgr.cols - 2 * MARGIN) / TILT_BUCKETS;
  const int top = bgr.rows - MARGIN - BAR_HEIGHT;

  for (int bucket = 0; bucket < TILT_BUCKETS; bucket += 1) {
    const cv::Rect bar(MARGIN + bucket * bar_width, top,
                       bar_width - 4, BAR_HEIGHT);
    const bool full = tilt[bucket] >= VIEWS_PER_TILT_BUCKET;

    cv::rectangle(bgr, bar,
                  full ? cv::Scalar(0, 200, 0) : cv::Scalar(0, 0, 200),
                  cv::FILLED);
    cv::rectangle(bgr, bar, cv::Scalar(255, 255, 255), 1);
  }

  cv::putText(bgr, "flat        tilt        more", cv::Point(MARGIN, top - 8),
              cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1);
}

// One instruction at a time. The bars and the grid say what has been banked,
// which is the wrong thing to read while holding a board: what the operator
// needs is the single next thing to do. Tilt outranks position because a
// missing angle is what makes a calibration wrong, where a missing cell only
// makes it weaker.
std::string next_action(const LensCalibration& calibrator,
                        const CameraProgress& camera) {
  if (camera.calibrated)
    return "done";

  if (camera.last_result == FrameResult::TooBlurry)
    return "hold still";

  if (camera.last_result == FrameResult::NoBoard)
    return "show the board";

  if (!calibrator.tilt_complete())
    return "tilt the board";

  if (!calibrator.coverage_complete())
    return "move to a red cell";

  return "keep going";
}

void draw_action(cv::Mat& bgr, const std::string& text, bool satisfied) {
  const cv::Scalar color = satisfied ? cv::Scalar(0, 220, 0)
                                     : cv::Scalar(0, 200, 255);

  int baseline = 0;
  const cv::Size size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.1, 3,
                                        &baseline);
  const cv::Point origin((bgr.cols - size.width) / 2, bgr.rows - 70);

  cv::putText(bgr, text, origin, cv::FONT_HERSHEY_SIMPLEX, 1.1,
              cv::Scalar(0, 0, 0), 7);
  cv::putText(bgr, text, origin, cv::FONT_HERSHEY_SIMPLEX, 1.1, color, 3);
}

void draw_label(cv::Mat& bgr, const std::string& text) {
  cv::putText(bgr, text, cv::Point(12, 40), cv::FONT_HERSHEY_SIMPLEX, 1.2,
              cv::Scalar(0, 0, 0), 6);
  cv::putText(bgr, text, cv::Point(12, 40), cv::FONT_HERSHEY_SIMPLEX, 1.2,
              cv::Scalar(255, 255, 255), 2);
}


void draw_pane(cv::Mat& pane, const PreviewPane& source) {
  // the grid and the corners are drawn in source coordinates, so they rotate
  // along with the image and keep meaning what they meant
  cv::Mat overlaid;
  cv::cvtColor(*source.frame, overlaid, cv::COLOR_GRAY2BGR);
  draw_coverage(overlaid, source.calibrator->coverage_map());

  if (source.progress->last_result != FrameResult::NoBoard)
    source.calibrator->draw_corners(overlaid);

  cv::Mat upright;
  cv::rotate(overlaid, upright, cv::ROTATE_90_COUNTERCLOCKWISE);

  // the bars and the text go on after the rotation, or they would be sideways
  draw_tilt(upright, source.calibrator->tilt_map());
  draw_action(upright, next_action(*source.calibrator, *source.progress),
              source.progress->calibrated);
  draw_label(upright, std::format("{} {}/{}", source.name,
                                  source.progress->accepted, MIN_LENS_FRAMES));

  cv::resize(upright, pane, cv::Size(PANE_WIDTH, PANE_HEIGHT));
}

} // namespace

std::vector<PreviewPane> make_preview_panes(
    const std::vector<CameraCalibrationContext>& contexts,
    const std::vector<cv::Mat>& frames) {
  std::vector<PreviewPane> panes;
  panes.reserve(contexts.size());

  // frames are indexed by camera id, the same way the acquisition loop fills
  // them, so a config whose ids are not in order still lines up
  for (const CameraCalibrationContext& context : contexts)
    panes.push_back(PreviewPane{
      context.camera.name, &frames[context.camera.id], &context.calibrator,
      &context.progress
    });

  return panes;
}

// Every camera gets a pane in one window rather than a window of its own. A
// tiling window manager stacks separate windows, which leaves only one camera
// visible, and the whole point of the preview is seeing all of them at once.
bool show_previews(const std::vector<PreviewPane>& panes) {
  static cv::Mat mosaic;

  const int width = PANE_WIDTH * static_cast<int>(panes.size());
  if (mosaic.cols != width)
    mosaic.create(PANE_HEIGHT, width, CV_8UC3);

  for (size_t i = 0; i < panes.size(); i += 1) {
    if (panes[i].frame->empty())
      continue;

    cv::Mat pane = mosaic(cv::Rect(static_cast<int>(i) * PANE_WIDTH, 0,
                                   PANE_WIDTH, PANE_HEIGHT));
    draw_pane(pane, panes[i]);
  }

  cv::imshow("lens calibration", mosaic);

  return cv::waitKey(1) != ESC_KEY;
}

} // namespace mocap
