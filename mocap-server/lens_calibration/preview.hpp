#ifndef MOCAP_LENS_PREVIEW_HPP
#define MOCAP_LENS_PREVIEW_HPP

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "lens_calibration.hpp"

namespace mocap {

// Everything one pane draws, by reference: the tool owns this state and the
// preview only reads it. Taking the pieces rather than the tool's own bundle
// keeps the dependency pointing one way, from display toward calibration.
//
// All of this is scaffolding around OpenCV's highgui and goes when the real
// renderer lands. It is behind one entry point so that swap is one call site.
struct PreviewPane {
  std::string name;
  const cv::Mat* frame;
  const LensCalibration* calibrator;
  const CameraProgress* progress;
};

// The panes point into the contexts and the frames, so this is built once and
// reused every frame. Resizing either afterwards would dangle.
std::vector<PreviewPane> make_preview_panes(
  const std::vector<CameraCalibrationContext>& contexts,
  const std::vector<cv::Mat>& frames
);

// Draws every pane into one window. Returns false if the operator asked to stop.
bool show_previews(const std::vector<PreviewPane>& panes);

} // namespace mocap

#endif // MOCAP_LENS_PREVIEW_HPP
