#ifndef MOCAP_CALIBRATION_PARAMS_HPP
#define MOCAP_CALIBRATION_PARAMS_HPP

#include <string>

#include <opencv2/opencv.hpp>

namespace mocap {

// fewest board views a calibration can be solved from, for intrinsics and for
// a stereo pair alike
constexpr int MIN_FRAMES = 10;

// Reading calibration files is shared: every tool downstream of calibration
// consumes them. Writing is not, so each calibration tool owns the format it
// produces.

// One camera's intrinsics, from <camera>_calibration.yaml
struct calibration_params {
  cv::Mat cam_matrix;
  cv::Mat dist_coeffs;
};

// One pair's extrinsics, from <camera1>_<camera2>_calibration.yaml
struct stereo_params {
  cv::Mat rotation;
  cv::Mat translation;
};

bool load_calibration_params(
  const std::string& filename,
  calibration_params& params
);

bool load_stereo_params(
  const std::string& filename,
  stereo_params& params
);

} // namespace mocap

#endif // MOCAP_CALIBRATION_PARAMS_HPP
