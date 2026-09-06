#include "calibration_params.hpp"

namespace mocap {

bool load_calibration_params(
  const std::string& filename,
  calibration_params& params
) {
  cv::FileStorage fs(filename, cv::FileStorage::READ);

  if (!fs.isOpened())
    return false;

  fs["cam_matrix"] >> params.cam_matrix;
  fs["dist_coeffs"] >> params.dist_coeffs;

  fs.release();
  return true;
}

bool load_stereo_params(
  const std::string& filename,
  stereo_params& params
) {
  cv::FileStorage fs(filename, cv::FileStorage::READ);

  if (!fs.isOpened())
    return false;

  fs["rotation_matrix"] >> params.rotation;
  fs["translation_matrix"] >> params.translation;

  fs.release();
  return true;
}

} // namespace mocap
