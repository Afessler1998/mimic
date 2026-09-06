#include <cstdint>
#include <opencv2/opencv.hpp>
#include <vector>

#include "lens_calibration.hpp"

namespace mocap {

LensCalibration::LensCalibration(
  int frame_width,
  int frame_height,
  int board_width,
  int board_height,
  float square_size
) :
  frame_width(frame_width),
  frame_height(frame_height),
  board_width(board_width),
  board_height(board_height),
  square_size(square_size),
  frame_count(0),
  reprojection_err(-1.0) {

  objp.reserve(board_width * board_height);
  for (int i = 0; i < board_height; i += 1) {
    for (int j = 0; j < board_width; j += 1)
      objp.push_back(cv::Point3f(j*square_size, i*square_size, 0));
  }

  corners.reserve(board_width * board_height);
  img_pts.reserve(MIN_FRAMES);
  obj_pts.reserve(MIN_FRAMES);
}

bool LensCalibration::try_frame(cv::Mat& gray_frame) {
  cv::Size board_size(board_width, board_height);
  corners.clear();

  bool found = cv::findChessboardCorners(
    gray_frame,
    board_size,
    corners,
    cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK
  );

  if (!found) return false;

  cv::TermCriteria criteria(
    cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
    30,
    0.001
  );
  cv::cornerSubPix(
    gray_frame,
    corners,
    cv::Size(11,11),
    cv::Size(-1,-1),
    criteria
  );

  img_pts.push_back(corners);
  obj_pts.push_back(objp);
  frame_count += 1;

  return true;
}

void LensCalibration::draw_corners(cv::Mat& bgr_frame) const {
  cv::drawChessboardCorners(
    bgr_frame,
    cv::Size(board_width, board_height),
    img_pts[frame_count - 1],
    true
  );
}

double LensCalibration::calibrate() {
  if (frame_count < MIN_FRAMES) return -1.0;

  cv::Size img_size(frame_width, frame_height);

  cam_matrix = cv::Mat::eye(3, 3, CV_64F);
  dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);

  // calibrateCamera returns RMS reprojection error directly. the previous
  // version recomputed it with projectPoints, which lives in OpenCV's 3d
  // module and is not built here, and computed mean squared error rather than
  // RMS. RMS is the conventional metric and MIN_ERR is in pixels.
  reprojection_err = cv::calibrateCamera(
    obj_pts,
    img_pts,
    img_size,
    cam_matrix,
    dist_coeffs,
    cv::noArray(),   // per view rotations, not used
    cv::noArray()    // per view translations, not used
  );

  return reprojection_err;
}

bool LensCalibration::check_status() {
  if (frame_count < MIN_FRAMES) return false;
  return reprojection_err < 1.0;
}

// image size, reproj_err and images_used are provenance: they record what this
// was solved from, and nothing reads them back
bool LensCalibration::save_params(const std::string& filename) const {
  cv::FileStorage fs(filename, cv::FileStorage::WRITE);

  if (!fs.isOpened())
    return false;

  fs << "cam_matrix" << cam_matrix;
  fs << "dist_coeffs" << dist_coeffs;

  fs << "image_width" << frame_width;
  fs << "image_height" << frame_height;
  fs << "reproj_err" << reprojection_err;
  fs << "images_used" << frame_count;

  fs.release();
  return true;
}


} // namespace mocap
