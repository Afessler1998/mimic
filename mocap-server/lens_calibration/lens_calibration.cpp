#include <cstdint>
#include <opencv2/opencv.hpp>
#include <vector>

#include <algorithm>
#include <cmath>

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

const char* describe(CalibrationStatus status) {
  switch (status) {
    case CalibrationStatus::Ok:                      return "ok";
    case CalibrationStatus::TooFewFrames:            return "too few views";
    case CalibrationStatus::HighError:               return "reprojection error too high";
    case CalibrationStatus::PrincipalPointOffCenter: return "principal point off the sensor center";
    case CalibrationStatus::FocalLengthsDisagree:    return "fx and fy disagree";
  }

  return "unknown";
}

// Reprojection error only says the solution fits the views it was fit to. Views
// that all share a pose let focal length and board distance trade off against
// each other, so a whole family of solutions fits them equally well and the
// optimizer returns whichever one it wandered into. These two checks ask the
// separate question of whether the answer describes a camera that could exist.
CalibrationStatus LensCalibration::status() const {
  if (frame_count < MIN_FRAMES)
    return CalibrationStatus::TooFewFrames;

  if (reprojection_err >= MIN_ERR)
    return CalibrationStatus::HighError;

  const double fx = cam_matrix.at<double>(0, 0);
  const double fy = cam_matrix.at<double>(1, 1);
  const double cx = cam_matrix.at<double>(0, 2);
  const double cy = cam_matrix.at<double>(1, 2);

  if (std::abs(cx - frame_width / 2.0) > MAX_CENTER_OFFSET * frame_width ||
      std::abs(cy - frame_height / 2.0) > MAX_CENTER_OFFSET * frame_height)
    return CalibrationStatus::PrincipalPointOffCenter;

  if (std::abs(fx - fy) > MAX_FOCAL_SKEW * std::max(fx, fy))
    return CalibrationStatus::FocalLengthsDisagree;

  return CalibrationStatus::Ok;
}

bool LensCalibration::check_status() {
  return status() == CalibrationStatus::Ok;
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
