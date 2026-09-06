#ifndef MOCAP_LENS_CALIBRATION_HPP
#define MOCAP_LENS_CALIBRATION_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include <vector>

#include "calibration_params.hpp"

namespace mocap {

constexpr double MIN_ERR = 1.0;

// The optical axis pierces the sensor near its center, within how the lens is
// mounted over the chip, so a principal point further out than this is not a
// camera that exists.
constexpr double MAX_CENTER_OFFSET = 0.15;

// fx and fy are the same focal length divided by the pixel pitch in each
// direction, and the sensor's pixels are square, so they describe one number
// twice. Anything further apart than this is the solve drifting, not the
// hardware.
constexpr double MAX_FOCAL_SKEW = 0.02;

// Why a solve was rejected, so the operator can tell a camera that needs more
// views from one that needs the board held differently.
enum class CalibrationStatus {
  Ok,
  TooFewFrames,
  HighError,
  PrincipalPointOffCenter,
  FocalLengthsDisagree
};

const char* describe(CalibrationStatus status);

class LensCalibration {
private:
  int frame_width;
  int frame_height;
  int board_width;
  int board_height;
  float square_size;

  int frame_count;

  std::vector<cv::Point3f> objp;
  std::vector<cv::Point2f> corners;
  std::vector<std::vector<cv::Point2f>> img_pts;
  std::vector<std::vector<cv::Point3f>> obj_pts;

  cv::Mat cam_matrix;
  cv::Mat dist_coeffs;

  double reprojection_err;

public:
  LensCalibration(
    int frame_width,
    int frame_height,
    int board_width,
    int board_height,
    float square_size
  );
  bool try_frame(cv::Mat& gray_frame);
  void draw_corners(cv::Mat& bgr_frame) const;
  double calibrate();
  CalibrationStatus status() const;
  bool check_status();
  bool save_params(const std::string& filename) const;
};

} // namespace mocap

#endif // MOCAP_LENS_CALIBRATION_HPP
