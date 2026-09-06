#ifndef MOCAP_LENS_CALIBRATION_HPP
#define MOCAP_LENS_CALIBRATION_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include <vector>

#include "calibration_params.hpp"

namespace mocap {

constexpr double MIN_ERR = 1.0;

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
  bool check_status();
  bool save_params(const std::string& filename) const;
};

} // namespace mocap

#endif // MOCAP_LENS_CALIBRATION_HPP
