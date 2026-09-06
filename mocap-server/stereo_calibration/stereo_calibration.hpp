#ifndef MOCAP_STEREO_CALIBRATION_HPP
#define MOCAP_STEREO_CALIBRATION_HPP

#include <array>
#include <opencv2/opencv.hpp>
#include <vector>

#include <string>
#include "calibration_params.hpp"
#include "config.hpp"

namespace mocap {

struct stereo_pair {
  // which cameras this pair relates, so the set can be walked flat instead of
  // reconstructing a triangular index from a nested loop
  int cam1;
  int cam2;

  std::vector<std::vector<cv::Point2f>> img_pts1;
  std::vector<std::vector<cv::Point2f>> img_pts2;
  cv::Mat cam1_matrix;
  cv::Mat cam1_dist_coeffs;
  cv::Mat cam2_matrix;
  cv::Mat cam2_dist_coeffs;
  cv::Mat rotation_mat;
  cv::Mat translation_mat;
};

// how many shared views each pair has gathered, so a tool can say which pairs
// still need the board held between them
struct PairProgress {
  int cam1;
  int cam2;
  size_t views;
};

class StereoCalibration {
private:
  // which cameras saw the board this frame, and where its corners landed
  struct BoardDetections {
    std::vector<bool> found;
    std::vector<std::vector<cv::Point2f>> corners;
  };

  std::vector<cv::Point3f> objp;

  std::vector<stereo_pair> stereo_pairs;
  int cam_count;
  int frame_width;
  int frame_height;
  int board_width;
  int board_height;
  float square_size;

public:
  StereoCalibration(
    struct calibration_params* calib_params,
    int cam_count,
    int frame_width,
    int frame_height,
    int board_width,
    int board_height,
    float square_size
  );

  void try_frames(cv::Mat* frames);

private:
  bool find_board(const cv::Mat& frame, std::vector<cv::Point2f>& corners) const;
  BoardDetections find_boards(cv::Mat* frames) const;
  void record_pairs(const BoardDetections& detections);

public:
  std::vector<PairProgress> progress() const;
  bool check_status() const;
  void calibrate();
  void save_params(const std::vector<Camera>& cameras);
};

} // namespace mocap

#endif // MOCAP_STEREO_CALIBRATION_HPP
