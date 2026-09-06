#include <opencv2/opencv.hpp>
#include <vector>

#include "calibration_params.hpp"
#include "stereo_calibration.hpp"

namespace mocap {

StereoCalibration::StereoCalibration(
  struct calibration_params* calib_params,
  int cam_count,
  int frame_width,
  int frame_height,
  int board_width,
  int board_height,
  float square_size
) :
  cam_count(cam_count),
  frame_width(frame_width),
  frame_height(frame_height),
  board_width(board_width),
  board_height(board_height),
  square_size(square_size) {

  objp.reserve(board_width * board_height);
  for (int i = 0; i < board_height; i += 1) {
    for (int j = 0; j < board_width; j += 1)
      objp.push_back(cv::Point3f(j*square_size, i*square_size, 0));
  }

  stereo_pairs.reserve(cam_count * (cam_count - 1) / 2);

  for (int i = 0; i < cam_count - 1; i += 1) {
    for (int j = i + 1; j < cam_count; j += 1) {
      struct stereo_pair pair;
      pair.cam1 = i;
      pair.cam2 = j;
      pair.img_pts1.reserve(MIN_FRAMES);
      pair.img_pts2.reserve(MIN_FRAMES);

      pair.cam1_matrix = calib_params[i].cam_matrix;
      pair.cam2_matrix = calib_params[j].cam_matrix;
      pair.cam1_dist_coeffs = calib_params[i].dist_coeffs;
      pair.cam2_dist_coeffs = calib_params[j].dist_coeffs;

      stereo_pairs.push_back(std::move(pair));
    }
  }
}

// one camera's frame: locate the board and refine to subpixel accuracy
bool StereoCalibration::find_board(const cv::Mat& frame,
                                   std::vector<cv::Point2f>& corners) const {
  auto flags = cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK;

  if (!cv::findChessboardCorners(frame, cv::Size(board_width, board_height), corners, flags))
    return false;

  cv::TermCriteria criteria(
    cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER,
    30,
    0.001
  );
  cv::cornerSubPix(frame, corners, cv::Size(11, 11), cv::Size(-1, -1), criteria);

  return true;
}

StereoCalibration::BoardDetections StereoCalibration::find_boards(cv::Mat* frames) const {
  BoardDetections detections{
    std::vector<bool>(cam_count, false),
    std::vector<std::vector<cv::Point2f>>(cam_count)
  };

  for (int i = 0; i < cam_count; i += 1)
    detections.found[i] = find_board(frames[i], detections.corners[i]);

  return detections;
}

// a pair only learns something when both of its cameras saw the board
void StereoCalibration::record_pairs(const BoardDetections& detections) {
  for (struct stereo_pair& pair : stereo_pairs) {
    if (!(detections.found[pair.cam1] && detections.found[pair.cam2]))
      continue;

    pair.img_pts1.push_back(detections.corners[pair.cam1]);
    pair.img_pts2.push_back(detections.corners[pair.cam2]);
  }
}

void StereoCalibration::try_frames(cv::Mat* frames) {
  const BoardDetections detections = find_boards(frames);
  record_pairs(detections);
}

std::vector<PairProgress> StereoCalibration::progress() const {
  std::vector<PairProgress> counts;
  counts.reserve(stereo_pairs.size());

  for (const struct stereo_pair& pair : stereo_pairs)
    counts.push_back(PairProgress{pair.cam1, pair.cam2, pair.img_pts1.size()});

  return counts;
}

// a pair with no shared views is one the board was never held between, which
// is allowed. a pair with too few is one that was attempted and is not
// trustworthy yet, which is not.
bool StereoCalibration::check_status() const {
  bool any_observed = false;

  for (const struct stereo_pair& pair : stereo_pairs) {
    size_t frames = pair.img_pts1.size();
    if (frames == 0)
      continue;

    if (frames <= MIN_FRAMES)
      return false;

    any_observed = true;
  }

  return any_observed;
}

// intrinsics are fixed: they came from lens calibration, and re solving them
// here from fewer views would be worse than what is already on disk
void StereoCalibration::calibrate() {
  for (struct stereo_pair& pair : stereo_pairs) {
    if (pair.img_pts1.empty())
      continue;

    cv::stereoCalibrate(
      std::vector<std::vector<cv::Point3f>>(pair.img_pts1.size(), objp),
      pair.img_pts1,
      pair.img_pts2,
      pair.cam1_matrix,
      pair.cam1_dist_coeffs,
      pair.cam2_matrix,
      pair.cam2_dist_coeffs,
      cv::Size(frame_width, frame_height),
      pair.rotation_mat,
      pair.translation_mat,
      cv::noArray(), // essential matrix
      cv::noArray(), // fundamental matrix
      cv::CALIB_FIX_INTRINSIC
    );
  }
}

void StereoCalibration::save_params(const std::vector<Camera>& cameras) {
  for (const struct stereo_pair& pair : stereo_pairs) {
    if (pair.img_pts1.empty())
      continue;

    std::string filename =
      cameras[pair.cam1].name + "_" + cameras[pair.cam2].name + "_calibration.yaml";
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    fs << "rotation_matrix" << pair.rotation_mat;
    fs << "translation_matrix" << pair.translation_mat;

    fs.release();
  }
}

} // namespace mocap
