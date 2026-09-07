#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/geometry/2d.hpp>

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
  frame_count(0),
  reprojection_err(-1.0),
  sharpness(0.0),
  spacing(0.0),
  discard_count(0),
  bins(COVERAGE_COLS * COVERAGE_ROWS * TILT_BUCKETS, 0),
  coverage(COVERAGE_COLS * COVERAGE_ROWS, 0),
  tilt(TILT_BUCKETS, 0) {

  objp.reserve(board_width * board_height);
  for (int i = 0; i < board_height; i += 1) {
    for (int j = 0; j < board_width; j += 1)
      objp.push_back(cv::Point3f(j*square_size, i*square_size, 0));
  }

  corners.reserve(board_width * board_height);
  img_pts.reserve(MIN_LENS_FRAMES);
  obj_pts.reserve(MIN_LENS_FRAMES);
}

namespace {

// cv::Laplacian convolves a 3x3 kernel, so a region narrower than that has no
// interior to differentiate and its variance says nothing about focus.
constexpr int LAPLACIAN_KERNEL = 3;

// Variance of the Laplacian over a region. A checkerboard in focus is almost
// entirely edges, so the second derivative is large and spread out; blurring it
// pulls every neighbouring pixel toward its neighbours and the variance falls.
double measure_sharpness(const cv::Mat& gray_frame, const cv::Rect& region) {
  cv::Mat laplacian;
  cv::Laplacian(gray_frame(region), laplacian, CV_64F);

  cv::Scalar mean;
  cv::Scalar stddev;
  cv::meanStdDev(laplacian, mean, stddev);

  return stddev[0] * stddev[0];
}

// The board's bounding box, clipped to the frame, since cornerSubPix can place
// a corner marginally outside it.
cv::Rect find_board_bounds(const std::vector<cv::Point2f>& corners,
                      int frame_width, int frame_height) {
  return cv::boundingRect(corners) & cv::Rect(0, 0, frame_width, frame_height);
}

// The smallest gap between neighbouring corners along a board row. Under
// perspective the far end of a tilted board is compressed, so this is where the
// sub pixel search window has to fit.
double measure_min_corner_spacing(const std::vector<cv::Point2f>& corners,
                          int board_width, int board_height) {
  double smallest = std::numeric_limits<double>::max();

  for (int row = 0; row < board_height; row += 1) {
    for (int col = 0; col + 1 < board_width; col += 1) {
      const int index = row * board_width + col;
      smallest = std::min(smallest, cv::norm(corners[index + 1] - corners[index]));
    }
  }

  return smallest;
}

double compute_median(const cv::Mat& errors) {
  std::vector<double> sorted;
  sorted.reserve(errors.total());
  for (size_t i = 0; i < errors.total(); i += 1)
    sorted.push_back(errors.at<double>(static_cast<int>(i)));

  std::sort(sorted.begin(), sorted.end());
  return sorted[sorted.size() / 2];
}

// The interior angle at b, in degrees.
double measure_corner_angle(const cv::Point2f& a, const cv::Point2f& b,
                    const cv::Point2f& c) {
  const cv::Point2f first = a - b;
  const cv::Point2f second = c - b;

  const double dot = first.x * second.x + first.y * second.y;
  const double lengths = cv::norm(first) * cv::norm(second);
  if (lengths < 1e-9)
    return 90.0;

  return std::acos(std::clamp(dot / lengths, -1.0, 1.0)) * 180.0 / CV_PI;
}

} // namespace

int LensCalibration::find_coverage_cell(const std::vector<cv::Point2f>& points) const {
  cv::Point2f center(0.0f, 0.0f);
  for (const cv::Point2f& point : points)
    center += point;

  center /= static_cast<float>(points.size());

  const float cells_across = center.x * COVERAGE_COLS / frame_width;
  const float cells_down = center.y * COVERAGE_ROWS / frame_height;

  const int col = std::clamp(static_cast<int>(cells_across), 0, COVERAGE_COLS - 1);
  const int row = std::clamp(static_cast<int>(cells_down), 0, COVERAGE_ROWS - 1);

  return row * COVERAGE_COLS + col;
}

// The board is a rectangle, so its corners are square when it faces the camera
// and depart from square as it turns away. The largest departure across the
// four corners is a single number for how oblique the view is, and it needs no
// pose solve to compute.
int LensCalibration::find_tilt_bucket(const std::vector<cv::Point2f>& points) const {
  const cv::Point2f& top_left = points[0];
  const cv::Point2f& top_right = points[board_width - 1];
  const cv::Point2f& bottom_right = points[board_width * board_height - 1];
  const cv::Point2f& bottom_left = points[board_width * (board_height - 1)];

  double worst = 0.0;
  worst = std::max(worst, std::abs(90.0 - measure_corner_angle(bottom_left, top_left, top_right)));
  worst = std::max(worst, std::abs(90.0 - measure_corner_angle(top_left, top_right, bottom_right)));
  worst = std::max(worst, std::abs(90.0 - measure_corner_angle(top_right, bottom_right, bottom_left)));
  worst = std::max(worst, std::abs(90.0 - measure_corner_angle(bottom_right, bottom_left, top_left)));

  for (int bucket = 0; bucket < TILT_BUCKETS; bucket += 1)
    if (worst < TILT_BUCKET_LIMITS[bucket])
      return bucket;

  return TILT_BUCKETS - 1;
}

bool LensCalibration::tilt_complete() const {
  for (int bucket : tilt)
    if (bucket < VIEWS_PER_TILT_BUCKET)
      return false;

  return true;
}

bool LensCalibration::coverage_complete() const {
  for (int cell : coverage)
    if (cell < VIEWS_PER_CELL)
      return false;

  return true;
}

bool LensCalibration::detect_board(const cv::Mat& gray_frame) {
  corners.clear();

  return cv::findChessboardCorners(
    gray_frame,
    cv::Size(board_width, board_height),
    corners,
    cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK
  );
}

// The search window has to stay inside one square, or the refinement fits
// whatever structure the neighbouring squares put in it. A tilted board
// compresses its far end, so the window is sized from the tightest corner
// spacing in this particular view rather than fixed.
void LensCalibration::refine_corners(const cv::Mat& gray_frame) {
  spacing = measure_min_corner_spacing(corners, board_width, board_height);

  const int window = std::clamp(static_cast<int>(spacing / 2.0) - 1, 2, 11);

  cv::cornerSubPix(
    gray_frame,
    corners,
    cv::Size(window, window),
    cv::Size(-1, -1),
    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001)
  );
}

// Where this view falls in the joint histogram of place and angle.
int LensCalibration::find_bin(const std::vector<cv::Point2f>& points) const {
  return find_coverage_cell(points) * TILT_BUCKETS + find_tilt_bucket(points);
}

bool LensCalibration::covers_new_ground() const {
  return bins[find_bin(corners)] < VIEWS_PER_CELL;
}

void LensCalibration::bank_view() {
  bins[find_bin(corners)] += 1;
  coverage[find_coverage_cell(corners)] += 1;
  tilt[find_tilt_bucket(corners)] += 1;

  img_pts.push_back(corners);
  obj_pts.push_back(objp);
  frame_count += 1;
}

// Two independent questions about a candidate view: can its corners be located
// precisely, and does it sit somewhere the solve has not already been shown.
// Sharpness answers the first, the coverage grid and tilt buckets the second. A
// view that fails either is worth nothing to the solve, and banking it only
// dilutes the ones that are.
FrameResult LensCalibration::try_bank_view(cv::Mat& gray_frame) {
  if (!detect_board(gray_frame))
    return FrameResult::NoBoard;

  refine_corners(gray_frame);

  const cv::Rect bounds = find_board_bounds(corners, frame_width, frame_height);
  if (bounds.width < LAPLACIAN_KERNEL || bounds.height < LAPLACIAN_KERNEL)
    return FrameResult::NoBoard;

  sharpness = measure_sharpness(gray_frame, bounds);
  if (sharpness < MIN_SHARPNESS)
    return FrameResult::TooBlurry;

  if (!covers_new_ground())
    return FrameResult::NoNewCoverage;

  bank_view();

  return FrameResult::Accepted;
}

void LensCalibration::draw_corners(cv::Mat& bgr_frame) const {
  cv::drawChessboardCorners(
    bgr_frame,
    cv::Size(board_width, board_height),
    corners,
    true
  );
}

double LensCalibration::solve_intrinsics() {
  cam_matrix = cv::Mat::eye(3, 3, CV_64F);
  dist_coeffs = cv::Mat::zeros(5, 1, CV_64F);

  return cv::calibrateCamera(
    obj_pts,
    img_pts,
    cv::Size(frame_width, frame_height),
    cam_matrix,
    dist_coeffs,
    cv::noArray(),   // per view rotations, not used
    cv::noArray(),   // per view translations, not used
    cv::noArray(),   // intrinsic standard deviations, not used
    cv::noArray(),   // extrinsic standard deviations, not used
    view_errors,
    // Three things we already know, given to the solve rather than tested for
    // afterwards. The sensor's pixels are square, so fx and fy are one number.
    // The third radial term is for fisheye lenses, and on a normal one it is so
    // correlated with the first two that the three oscillate against each other
    // into a fit that matches the views and describes no lens. Tangential
    // distortion is the lens sitting non parallel to the sensor, which for a
    // manufactured module is a thousandth, not the fiftieth the free solve
    // produced on all three cameras at once.
    cv::CALIB_FIX_ASPECT_RATIO | cv::CALIB_FIX_K3 | cv::CALIB_ZERO_TANGENT_DIST
  );
}

// Views the rest of the set disagrees with, dropped for good so a bad grab
// cannot come back on the next solve. Returns whether anything went, since
// there is no point solving again if nothing did.
bool LensCalibration::discard_outlier_views() {
  const double limit = OUTLIER_FACTOR * compute_median(view_errors);

  std::vector<std::vector<cv::Point2f>> kept;
  kept.reserve(img_pts.size());

  for (size_t i = 0; i < img_pts.size(); i += 1) {
    if (view_errors.at<double>(static_cast<int>(i)) > limit)
      continue;

    kept.push_back(img_pts[i]);
  }

  discard_count = static_cast<int>(img_pts.size() - kept.size());

  // below the floor there is not enough left to solve on, so the outliers stay
  // and the frame count requirement asks for replacements instead
  if (discard_count == 0 || static_cast<int>(kept.size()) < MIN_SOLVE_FRAMES) {
    discard_count = 0;
    return false;
  }

  img_pts = std::move(kept);
  obj_pts.resize(img_pts.size());
  frame_count = static_cast<int>(img_pts.size());

  return true;
}

// Least squares has no way to tell a view it should not have trusted from one
// it merely fits poorly, so it spreads a bad view's error over every parameter
// instead of setting it aside. Solve once to find out which views the rest
// disagree with, drop those, and solve again on what is left. Same shape as
// dropping a camera that disagrees when triangulating a joint.
double LensCalibration::calibrate() {
  if (frame_count < MIN_LENS_FRAMES)
    return -1.0;

  reprojection_err = solve_intrinsics();

  if (discard_outlier_views())
    reprojection_err = solve_intrinsics();

  return reprojection_err;
}

const char* describe(CalibrationStatus status) {
  switch (status) {
    case CalibrationStatus::Ok:                      return "ok";
    case CalibrationStatus::TooFewFrames:            return "too few views";
    case CalibrationStatus::HighError:               return "reprojection error too high";
    case CalibrationStatus::PrincipalPointOffCenter: return "principal point off the sensor center";
    case CalibrationStatus::DistortionImplausible:   return "distortion absorbing a bad focal length";
    case CalibrationStatus::NeedsTiltedViews:        return "needs the board tilted more";
  }

  return "unknown";
}

// Reprojection error only says the solution fits the views it was fit to. Views
// that all share a pose let focal length and board distance trade off against
// each other, so a whole family of solutions fits them equally well and the
// optimizer returns whichever one it wandered into. These two checks ask the
// separate question of whether the answer describes a camera that could exist.
CalibrationStatus LensCalibration::status() const {
  if (frame_count < MIN_LENS_FRAMES)
    return CalibrationStatus::TooFewFrames;

  if (reprojection_err >= MIN_ERR)
    return CalibrationStatus::HighError;

  const double cx = cam_matrix.at<double>(0, 2);
  const double cy = cam_matrix.at<double>(1, 2);

  if (std::abs(cx - frame_width / 2.0) > MAX_CENTER_OFFSET * frame_width ||
      std::abs(cy - frame_height / 2.0) > MAX_CENTER_OFFSET * frame_height)
    return CalibrationStatus::PrincipalPointOffCenter;

  // A focal length the views could not pin down leaves its slack here, so an
  // implausible k1 is the symptom that shows even when fx and fy look fine.
  // k2 as well as k1: rpicam01 once passed on k1 alone while its higher terms
  // were plainly running away
  if (std::abs(dist_coeffs.at<double>(0)) > MAX_DISTORTION ||
      std::abs(dist_coeffs.at<double>(1)) > MAX_DISTORTION)
    return CalibrationStatus::DistortionImplausible;

  if (!tilt_complete())
    return CalibrationStatus::NeedsTiltedViews;

  return CalibrationStatus::Ok;
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
