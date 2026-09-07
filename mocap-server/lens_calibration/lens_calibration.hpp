#ifndef MOCAP_LENS_CALIBRATION_HPP
#define MOCAP_LENS_CALIBRATION_HPP

#include <string>
#include <opencv2/opencv.hpp>
#include <vector>

#include "config.hpp"

namespace mocap {

constexpr double MIN_ERR = 1.0;

// Lens calibration wants more views than a stereo pair does. Stopping at the
// shared minimum let two runs finish before focal length had settled, and the
// only cost of asking for more is a few more seconds of waving the board.
constexpr int MIN_LENS_FRAMES = 25;

// A view whose reprojection error is this many times the median is not a view
// the model failed to fit, it is a view that was wrong: a blurred grab, or a
// corner set that latched onto the wrong structure. Nothing later removes it,
// so one such view banked early poisons every solve after it.
constexpr double OUTLIER_FACTOR = 3.0;

// Dropping outliers has to be able to take the count below the minimum, or it
// can never fire: the tool solves the moment it reaches MIN_LENS_FRAMES, so
// every solve is at exactly the minimum. Below this floor there is not enough
// left to solve on at all, and the count requirement asks for replacements.
constexpr int MIN_SOLVE_FRAMES = 15;

// The optical axis pierces the sensor near its center, within how the lens is
// mounted over the chip, so a principal point further out than this is not a
// camera that exists.
constexpr double MAX_CENTER_OFFSET = 0.15;

// Variance of the Laplacian over the board measures how well the corners can
// be located: a sharp checkerboard is nothing but edges, and blur collapses
// them. Measured inside the board's bounding box rather than over the whole
// frame, so a busy background cannot pass for focus.
constexpr double MIN_SHARPNESS = 150.0;


// The frame is divided into cells, and a view is banked only if the board
// center lands in a cell that still wants one. Distortion is estimated from
// how straight lines bend, which only shows up away from the center, so a
// calibration solved from the middle of the frame has nothing to fit the edge
// coefficients to.
constexpr int COVERAGE_COLS = 6;
constexpr int COVERAGE_ROWS = 4;
constexpr int VIEWS_PER_CELL = 1;

// A board twice as far away seen through a lens twice as long projects to the
// same pixels, so no amount of moving it around the frame separates focal
// length from distance. Tilting does: the near half of the board grows and the
// far half shrinks, and only perspective explains that. Without tilted views
// the solve leaves focal length free and absorbs the slack into the distortion
// coefficients, which is a calibration that fits its own views perfectly and
// reprojects anything else wrong.
//
// Measured as how far the board's corner angles depart from square, which is
// zero when it faces the camera. Buckets are upper bounds in degrees.
constexpr int TILT_BUCKETS = 3;
constexpr double TILT_BUCKET_LIMITS[TILT_BUCKETS] = {8.0, 18.0, 180.0};
constexpr int VIEWS_PER_TILT_BUCKET = 4;

// A real lens does not need coefficients this large. With k3 fixed at zero the
// remaining two have nothing to oscillate against, so anything past this is the
// model still straining rather than describing the optics.
constexpr double MAX_DISTORTION = 0.3;

// What happened to a candidate frame, so the operator can be told whether to
// hold the board still or move it somewhere new.
enum class FrameResult {
  NoBoard,
  TooBlurry,
  NoNewCoverage,
  Accepted
};

// Why a solve was rejected, so the operator can tell a camera that needs more
// views from one that needs the board held differently.
enum class CalibrationStatus {
  Ok,
  TooFewFrames,
  HighError,
  PrincipalPointOffCenter,
  DistortionImplausible,
  NeedsTiltedViews
};

const char* describe(CalibrationStatus status);

// How one camera is doing, which is the tool's business rather than the
// calibration's: the calibrator knows what it has banked, this knows what has
// been done about it.
struct CameraProgress {
  int accepted = 0;
  int last_attempt = 0;
  bool calibrated = false;
  double error = 0.0;
  FrameResult last_result = FrameResult::NoBoard;
};


class LensCalibration {
private:
  int frame_width;
  int frame_height;
  int board_width;
  int board_height;

  int frame_count;

  std::vector<cv::Point3f> objp;
  std::vector<cv::Point2f> corners;
  std::vector<std::vector<cv::Point2f>> img_pts;
  std::vector<std::vector<cv::Point3f>> obj_pts;

  cv::Mat cam_matrix;
  cv::Mat dist_coeffs;
  cv::Mat view_errors;

  double reprojection_err;
  double sharpness;
  double spacing;
  int discard_count;

  // Acceptance reads the joint bins: a view is new if no view has landed in
  // this cell at this angle before. Either moving the board or tilting it opens
  // a fresh bin, so neither dimension can deadlock the other, and a motionless
  // board fills one bin and is refused after that.
  std::vector<int> bins;

  // marginals of the above, kept for the overlay and for tilt_complete
  std::vector<int> coverage;
  std::vector<int> tilt;

  // steps of try_frame, in the order it runs them
  bool detect_board(const cv::Mat& gray_frame);
  void refine_corners(const cv::Mat& gray_frame);
  bool covers_new_ground() const;
  void bank_view();

  // steps of calibrate
  double solve_intrinsics();
  bool discard_outlier_views();

  int find_bin(const std::vector<cv::Point2f>& points) const;
  int find_coverage_cell(const std::vector<cv::Point2f>& points) const;
  int find_tilt_bucket(const std::vector<cv::Point2f>& points) const;

public:
  LensCalibration(
    int frame_width,
    int frame_height,
    int board_width,
    int board_height,
    float square_size
  );

  FrameResult try_bank_view(cv::Mat& gray_frame);
  void draw_corners(cv::Mat& bgr_frame) const;
  double calibrate();
  CalibrationStatus status() const;
  bool save_params(const std::string& filename) const;

  // what the last accepted frame measured, for tuning the thresholds
  double last_sharpness() const { return sharpness; }
  double last_spacing() const { return spacing; }

  // how many views the last solve threw out as outliers
  int discarded() const { return discard_count; }

  // per view reprojection errors from the last solve, in pixels
  const cv::Mat& per_view_errors() const { return view_errors; }

  // where views have landed, marginally, which is what the overlay shows
  const std::vector<int>& coverage_map() const { return coverage; }
  const std::vector<int>& tilt_map() const { return tilt; }
  bool coverage_complete() const;
  bool tilt_complete() const;
};

// One camera and everything the tool knows about it, so the loops walk a single
// sequence rather than indexing three in parallel.
struct CameraCalibrationContext {
  Camera camera;
  LensCalibration calibrator;
  CameraProgress progress;
};


} // namespace mocap

#endif // MOCAP_LENS_CALIBRATION_HPP
