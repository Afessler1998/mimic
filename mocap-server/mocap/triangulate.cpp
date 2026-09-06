#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/geometry/3d.hpp>

#include "calibration_params.hpp"
#include "triangulate.hpp"

namespace mocap {

namespace {

// A joint the network is unsure about in one view drags the whole solve, and
// SimCC peaks on occluded joints are low, so a view only contributes above
// this.
constexpr float MIN_CONFIDENCE = 0.3f;

// two rays are the minimum that intersect anywhere
constexpr int MIN_VIEWS = 2;

// How far a view's ray may miss the solved point before that view is treated
// as wrong about the joint rather than noisy. The network sees a frame scaled
// down by more than three, so a single network pixel of disagreement is
// already several source pixels.
constexpr double MAX_REPROJECTION_ERROR = 12.0;

// Builds the pair filename stereo calibration wrote, which is ordered by the
// camera's position in the config rather than by name.
std::string pair_filename(const Camera& reference, const Camera& other) {
  return reference.name + "_" + other.name + "_calibration.yaml";
}

// Homogeneous 3x4 pose from the rotation and translation stereo calibration
// solved, which map a point in the reference camera's frame into this one.
cv::Matx34d make_pose(const cv::Mat& rotation, const cv::Mat& translation) {
  cv::Matx34d pose = cv::Matx34d::zeros();

  for (int row = 0; row < 3; row += 1) {
    for (int col = 0; col < 3; col += 1)
      pose(row, col) = rotation.at<double>(row, col);

    pose(row, 3) = translation.at<double>(row);
  }

  return pose;
}

} // namespace

Result<Triangulator> Triangulator::load(const std::vector<Camera>& cameras) {
  if (cameras.size() < static_cast<size_t>(MIN_VIEWS))
    return std::unexpected(invalid("triangulation needs at least two cameras"));

  Triangulator triangulator;
  triangulator.m_views.reserve(cameras.size());

  for (size_t i = 0; i < cameras.size(); i += 1) {
    calibration_params intrinsics;
    const std::string filename = cameras[i].name + "_calibration.yaml";
    if (!load_calibration_params(filename, intrinsics))
      return std::unexpected(invalid("missing " + filename));

    // the loader reports success for a file that opened, so a file written by
    // something else reaches here as empty matrices
    if (intrinsics.cam_matrix.rows != 3 || intrinsics.cam_matrix.cols != 3)
      return std::unexpected(invalid("no 3x3 cam_matrix in " + filename));

    View view;
    view.camera_id = cameras[i].id;
    view.cam_matrix = intrinsics.cam_matrix;
    view.dist_coeffs = intrinsics.dist_coeffs;

    // the first camera defines the world, so its pose is the identity and no
    // pair file is involved
    if (i == 0) {
      view.pose = cv::Matx34d::eye();
    } else {
      stereo_params extrinsics;
      const std::string pair = pair_filename(cameras[0], cameras[i]);
      if (!load_stereo_params(pair, extrinsics))
        return std::unexpected(invalid("missing " + pair));

      if (extrinsics.rotation.rows != 3 || extrinsics.rotation.cols != 3 ||
          extrinsics.translation.total() != 3)
        return std::unexpected(invalid("no 3x3 rotation and 3x1 translation in " + pair));

      view.pose = make_pose(extrinsics.rotation, extrinsics.translation);
    }

    triangulator.m_views.push_back(std::move(view));
  }

  triangulator.m_distorted.resize(NUM_KEYPOINTS);
  triangulator.m_normalized.resize(NUM_KEYPOINTS);
  triangulator.m_rays.assign(triangulator.m_views.size(),
                             std::vector<cv::Point2f>(NUM_KEYPOINTS));
  triangulator.m_present.assign(triangulator.m_views.size(), nullptr);
  triangulator.m_contributions.reserve(triangulator.m_views.size());

  return triangulator;
}

// Each contributing view gives two equations saying its ray passes through the
// point, weighted by how sure the network was, and the point is the direction
// that satisfies all of them best. A projection is only equal to its ray up to
// scale, so cross multiplying x = (P0.X)/(P2.X) is what makes it linear.
bool Triangulator::solve(const std::vector<Contribution>& contributions,
                         cv::Vec3d& point) const {
  cv::Matx<double, 2 * MAX_BATCH, 4> equations;
  int rows = 0;

  for (const Contribution& contribution : contributions) {
    const cv::Matx34d& pose = m_views[contribution.view].pose;
    const cv::Point2f& ray = contribution.ray;
    const double weight = contribution.confidence;

    for (int col = 0; col < 4; col += 1) {
      equations(rows, col) = weight * (ray.x * pose(2, col) - pose(0, col));
      equations(rows + 1, col) = weight * (ray.y * pose(2, col) - pose(1, col));
    }

    rows += 2;
  }

  // only the rows that were filled, so a joint two cameras saw is not solved
  // against a third camera's stale equations
  cv::Vec4d solution;
  cv::SVD::solveZ(
    cv::Mat(equations.rows, 4, CV_64F, equations.val).rowRange(0, rows),
    solution);

  // a point on the plane through the cameras projects to the same ray from
  // every one of them, which leaves the solve with no scale to recover
  if (std::abs(solution[3]) < 1e-9)
    return false;

  point = cv::Vec3d(solution[0] / solution[3],
                    solution[1] / solution[3],
                    solution[2] / solution[3]);
  return true;
}

double Triangulator::reprojection_error(const Contribution& contribution,
                                        const cv::Vec3d& point) const {
  const View& view = m_views[contribution.view];
  const cv::Vec3d projected =
    view.pose * cv::Vec4d(point[0], point[1], point[2], 1.0);

  // a point behind the camera projects to a mirrored point in front of it,
  // which can land close enough to the ray to look like agreement
  if (projected[2] <= 0.0)
    return std::numeric_limits<double>::max();

  const double dx = projected[0] / projected[2] - contribution.ray.x;
  const double dy = projected[1] / projected[2] - contribution.ray.y;

  // the rays are normalized, so scaling by focal length puts the error back in
  // source pixels and lets the threshold be stated in them
  return std::hypot(dx * view.cam_matrix(0, 0), dy * view.cam_matrix(1, 1));
}

const Pose3d& Triangulator::triangulate(std::span<const PoseResult> results) {
  // Undistorting is a per view batch, so it runs once over all the joints
  // before the per joint solve rather than a call per ray. With no projection
  // matrix passed the output is normalized image coordinates, which is what
  // lets the pose alone act as the projection below.
  std::fill(m_present.begin(), m_present.end(), nullptr);

  for (size_t i = 0; i < m_views.size(); i += 1) {
    const auto match = std::find_if(
      results.begin(), results.end(),
      [&](const PoseResult& result) {
        return result.camera_id == m_views[i].camera_id;
      });

    if (match == results.end())
      continue;

    for (int joint = 0; joint < NUM_KEYPOINTS; joint += 1)
      m_distorted[joint] = cv::Point2f(match->keypoints[joint].x,
                                       match->keypoints[joint].y);

    cv::undistortPoints(m_distorted, m_normalized,
                        m_views[i].cam_matrix, m_views[i].dist_coeffs);

    m_rays[i] = m_normalized;
    m_present[i] = &(*match);
  }

  for (int joint = 0; joint < NUM_KEYPOINTS; joint += 1) {
    m_contributions.clear();

    for (size_t i = 0; i < m_views.size(); i += 1) {
      if (!m_present[i])
        continue;

      const float confidence = m_present[i]->keypoints[joint].confidence;
      if (confidence < MIN_CONFIDENCE)
        continue;

      m_contributions.push_back(Contribution{i, confidence, m_rays[i][joint], 0.0});
    }

    Keypoint3d& keypoint = m_pose.keypoints[joint];
    keypoint = Keypoint3d{0.0f, 0.0f, 0.0f, 0.0f,
                          static_cast<uint8_t>(m_contributions.size())};

    if (m_contributions.size() < static_cast<size_t>(MIN_VIEWS))
      continue;

    // Least squares has no way to notice a view it should not have trusted, so
    // the point is solved, measured against every ray that produced it, and
    // solved again without the worst if that one is too far off. Two views is
    // the floor: below that there is nothing left to disagree with.
    cv::Vec3d point;
    bool solved = solve(m_contributions, point);

    while (solved) {
      size_t worst = 0;
      for (size_t i = 0; i < m_contributions.size(); i += 1) {
        m_contributions[i].error = reprojection_error(m_contributions[i], point);
        if (m_contributions[i].error > m_contributions[worst].error)
          worst = i;
      }

      if (m_contributions[worst].error <= MAX_REPROJECTION_ERROR)
        break;

      // the rays that are left still do not meet, so the joint is wrong in
      // every view rather than in one of them
      if (m_contributions.size() == static_cast<size_t>(MIN_VIEWS)) {
        solved = false;
        break;
      }

      m_contributions.erase(m_contributions.begin() + worst);
      solved = solve(m_contributions, point);
    }

    if (!solved) {
      keypoint.views = 0;
      continue;
    }

    float confidence_sum = 0.0f;
    for (const Contribution& contribution : m_contributions)
      confidence_sum += contribution.confidence;

    keypoint.x = static_cast<float>(point[0]);
    keypoint.y = static_cast<float>(point[1]);
    keypoint.z = static_cast<float>(point[2]);
    keypoint.confidence = confidence_sum / m_contributions.size();
    keypoint.views = static_cast<uint8_t>(m_contributions.size());
  }

  return m_pose;
}

} // namespace mocap
