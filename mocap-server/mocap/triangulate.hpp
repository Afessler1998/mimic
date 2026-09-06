#ifndef MOCAP_TRIANGULATE_HPP
#define MOCAP_TRIANGULATE_HPP

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "error.hpp"
#include "pose_constants.hpp"
#include "pose_model.hpp"

namespace mocap {

// A joint in the reference camera's frame, in whatever units the calibration
// board's square size was given in. Views is how many cameras contributed, so
// a consumer can tell a two view guess from a three view agreement.
struct Keypoint3d {
  float x;
  float y;
  float z;
  float confidence;
  uint8_t views;
};

struct Pose3d {
  std::array<Keypoint3d, NUM_KEYPOINTS> keypoints;
};

// Lifts per camera 2D keypoints into one 3D pose.
//
// The world frame is the first camera in the config. Stereo calibration solves
// each pair independently, so the pairs that relate the reference camera to
// every other one are the ones that chain into a single frame, and those are
// what this loads.
class Triangulator {
public:
  static Result<Triangulator> load(const std::vector<Camera>& cameras);

  // Solves every joint from whichever cameras saw it with enough confidence.
  // The result is owned by the triangulator and valid until the next call.
  const Pose3d& triangulate(std::span<const PoseResult> results);

private:
  // One camera's contribution to the linear system: the intrinsics needed to
  // undistort its keypoints, and the pose that places its rays in the world.
  struct View {
    uint8_t camera_id;
    cv::Matx33d cam_matrix;
    cv::Mat dist_coeffs;
    cv::Matx34d pose;
  };

  // One view's ray for one joint, kept in a flat list so a view that
  // disagrees with the others can be dropped and the joint solved again.
  struct Contribution {
    size_t view;
    float confidence;
    cv::Point2f ray;
    double error;
  };

  bool solve(const std::vector<Contribution>& contributions,
             cv::Vec3d& point) const;

  // How far the solved point lands from this view's ray, in source pixels.
  double reprojection_error(const Contribution& contribution,
                            const cv::Vec3d& point) const;

  std::vector<View> m_views;

  // reused across calls so a per frame solve does not allocate
  std::vector<cv::Point2f> m_distorted;
  std::vector<cv::Point2f> m_normalized;
  std::vector<std::vector<cv::Point2f>> m_rays;
  std::vector<const PoseResult*> m_present;
  std::vector<Contribution> m_contributions;
  Pose3d m_pose;
};

} // namespace mocap

#endif // MOCAP_TRIANGULATE_HPP
