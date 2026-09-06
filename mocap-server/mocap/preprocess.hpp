#ifndef MOCAP_PREPROCESS_HPP
#define MOCAP_PREPROCESS_HPP

#include <cstdint>

#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include "pose_constants.hpp"

namespace mocap {

// Maps network coordinates to the source frame they came from, as a 3x3 in
// homogeneous coordinates:
//
//   src_x = m(0,0)*dst_x + m(0,1)*dst_y + m(0,2)
//   src_y = m(1,0)*dst_x + m(1,1)*dst_y + m(1,2)
//
// This direction is what the kernel needs, since each thread owns a
// destination pixel and has to find where it reads from. It is also what the
// keypoint decode needs, so both use the same description and cannot disagree
// about the geometry.
//
// Built once at startup, not per frame.
//
// Compose the forward transform the way you would describe it out loud, then
// invert it. cv::Matx33d has a constructor taking nine values row major, an
// operator* that composes, and an analytic inv(). Rightmost applies first, so
// translate * scale * rotate rotates before it scales.
//
// Check it before trusting it: source (src_width-1, 0), the top right corner,
// should land at the top left of the content region once the forward transform
// runs. If it comes out mirrored or transposed, the composition order is wrong.
cv::Matx33d make_transform(uint32_t src_width, uint32_t src_height);

// Applies the transform to one point. Used by the keypoint decode to turn
// network coordinates back into source frame pixels.
void network_to_source(
  const cv::Matx33d& transform,
  float network_x,
  float network_y,
  float& source_x,
  float& source_y
);

// ---------------------------------------------------------------------------
// Turns one decoded camera frame into one batch element of the network input.
//
// Source is NV12 as NVDEC produces it:
//   - luma plane at nv12, src_height rows of src_width bytes
//   - each row padded to src_pitch bytes (1536 for a 1280 wide frame)
//   - chroma plane follows immediately at nv12 + src_pitch * src_height,
//     interleaved UV, half resolution in both axes, same pitch
//
// Destination is one slot of the TensorRT input buffer, already offset by the
// caller, so the kernel writes to dst[0 .. INPUT_CHANNELS * INPUT_HEIGHT *
// INPUT_WIDTH). Layout is NCHW planar float32: all of R, then all of G, then
// all of B, each INPUT_HEIGHT rows of INPUT_WIDTH, no padding.
//
// Geometry comes in through `transform`, so the kernel knows nothing about
// rotation, scale or letterboxing. It maps its destination pixel through the
// six coefficients, and a result outside the source bounds is padding. That
// one test covers letterbox, pillarbox and crop alike.
//
// Normalization is (value - mean) / std per channel. The constants in
// pose_constants.hpp are in 0-255 RGB, so either scale them or the pixels, but
// not both. This one is worth checking against a reference implementation
// before trusting any keypoints that come out.
//
// Runs on the caller's stream and must not synchronize: the caller records an
// event afterwards and only releases the source surface once that event fires.
// ---------------------------------------------------------------------------
void preprocess_nv12(
  const uint8_t* nv12,
  uint32_t src_width,
  uint32_t src_height,
  uint32_t src_pitch,
  const cv::Matx33d& transform,
  float* dst,
  cudaStream_t stream
);

} // namespace mocap

#endif // MOCAP_PREPROCESS_HPP
