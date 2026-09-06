#ifndef MOCAP_POSE_CONSTANTS_HPP
#define MOCAP_POSE_CONSTANTS_HPP

#include <cstddef>

namespace mocap {

constexpr int INPUT_CHANNELS = 3;
constexpr int INPUT_HEIGHT = 384;
constexpr int INPUT_WIDTH = 288;

constexpr int NUM_KEYPOINTS = 133;   // COCO WholeBody

// RTMW predicts coordinates as two 1D distributions per joint rather than a 2D
// heatmap. The bin counts are exactly twice the input dimensions, so a bin
// index divided by SIMCC_SPLIT_RATIO is a network pixel coordinate.
constexpr int SIMCC_X_BINS = 576;
constexpr int SIMCC_Y_BINS = 768;
constexpr float SIMCC_SPLIT_RATIO = 2.0f;

// The engine was built with min 1, opt 3, max 3, so a frameset missing a
// camera still runs without rebuilding.
constexpr int MAX_BATCH = 3;

constexpr size_t INPUT_ELEMS_PER_IMAGE =
  static_cast<size_t>(INPUT_CHANNELS) * INPUT_HEIGHT * INPUT_WIDTH;

// ImageNet statistics in 0-255 RGB, which is what mmpose trains RTMW with.
constexpr float PIXEL_MEAN[3] = {123.675f, 116.28f, 103.53f};
constexpr float PIXEL_STD[3] = {58.395f, 57.12f, 57.375f};

} // namespace mocap

#endif // MOCAP_POSE_CONSTANTS_HPP
