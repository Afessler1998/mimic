#ifndef MOCAP_COPY_GRAY_TO_HOST_HPP
#define MOCAP_COPY_GRAY_TO_HOST_HPP

#include <opencv2/opencv.hpp>

#include "session.hpp"

namespace mocap {

// Copies a frame's luma plane across the bus into a host grayscale Mat. This
// is a real device to host transfer, not a colour conversion.
//
// NV12 stores luma as a full resolution 8 bit plane, so it is already exactly
// what findChessboardCorners wants. Nothing here touches chroma, and no colour
// conversion happens.
bool copy_gray_to_host(const FrameView& view, cv::Mat& out);

} // namespace mocap

#endif // MOCAP_COPY_GRAY_TO_HOST_HPP
