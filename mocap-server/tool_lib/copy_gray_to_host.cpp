#include <cuda_runtime.h>

#include "copy_gray_to_host.hpp"

namespace mocap {

bool copy_gray_to_host(const FrameView& view, cv::Mat& out) {
  if (out.rows != static_cast<int>(view.height) || out.cols != static_cast<int>(view.width) ||
      out.type() != CV_8UC1)
    out.create(static_cast<int>(view.height), static_cast<int>(view.width), CV_8UC1);

  // the decoder's pitch is padded for alignment and wider than the frame, so
  // the copy is strided rather than a flat memcpy
  cudaError_t status = cudaMemcpy2D(
    out.data,
    out.step,
    view.device_ptr,
    view.pitch,
    view.width,
    view.height,
    cudaMemcpyDeviceToHost
  );

  return status == cudaSuccess;
}

} // namespace mocap
