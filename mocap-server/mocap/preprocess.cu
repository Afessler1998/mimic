#include "preprocess.hpp"

#include <algorithm>
#include <opencv2/core.hpp>

namespace mocap {

cv::Matx33d make_transform(uint32_t src_width, uint32_t src_height) {
  const double src_width_minus_one = static_cast<double>(src_width - 1);
  cv::Matx33d rotation{0.0, 1.0, 0.0, -1.0, 0.0, src_width_minus_one,
                       0.0, 0.0, 1.0};

  const double scale_x =
      static_cast<double>(src_height) / static_cast<double>(INPUT_WIDTH);
  const double scale_y =
      static_cast<double>(src_width) / static_cast<double>(INPUT_HEIGHT);
  const double s = std::max(scale_x, scale_y);

  cv::Matx33d scale{1.0 / s, 0.0, 0.0, 0.0, 1.0 / s, 0.0, 0.0, 0.0, 1.0};

  const double offset_x = (INPUT_WIDTH - src_height / s) / 2;
  const double offset_y = (INPUT_HEIGHT - src_width / s) / 2;

  cv::Matx33d translation{1.0,      0.0, offset_x, 0.0, 1.0,
                          offset_y, 0.0, 0.0,      1.0};

  cv::Matx33d pixel_center_translation(1.0, 0.0, 0.5, 0.0, 1.0, 0.5, 0.0, 0.0,
                                       1.0);

  return pixel_center_translation.inv() *
         (translation * scale * rotation).inv() * pixel_center_translation;
}

void network_to_source(const cv::Matx33d &transform, float network_x,
                       float network_y, float &source_x, float &source_y) {
  const double x = static_cast<double>(network_x);
  const double y = static_cast<double>(network_y);

  source_x = static_cast<float>(transform(0, 0) * x + transform(0, 1) * y +
                                transform(0, 2));
  source_y = static_cast<float>(transform(1, 0) * x + transform(1, 1) * y +
                                transform(1, 2));
}

struct Affine {
  float params[6];
};

// BT.709 limited range, which is what NVDEC produces for 720p. The offsets are
// why the range matters: luma is stored 16..235 rather than 0..255, and the
// chroma channels are signed quantities stored centred on 128.
__device__ inline void yuv_to_rgb(float y, float u, float v, float &r, float &g,
                                  float &b) {
  const float yn = 1.164f * (y - 16.0f);
  const float un = u - 128.0f;
  const float vn = v - 128.0f;

  r = yn + 1.793f * vn;
  g = yn - 0.213f * un - 0.534f * vn;
  b = yn + 2.115f * un;
}

__global__ void preprocess_nv12_kernel(const uint8_t *nv12, uint32_t src_width,
                                       uint32_t src_height, uint32_t src_pitch,
                                       float *dst, Affine affine) {
  const uint32_t dst_x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t dst_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (dst_x >= INPUT_WIDTH || dst_y >= INPUT_HEIGHT)
    return;

  // constexpr arrays at namespace scope are not addressable from device code,
  // so these are local copies of the constants in pose_constants.hpp
  const float mean[INPUT_CHANNELS] = {PIXEL_MEAN[0], PIXEL_MEAN[1],
                                      PIXEL_MEAN[2]};
  const float stddev[INPUT_CHANNELS] = {PIXEL_STD[0], PIXEL_STD[1],
                                        PIXEL_STD[2]};

  const uint32_t plane_stride = INPUT_HEIGHT * INPUT_WIDTH;
  const uint32_t dst_index = dst_y * INPUT_WIDTH + dst_x;

  const float dst_xf = static_cast<float>(dst_x);
  const float dst_yf = static_cast<float>(dst_y);

  const float src_x =
      affine.params[0] * dst_xf + affine.params[1] * dst_yf + affine.params[2];
  const float src_y =
      affine.params[3] * dst_xf + affine.params[4] * dst_yf + affine.params[5];

  // outside the source is the letterbox. one test covers pad, pillarbox and
  // crop alike, which is the point of carrying the geometry as a matrix.
  const float max_x = static_cast<float>(src_width) - 1.0f;
  const float max_y = static_cast<float>(src_height) - 1.0f;

  if (src_x < 0.0f || src_x > max_x || src_y < 0.0f || src_y > max_y) {
    // the pad has to survive normalisation as a constant, so fill with black
    // in source terms and let the same normalisation below apply to it
    for (int channel = 0; channel < INPUT_CHANNELS; channel += 1)
      dst[channel * plane_stride + dst_index] =
          (0.0f - mean[channel]) / stddev[channel];
    return;
  }

  const float x0f = floorf(src_x);
  const float y0f = floorf(src_y);
  const float x_weight = src_x - x0f;
  const float y_weight = src_y - y0f;

  const int x0 = static_cast<int>(x0f);
  const int y0 = static_cast<int>(y0f);

  // x1 and y1 run one past the last row and column when the sample lands on
  // the final pixel, so the far tap collapses onto its neighbour there
  const int x1 = min(x0 + 1, static_cast<int>(src_width) - 1);
  const int y1 = min(y0 + 1, static_cast<int>(src_height) - 1);

  const float luma00 = nv12[y0 * src_pitch + x0];
  const float luma10 = nv12[y0 * src_pitch + x1];
  const float luma01 = nv12[y1 * src_pitch + x0];
  const float luma11 = nv12[y1 * src_pitch + x1];

  const float top = luma00 * (1.0f - x_weight) + luma10 * x_weight;
  const float bottom = luma01 * (1.0f - x_weight) + luma11 * x_weight;
  const float luma = top * (1.0f - y_weight) + bottom * y_weight;

  // chroma is already subsampled 2x by the encoder, so interpolating it back
  // up recovers detail that is not there. nearest is what production paths use.
  const uint8_t *chroma_plane = nv12 + src_pitch * src_height;
  const uint32_t chroma_x = (static_cast<uint32_t>(src_x) / 2u) * 2u;
  const uint32_t chroma_y = static_cast<uint32_t>(src_y) / 2u;

  const float chroma_u = chroma_plane[chroma_y * src_pitch + chroma_x];
  const float chroma_v = chroma_plane[chroma_y * src_pitch + chroma_x + 1u];

  float rgb[INPUT_CHANNELS];
  yuv_to_rgb(luma, chroma_u, chroma_v, rgb[0], rgb[1], rgb[2]);

  for (int channel = 0; channel < INPUT_CHANNELS; channel += 1) {
    // cv2 converts through uint8, so it saturates. matching that keeps the
    // input distribution the same as the one the model trained on.
    const float clamped = fminf(fmaxf(rgb[channel], 0.0f), 255.0f);
    const float normalised = (clamped - mean[channel]) / stddev[channel];

    dst[channel * plane_stride + dst_index] = normalised;
  }
}

void preprocess_nv12(const uint8_t *nv12_dev_ptr, uint32_t src_width,
                     uint32_t src_height, uint32_t src_pitch,
                     const cv::Matx33d &transform, float *dst_dev_ptr,
                     cudaStream_t stream) {
  dim3 block(32, 8);
  dim3 grid((INPUT_WIDTH + block.x - 1) / block.x,
            (INPUT_HEIGHT + block.y - 1) / block.y);

  Affine affine;
  for (int i = 0; i < 6; i += 1) {
    affine.params[i] = static_cast<float>(transform.val[i]);
  }

  preprocess_nv12_kernel<<<grid, block, 0, stream>>>(
      nv12_dev_ptr, src_width, src_height, src_pitch, dst_dev_ptr, affine);
}

} // namespace mocap
