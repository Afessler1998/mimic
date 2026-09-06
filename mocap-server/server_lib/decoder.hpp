#ifndef MOCAP_DECODER_HPP
#define MOCAP_DECODER_HPP

#include <cstdint>
#include <expected>
#include <optional>
#include <span>

#include "config.hpp"
#include "error.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
}

namespace mocap {

// CUDA device context shared by every Decoder, so all streams decode onto the
// same GPU and their surfaces live in one address space.
class HwContext {
public:
  static Result<HwContext> open();

  HwContext(HwContext&& other) noexcept;
  HwContext& operator=(HwContext&& other) noexcept;
  HwContext(const HwContext&) = delete;
  HwContext& operator=(const HwContext&) = delete;
  ~HwContext();

  AVBufferRef* get() const { return m_ctx; }

private:
  explicit HwContext(AVBufferRef* ctx);

  AVBufferRef* m_ctx = nullptr;
};

// One decoded frame, still resident on the GPU. Holds a reference to its NVDEC
// surface, so the surface cannot be recycled while this object is alive. The
// surface pool is fixed size: hold too many of these and the Decoder stalls.
class DecodedFrame {
public:
  DecodedFrame() = default;

  DecodedFrame(DecodedFrame&& other) noexcept;
  DecodedFrame& operator=(DecodedFrame&& other) noexcept;
  DecodedFrame(const DecodedFrame&) = delete;
  DecodedFrame& operator=(const DecodedFrame&) = delete;
  ~DecodedFrame();

  // NV12 luma plane in device memory. Chroma follows at pitch * height.
  const uint8_t* device_ptr() const;
  uint32_t pitch() const;

  // capture timestamp, carried through the Decoder as the packet pts
  uint64_t timestamp() const;

  bool valid() const { return m_frame != nullptr; }

private:
  explicit DecodedFrame(AVFrame* frame);
  friend class Decoder;

  AVFrame* m_frame = nullptr;
};

// h264_cuvid Decoder for a single Camera stream.
class Decoder {
public:
  static Result<Decoder> open(const StreamParams& params,
                                            uint32_t surfaces,
                                            const HwContext& hw);

  Decoder(Decoder&& other) noexcept;
  Decoder& operator=(Decoder&& other) noexcept;
  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;
  ~Decoder();

  Result<void> send_packet(std::span<const uint8_t> data, uint64_t timestamp);

  Result<std::optional<DecodedFrame>> receive_frame();

private:
  Decoder(AVCodecContext* ctx, AVPacket* packet);

  AVCodecContext* m_ctx = nullptr;
  AVPacket* m_packet = nullptr;
};

} // namespace mocap

#endif // MOCAP_DECODER_HPP
