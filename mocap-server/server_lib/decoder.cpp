#include <utility>

#include "decoder.hpp"

namespace mocap {

namespace {

constexpr const char* CODEC_NAME = "h264_cuvid";

// picam stamps capture time in nanoseconds, so a 1 ns tick keeps the pts
// exact rather than quantising it into the decoder's timebase
constexpr AVRational TIMEBASE{1, 1000000000};

Error av_error(const char* what, int ret) {
  char buf[AV_ERROR_MAX_STRING_SIZE];
  av_strerror(ret, buf, sizeof(buf));
  return Error{
    std::make_error_code(std::errc::io_error),
    std::string(what) + ": " + buf
  };
}

} // namespace

//  HwContext 

HwContext::HwContext(AVBufferRef* ctx) : m_ctx(ctx) {}

HwContext::HwContext(HwContext&& other) noexcept
  : m_ctx(std::exchange(other.m_ctx, nullptr)) {}

HwContext& HwContext::operator=(HwContext&& other) noexcept {
  if (this != &other) {
    if (m_ctx)
      av_buffer_unref(&m_ctx);
    m_ctx = std::exchange(other.m_ctx, nullptr);
  }
  return *this;
}

HwContext::~HwContext() {
  if (m_ctx)
    av_buffer_unref(&m_ctx);
}

Result<HwContext> HwContext::open() {
  AVBufferRef* ctx = nullptr;

  int ret = av_hwdevice_ctx_create(&ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
  if (ret < 0)
    return std::unexpected(av_error("failed to create CUDA device context", ret));

  return HwContext(ctx);
}

//  DecodedFrame 

DecodedFrame::DecodedFrame(AVFrame* frame) : m_frame(frame) {}

DecodedFrame::DecodedFrame(DecodedFrame&& other) noexcept
  : m_frame(std::exchange(other.m_frame, nullptr)) {}

DecodedFrame& DecodedFrame::operator=(DecodedFrame&& other) noexcept {
  if (this != &other) {
    if (m_frame)
      av_frame_free(&m_frame);
    m_frame = std::exchange(other.m_frame, nullptr);
  }
  return *this;
}

// releases this frame's reference, returning its surface to the decoder pool
DecodedFrame::~DecodedFrame() {
  if (m_frame)
    av_frame_free(&m_frame);
}

const uint8_t* DecodedFrame::device_ptr() const {
  return m_frame ? m_frame->data[0] : nullptr;
}

uint32_t DecodedFrame::pitch() const {
  return m_frame ? static_cast<uint32_t>(m_frame->linesize[0]) : 0;
}

uint64_t DecodedFrame::timestamp() const {
  return m_frame ? static_cast<uint64_t>(m_frame->pts) : 0;
}

//  Decoder 

Decoder::Decoder(AVCodecContext* ctx, AVPacket* packet)
  : m_ctx(ctx), m_packet(packet) {}

Decoder::Decoder(Decoder&& other) noexcept
  : m_ctx(std::exchange(other.m_ctx, nullptr)),
    m_packet(std::exchange(other.m_packet, nullptr)) {}

Decoder& Decoder::operator=(Decoder&& other) noexcept {
  if (this != &other) {
    if (m_packet)
      av_packet_free(&m_packet);
    if (m_ctx)
      avcodec_free_context(&m_ctx);
    m_ctx = std::exchange(other.m_ctx, nullptr);
    m_packet = std::exchange(other.m_packet, nullptr);
  }
  return *this;
}

Decoder::~Decoder() {
  if (m_packet)
    av_packet_free(&m_packet);
  if (m_ctx)
    avcodec_free_context(&m_ctx);
}

Result<Decoder> Decoder::open(const StreamParams& params,
                                            uint32_t surfaces,
                                            const HwContext& hw) {
  const AVCodec* codec = avcodec_find_decoder_by_name(CODEC_NAME);
  if (!codec)
    return std::unexpected(invalid("h264_cuvid decoder not available"));

  AVCodecContext* ctx = avcodec_alloc_context3(codec);
  if (!ctx)
    return std::unexpected(invalid("failed to allocate decoder context"));

  Decoder decoder(ctx, nullptr);

  ctx->hw_device_ctx = av_buffer_ref(hw.get());
  if (!ctx->hw_device_ctx)
    return std::unexpected(invalid("failed to reference CUDA device context"));

  ctx->width = static_cast<int>(params.frame_width);
  ctx->height = static_cast<int>(params.frame_height);
  ctx->pix_fmt = AV_PIX_FMT_NONE;
  ctx->pkt_timebase = TIMEBASE;
  // emit each frame as soon as it decodes rather than buffering for reorder
  ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;

  AVDictionary* opts = nullptr;
  av_dict_set_int(&opts, "surfaces", surfaces, 0);

  int ret = avcodec_open2(ctx, codec, &opts);
  av_dict_free(&opts);
  if (ret < 0)
    return std::unexpected(av_error("failed to open decoder", ret));

  decoder.m_packet = av_packet_alloc();
  if (!decoder.m_packet)
    return std::unexpected(invalid("failed to allocate packet"));

  return decoder;
}

Result<void> Decoder::send_packet(std::span<const uint8_t> data, uint64_t timestamp) {
  // the packet only borrows the caller's buffer; avcodec_send_packet copies
  // what it needs before returning
  m_packet->data = const_cast<uint8_t*>(data.data());
  m_packet->size = static_cast<int>(data.size());
  m_packet->pts = static_cast<int64_t>(timestamp);

  int ret = avcodec_send_packet(m_ctx, m_packet);
  if (ret < 0)
    return std::unexpected(av_error("failed to send packet to decoder", ret));

  return {};
}

Result<std::optional<DecodedFrame>> Decoder::receive_frame() {
  AVFrame* frame = av_frame_alloc();
  if (!frame)
    return std::unexpected(invalid("failed to allocate frame"));

  int ret = avcodec_receive_frame(m_ctx, frame);
  if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
    av_frame_free(&frame);
    return std::nullopt;
  }
  if (ret < 0) {
    av_frame_free(&frame);
    return std::unexpected(av_error("failed to receive frame from decoder", ret));
  }

  return DecodedFrame(frame);
}

} // namespace mocap
