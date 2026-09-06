#include <cerrno>
#include <cstdio>
#include <cstring>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <unistd.h>
#include <utility>

#include "event_loop.hpp"

namespace mocap {

namespace {

constexpr size_t MAX_EVENTS = 16;

// picam refuses to send a frame larger than PKT_MAX_SIZE - 12, so a buffer
// this size can always hold one complete frame plus its header
constexpr size_t MAX_FRAME_SIZE = 262144;
constexpr size_t FRAME_HEADER_SIZE = 12;
constexpr size_t MAX_PAYLOAD_SIZE = MAX_FRAME_SIZE - FRAME_HEADER_SIZE;

// NVDEC surface pool per stream. Every frame handed out holds one, so this is
// the ceiling on frames in flight before the decoder stalls.
// framesets the consumer may have outstanding. each one pins a surface per
// camera, so the decoder pool has to cover this plus its own working set.
constexpr size_t POOL_SLOTS = 4;
constexpr uint32_t DECODE_SURFACES = POOL_SLOTS + 12;

// picam frames a packet as [uint64 capture timestamp][uint32 payload size]
struct FrameHeader {
  uint64_t timestamp;
  uint32_t size;
};

FrameHeader read_frame_header(const uint8_t* data) {
  FrameHeader header;
  std::memcpy(&header.timestamp, data, sizeof(header.timestamp));
  std::memcpy(&header.size, data + sizeof(header.timestamp), sizeof(header.size));
  return header;
}

// epoll hands back one 64-bit token per ready fd. We split it into what kind
// of fd fired and which Camera it belongs to.
enum class SourceKind : uint32_t { stop, listener, stream };

struct EventSource {
  SourceKind kind;
  uint32_t index;
};

uint64_t encode(SourceKind kind, size_t index) {
  return (static_cast<uint64_t>(kind) << 32) | static_cast<uint32_t>(index);
}

EventSource decode(uint64_t key) {
  return EventSource{
    static_cast<SourceKind>(key >> 32),
    static_cast<uint32_t>(key)
  };
}

} // namespace

EventLoop::EventLoop(UniqueFd epoll_fd, UniqueFd stop_fd, HwContext hw,
                     std::vector<StreamListener> listeners, std::vector<StreamState> streams,
                     size_t pool_slots, uint64_t session_start,
                     uint32_t frame_width, uint32_t frame_height)
  : m_epoll_fd(std::move(epoll_fd)),
    m_stop_fd(std::move(stop_fd)),
    m_hw(std::move(hw)),
    m_listeners(std::move(listeners)),
    m_pool(std::make_unique<FramesetPool>(m_listeners.size(), pool_slots, session_start,
                                         frame_width, frame_height)),
    m_streams(std::move(streams)),
    m_events(MAX_EVENTS) {}

Result<EventLoop> EventLoop::open(const Config& conf, uint64_t session_start) {
  UniqueFd epoll_fd{epoll_create1(EPOLL_CLOEXEC)};
  if (!epoll_fd.valid())
    return std::unexpected(errno_error("failed to create epoll fd"));

  UniqueFd stop_fd{eventfd(0, EFD_CLOEXEC)};
  if (!stop_fd.valid())
    return std::unexpected(errno_error("failed to create stop eventfd"));

  Result<HwContext> hw = HwContext::open();
  if (!hw)
    return std::unexpected(hw.error());

  std::vector<StreamListener> listeners;
  std::vector<StreamState> streams;
  listeners.reserve(conf.cameras.size());
  streams.reserve(conf.cameras.size());

  for (const Camera& cam : conf.cameras) {
    Result<StreamListener> listener =
      StreamListener::open(cam.tcp_port, cam.name);
    if (!listener)
      return std::unexpected(listener.error());

    Result<Decoder> decoder =
      Decoder::open(conf.stream, DECODE_SURFACES, *hw);
    if (!decoder)
      return std::unexpected(decoder.error());

    listeners.push_back(std::move(*listener));
    streams.push_back(StreamState{
      UniqueFd{}, std::vector<uint8_t>(MAX_PAYLOAD_SIZE), {}, 0, 0, 0, std::move(*decoder)
    });
  }

  EventLoop loop(std::move(epoll_fd), std::move(stop_fd), std::move(*hw),
                 std::move(listeners), std::move(streams), POOL_SLOTS, session_start,
                 conf.stream.frame_width, conf.stream.frame_height);

  Result<void> watched =
    loop.watch(loop.m_stop_fd.get(), encode(SourceKind::stop, 0));
  if (!watched)
    return std::unexpected(watched.error());

  for (size_t i = 0; i < loop.m_listeners.size(); i += 1) {
    watched = loop.watch(loop.m_listeners[i].fd(), encode(SourceKind::listener, i));
    if (!watched)
      return std::unexpected(watched.error());
  }

  return loop;
}

Result<void> EventLoop::run() {
  m_stopping = false;

  while (!m_stopping) {
    Result<std::span<const epoll_event>> events = wait();
    if (!events)
      return std::unexpected(events.error());

    Result<void> serviced = service(*events);
    if (!serviced)
      return std::unexpected(serviced.error());
  }

  return {};
}

Result<std::span<const epoll_event>> EventLoop::wait() {
  for (;;) {
    int ready = epoll_wait(m_epoll_fd.get(), m_events.data(), m_events.size(), -1);
    if (ready >= 0)
      return std::span<const epoll_event>(m_events.data(), static_cast<size_t>(ready));

    // a signal delivered to this thread is not a failure, just resume waiting
    if (errno != EINTR)
      return std::unexpected(errno_error("epoll_wait failed"));
  }
}

Result<void> EventLoop::service(std::span<const epoll_event> events) {
  for (const epoll_event& event : events) {
    // only fatal errors reach here now, per stream ones are absorbed below
    Result<void> handled = dispatch(event.data.u64);
    if (!handled)
      return std::unexpected(handled.error());
  }

  return {};
}

Result<void> EventLoop::dispatch(uint64_t key) {
  const EventSource source = decode(key);

  switch (source.kind) {
    case SourceKind::stop:
      m_stopping = true;
      return {};

    case SourceKind::listener:
      return accept_camera(source.index)
        .or_else([this, i = source.index](Error err) { return absorb_stream_error(i, std::move(err)); });

    case SourceKind::stream:
      return read_camera(source.index);
  }

  return std::unexpected(invalid("unknown epoll event source"));
}

Result<void> EventLoop::watch(int fd, uint64_t key) const {
  epoll_event ev{};
  ev.events = EPOLLIN;
  ev.data.u64 = key;

  if (epoll_ctl(m_epoll_fd.get(), EPOLL_CTL_ADD, fd, &ev) < 0)
    return std::unexpected(errno_error("failed to add fd to epoll set"));

  return {};
}

Result<void> EventLoop::accept_camera(size_t index) {
  Result<UniqueFd> conn = m_listeners[index].accept();
  if (!conn)
    return std::unexpected(conn.error());

  Result<void> watched = watch(conn->get(), encode(SourceKind::stream, index));
  if (!watched)
    return std::unexpected(watched.error());

  std::printf("[%s] connected\n", m_listeners[index].name().c_str());
  m_streams[index].fd = std::move(*conn);
  return {};
}

// a camera failing is not a reason to stop serving the others, so everything
// below this point is absorbed rather than propagated. the listener stays
// registered, so a dropped camera rejoins on its next connect attempt.
Result<void> EventLoop::read_camera(size_t index) {
  if (m_streams[index].expected == 0)
    return read_header(index)
      .or_else([this, index](Error err) { return absorb_stream_error(index, std::move(err)); });

  return read_payload(index)
    .and_then([this, index] { return submit_frame(index); })
    .and_then([this, index] { return drain_frames(index); })
    .or_else([this, index](Error err) { return absorb_stream_error(index, std::move(err)); });
}

void EventLoop::drop_camera(size_t index) {
  StreamState& stream = m_streams[index];
  if (!stream.fd.valid())
    return;

  epoll_ctl(m_epoll_fd.get(), EPOLL_CTL_DEL, stream.fd.get(), nullptr);
  stream.fd.reset();
  stream.filled = 0;
  stream.expected = 0;
}

// the one place a stream is torn down. a desynced stream cannot be resynced in
// place and a failed decoder keeps failing, so anything short of "try again"
// means dropping the camera and letting it reconnect clean.
Result<void> EventLoop::absorb_stream_error(size_t index, Error err) {
  if (is_retry(err))
    return {};

  if (is_closed(err))
    std::printf("[%s] disconnected\n", m_listeners[index].name().c_str());
  else
    std::printf("[%s] dropped: %s: %s\n",
                m_listeners[index].name().c_str(),
                err.detail.c_str(),
                err.ec.message().c_str());

  drop_camera(index);
  return {};
}

// reads at most the bytes still missing, so a read never crosses into the
// next frame. whatever else arrived stays in the socket buffer and wakes us
// again, which keeps every read aligned to a frame boundary.
Result<size_t> EventLoop::fill(size_t index, uint8_t* dst, size_t want) {
  StreamState& stream = m_streams[index];

  ssize_t got = read(stream.fd.get(), dst + stream.filled, want - stream.filled);
  if (got < 0) {
    // nothing left to read right now, epoll will wake us when there is
    if (errno == EAGAIN || errno == EWOULDBLOCK)
      return stream.filled;

    return std::unexpected(errno_error("failed to read from stream"));
  }

  if (got == 0)
    return std::unexpected(closed());

  stream.filled += static_cast<size_t>(got);
  return stream.filled;
}

Result<void> EventLoop::read_header(size_t index) {
  StreamState& stream = m_streams[index];

  Result<size_t> filled = fill(index, stream.header, FRAME_HEADER_SIZE);
  if (!filled)
    return std::unexpected(filled.error());

  if (*filled < FRAME_HEADER_SIZE)
    return std::unexpected(retry());

  const FrameHeader header = read_frame_header(stream.header);
  if (header.size == 0 || header.size > MAX_PAYLOAD_SIZE)
    return std::unexpected(invalid("frame size outside protocol bounds"));

  stream.timestamp = header.timestamp;
  stream.expected = header.size;
  stream.filled = 0;
  return {};
}

Result<void> EventLoop::read_payload(size_t index) {
  StreamState& stream = m_streams[index];

  Result<size_t> filled = fill(index, stream.buffer.data(), stream.expected);
  if (!filled)
    return std::unexpected(filled.error());

  if (*filled < stream.expected)
    return std::unexpected(retry());

  return {};
}

Result<void> EventLoop::submit_frame(size_t index) {
  StreamState& stream = m_streams[index];

  Result<void> sent = stream.decoder.send_packet(
    std::span<const uint8_t>(stream.buffer.data(), stream.expected),
    stream.timestamp
  );
  if (!sent)
    return std::unexpected(sent.error());

  stream.filled = 0;
  stream.expected = 0;
  return {};
}

// the decoder stops accepting input once its queue fills, so every send has to
// be followed by draining whatever became ready
Result<void> EventLoop::drain_frames(size_t index) {
  StreamState& stream = m_streams[index];

  for (;;) {
    Result<std::optional<DecodedFrame>> frame = stream.decoder.receive_frame();
    if (!frame)
      return std::unexpected(frame.error());

    if (!frame->has_value())
      return {};


    m_pool->push(index, std::move(**frame));

  }
}

Result<void> EventLoop::stop() const {
  uint64_t one = 1;
  if (write(m_stop_fd.get(), &one, sizeof(one)) != sizeof(one))
    return std::unexpected(errno_error("failed to signal stop eventfd"));

  return {};
}

} // namespace mocap
