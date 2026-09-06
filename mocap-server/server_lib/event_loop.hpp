#ifndef MOCAP_EVENT_LOOP_HPP
#define MOCAP_EVENT_LOOP_HPP

#include <cstdint>
#include <expected>
#include <memory>
#include <span>
#include <sys/epoll.h>
#include <vector>

#include "config.hpp"
#include "decoder.hpp"
#include "frameset_pool.hpp"
#include "error.hpp"
#include "fd.hpp"
#include "stream.hpp"

namespace mocap {

// Per-Camera receive state. Streams interleave, so a partial frame from one
// Camera has to survive reads from the others, the buffer cannot be shared.
struct StreamState {
  UniqueFd fd;
  std::vector<uint8_t> buffer;
  uint8_t header[12];

  // bytes of the header or payload gathered so far. reads are bounded to
  // exactly what is still missing, so a read never crosses a frame boundary
  size_t filled = 0;
  uint32_t expected = 0;   // payload size, 0 while still gathering the header
  uint64_t timestamp = 0;

  Decoder decoder;
};

// Single-threaded epoll loop owning every Camera stream. One thread services
// all cameras, so the work done per wakeup needs to stay bounded.
class EventLoop {
public:
  static Result<EventLoop> open(const Config& conf, uint64_t session_start);

  EventLoop(EventLoop&&) noexcept = default;
  EventLoop& operator=(EventLoop&&) noexcept = default;

  // blocks until stop() is called
  Result<void> run();

  // wakes run() from another thread
  Result<void> stop() const;

  FramesetPool& pool() { return *m_pool; }

private:
  EventLoop(UniqueFd epoll_fd, UniqueFd stop_fd, HwContext hw,
            std::vector<StreamListener> listeners, std::vector<StreamState> streams,
            size_t pool_slots, uint64_t session_start,
            uint32_t frame_width, uint32_t frame_height);

  Result<void> watch(int fd, uint64_t key) const;

  Result<std::span<const epoll_event>> wait();
  Result<void> service(std::span<const epoll_event> events);
  Result<void> dispatch(uint64_t key);

  Result<void> accept_camera(size_t index);
  Result<void> read_camera(size_t index);
  Result<void> read_header(size_t index);
  Result<void> read_payload(size_t index);
  Result<void> submit_frame(size_t index);
  Result<void> drain_frames(size_t index);
  Result<size_t> fill(size_t index, uint8_t* dst, size_t want);

  void drop_camera(size_t index);
  Result<void> absorb_stream_error(size_t index, Error err);

  UniqueFd m_epoll_fd;
  UniqueFd m_stop_fd;
  HwContext m_hw;
  // declared before m_pool: members initialise in declaration order, and the
  // pool is sized from m_listeners.size()
  std::vector<StreamListener> m_listeners;

  // by pointer so EventLoop stays movable (the pool holds a mutex) and so the
  // consumer thread's reference survives EventLoop being moved into the session
  std::unique_ptr<FramesetPool> m_pool;
  std::vector<StreamState> m_streams;
  std::vector<epoll_event> m_events;
  bool m_stopping = false;
};

} // namespace mocap

#endif // MOCAP_EVENT_LOOP_HPP
