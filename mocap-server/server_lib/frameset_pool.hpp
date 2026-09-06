#ifndef MOCAP_FRAMESET_POOL_HPP
#define MOCAP_FRAMESET_POOL_HPP

#include <cstdint>
#include <optional>
#include <vector>

#include "decoder.hpp"
#include "locked_queue.hpp"
#include "session.hpp"

namespace mocap {

// Groups decoded frames into timestamp matched sets and hands them to the
// library consumer.
//
// Slots are preallocated and recycled through two queues, so steady state does
// no allocation. A slot is in exactly one of three places: available, ready, or
// held by the consumer between try_acquire and release.
//
// Crossed by two threads, the event loop pushing and the consumer acquiring,
// but only the two queues are shared. A slot is in exactly one place at a
// time, so whoever holds it owns its contents outright and needs no lock to
// touch them.
//
// Locking the queues rather than going lock free is what buys the ability to
// drop the oldest completed set: an SPSC pair cannot, because dropping means
// the producer popping the consumer's end.
class FramesetPool {
public:
  FramesetPool(size_t cameras, size_t slots, uint64_t session_start,
               uint32_t frame_width, uint32_t frame_height);

  // event loop thread
  void push(size_t camera_index, DecodedFrame frame);

  // consumer thread
  std::optional<Frameset> try_acquire();
  void release(uint32_t slot);

private:
  struct Slot {
    uint64_t timestamp = 0;
    std::vector<DecodedFrame> surfaces;   // hold the GPU surfaces open
    std::vector<FrameView> views;      // what the consumer reads
  };

  std::optional<uint32_t> claim_slot();
  void emit_open_set();
  void clear_slot(uint32_t slot);     // empty it, leaving ownership with the caller
  void recycle_slot(uint32_t slot);   // empty it and return it to the free list

  std::vector<Slot> m_slots;
  LockedQueue<uint32_t> m_unused_slots;
  LockedQueue<uint32_t> m_completed_sets;
  std::optional<uint32_t> m_open_set;
  size_t m_camera_count;

  // cameras can carry a frame from a previous session across a reconnect, and
  // every camera agrees on its stale timestamp, so it forms a set that looks
  // valid. nothing captured before the session started is real.
  uint64_t m_session_start;
  uint32_t m_frame_width;
  uint32_t m_frame_height;
};

} // namespace mocap

#endif // MOCAP_FRAMESET_POOL_HPP
