#include <cstdio>
#include <utility>

#include "frameset_pool.hpp"

namespace mocap {

namespace {

// two views is the floor for triangulation, so a set holding at least this
// many is still worth handing out
constexpr size_t MIN_USEFUL_FRAMES = 2;

} // namespace

FramesetPool::FramesetPool(size_t cameras, size_t slots, uint64_t session_start,
                           uint32_t frame_width, uint32_t frame_height)
  : m_slots(slots), m_camera_count(cameras),
    m_session_start(session_start),
    m_frame_width(frame_width), m_frame_height(frame_height) {
  for (uint32_t i = 0; i < slots; i += 1) {
    m_slots[i].surfaces.reserve(cameras);
    m_slots[i].views.reserve(cameras);
    m_unused_slots.push(i);
  }
}

void FramesetPool::clear_slot(uint32_t slot) {
  m_slots[slot].surfaces.clear();   // releases the surfaces back to the decoders
  m_slots[slot].views.clear();
  m_slots[slot].timestamp = 0;
}

void FramesetPool::recycle_slot(uint32_t slot) {
  clear_slot(slot);
  m_unused_slots.push(slot);
}

std::optional<uint32_t> FramesetPool::claim_slot() {
  if (std::optional<uint32_t> slot = m_unused_slots.try_pop())
    return slot;

  // the consumer is behind. drop the oldest completed set rather than the
  // newest, so it resumes on current data instead of working through a backlog.
  if (std::optional<uint32_t> oldest = m_completed_sets.try_pop()) {
    std::printf("[frameset] consumer behind, dropped set %lu\n", m_slots[*oldest].timestamp);
    clear_slot(*oldest);
    return oldest;
  }

  // every slot is held by the consumer
  return std::nullopt;
}

void FramesetPool::emit_open_set() {
  Slot& set = m_slots[*m_open_set];

  if (set.surfaces.size() >= MIN_USEFUL_FRAMES) {
    if (set.surfaces.size() < m_camera_count)
      std::printf("[frameset] partial set at %lu, %zu of %zu cameras\n",
                  set.timestamp, set.surfaces.size(), m_camera_count);
    m_completed_sets.push(*m_open_set);
  } else {
    std::printf("[frameset] dropped set at %lu, only %zu camera(s) delivered\n",
                set.timestamp, set.surfaces.size());
    recycle_slot(*m_open_set);
  }

  m_open_set.reset();
}

void FramesetPool::push(size_t camera_index, DecodedFrame frame) {
  const uint64_t timestamp = frame.timestamp();

  if (timestamp < m_session_start) {
    std::printf("[frameset] discarded pre session frame from cam%zu\n", camera_index);
    return;
  }

  // a newer frame means this camera has moved on, so whatever the current set
  // is still missing has run out of time. emit what we have and start over.
  if (m_open_set && timestamp > m_slots[*m_open_set].timestamp)
    emit_open_set();

  if (!m_open_set) {
    m_open_set = claim_slot();
    if (!m_open_set)
      return;   // consumer holds every slot, drop the frame

    m_slots[*m_open_set].timestamp = timestamp;
  }

  Slot& set = m_slots[*m_open_set];

  // a straggler from a set that has already been emitted
  if (timestamp < set.timestamp)
    return;

  set.views.push_back(FrameView{
    static_cast<uint8_t>(camera_index),
    frame.device_ptr(),
    m_frame_width,
    m_frame_height,
    frame.pitch()
  });
  set.surfaces.push_back(std::move(frame));

  if (set.surfaces.size() == m_camera_count)
    emit_open_set();
}

std::optional<Frameset> FramesetPool::try_acquire() {
  std::optional<uint32_t> slot = m_completed_sets.try_pop();
  if (!slot)
    return std::nullopt;

  // the slot is ours now, so reading it needs no lock
  return Frameset{
    m_slots[*slot].timestamp,
    std::span<const FrameView>(m_slots[*slot].views),
    *slot
  };
}

void FramesetPool::release(uint32_t slot) {
  // clear_slot runs av_frame_free per surface. we still own the slot here, so
  // that happens outside any lock the event loop could be waiting on.
  recycle_slot(slot);
}

} // namespace mocap
