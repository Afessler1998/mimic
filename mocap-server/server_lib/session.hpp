#ifndef MOCAP_SESSION_HPP
#define MOCAP_SESSION_HPP

#include <cstdint>
#include <expected>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>

#include "config.hpp"
#include "error.hpp"
#include "visibility.hpp"

namespace mocap {

// One camera's decoded frame, still resident on the GPU. NV12, chroma follows
// the luma plane at pitch * height.
struct FrameView {
  uint8_t camera_id;
  const uint8_t* device_ptr;
  uint32_t width;
  uint32_t height;
  uint32_t pitch;    // padded wider than width for alignment, so copies stride
};

// A set of frames that share a capture timestamp. May hold fewer than one per
// camera if a camera missed the window, so read frames.size() rather than
// assuming a full set. Every acquired set must be handed back to
// release_frameset or its GPU surfaces are never reused.
struct Frameset {
  uint64_t timestamp;
  std::span<const FrameView> frames;
  uint32_t slot;   // opaque, hand back to release_frameset
};

// A running capture session. Starting one reads the camera config and
// broadcasts a start timestamp to every camera at once, so they arm their
// capture timers against the same absolute time and begin in lockstep
// regardless of delivery jitter. Letting it go out of scope stops them.
//
// Framesets may be acquired from a different thread than the one holding the
// session, but the session itself is not thread safe to move or destroy.
//
// The destructor cannot report failure, so a stop that never reaches the
// cameras is written to stderr. It is the only way a stop fails, and nothing
// a caller would do differently.
class MOCAP_API Session {
public:
  static Result<Session> start(const std::filesystem::path& config_path);

  Session(Session&& other) noexcept;
  Session& operator=(Session&& other) noexcept;
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  ~Session();

  // The config this session was started from, valid for its lifetime.
  const Config& config() const;

  // The oldest ready frameset, or nullopt if none is ready. Every acquired set
  // must be handed back to release_frameset or its GPU surfaces are never
  // reused.
  std::optional<Frameset> try_acquire_frameset();
  void release_frameset(const Frameset& set);

private:
  // held by pointer so this header needs no decoder, socket or epoll types,
  // which keeps ffmpeg out of anything that links the library
  struct State;
  explicit Session(std::unique_ptr<State> state);

  std::unique_ptr<State> m_state;
};


} // namespace mocap

#endif // MOCAP_SESSION_HPP
