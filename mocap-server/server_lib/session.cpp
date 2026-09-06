#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <utility>

#include "config.hpp"
#include "control.hpp"
#include "event_loop.hpp"
#include "session.hpp"

namespace mocap {

// everything a running session owns. lives on the heap so its address is
// stable across moves, which the loop thread relies on.
struct Session::State {
  Config conf;
  ControlSocket control;
  EventLoop loop;
  std::thread thread;
};

namespace {

// cameras need roughly a second between receiving the timestamp and having a
// TCP stream up, so the start time is placed far enough ahead for every camera
// to be armed before it arrives
constexpr std::chrono::seconds START_DELAY{2};

uint64_t start_timestamp() {
  std::chrono::nanoseconds now =
    std::chrono::system_clock::now().time_since_epoch();
  return static_cast<uint64_t>((now + START_DELAY).count());
}

} // namespace

Session::Session(std::unique_ptr<State> state) : m_state(std::move(state)) {}

Session::Session(Session&& other) noexcept = default;
Session& Session::operator=(Session&& other) noexcept = default;

Result<Session> Session::start(const std::filesystem::path& config_path) {
  Result<Config> conf = parse_config(config_path);
  if (!conf)
    return std::unexpected(conf.error());

  Result<ControlSocket> control =
    ControlSocket::open(conf->control.broadcast, conf->control.port);
  if (!control)
    return std::unexpected(control.error());

  // computed before the loop opens so the frameset builder can reject frames
  // captured before this session began
  const uint64_t timestamp = start_timestamp();

  // listeners must be bound before the cameras are told to start, otherwise
  // their first connect is refused and they back off for a retry interval
  Result<EventLoop> loop = EventLoop::open(*conf, timestamp);
  if (!loop)
    return std::unexpected(loop.error());

  Result<void> started = control->broadcast_start(timestamp);
  if (!started)
    return std::unexpected(started.error());

  std::unique_ptr<State> state = std::make_unique<State>(
    std::move(*conf), std::move(*control), std::move(*loop), std::thread{}
  );

  // the state is heap allocated and never relocates, so the thread can hold a
  // raw pointer to it even as the Session that owns it moves
  State* running = state.get();
  state->thread = std::thread([running] { (void)running->loop.run(); });

  return Session(std::move(state));
}

Session::~Session() {
  if (!m_state)
    return;

  Result<void> stopped = m_state->control.broadcast_stop();

  // the eventfd is the only way to wake the loop, so if the write fails the
  // join below would block forever. not exit: the loop thread is still
  // joinable, so running its destructor on the way out would terminate anyway.
  Result<void> signalled = m_state->loop.stop();
  if (!signalled) {
    std::fprintf(stderr, "[session] cannot signal the event loop to stop: %s: %s\n",
                 signalled.error().detail.c_str(),
                 signalled.error().ec.message().c_str());
    std::abort();
  }

  if (m_state->thread.joinable())
    m_state->thread.join();

  // the only failure worth reporting: the cameras were never told to stop, so
  // they keep capturing until their stream write fails
  if (!stopped)
    std::fprintf(stderr, "[session] stop did not reach the cameras: %s: %s\n",
                 stopped.error().detail.c_str(),
                 stopped.error().ec.message().c_str());
}

const Config& Session::config() const {
  return m_state->conf;
}

std::optional<Frameset> Session::try_acquire_frameset() {
  return m_state->loop.pool().try_acquire();
}

void Session::release_frameset(const Frameset& set) {
  m_state->loop.pool().release(set.slot);
}

} // namespace mocap
