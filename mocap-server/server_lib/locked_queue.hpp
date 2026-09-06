#ifndef MOCAP_LOCKED_QUEUE_HPP
#define MOCAP_LOCKED_QUEUE_HPP

#include <mutex>
#include <optional>
#include <queue>
#include <utility>

namespace mocap {

// A queue safe for any number of threads on either end, holding its lock only
// for the queue operation itself.
//
// There is deliberately no empty() or front(): either would be stale the
// moment it returned, and callers would have to retake the lock to act on it.
// try_pop is the only safe shape, so it is the only one offered.
template <typename T>
class LockedQueue {
public:
  void push(T value) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_queue.push(std::move(value));
  }

  std::optional<T> try_pop() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_queue.empty())
      return std::nullopt;

    T value = std::move(m_queue.front());
    m_queue.pop();
    return value;
  }

private:
  std::mutex m_mutex;
  std::queue<T> m_queue;
};

} // namespace mocap

#endif // MOCAP_LOCKED_QUEUE_HPP
