#include <unistd.h>
#include <utility>

#include "fd.hpp"

namespace mocap {

UniqueFd::UniqueFd(int fd) : m_fd(fd) {}

UniqueFd::~UniqueFd() {
  reset();
}

UniqueFd::UniqueFd(UniqueFd&& other) noexcept
  : m_fd(std::exchange(other.m_fd, -1)) {}

UniqueFd& UniqueFd::operator=(UniqueFd&& other) noexcept {
  if (this != &other) {
    reset();
    m_fd = std::exchange(other.m_fd, -1);
  }
  return *this;
}

void UniqueFd::reset() {
  if (m_fd >= 0) {
    close(m_fd);
    m_fd = -1;
  }
}

} // namespace mocap
