#include <cstring>
#include <sys/socket.h>
#include <utility>

#include "control.hpp"

namespace mocap {

namespace {
constexpr const char* STOP_SENTINEL = "STOP";
}

ControlSocket::ControlSocket(UniqueFd fd, sockaddr_in dest)
  : m_fd(std::move(fd)), m_dest(dest) {}

Result<ControlSocket> ControlSocket::open(in_addr broadcast_addr, uint16_t port) {
  UniqueFd fd{socket(AF_INET, SOCK_DGRAM, 0)};
  if (!fd.valid())
    return std::unexpected(errno_error("failed to create control socket"));

  int enable = 1;
  if (setsockopt(fd.get(), SOL_SOCKET, SO_BROADCAST, &enable, sizeof(enable)) < 0)
    return std::unexpected(errno_error("failed to enable broadcast on control socket"));

  sockaddr_in dest{};
  dest.sin_family = AF_INET;
  dest.sin_port = htons(port);
  dest.sin_addr = broadcast_addr;

  return ControlSocket(std::move(fd), dest);
}

Result<void> ControlSocket::broadcast(const void* msg, size_t len) const {
  ssize_t sent = sendto(
    m_fd.get(),
    msg,
    len,
    0,
    reinterpret_cast<const sockaddr*>(&m_dest),
    sizeof(m_dest)
  );
  if (sent < 0)
    return std::unexpected(errno_error("failed to broadcast control message"));

  if (static_cast<size_t>(sent) != len)
    return std::unexpected(invalid("control message truncated in transit"));

  return {};
}

Result<void> ControlSocket::broadcast_start(uint64_t timestamp) const {
  return broadcast(&timestamp, sizeof(timestamp));
}

Result<void> ControlSocket::broadcast_stop() const {
  return broadcast(STOP_SENTINEL, std::strlen(STOP_SENTINEL));
}
} // namespace mocap
