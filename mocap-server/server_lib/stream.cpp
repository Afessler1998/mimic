#include <netinet/in.h>
#include <sys/socket.h>
#include <utility>

#include "stream.hpp"

namespace mocap {

namespace {
constexpr int LISTEN_BACKLOG = 1;
}

StreamListener::StreamListener(UniqueFd fd, std::string name)
  : m_fd(std::move(fd)), m_name(std::move(name)) {}

Result<StreamListener> StreamListener::open(uint16_t port, std::string name) {
  UniqueFd fd{socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0)};
  if (!fd.valid())
    return std::unexpected(errno_error("failed to create listen socket for " + name));

  // without this a crashed session leaves the port in TIME_WAIT and the next
  // session_start fails to bind
  int enable = 1;
  if (setsockopt(fd.get(), SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(enable)) < 0)
    return std::unexpected(errno_error("failed to set SO_REUSEADDR for " + name));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);

  if (bind(fd.get(), reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) < 0)
    return std::unexpected(errno_error("failed to bind listen socket for " + name));

  if (listen(fd.get(), LISTEN_BACKLOG) < 0)
    return std::unexpected(errno_error("failed to listen for " + name));

  return StreamListener(std::move(fd), std::move(name));
}

// non blocking, so a read that outruns the data returns EAGAIN instead of
// parking the loop thread and stalling every other camera
Result<UniqueFd> StreamListener::accept() const {
  UniqueFd conn{::accept4(m_fd.get(), nullptr, nullptr, SOCK_NONBLOCK | SOCK_CLOEXEC)};
  if (!conn.valid())
    return std::unexpected(errno_error("failed to accept connection for " + m_name));

  return conn;
}

} // namespace mocap
