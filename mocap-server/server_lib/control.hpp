#ifndef MOCAP_CONTROL_HPP
#define MOCAP_CONTROL_HPP

#include <cstdint>
#include <expected>
#include <netinet/in.h>

#include "error.hpp"
#include "fd.hpp"

namespace mocap {

// UDP socket for the session control plane. Every camera binds the same
// control port, so one datagram to the subnet broadcast address starts or
// stops the whole rig.
class ControlSocket {
public:
  static Result<ControlSocket> open(in_addr broadcast_addr, uint16_t port);

  // cameras arm their interval timer against this absolute time, so delivery
  // jitter between cameras does not affect capture alignment
  Result<void> broadcast_start(uint64_t timestamp) const;
  Result<void> broadcast_stop() const;

private:
  ControlSocket(UniqueFd fd, sockaddr_in dest);

  Result<void> broadcast(const void* msg, size_t len) const;

  UniqueFd m_fd;
  sockaddr_in m_dest;
};

} // namespace mocap

#endif // MOCAP_CONTROL_HPP
