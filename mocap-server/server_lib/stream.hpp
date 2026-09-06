#ifndef MOCAP_STREAM_HPP
#define MOCAP_STREAM_HPP

#include <cstdint>
#include <expected>
#include <string>

#include "error.hpp"
#include "fd.hpp"

namespace mocap {

// Listening socket for one Camera. Cameras dial out to the server, so each
// Camera's identity is established by which listener accepted it.
class StreamListener {
public:
  static Result<StreamListener> open(uint16_t port, std::string name);

  Result<UniqueFd> accept() const;

  int fd() const { return m_fd.get(); }
  const std::string& name() const { return m_name; }

private:
  StreamListener(UniqueFd fd, std::string name);

  UniqueFd m_fd;
  std::string m_name;
};

} // namespace mocap

#endif // MOCAP_STREAM_HPP
