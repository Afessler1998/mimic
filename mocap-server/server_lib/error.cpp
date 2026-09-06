#include <cerrno>
#include <utility>

#include "error.hpp"

namespace mocap {

Error errno_error(std::string detail) {
  return Error{
    std::error_code(errno ? errno : EIO, std::generic_category()),
    std::move(detail)
  };
}

Error retry() {
  return Error{
    std::make_error_code(std::errc::resource_unavailable_try_again),
    "incomplete"
  };
}

bool is_retry(const Error& err) {
  return err.ec == std::errc::resource_unavailable_try_again;
}

Error closed() {
  return Error{
    std::make_error_code(std::errc::connection_reset),
    "stream closed"
  };
}

bool is_closed(const Error& err) {
  return err.ec == std::errc::connection_reset;
}

Error invalid(std::string detail) {
  return Error{
    std::make_error_code(std::errc::invalid_argument),
    std::move(detail)
  };
}

} // namespace mocap
