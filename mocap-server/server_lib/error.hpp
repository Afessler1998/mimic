#ifndef MOCAP_ERROR_HPP
#define MOCAP_ERROR_HPP

#include <expected>
#include <string>
#include <system_error>

namespace mocap {

struct Error {
  std::error_code ec;
  std::string detail;
};

// Every fallible operation in this codebase fails the same way, so the error
// half never varies and spelling it out at each site is noise.
template <typename T>
using Result = std::expected<T, Error>;

Error errno_error(std::string detail);

// not a failure: the operation made no progress and should be retried when
// its fd next wakes up. short circuits an and_then chain without ending it.
Error retry();
bool is_retry(const Error& err);

// not a failure either: the stream is gone and has already been torn down.
// the listener stays registered, so the camera is re-accepted when it retries.
Error closed();
bool is_closed(const Error& err);
Error invalid(std::string detail);

} // namespace mocap

#endif // MOCAP_ERROR_HPP
