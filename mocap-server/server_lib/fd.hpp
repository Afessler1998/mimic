#ifndef MOCAP_FD_HPP
#define MOCAP_FD_HPP

namespace mocap {

// Owns a file descriptor. Move-only: copying would give two owners one
// descriptor, and the second close would hit whatever fd number the kernel
// had since recycled.
class UniqueFd {
public:
  UniqueFd() = default;
  explicit UniqueFd(int fd);
  ~UniqueFd();

  UniqueFd(UniqueFd&& other) noexcept;
  UniqueFd& operator=(UniqueFd&& other) noexcept;
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;

  int get() const { return m_fd; }
  bool valid() const { return m_fd >= 0; }
  void reset();

private:
  int m_fd = -1;
};

} // namespace mocap

#endif // MOCAP_FD_HPP
