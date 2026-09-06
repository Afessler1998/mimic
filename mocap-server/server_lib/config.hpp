#ifndef MOCAP_CONFIG_HPP
#define MOCAP_CONFIG_HPP

#include <cstdint>
#include <expected>
#include <netinet/in.h>
#include <filesystem>
#include <string>
#include <vector>

#include "error.hpp"

namespace mocap {

struct StreamParams {
  uint32_t frame_width;
  uint32_t frame_height;
  uint32_t fps;
};

struct Camera {
  std::string name;
  uint8_t id;
  in_addr eth_ip;
  uint16_t tcp_port;
};

struct Control {
  in_addr broadcast;
  uint16_t port;
};

struct Config {
  StreamParams stream;
  Control control;
  std::vector<Camera> cameras;
};

Result<Config> parse_config(const std::filesystem::path& path);

} // namespace mocap

#endif // MOCAP_CONFIG_HPP
