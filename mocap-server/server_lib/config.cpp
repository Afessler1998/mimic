#include <arpa/inet.h>
#include <cctype>
#include <cerrno>
#include <format>
#include <set>
#include <string_view>
#include <fstream>
#include <iterator>
#include <optional>
#include <utility>

#define TOML_EXCEPTIONS 0
#include <toml++/toml.h>

#include "config.hpp"

namespace mocap {

namespace fs = std::filesystem;

static Result<std::string> read_file(const fs::path& path) {
  errno = 0;
  std::ifstream file(path, std::ios::binary);
  if (!file)
    return std::unexpected(errno_error(path.string()));

  std::string contents{
    std::istreambuf_iterator<char>(file),
    std::istreambuf_iterator<char>()
  };
  if (file.bad())
    return std::unexpected(errno_error(path.string()));

  return contents;
}

static Result<in_addr> parse_ipv4(const std::string& text) {
  in_addr addr;
  if (inet_pton(AF_INET, text.c_str(), &addr) != 1)
    return std::unexpected(invalid(std::format("invalid ipv4 address: {}", text)));

  return addr;
}

// names are the wire identity for a Camera, expected format is rpicamXX
static bool valid_camera_name(std::string_view name) {
  constexpr std::string_view prefix = "rpicam";
  return name.size() == prefix.size() + 2
      && name.starts_with(prefix)
      && std::isdigit(static_cast<unsigned char>(name[6]))
      && std::isdigit(static_cast<unsigned char>(name[7]));
}

static Result<StreamParams> parse_stream_params(const toml::table& root) {
  const toml::node_view<const toml::node> params = root["stream_params"];

  std::optional<uint32_t> width = params["frame_width"].value<uint32_t>();
  std::optional<uint32_t> height = params["frame_height"].value<uint32_t>();
  std::optional<uint32_t> fps = params["fps"].value<uint32_t>();
  if (!width || !height || !fps)
    return std::unexpected(invalid("stream_params: missing or invalid field"));

  return StreamParams{*width, *height, *fps};
}

static Result<Control> parse_control(const toml::table& root) {
  const toml::node_view<const toml::node> control = root["control"];

  std::optional<std::string> broadcast = control["broadcast"].value<std::string>();
  std::optional<uint16_t> port = control["port"].value<uint16_t>();
  if (!broadcast || !port)
    return std::unexpected(invalid("control: missing or invalid field"));

  Result<in_addr> addr = parse_ipv4(*broadcast);
  if (!addr)
    return std::unexpected(addr.error());

  return Control{*addr, *port};
}

static Result<Camera> parse_camera(const toml::table& entry) {
  std::optional<std::string> name = entry["name"].value<std::string>();
  std::optional<uint8_t> id = entry["id"].value<uint8_t>();
  std::optional<std::string> eth_ip = entry["eth_ip"].value<std::string>();
  std::optional<uint16_t> tcp_port = entry["tcp_port"].value<uint16_t>();
  if (!name || !id || !eth_ip || !tcp_port)
    return std::unexpected(invalid("missing or invalid field"));

  if (!valid_camera_name(*name))
    return std::unexpected(invalid(std::format("name must match rpicamXX, got {}", *name)));

  Result<in_addr> eth = parse_ipv4(*eth_ip);
  if (!eth)
    return std::unexpected(eth.error());

  return Camera{*name, *id, *eth, *tcp_port};
}

// invariants that only make sense across the whole set: ids index the set,
// and no two cameras share a stream port. the control port is deliberately
// shared, and tcp/udp are separate port spaces, so it needs no cross-check.
static Result<void> validate_cameras(const std::vector<Camera>& cameras) {
  std::set<uint16_t> ports;

  for (size_t i = 0; i < cameras.size(); i += 1) {
    if (static_cast<size_t>(cameras[i].id) != i)
      return std::unexpected(invalid(std::format(
        "cameras[{}]: id must be {}, got {}", i, i, cameras[i].id)));

    if (!ports.insert(cameras[i].tcp_port).second)
      return std::unexpected(invalid(std::format(
        "cameras[{}]: duplicate tcp port {}", i, cameras[i].tcp_port)));
  }

  return {};
}

static Result<std::vector<Camera>> parse_cameras(const toml::table& root) {
  const toml::array* entries = root["cameras"].as_array();
  if (!entries || entries->empty())
    return std::unexpected(invalid("cameras: missing or empty"));

  std::vector<Camera> cameras;
  cameras.reserve(entries->size());

  for (size_t i = 0; i < entries->size(); i += 1) {
    const toml::table* entry = entries->at(i).as_table();
    if (!entry)
      return std::unexpected(invalid(std::format("cameras[{}]: not a table", i)));

    Result<Camera> cam = parse_camera(*entry);
    if (!cam)
      return std::unexpected(invalid(std::format("cameras[{}]: {}", i, cam.error().detail)));

    cameras.push_back(std::move(*cam));
  }

  Result<void> valid = validate_cameras(cameras);
  if (!valid)
    return std::unexpected(valid.error());

  return cameras;
}

static Result<Config> parse_toml(std::string contents) {
  toml::parse_result result = toml::parse(contents);
  if (!result)
    return std::unexpected(invalid(std::string(result.error().description())));

  const toml::table& root = result.table();

  Result<StreamParams> stream = parse_stream_params(root);
  if (!stream)
    return std::unexpected(stream.error());

  Result<Control> control = parse_control(root);
  if (!control)
    return std::unexpected(control.error());

  Result<std::vector<Camera>> cameras = parse_cameras(root);
  if (!cameras)
    return std::unexpected(cameras.error());

  return Config{*stream, *control, std::move(*cameras)};
}

Result<Config> parse_config(const fs::path& path) {
  return read_file(path).and_then(parse_toml);
}

} // namespace mocap
