#include <cstdio>
#include <fstream>
#include <utility>

#include <NvInfer.h>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>

#include "pose_model.hpp"

namespace mocap {

namespace {

constexpr const char* INPUT_TENSOR = "input";
constexpr const char* SIMCC_X_TENSOR = "simcc_x";
constexpr const char* SIMCC_Y_TENSOR = "simcc_y";

class TrtLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING)
      std::fprintf(stderr, "[trt] %s\n", msg);
  }
};

TrtLogger g_logger;

Error cuda_error(const char* what, cudaError_t status) {
  return Error{
    std::make_error_code(std::errc::io_error),
    std::string(what) + ": " + cudaGetErrorString(status)
  };
}

// TensorRT hands back raw pointers with their own destroy semantics, so each
// one gets a deleter rather than a bare pointer nobody is responsible for.
struct TrtDelete {
  void operator()(nvinfer1::IRuntime* p) const { delete p; }
  void operator()(nvinfer1::ICudaEngine* p) const { delete p; }
  void operator()(nvinfer1::IExecutionContext* p) const { delete p; }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDelete>;

} // namespace

struct PoseModel::State {
  TrtPtr<nvinfer1::IRuntime> runtime;
  TrtPtr<nvinfer1::ICudaEngine> engine;
  TrtPtr<nvinfer1::IExecutionContext> context;

  cudaStream_t stream = nullptr;
  cudaEvent_t staged = nullptr;

  float* dev_input = nullptr;
  float* dev_simcc_x = nullptr;
  float* dev_simcc_y = nullptr;

  // pinned so the copy back does not stage through a bounce buffer
  float* host_simcc_x = nullptr;
  float* host_simcc_y = nullptr;

  cv::Matx33d transform;

  // which camera each batch slot came from, since a partial set leaves gaps
  std::vector<uint8_t> batch_cameras;
  std::vector<PoseResult> results;
};

namespace {

Result<void> check_engine_shape(const nvinfer1::ICudaEngine& engine) {
  auto dims_match = [](nvinfer1::Dims d, std::initializer_list<int> want) {
    if (d.nbDims != static_cast<int>(want.size()) + 1)
      return false;

    int i = 1;
    for (int v : want)
      if (d.d[i++] != v)
        return false;

    return true;
  };

  if (!dims_match(engine.getTensorShape(INPUT_TENSOR),
                  {INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH}))
    return std::unexpected(invalid("engine input is not 3x384x288, wrong model"));

  if (!dims_match(engine.getTensorShape(SIMCC_X_TENSOR), {NUM_KEYPOINTS, SIMCC_X_BINS}))
    return std::unexpected(invalid("engine simcc_x does not match, wrong model"));

  if (!dims_match(engine.getTensorShape(SIMCC_Y_TENSOR), {NUM_KEYPOINTS, SIMCC_Y_BINS}))
    return std::unexpected(invalid("engine simcc_y does not match, wrong model"));

  nvinfer1::Dims max =
    engine.getProfileShape(INPUT_TENSOR, 0, nvinfer1::OptProfileSelector::kMAX);
  if (max.d[0] < MAX_BATCH)
    return std::unexpected(invalid("engine max batch is below the camera count"));

  return {};
}

} // namespace

PoseModel::PoseModel(std::unique_ptr<State> state) : m_state(std::move(state)) {}
PoseModel::PoseModel(PoseModel&&) noexcept = default;
PoseModel& PoseModel::operator=(PoseModel&&) noexcept = default;

PoseModel::~PoseModel() {
  if (!m_state)
    return;

  cudaFreeHost(m_state->host_simcc_y);
  cudaFreeHost(m_state->host_simcc_x);
  cudaFree(m_state->dev_simcc_y);
  cudaFree(m_state->dev_simcc_x);
  cudaFree(m_state->dev_input);

  if (m_state->staged)
    cudaEventDestroy(m_state->staged);
  if (m_state->stream)
    cudaStreamDestroy(m_state->stream);
}

Result<PoseModel> PoseModel::load(const std::filesystem::path& engine_path,
                                  uint32_t frame_width,
                                  uint32_t frame_height) {
  std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
  if (!file)
    return std::unexpected(errno_error("failed to open " + engine_path.string()));

  const std::streamsize size = file.tellg();
  file.seekg(0);

  std::vector<char> plan(static_cast<size_t>(size));
  if (!file.read(plan.data(), size))
    return std::unexpected(errno_error("failed to read " + engine_path.string()));

  std::unique_ptr<State> state = std::make_unique<State>();

  state->runtime.reset(nvinfer1::createInferRuntime(g_logger));
  if (!state->runtime)
    return std::unexpected(invalid("failed to create TensorRT runtime"));

  state->engine.reset(
    state->runtime->deserializeCudaEngine(plan.data(), static_cast<size_t>(size)));
  if (!state->engine)
    return std::unexpected(invalid("failed to deserialize " + engine_path.string()));

  Result<void> shape = check_engine_shape(*state->engine);
  if (!shape)
    return std::unexpected(shape.error());

  state->context.reset(state->engine->createExecutionContext());
  if (!state->context)
    return std::unexpected(invalid("failed to create execution context"));

  cudaError_t status = cudaStreamCreate(&state->stream);
  if (status != cudaSuccess)
    return std::unexpected(cuda_error("failed to create stream", status));

  // timing is not needed and disabling it makes the event cheaper
  status = cudaEventCreateWithFlags(&state->staged, cudaEventDisableTiming);
  if (status != cudaSuccess)
    return std::unexpected(cuda_error("failed to create event", status));

  const size_t input_bytes = MAX_BATCH * INPUT_ELEMS_PER_IMAGE * sizeof(float);
  const size_t simcc_x_bytes = MAX_BATCH * NUM_KEYPOINTS * SIMCC_X_BINS * sizeof(float);
  const size_t simcc_y_bytes = MAX_BATCH * NUM_KEYPOINTS * SIMCC_Y_BINS * sizeof(float);

  struct Alloc { void** ptr; size_t bytes; bool pinned; const char* what; };
  const Alloc allocs[] = {
    {reinterpret_cast<void**>(&state->dev_input),     input_bytes,    false, "network input"},
    {reinterpret_cast<void**>(&state->dev_simcc_x),   simcc_x_bytes,  false, "simcc_x"},
    {reinterpret_cast<void**>(&state->dev_simcc_y),   simcc_y_bytes,  false, "simcc_y"},
    {reinterpret_cast<void**>(&state->host_simcc_x),  simcc_x_bytes,  true,  "host simcc_x"},
    {reinterpret_cast<void**>(&state->host_simcc_y),  simcc_y_bytes,  true,  "host simcc_y"},
  };

  for (const Alloc& alloc : allocs) {
    status = alloc.pinned ? cudaHostAlloc(alloc.ptr, alloc.bytes, cudaHostAllocDefault)
                          : cudaMalloc(alloc.ptr, alloc.bytes);
    if (status != cudaSuccess)
      return std::unexpected(cuda_error(alloc.what, status));
  }

  state->transform = make_transform(frame_width, frame_height);
  state->batch_cameras.reserve(MAX_BATCH);
  state->results.reserve(MAX_BATCH);

  return PoseModel(std::move(state));
}

Result<void> PoseModel::stage(const Frameset& set) {
  State& s = *m_state;

  if (set.frames.size() > MAX_BATCH)
    return std::unexpected(invalid("frameset holds more cameras than the engine batch"));

  s.batch_cameras.clear();

  for (const FrameView& view : set.frames) {
    // the kernel takes one pitch, and every camera decodes with the same
    // config, so a mismatch means an assumption broke rather than a case to
    // handle
    if (view.pitch != set.frames.front().pitch)
      return std::unexpected(invalid("cameras disagree on decoder pitch"));

    float* slot = s.dev_input + s.batch_cameras.size() * INPUT_ELEMS_PER_IMAGE;
    preprocess_nv12(view.device_ptr, view.width, view.height, view.pitch,
                    s.transform, slot, s.stream);

    s.batch_cameras.push_back(view.camera_id);
  }

  // the source surfaces are only read by the kernels above, so once this event
  // fires the caller can hand the frameset back
  cudaError_t status = cudaEventRecord(s.staged, s.stream);
  if (status != cudaSuccess)
    return std::unexpected(cuda_error("failed to record staging event", status));

  status = cudaEventSynchronize(s.staged);
  if (status != cudaSuccess)
    return std::unexpected(cuda_error("failed to wait on staging event", status));

  return {};
}

namespace {

// SimCC gives two 1D distributions per joint instead of a heatmap, so the
// coordinate is the argmax bin divided by the split ratio, and the confidence
// is how peaked that argmax was.
//
// This runs on the host over 3 x 133 x 1344 floats, which is a 2.1 MB copy
// back per frameset. Doing the argmax on device instead would make the copy
// about 5 KB. Left here so the pipeline works without a second kernel.
Keypoint decode_joint(const float* simcc_x, const float* simcc_y,
                      const cv::Matx33d& transform) {
  int best_x = 0;
  int best_y = 0;

  for (int i = 1; i < SIMCC_X_BINS; i += 1)
    if (simcc_x[i] > simcc_x[best_x])
      best_x = i;

  for (int i = 1; i < SIMCC_Y_BINS; i += 1)
    if (simcc_y[i] > simcc_y[best_y])
      best_y = i;

  Keypoint keypoint{};
  network_to_source(
    transform,
    static_cast<float>(best_x) / SIMCC_SPLIT_RATIO,
    static_cast<float>(best_y) / SIMCC_SPLIT_RATIO,
    keypoint.x,
    keypoint.y
  );

  // mmpose implementations differ on mean versus min of the two peaks. worth
  // checking against a reference before trusting the threshold downstream.
  keypoint.confidence = 0.5f * (simcc_x[best_x] + simcc_y[best_y]);
  return keypoint;
}

} // namespace

Result<std::span<const PoseResult>> PoseModel::infer() {
  State& s = *m_state;

  if (s.batch_cameras.empty())
    return std::span<const PoseResult>{};

  const int batch = static_cast<int>(s.batch_cameras.size());

  // dynamic batch: a frameset short a camera runs at batch 2 rather than
  // padding the input with a frame the network would waste time on
  if (!s.context->setInputShape(INPUT_TENSOR,
                                nvinfer1::Dims4{batch, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH}))
    return std::unexpected(invalid("failed to set input shape"));

  if (!s.context->setTensorAddress(INPUT_TENSOR, s.dev_input) ||
      !s.context->setTensorAddress(SIMCC_X_TENSOR, s.dev_simcc_x) ||
      !s.context->setTensorAddress(SIMCC_Y_TENSOR, s.dev_simcc_y))
    return std::unexpected(invalid("failed to bind tensors"));

  if (!s.context->enqueueV3(s.stream))
    return std::unexpected(invalid("failed to enqueue inference"));

  const size_t x_bytes = static_cast<size_t>(batch) * NUM_KEYPOINTS * SIMCC_X_BINS * sizeof(float);
  const size_t y_bytes = static_cast<size_t>(batch) * NUM_KEYPOINTS * SIMCC_Y_BINS * sizeof(float);

  cudaError_t status = cudaMemcpyAsync(
    s.host_simcc_x, s.dev_simcc_x, x_bytes, cudaMemcpyDeviceToHost, s.stream);
  if (status != cudaSuccess)
    return std::unexpected(cuda_error("failed to copy simcc_x back", status));

  status = cudaMemcpyAsync(
    s.host_simcc_y, s.dev_simcc_y, y_bytes, cudaMemcpyDeviceToHost, s.stream);
  if (status != cudaSuccess)
    return std::unexpected(cuda_error("failed to copy simcc_y back", status));

  status = cudaStreamSynchronize(s.stream);
  if (status != cudaSuccess)
    return std::unexpected(cuda_error("inference stream failed", status));

  s.results.clear();

  for (int b = 0; b < batch; b += 1) {
    PoseResult result{};
    result.camera_id = s.batch_cameras[static_cast<size_t>(b)];

    const float* x_base = s.host_simcc_x + static_cast<size_t>(b) * NUM_KEYPOINTS * SIMCC_X_BINS;
    const float* y_base = s.host_simcc_y + static_cast<size_t>(b) * NUM_KEYPOINTS * SIMCC_Y_BINS;

    for (int joint = 0; joint < NUM_KEYPOINTS; joint += 1)
      result.keypoints[static_cast<size_t>(joint)] = decode_joint(
        x_base + static_cast<size_t>(joint) * SIMCC_X_BINS,
        y_base + static_cast<size_t>(joint) * SIMCC_Y_BINS,
        s.transform
      );

    s.results.push_back(result);
  }

  return std::span<const PoseResult>(s.results);
}

} // namespace mocap
