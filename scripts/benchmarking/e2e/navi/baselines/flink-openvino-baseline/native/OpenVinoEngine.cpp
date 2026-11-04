#include "OpenVinoEngine.hpp"

#include <openvino/openvino.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

/// Compiled-model sharing, the counterpart of the NES inference option
/// `openvino_share_compiled_model`. Off by default, so the baseline keeps the straightforward
/// behaviour of one compiled model per operator instance; set FLINK_OV_SHARE_COMPILED_MODEL=1 to
/// give Flink the same advantage NES has. It matters on a Pi: three lanes each holding their own
/// copy of the 190 MB SARAD model push per-inference time from ~78 ms to ~226 ms, because the
/// copies contend for memory bandwidth.
bool share_compiled_model_enabled() {
    const char* value = std::getenv("FLINK_OV_SHARE_COMPILED_MODEL");
    if (value == nullptr) {
        return false;
    }
    const std::string v{value};
    return v == "1" || v == "true" || v == "TRUE" || v == "yes";
}

/// How many lanes share one compiled model: FLINK_OV_SHARED_LANES, which the runners set to the job's parallelism.
int shared_lanes() {
    const char* value = std::getenv("FLINK_OV_SHARED_LANES");
    if (value == nullptr) {
        return 1;
    }
    try {
        return std::max(1, std::stoi(value));
    } catch (...) {
        return 1;
    }
}

/// One Core for the process. Core is internally reference counted, but sharing it keeps the plugin
/// registry and its caches in one place.
ov::Core& shared_core() {
    static ov::Core core;
    return core;
}

std::mutex& compile_mutex() {
    static std::mutex mutex;
    return mutex;
}

/// Compiled models already built in this process, keyed by everything that affects the result.
/// Weak, so the last engine releasing a model frees it -- the same ownership NES's shared
/// compiled-model cache uses, and it keeps a finished query from pinning the weights.
std::map<std::string, std::weak_ptr<ov::CompiledModel>>& compiled_model_cache() {
    static std::map<std::string, std::weak_ptr<ov::CompiledModel>> cache;
    return cache;
}

}

OpenVinoEngine::OpenVinoEngine(
    const std::string& model_path,
    const std::string& device,
    int num_streams,
    int num_threads
) {
    const bool share = share_compiled_model_enabled();
    // OpenVINO runs one inference per stream at a time, so lanes sharing a compiled model with fewer streams than lanes would
    // queue on a stream's graph. Like NES's resolveSessionOptions, a shared model gets one stream per lane, each with the
    // per-lane thread count (inference_num_threads counts the threads of all streams).
    if (share && num_streams >= 0) {
        num_streams = std::max(num_streams, shared_lanes());
        if (num_threads > 0) {
            num_threads *= num_streams;
        }
    }
    const std::string cache_key = model_path + '|' + device + '|' + std::to_string(num_streams) + '|'
        + std::to_string(num_threads);

    // Compilation is serialised across lanes whether or not the model is shared. Three concurrent
    // compilations of a large IR reproducibly wedge a 4-core Pi: two lanes come up and the third
    // never finishes. It costs nothing to serialise -- this runs before any measurement starts.
    std::lock_guard<std::mutex> compile_guard(compile_mutex());

    if (share) {
        auto& cache = compiled_model_cache();
        const auto cached = cache.find(cache_key);
        if (cached != cache.end()) {
            if (auto existing = cached->second.lock()) {
                compiled_model_ = std::move(existing);
                std::cout << "OpenVINO compiled model: reusing the shared one for " << model_path
                          << std::endl;
            }
        }
    }

    if (!compiled_model_) {
        compiled_model_ = compile(model_path, device, num_streams, num_threads);
        std::cout << "OpenVINO compiled model: " << (share ? "compiled, shared with later lanes"
                                                           : "compiled, private to this lane")
                  << " (" << model_path << ", num_streams " << num_streams << ", inference_num_threads " << num_threads
                  << ")" << std::endl;
        if (share) {
            compiled_model_cache()[cache_key] = compiled_model_;
        }
    }

    infer_request_ = compiled_model_->create_infer_request();

    ov::Output<const ov::Node> input_port = compiled_model_->input();
    ov::Output<const ov::Node> output_port = compiled_model_->output();

    ov::Shape input_shape = input_port.get_shape();
    ov::Shape output_shape = output_port.get_shape();

    ov::element::Type input_type = input_port.get_element_type();
    ov::element::Type output_type = output_port.get_element_type();

    if (input_type != ov::element::f32) {
        throw std::runtime_error(
            "This baseline expects f32 input. Actual input type: " +
            input_type.to_string()
        );
    }

    if (output_type != ov::element::f32) {
        throw std::runtime_error(
            "This baseline expects f32 output. Actual output type: " +
            output_type.to_string()
        );
    }

    input_size_ = shape_size(input_shape);
    output_size_ = shape_size(output_shape);

    input_tensor_ = ov::Tensor(input_type, input_shape);
    output_tensor_ = ov::Tensor(output_type, output_shape);

    infer_request_.set_input_tensor(input_tensor_);
    infer_request_.set_output_tensor(output_tensor_);
}

std::shared_ptr<ov::CompiledModel> OpenVinoEngine::compile(
    const std::string& model_path,
    const std::string& device,
    int num_streams,
    int num_threads
) {
    ov::Core& core = shared_core();

    std::shared_ptr<ov::Model> model = core.read_model(model_path);

    // Pin a dynamic batch dimension to 1: the job feeds one frame per predict() call, and
    // get_shape()/ov::Tensor below need fully static shapes. The SqueezeDet export carries a
    // dynamic leading dim ([?, H, W, 3]); NanoDet is already static.
    ov::PartialShape input_pshape = model->input().get_partial_shape();
    if (input_pshape.rank().is_static() && input_pshape[0].is_dynamic()) {
        input_pshape[0] = 1;
        model->reshape(input_pshape);
    }

    // Mirror the NES OpenVINO backend's compile options (OpenVinoRuntimeBackend.cpp): ACCURACY +
    // LATENCY hints with EXPLICIT num_streams / inference_num_threads values. NES compiles with
    // num_streams(0) and inference_num_threads(1) by default, so 0 must reach the plugin as an
    // explicit value, not be dropped. Pass a negative value to leave a property unset.
    ov::AnyMap compile_options{
        ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY),
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
    };
    if (num_streams >= 0) {
        compile_options.insert(ov::num_streams(num_streams));
    }
    if (num_threads >= 0) {
        compile_options.insert(ov::inference_num_threads(num_threads));
    }
    return std::make_shared<ov::CompiledModel>(core.compile_model(model, device, compile_options));
}

std::size_t OpenVinoEngine::shape_size(const ov::Shape& shape) {
    return std::accumulate(
        shape.begin(),
        shape.end(),
        static_cast<std::size_t>(1),
        std::multiplies<>()
    );
}

std::size_t OpenVinoEngine::input_size() const {
    return input_size_;
}

std::size_t OpenVinoEngine::output_size() const {
    return output_size_;
}

std::atomic<std::uint64_t> OpenVinoEngine::calls_{0};
std::atomic<std::uint64_t> OpenVinoEngine::predict_ns_{0};
std::atomic<std::uint64_t> OpenVinoEngine::infer_ns_{0};

void OpenVinoEngine::timings(std::uint64_t& calls, std::uint64_t& predict_ns, std::uint64_t& infer_ns) {
    calls = calls_.load(std::memory_order_relaxed);
    predict_ns = predict_ns_.load(std::memory_order_relaxed);
    infer_ns = infer_ns_.load(std::memory_order_relaxed);
}

void OpenVinoEngine::predict(const float* input, float* output) {
    const auto predict_start = std::chrono::steady_clock::now();

    auto* input_data = input_tensor_.data<float>();

    std::memcpy(
        input_data,
        input,
        input_size_ * sizeof(float)
    );

    const auto infer_start = std::chrono::steady_clock::now();
    infer_request_.infer();
    const auto infer_end = std::chrono::steady_clock::now();

    const auto* output_data = output_tensor_.data<const float>();

    std::memcpy(
        output,
        output_data,
        output_size_ * sizeof(float)
    );

    const auto predict_end = std::chrono::steady_clock::now();
    using ns = std::chrono::nanoseconds;
    calls_.fetch_add(1, std::memory_order_relaxed);
    predict_ns_.fetch_add(
        std::chrono::duration_cast<ns>(predict_end - predict_start).count(), std::memory_order_relaxed);
    infer_ns_.fetch_add(
        std::chrono::duration_cast<ns>(infer_end - infer_start).count(), std::memory_order_relaxed);
}
