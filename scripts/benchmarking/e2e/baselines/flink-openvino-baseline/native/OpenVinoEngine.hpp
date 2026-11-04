#pragma once

#include <atomic>
#include <cstdint>

#include <openvino/openvino.hpp>

#include <cstddef>
#include <memory>
#include <string>

class OpenVinoEngine {
public:
    OpenVinoEngine(
        const std::string& model_path,
        const std::string& device,
        int num_streams,
        int num_threads
    );

    std::size_t input_size() const;
    std::size_t output_size() const;

    void predict(const float* input, float* output);

    /// Process-wide totals for the cost breakdown of one predict() call, in nanoseconds.
    ///
    /// Three nested spans separate what NAVI does not pay from the model itself: the Java side
    /// times the bridge call, `predict_ns` covers this function (the two tensor memcpys plus the
    /// inference), and `infer_ns` covers only `infer_request_.infer()`. Java-minus-predict is then
    /// the JNI array access (GetFloatArrayElements may copy the whole array), and
    /// predict-minus-infer is the marshalling into and out of OpenVINO's own tensors. NAVI skips
    /// both: its InferModelPhysicalOperator hands OpenVINO the tuple buffer directly whenever the
    /// field size already matches the model input.
    ///
    /// Static because each inference lane builds its own engine and the run wants the aggregate.
    /// Relaxed ordering: summed once after the job stops, so only atomicity matters.
    static void timings(std::uint64_t& calls, std::uint64_t& predict_ns, std::uint64_t& infer_ns);


private:
    static std::size_t shape_size(const ov::Shape& shape);

    /// Reads the model and compiles it with the NES compile options. Called under the compile
    /// mutex, once per distinct (model, device, streams, threads) when sharing is on.
    static std::shared_ptr<ov::CompiledModel> compile(
        const std::string& model_path,
        const std::string& device,
        int num_streams,
        int num_threads
    );

private:
    /// Shared with the other engines of this process when compiled-model sharing is enabled, so the
    /// lanes hold one copy of the weights between them instead of one each. The infer request and
    /// the tensors below stay per-engine: a request is not thread-safe, and each lane runs its own.
    std::shared_ptr<ov::CompiledModel> compiled_model_;
    ov::InferRequest infer_request_;

    ov::Tensor input_tensor_;
    ov::Tensor output_tensor_;

    std::size_t input_size_ = 0;
    std::size_t output_size_ = 0;

private:
    static std::atomic<std::uint64_t> calls_;
    static std::atomic<std::uint64_t> predict_ns_;
    static std::atomic<std::uint64_t> infer_ns_;
};
