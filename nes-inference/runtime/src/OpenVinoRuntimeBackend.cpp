/*
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#include <OpenVinoRuntimeBackend.hpp>

#include <Util/Logger/Logger.hpp>
#include <ErrorHandling.hpp>
#include <Inference.hpp>
#include <Model.hpp>
#include <RuntimeBackend.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <ios>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/runtime/tensor.hpp>

namespace
{
ov::PartialShape makeDynamicBatchShape(const std::vector<size_t>& shape)
{
    std::vector<ov::Dimension> dimensions;
    dimensions.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i == 0)
        {
            dimensions.emplace_back(ov::Dimension::dynamic());
        }
        else
        {
            dimensions.emplace_back(static_cast<int64_t>(shape.at(i)));
        }
    }
    return ov::PartialShape(dimensions);
}

ov::Shape makeRuntimeShape(std::vector<size_t> shape, size_t batchSize)
{
    if (!shape.empty())
    {
        shape.front() = batchSize;
    }
    return ov::Shape(shape.begin(), shape.end());
}

/// Identifies a compiled artifact for the shared-compiled-model cache. Anything that
/// changes the compiled result must appear here: the model bytes (hash + length, to
/// keep collisions astronomically unlikely without storing weight-sized keys) and
/// every compile option. Two inference operators over the same model with identical
/// options therefore share one ov::CompiledModel.
struct CompiledModelKey
{
    size_t dataHash;
    size_t auxHash;
    size_t dataSize;
    size_t auxSize;
    size_t batchSize;
    int numThreads;
    int numStreams;
    bool pinning;
    bool dynamicBatch;
    bool operator==(const CompiledModelKey&) const = default;
};

struct CompiledModelKeyHash
{
    size_t operator()(const CompiledModelKey& key) const
    {
        size_t seed = key.dataHash;
        const auto mix = [&seed](size_t value) { seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2); };
        mix(key.auxHash);
        mix(key.dataSize);
        mix(key.auxSize);
        mix(key.batchSize);
        mix(static_cast<size_t>(key.numThreads));
        mix(static_cast<size_t>(key.numStreams));
        mix(static_cast<size_t>(key.pinning));
        mix(static_cast<size_t>(key.dynamicBatch));
        return seed;
    }
};
}

namespace NES
{
RuntimeMetadata OpenVinoRuntimeBackend::setup(const CompiledModel& model, size_t batchSize, const InferenceRuntimeOptions& options)
{
    /// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) byte-to-text for OpenVINO XML payload
    const std::string modelXml(reinterpret_cast<const char*>(model.getData().data()), model.getData().size());
    std::vector<std::uint8_t> modelBin(model.getAuxiliaryData().size());
    std::ranges::transform(model.getAuxiliaryData(), modelBin.begin(), [](std::byte value) { return static_cast<std::uint8_t>(value); });

    static ov::Core sharedCore;
    static std::mutex coreMutex;
    /// Cache of compiled models shared across worker-thread sessions and inference
    /// operators, guarded by coreMutex (only touched at setup, never on the infer
    /// hot path). Enabled by openvino_share_compiled_model; otherwise every session
    /// compiles its own private copy exactly as before.
    static std::unordered_map<CompiledModelKey, ov::CompiledModel, CompiledModelKeyHash> compiledModelCache;

    auto runtimeInputShape = model.getInputShape();
    auto runtimeOutputShape = model.getOutputShape();
    const auto maxInputShape = makeRuntimeShape(runtimeInputShape, batchSize);
    const auto maxOutputShape = makeRuntimeShape(runtimeOutputShape, batchSize);
    dynamicBatchEnabled = batchSize > 1 && options.openvinoAllowDynamicBatch;

    const auto compile = [&]
    {
        ov::Tensor weights(ov::element::u8, {modelBin.size()});
        if (!modelBin.empty())
        {
            std::memcpy(weights.data<std::uint8_t>(), modelBin.data(), modelBin.size());
        }
        auto openVinoModel = sharedCore.read_model(modelXml, weights);
        if (dynamicBatchEnabled)
        {
            openVinoModel->reshape(makeDynamicBatchShape(runtimeInputShape));
        }
        else
        {
            openVinoModel->reshape(ov::PartialShape(maxInputShape));
        }
        /// A shared model with >1 stream is a throughput deployment (N concurrent
        /// requests over one weight copy). Only then switch off the latency hint —
        /// the non-shared path keeps LATENCY unconditionally, matching prior behavior
        /// for the existing openvino_num_streams ablation.
        const auto performanceMode = (options.openvinoShareCompiledModel && options.openvinoNumStreams > 1)
            ? ov::hint::PerformanceMode::THROUGHPUT
            : ov::hint::PerformanceMode::LATENCY;
        ov::AnyMap compileOptions{
            ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY),
            ov::hint::performance_mode(performanceMode),
            ov::inference_num_threads(static_cast<int>(options.openvinoInferenceNumThreads)),
            ov::num_streams(static_cast<int>(options.openvinoNumStreams)),
            ov::hint::enable_cpu_pinning(options.openvinoEnableCpuPinning)};
        return sharedCore.compile_model(openVinoModel, "CPU", compileOptions);
    };

    const std::scoped_lock lock(coreMutex);
    ov::CompiledModel compiledModel;
    if (options.openvinoShareCompiledModel)
    {
        const CompiledModelKey key{
            .dataHash = std::hash<std::string_view>{}(std::string_view(modelXml)),
            .auxHash = std::hash<std::string_view>{}(
                std::string_view(reinterpret_cast<const char*>(modelBin.data()), modelBin.size())),
            .dataSize = modelXml.size(),
            .auxSize = modelBin.size(),
            .batchSize = batchSize,
            .numThreads = static_cast<int>(options.openvinoInferenceNumThreads),
            .numStreams = static_cast<int>(options.openvinoNumStreams),
            .pinning = options.openvinoEnableCpuPinning,
            .dynamicBatch = dynamicBatchEnabled};
        if (const auto it = compiledModelCache.find(key); it != compiledModelCache.end())
        {
            compiledModel = it->second;
        }
        else
        {
            compiledModel = compile();
            compiledModelCache.emplace(key, compiledModel);
        }
    }
    else
    {
        compiledModel = compile();
    }

    inferRequest = compiledModel.create_infer_request();
    inputElementType = compiledModel.input(0).get_element_type();
    inputShape = maxInputShape;
    outputElementType = compiledModel.output(0).get_element_type();
    outputShape = maxOutputShape;
    inputTupleSize = inputShape.empty() ? ov::Tensor(inputElementType, inputShape).get_byte_size()
                                        : ov::Tensor(inputElementType, inputShape).get_byte_size() / inputShape.front();
    outputTupleSize = outputShape.empty() ? ov::Tensor(outputElementType, outputShape).get_byte_size()
                                          : ov::Tensor(outputElementType, outputShape).get_byte_size() / outputShape.front();

    return RuntimeMetadata{
        .inputShape = std::vector<size_t>(inputShape.begin(), inputShape.end()),
        .outputShape = std::vector<size_t>(outputShape.begin(), outputShape.end()),
        .nDim = runtimeInputShape.size(),
        .functionName = model.getFunctionName(),
        .inputSize = ov::Tensor(inputElementType, inputShape).get_byte_size(),
        .outputSize = ov::Tensor(outputElementType, outputShape).get_byte_size()};
}

void OpenVinoRuntimeBackend::infer(std::byte* inputBuffer, size_t inputBufferSize, std::byte* outputBuffer, size_t outputBufferSize)
{
    auto currentInputShape = inputShape;
    auto currentOutputShape = outputShape;
    if (!currentInputShape.empty())
    {
        if (inputBufferSize % inputTupleSize != 0)
        {
            throw NES::InferenceRuntimeFailure(
                "Model Execution failed. Input buffer size {} B is not a multiple of tuple size {} B",
                inputBufferSize,
                inputTupleSize);
        }
        const auto currentBatchSize = inputBufferSize / inputTupleSize;
        currentInputShape.front() = currentBatchSize;
        currentOutputShape.front() = currentBatchSize;
    }

    const auto outputSizeBytes = currentOutputShape.empty() ? outputTupleSize : currentOutputShape.front() * outputTupleSize;
    if (outputSizeBytes > outputBufferSize)
    {
        throw NES::InferenceRuntimeFailure(
            "Model Execution failed. Model output size {} B exceeds buffer capacity {} B", outputSizeBytes, outputBufferSize);
    }

    if (!dynamicBatchEnabled)
    {
        if (currentInputShape != inputShape || currentOutputShape != outputShape)
        {
            throw NES::InferenceRuntimeFailure("Model Execution failed. OpenVINO runtime was set up with static batch size {}", inputShape.front());
        }
        prepareExternalTensors(inputShape, outputShape, inputBuffer, outputBuffer);
        inferRequest.infer();
        return;
    }

    if (currentInputShape == inputShape && currentOutputShape == outputShape)
    {
        prepareExternalTensors(currentInputShape, currentOutputShape, inputBuffer, outputBuffer);
        inferRequest.infer();
        return;
    }

    prepareOwnedTensors(currentInputShape, currentOutputShape);
    std::memcpy(inputTensor.data(), inputBuffer, inputBufferSize);
    inferRequest.infer();
    std::memcpy(outputBuffer, outputTensor.data(), outputSizeBytes);
}

void OpenVinoRuntimeBackend::prepareExternalTensors(
    const ov::Shape& currentInputShape,
    const ov::Shape& currentOutputShape,
    std::byte* inputBuffer,
    std::byte* outputBuffer)
{
    if (!usingExternalTensors || externalInputBuffer != inputBuffer || inputTensor.get_shape() != currentInputShape)
    {
        inputTensor = ov::Tensor(inputElementType, currentInputShape, inputBuffer);
        inferRequest.set_input_tensor(inputTensor);
        externalInputBuffer = inputBuffer;
    }

    if (!usingExternalTensors || externalOutputBuffer != outputBuffer || outputTensor.get_shape() != currentOutputShape)
    {
        outputTensor = ov::Tensor(outputElementType, currentOutputShape, outputBuffer);
        inferRequest.set_output_tensor(0, outputTensor);
        externalOutputBuffer = outputBuffer;
    }
    usingExternalTensors = true;
}

void OpenVinoRuntimeBackend::prepareOwnedTensors(const ov::Shape& currentInputShape, const ov::Shape& currentOutputShape)
{
    if (usingExternalTensors || !inputTensor || inputTensor.get_shape() != currentInputShape)
    {
        inputTensor = ov::Tensor(inputElementType, currentInputShape);
        inferRequest.set_input_tensor(inputTensor);
        externalInputBuffer = nullptr;
    }

    if (usingExternalTensors || !outputTensor || outputTensor.get_shape() != currentOutputShape)
    {
        outputTensor = ov::Tensor(outputElementType, currentOutputShape);
        inferRequest.set_output_tensor(0, outputTensor);
        externalOutputBuffer = nullptr;
    }
    usingExternalTensors = false;
}
}
