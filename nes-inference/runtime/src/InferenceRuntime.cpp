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

#include <InferenceRuntime.hpp>

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ErrorHandling.hpp>
#ifdef NES_ENABLE_INFERENCE_BACKEND_IREE
    #include <IreeRuntimeBackend.hpp>
#endif
#include <Model.hpp>
#ifdef NES_ENABLE_INFERENCE_BACKEND_OPENVINO
    #include <OpenVinoRuntimeBackend.hpp>
#endif
#include <RuntimeBackend.hpp>

namespace NES
{

namespace
{

std::unique_ptr<RuntimeBackend> createRuntimeBackend(ModelBackend backend)
{
    switch (backend)
    {
        case ModelBackend::IREE:
#ifdef NES_ENABLE_INFERENCE_BACKEND_IREE
            return std::make_unique<IreeRuntimeBackend>();
#else
            throw NES::InferenceRuntimeFailure("IREE inference backend was not built");
#endif
        case ModelBackend::OPENVINO:
#ifdef NES_ENABLE_INFERENCE_BACKEND_OPENVINO
            return std::make_unique<OpenVinoRuntimeBackend>();
#else
            throw NES::InferenceRuntimeFailure("OpenVINO inference backend was not built");
#endif
    }
    std::unreachable();
}

}

struct InferenceRuntime::Impl
{
    std::unique_ptr<RuntimeBackend> backend;
};

InferenceRuntime::InferenceRuntime() : impl(std::make_unique<Impl>())
{
}

InferenceRuntime::~InferenceRuntime() = default;
InferenceRuntime::InferenceRuntime(InferenceRuntime&&) noexcept = default;
InferenceRuntime& InferenceRuntime::operator=(InferenceRuntime&&) noexcept = default;

void InferenceRuntime::setup(const CompiledModel& model, size_t batchSize, const InferenceRuntimeOptions& options)
{
    impl->backend = createRuntimeBackend(model.getBackend());
    auto metadata = impl->backend->setup(model, batchSize, options);

    this->inputShape = std::move(metadata.inputShape);
    this->outputShape = std::move(metadata.outputShape);
    this->nDim = metadata.nDim;
    this->functionName = std::move(metadata.functionName);

    /// NOLINTNEXTLINE(modernize-avoid-c-arrays) dynamic byte buffer requires array form
    this->inputData = std::make_unique<std::byte[]>(metadata.inputSize);
    this->inputSize = metadata.inputSize;
    this->inputTupleSize = inputShape.empty() ? inputSize : inputSize / inputShape.front();
    /// NOLINTNEXTLINE(modernize-avoid-c-arrays) dynamic byte buffer requires array form
    this->outputData = std::make_unique<std::byte[]>(metadata.outputSize);
    this->outputSize = metadata.outputSize;
    this->outputTupleSize = outputShape.empty() ? outputSize : outputSize / outputShape.front();
}

void InferenceRuntime::infer()
{
    infer(inputData.get(), inputSize);
}

void InferenceRuntime::infer(std::byte* inputBuffer, size_t inputBufferSize)
{
    if (impl->backend == nullptr)
    {
        throw NES::InferenceRuntimeFailure("Model Execution failed. Runtime backend was not set up");
    }

    PRECONDITION(inputBuffer != nullptr, "Cannot run inference with a null input buffer");
    PRECONDITION(inputBufferSize == inputSize, "Expected single-tuple input size {} B, but got {} B", inputSize, inputBufferSize);

    if (inputBuffer != inputData.get())
    {
        std::memcpy(inputData.get(), inputBuffer, inputBufferSize);
        inputBuffer = inputData.get();
    }

    impl->backend->infer(inputBuffer, inputBufferSize, outputData.get(), outputSize);
}

void InferenceRuntime::infer(size_t numberOfTuples)
{
    if (impl->backend == nullptr)
    {
        throw NES::InferenceRuntimeFailure("Model Execution failed. Runtime backend was not set up");
    }

    PRECONDITION(numberOfTuples > 0, "Cannot run inference on an empty batch");
    PRECONDITION(numberOfTuples * inputTupleSize <= inputSize, "Requested input batch exceeds runtime input buffer");
    PRECONDITION(numberOfTuples * outputTupleSize <= outputSize, "Requested output batch exceeds runtime output buffer");

    if (impl->backend->supportsActiveBatchSize())
    {
        impl->backend->infer(inputData.get(), numberOfTuples * inputTupleSize, outputData.get(), numberOfTuples * outputTupleSize);
        return;
    }

    const auto paddingOffset = numberOfTuples * inputTupleSize;
    std::memset(inputData.get() + paddingOffset, 0, inputSize - paddingOffset);
    impl->backend->infer(inputData.get(), inputSize, outputData.get(), outputSize);
}

}
