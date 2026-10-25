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

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include <ErrorHandling.hpp>
#include <Model.hpp>

namespace NES
{

/// The OpenVINO session of a model UDF. A UDF stands for inference in a stream processor without an inference operator, so it
/// runs the model the way a UDF author would with OpenVINO alone: compiled for the CPU with OpenVINO's default configuration (no
/// stream, thread, performance-hint or precision properties), with none of the inference operators' runtime or session options.
/// The only thing shared with them is the model conversion: the session reads the OpenVINO IR that importing and compiling the
/// model produced, which ovc would produce offline. Each worker thread owns one session: compiled model, infer request and the
/// request's input and output tensors.
class StandaloneOpenVinoSession
{
public:
    explicit StandaloneOpenVinoSession(const CompiledModel& model)
    {
        if (model.getBackend() != ModelBackend::OPENVINO)
        {
            throw CannotLoadModel("A model UDF runs OpenVINO models only");
        }
        try
        {
            /// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) byte-to-text for the OpenVINO XML payload
            const std::string xml(reinterpret_cast<const char*>(model.getData().data()), model.getData().size());
            ov::Tensor weights(ov::element::u8, {model.getAuxiliaryData().size()});
            if (!model.getAuxiliaryData().empty())
            {
                std::memcpy(weights.data(), model.getAuxiliaryData().data(), model.getAuxiliaryData().size());
            }

            /// One window per invocation: the batch dimension, dynamic in some exports, is pinned to 1.
            std::vector<size_t> inputShape = model.getInputShape();
            if (!inputShape.empty())
            {
                inputShape.front() = 1;
            }

            /// Compilation is serialized, as concurrent compilations of a large model can exhaust a small worker's memory; it
            /// happens once per worker thread, before the first invocation.
            const std::scoped_lock lock(compileMutex());
            auto openVinoModel = core().read_model(xml, weights);
            openVinoModel->reshape(ov::PartialShape(ov::Shape(inputShape.begin(), inputShape.end())));
            compiledModel = core().compile_model(openVinoModel, "CPU");
            inferRequest = compiledModel.create_infer_request();
            inputTensor = inferRequest.get_input_tensor();
            outputTensor = inferRequest.get_output_tensor();
        }
        catch (const std::exception& exception)
        {
            throw CannotLoadModel("OpenVINO failed to set up a model UDF session: {}", exception.what());
        }
    }

    [[nodiscard]] std::byte* getInputData() { return static_cast<std::byte*>(inputTensor.data()); }

    [[nodiscard]] size_t getInputSize() const { return inputTensor.get_byte_size(); }

    [[nodiscard]] std::byte* getOutputData() { return static_cast<std::byte*>(outputTensor.data()); }

    [[nodiscard]] size_t getOutputSize() const { return outputTensor.get_byte_size(); }

    void infer() { inferRequest.infer(); }

private:
    /// The UDFs' own OpenVINO core, separate from the one the inference operators use.
    static ov::Core& core()
    {
        static ov::Core instance;
        return instance;
    }

    static std::mutex& compileMutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    ov::CompiledModel compiledModel;
    ov::InferRequest inferRequest;
    ov::Tensor inputTensor;
    ov::Tensor outputTensor;
};

}
