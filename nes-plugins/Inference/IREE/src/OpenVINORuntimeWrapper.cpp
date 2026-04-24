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

#include <OpenVINORuntimeWrapper.hpp>

#include <algorithm>
#include <cstring>
#include <vector>
#include <ErrorHandling.hpp>

namespace NES
{

void OpenVINORuntimeWrapper::setup(
    const std::string& modelXml,
    const std::span<const std::byte> modelBin,
    const ov::element::Type& inputElementType,
    const std::vector<size_t>& inputShape)
{
    weightsBuffer.resize(modelBin.size());
    std::ranges::transform(modelBin, weightsBuffer.begin(), [](const std::byte value) { return static_cast<std::uint8_t>(value); });

    ov::Tensor weights(ov::element::u8, {weightsBuffer.size()}, weightsBuffer.data());
    auto model = core.read_model(modelXml, weights);
    compiledModel = core.compile_model(model, "CPU");
        // ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
        // ov::inference_num_threads(1),
        // ov::num_streams(1));
    inferRequest = compiledModel.create_infer_request();
    inputTensor = ov::Tensor(inputElementType, inputShape);
    inferRequest.set_input_tensor(inputTensor);
}

void OpenVINORuntimeWrapper::execute(const void* inputData, size_t inputDataSize, void* outputData, const size_t outputDataSize)
{
    PRECONDITION(inputData != nullptr, "Input data pointer must not be null");
    PRECONDITION(outputData != nullptr, "Output data pointer must not be null");

    const auto tensorInputSize = inputTensor.get_byte_size();
    std::memset(inputTensor.data(), 0, tensorInputSize);
    std::memcpy(inputTensor.data(), inputData, std::min(inputDataSize, tensorInputSize));

    inferRequest.infer();

    const auto outputTensor = inferRequest.get_output_tensor(0);
    const auto tensorOutputSize = outputTensor.get_byte_size();
    std::memcpy(outputData, outputTensor.data(), std::min(outputDataSize, tensorOutputSize));
}

}
