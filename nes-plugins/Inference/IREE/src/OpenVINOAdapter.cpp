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

#include <OpenVINOAdapter.hpp>

#include <algorithm>
#include <bit>
#include <ranges>
#include <string>

namespace NES
{

void OpenVINOAdapter::initializeModel(Nebuli::Inference::Model& model, const uint64_t batchSize)
{
    PRECONDITION(model.getBackend() == Nebuli::Inference::ModelBackend::OPENVINO, "OpenVINOAdapter expects an OpenVINO model");

    auto inputShape = model.getInputShape();
    PRECONDITION(!inputShape.empty(), "Model input shape must not be empty");
    inputShape[0] = batchSize;

    this->inputSize = model.inputSize() * batchSize;
    this->inputData = std::make_unique<std::byte[]>(inputSize);

    this->outputSize = model.outputSize() * batchSize;
    this->outputData = std::make_unique<std::byte[]>(outputSize);

    const auto xmlBuffer = model.getOpenVinoXml();
    std::string xmlContent(xmlBuffer.size(), '\0');
    std::ranges::transform(xmlBuffer, xmlContent.begin(), [](const auto value) { return static_cast<char>(value); });
    PRECONDITION(dtypeMap.contains(model.getInputDtype()), "Unsupported OpenVINO input dtype");
    runtimeWrapper.setup(xmlContent, model.getOpenVinoBin(), dtypeMap.at(model.getInputDtype()), inputShape);
}

template <class T>
void OpenVINOAdapter::addModelInput(const size_t index, const T value)
{
    PRECONDITION(index < inputSize / sizeof(T), "Index is too large");
    std::bit_cast<T*>(inputData.get())[index] = value;
}

void OpenVINOAdapter::addModelInput(const std::span<std::byte> content)
{
    std::ranges::copy_n(content.data(), std::min(content.size(), inputSize), inputData.get());
}

template <class T>
T OpenVINOAdapter::getResultAt(const size_t idx)
{
    PRECONDITION(idx < outputSize / sizeof(T), "Index is too large");
    return std::bit_cast<T*>(outputData.get())[idx];
}

void OpenVINOAdapter::copyResultTo(const std::span<std::byte> content)
{
    PRECONDITION(outputSize == content.size(), "Output size does not match");
    std::ranges::copy_n(outputData.get(), std::min(content.size(), outputSize), content.data());
}

template <class T>
void OpenVINOAdapter::infer()
{
    runtimeWrapper.execute(inputData.get(), inputSize, outputData.get(), outputSize);
}

std::shared_ptr<OpenVINOAdapter> OpenVINOAdapter::create()
{
    return std::make_shared<OpenVINOAdapter>();
}

#define NES_OPENVINO_ADAPTER_INSTANTIATE(T)      \
    template void OpenVINOAdapter::addModelInput<T>(size_t, T); \
    template T OpenVINOAdapter::getResultAt<T>(size_t); \
    template void OpenVINOAdapter::infer<T>();

NES_OPENVINO_ADAPTER_INSTANTIATE(uint8_t)
NES_OPENVINO_ADAPTER_INSTANTIATE(uint16_t)
NES_OPENVINO_ADAPTER_INSTANTIATE(uint32_t)
NES_OPENVINO_ADAPTER_INSTANTIATE(uint64_t)
NES_OPENVINO_ADAPTER_INSTANTIATE(int8_t)
NES_OPENVINO_ADAPTER_INSTANTIATE(int16_t)
NES_OPENVINO_ADAPTER_INSTANTIATE(int32_t)
NES_OPENVINO_ADAPTER_INSTANTIATE(int64_t)
NES_OPENVINO_ADAPTER_INSTANTIATE(float)
NES_OPENVINO_ADAPTER_INSTANTIATE(double)

#undef NES_OPENVINO_ADAPTER_INSTANTIATE

}
