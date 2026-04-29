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
#include <mutex>
#include <sstream>
#include <type_traits>
#include <vector>
#include <ErrorHandling.hpp>
#include <Util/Logger/Logger.hpp>

namespace NES
{
#ifndef NO_ASSERT
namespace
{
constexpr size_t MAX_DEBUG_TENSOR_STRING_SIZE = 4096;

std::string shapeToString(const ov::Shape& shape)
{
    std::ostringstream stream;
    stream << "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
        {
            stream << ", ";
        }
        stream << shape.at(i);
    }
    stream << "]";
    return stream.str();
}

template <typename T>
void appendTensorValues(std::ostringstream& stream, const ov::Tensor& tensor)
{
    constexpr auto DEBUG_MARGIN = std::streamoff{64};
    const auto* data = tensor.data<const T>();
    const auto numberOfValues = tensor.get_size();
    for (size_t i = 0; i < numberOfValues; ++i)
    {
        if (i > 0)
        {
            stream << ", ";
        }

        if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>)
        {
            stream << static_cast<int>(data[i]);
        }
        else if constexpr (std::is_same_v<T, bool>)
        {
            stream << (data[i] ? "true" : "false");
        }
        else if constexpr (std::is_same_v<T, ov::float16>)
        {
            stream << static_cast<float>(data[i]);
        }
        else
        {
            stream << data[i];
        }

        if (stream.tellp() >= static_cast<std::streamoff>(MAX_DEBUG_TENSOR_STRING_SIZE) - DEBUG_MARGIN)
        {
            stream << ", ...";
            return;
        }
    }
}

std::string formatTensor(const ov::Tensor& tensor)
{
    std::ostringstream stream;
    stream << "shape=" << shapeToString(tensor.get_shape()) << ", type=" << tensor.get_element_type().get_type_name() << ", values=[";

    const auto elementType = tensor.get_element_type();
    if (elementType == ov::element::u8)
    {
        appendTensorValues<uint8_t>(stream, tensor);
    }
    else if (elementType == ov::element::u16)
    {
        appendTensorValues<uint16_t>(stream, tensor);
    }
    else if (elementType == ov::element::u32)
    {
        appendTensorValues<uint32_t>(stream, tensor);
    }
    else if (elementType == ov::element::u64)
    {
        appendTensorValues<uint64_t>(stream, tensor);
    }
    else if (elementType == ov::element::i8)
    {
        appendTensorValues<int8_t>(stream, tensor);
    }
    else if (elementType == ov::element::i16)
    {
        appendTensorValues<int16_t>(stream, tensor);
    }
    else if (elementType == ov::element::i32)
    {
        appendTensorValues<int32_t>(stream, tensor);
    }
    else if (elementType == ov::element::i64)
    {
        appendTensorValues<int64_t>(stream, tensor);
    }
    else if (elementType == ov::element::f16)
    {
        appendTensorValues<ov::float16>(stream, tensor);
    }
    else if (elementType == ov::element::f32)
    {
        appendTensorValues<float>(stream, tensor);
    }
    else if (elementType == ov::element::f64)
    {
        appendTensorValues<double>(stream, tensor);
    }
    else if (elementType == ov::element::boolean)
    {
        appendTensorValues<bool>(stream, tensor);
    }
    else
    {
        stream << "<unsupported element type>";
    }

    stream << "]";
    auto result = stream.str();
    if (result.size() > MAX_DEBUG_TENSOR_STRING_SIZE)
    {
        result.resize(MAX_DEBUG_TENSOR_STRING_SIZE - 3);
        result += "...";
    }
    return result;
}
}
#endif

namespace
{
struct InferRequestContext
{
    ov::InferRequest inferRequest;
    ov::element::Type outputElementType;
    ov::Shape outputShape;
    size_t outputTensorSize = 0;
};

InferRequestContext createInferRequest(const std::string& modelXml, std::span<const std::byte> modelBin, const ov::Shape& shape)
{
    static ov::Core sharedCore;
    static std::mutex coreMutex;

    std::vector<std::uint8_t> modelBinBuffer(modelBin.size());
    std::ranges::transform(modelBin, modelBinBuffer.begin(), [](const std::byte value) { return static_cast<std::uint8_t>(value); });
    std::scoped_lock lock(coreMutex);

    ov::Tensor weights(ov::element::u8, {modelBinBuffer.size()}, modelBinBuffer.data());
    auto model = sharedCore.read_model(modelXml, weights);
    model->reshape(shape);
    auto compiledModel = sharedCore.compile_model(model, "CPU",
        ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY), // to avoid implicit demotion to bfloat16
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
        ov::hint::enable_cpu_pinning(false),
        ov::inference_num_threads(1),
        ov::num_streams(1));

    auto outputElementType = compiledModel.output(0).get_element_type();
    auto outputShape = compiledModel.output(0).get_shape();
    const auto outputTensorSize = ov::Tensor(outputElementType, outputShape).get_byte_size();

    return InferRequestContext{
        .inferRequest = compiledModel.create_infer_request(),
        .outputElementType = std::move(outputElementType),
        .outputShape = std::move(outputShape),
        .outputTensorSize = outputTensorSize};
}
}

void OpenVINORuntimeWrapper::setup(
    const std::string& modelXml,
    const std::span<const std::byte> modelBin,
    const ov::element::Type& inputElementType,
    const std::vector<size_t>& inputShape)
{
    auto shape = ov::Shape(inputShape.begin(), inputShape.end());
    auto inferRequestContext = createInferRequest(modelXml, modelBin, shape);
    inferRequest = std::move(inferRequestContext.inferRequest);
    this->inputElementType = inputElementType;
    this->inputShape = shape;
    this->outputElementType = std::move(inferRequestContext.outputElementType);
    this->outputShape = std::move(inferRequestContext.outputShape);
    this->outputTensorSize = inferRequestContext.outputTensorSize;
}

void OpenVINORuntimeWrapper::execute(const void* inputData, void* outputData)
{
    auto inputTensor = ov::Tensor(inputElementType, inputShape, const_cast<void*>(inputData));
    inferRequest.set_input_tensor(inputTensor);

    auto outputTensor = ov::Tensor(outputElementType, outputShape, outputData);
    inferRequest.set_output_tensor(0, outputTensor);

#ifndef NO_ASSERT
    NES_DEBUG("Model input: {}", formatTensor(inputTensor))
#endif

    inferRequest.infer();

#ifndef NO_ASSERT
    NES_DEBUG("Model output: {}", formatTensor(outputTensor))
#endif
}

}
