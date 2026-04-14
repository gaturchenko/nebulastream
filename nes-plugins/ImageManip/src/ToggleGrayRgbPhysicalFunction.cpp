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

#include "../include/ToggleGrayRgbPhysicalFunction.hpp"

#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <Functions/PhysicalFunction.hpp>
#include <Nautilus/DataTypes/VarVal.hpp>
#include <Nautilus/DataTypes/VariableSizedData.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nautilus/function.hpp>
#include <Arena.hpp>
#include <ErrorHandling.hpp>
#include <PhysicalFunctionRegistry.hpp>

namespace NES
{

namespace
{
std::vector<uint8_t> toggleGrayRgbAndEncodePng(const int8_t* inputData, uint32_t inputSize)
{
    if (inputData == nullptr || inputSize == 0)
    {
        return {};
    }

    const auto inputSizeInt = static_cast<int>(inputSize);
    cv::Mat encodedBytes(1, inputSizeInt, CV_8U, const_cast<int8_t*>(inputData)); /// NOLINT(cppcoreguidelines-pro-type-const-cast)
    cv::Mat inputImage = cv::imdecode(encodedBytes, cv::IMREAD_UNCHANGED);
    if (inputImage.empty())
    {
        return {};
    }

    cv::Mat convertedImage;
    if (inputImage.channels() == 1)
    {
        cv::cvtColor(inputImage, convertedImage, cv::COLOR_GRAY2BGR);
    }
    else if (inputImage.channels() == 3)
    {
        cv::cvtColor(inputImage, convertedImage, cv::COLOR_BGR2GRAY);
    }
    else if (inputImage.channels() == 4)
    {
        cv::cvtColor(inputImage, convertedImage, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        return {};
    }

    std::vector<uint8_t> encodedOutput;
    if (!cv::imencode(".png", convertedImage, encodedOutput))
    {
        return {};
    }
    return encodedOutput;
}

uint32_t getToggleGrayRgbImageSize(int8_t* inputData, uint32_t inputSize)
{
    const auto encodedOutput = toggleGrayRgbAndEncodePng(inputData, inputSize);
    if (encodedOutput.empty())
    {
        return 0U;
    }

    PRECONDITION(
        encodedOutput.size() <= std::numeric_limits<uint32_t>::max(),
        "converted image size ({}) exceeds uint32_t max",
        encodedOutput.size());
    return static_cast<uint32_t>(encodedOutput.size());
}

uint32_t writeToggleGrayRgbImage(int8_t* inputData, uint32_t inputSize, int8_t* outputData, uint32_t outputCapacity)
{
    PRECONDITION(outputData != nullptr, "output buffer must not be null");
    const auto encodedOutput = toggleGrayRgbAndEncodePng(inputData, inputSize);
    if (encodedOutput.empty() || encodedOutput.size() > outputCapacity)
    {
        return 0U;
    }

    std::memcpy(outputData, encodedOutput.data(), encodedOutput.size());
    return static_cast<uint32_t>(encodedOutput.size());
}
}

ToggleGrayRgbPhysicalFunction::ToggleGrayRgbPhysicalFunction(PhysicalFunction imagePhysicalFunction)
    : imagePhysicalFunction(std::move(imagePhysicalFunction))
{
}

VarVal ToggleGrayRgbPhysicalFunction::execute(const Record& record, ArenaRef& arena) const
{
    const auto inputImage = imagePhysicalFunction.execute(record, arena).cast<VariableSizedData>();
    const auto inputImageSize = inputImage.getContentSize();

    const auto outputSize = nautilus::invoke(getToggleGrayRgbImageSize, inputImage.getContent(), inputImageSize);
    if (outputSize == 0)
    {
        return inputImage;
    }

    auto outputImage = arena.allocateVariableSizedData(outputSize);
    const auto writtenSize =
        nautilus::invoke(writeToggleGrayRgbImage, inputImage.getContent(), inputImageSize, outputImage.getContent(), outputSize);

    if (writtenSize == 0)
    {
        return inputImage;
    }

    VarVal(writtenSize).writeToMemory(outputImage.getReference());
    return VariableSizedData(outputImage.getReference(), writtenSize);
}

PhysicalFunctionRegistryReturnType
PhysicalFunctionGeneratedRegistrar::RegisterToggleGrayRgbPhysicalFunction(PhysicalFunctionRegistryArguments arguments)
{
    PRECONDITION(arguments.childFunctions.size() == 1, "ToggleGrayRgb function must have exactly one child function");
    PRECONDITION(arguments.inputTypes.size() == 1, "ToggleGrayRgb function expects exactly one input type descriptor");
    PRECONDITION(
        arguments.inputTypes[0].isType(DataType::Type::VARSIZED),
        "ToggleGrayRgb argument must be VARSIZED, but got {}",
        arguments.inputTypes[0]);

    return ToggleGrayRgbPhysicalFunction(arguments.childFunctions[0]);
}

}
