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

#include "../include/ResizeImagePhysicalFunction.hpp"

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
std::vector<uint8_t> resizeAndEncodePng(const int8_t* inputData, uint32_t inputSize, int32_t width, int32_t height)
{
    if (inputData == nullptr || inputSize == 0 || width <= 0 || height <= 0)
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

    cv::Mat resizedImage;
    cv::resize(inputImage, resizedImage, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);

    std::vector<uint8_t> encodedOutput;
    if (!cv::imencode(".png", resizedImage, encodedOutput))
    {
        return {};
    }
    return encodedOutput;
}

uint32_t getResizedImageSize(int8_t* inputData, uint32_t inputSize, int32_t width, int32_t height)
{
    const auto encodedOutput = resizeAndEncodePng(inputData, inputSize, width, height);
    if (encodedOutput.empty())
    {
        return 0U;
    }

    PRECONDITION(
        encodedOutput.size() <= std::numeric_limits<uint32_t>::max(),
        "resized image size ({}) exceeds uint32_t max",
        encodedOutput.size());
    return static_cast<uint32_t>(encodedOutput.size());
}

uint32_t writeResizedImage(
    int8_t* inputData, uint32_t inputSize, int32_t width, int32_t height, int8_t* outputData, uint32_t outputCapacity)
{
    PRECONDITION(outputData != nullptr, "output buffer must not be null");
    const auto encodedOutput = resizeAndEncodePng(inputData, inputSize, width, height);
    if (encodedOutput.empty() || encodedOutput.size() > outputCapacity)
    {
        return 0U;
    }

    std::memcpy(outputData, encodedOutput.data(), encodedOutput.size());
    return static_cast<uint32_t>(encodedOutput.size());
}
}

ResizeImagePhysicalFunction::ResizeImagePhysicalFunction(
    PhysicalFunction imagePhysicalFunction, PhysicalFunction widthPhysicalFunction, PhysicalFunction heightPhysicalFunction)
    : imagePhysicalFunction(std::move(imagePhysicalFunction))
    , widthPhysicalFunction(std::move(widthPhysicalFunction))
    , heightPhysicalFunction(std::move(heightPhysicalFunction))
{
}

VarVal ResizeImagePhysicalFunction::execute(const Record& record, ArenaRef& arena) const
{
    const auto inputImage = imagePhysicalFunction.execute(record, arena).cast<VariableSizedData>();
    const auto inputImageSize = inputImage.getContentSize();
    const auto width = widthPhysicalFunction.execute(record, arena).castToType(DataType::Type::INT32).cast<nautilus::val<int32_t>>();
    const auto height = heightPhysicalFunction.execute(record, arena).castToType(DataType::Type::INT32).cast<nautilus::val<int32_t>>();

    const auto outputSize = nautilus::invoke(getResizedImageSize, inputImage.getContent(), inputImageSize, width, height);
    if (outputSize == 0)
    {
        return inputImage;
    }

    auto outputImage = arena.allocateVariableSizedData(outputSize);
    const auto writtenSize = nautilus::invoke(
        writeResizedImage, inputImage.getContent(), inputImageSize, width, height, outputImage.getContent(), outputSize);

    if (writtenSize == 0)
    {
        return inputImage;
    }

    VarVal(writtenSize).writeToMemory(outputImage.getReference());
    return VariableSizedData(outputImage.getReference(), writtenSize);
}

PhysicalFunctionRegistryReturnType
PhysicalFunctionGeneratedRegistrar::RegisterResizeImagePhysicalFunction(PhysicalFunctionRegistryArguments arguments)
{
    PRECONDITION(arguments.childFunctions.size() == 3, "ResizeImage function must have exactly three child functions");
    PRECONDITION(arguments.inputTypes.size() == 3, "ResizeImage function expects exactly three input type descriptors");
    PRECONDITION(
        arguments.inputTypes[0].isType(DataType::Type::VARSIZED),
        "ResizeImage first argument must be VARSIZED, but got {}",
        arguments.inputTypes[0]);
    PRECONDITION(arguments.inputTypes[1].isInteger(), "ResizeImage second argument must be integer, but got {}", arguments.inputTypes[1]);
    PRECONDITION(arguments.inputTypes[2].isInteger(), "ResizeImage third argument must be integer, but got {}", arguments.inputTypes[2]);

    return ResizeImagePhysicalFunction(arguments.childFunctions[0], arguments.childFunctions[1], arguments.childFunctions[2]);
}

}
