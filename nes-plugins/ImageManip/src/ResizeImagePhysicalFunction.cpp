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
struct ResizeImageSizeCacheEntry
{
    int32_t width = 0;
    int32_t height = 0;
    uint32_t outputSize = 0;
    bool initialized = false;
};

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

uint32_t getCachedResizedImageSize(int32_t width, int32_t height)
{
    thread_local ResizeImageSizeCacheEntry cache;
    if (cache.initialized && cache.width == width && cache.height == height)
    {
        return cache.outputSize;
    }
    return 0U;
}

void updateCachedResizedImageSize(int32_t width, int32_t height, uint32_t outputSize)
{
    thread_local ResizeImageSizeCacheEntry cache;
    cache.width = width;
    cache.height = height;
    cache.outputSize = outputSize;
    cache.initialized = true;
}

uint32_t writeResizedImage(
    int8_t* inputData, uint32_t inputSize, int32_t width, int32_t height, int8_t* outputData, uint32_t outputCapacity)
{
    PRECONDITION(outputData != nullptr, "output buffer must not be null");
    const auto encodedOutput = resizeAndEncodePng(inputData, inputSize, width, height);
    if (encodedOutput.empty())
    {
        return 0U;
    }

    PRECONDITION(
        encodedOutput.size() <= std::numeric_limits<uint32_t>::max(),
        "resized image size ({}) exceeds uint32_t max",
        encodedOutput.size());
    const auto encodedOutputSize = static_cast<uint32_t>(encodedOutput.size());

    if (encodedOutputSize > outputCapacity)
    {
        /// Caller can retry with the returned required size
        return encodedOutputSize;
    }

    std::memcpy(outputData, encodedOutput.data(), encodedOutput.size());
    return encodedOutputSize;
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

    nautilus::val<uint32_t> outputCapacity = nautilus::invoke(getCachedResizedImageSize, width, height);
    if (outputCapacity == 0)
    {
        /// Best-effort first guess if no cached size exists for these dimensions.
        /// This avoids a dedicated "size" pass in the common case.
        outputCapacity = inputImageSize;
    }

    if (outputCapacity == 0)
    {
        return inputImage;
    }

    auto outputImage = arena.allocateVariableSizedData(outputCapacity);
    nautilus::val<uint32_t> writtenSize = nautilus::invoke(
        writeResizedImage, inputImage.getContent(), inputImageSize, width, height, outputImage.getContent(), outputCapacity);

    if (writtenSize == 0)
    {
        return inputImage;
    }

    if (writtenSize > outputCapacity)
    {
        outputCapacity = writtenSize;
        outputImage = arena.allocateVariableSizedData(outputCapacity);
        writtenSize = nautilus::invoke(
            writeResizedImage, inputImage.getContent(), inputImageSize, width, height, outputImage.getContent(), outputCapacity);

        if (writtenSize == 0 || writtenSize > outputCapacity)
        {
            return inputImage;
        }
    }

    nautilus::invoke(updateCachedResizedImageSize, width, height, writtenSize);

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
