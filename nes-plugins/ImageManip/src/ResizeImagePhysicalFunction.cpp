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
#include <nautilus/function.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
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
    uint64_t outputSize = 0;
    bool initialized = false;
};

std::vector<uint8_t> resizeAndEncodePng(const int8_t* inputData, uint64_t inputSize, int32_t width, int32_t height)
{
    if (inputData == nullptr || inputSize == 0 || inputSize > static_cast<uint64_t>(std::numeric_limits<int>::max()) || width <= 0
        || height <= 0)
    {
        return {};
    }

    try
    {
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
    catch (const cv::Exception&)
    {
        return {};
    }
}

ResizeImageSizeCacheEntry& getResizedImageSizeCache()
{
    thread_local ResizeImageSizeCacheEntry cache;
    return cache;
}

uint64_t getCachedResizedImageSize(int32_t width, int32_t height)
{
    auto& cache = getResizedImageSizeCache();
    if (cache.initialized && cache.width == width && cache.height == height)
    {
        return cache.outputSize;
    }
    return 0U;
}

void updateCachedResizedImageSize(int32_t width, int32_t height, uint64_t outputSize)
{
    auto& cache = getResizedImageSizeCache();
    cache.width = width;
    cache.height = height;
    cache.outputSize = outputSize;
    cache.initialized = true;
}

uint64_t
writeResizedImage(int8_t* inputData, uint64_t inputSize, int32_t width, int32_t height, int8_t* outputData, uint64_t outputCapacity)
{
    PRECONDITION(outputData != nullptr, "output buffer must not be null");
    const auto encodedOutput = resizeAndEncodePng(inputData, inputSize, width, height);
    if (encodedOutput.empty())
    {
        return 0U;
    }

    if (encodedOutput.size() > outputCapacity)
    {
        /// Caller can retry with the returned required size.
        return encodedOutput.size();
    }

    std::memcpy(outputData, encodedOutput.data(), encodedOutput.size());
    return encodedOutput.size();
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
    const auto inputValue = imagePhysicalFunction.execute(record, arena);
    const auto inputImage = inputValue.getRawValueAs<VariableSizedData>();
    const auto inputImageSize = inputImage.getSize();
    const auto width
        = widthPhysicalFunction.execute(record, arena).castToType(DataType::Type::INT32).getRawValueAs<nautilus::val<int32_t>>();
    const auto height
        = heightPhysicalFunction.execute(record, arena).castToType(DataType::Type::INT32).getRawValueAs<nautilus::val<int32_t>>();

    nautilus::val<uint64_t> outputCapacity = nautilus::invoke(getCachedResizedImageSize, width, height);
    if (outputCapacity == 0U)
    {
        /// Best-effort first guess if no cached size exists for these dimensions.
        /// This avoids a dedicated size pass in the common case.
        outputCapacity = inputImageSize;
    }

    if (outputCapacity == 0U)
    {
        return inputValue;
    }

    auto outputImage = arena.allocateVariableSizedData(outputCapacity);
    nautilus::val<uint64_t> writtenSize = nautilus::invoke(
        writeResizedImage, inputImage.getContent(), inputImageSize, width, height, outputImage.getContent(), outputCapacity);

    if (writtenSize == 0U)
    {
        return inputValue;
    }

    if (writtenSize > outputCapacity)
    {
        outputCapacity = writtenSize;
        outputImage = arena.allocateVariableSizedData(outputCapacity);
        writtenSize = nautilus::invoke(
            writeResizedImage, inputImage.getContent(), inputImageSize, width, height, outputImage.getContent(), outputCapacity);

        if (writtenSize == 0U || writtenSize > outputCapacity)
        {
            return inputValue;
        }
    }

    nautilus::invoke(updateCachedResizedImageSize, width, height, writtenSize);
    return VariableSizedData(outputImage.getContent(), writtenSize);
}

PhysicalFunctionRegistryReturnType
PhysicalFunctionGeneratedRegistrar::RegisterRESIZE_IMAGEPhysicalFunction(PhysicalFunctionRegistryArguments arguments)
{
    PRECONDITION(arguments.childFunctions.size() == 3, "RESIZE_IMAGE function must have exactly three child functions");
    PRECONDITION(arguments.inputTypes.size() == 3, "RESIZE_IMAGE function expects exactly three input type descriptors");
    PRECONDITION(
        arguments.inputTypes[0].isType(DataType::Type::VARSIZED),
        "RESIZE_IMAGE first argument must be VARSIZED, but got {}",
        arguments.inputTypes[0]);
    PRECONDITION(arguments.inputTypes[1].isInteger(), "RESIZE_IMAGE second argument must be integer, but got {}", arguments.inputTypes[1]);
    PRECONDITION(arguments.inputTypes[2].isInteger(), "RESIZE_IMAGE third argument must be integer, but got {}", arguments.inputTypes[2]);

    return ResizeImagePhysicalFunction(arguments.childFunctions[0], arguments.childFunctions[1], arguments.childFunctions[2]);
}

}
