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
/// Deterministic float-tensor output size: width * height * 3 channels * sizeof(float).
/// (IMREAD_COLOR below always yields 3 channels, so the size does not depend on input content.)
uint64_t floatOutputSize(int32_t width, int32_t height)
{
    if (width <= 0 || height <= 0)
    {
        return 0U;
    }
    return static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 3ULL * sizeof(float);
}

/// Decode the encoded input image, resize to width x height (INTER_LINEAR, as before) and
/// write a CHW, BGR, float32 tensor with unscaled pixel values (range [0, 255]) into
/// outputData. Channel-planar layout (all B, then all G, then all R) is what NCHW models
/// expect for a batch of one. Returns the number of bytes written, or 0 on failure.
uint64_t writeResizedFloat(
    int8_t* inputData, uint64_t inputSize, int32_t width, int32_t height, int8_t* outputData, uint64_t outputCapacity)
{
    PRECONDITION(outputData != nullptr, "output buffer must not be null");
    if (inputData == nullptr || inputSize == 0 || inputSize > static_cast<uint64_t>(std::numeric_limits<int>::max()) || width <= 0
        || height <= 0)
    {
        return 0U;
    }

    const auto requiredBytes = floatOutputSize(width, height);
    if (requiredBytes > outputCapacity)
    {
        return 0U;
    }

    try
    {
        const auto inputSizeInt = static_cast<int>(inputSize);
        cv::Mat encodedBytes(1, inputSizeInt, CV_8U, const_cast<int8_t*>(inputData)); /// NOLINT(cppcoreguidelines-pro-type-const-cast)
        /// IMREAD_COLOR forces a 3-channel BGR image, keeping the output size deterministic.
        cv::Mat inputImage = cv::imdecode(encodedBytes, cv::IMREAD_COLOR);
        if (inputImage.empty())
        {
            return 0U;
        }

        cv::Mat resizedImage;
        cv::resize(inputImage, resizedImage, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);

        cv::Mat floatImage;
        resizedImage.convertTo(floatImage, CV_32FC3);

        /// Write channel-planar (CHW): B plane, then G, then R. cv::split yields continuous
        /// single-channel planes, so each is a straight memcpy.
        std::vector<cv::Mat> channelPlanes;
        cv::split(floatImage, channelPlanes);
        auto* out = reinterpret_cast<float*>(outputData); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
        const auto planeElements = static_cast<size_t>(width) * static_cast<size_t>(height);
        for (size_t c = 0; c < channelPlanes.size(); ++c)
        {
            std::memcpy(out + (c * planeElements), channelPlanes[c].ptr<float>(), planeElements * sizeof(float));
        }
        return requiredBytes;
    }
    catch (const cv::Exception&)
    {
        return 0U;
    }
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

    /// The float-tensor output size is fully determined by width/height (3 channels,
    /// float32), so allocate it directly — no size-probe or retry needed.
    const nautilus::val<uint64_t> outputSize = nautilus::invoke(floatOutputSize, width, height);
    if (outputSize == 0U)
    {
        return inputValue;
    }

    auto outputImage = arena.allocateVariableSizedData(outputSize);
    const nautilus::val<uint64_t> writtenSize = nautilus::invoke(
        writeResizedFloat, inputImage.getContent(), inputImageSize, width, height, outputImage.getContent(), outputSize);

    if (writtenSize == 0U)
    {
        return inputValue;
    }

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
