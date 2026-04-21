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

#include "../include/ConvertColorSchemePhysicalFunction.hpp"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
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
#include <magic_enum/magic_enum.hpp>
#include <nautilus/function.hpp>
#include <Arena.hpp>
#include <ErrorHandling.hpp>
#include <PhysicalFunctionRegistry.hpp>

namespace NES
{

namespace
{
enum class ConversionMode
{
    RGB_TO_BRG,
    BRG_TO_RGB,
    BRG_TO_GRAYSCALE,
    GRAYSCALE_TO_BRG,
    RGB_TO_GRAYSCALE,
    GRAYSCALE_TO_RGB
};

std::string uppercaseConversionMode(std::string_view conversionMode)
{
    std::string uppercaseMode;
    uppercaseMode.reserve(conversionMode.size());
    for (const char modeChar : conversionMode)
    {
        const auto unsignedModeChar = static_cast<unsigned char>(modeChar);
        uppercaseMode.push_back(static_cast<char>(std::toupper(unsignedModeChar)));
    }
    return uppercaseMode;
}

std::optional<ConversionMode> decodeConversionMode(const int8_t* conversionModeData, uint32_t conversionModeSize)
{
    if (conversionModeData == nullptr || conversionModeSize == 0)
    {
        return std::nullopt;
    }
    const auto conversionMode = std::string_view{reinterpret_cast<const char*>(conversionModeData), conversionModeSize};
    const auto uppercaseMode = uppercaseConversionMode(conversionMode);
    if (uppercaseMode.empty())
    {
        return std::nullopt;
    }
    return magic_enum::enum_cast<ConversionMode>(uppercaseMode);
}

cv::Mat convertRgbToBrg(const cv::Mat& inputImage)
{
    if (inputImage.channels() != 3)
    {
        return {};
    }

    cv::Mat converted(inputImage.rows, inputImage.cols, inputImage.type());
    constexpr int fromTo[] = {1, 0, 2, 1, 0, 2};
    cv::mixChannels(&inputImage, 1, &converted, 1, fromTo, 3);
    return converted;
}

cv::Mat convertBrgToRgb(const cv::Mat& inputImage)
{
    if (inputImage.channels() != 3)
    {
        return {};
    }

    cv::Mat converted(inputImage.rows, inputImage.cols, inputImage.type());
    constexpr int fromTo[] = {2, 0, 0, 1, 1, 2};
    cv::mixChannels(&inputImage, 1, &converted, 1, fromTo, 3);
    return converted;
}

cv::Mat applyConversionMode(const cv::Mat& inputImage, const ConversionMode conversionMode)
{
    cv::Mat convertedImage;
    switch (conversionMode)
    {
        case ConversionMode::RGB_TO_BRG:
            return convertRgbToBrg(inputImage);
        case ConversionMode::BRG_TO_RGB:
            return convertBrgToRgb(inputImage);
        case ConversionMode::BRG_TO_GRAYSCALE:
        {
            const auto rgbImage = convertBrgToRgb(inputImage);
            if (rgbImage.empty())
            {
                return {};
            }
            cv::cvtColor(rgbImage, convertedImage, cv::COLOR_BGR2GRAY);
            return convertedImage;
        }
        case ConversionMode::GRAYSCALE_TO_BRG:
            if (inputImage.channels() != 1)
            {
                return {};
            }
            cv::cvtColor(inputImage, convertedImage, cv::COLOR_GRAY2BGR);
            return convertedImage;
        case ConversionMode::RGB_TO_GRAYSCALE:
            if (inputImage.channels() == 3)
            {
                cv::cvtColor(inputImage, convertedImage, cv::COLOR_BGR2GRAY);
                return convertedImage;
            }
            if (inputImage.channels() == 4)
            {
                cv::cvtColor(inputImage, convertedImage, cv::COLOR_BGRA2GRAY);
                return convertedImage;
            }
            return {};
        case ConversionMode::GRAYSCALE_TO_RGB:
            if (inputImage.channels() != 1)
            {
                return {};
            }
            cv::cvtColor(inputImage, convertedImage, cv::COLOR_GRAY2BGR);
            return convertedImage;
    }
    return {};
}

std::vector<uint8_t> convertColorSchemeAndEncodePng(
    const int8_t* inputData, uint32_t inputSize, const int8_t* conversionModeData, uint32_t conversionModeSize)
{
    if (inputData == nullptr || inputSize == 0)
    {
        return {};
    }
    const auto decodedConversionMode = decodeConversionMode(conversionModeData, conversionModeSize);
    if (!decodedConversionMode.has_value())
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

    const auto convertedImage = applyConversionMode(inputImage, *decodedConversionMode);
    if (convertedImage.empty())
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

uint32_t getConvertedColorSchemeImageSize(
    int8_t* inputData, uint32_t inputSize, int8_t* conversionModeData, uint32_t conversionModeSize)
{
    const auto encodedOutput = convertColorSchemeAndEncodePng(inputData, inputSize, conversionModeData, conversionModeSize);
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

uint32_t writeConvertedColorSchemeImage(
    int8_t* inputData,
    uint32_t inputSize,
    int8_t* conversionModeData,
    uint32_t conversionModeSize,
    int8_t* outputData,
    uint32_t outputCapacity)
{
    PRECONDITION(outputData != nullptr, "output buffer must not be null");
    const auto encodedOutput = convertColorSchemeAndEncodePng(inputData, inputSize, conversionModeData, conversionModeSize);
    if (encodedOutput.empty() || encodedOutput.size() > outputCapacity)
    {
        return 0U;
    }

    std::memcpy(outputData, encodedOutput.data(), encodedOutput.size());
    return static_cast<uint32_t>(encodedOutput.size());
}
}

ConvertColorSchemePhysicalFunction::ConvertColorSchemePhysicalFunction(
    PhysicalFunction imagePhysicalFunction, PhysicalFunction conversionModePhysicalFunction)
    : imagePhysicalFunction(std::move(imagePhysicalFunction))
    , conversionModePhysicalFunction(std::move(conversionModePhysicalFunction))
{
}

VarVal ConvertColorSchemePhysicalFunction::execute(const Record& record, ArenaRef& arena) const
{
    const auto inputImage = imagePhysicalFunction.execute(record, arena).cast<VariableSizedData>();
    const auto inputImageSize = inputImage.getContentSize();
    const auto conversionMode = conversionModePhysicalFunction.execute(record, arena).cast<VariableSizedData>();

    const auto outputSize = nautilus::invoke(
        getConvertedColorSchemeImageSize,
        inputImage.getContent(),
        inputImageSize,
        conversionMode.getContent(),
        conversionMode.getContentSize());
    if (outputSize == 0)
    {
        return inputImage;
    }

    auto outputImage = arena.allocateVariableSizedData(outputSize);
    const auto writtenSize = nautilus::invoke(
        writeConvertedColorSchemeImage,
        inputImage.getContent(),
        inputImageSize,
        conversionMode.getContent(),
        conversionMode.getContentSize(),
        outputImage.getContent(),
        outputSize);

    if (writtenSize == 0)
    {
        return inputImage;
    }

    VarVal(writtenSize).writeToMemory(outputImage.getReference());
    return VariableSizedData(outputImage.getReference(), writtenSize);
}

PhysicalFunctionRegistryReturnType
PhysicalFunctionGeneratedRegistrar::RegisterConvertColorSchemePhysicalFunction(PhysicalFunctionRegistryArguments arguments)
{
    PRECONDITION(
        arguments.childFunctions.size() == arguments.inputTypes.size(),
        "ConvertColorScheme function requires the same number of child functions and input type descriptors");
    PRECONDITION(
        arguments.childFunctions.size() == 2,
        "ConvertColorScheme function must have exactly two child functions");
    PRECONDITION(arguments.inputTypes.size() == 2, "ConvertColorScheme function expects exactly two input type descriptors");
    PRECONDITION(
        arguments.inputTypes[0].isType(DataType::Type::VARSIZED),
        "ConvertColorScheme first argument must be VARSIZED, but got {}",
        arguments.inputTypes[0]);
    PRECONDITION(
        arguments.inputTypes[1].isType(DataType::Type::VARSIZED),
        "ConvertColorScheme second argument must be VARSIZED, but got {}",
        arguments.inputTypes[1]);

    return ConvertColorSchemePhysicalFunction(arguments.childFunctions[0], arguments.childFunctions[1]);
}

}
