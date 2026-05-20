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

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <Functions/PhysicalFunction.hpp>
#include <Nautilus/DataTypes/VarVal.hpp>
#include <Nautilus/DataTypes/VariableSizedData.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <openvino/core/layout.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/core/preprocess/resize_algorithm.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/tensor.hpp>
#include <zlib.h>
#include <Arena.hpp>
#include <ErrorHandling.hpp>
#include <PhysicalFunctionRegistry.hpp>
#include <nautilus/function.hpp>

namespace NES
{

namespace
{
constexpr std::array<uint8_t, 8> PNG_SIGNATURE{137U, 80U, 78U, 71U, 13U, 10U, 26U, 10U};

struct DecodedPng
{
    uint32_t width = 0;
    uint32_t height = 0;
    uint8_t colorType = 0;
    uint8_t channels = 0;
    std::vector<uint8_t> pixels;
};

struct ResizeImageSizeCacheEntry
{
    int32_t width = 0;
    int32_t height = 0;
    uint64_t outputSize = 0;
    bool initialized = false;
};

struct OpenVinoResizeCacheEntry
{
    uint32_t sourceWidth = 0;
    uint32_t sourceHeight = 0;
    uint32_t targetWidth = 0;
    uint32_t targetHeight = 0;
    uint8_t channels = 0;
    ov::CompiledModel compiledModel;
    bool initialized = false;
};

uint32_t readBigEndian32(const uint8_t* data)
{
    return (static_cast<uint32_t>(data[0]) << 24U) | (static_cast<uint32_t>(data[1]) << 16U)
        | (static_cast<uint32_t>(data[2]) << 8U) | static_cast<uint32_t>(data[3]);
}

void appendBigEndian32(std::vector<uint8_t>& output, uint32_t value)
{
    output.push_back(static_cast<uint8_t>((value >> 24U) & 0xFFU));
    output.push_back(static_cast<uint8_t>((value >> 16U) & 0xFFU));
    output.push_back(static_cast<uint8_t>((value >> 8U) & 0xFFU));
    output.push_back(static_cast<uint8_t>(value & 0xFFU));
}

uint8_t channelsForColorType(uint8_t colorType)
{
    switch (colorType)
    {
        case 0:
            return 1;
        case 2:
            return 3;
        case 4:
            return 2;
        case 6:
            return 4;
        default:
            return 0;
    }
}

uint8_t paethPredictor(uint8_t left, uint8_t above, uint8_t upperLeft)
{
    const auto estimate = static_cast<int>(left) + static_cast<int>(above) - static_cast<int>(upperLeft);
    const auto distanceLeft = std::abs(estimate - static_cast<int>(left));
    const auto distanceAbove = std::abs(estimate - static_cast<int>(above));
    const auto distanceUpperLeft = std::abs(estimate - static_cast<int>(upperLeft));
    if (distanceLeft <= distanceAbove && distanceLeft <= distanceUpperLeft)
    {
        return left;
    }
    if (distanceAbove <= distanceUpperLeft)
    {
        return above;
    }
    return upperLeft;
}

bool decodePng(const int8_t* inputData, uint64_t inputSize, DecodedPng& output)
{
    if (inputData == nullptr || inputSize < PNG_SIGNATURE.size())
    {
        return false;
    }

    /// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) byte-oriented PNG parsing
    const auto* bytes = reinterpret_cast<const uint8_t*>(inputData);
    if (!std::equal(PNG_SIGNATURE.begin(), PNG_SIGNATURE.end(), bytes))
    {
        return false;
    }

    bool sawHeader = false;
    std::vector<uint8_t> compressedImageData;
    auto offset = uint64_t{PNG_SIGNATURE.size()};
    while (offset + 8U <= inputSize)
    {
        const auto chunkLength = readBigEndian32(bytes + offset);
        offset += 4U;
        const std::string_view chunkType(reinterpret_cast<const char*>(bytes + offset), 4U); /// NOLINT
        offset += 4U;
        if (chunkLength > inputSize - offset || inputSize - offset - chunkLength < 4U)
        {
            return false;
        }

        const auto* chunkData = bytes + offset;
        if (chunkType == "IHDR")
        {
            if (chunkLength != 13U)
            {
                return false;
            }

            output.width = readBigEndian32(chunkData);
            output.height = readBigEndian32(chunkData + 4U);
            const auto bitDepth = chunkData[8];
            output.colorType = chunkData[9];
            const auto compressionMethod = chunkData[10];
            const auto filterMethod = chunkData[11];
            const auto interlaceMethod = chunkData[12];
            output.channels = channelsForColorType(output.colorType);

            if (output.width == 0 || output.height == 0 || bitDepth != 8U || output.channels == 0 || compressionMethod != 0U
                || filterMethod != 0U || interlaceMethod != 0U)
            {
                return false;
            }
            sawHeader = true;
        }
        else if (chunkType == "IDAT")
        {
            compressedImageData.insert(compressedImageData.end(), chunkData, chunkData + chunkLength);
        }
        else if (chunkType == "IEND")
        {
            break;
        }

        offset += static_cast<uint64_t>(chunkLength) + 4U;
    }

    if (!sawHeader || compressedImageData.empty())
    {
        return false;
    }

    const auto rowBytes = static_cast<uint64_t>(output.width) * output.channels;
    const auto decompressedSize = (rowBytes + 1U) * output.height;
    if (decompressedSize > std::numeric_limits<uLongf>::max())
    {
        return false;
    }

    std::vector<uint8_t> filteredImageData(decompressedSize);
    auto actualSize = static_cast<uLongf>(filteredImageData.size());
    if (::uncompress(
            filteredImageData.data(),
            &actualSize,
            compressedImageData.data(),
            static_cast<uLong>(compressedImageData.size()))
        != Z_OK)
    {
        return false;
    }
    if (actualSize != filteredImageData.size())
    {
        return false;
    }

    output.pixels.assign(rowBytes * output.height, 0U);
    const auto bytesPerPixel = static_cast<uint64_t>(output.channels);
    for (uint32_t row = 0; row < output.height; ++row)
    {
        const auto filteredOffset = static_cast<uint64_t>(row) * (rowBytes + 1U);
        const auto filter = filteredImageData[filteredOffset];
        if (filter > 4U)
        {
            return false;
        }

        const auto outputOffset = static_cast<uint64_t>(row) * rowBytes;
        for (uint64_t column = 0; column < rowBytes; ++column)
        {
            const auto value = filteredImageData[filteredOffset + 1U + column];
            const auto left = column >= bytesPerPixel ? output.pixels[outputOffset + column - bytesPerPixel] : 0U;
            const auto above = row > 0 ? output.pixels[outputOffset + column - rowBytes] : 0U;
            const auto upperLeft = row > 0 && column >= bytesPerPixel ? output.pixels[outputOffset + column - rowBytes - bytesPerPixel] : 0U;

            uint8_t predictor = 0;
            switch (filter)
            {
                case 1:
                    predictor = left;
                    break;
                case 2:
                    predictor = above;
                    break;
                case 3:
                    predictor = static_cast<uint8_t>((static_cast<uint16_t>(left) + static_cast<uint16_t>(above)) / 2U);
                    break;
                case 4:
                    predictor = paethPredictor(left, above, upperLeft);
                    break;
                default:
                    break;
            }
            output.pixels[outputOffset + column] = static_cast<uint8_t>(value + predictor);
        }
    }

    return true;
}

void appendPngChunk(std::vector<uint8_t>& png, std::string_view type, const std::vector<uint8_t>& data)
{
    PRECONDITION(type.size() == 4U, "PNG chunk types must be four bytes");
    PRECONDITION(data.size() <= std::numeric_limits<uint32_t>::max(), "PNG chunk size {} exceeds uint32_t max", data.size());

    appendBigEndian32(png, static_cast<uint32_t>(data.size()));
    const auto chunkStart = png.size();
    png.insert(png.end(), type.begin(), type.end());
    png.insert(png.end(), data.begin(), data.end());
    const auto crc = ::crc32(0L, reinterpret_cast<const Bytef*>(png.data() + chunkStart), static_cast<uInt>(png.size() - chunkStart)); /// NOLINT
    appendBigEndian32(png, static_cast<uint32_t>(crc));
}

std::vector<uint8_t> encodePng(const DecodedPng& image)
{
    const auto rowBytes = static_cast<uint64_t>(image.width) * image.channels;
    const auto filteredSize = (rowBytes + 1U) * image.height;
    if (filteredSize > std::numeric_limits<uLong>::max())
    {
        return {};
    }

    std::vector<uint8_t> filteredImageData(filteredSize, 0U);
    for (uint32_t row = 0; row < image.height; ++row)
    {
        const auto filteredOffset = static_cast<uint64_t>(row) * (rowBytes + 1U);
        const auto inputOffset = static_cast<uint64_t>(row) * rowBytes;
        std::memcpy(filteredImageData.data() + filteredOffset + 1U, image.pixels.data() + inputOffset, rowBytes);
    }

    auto compressedSize = ::compressBound(static_cast<uLong>(filteredImageData.size()));
    std::vector<uint8_t> compressedImageData(compressedSize);
    if (::compress2(
            compressedImageData.data(),
            &compressedSize,
            filteredImageData.data(),
            static_cast<uLong>(filteredImageData.size()),
            Z_BEST_SPEED)
        != Z_OK)
    {
        return {};
    }
    compressedImageData.resize(compressedSize);

    std::vector<uint8_t> header;
    header.reserve(13U);
    appendBigEndian32(header, image.width);
    appendBigEndian32(header, image.height);
    header.push_back(8U);
    header.push_back(image.colorType);
    header.push_back(0U);
    header.push_back(0U);
    header.push_back(0U);

    std::vector<uint8_t> png(PNG_SIGNATURE.begin(), PNG_SIGNATURE.end());
    appendPngChunk(png, "IHDR", header);
    appendPngChunk(png, "IDAT", compressedImageData);
    appendPngChunk(png, "IEND", {});
    return png;
}

OpenVinoResizeCacheEntry& getOpenVinoResizeCache()
{
    thread_local OpenVinoResizeCacheEntry cache;
    return cache;
}

ov::CompiledModel compileResizeModel(uint32_t sourceWidth, uint32_t sourceHeight, uint32_t targetWidth, uint32_t targetHeight, uint8_t channels)
{
    auto parameter = std::make_shared<ov::opset8::Parameter>(ov::element::u8, ov::Shape{1U, targetHeight, targetWidth, channels});
    auto result = std::make_shared<ov::opset8::Result>(parameter);
    auto model = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter});

    ov::preprocess::PrePostProcessor preprocessor(model);
    preprocessor.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_spatial_static_shape(sourceHeight, sourceWidth);
    preprocessor.input().model().set_layout("NHWC");
    preprocessor.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR, targetHeight, targetWidth);
    model = preprocessor.build();

    static ov::Core core;
    static std::mutex coreMutex;
    const std::scoped_lock lock(coreMutex);
    return core.compile_model(model, "CPU");
}

ov::CompiledModel& getResizeModel(uint32_t sourceWidth, uint32_t sourceHeight, uint32_t targetWidth, uint32_t targetHeight, uint8_t channels)
{
    auto& cache = getOpenVinoResizeCache();
    if (!cache.initialized || cache.sourceWidth != sourceWidth || cache.sourceHeight != sourceHeight || cache.targetWidth != targetWidth
        || cache.targetHeight != targetHeight || cache.channels != channels)
    {
        cache.sourceWidth = sourceWidth;
        cache.sourceHeight = sourceHeight;
        cache.targetWidth = targetWidth;
        cache.targetHeight = targetHeight;
        cache.channels = channels;
        cache.compiledModel = compileResizeModel(sourceWidth, sourceHeight, targetWidth, targetHeight, channels);
        cache.initialized = true;
    }
    return cache.compiledModel;
}

bool resizeWithOpenVino(const DecodedPng& inputImage, uint32_t targetWidth, uint32_t targetHeight, DecodedPng& outputImage)
{
    if (inputImage.width == targetWidth && inputImage.height == targetHeight)
    {
        outputImage = inputImage;
        return true;
    }

    auto& compiledModel = getResizeModel(inputImage.width, inputImage.height, targetWidth, targetHeight, inputImage.channels);
    auto inferRequest = compiledModel.create_infer_request();

    ov::Tensor inputTensor(
        ov::element::u8,
        ov::Shape{1U, inputImage.height, inputImage.width, inputImage.channels},
        const_cast<uint8_t*>(inputImage.pixels.data())); /// NOLINT(cppcoreguidelines-pro-type-const-cast)
    inferRequest.set_input_tensor(inputTensor);

    outputImage = inputImage;
    outputImage.width = targetWidth;
    outputImage.height = targetHeight;
    outputImage.pixels.assign(static_cast<uint64_t>(targetWidth) * targetHeight * inputImage.channels, 0U);

    ov::Tensor outputTensor(
        ov::element::u8,
        ov::Shape{1U, targetHeight, targetWidth, inputImage.channels},
        outputImage.pixels.data());
    inferRequest.set_output_tensor(outputTensor);
    inferRequest.infer();
    return true;
}

std::vector<uint8_t> resizeAndEncodePng(const int8_t* inputData, uint64_t inputSize, int32_t width, int32_t height)
{
    if (inputData == nullptr || inputSize == 0 || width <= 0 || height <= 0)
    {
        return {};
    }

    DecodedPng inputImage;
    if (!decodePng(inputData, inputSize, inputImage))
    {
        return {};
    }

    DecodedPng resizedImage;
    try
    {
        if (!resizeWithOpenVino(inputImage, static_cast<uint32_t>(width), static_cast<uint32_t>(height), resizedImage))
        {
            return {};
        }
    }
    catch (const std::exception&)
    {
        return {};
    }

    return encodePng(resizedImage);
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

uint64_t writeResizedImage(
    int8_t* inputData,
    uint64_t inputSize,
    int32_t width,
    int32_t height,
    int8_t* outputData,
    uint64_t outputCapacity)
{
    PRECONDITION(outputData != nullptr, "output buffer must not be null");
    const auto encodedOutput = resizeAndEncodePng(inputData, inputSize, width, height);
    if (encodedOutput.empty())
    {
        return 0U;
    }

    if (encodedOutput.size() > outputCapacity)
    {
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
    const auto width = widthPhysicalFunction.execute(record, arena).castToType(DataType::Type::INT32).getRawValueAs<nautilus::val<int32_t>>();
    const auto height = heightPhysicalFunction.execute(record, arena).castToType(DataType::Type::INT32).getRawValueAs<nautilus::val<int32_t>>();

    nautilus::val<uint64_t> outputCapacity = nautilus::invoke(getCachedResizedImageSize, width, height);
    if (outputCapacity == 0U)
    {
        outputCapacity = inputImageSize;
    }

    if (outputCapacity == 0U)
    {
        return inputValue;
    }

    auto outputImage = arena.allocateVariableSizedData(outputCapacity);
    nautilus::val<uint64_t> writtenSize = nautilus::invoke(
        writeResizedImage,
        inputImage.getContent(),
        inputImageSize,
        width,
        height,
        outputImage.getContent(),
        outputCapacity);

    if (writtenSize == 0U)
    {
        return inputValue;
    }

    if (writtenSize > outputCapacity)
    {
        outputCapacity = writtenSize;
        outputImage = arena.allocateVariableSizedData(outputCapacity);
        writtenSize = nautilus::invoke(
            writeResizedImage,
            inputImage.getContent(),
            inputImageSize,
            width,
            height,
            outputImage.getContent(),
            outputCapacity);

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
