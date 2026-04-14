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

#include "../include/SSIMPhysicalFunction.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
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

struct SSIMFunctionState
{
    mutable std::mutex mutex;
    std::vector<int8_t> previousImage;
};

namespace
{
cv::Mat gradientImage(const cv::Mat& gray)
{
    cv::Mat gx, gy, mag;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    mag = cv::abs(gx) + cv::abs(gy);
    return mag;
}

float computeSsim(const int8_t* lhsData, uint32_t lhsSize, const int8_t* rhsData, uint32_t rhsSize)
{
    if (lhsData == nullptr || rhsData == nullptr || lhsSize == 0 || rhsSize == 0)
    {
        return 0.0F;
    }

    const auto lhsSizeInt = static_cast<int>(lhsSize);
    const auto rhsSizeInt = static_cast<int>(rhsSize);

    cv::Mat lhsBytes(1, lhsSizeInt, CV_8U, const_cast<int8_t*>(lhsData)); /// NOLINT(cppcoreguidelines-pro-type-const-cast)
    cv::Mat rhsBytes(1, rhsSizeInt, CV_8U, const_cast<int8_t*>(rhsData)); /// NOLINT(cppcoreguidelines-pro-type-const-cast)

    cv::Mat lhsImage = cv::imdecode(lhsBytes, cv::IMREAD_UNCHANGED);
    cv::Mat rhsImage = cv::imdecode(rhsBytes, cv::IMREAD_UNCHANGED);

    if (lhsImage.empty() || rhsImage.empty() || lhsImage.size() != rhsImage.size())
    {
        return 0.0F;
    }

    cv::Mat lhsProcessed = lhsImage;
    cv::Mat rhsProcessed = rhsImage;
    if (lhsImage.channels() == 3)
    {
        cv::cvtColor(lhsImage, lhsProcessed, cv::COLOR_BGR2GRAY);
    }
    if (rhsImage.channels() == 3)
    {
        cv::cvtColor(rhsImage, rhsProcessed, cv::COLOR_BGR2GRAY);
    }
    if (lhsProcessed.channels() != rhsProcessed.channels())
    {
        return 0.0F;
    }

    lhsProcessed = gradientImage(lhsProcessed);
    rhsProcessed = gradientImage(rhsProcessed);

    cv::Mat lhsFloat;
    cv::Mat rhsFloat;
    lhsProcessed.convertTo(lhsFloat, CV_32F);
    rhsProcessed.convertTo(rhsFloat, CV_32F);

    cv::Mat mu1;
    cv::Mat mu2;
    cv::GaussianBlur(lhsFloat, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(rhsFloat, mu2, cv::Size(11, 11), 1.5);

    const cv::Mat mu1Sq = mu1.mul(mu1);
    const cv::Mat mu2Sq = mu2.mul(mu2);
    const cv::Mat mu1Mu2 = mu1.mul(mu2);

    cv::Mat sigma1Sq;
    cv::Mat sigma2Sq;
    cv::Mat sigma12;

    cv::GaussianBlur(lhsFloat.mul(lhsFloat), sigma1Sq, cv::Size(11, 11), 1.5);
    sigma1Sq -= mu1Sq;

    cv::GaussianBlur(rhsFloat.mul(rhsFloat), sigma2Sq, cv::Size(11, 11), 1.5);
    sigma2Sq -= mu2Sq;

    cv::GaussianBlur(lhsFloat.mul(rhsFloat), sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1Mu2;

    constexpr double c1 = (0.01 * 255.0) * (0.01 * 255.0);
    constexpr double c2 = (0.03 * 255.0) * (0.03 * 255.0);

    cv::Mat numerator = (2.0 * mu1Mu2 + c1).mul(2.0 * sigma12 + c2);
    cv::Mat denominator = (mu1Sq + mu2Sq + c1).mul(sigma1Sq + sigma2Sq + c2);

    constexpr double denominatorEpsilon = 1e-12;
    denominator += denominatorEpsilon;

    cv::Mat ssimMap;
    cv::divide(numerator, denominator, ssimMap);

    return cv::mean(ssimMap)[0];
}

uint32_t getPreviousImageSize(SSIMFunctionState* state)
{
    std::scoped_lock lock(state->mutex);
    PRECONDITION(
        state->previousImage.size() <= std::numeric_limits<uint32_t>::max(),
        "previous image size ({}) exceeds uint32_t max",
        state->previousImage.size());
    return static_cast<uint32_t>(state->previousImage.size());
}

uint32_t applySsimFilter(
    SSIMFunctionState* state, const int8_t* currentImageData, uint32_t currentImageSize, float threshold, int8_t* outputImageData)
{
    PRECONDITION(state != nullptr, "SSIM state must not be null");
    PRECONDITION(currentImageData != nullptr, "current image must not be null");
    PRECONDITION(outputImageData != nullptr, "output image must not be null");

    std::scoped_lock lock(state->mutex);

    if (state->previousImage.empty())
    {
        std::memcpy(outputImageData, currentImageData, currentImageSize);
        state->previousImage.assign(currentImageData, currentImageData + currentImageSize);
        return currentImageSize;
    }

    const auto previousSize = static_cast<uint32_t>(state->previousImage.size());
    const auto similarity = computeSsim(state->previousImage.data(), previousSize, currentImageData, currentImageSize);

    if (similarity >= threshold)
    {
        std::memcpy(outputImageData, state->previousImage.data(), previousSize);
        return previousSize;
    }

    std::memcpy(outputImageData, currentImageData, currentImageSize);
    state->previousImage.assign(currentImageData, currentImageData + currentImageSize);
    return currentImageSize;
}
}

SSIMPhysicalFunction::SSIMPhysicalFunction(PhysicalFunction imagePhysicalFunction, PhysicalFunction thresholdPhysicalFunction)
    : imagePhysicalFunction(std::move(imagePhysicalFunction))
    , thresholdPhysicalFunction(std::move(thresholdPhysicalFunction))
    , state(std::make_shared<SSIMFunctionState>())
{
}

VarVal SSIMPhysicalFunction::execute(const Record& record, ArenaRef& arena) const
{
    const auto currentImage = imagePhysicalFunction.execute(record, arena).cast<VariableSizedData>();
    const auto currentImageSize = currentImage.getContentSize();
    const auto threshold = thresholdPhysicalFunction.execute(record, arena).cast<nautilus::val<float>>();
    const nautilus::val<SSIMFunctionState*> stateRef{state.get()};

    const auto previousImageSize = nautilus::invoke(getPreviousImageSize, stateRef);
    auto maxOutputSize = currentImageSize;
    if (previousImageSize > maxOutputSize)
    {
        maxOutputSize = previousImageSize;
    }

    auto outputImage = arena.allocateVariableSizedData(maxOutputSize);
    const auto outputSize = nautilus::invoke(
        applySsimFilter, stateRef, currentImage.getContent(), currentImageSize, threshold, outputImage.getContent());

    VarVal(outputSize).writeToMemory(outputImage.getReference());
    return VariableSizedData(outputImage.getReference(), outputSize);
}

PhysicalFunctionRegistryReturnType
PhysicalFunctionGeneratedRegistrar::RegisterSSIMPhysicalFunction(PhysicalFunctionRegistryArguments physicalFunctionRegistryArguments)
{
    PRECONDITION(
        physicalFunctionRegistryArguments.childFunctions.size() == 2,
        "SSIM function must have exactly two child functions");
    PRECONDITION(
        physicalFunctionRegistryArguments.inputTypes.size() == 2, "SSIM function expects exactly two input type descriptors");
    PRECONDITION(
        physicalFunctionRegistryArguments.inputTypes[0].isType(DataType::Type::VARSIZED),
        "SSIM first argument must be VARSIZED, but got {}",
        physicalFunctionRegistryArguments.inputTypes[0]);
    PRECONDITION(
        physicalFunctionRegistryArguments.inputTypes[1].isFloat(),
        "SSIM second argument must be FLOAT32/FLOAT64, but got {}",
        physicalFunctionRegistryArguments.inputTypes[1]);

    return SSIMPhysicalFunction(physicalFunctionRegistryArguments.childFunctions[0], physicalFunctionRegistryArguments.childFunctions[1]);
}
}
