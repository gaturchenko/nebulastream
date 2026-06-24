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

#include "../include/AudioDenoising.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

#include <fftw3.h>

namespace NES::AudioDenoising
{

namespace
{
constexpr uint64_t FRAME_SIZE = 512;
constexpr uint64_t HOP_SIZE = FRAME_SIZE / 2;
constexpr double OVER_SUBTRACTION_FACTOR = 1.0;
constexpr double SPECTRAL_FLOOR = 0.02;

uint32_t readU32LE(const uint8_t* data)
{
    return static_cast<uint32_t>(data[0]) | (static_cast<uint32_t>(data[1]) << 8U) | (static_cast<uint32_t>(data[2]) << 16U)
        | (static_cast<uint32_t>(data[3]) << 24U);
}

void writeU32LE(uint8_t* data, uint32_t value)
{
    data[0] = static_cast<uint8_t>(value & 0xFFU);
    data[1] = static_cast<uint8_t>((value >> 8U) & 0xFFU);
    data[2] = static_cast<uint8_t>((value >> 16U) & 0xFFU);
    data[3] = static_cast<uint8_t>((value >> 24U) & 0xFFU);
}

float readFloatLE(const uint8_t* data)
{
    const auto bits = readU32LE(data);
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return std::isfinite(value) ? value : 0.0F;
}

void writeFloatLE(uint8_t* data, float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(value));
    writeU32LE(data, bits);
}

std::mutex& getFftwPlannerMutex()
{
    static std::mutex mutex;
    return mutex;
}

class FftwTransform
{
public:
    explicit FftwTransform(uint64_t size) : size(size)
    {
        values = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * size));
        if (values == nullptr)
        {
            return;
        }

        std::lock_guard lock(getFftwPlannerMutex());
        forwardPlan = fftw_plan_dft_1d(static_cast<int>(size), values, values, FFTW_FORWARD, FFTW_ESTIMATE);
        inversePlan = fftw_plan_dft_1d(static_cast<int>(size), values, values, FFTW_BACKWARD, FFTW_ESTIMATE);
    }

    ~FftwTransform()
    {
        std::lock_guard lock(getFftwPlannerMutex());
        if (forwardPlan != nullptr)
        {
            fftw_destroy_plan(forwardPlan);
        }
        if (inversePlan != nullptr)
        {
            fftw_destroy_plan(inversePlan);
        }
        fftw_free(values);
    }

    FftwTransform(const FftwTransform&) = delete;
    FftwTransform& operator=(const FftwTransform&) = delete;
    FftwTransform(FftwTransform&&) = delete;
    FftwTransform& operator=(FftwTransform&&) = delete;

    [[nodiscard]] bool isValid() const { return values != nullptr && forwardPlan != nullptr && inversePlan != nullptr; }

    void clear()
    {
        for (uint64_t index = 0; index < size; ++index)
        {
            values[index][0] = 0.0;
            values[index][1] = 0.0;
        }
    }

    void setReal(uint64_t index, double real)
    {
        values[index][0] = real;
        values[index][1] = 0.0;
    }

    [[nodiscard]] double real(uint64_t index) const { return values[index][0]; }

    [[nodiscard]] double magnitude(uint64_t index) const { return std::hypot(values[index][0], values[index][1]); }

    void scale(uint64_t index, double factor)
    {
        values[index][0] *= factor;
        values[index][1] *= factor;
    }

    void forward() { fftw_execute(forwardPlan); }

    void inverse()
    {
        fftw_execute(inversePlan);
        const auto normalization = static_cast<double>(size);
        for (uint64_t index = 0; index < size; ++index)
        {
            values[index][0] /= normalization;
            values[index][1] /= normalization;
        }
    }

private:
    uint64_t size;
    fftw_complex* values = nullptr;
    fftw_plan forwardPlan = nullptr;
    fftw_plan inversePlan = nullptr;
};

uint64_t calculateFrameCount(uint64_t sampleCount)
{
    if (sampleCount == 0)
    {
        return 0;
    }
    return sampleCount <= FRAME_SIZE ? 1 : ((sampleCount - FRAME_SIZE + HOP_SIZE - 1) / HOP_SIZE) + 1;
}

void fillSpectrumForFrame(
    const std::vector<double>& samples, const std::vector<double>& window, uint64_t frameIndex, FftwTransform& transform)
{
    const auto frameStart = frameIndex * HOP_SIZE;
    transform.clear();
    for (uint64_t index = 0; index < FRAME_SIZE; ++index)
    {
        const auto sampleIndex = frameStart + index;
        if (sampleIndex < samples.size())
        {
            transform.setReal(index, samples[sampleIndex] * window[index]);
        }
    }
    transform.forward();
}

double estimateWhiteNoiseBinMagnitude(const std::vector<double>& window, double noiseRmsDbfs)
{
    const auto noiseRms = std::pow(10.0, noiseRmsDbfs / 20.0);
    double windowEnergy = 0.0;
    for (const auto coefficient : window)
    {
        windowEnergy += coefficient * coefficient;
    }
    return noiseRms * std::sqrt(windowEnergy);
}

void applySpectralDenoising(double magnitude, double noiseMagnitude, SpectralDenoisingMode mode, FftwTransform& transform, uint64_t bin)
{
    if (magnitude <= std::numeric_limits<double>::epsilon())
    {
        return;
    }

    switch (mode)
    {
        case SpectralDenoisingMode::Subtraction: {
            const auto denoisedMagnitude = std::max(magnitude - OVER_SUBTRACTION_FACTOR * noiseMagnitude, SPECTRAL_FLOOR * noiseMagnitude);
            transform.scale(bin, denoisedMagnitude / magnitude);
            break;
        }
        case SpectralDenoisingMode::HardGate:
            if (magnitude <= noiseMagnitude)
            {
                transform.scale(bin, 0.0);
            }
            break;
    }
}

void denoiseMonoSamples(std::vector<double>& samples, double noiseRmsDbfs, SpectralDenoisingMode mode)
{
    const auto sampleCount = static_cast<uint64_t>(samples.size());
    const auto frameCount = calculateFrameCount(sampleCount);
    if (frameCount == 0)
    {
        return;
    }

    std::vector<double> window(FRAME_SIZE);
    for (uint64_t index = 0; index < FRAME_SIZE; ++index)
    {
        window[index] = std::sin(std::acos(-1.0) * (static_cast<double>(index) + 0.5) / static_cast<double>(FRAME_SIZE));
    }

    const auto noiseMagnitude = estimateWhiteNoiseBinMagnitude(window, noiseRmsDbfs);
    FftwTransform transform(FRAME_SIZE);
    if (!transform.isValid())
    {
        return;
    }

    std::vector<double> output(sampleCount, 0.0);
    std::vector<double> normalization(sampleCount, 0.0);
    for (uint64_t frame = 0; frame < frameCount; ++frame)
    {
        fillSpectrumForFrame(samples, window, frame, transform);
        for (uint64_t bin = 0; bin < FRAME_SIZE; ++bin)
        {
            applySpectralDenoising(transform.magnitude(bin), noiseMagnitude, mode, transform, bin);
        }

        transform.inverse();
        const auto frameStart = frame * HOP_SIZE;
        for (uint64_t index = 0; index < FRAME_SIZE && frameStart + index < sampleCount; ++index)
        {
            const auto sampleIndex = frameStart + index;
            const auto weight = window[index];
            output[sampleIndex] += transform.real(index) * weight;
            normalization[sampleIndex] += weight * weight;
        }
    }

    for (uint64_t index = 0; index < sampleCount; ++index)
    {
        if (normalization[index] > std::numeric_limits<double>::epsilon())
        {
            samples[index] = output[index] / normalization[index];
        }
    }
}

}

uint64_t denoiseRawFloat32(
    int8_t* inputData, uint64_t inputSize, double noiseRmsDbfs, SpectralDenoisingMode mode, int8_t* outputData, uint64_t outputCapacity)
{
    if (inputData == nullptr || outputCapacity < inputSize || inputSize == 0 || inputSize % sizeof(float) != 0
        || !std::isfinite(noiseRmsDbfs) || outputData == nullptr)
    {
        return 0;
    }

    const auto* inputBytes = reinterpret_cast<const uint8_t*>(inputData); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    auto* outputBytes = reinterpret_cast<uint8_t*>(outputData); /// NOLINT(cppcoreguidelines-pro-type-reinterpret-cast)
    const auto sampleCount = inputSize / sizeof(float);
    std::vector<double> samples(sampleCount);
    for (uint64_t index = 0; index < sampleCount; ++index)
    {
        samples[index] = readFloatLE(inputBytes + index * sizeof(float));
    }

    denoiseMonoSamples(samples, noiseRmsDbfs, mode);

    for (uint64_t index = 0; index < sampleCount; ++index)
    {
        writeFloatLE(outputBytes + index * sizeof(float), static_cast<float>(samples[index]));
    }
    return inputSize;
}

}
