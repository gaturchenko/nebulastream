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

#include <ModelProfiler.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include <ErrorHandling.hpp>
#include <Inference.hpp>
#include <InferenceRuntime.hpp>
#include <Model.hpp>
#include <ModelProfile.hpp>

namespace NES
{

namespace
{

/// Fill the input with floats in [0, 1) instead of random bytes, which could form NaNs or denormals and distort the timing.
void fillSyntheticInput(std::byte* input, const size_t inputSize)
{
    constexpr uint32_t seed = 42;
    std::mt19937 generator{seed};
    std::uniform_real_distribution<float> distribution{0.0F, 1.0F};
    const auto numberOfFloats = inputSize / sizeof(float);
    for (size_t i = 0; i < numberOfFloats; ++i)
    {
        const auto value = distribution(generator);
        std::memcpy(input + (i * sizeof(float)), &value, sizeof(float));
    }
    std::memset(input + (numberOfFloats * sizeof(float)), 0, inputSize - (numberOfFloats * sizeof(float)));
}

BatchProfile profileBatchSize(
    const CompiledModel& model,
    const size_t batchSize,
    const size_t warmupIterations,
    const size_t measurementIterations,
    const InferenceRuntimeOptions& options)
{
    InferenceRuntime runtime;
    runtime.setup(model, batchSize, options);
    fillSyntheticInput(runtime.getInputData(), runtime.getInputSize());

    for (size_t i = 0; i < warmupIterations; ++i)
    {
        runtime.infer();
    }

    std::vector<std::chrono::nanoseconds> latencies;
    latencies.reserve(measurementIterations);
    for (size_t i = 0; i < measurementIterations; ++i)
    {
        const auto start = std::chrono::steady_clock::now();
        runtime.infer();
        latencies.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start));
    }

    std::ranges::sort(latencies);
    const auto total = std::accumulate(latencies.begin(), latencies.end(), std::chrono::nanoseconds{0});
    return BatchProfile{
        .batchSize = batchSize,
        .iterations = measurementIterations,
        .medianLatency = latencies.at(latencies.size() / 2),
        .meanLatency = total / static_cast<int64_t>(latencies.size()),
        .minLatency = latencies.front()};
}

}

ModelProfile profileModel(
    const CompiledModel& model,
    const std::vector<size_t>& batchSizes,
    const size_t warmupIterations,
    const size_t measurementIterations,
    const ProfiledInvocationOptions& options)
{
    PRECONDITION(measurementIterations > 0, "Profiling requires at least one measured invocation");
    ModelProfile profile;
    profile.batches.reserve(batchSizes.size());
    for (const auto batchSize : batchSizes)
    {
        PRECONDITION(batchSize > 0, "Cannot profile a model invocation with an empty batch");
        const auto& invocationOptions = batchSize == 1 ? options.singleTupleOptions : options.batchedOptions;
        profile.batches.push_back(profileBatchSize(model, batchSize, warmupIterations, measurementIterations, invocationOptions));
    }
    return profile;
}

}
