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
#include <fstream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#include <unistd.h>
#ifdef __GLIBC__
    #include <malloc.h>
#endif

#include <Util/Logger/Logger.hpp>
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

/// Resident set size of this process. Reading it from /proc is cheap enough: it is only sampled around session setup, never on
/// the inference path.
size_t residentSetBytes()
{
    std::ifstream statm{"/proc/self/statm"};
    size_t totalPages = 0;
    size_t residentPages = 0;
    if (!(statm >> totalPages >> residentPages))
    {
        return 0;
    }
    return residentPages * static_cast<size_t>(::sysconf(_SC_PAGE_SIZE));
}

/// Returns unused heap to the operating system so that the next measurement sees the memory of the session under test rather than
/// the allocator's free lists. Without this, freeing a session would leave its memory resident and hide the next growth.
void releaseFreedMemory()
{
#ifdef __GLIBC__
    ::malloc_trim(0);
#endif
}

size_t residentSetGrowth(const size_t before)
{
    const auto after = residentSetBytes();
    return after > before ? after - before : 0;
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

SessionMemoryProfile measureSessionMemory(
    const CompiledModel& model,
    const size_t batchSize,
    const InferenceRuntimeOptions& privateOptions,
    const InferenceRuntimeOptions& sharedOptions)
{
    PRECONDITION(batchSize > 0, "Cannot measure a model invocation with an empty batch");
    SessionMemoryProfile memory{
        .measuredBatchSize = batchSize, .modelPayloadBytes = model.getData().size() + model.getAuxiliaryData().size(), .measured = true};

    /// The very first session of a process also loads the backend's plugins and warms its allocators, which would otherwise be
    /// attributed to the model. A throwaway session absorbs that one-time cost so the measurements below are per session.
    {
        InferenceRuntime warmup;
        warmup.setup(model, batchSize, privateOptions);
    }

    const auto measureDeployment
        = [&model, batchSize](InferenceRuntimeOptions deploymentOptions, size_t& firstSession, size_t& additionalSession)
    {
        releaseFreedMemory();
        auto beforeFirst = residentSetBytes();
        InferenceRuntime first;
        first.setup(model, batchSize, deploymentOptions);
        firstSession = residentSetGrowth(beforeFirst);

        auto beforeSecond = residentSetBytes();
        InferenceRuntime second;
        second.setup(model, batchSize, deploymentOptions);
        additionalSession = residentSetGrowth(beforeSecond);

        return std::pair{first.getInputSize(), first.getOutputSize()};
    };

    releaseFreedMemory();
    const auto beforeAssessment = residentSetBytes();
    const auto [inputBufferBytes, outputBufferBytes]
        = measureDeployment(privateOptions, memory.firstPrivateSessionBytes, memory.additionalPrivateSessionBytes);
    memory.inputBufferBytes = inputBufferBytes;
    memory.outputBufferBytes = outputBufferBytes;

    measureDeployment(sharedOptions, memory.firstSharedSessionBytes, memory.additionalSharedSessionBytes);

    /// Everything the assessment allocated is released once its sessions are gone. What stays resident is what the backend keeps
    /// behind its own API, notably a compiled model it holds in a process-wide cache.
    releaseFreedMemory();
    NES_INFO(
        "Memory assessment at batch size {} left {} B resident after its sessions were destroyed",
        batchSize,
        residentSetGrowth(beforeAssessment));

    return memory;
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
