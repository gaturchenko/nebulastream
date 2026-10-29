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

#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <optional>
#include <vector>

#include <Model.hpp>

namespace NES
{

/// A profiling job the optimizer submits to a worker: invoke `model` with each of the `batchSizes` on the worker's inference
/// backend and report the observed per-invocation latency. The worker applies its own inference runtime configuration, so the
/// profile reflects the backend, thread and stream settings the deployed operator will run with.
struct ModelProfilingRequest
{
    ImportedModel model;
    std::vector<size_t> batchSizes;
    size_t warmupIterations = 0;
    size_t measurementIterations = 1;
    /// The batch size to assess the model's memory for, if the optimizer asks for an assessment. Requires additional runtime
    /// sessions to be set up, so the optimizer asks for it once per model and worker, after it has decided how the model will be
    /// invoked: the sessions are then set up like the deployed ones, and the measurement covers the activations of its batch.
    std::optional<size_t> assessMemoryForBatchSize;
};

/// Observed latency of a single model invocation with `batchSize` tuples.
struct BatchProfile
{
    size_t batchSize = 0;
    size_t iterations = 0;
    std::chrono::nanoseconds medianLatency{0};
    std::chrono::nanoseconds meanLatency{0};
    std::chrono::nanoseconds minLatency{0};

    bool operator==(const BatchProfile&) const = default;
};

/// Memory one inference runtime session needs, measured as the resident-set growth around setting the session up and invoking it
/// once. The growth covers everything the backend allocates for it: the compiled model, its activations (some backends allocate
/// them only at the first invocation) and the session's input and output buffers. Sharing the compiled model replaces the per-thread weight copies of the private sessions with one copy, which is why
/// the marginal cost of an additional session is reported for both deployments.
struct SessionMemoryProfile
{
    size_t measuredBatchSize = 0;
    size_t modelPayloadBytes = 0;
    size_t inputBufferBytes = 0;
    size_t outputBufferBytes = 0;
    size_t firstPrivateSessionBytes = 0;
    size_t additionalPrivateSessionBytes = 0;
    size_t firstSharedSessionBytes = 0;
    size_t additionalSharedSessionBytes = 0;
    bool measured = false;

    /// Memory `numberOfWorkerThreads` sessions need when every session compiles its own copy of the model.
    [[nodiscard]] size_t privateDeploymentBytes(const size_t numberOfWorkerThreads) const
    {
        return numberOfWorkerThreads == 0 ? 0 : firstPrivateSessionBytes + ((numberOfWorkerThreads - 1) * additionalPrivateSessionBytes);
    }

    /// Memory `numberOfWorkerThreads` sessions need when they share one compiled model.
    [[nodiscard]] size_t sharedDeploymentBytes(const size_t numberOfWorkerThreads) const
    {
        return numberOfWorkerThreads == 0 ? 0 : firstSharedSessionBytes + ((numberOfWorkerThreads - 1) * additionalSharedSessionBytes);
    }

    bool operator==(const SessionMemoryProfile&) const = default;
};

/// Memory the prediction cache of one inference operator needs, derived from the cache configuration and the model's input and
/// output sizes. A thread-local cache is replicated per worker thread, a global cache exists once and adds a per-thread output
/// scratch buffer.
struct PredictionCacheMemoryProfile
{
    size_t entries = 0;
    size_t bytesPerEntry = 0;
    size_t threadLocalBytes = 0;
    size_t globalBytes = 0;

    bool operator==(const PredictionCacheMemoryProfile&) const = default;
};

/// What the worker can spend on inference: its physical memory, minus what the global buffer pool already reserves.
struct WorkerMemoryProfile
{
    size_t physicalMemoryBytes = 0;
    size_t bufferPoolBytes = 0;
    size_t numberOfWorkerThreads = 0;

    [[nodiscard]] size_t availableBytes() const
    {
        return physicalMemoryBytes > bufferPoolBytes ? physicalMemoryBytes - bufferPoolBytes : 0;
    }

    bool operator==(const WorkerMemoryProfile&) const = default;
};

/// What inference costs in memory on one worker: the runtime sessions of a model, the prediction cache of an operator using that
/// model, and the memory the worker has for both.
struct InferenceMemoryProfile
{
    SessionMemoryProfile session;
    PredictionCacheMemoryProfile predictionCache;
    WorkerMemoryProfile worker;

    bool operator==(const InferenceMemoryProfile&) const = default;
};

struct ModelProfile
{
    std::vector<BatchProfile> batches;
    InferenceMemoryProfile memory;

    [[nodiscard]] std::optional<BatchProfile> forBatchSize(const size_t batchSize) const
    {
        const auto profile = std::ranges::find(batches, batchSize, &BatchProfile::batchSize);
        if (profile == batches.end())
        {
            return std::nullopt;
        }
        return *profile;
    }

    bool operator==(const ModelProfile&) const = default;
};

}
