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

struct ModelProfile
{
    std::vector<BatchProfile> batches;

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
