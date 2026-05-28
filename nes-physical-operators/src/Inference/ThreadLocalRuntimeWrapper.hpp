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

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <Identifiers/Identifiers.hpp>
#include <InferenceRuntime.hpp>
#include <Model.hpp>

namespace NES::detail
{

/// Per-operator pool of inference runtimes, one per worker thread. Owns the
/// compiled model and creates thread-local runtime sessions at setup.
struct ThreadLocalRuntimeWrapper
{
    explicit ThreadLocalRuntimeWrapper(CompiledModel model, InferenceRuntimeOptions options) : model(std::move(model)), options(options) { }

    void setup(size_t numThreads, size_t batchSize = 1)
    {
        wrappers.clear();
        deduplicatedOutputRowIndices.clear();
        wrappers.reserve(numThreads);
        deduplicatedOutputRowIndices.reserve(numThreads);
        for (size_t i = 0; i < numThreads; ++i)
        {
            wrappers.emplace_back();
            wrappers.back().setup(model, batchSize, options);
            deduplicatedOutputRowIndices.emplace_back(batchSize);
        }
    }

    [[nodiscard]] InferenceRuntime& getHandle(WorkerThreadId thread) { return wrappers[thread.getRawValue() % wrappers.size()]; }

    [[nodiscard]] uint64_t* getDeduplicatedOutputRowIndices(WorkerThreadId thread)
    {
        return deduplicatedOutputRowIndices[thread.getRawValue() % deduplicatedOutputRowIndices.size()].data();
    }

    CompiledModel model;
    InferenceRuntimeOptions options;
    std::vector<InferenceRuntime> wrappers;
    std::vector<std::vector<uint64_t>> deduplicatedOutputRowIndices;
};

}
