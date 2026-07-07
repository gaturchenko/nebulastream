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

        /// When sharing the compiled model, all sessions reuse one ov::CompiledModel
        /// (one weight copy). To keep the per-thread parallelism the old per-session
        /// compilation gave for free, that single model must have enough streams for
        /// every worker thread AND enough total threads to feed those streams (streams
        /// share one inference_num_threads budget, unlike the old N private models that
        /// each grabbed their own thread). Only override "auto" (0) streams so an
        /// explicit ablation value still wins; raise the thread budget to at least one
        /// per stream so the streams can actually run concurrently.
        auto effectiveOptions = options;
        if (effectiveOptions.openvinoShareCompiledModel && effectiveOptions.openvinoNumStreams == 0)
        {
            effectiveOptions.openvinoNumStreams = numThreads;
            effectiveOptions.openvinoInferenceNumThreads = std::max(effectiveOptions.openvinoInferenceNumThreads, numThreads);
        }

        for (size_t i = 0; i < numThreads; ++i)
        {
            wrappers.emplace_back();
            wrappers.back().setup(model, batchSize, effectiveOptions);
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
