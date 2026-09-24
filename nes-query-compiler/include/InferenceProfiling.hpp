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

#include <Configuration/WorkerConfiguration.hpp>
#include <Inference.hpp>
#include <InferenceConfiguration.hpp>
#include <ModelProfile.hpp>
#include <QueryExecutionConfiguration.hpp>

namespace NES
{

/// Runtime options of inference operators lowered under `configuration`.
[[nodiscard]] InferenceRuntimeOptions getInferenceRuntimeOptions(const QueryExecutionConfiguration& configuration);

/// Whether batched inference operators lowered under `configuration` let the backend run partially filled batches.
[[nodiscard]] bool usesDynamicInferenceBatches(const QueryExecutionConfiguration& configuration);

/// Memory the prediction cache of one inference operator needs, for both scopes. The record and prediction bytes of the entries
/// are exact; the bookkeeping per entry is approximated, which is negligible for the models NES targets, where one cached record
/// is orders of magnitude larger than its metadata.
[[nodiscard]] PredictionCacheMemoryProfile estimatePredictionCacheMemory(
    PredictionCacheType cacheType, size_t entries, size_t inputBytes, size_t outputBytes, size_t numberOfWorkerThreads);

/// The worker's physical memory and what its global buffer pool reserves of it.
[[nodiscard]] WorkerMemoryProfile getWorkerMemoryProfile(const WorkerConfiguration& configuration);

/// Executes a profiling job on this worker: compiles the model, measures its invocation latency per requested batch size with the
/// runtime options the lowered inference operators would use, and reports what the model and its prediction cache would cost in
/// memory on this worker.
[[nodiscard]] ModelProfile profileInferenceModel(const ModelProfilingRequest& request, const WorkerConfiguration& configuration);

}
