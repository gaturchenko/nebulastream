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

#include <Inference.hpp>
#include <ModelProfile.hpp>
#include <QueryExecutionConfiguration.hpp>

namespace NES
{

/// Runtime options of inference operators lowered under `configuration`.
[[nodiscard]] InferenceRuntimeOptions getInferenceRuntimeOptions(const QueryExecutionConfiguration& configuration);

/// Whether batched inference operators lowered under `configuration` let the backend run partially filled batches.
[[nodiscard]] bool usesDynamicInferenceBatches(const QueryExecutionConfiguration& configuration);

/// Executes a profiling job on this worker: compiles the model and measures its invocation latency per requested batch size with
/// the runtime options the lowered inference operators would use on a worker with `numberOfWorkerThreads` threads.
[[nodiscard]] ModelProfile
profileInferenceModel(const ModelProfilingRequest& request, const QueryExecutionConfiguration& configuration, size_t numberOfWorkerThreads);

}
