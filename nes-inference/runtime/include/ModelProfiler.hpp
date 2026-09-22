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
#include <vector>

#include <Inference.hpp>
#include <Model.hpp>
#include <ModelProfile.hpp>

namespace NES
{

/// Runtime options a profiled invocation with `batchSize` tuples is set up with.
struct ProfiledInvocationOptions
{
    InferenceRuntimeOptions singleTupleOptions;
    InferenceRuntimeOptions batchedOptions;
};

/// Measures the latency of single model invocations: for every requested batch size, a fresh runtime session is set up exactly
/// as a worker thread would set it up, warmed up and then invoked `measurementIterations` times on a deterministic synthetic
/// input. Only the backend invocation is timed, setup and input preparation are not.
[[nodiscard]] ModelProfile profileModel(
    const CompiledModel& model,
    const std::vector<size_t>& batchSizes,
    size_t warmupIterations,
    size_t measurementIterations,
    const ProfiledInvocationOptions& options);

}
