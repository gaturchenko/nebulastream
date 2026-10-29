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
/// Measures how much memory the runtime sessions of a model need, once with every session compiling its own copy of the model
/// (`privateOptions`) and once with the sessions sharing one compiled model (`sharedOptions`). Both option sets must be the ones
/// the deployed sessions would be set up with, because the backend compiles a shared model differently: one stream per worker
/// thread in a throughput deployment rather than a single-stream latency one. Each step sets up a session, invokes it once on a
/// synthetic input, and records the growth of the process's resident set, so the numbers include everything the backend allocates
/// behind its own API, also what it allocates only at a session's first invocation.
///
/// The backend's cache of shared compiled models references them weakly, so neither does a model compiled by an earlier session
/// distort this measurement, nor does the one compiled here outlive the sessions that measured it.
[[nodiscard]] SessionMemoryProfile measureSessionMemory(
    const CompiledModel& model,
    size_t batchSize,
    const InferenceRuntimeOptions& privateOptions,
    const InferenceRuntimeOptions& sharedOptions);

[[nodiscard]] ModelProfile profileModel(
    const CompiledModel& model,
    const std::vector<size_t>& batchSizes,
    size_t warmupIterations,
    size_t measurementIterations,
    const ProfiledInvocationOptions& options);

}
