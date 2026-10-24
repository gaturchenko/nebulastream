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
#include <expected>
#include <filesystem>
#include <string>

#include <Model.hpp>

namespace NES
{

struct ImportError
{
    std::string message;
};

struct CompileError
{
    std::string message;
};

struct InferenceRuntimeOptions
{
    uint64_t openvinoInferenceNumThreads = 1;
    uint64_t openvinoNumStreams = 0;
    bool openvinoEnableCpuPinning = false;
    bool openvinoAllowDynamicBatch = false;
    /// When set, worker-thread sessions reuse a single cached ov::CompiledModel
    /// (one weight copy) and rely on OpenVINO streams for concurrency instead of
    /// compiling a private per-thread copy. See openvino_share_compiled_model.
    bool openvinoShareCompiledModel = false;
};

/// Options a worker-thread session is set up with when `numberOfWorkerThreads` sessions run the same model. When sharing the
/// compiled model, all sessions reuse one ov::CompiledModel (one weight copy). A compiled model holds one graph per stream and an
/// inference locks its graph, so that single model needs at least one stream per worker thread, or the worker threads would run
/// their inferences one after another; a larger explicit stream count still wins. inference_num_threads stays per session: the
/// streams share one thread budget, so the shared model gets inference_num_threads per stream, the threads each private model
/// would have had.
[[nodiscard]] InferenceRuntimeOptions resolveSessionOptions(InferenceRuntimeOptions options, size_t numberOfWorkerThreads);

/// Options of a session that a model UDF sets up. A UDF receives no worker configuration, so OpenVINO's stream and thread counts
/// come from NES_UDF_OPENVINO_NUM_STREAMS and NES_UDF_OPENVINO_INFERENCE_NUM_THREADS, which let a UDF baseline run the session
/// the worker's inference operators run. Either variable left unset keeps the InferenceRuntimeOptions default.
[[nodiscard]] InferenceRuntimeOptions udfSessionOptions();

/// Import a model from a file.
[[nodiscard]] ModelBackend defaultModelBackend();
std::expected<ImportedModel, ImportError> importModel(const std::filesystem::path& modelPath);
std::expected<ImportedModel, ImportError> importModel(const std::filesystem::path& modelPath, ModelBackend backend);

/// Compile a previously imported model so it can be executed by the runtime.
std::expected<CompiledModel, CompileError> compileModel(const ImportedModel& imported);

}
