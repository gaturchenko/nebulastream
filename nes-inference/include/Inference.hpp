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
/// compiled model, all sessions reuse one ov::CompiledModel (one weight copy). To keep the per-thread parallelism the old
/// per-session compilation gave for free, that single model must have enough streams for every worker thread AND enough total
/// threads to feed those streams (streams share one inference_num_threads budget, unlike the old N private models that each
/// grabbed their own thread). Only "auto" (0) streams are overridden so an explicit ablation value still wins; the thread budget
/// is raised to at least one per stream so the streams can actually run concurrently.
[[nodiscard]] InferenceRuntimeOptions resolveSessionOptions(InferenceRuntimeOptions options, size_t numberOfWorkerThreads);

/// Import a model from a file.
[[nodiscard]] ModelBackend defaultModelBackend();
std::expected<ImportedModel, ImportError> importModel(const std::filesystem::path& modelPath);
std::expected<ImportedModel, ImportError> importModel(const std::filesystem::path& modelPath, ModelBackend backend);

/// Compile a previously imported model so it can be executed by the runtime.
std::expected<CompiledModel, CompileError> compileModel(const ImportedModel& imported);

}
