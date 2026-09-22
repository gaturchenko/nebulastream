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

#include <InferenceProfiling.hpp>

#include <cstddef>

#include <Util/Logger/Logger.hpp>
#include <ErrorHandling.hpp>
#include <Inference.hpp>
#include <InferenceConfiguration.hpp>
#include <ModelProfile.hpp>
#include <ModelProfiler.hpp>
#include <QueryExecutionConfiguration.hpp>

namespace NES
{

InferenceRuntimeOptions getInferenceRuntimeOptions(const QueryExecutionConfiguration& configuration)
{
    const auto& inference = configuration.inferenceConfiguration;
    return {
        .openvinoInferenceNumThreads = inference.openvinoInferenceNumThreads.getValue(),
        .openvinoNumStreams = inference.openvinoNumStreams.getValue(),
        .openvinoEnableCpuPinning = inference.openvinoEnableCpuPinning.getValue(),
        .openvinoShareCompiledModel = inference.openvinoShareCompiledModel.getValue()};
}

bool usesDynamicInferenceBatches(const QueryExecutionConfiguration& configuration)
{
    const auto& inference = configuration.inferenceConfiguration;
    return inference.predictionCacheType.getValue() != PredictionCacheType::NONE || inference.useBatchDeduplication.getValue();
}

ModelProfile profileInferenceModel(
    const ModelProfilingRequest& request, const QueryExecutionConfiguration& configuration, const size_t numberOfWorkerThreads)
{
    auto compiled = compileModel(request.model);
    if (!compiled)
    {
        throw CannotLoadModel("Failed to compile model for profiling: {}", compiled.error().message);
    }

    const auto sessionOptions = resolveSessionOptions(getInferenceRuntimeOptions(configuration), numberOfWorkerThreads);
    auto batchedOptions = sessionOptions;
    batchedOptions.openvinoAllowDynamicBatch = usesDynamicInferenceBatches(configuration);

    auto profile = profileModel(
        *compiled,
        request.batchSizes,
        request.warmupIterations,
        request.measurementIterations,
        ProfiledInvocationOptions{.singleTupleOptions = sessionOptions, .batchedOptions = batchedOptions});
    for (const auto& batch : profile.batches)
    {
        NES_INFO(
            "Profiled model invocation with batch size {}: median {} ns, mean {} ns, min {} ns over {} iterations",
            batch.batchSize,
            batch.medianLatency.count(),
            batch.meanLatency.count(),
            batch.minLatency.count(),
            batch.iterations);
    }
    return profile;
}

}
