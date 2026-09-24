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
#include <cstdint>
#include <optional>
#include <vector>
#include <unistd.h>

#include <Configuration/WorkerConfiguration.hpp>
#include <Inference/PredictionCache/PredictionCacheUtil.hpp>
#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
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

PredictionCacheMemoryProfile estimatePredictionCacheMemory(
    const PredictionCacheType cacheType,
    const size_t entries,
    const size_t inputBytes,
    const size_t outputBytes,
    const size_t numberOfWorkerThreads)
{
    if (cacheType == PredictionCacheType::NONE || entries == 0)
    {
        return {};
    }

    /// A thread-local cache stores the entry structs, the cached records and the cached predictions in three arrays, and indexes
    /// them with a chained hash map of one bucket per entry.
    const auto entryBytes = Util::getPredictionCacheEntrySize(cacheType);
    const auto lookupIndexBytes = sizeof(ChainedHashMapEntry) + sizeof(uint64_t) + sizeof(void*);
    const auto threadLocalBytesPerEntry = entryBytes + lookupIndexBytes + inputBytes + outputBytes;
    const auto threadLocalBytes = numberOfWorkerThreads * ((entries * threadLocalBytesPerEntry) + sizeof(HitsAndMisses));

    /// The global cache holds one slot per entry, each owning its record and prediction, plus a hash index over the slots. Its
    /// per-slot bookkeeping is approximated by the two vector headers, the flags and one index entry. Every worker thread
    /// additionally owns a scratch buffer that hits are copied into.
    constexpr size_t globalSlotOverheadBytes = (2 * sizeof(std::vector<std::byte>)) + (2 * sizeof(uint64_t)) + (2 * sizeof(void*));
    const auto globalBytes = (entries * (inputBytes + outputBytes + globalSlotOverheadBytes)) + (numberOfWorkerThreads * outputBytes);

    return PredictionCacheMemoryProfile{
        .entries = entries, .bytesPerEntry = threadLocalBytesPerEntry, .threadLocalBytes = threadLocalBytes, .globalBytes = globalBytes};
}

WorkerMemoryProfile getWorkerMemoryProfile(const WorkerConfiguration& configuration)
{
    /// The same source the buffer manager checks its own allocation against.
    const auto pages = static_cast<size_t>(sysconf(_SC_PHYS_PAGES));
    const auto pageSize = static_cast<size_t>(sysconf(_SC_PAGE_SIZE));
    return WorkerMemoryProfile{
        .physicalMemoryBytes = pages * pageSize,
        .bufferPoolBytes
        = configuration.defaultQueryExecution.operatorBufferSize.getValue() * configuration.numberOfBuffersInGlobalBufferManager.getValue(),
        .numberOfWorkerThreads = configuration.queryEngine.numberOfWorkerThreads.getValue()};
}

ModelProfile profileInferenceModel(const ModelProfilingRequest& request, const WorkerConfiguration& configuration)
{
    auto compiled = compileModel(request.model);
    if (!compiled)
    {
        throw CannotLoadModel("Failed to compile model for profiling: {}", compiled.error().message);
    }

    const auto& queryExecution = configuration.defaultQueryExecution;
    const auto numberOfWorkerThreads = configuration.queryEngine.numberOfWorkerThreads.getValue();
    const auto sessionOptions = resolveSessionOptions(getInferenceRuntimeOptions(queryExecution), numberOfWorkerThreads);
    auto batchedOptions = sessionOptions;
    batchedOptions.openvinoAllowDynamicBatch = usesDynamicInferenceBatches(queryExecution);

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

    profile.memory.worker = getWorkerMemoryProfile(configuration);
    profile.memory.predictionCache = estimatePredictionCacheMemory(
        queryExecution.inferenceConfiguration.predictionCacheType.getValue(),
        queryExecution.inferenceConfiguration.numberOfEntriesPredictionCache.getValue(),
        compiled->inputSize(),
        compiled->outputSize(),
        numberOfWorkerThreads);

    if (request.assessMemoryForBatchSize.has_value())
    {
        /// Set the sessions up exactly as the deployed operator will, so that the measurement covers the activations of its batch
        /// size and the compiled model it leaves in the shared cache is the one the operator reuses.
        const auto memoryBatchSize = request.assessMemoryForBatchSize.value();
        /// Sharing changes how the backend compiles the model, so the options of both deployments are resolved separately.
        auto baseOptions = getInferenceRuntimeOptions(queryExecution);
        baseOptions.openvinoAllowDynamicBatch = memoryBatchSize > 1 && usesDynamicInferenceBatches(queryExecution);
        auto privateOptions = baseOptions;
        privateOptions.openvinoShareCompiledModel = false;
        auto sharedOptions = baseOptions;
        sharedOptions.openvinoShareCompiledModel = true;
        profile.memory.session = measureSessionMemory(
            *compiled,
            memoryBatchSize,
            resolveSessionOptions(privateOptions, numberOfWorkerThreads),
            resolveSessionOptions(sharedOptions, numberOfWorkerThreads));
        const auto& memory = profile.memory.session;
        NES_INFO(
            "Assessed inference memory on this worker for batch size {}: model payload {} B, session buffers {} B in / {} B out, {} "
            "sessions need {} B "
            "with private compiled models or {} B with a shared one; prediction cache needs {} B thread-local or {} B global; worker "
            "has {} B physical memory with {} B reserved by the buffer pool",
            memory.measuredBatchSize,
            memory.modelPayloadBytes,
            memory.inputBufferBytes,
            memory.outputBufferBytes,
            numberOfWorkerThreads,
            memory.privateDeploymentBytes(numberOfWorkerThreads),
            memory.sharedDeploymentBytes(numberOfWorkerThreads),
            profile.memory.predictionCache.threadLocalBytes,
            profile.memory.predictionCache.globalBytes,
            profile.memory.worker.physicalMemoryBytes,
            profile.memory.worker.bufferPoolBytes);
    }
    return profile;
}

}
