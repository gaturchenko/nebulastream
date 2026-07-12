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

#include <Inference/CacheInferModelPhysicalOperator.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <Identifiers/Identifiers.hpp>
#include <Inference/PredictionCache/GlobalPredictionCache.hpp>
#include <Inference/PredictionCache/PredictionCache.hpp>
#include <Inference/PredictionCache/PredictionCacheEntry.hpp>
#include <Inference/PredictionCache/PredictionCacheUtil.hpp>
#include <Nautilus/DataTypes/VarVal.hpp>
#include <Nautilus/DataTypes/VariableSizedData.hpp>
#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <Nautilus/Interface/RecordBuffer.hpp>
#include <nautilus/function.hpp>
#include <nautilus/std/cstring.h>
#include <InferenceConfiguration.hpp>
#include <InferenceRuntime.hpp>
#include <val_ptr.hpp>

#include <Arena.hpp>
#include <CompilationContext.hpp>
#include <ExecutionContext.hpp>
#include <Model.hpp>
#include <OperatorState.hpp>
#include <PhysicalOperator.hpp>
#include <PipelineExecutionContext.hpp>
#include <static.hpp>
#include <val_arith.hpp>

namespace NES
{

namespace detail
{
namespace
{
constexpr uint64_t MIN_PREDICTION_CACHE_LOOKUP_INDEX_PAGE_SIZE = 4096;
constexpr uint64_t PREDICTION_CACHE_LOOKUP_INDEX_KEY_SIZE = 0;
constexpr uint64_t PREDICTION_CACHE_LOOKUP_INDEX_VALUE_SIZE = sizeof(uint64_t);

uint64_t getPredictionCacheLookupIndexPageSize()
{
    const auto mapEntrySize
        = sizeof(ChainedHashMapEntry) + PREDICTION_CACHE_LOOKUP_INDEX_KEY_SIZE + PREDICTION_CACHE_LOOKUP_INDEX_VALUE_SIZE;
    return std::max(MIN_PREDICTION_CACHE_LOOKUP_INDEX_PAGE_SIZE, mapEntrySize);
}

}

struct ThreadLocalPredictionCacheWrapper
{
    ThreadLocalPredictionCacheWrapper(
        CompiledModel model,
        InferenceRuntimeOptions options,
        PredictionCacheType cacheType,
        PredictionCacheScope cacheScope,
        size_t numberOfCacheEntries)
        : model(std::move(model)), options(options), cacheType(cacheType), cacheScope(cacheScope), numberOfCacheEntries(numberOfCacheEntries)
    {
    }

    void setup(const size_t numThreads)
    {
        wrappers.clear();
        cacheStorage.clear();
        lookupIndexes.clear();
        replacementPositions.clear();
        recordStorage.clear();
        outputStorage.clear();
        hitOutputScratch.clear();
        globalCache.reset();
        wrappers.reserve(numThreads);
        for (size_t i = 0; i < numThreads; ++i)
        {
            wrappers.emplace_back();
            wrappers.back().setup(model, 1, options);
        }

        if (cacheScope == PredictionCacheScope::GLOBAL)
        {
            globalCache = std::make_unique<GlobalPredictionCache>(cacheType, numberOfCacheEntries, model.inputSize(), model.outputSize());
            hitOutputScratch.reserve(numThreads);
            for (size_t i = 0; i < numThreads; ++i)
            {
                hitOutputScratch.emplace_back(model.outputSize());
            }
            return;
        }

        cacheStorage.reserve(numThreads);
        lookupIndexes.reserve(numThreads);
        replacementPositions.reserve(numThreads);
        recordStorage.reserve(numThreads);
        outputStorage.reserve(numThreads);
        const auto entrySize = Util::getPredictionCacheEntrySize(cacheType);
        const auto cacheMemorySize = sizeof(HitsAndMisses) + numberOfCacheEntries * entrySize;
        const auto lookupIndexPageSize = getPredictionCacheLookupIndexPageSize();
        for (size_t i = 0; i < numThreads; ++i)
        {
            cacheStorage.emplace_back(cacheMemorySize);
            std::ranges::fill(cacheStorage.back(), std::byte{0});
            recordStorage.emplace_back(numberOfCacheEntries * model.inputSize());
            outputStorage.emplace_back(numberOfCacheEntries * model.outputSize());
            lookupIndexes.emplace_back(std::make_unique<ChainedHashMap>(
                PREDICTION_CACHE_LOOKUP_INDEX_KEY_SIZE,
                PREDICTION_CACHE_LOOKUP_INDEX_VALUE_SIZE,
                numberOfCacheEntries,
                lookupIndexPageSize));
            replacementPositions.emplace_back(0);
        }
    }

    [[nodiscard]] InferenceRuntime& getHandle(const WorkerThreadId thread) { return wrappers[thread.getRawValue() % wrappers.size()]; }

    [[nodiscard]] int8_t* getCacheStart(const WorkerThreadId thread)
    {
        return reinterpret_cast<int8_t*>(cacheStorage[thread.getRawValue() % cacheStorage.size()].data());
    }

    [[nodiscard]] ChainedHashMap* getLookupIndex(const WorkerThreadId thread)
    {
        return lookupIndexes[thread.getRawValue() % lookupIndexes.size()].get();
    }

    [[nodiscard]] uint64_t getReplacementPosition(const WorkerThreadId thread) const
    {
        return replacementPositions[thread.getRawValue() % replacementPositions.size()];
    }

    void setReplacementPosition(const WorkerThreadId thread, const uint64_t replacementPosition)
    {
        replacementPositions[thread.getRawValue() % replacementPositions.size()] = replacementPosition;
    }

    [[nodiscard]] std::byte* replaceEntry(
        const WorkerThreadId thread,
        PredictionCacheEntry* entry,
        const uint64_t replacementIndex,
        std::byte* inputRecord,
        const size_t inputRecordSize)
    {
        const auto threadIndex = thread.getRawValue() % wrappers.size();
        auto& runtime = getHandle(thread);
        runtime.infer(inputRecord, inputRecordSize);

        entry->recordSize = inputRecordSize;
        entry->record = recordStorage[threadIndex].data() + replacementIndex * entry->recordSize;
        std::memcpy(entry->record, inputRecord, entry->recordSize);

        entry->dataSize = runtime.getOutputSize();
        entry->dataStructure = outputStorage[threadIndex].data() + replacementIndex * entry->dataSize;
        std::memcpy(entry->dataStructure, runtime.getOutputData(), entry->dataSize);
        return entry->dataStructure;
    }

    /// Global-cache path: probe the shared cache and run inference on this thread's
    /// runtime on a miss. Inference happens outside the cache lock, so misses on
    /// different threads still run in parallel.
    [[nodiscard]] std::byte* lookupOrInfer(const WorkerThreadId thread, std::byte* inputRecord, const size_t inputRecordSize)
    {
        auto* const hitOutput = hitOutputScratch[thread.getRawValue() % hitOutputScratch.size()].data();
        if (globalCache->lookup(inputRecord, hitOutput))
        {
            return hitOutput;
        }

        auto& runtime = getHandle(thread);
        runtime.infer(inputRecord, inputRecordSize);
        globalCache->insert(inputRecord, runtime.getOutputData());
        return runtime.getOutputData();
    }

    [[nodiscard]] HitsAndMisses getAggregatedHitsAndMisses() const
    {
        if (globalCache)
        {
            return globalCache->getHitsAndMisses();
        }

        HitsAndMisses total{.hits = 0, .misses = 0};

        for (const auto& storage : cacheStorage)
        {
            const auto* stats = reinterpret_cast<const HitsAndMisses*>(storage.data());
            total.hits += stats->hits;
            total.misses += stats->misses;
        }

        return total;
    }

    CompiledModel model;
    InferenceRuntimeOptions options;
    PredictionCacheType cacheType;
    PredictionCacheScope cacheScope;
    size_t numberOfCacheEntries;
    std::vector<InferenceRuntime> wrappers;
    std::vector<std::vector<std::byte>> cacheStorage;
    std::vector<std::vector<std::byte>> recordStorage;
    std::vector<std::vector<std::byte>> outputStorage;
    std::vector<std::unique_ptr<ChainedHashMap>> lookupIndexes;
    std::vector<uint64_t> replacementPositions;
    std::unique_ptr<GlobalPredictionCache> globalCache;
    std::vector<std::vector<std::byte>> hitOutputScratch;
};

}

namespace
{

using detail::ThreadLocalPredictionCacheWrapper;

class CachedInferenceLocalState final : public OperatorState
{
public:
    explicit CachedInferenceLocalState(std::unique_ptr<PredictionCache> predictionCache) : predictionCache(std::move(predictionCache)) { }

    [[nodiscard]] PredictionCache* getPredictionCache() const { return predictionCache.get(); }

private:
    std::unique_ptr<PredictionCache> predictionCache;
};

void setupCachedSessions(ThreadLocalPredictionCacheWrapper* twl, PipelineExecutionContext* pec)
{
    twl->setup(pec->getNumberOfWorkerThreads());
}

std::byte* getInputBuffer(ThreadLocalPredictionCacheWrapper* twl, WorkerThreadId thread)
{
    return twl->getHandle(thread).getInputData();
}

int8_t* getPredictionCacheStart(ThreadLocalPredictionCacheWrapper* twl, WorkerThreadId thread)
{
    return twl->getCacheStart(thread);
}

ChainedHashMap* getPredictionCacheLookupIndex(ThreadLocalPredictionCacheWrapper* twl, WorkerThreadId thread)
{
    return twl->getLookupIndex(thread);
}

uint64_t getReplacementPosition(ThreadLocalPredictionCacheWrapper* twl, WorkerThreadId thread)
{
    return twl->getReplacementPosition(thread);
}

void setReplacementPosition(ThreadLocalPredictionCacheWrapper* twl, WorkerThreadId thread, uint64_t replacementPosition)
{
    twl->setReplacementPosition(thread, replacementPosition);
}

std::byte* replacePredictionCacheEntry(
    ThreadLocalPredictionCacheWrapper* twl,
    WorkerThreadId thread,
    PredictionCacheEntry* entry,
    uint64_t replacementIndex,
    std::byte* inputRecord,
    uint64_t inputRecordSize)
{
    return twl->replaceEntry(thread, entry, replacementIndex, inputRecord, inputRecordSize);
}

std::byte* globalPredictionCacheLookupOrInfer(
    ThreadLocalPredictionCacheWrapper* twl, WorkerThreadId thread, std::byte* inputRecord, uint64_t inputRecordSize)
{
    return twl->lookupOrInfer(thread, inputRecord, inputRecordSize);
}

void logCacheHitsAndMisses(ThreadLocalPredictionCacheWrapper* twl)
{
    const auto stats = twl->getAggregatedHitsAndMisses();
    NES_INFO("CacheInferModelPhysicalOperator cache hits={}, misses={}", stats.hits, stats.misses);
}

}

CacheInferModelPhysicalOperator::CacheInferModelPhysicalOperator(
    CompiledModel model,
    std::vector<std::string> inputFieldNames,
    std::vector<std::string> outputFieldNames,
    InferenceRuntimeOptions runtimeOptions,
    PredictionCacheType predictionCacheType,
    PredictionCacheScope predictionCacheScope,
    size_t numberOfCacheEntries,
    bool varsizedInput,
    bool varsizedOutput)
    : threadLocal(std::make_shared<ThreadLocalPredictionCacheWrapper>(
          model, runtimeOptions, predictionCacheType, predictionCacheScope, numberOfCacheEntries))
    , inputFieldNames(std::move(inputFieldNames))
    , outputFieldNames(std::move(outputFieldNames))
    , inputSize(model.inputSize())
    , outputSize(model.outputSize())
    , varsizedInput(varsizedInput)
    , varsizedOutput(varsizedOutput)
{
}

void CacheInferModelPhysicalOperator::setup(ExecutionContext& executionCtx, CompilationContext& compilationContext) const
{
    setupChild(executionCtx, compilationContext);
    nautilus::invoke(
        setupCachedSessions, nautilus::val<ThreadLocalPredictionCacheWrapper*>(threadLocal.get()), executionCtx.pipelineContext);
}

void CacheInferModelPhysicalOperator::open(ExecutionContext& ctx, RecordBuffer& recordBuffer) const
{
    if (threadLocal->cacheScope == PredictionCacheScope::GLOBAL)
    {
        /// The global cache is a shared runtime object; there is no per-task traced
        /// cache state to set up or restore.
        openChild(ctx, recordBuffer);
        return;
    }

    const auto runtime = nautilus::val<ThreadLocalPredictionCacheWrapper*>(threadLocal.get());
    const auto startOfEntries = nautilus::invoke(getPredictionCacheStart, runtime, ctx.workerThreadId);
    auto predictionCache = Util::createPredictionCache(
        threadLocal->cacheType, threadLocal->numberOfCacheEntries, startOfEntries, nautilus::val<size_t>(inputSize));
    predictionCache->configureLookupIndex(
        nautilus::invoke(getPredictionCacheLookupIndex, runtime, ctx.workerThreadId), ctx.pipelineMemoryProvider.bufferProvider);
    predictionCache->setReplacementPos(nautilus::invoke(getReplacementPosition, runtime, ctx.workerThreadId));
    ctx.setLocalOperatorState(id, std::make_unique<CachedInferenceLocalState>(std::move(predictionCache)));
    openChild(ctx, recordBuffer);
}

void CacheInferModelPhysicalOperator::execute(ExecutionContext& ctx, Record& record) const
{
    const auto runtime = nautilus::val<ThreadLocalPredictionCacheWrapper*>(threadLocal.get());
    const auto inputBuffer = nautilus::invoke(getInputBuffer, runtime, ctx.workerThreadId);
    auto cacheInputBuffer = inputBuffer;

    if (varsizedInput)
    {
        const auto& value = record.read(inputFieldNames.at(0));
        auto varSized = value.getRawValueAs<VariableSizedData>();
        const auto inputSizeVal = nautilus::val<uint64_t>(inputSize);
        if (varSized.getSize() == inputSizeVal)
        {
            cacheInputBuffer = static_cast<nautilus::val<std::byte*>>(varSized.getContent());
        }
        else
        {
            const auto bytesToCopy = varSized.getSize() < inputSizeVal ? varSized.getSize() : inputSizeVal;
            nautilus::memcpy(inputBuffer, varSized.getContent(), bytesToCopy);
            if (bytesToCopy < inputSizeVal)
            {
                nautilus::memset(inputBuffer + bytesToCopy, 0, inputSizeVal - bytesToCopy);
            }
        }
    }
    else
    {
        for (nautilus::static_val<size_t> i = 0; i < inputFieldNames.size(); ++i)
        {
            const auto value = record.read(inputFieldNames.at(nautilus::static_val<int>(i)));
            const auto memPos = inputBuffer + nautilus::val<uint64_t>(i * sizeof(float));
            value.writeToMemory(memPos);
        }
    }

    const auto outputBuffer = [&]() -> nautilus::val<std::byte*>
    {
        if (threadLocal->cacheScope == PredictionCacheScope::GLOBAL)
        {
            return nautilus::invoke(
                globalPredictionCacheLookupOrInfer, runtime, ctx.workerThreadId, cacheInputBuffer, nautilus::val<uint64_t>(inputSize));
        }

        auto* localState = dynamic_cast<CachedInferenceLocalState*>(ctx.getLocalState(id));
        auto* predictionCache = localState->getPredictionCache();
        return predictionCache->getDataStructureRef(
            cacheInputBuffer,
            [&](const nautilus::val<PredictionCacheEntry*>& predictionCacheEntryToReplace, const nautilus::val<uint64_t>& replacementIndex)
            {
                return nautilus::invoke(
                    replacePredictionCacheEntry,
                    runtime,
                    ctx.workerThreadId,
                    predictionCacheEntryToReplace,
                    replacementIndex,
                    cacheInputBuffer,
                    nautilus::val<uint64_t>(inputSize));
            });
    }();

    if (varsizedOutput)
    {
        const auto outputSizeBytes = nautilus::val<uint64_t>(outputSize);
        auto output = ctx.pipelineMemoryProvider.arena.allocateVariableSizedData(outputSizeBytes);
        nautilus::memcpy(output.getContent(), outputBuffer, outputSizeBytes);
        record.write(outputFieldNames.at(0), VarVal(output));
    }
    else
    {
        const DataType floatType{DataType::Type::FLOAT32, DataType::NULLABLE::NOT_NULLABLE};
        for (nautilus::static_val<size_t> i = 0; i < outputFieldNames.size(); ++i)
        {
            const auto memPos = outputBuffer + nautilus::val<uint64_t>(i * sizeof(float));
            const auto result = VarVal::readNonNullableVarValFromMemory(memPos, floatType);
            record.write(outputFieldNames.at(i), result);
        }
    }

    executeChild(ctx, record);
}

void CacheInferModelPhysicalOperator::close(ExecutionContext& ctx, RecordBuffer& recordBuffer) const
{
    if (threadLocal->cacheScope == PredictionCacheScope::GLOBAL)
    {
        closeChild(ctx, recordBuffer);
        return;
    }

    auto* localState = dynamic_cast<CachedInferenceLocalState*>(ctx.getLocalState(id));
    auto* predictionCache = localState->getPredictionCache();
    nautilus::invoke(
        setReplacementPosition,
        nautilus::val<ThreadLocalPredictionCacheWrapper*>(threadLocal.get()),
        ctx.workerThreadId,
        predictionCache->getReplacementPos());
    closeChild(ctx, recordBuffer);
}

void CacheInferModelPhysicalOperator::terminate(ExecutionContext& executionCtx) const
{
    nautilus::invoke(logCacheHitsAndMisses, nautilus::val<ThreadLocalPredictionCacheWrapper*>(threadLocal.get()));
    terminateChild(executionCtx);
}

std::optional<PhysicalOperator> CacheInferModelPhysicalOperator::getChild() const
{
    return child;
}

void CacheInferModelPhysicalOperator::setChild(PhysicalOperator newChild)
{
    this->child = std::move(newChild);
}

}
