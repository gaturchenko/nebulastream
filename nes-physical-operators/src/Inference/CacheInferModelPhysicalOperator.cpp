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

uint64_t getPredictionCacheLookupIndexPageSize(const uint64_t keySize)
{
    const auto mapEntrySize = sizeof(ChainedHashMapEntry) + keySize + 2 * sizeof(uint64_t);
    return std::max(MIN_PREDICTION_CACHE_LOOKUP_INDEX_PAGE_SIZE, mapEntrySize);
}

}

struct ThreadLocalPredictionCacheWrapper
{
    ThreadLocalPredictionCacheWrapper(
        CompiledModel model, InferenceRuntimeOptions options, PredictionCacheType cacheType, size_t numberOfCacheEntries)
        : model(std::move(model)), options(options), cacheType(cacheType), numberOfCacheEntries(numberOfCacheEntries)
    {
    }

    void setup(const size_t numThreads)
    {
        wrappers.clear();
        cacheStorage.clear();
        lookupIndexes.clear();
        replacementPositions.clear();
        wrappers.reserve(numThreads);
        cacheStorage.reserve(numThreads);
        lookupIndexes.reserve(numThreads);
        replacementPositions.reserve(numThreads);
        const auto entrySize = Util::getPredictionCacheEntrySize(cacheType);
        const auto cacheMemorySize = sizeof(HitsAndMisses) + numberOfCacheEntries * entrySize;
        const auto lookupIndexPageSize = getPredictionCacheLookupIndexPageSize(model.inputSize());
        for (size_t i = 0; i < numThreads; ++i)
        {
            wrappers.emplace_back();
            wrappers.back().setup(model, 1, options);
            cacheStorage.emplace_back(cacheMemorySize);
            std::ranges::fill(cacheStorage.back(), std::byte{0});
            lookupIndexes.emplace_back(
                std::make_unique<ChainedHashMap>(model.inputSize(), 2 * sizeof(uint64_t), numberOfCacheEntries, lookupIndexPageSize));
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

    [[nodiscard]] std::byte* replaceEntry(const WorkerThreadId thread, PredictionCacheEntry* entry)
    {
        auto& runtime = getHandle(thread);
        runtime.infer();

        delete[] entry->record;
        delete[] entry->dataStructure;
        entry->recordSize = runtime.getInputSize();
        entry->record = new std::byte[entry->recordSize];
        std::memcpy(entry->record, runtime.getInputData(), entry->recordSize);

        entry->dataSize = runtime.getOutputSize();
        entry->dataStructure = new std::byte[entry->dataSize];
        std::memcpy(entry->dataStructure, runtime.getOutputData(), entry->dataSize);
        return entry->dataStructure;
    }

    CompiledModel model;
    InferenceRuntimeOptions options;
    PredictionCacheType cacheType;
    size_t numberOfCacheEntries;
    std::vector<InferenceRuntime> wrappers;
    std::vector<std::vector<std::byte>> cacheStorage;
    std::vector<std::unique_ptr<ChainedHashMap>> lookupIndexes;
    std::vector<uint64_t> replacementPositions;
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

std::byte* replacePredictionCacheEntry(ThreadLocalPredictionCacheWrapper* twl, WorkerThreadId thread, PredictionCacheEntry* entry)
{
    return twl->replaceEntry(thread, entry);
}

}

CacheInferModelPhysicalOperator::CacheInferModelPhysicalOperator(
    CompiledModel model,
    std::vector<std::string> inputFieldNames,
    std::vector<std::string> outputFieldNames,
    InferenceRuntimeOptions runtimeOptions,
    PredictionCacheType predictionCacheType,
    size_t numberOfCacheEntries,
    bool varsizedInput,
    bool varsizedOutput)
    : threadLocal(std::make_shared<ThreadLocalPredictionCacheWrapper>(model, runtimeOptions, predictionCacheType, numberOfCacheEntries))
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

    if (varsizedInput)
    {
        const auto& value = record.read(inputFieldNames.at(0));
        auto varSized = value.getRawValueAs<VariableSizedData>();
        const auto inputSizeVal = nautilus::val<uint64_t>(inputSize);
        const auto bytesToCopy = varSized.getSize() < inputSizeVal ? varSized.getSize() : inputSizeVal;
        nautilus::memcpy(inputBuffer, varSized.getContent(), bytesToCopy);
        if (bytesToCopy < inputSizeVal)
        {
            nautilus::memset(inputBuffer + bytesToCopy, 0, inputSizeVal - bytesToCopy);
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

    auto* localState = dynamic_cast<CachedInferenceLocalState*>(ctx.getLocalState(id));
    auto* predictionCache = localState->getPredictionCache();
    const auto outputBuffer = predictionCache->getDataStructureRef(
        inputBuffer,
        [&](const nautilus::val<PredictionCacheEntry*>& predictionCacheEntryToReplace, const nautilus::val<uint64_t>&)
        { return nautilus::invoke(replacePredictionCacheEntry, runtime, ctx.workerThreadId, predictionCacheEntryToReplace); });

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
