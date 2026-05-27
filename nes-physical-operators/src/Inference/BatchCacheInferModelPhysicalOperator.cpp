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

#include <Inference/BatchCacheInferModelPhysicalOperator.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <Identifiers/Identifiers.hpp>
#include <Inference/BatchInferenceOperatorHandler.hpp>
#include <Inference/PredictionCache/PredictionCache.hpp>
#include <Inference/PredictionCache/PredictionCacheEntry.hpp>
#include <Inference/PredictionCache/PredictionCacheUtil.hpp>
#include <Nautilus/DataTypes/VarVal.hpp>
#include <Nautilus/DataTypes/VariableSizedData.hpp>
#include <Nautilus/Interface/BufferRef/TupleBufferRef.hpp>
#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedHashMapRef.hpp>
#include <Nautilus/Interface/PagedVector/PagedVectorRef.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <Nautilus/Interface/RecordBuffer.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Util/StdInt.hpp>
#include <nautilus/std/cstring.h>
#include <CompilationContext.hpp>
#include <ExecutionContext.hpp>
#include <Inference.hpp>
#include <InferenceRuntime.hpp>
#include <Model.hpp>
#include <PhysicalOperator.hpp>
#include <PipelineExecutionContext.hpp>
#include <static.hpp>
#include <val.hpp>
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

struct ThreadLocalBatchPredictionCacheWrapper
{
    ThreadLocalBatchPredictionCacheWrapper(
        CompiledModel model,
        InferenceRuntimeOptions options,
        PredictionCacheType cacheType,
        size_t numberOfCacheEntries,
        size_t inputTupleSize)
        : model(std::move(model))
        , options(options)
        , cacheType(cacheType)
        , numberOfCacheEntries(numberOfCacheEntries)
        , inputTupleSize(inputTupleSize)
    {
    }

    void setup(const size_t numThreads, const size_t batchSize)
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
        const auto lookupIndexPageSize = getPredictionCacheLookupIndexPageSize(inputTupleSize);
        for (size_t i = 0; i < numThreads; ++i)
        {
            wrappers.emplace_back();
            wrappers.back().setup(model, batchSize, options);
            cacheStorage.emplace_back(cacheMemorySize);
            std::ranges::fill(cacheStorage.back(), std::byte{0});
            lookupIndexes.emplace_back(
                std::make_unique<ChainedHashMap>(inputTupleSize, 2 * sizeof(uint64_t), numberOfCacheEntries, lookupIndexPageSize));
            replacementPositions.emplace_back(0);
        }
    }

    [[nodiscard]] InferenceRuntime& getHandle(const WorkerThreadId thread) { return wrappers[thread.getRawValue() % wrappers.size()]; }

    [[nodiscard]] bool supportsReducedBatchInvocation() const { return model.getBackend() == ModelBackend::OPENVINO; }

    void infer(const WorkerThreadId thread, const size_t numberOfTuples)
    {
        auto& runtime = getHandle(thread);
        if (supportsReducedBatchInvocation())
        {
            runtime.infer(numberOfTuples);
        }
        else
        {
            runtime.infer();
        }
    }

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

    CompiledModel model;
    InferenceRuntimeOptions options;
    PredictionCacheType cacheType;
    size_t numberOfCacheEntries;
    size_t inputTupleSize;
    std::vector<InferenceRuntime> wrappers;
    std::vector<std::vector<std::byte>> cacheStorage;
    std::vector<std::unique_ptr<ChainedHashMap>> lookupIndexes;
    std::vector<uint64_t> replacementPositions;
};

}

namespace
{

using detail::ThreadLocalBatchPredictionCacheWrapper;

size_t tupleSizeFor(const std::vector<size_t>& shape, size_t tensorSize)
{
    if (shape.empty() || shape.front() == 0)
    {
        return tensorSize;
    }
    return tensorSize / shape.front();
}

void setupBatchCacheSessions(ThreadLocalBatchPredictionCacheWrapper* twl, PipelineExecutionContext* pec, size_t batchSize)
{
    twl->setup(pec->getNumberOfWorkerThreads(), batchSize);
}

int8_t* getInputBuffer(ThreadLocalBatchPredictionCacheWrapper* twl, WorkerThreadId thread)
{
    return reinterpret_cast<int8_t*>(twl->getHandle(thread).getInputData());
}

int8_t* getOutputBuffer(ThreadLocalBatchPredictionCacheWrapper* twl, WorkerThreadId thread)
{
    return reinterpret_cast<int8_t*>(twl->getHandle(thread).getOutputData());
}

int8_t* getPredictionCacheStart(ThreadLocalBatchPredictionCacheWrapper* twl, WorkerThreadId thread)
{
    return twl->getCacheStart(thread);
}

ChainedHashMap* getPredictionCacheLookupIndex(ThreadLocalBatchPredictionCacheWrapper* twl, WorkerThreadId thread)
{
    return twl->getLookupIndex(thread);
}

uint64_t getReplacementPosition(ThreadLocalBatchPredictionCacheWrapper* twl, WorkerThreadId thread)
{
    return twl->getReplacementPosition(thread);
}

void setReplacementPosition(ThreadLocalBatchPredictionCacheWrapper* twl, WorkerThreadId thread, uint64_t replacementPosition)
{
    twl->setReplacementPosition(thread, replacementPosition);
}

void infer(ThreadLocalBatchPredictionCacheWrapper* twl, WorkerThreadId thread, uint64_t numberOfTuples)
{
    twl->infer(thread, numberOfTuples);
}

bool supportsReducedBatchInvocation(ThreadLocalBatchPredictionCacheWrapper* twl)
{
    return twl->supportsReducedBatchInvocation();
}

void allocateBatchDeduplicationHashMaps(
    OperatorHandler* ptrOpHandler,
    PipelineExecutionContext* pipelineExecutionContext,
    uint64_t keySize,
    uint64_t valueSize,
    uint64_t numberOfBuckets,
    uint64_t pageSize)
{
    PRECONDITION(ptrOpHandler != nullptr, "opHandler context should not be null!");
    PRECONDITION(pipelineExecutionContext != nullptr, "pipeline execution context should not be null!");
    auto* opHandler = dynamic_cast<BatchInferenceOperatorHandler*>(ptrOpHandler);
    PRECONDITION(opHandler != nullptr, "operator handler should be a BatchInferenceOperatorHandler");
    opHandler->allocateHashMaps(pipelineExecutionContext->getNumberOfWorkerThreads(), keySize, valueSize, numberOfBuckets, pageSize);
}

HashMap* getHashMapPtr(OperatorHandler* ptrOpHandler, WorkerThreadId thread)
{
    PRECONDITION(ptrOpHandler != nullptr, "opHandler context should not be null!");
    auto* opHandler = dynamic_cast<BatchInferenceOperatorHandler*>(ptrOpHandler);
    PRECONDITION(opHandler != nullptr, "operator handler should be a BatchInferenceOperatorHandler");
    return opHandler->getHashMapPtr(thread);
}

void clearHashMap(OperatorHandler* ptrOpHandler, WorkerThreadId thread)
{
    PRECONDITION(ptrOpHandler != nullptr, "opHandler context should not be null!");
    auto* opHandler = dynamic_cast<BatchInferenceOperatorHandler*>(ptrOpHandler);
    PRECONDITION(opHandler != nullptr, "operator handler should be a BatchInferenceOperatorHandler");
    opHandler->clearHashMap(thread);
}

uint64_t* allocateRowIndexArray(uint64_t numberOfRows)
{
    return new uint64_t[numberOfRows];
}

void freeRowIndexArray(uint64_t* rowIndices)
{
    delete[] rowIndices;
}

std::byte** allocatePredictionArray(uint64_t numberOfRows)
{
    return new std::byte*[numberOfRows];
}

std::byte* allocatePredictionCopyBuffer(uint64_t numberOfRows, size_t outputTupleSize)
{
    return new std::byte[numberOfRows * outputTupleSize];
}

void freePredictionArray(std::byte** predictions)
{
    delete[] predictions;
}

void freePredictionCopyBuffer(std::byte* predictionCopyBuffer)
{
    delete[] predictionCopyBuffer;
}

void clearPredictionArray(std::byte** predictions, uint64_t numberOfRows)
{
    std::fill_n(predictions, numberOfRows, nullptr);
}

void copyPrediction(std::byte** predictions, std::byte* predictionCopyBuffer, uint64_t row, std::byte* prediction, size_t outputTupleSize)
{
    auto* predictionCopy = predictionCopyBuffer + row * outputTupleSize;
    std::memcpy(predictionCopy, prediction, outputTupleSize);
    predictions[row] = predictionCopy;
}

std::byte* getPrediction(std::byte** predictions, uint64_t row)
{
    return predictions[row];
}

void copyOutputToCacheEntry(PredictionCacheEntry* predictionCacheEntry, std::byte* outputTupleBuffer, size_t outputTupleSize)
{
    delete[] predictionCacheEntry->dataStructure;
    predictionCacheEntry->dataSize = outputTupleSize;
    predictionCacheEntry->dataStructure = new std::byte[outputTupleSize];
    std::memcpy(predictionCacheEntry->dataStructure, outputTupleBuffer, outputTupleSize);
}

void copyRecordToCacheEntry(PredictionCacheEntry* predictionCacheEntry, std::byte* inputTupleBuffer, size_t inputTupleSize)
{
    delete[] predictionCacheEntry->record;
    delete[] predictionCacheEntry->dataStructure;
    predictionCacheEntry->recordSize = inputTupleSize;
    predictionCacheEntry->record = new std::byte[inputTupleSize];
    std::memcpy(predictionCacheEntry->record, inputTupleBuffer, inputTupleSize);
    predictionCacheEntry->dataSize = 0;
    predictionCacheEntry->dataStructure = nullptr;
}

Batch* getBatchFromEmittedBuffer(OperatorHandler* ptrOpHandler, const EmittedBatch* currentBatch)
{
    PRECONDITION(ptrOpHandler != nullptr, "opHandler context should not be null!");
    PRECONDITION(currentBatch != nullptr, "emitted batch should not be null!");
    auto* opHandler = dynamic_cast<BatchInferenceOperatorHandler*>(ptrOpHandler);
    PRECONDITION(opHandler != nullptr, "operator handler should be a BatchInferenceOperatorHandler");
    auto batch = opHandler->getBatch(currentBatch->batchId);
    PRECONDITION(batch != nullptr, "emitted batch {} should exist in the batch handler", currentBatch->batchId);
    return batch.get();
}

PagedVector* getBatchPagedVector(const Batch* batch)
{
    PRECONDITION(batch != nullptr, "batch context should not be null!");
    return batch->getPagedVectorRef();
}

void markBatchProcessed(OperatorHandler* ptrOpHandler, const EmittedBatch* currentBatch)
{
    PRECONDITION(ptrOpHandler != nullptr, "opHandler context should not be null!");
    PRECONDITION(currentBatch != nullptr, "emitted batch should not be null!");
    auto* opHandler = dynamic_cast<BatchInferenceOperatorHandler*>(ptrOpHandler);
    PRECONDITION(opHandler != nullptr, "operator handler should be a BatchInferenceOperatorHandler");
    auto batch = opHandler->getBatch(currentBatch->batchId);
    PRECONDITION(batch != nullptr, "emitted batch {} should exist in the batch handler", currentBatch->batchId);
    batch->setState(BatchState::MARKED_AS_PROCESSED);
}

void garbageCollectBatches(OperatorHandler* ptrOpHandler)
{
    PRECONDITION(ptrOpHandler != nullptr, "opHandler context should not be null!");
    auto* opHandler = dynamic_cast<BatchInferenceOperatorHandler*>(ptrOpHandler);
    PRECONDITION(opHandler != nullptr, "operator handler should be a BatchInferenceOperatorHandler");
    opHandler->garbageCollectBatches();
}

}

BatchCacheInferModelPhysicalOperator::BatchCacheInferModelPhysicalOperator(
    CompiledModel model,
    std::shared_ptr<TupleBufferRef> bufferRef,
    std::vector<Record::RecordFieldIdentifier> projections,
    std::vector<std::string> inputFieldNames,
    std::vector<std::string> outputFieldNames,
    size_t batchSize,
    InferenceRuntimeOptions runtimeOptions,
    PredictionCacheType predictionCacheType,
    size_t numberOfCacheEntries,
    HashMapOptions hashMapOptions,
    bool varsizedInput,
    bool varsizedOutput,
    bool useBatchDeduplication,
    OperatorHandlerId operatorHandlerId)
    : threadLocal(std::make_shared<ThreadLocalBatchPredictionCacheWrapper>(
          model, runtimeOptions, predictionCacheType, numberOfCacheEntries, tupleSizeFor(model.getInputShape(), model.inputSize())))
    , bufferRef(std::move(bufferRef))
    , projections(std::move(projections))
    , inputFieldNames(std::move(inputFieldNames))
    , outputFieldNames(std::move(outputFieldNames))
    , batchSize(batchSize)
    , hashMapOptions(std::move(hashMapOptions))
    , inputTupleSize(tupleSizeFor(model.getInputShape(), model.inputSize()))
    , outputTupleSize(tupleSizeFor(model.getOutputShape(), model.outputSize()))
    , varsizedInput(varsizedInput)
    , varsizedOutput(varsizedOutput)
    , useBatchDeduplication(useBatchDeduplication)
    , operatorHandlerId(operatorHandlerId)
{
}

void BatchCacheInferModelPhysicalOperator::setup(ExecutionContext& executionCtx, CompilationContext& compilationContext) const
{
    setupChild(executionCtx, compilationContext);
    nautilus::invoke(
        setupBatchCacheSessions,
        nautilus::val<ThreadLocalBatchPredictionCacheWrapper*>(threadLocal.get()),
        executionCtx.pipelineContext,
        nautilus::val<size_t>(batchSize));
    if (useBatchDeduplication)
    {
        nautilus::invoke(
            allocateBatchDeduplicationHashMaps,
            executionCtx.getGlobalOperatorHandler(operatorHandlerId),
            executionCtx.pipelineContext,
            nautilus::val<uint64_t>(hashMapOptions.keySize),
            nautilus::val<uint64_t>(hashMapOptions.valueSize),
            nautilus::val<uint64_t>(hashMapOptions.numberOfBuckets),
            nautilus::val<uint64_t>(hashMapOptions.pageSize));
    }
}

void BatchCacheInferModelPhysicalOperator::open(ExecutionContext& ctx, RecordBuffer& recordBuffer) const
{
    ctx.watermarkTs = recordBuffer.getWatermarkTs();
    ctx.originId = recordBuffer.getOriginId();
    ctx.currentTs = recordBuffer.getCreatingTs();
    ctx.sequenceNumber = recordBuffer.getSequenceNumber();
    ctx.chunkNumber = recordBuffer.getChunkNumber();
    ctx.lastChunk = recordBuffer.isLastChunk();

    openChild(ctx, recordBuffer);

    {
        const auto operatorHandler = ctx.getGlobalOperatorHandler(operatorHandlerId);
        const auto emittedBatch = static_cast<nautilus::val<EmittedBatch*>>(recordBuffer.getMemArea());
        const auto batchRef = nautilus::invoke(getBatchFromEmittedBuffer, operatorHandler, emittedBatch);
        const auto batchPagedVectorMemRef = nautilus::invoke(getBatchPagedVector, batchRef);
        const PagedVectorRef batchPagedVectorRef(batchPagedVectorMemRef, bufferRef);

        const auto batchRuntime = nautilus::val<ThreadLocalBatchPredictionCacheWrapper*>(threadLocal.get());
        const auto inputBuffer = nautilus::invoke(getInputBuffer, batchRuntime, ctx.workerThreadId);
        const auto outputBuffer = nautilus::invoke(getOutputBuffer, batchRuntime, ctx.workerThreadId);
        const auto numberOfRecords = batchPagedVectorRef.getNumberOfTuples();
        const auto configuredBatchSize = nautilus::val<uint64_t>(batchSize);
        const auto inputTupleSizeVal = nautilus::val<uint64_t>(inputTupleSize);
        const auto outputTupleSizeVal = nautilus::val<uint64_t>(outputTupleSize);
        const auto inputBufferSize = nautilus::val<uint64_t>(batchSize * inputTupleSize);
        const auto reducedBatchInvocation = nautilus::invoke(supportsReducedBatchInvocation, batchRuntime);

        const auto startOfEntries = nautilus::invoke(getPredictionCacheStart, batchRuntime, ctx.workerThreadId);
        const auto lookupIndex = nautilus::invoke(getPredictionCacheLookupIndex, batchRuntime, ctx.workerThreadId);
        auto predictionCache = Util::createPredictionCache(
            threadLocal->cacheType, threadLocal->numberOfCacheEntries, startOfEntries, nautilus::val<size_t>(inputTupleSize));
        predictionCache->configureLookupIndex(lookupIndex, ctx.pipelineMemoryProvider.bufferProvider);
        predictionCache->setReplacementPos(nautilus::invoke(getReplacementPosition, batchRuntime, ctx.workerThreadId));

        auto hashMapPtr = nautilus::invoke(+[]() { return static_cast<HashMap*>(nullptr); });
        if (useBatchDeduplication)
        {
            hashMapPtr = nautilus::invoke(getHashMapPtr, operatorHandler, ctx.workerThreadId);
        }
        ChainedHashMapRef hashMap{
            hashMapPtr,
            hashMapOptions.fieldKeys,
            hashMapOptions.fieldValues,
            nautilus::val<uint64_t>(hashMapOptions.entriesPerPage),
            nautilus::val<uint64_t>(hashMapOptions.entrySize)};
        const auto chainedHashMapPtr = static_cast<nautilus::val<ChainedHashMap*>>(hashMapPtr);

        const auto missCachePositions = nautilus::invoke(allocateRowIndexArray, configuredBatchSize);
        const auto missOutputRows = nautilus::invoke(allocateRowIndexArray, configuredBatchSize);
        const auto cachedPredictions = nautilus::invoke(allocatePredictionArray, configuredBatchSize);
        const auto cachedPredictionCopies
            = nautilus::invoke(allocatePredictionCopyBuffer, configuredBatchSize, nautilus::val<size_t>(outputTupleSize));
        auto deduplicatedOutputRowIndices = nautilus::invoke(+[]() { return static_cast<uint64_t*>(nullptr); });
        if (useBatchDeduplication)
        {
            deduplicatedOutputRowIndices = nautilus::invoke(allocateRowIndexArray, configuredBatchSize);
        }

        const auto writeInputRecord = [&](const Record& record, const nautilus::val<uint64_t> inputRowIndex)
        {
            const auto inputTupleBuffer = inputBuffer + (inputRowIndex * inputTupleSizeVal);

            if (varsizedInput)
            {
                const auto& value = record.read(inputFieldNames.at(0));
                auto varSized = value.getRawValueAs<VariableSizedData>();
                const auto bytesToCopy = varSized.getSize() < inputTupleSizeVal ? varSized.getSize() : inputTupleSizeVal;
                nautilus::memcpy(inputTupleBuffer, varSized.getContent(), bytesToCopy);
                if (bytesToCopy < inputTupleSizeVal)
                {
                    nautilus::memset(inputTupleBuffer + bytesToCopy, 0, inputTupleSizeVal - bytesToCopy);
                }
            }
            else
            {
                for (nautilus::static_val<size_t> i = 0; i < inputFieldNames.size(); ++i)
                {
                    const auto value = record.read(inputFieldNames.at(nautilus::static_val<int>(i)));
                    const auto memPos = inputTupleBuffer + nautilus::val<uint64_t>(i * sizeof(float));
                    value.writeToMemory(memPos);
                }
            }
        };

        const auto updateCacheValue = [&](const nautilus::val<uint64_t> cachePos, const nautilus::val<uint64_t> outputRow)
        {
            predictionCache->updateValues(
                cachePos,
                [&](const nautilus::val<PredictionCacheEntry*>& predictionCacheEntryToUpdate, const nautilus::val<uint64_t>&)
                {
                    nautilus::invoke(
                        copyOutputToCacheEntry,
                        predictionCacheEntryToUpdate,
                        static_cast<nautilus::val<std::byte*>>(outputBuffer + (outputRow * outputTupleSizeVal)),
                        nautilus::val<size_t>(outputTupleSize));
                });
        };

        const auto processBatch = [&](const nautilus::val<uint64_t> currentBatchStart, const nautilus::val<uint64_t> recordsInBatch)
        {
            nautilus::invoke(clearPredictionArray, cachedPredictions, recordsInBatch);
            if (useBatchDeduplication)
            {
                nautilus::invoke(clearHashMap, operatorHandler, ctx.workerThreadId);
            }
            nautilus::val<uint64_t> numberOfMisses = 0_u64;

            for (nautilus::val<uint64_t> batchOffset = 0_u64; batchOffset < recordsInBatch; batchOffset = batchOffset + 1_u64)
            {
                auto recordIndex = currentBatchStart + batchOffset;
                auto record = batchPagedVectorRef.readRecord(recordIndex, projections);

                if (useBatchDeduplication)
                {
                    const auto hashMapEntry = hashMap.findOrCreateEntry(
                        record,
                        *hashMapOptions.hashFunction,
                        [&](const nautilus::val<AbstractHashMapEntry*>& entry)
                        {
                            const auto chainedEntry = static_cast<nautilus::val<ChainedHashMapEntry*>>(entry);
                            const ChainedHashMapRef::ChainedEntryRef ref(
                                chainedEntry, chainedHashMapPtr, hashMapOptions.fieldKeys, hashMapOptions.fieldValues);
                            Record valueRecord;
                            valueRecord.write("rowInputIndex", VarVal(batchOffset));
                            valueRecord.write("rowOutputIndex", VarVal(batchOffset));
                            ref.copyValuesToEntry(valueRecord, ctx.pipelineMemoryProvider.bufferProvider);
                        },
                        ctx.pipelineMemoryProvider.bufferProvider);

                    const auto chainedEntry = static_cast<nautilus::val<ChainedHashMapEntry*>>(hashMapEntry);
                    const ChainedHashMapRef::ChainedEntryRef entryRef(
                        chainedEntry, chainedHashMapPtr, hashMapOptions.fieldKeys, hashMapOptions.fieldValues);
                    auto valueRecord = entryRef.getValue();
                    const auto entryRowIndex = valueRecord.read("rowInputIndex").getRawValueAs<nautilus::val<uint64_t>>();
                    auto outputRowIndex = valueRecord.read("rowOutputIndex").getRawValueAs<nautilus::val<uint64_t>>();
                    *(deduplicatedOutputRowIndices + batchOffset) = outputRowIndex;
                    if (entryRowIndex != batchOffset)
                    {
                        continue;
                    }
                }

                writeInputRecord(record, batchOffset);

                const auto inputTupleBuffer = inputBuffer + (batchOffset * inputTupleSizeVal);
                const auto cacheProbeTuple = static_cast<nautilus::val<std::byte*>>(inputTupleBuffer);
                const auto cacheKeyIndex = predictionCache->updateKeys(
                    cacheProbeTuple,
                    [&](const nautilus::val<PredictionCacheEntry*>& predictionCacheEntryToReplace, const nautilus::val<uint64_t>&)
                    {
                        nautilus::invoke(
                            copyRecordToCacheEntry, predictionCacheEntryToReplace, cacheProbeTuple, nautilus::val<size_t>(inputTupleSize));
                    });

                auto cachePos = cacheKeyIndex;
                if (cacheKeyIndex == PredictionCache::NOT_FOUND)
                {
                    cachePos = predictionCache->getReplacementIndex();
                }

                const auto cachedPrediction = predictionCache->getDataStructure(cachePos);
                const auto hasCachedPrediction = cachedPrediction != nautilus::val<std::byte*>(nullptr);
                if (hasCachedPrediction)
                {
                    nautilus::invoke(
                        copyPrediction,
                        cachedPredictions,
                        cachedPredictionCopies,
                        batchOffset,
                        cachedPrediction,
                        nautilus::val<size_t>(outputTupleSize));
                }
                else
                {
                    if (numberOfMisses != batchOffset)
                    {
                        nautilus::memcpy(inputBuffer + (numberOfMisses * inputTupleSizeVal), inputTupleBuffer, inputTupleSizeVal);
                    }
                    *(missCachePositions + numberOfMisses) = cachePos;
                    *(missOutputRows + numberOfMisses) = batchOffset;
                    numberOfMisses = numberOfMisses + 1_u64;
                }
            }

            if (numberOfMisses > 0_u64)
            {
                if (!reducedBatchInvocation)
                {
                    const auto paddingOffset = numberOfMisses * inputTupleSizeVal;
                    const auto paddingSize = inputBufferSize - paddingOffset;
                    nautilus::memset(inputBuffer + paddingOffset, 0, paddingSize);
                }
                nautilus::invoke(infer, batchRuntime, ctx.workerThreadId, numberOfMisses);

                for (nautilus::val<uint64_t> missOffset = 0_u64; missOffset < numberOfMisses; missOffset = missOffset + 1_u64)
                {
                    updateCacheValue(*(missCachePositions + missOffset), missOffset);
                }

                for (auto missOffset = numberOfMisses; missOffset > 0_u64; missOffset = missOffset - 1_u64)
                {
                    const auto compactOutputRow = missOffset - 1_u64;
                    auto originalOutputRow = compactOutputRow;
                    originalOutputRow = *(missOutputRows + compactOutputRow);
                    if (originalOutputRow != compactOutputRow)
                    {
                        nautilus::memcpy(
                            outputBuffer + (originalOutputRow * outputTupleSizeVal),
                            outputBuffer + (compactOutputRow * outputTupleSizeVal),
                            outputTupleSizeVal);
                    }
                }
            }

            for (nautilus::val<uint64_t> batchOffset = 0_u64; batchOffset < recordsInBatch; batchOffset = batchOffset + 1_u64)
            {
                const auto cachedPrediction = nautilus::invoke(getPrediction, cachedPredictions, batchOffset);
                if (cachedPrediction != nautilus::val<std::byte*>(nullptr))
                {
                    nautilus::memcpy(
                        outputBuffer + (batchOffset * outputTupleSizeVal),
                        static_cast<nautilus::val<int8_t*>>(cachedPrediction),
                        outputTupleSizeVal);
                }
            }
        };

        const auto emitBatchOutputs = [&](const nautilus::val<uint64_t> currentBatchStart, const nautilus::val<uint64_t> recordsInBatch)
        {
            for (nautilus::val<uint64_t> batchOffset = 0_u64; batchOffset < recordsInBatch; batchOffset = batchOffset + 1_u64)
            {
                auto recordIndex = currentBatchStart + batchOffset;
                auto record = batchPagedVectorRef.readRecord(recordIndex, projections);
                auto outputRowIndex = batchOffset;
                if (useBatchDeduplication)
                {
                    outputRowIndex = *(deduplicatedOutputRowIndices + batchOffset);
                }
                const auto outputTupleBuffer = outputBuffer + (outputRowIndex * outputTupleSizeVal);

                if (varsizedOutput)
                {
                    auto output = ctx.pipelineMemoryProvider.arena.allocateVariableSizedData(outputTupleSizeVal);
                    nautilus::memcpy(output.getContent(), outputTupleBuffer, outputTupleSizeVal);
                    record.write(outputFieldNames.at(0), VarVal(output));
                }
                else
                {
                    const DataType floatType{DataType::Type::FLOAT32, DataType::NULLABLE::NOT_NULLABLE};
                    for (nautilus::static_val<size_t> i = 0; i < outputFieldNames.size(); ++i)
                    {
                        const auto memPos = outputTupleBuffer + nautilus::val<uint64_t>(i * sizeof(float));
                        const auto result = VarVal::readNonNullableVarValFromMemory(memPos, floatType);
                        record.write(outputFieldNames.at(i), result);
                    }
                }

                executeChild(ctx, record);
            }
        };

        nautilus::val<uint64_t> batchStart = 0_u64;
        for (; batchStart + configuredBatchSize <= numberOfRecords; batchStart = batchStart + configuredBatchSize)
        {
            processBatch(batchStart, configuredBatchSize);
            emitBatchOutputs(batchStart, configuredBatchSize);
        }
        if (batchStart < numberOfRecords)
        {
            const auto recordsInTailBatch = numberOfRecords - batchStart;
            processBatch(batchStart, recordsInTailBatch);
            emitBatchOutputs(batchStart, recordsInTailBatch);
        }

        nautilus::invoke(setReplacementPosition, batchRuntime, ctx.workerThreadId, predictionCache->getReplacementPos());
        if (useBatchDeduplication)
        {
            nautilus::invoke(freeRowIndexArray, deduplicatedOutputRowIndices);
        }
        nautilus::invoke(freePredictionCopyBuffer, cachedPredictionCopies);
        nautilus::invoke(freePredictionArray, cachedPredictions);
        nautilus::invoke(freeRowIndexArray, missOutputRows);
        nautilus::invoke(freeRowIndexArray, missCachePositions);
        nautilus::invoke(markBatchProcessed, operatorHandler, emittedBatch);
        closeChild(ctx, recordBuffer);
    }
}

void BatchCacheInferModelPhysicalOperator::terminate(ExecutionContext& executionCtx) const
{
    nautilus::invoke(garbageCollectBatches, executionCtx.getGlobalOperatorHandler(operatorHandlerId));
    terminateChild(executionCtx);
}

std::optional<PhysicalOperator> BatchCacheInferModelPhysicalOperator::getChild() const
{
    return child;
}

void BatchCacheInferModelPhysicalOperator::setChild(PhysicalOperator newChild)
{
    this->child = std::move(newChild);
}

}
