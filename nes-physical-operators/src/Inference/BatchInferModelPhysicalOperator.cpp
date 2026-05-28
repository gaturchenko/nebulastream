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

#include <Inference/BatchInferModelPhysicalOperator.hpp>
#include "ThreadLocalRuntimeWrapper.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <Identifiers/Identifiers.hpp>
#include <Inference/BatchInferenceOperatorHandler.hpp>
#include <Nautilus/DataTypes/VarVal.hpp>
#include <Nautilus/DataTypes/VariableSizedData.hpp>
#include <Nautilus/Interface/BufferRef/TupleBufferRef.hpp>
#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedHashMapRef.hpp>
#include <Nautilus/Interface/PagedVector/PagedVectorRef.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <Nautilus/Interface/RecordBuffer.hpp>
#include <Util/StdInt.hpp>
#include <nautilus/std/cstring.h>
#include <CompilationContext.hpp>
#include <ExecutionContext.hpp>
#include <Model.hpp>
#include <PhysicalOperator.hpp>
#include <PipelineExecutionContext.hpp>
#include <static.hpp>
#include <val.hpp>
#include <val_arith.hpp>

namespace NES
{

namespace
{

using detail::ThreadLocalRuntimeWrapper;

size_t tupleSizeFor(const std::vector<size_t>& shape, size_t tensorSize)
{
    if (shape.empty() || shape.front() == 0)
    {
        return tensorSize;
    }
    return tensorSize / shape.front();
}

void setupBatchSessions(ThreadLocalRuntimeWrapper* twl, PipelineExecutionContext* pec, size_t batchSize)
{
    twl->setup(pec->getNumberOfWorkerThreads(), batchSize);
}

int8_t* getInputBuffer(ThreadLocalRuntimeWrapper* twl, WorkerThreadId thread)
{
    /// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) std::byte* to int8_t* for nautilus pointer arithmetic
    return reinterpret_cast<int8_t*>(twl->getHandle(thread).getInputData());
}

int8_t* getOutputBuffer(ThreadLocalRuntimeWrapper* twl, WorkerThreadId thread)
{
    /// NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast) std::byte* to int8_t* for nautilus pointer arithmetic
    return reinterpret_cast<int8_t*>(twl->getHandle(thread).getOutputData());
}

void infer(ThreadLocalRuntimeWrapper* twl, WorkerThreadId thread, uint64_t numberOfTuples)
{
    twl->getHandle(thread).infer(numberOfTuples);
}

uint64_t* getDeduplicatedOutputRowIndices(ThreadLocalRuntimeWrapper* twl, WorkerThreadId thread)
{
    return twl->getDeduplicatedOutputRowIndices(thread);
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

BatchInferModelPhysicalOperator::BatchInferModelPhysicalOperator(
    CompiledModel model,
    std::shared_ptr<TupleBufferRef> bufferRef,
    std::vector<Record::RecordFieldIdentifier> projections,
    std::vector<std::string> inputFieldNames,
    std::vector<std::string> outputFieldNames,
    size_t batchSize,
    InferenceRuntimeOptions runtimeOptions,
    HashMapOptions hashMapOptions,
    bool varsizedInput,
    bool varsizedOutput,
    bool useBatchDeduplication,
    OperatorHandlerId operatorHandlerId)
    : threadLocal(std::make_shared<ThreadLocalRuntimeWrapper>(model, runtimeOptions))
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

void BatchInferModelPhysicalOperator::setup(ExecutionContext& executionCtx, CompilationContext& compilationContext) const
{
    setupChild(executionCtx, compilationContext);
    nautilus::invoke(
        setupBatchSessions,
        nautilus::val<ThreadLocalRuntimeWrapper*>(threadLocal.get()),
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

void BatchInferModelPhysicalOperator::open(ExecutionContext& ctx, RecordBuffer& recordBuffer) const
{
    /// set the execution context
    ctx.watermarkTs = recordBuffer.getWatermarkTs();
    ctx.originId = recordBuffer.getOriginId();
    ctx.currentTs = recordBuffer.getCreatingTs();
    ctx.sequenceNumber = recordBuffer.getSequenceNumber();
    ctx.chunkNumber = recordBuffer.getChunkNumber();
    ctx.lastChunk = recordBuffer.isLastChunk();

    openChild(ctx, recordBuffer);

    {
        /// extract the batch from the handler
        const auto operatorHandler = ctx.getGlobalOperatorHandler(operatorHandlerId);
        const auto emittedBatch = static_cast<nautilus::val<EmittedBatch*>>(recordBuffer.getMemArea());
        const auto batchRef = nautilus::invoke(getBatchFromEmittedBuffer, operatorHandler, emittedBatch);
        const auto batchPagedVectorMemRef = nautilus::invoke(getBatchPagedVector, batchRef);
        const PagedVectorRef batchPagedVectorRef(batchPagedVectorMemRef, bufferRef);

        const auto batchRuntime = nautilus::val<ThreadLocalRuntimeWrapper*>(threadLocal.get());
        const auto inputBuffer = nautilus::invoke(getInputBuffer, batchRuntime, ctx.workerThreadId);
        const auto numberOfRecords = batchPagedVectorRef.getNumberOfTuples();
        const auto configuredBatchSize = nautilus::val<uint64_t>(batchSize);
        const auto inputTupleSizeVal = nautilus::val<uint64_t>(inputTupleSize);
        const auto outputTupleSizeVal = nautilus::val<uint64_t>(outputTupleSize);
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
        auto deduplicatedOutputRowIndices = nautilus::invoke(+[]() { return static_cast<uint64_t*>(nullptr); });
        if (useBatchDeduplication)
        {
            deduplicatedOutputRowIndices = nautilus::invoke(getDeduplicatedOutputRowIndices, batchRuntime, ctx.workerThreadId);
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

                /// if the number of bytes to copy is less than the tuple size, pad the remaining bytes with zeros
                /// this is important, because some models may produce biased outputs if the input tensor has stale values
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

        /// copies the input data into the backend
        const auto writeBatchInputs = [&](const nautilus::val<uint64_t> currentBatchStart, const nautilus::val<uint64_t> recordsInBatch)
        {
            for (nautilus::val<uint64_t> batchOffset = 0_u64; batchOffset < recordsInBatch; batchOffset = batchOffset + 1_u64)
            {
                auto recordIndex = currentBatchStart + batchOffset;
                auto record = batchPagedVectorRef.readRecord(recordIndex, projections);
                writeInputRecord(record, batchOffset);
            }
        };

        const auto writeDeduplicatedBatchInputs
            = [&](const nautilus::val<uint64_t> currentBatchStart, const nautilus::val<uint64_t> recordsInBatch)
        {
            nautilus::invoke(clearHashMap, operatorHandler, ctx.workerThreadId);

            nautilus::val<uint64_t> uniqueRecords = 0_u64;
            for (nautilus::val<uint64_t> batchOffset = 0_u64; batchOffset < recordsInBatch; batchOffset = batchOffset + 1_u64)
            {
                auto recordIndex = currentBatchStart + batchOffset;
                auto record = batchPagedVectorRef.readRecord(recordIndex, projections);

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
                        valueRecord.write("rowOutputIndex", VarVal(0_u64));
                        ref.copyValuesToEntry(valueRecord, ctx.pipelineMemoryProvider.bufferProvider);
                    },
                    ctx.pipelineMemoryProvider.bufferProvider);

                const auto chainedEntry = static_cast<nautilus::val<ChainedHashMapEntry*>>(hashMapEntry);
                const ChainedHashMapRef::ChainedEntryRef entryRef(
                    chainedEntry, chainedHashMapPtr, hashMapOptions.fieldKeys, hashMapOptions.fieldValues);
                auto valueRecord = entryRef.getValue();
                const auto entryRowIndex = valueRecord.read("rowInputIndex").getRawValueAs<nautilus::val<uint64_t>>();
                auto outputRowIndex = valueRecord.read("rowOutputIndex").getRawValueAs<nautilus::val<uint64_t>>();

                if (entryRowIndex == batchOffset)
                {
                    outputRowIndex = uniqueRecords;
                    writeInputRecord(record, uniqueRecords);
                    valueRecord.write("rowOutputIndex", VarVal(outputRowIndex));
                    entryRef.copyValuesToEntry(valueRecord, ctx.pipelineMemoryProvider.bufferProvider);
                    uniqueRecords = uniqueRecords + 1_u64;
                }

                *(deduplicatedOutputRowIndices + batchOffset) = outputRowIndex;
            }
            return uniqueRecords;
        };

        /// copies the output data from the backend
        const auto emitBatchOutputs = [&](const nautilus::val<uint64_t> currentBatchStart, const nautilus::val<uint64_t> recordsInBatch)
        {
            const auto outputBuffer = nautilus::invoke(getOutputBuffer, batchRuntime, ctx.workerThreadId);
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

        /// triggers the processing pipeline: input -> batch inference -> output
        const auto processBatch = [&](const nautilus::val<uint64_t> currentBatchStart, const nautilus::val<uint64_t> recordsInBatch)
        {
            auto recordsToInfer = recordsInBatch;
            if (useBatchDeduplication)
            {
                recordsToInfer = writeDeduplicatedBatchInputs(currentBatchStart, recordsInBatch);
            }
            else
            {
                writeBatchInputs(currentBatchStart, recordsInBatch);
            }

            nautilus::invoke(infer, batchRuntime, ctx.workerThreadId, recordsToInfer);
            emitBatchOutputs(currentBatchStart, recordsInBatch);
        };

        nautilus::val<uint64_t> batchStart = 0_u64;
        /// general case: the batch size is equal to the configured size
        for (; batchStart + configuredBatchSize <= numberOfRecords; batchStart = batchStart + configuredBatchSize)
        {
            processBatch(batchStart, configuredBatchSize);
        }
        /// special case: the batch size is smaller than the configured size
        if (batchStart < numberOfRecords)
        {
            processBatch(batchStart, numberOfRecords - batchStart);
        }

        nautilus::invoke(markBatchProcessed, operatorHandler, emittedBatch);
        closeChild(ctx, recordBuffer);
    }
}

void BatchInferModelPhysicalOperator::terminate(ExecutionContext& executionCtx) const
{
    nautilus::invoke(garbageCollectBatches, executionCtx.getGlobalOperatorHandler(operatorHandlerId));
    terminateChild(executionCtx);
}

std::optional<PhysicalOperator> BatchInferModelPhysicalOperator::getChild() const
{
    return child;
}

void BatchInferModelPhysicalOperator::setChild(PhysicalOperator newChild)
{
    this->child = std::move(newChild);
}

}
