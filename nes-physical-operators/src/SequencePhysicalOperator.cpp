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

#include <SequencePhysicalOperator.hpp>

#include <cstdint>
#include <utility>
#include <MemoryLayout/MemoryLayout.hpp>
#include <Nautilus/Interface/RecordBuffer.hpp>
#include <Nautilus/Interface/PagedVector/PagedVectorRef.hpp>
#include <Runtime/Execution/OperatorHandler.hpp>
#include <Runtime/QueryTerminationType.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <ExecutionContext.hpp>
#include <PipelineExecutionContext.hpp>
#include <SequenceOperatorHandler.hpp>
#include <function.hpp>

namespace NES
{

void
SequencePhysicalOperator::sequentialProcessing(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const
{
    auto buffer = nautilus::invoke(
        +[](OperatorHandler* handler, TupleBuffer* tupleBuffer) -> TupleBuffer*
        {
            return dynamic_cast<SequenceOperatorHandler*>(handler)->getNextBuffer(tupleBuffer).value_or(nullptr);
        }, executionCtx.getGlobalOperatorHandler(operatorHandlerIndex), recordBuffer.getReference());

    while (buffer)
    {
        RecordBuffer nextBufferInSequence{buffer};

        scan.open(executionCtx, nextBufferInSequence);
        scan.close(executionCtx, nextBufferInSequence);

        buffer = nautilus::invoke(
            +[](OperatorHandler* handler, TupleBuffer* tupleBuffer) -> TupleBuffer*
            {
                return dynamic_cast<SequenceOperatorHandler*>(handler)->markBufferAsDone(tupleBuffer).value_or(nullptr);
            },
            executionCtx.getGlobalOperatorHandler(operatorHandlerIndex),
            buffer);
    }
}

void SequencePhysicalOperator::sequentialBatchProcessing(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const
{
    /// As this operator functions as a scan, we have to set the execution context for this pipeline
    executionCtx.watermarkTs = recordBuffer.getWatermarkTs();
    executionCtx.sequenceNumber = recordBuffer.getSequenceNumber();
    executionCtx.chunkNumber = recordBuffer.getChunkNumber();
    executionCtx.lastChunk = recordBuffer.isLastChunk();
    executionCtx.originId = recordBuffer.getOriginId();
    openChild(executionCtx, recordBuffer);

    const auto emittedBatch = static_cast<nautilus::val<EmittedBatch*>>(recordBuffer.getMemArea());
    const auto operatorHandlerMemRef = executionCtx.getGlobalOperatorHandler(operatorHandlerId);

    const auto batchMemRef = nautilus::invoke(
        +[](OperatorHandler* ptrOpHandler, const EmittedBatch* currentBatch)
        {
            PRECONDITION(ptrOpHandler != nullptr, "opHandler context should not be null!");
            const auto* opHandler = dynamic_cast<SequenceOperatorHandler*>(ptrOpHandler);
            std::shared_ptr<Batch> batch = opHandler->getBatch(currentBatch->batchId);
            return batch.get();
        }, operatorHandlerMemRef, emittedBatch);

    const auto batchPagedVectorMemRef = nautilus::invoke(
        +[](const Batch* batch)
        {
            PRECONDITION(batch != nullptr, "batch context should not be null!");
            return batch->getPagedVectorRef();
        }, batchMemRef);

    const PagedVectorRef batchPagedVectorRef(batchPagedVectorMemRef, tupleBufferRef);

    const auto resultBufferRef = executionCtx.allocateBuffer();
    auto resultBuffer = RecordBuffer(resultBufferRef);

    const auto fields = tupleBufferRef->getMemoryLayout()->getSchema().getFieldNames();
    nautilus::val<uint64_t> rowIdx(0);

    for (auto it = batchPagedVectorRef.begin(fields); it != batchPagedVectorRef.end(fields); ++it)
    {
        auto record = createRecord(*it, fields);
        tupleBufferRef->writeRecord(rowIdx, resultBuffer, record, executionCtx.pipelineMemoryProvider.bufferProvider);
        ++rowIdx;
    }

    resultBuffer.setWatermarkTs(executionCtx.watermarkTs);
    resultBuffer.setSequenceNumber(executionCtx.sequenceNumber);
    resultBuffer.setChunkNumber(executionCtx.chunkNumber);
    resultBuffer.setLastChunk(executionCtx.lastChunk);
    resultBuffer.setOriginId(executionCtx.originId);
    resultBuffer.setNumRecords(batchPagedVectorRef.getNumberOfTuples());

    sequentialProcessing(executionCtx, resultBuffer);
}

void SequencePhysicalOperator::open(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const
{
    if (batchProcessing)
    {
        sequentialBatchProcessing(executionCtx, recordBuffer);
    }

    sequentialProcessing(executionCtx, recordBuffer);
}

void SequencePhysicalOperator::setup(ExecutionContext& executionCtx, CompilationContext& compilationCtx) const
{
    nautilus::invoke(
        +[](OperatorHandler* handler, PipelineExecutionContext* ctx) { handler->start(*ctx, 0); },
        executionCtx.getGlobalOperatorHandler(operatorHandlerIndex),
        executionCtx.pipelineContext);
    scan.setup(executionCtx, compilationCtx);
}

void SequencePhysicalOperator::terminate(ExecutionContext& executionCtx) const
{
    scan.terminate(executionCtx);
    nautilus::invoke(
        +[](OperatorHandler* handler, PipelineExecutionContext* ctx) { handler->stop(QueryTerminationType::Graceful, *ctx); },
        executionCtx.getGlobalOperatorHandler(operatorHandlerIndex),
        executionCtx.pipelineContext);
}

void SequencePhysicalOperator::setChild(PhysicalOperator child)
{
    scan.setChild(child);
}

std::optional<struct PhysicalOperator> SequencePhysicalOperator::getChild() const
{
    return scan.getChild();
}

Record
SequencePhysicalOperator::createRecord(const Record& featureRecord, const std::vector<Record::RecordFieldIdentifier>& projections) const
{
    Record record;
    for (const auto& fieldName : nautilus::static_iterable(projections))
    {
        record.write(fieldName, featureRecord.read(fieldName));
    }
    return record;
}
}
