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

#include <BatchingPhysicalOperator.hpp>

#include <cstddef>
#include <memory>
#include <utility>

#include <BatchInferenceOperatorHandler.hpp>
#include <ErrorHandling.hpp>
#include <ExecutionContext.hpp>
#include <Nautilus/Interface/PagedVector/PagedVectorRef.hpp>
#include <OperatorState.hpp>
#include <PipelineExecutionContext.hpp>
#include <function.hpp>

namespace NES
{

namespace
{

class BatchingOperatorLocalState final : public OperatorState
{
public:
    explicit BatchingOperatorLocalState(const nautilus::val<OperatorHandler*>& operatorHandler) : operatorHandler(operatorHandler) { }

    [[nodiscard]] nautilus::val<OperatorHandler*> getOperatorHandler() const { return operatorHandler; }

private:
    nautilus::val<OperatorHandler*> operatorHandler;
};

BatchInferenceOperatorHandler* getBatchHandler(OperatorHandler* ptrOpHandler)
{
    PRECONDITION(ptrOpHandler != nullptr, "opHandler context should not be null!");
    auto* opHandler = dynamic_cast<BatchInferenceOperatorHandler*>(ptrOpHandler);
    PRECONDITION(opHandler != nullptr, "operator handler should be a BatchInferenceOperatorHandler");
    return opHandler;
}

Batch* getOrCreateBatch(OperatorHandler* ptrOpHandler)
{
    return getBatchHandler(ptrOpHandler)->getOrCreateNewBatch();
}

PagedVector* getBatchPagedVector(Batch* batch)
{
    PRECONDITION(batch != nullptr, "batch context should not be null!");
    return batch->getPagedVectorRef();
}

void emitBatchesProxy(
    OperatorHandler* ptrOpHandler,
    PipelineExecutionContext* pipelineCtx,
    const Timestamp watermarkTs,
    const SequenceNumber sequenceNumber)
{
    PRECONDITION(pipelineCtx != nullptr, "pipeline context should not be null");
    auto* opHandler = getBatchHandler(ptrOpHandler);

    auto batchesToBeEmitted = opHandler->getCreatedBatches(false);

    for (size_t index = 0; index < batchesToBeEmitted.size(); ++index)
    {
        const auto chunkNumber = INITIAL_CHUNK_NUMBER.getRawValue() + index;
        const bool lastChunk = index + 1 == batchesToBeEmitted.size();
        const SequenceData sequenceData{sequenceNumber, ChunkNumber(chunkNumber), lastChunk};

        opHandler->emitBatchesToProbe(*batchesToBeEmitted[index], sequenceData, pipelineCtx, watermarkTs);
    }
}

}

BatchingPhysicalOperator::BatchingPhysicalOperator(
    const OperatorHandlerId operatorHandlerId,
    std::shared_ptr<TupleBufferRef> tupleBufferRef)
    : operatorHandlerId(operatorHandlerId), tupleBufferRef(std::move(tupleBufferRef))
{
}

void BatchingPhysicalOperator::open(ExecutionContext& executionCtx, RecordBuffer&) const
{
    const auto operatorHandler = executionCtx.getGlobalOperatorHandler(operatorHandlerId);
    executionCtx.setLocalOperatorState(id, std::make_unique<BatchingOperatorLocalState>(operatorHandler));
}

void BatchingPhysicalOperator::execute(ExecutionContext& executionCtx, Record& record) const
{
    auto* const localState = dynamic_cast<BatchingOperatorLocalState*>(executionCtx.getLocalState(id));
    auto operatorHandler = localState->getOperatorHandler();

    const auto batchMemRef = nautilus::invoke(getOrCreateBatch, operatorHandler);
    const auto batchPagedVectorMemRef = nautilus::invoke(getBatchPagedVector, batchMemRef);

    const PagedVectorRef batchPagedVectorRef(batchPagedVectorMemRef, tupleBufferRef);
    batchPagedVectorRef.writeRecord(record, executionCtx.pipelineMemoryProvider.bufferProvider);
}

void BatchingPhysicalOperator::close(ExecutionContext& executionCtx, RecordBuffer&) const
{
    const auto operatorHandlerMemRef = executionCtx.getGlobalOperatorHandler(operatorHandlerId);
    nautilus::invoke(
        emitBatchesProxy,
        operatorHandlerMemRef,
        executionCtx.pipelineContext,
        executionCtx.watermarkTs,
        executionCtx.sequenceNumber);
}

std::optional<PhysicalOperator> BatchingPhysicalOperator::getChild() const
{
    return child;
}

void BatchingPhysicalOperator::setChild(PhysicalOperator newChild)
{
    child = std::move(newChild);
}

}
