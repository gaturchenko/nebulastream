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

#include <InterBufferBatchingPhysicalOperator.hpp>

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

class InterBufferBatchingOperatorLocalState final : public OperatorState
{
public:
    explicit InterBufferBatchingOperatorLocalState(const nautilus::val<OperatorHandler*>& operatorHandler)
        : operatorHandler(operatorHandler)
    {
    }

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

void emitInterBufferBatchesProxy(
    OperatorHandler* ptrOpHandler,
    PipelineExecutionContext* pipelineCtx,
    const Timestamp watermarkTs)
{
    PRECONDITION(pipelineCtx != nullptr, "pipeline context should not be null");
    auto* opHandler = getBatchHandler(ptrOpHandler);

    auto batchesToBeEmitted = opHandler->getCreatedBatches(true);

    for (const auto& batch : batchesToBeEmitted)
    {
        const SequenceData sequenceData{SequenceNumber(batch->batchId), INITIAL_CHUNK_NUMBER, true};

        opHandler->emitBatchesToProbe(*batch, sequenceData, pipelineCtx, watermarkTs);
    }
}

void emitRemainingBatchesProxy(
    OperatorHandler* ptrOpHandler,
    PipelineExecutionContext* pipelineCtx,
    const Timestamp watermarkTs)
{
    PRECONDITION(pipelineCtx != nullptr, "pipeline context should not be null");
    auto* opHandler = getBatchHandler(ptrOpHandler);

    auto batchesToBeEmitted = opHandler->getCreatedBatches(false);

    for (const auto& batch : batchesToBeEmitted)
    {
        const SequenceData sequenceData{SequenceNumber(batch->batchId), INITIAL_CHUNK_NUMBER, true};

        opHandler->emitBatchesToProbe(*batch, sequenceData, pipelineCtx, watermarkTs);
    }
}

}

InterBufferBatchingPhysicalOperator::InterBufferBatchingPhysicalOperator(
    const OperatorHandlerId operatorHandlerId,
    std::shared_ptr<TupleBufferRef> tupleBufferRef)
    : operatorHandlerId(operatorHandlerId), tupleBufferRef(std::move(tupleBufferRef))
{
}

void InterBufferBatchingPhysicalOperator::open(ExecutionContext& executionCtx, RecordBuffer&) const
{
    const auto operatorHandler = executionCtx.getGlobalOperatorHandler(operatorHandlerId);
    executionCtx.setLocalOperatorState(id, std::make_unique<InterBufferBatchingOperatorLocalState>(operatorHandler));
}

void InterBufferBatchingPhysicalOperator::execute(ExecutionContext& executionCtx, Record& record) const
{
    auto* const localState = dynamic_cast<InterBufferBatchingOperatorLocalState*>(executionCtx.getLocalState(id));
    auto operatorHandler = localState->getOperatorHandler();

    const auto batchMemRef = nautilus::invoke(getOrCreateBatch, operatorHandler);
    const auto batchPagedVectorMemRef = nautilus::invoke(getBatchPagedVector, batchMemRef);

    const PagedVectorRef batchPagedVectorRef(batchPagedVectorMemRef, tupleBufferRef);
    batchPagedVectorRef.writeRecord(record, executionCtx.pipelineMemoryProvider.bufferProvider);
}

void InterBufferBatchingPhysicalOperator::close(ExecutionContext& executionCtx, RecordBuffer&) const
{
    const auto operatorHandlerMemRef = executionCtx.getGlobalOperatorHandler(operatorHandlerId);
    nautilus::invoke(
        emitInterBufferBatchesProxy,
        operatorHandlerMemRef,
        executionCtx.pipelineContext,
        executionCtx.watermarkTs);
}

void InterBufferBatchingPhysicalOperator::terminate(ExecutionContext& executionCtx) const
{
    const auto operatorHandlerMemRef = executionCtx.getGlobalOperatorHandler(operatorHandlerId);
    nautilus::invoke(
        emitRemainingBatchesProxy,
        operatorHandlerMemRef,
        executionCtx.pipelineContext,
        executionCtx.watermarkTs);
}

std::optional<PhysicalOperator> InterBufferBatchingPhysicalOperator::getChild() const
{
    return child;
}

void InterBufferBatchingPhysicalOperator::setChild(PhysicalOperator newChild)
{
    child = std::move(newChild);
}

}
