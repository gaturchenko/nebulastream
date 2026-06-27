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

#include <Inference/PostJoinBatchingPhysicalOperator.hpp>

#include <cstdint>
#include <memory>
#include <utility>

#include <Inference/BatchInferenceOperatorHandler.hpp>
#include <Nautilus/Interface/PagedVector/PagedVectorRef.hpp>
#include <Sequencing/SequenceData.hpp>
#include <ErrorHandling.hpp>
#include <ExecutionContext.hpp>
#include <OperatorState.hpp>
#include <PipelineExecutionContext.hpp>
#include <function.hpp>
#include <static.hpp>

namespace NES
{

namespace
{

class PostJoinBatchingOperatorLocalState final : public OperatorState
{
public:
    explicit PostJoinBatchingOperatorLocalState(const nautilus::val<OperatorHandler*>& operatorHandler) : operatorHandler(operatorHandler)
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

Batch* createPostJoinBatch(
    OperatorHandler* ptrOpHandler, const std::vector<std::string>* payloadFieldNames, const std::vector<std::string>* outputFieldNames)
{
    PRECONDITION(payloadFieldNames != nullptr, "Post-join payload field names should not be null");
    PRECONDITION(outputFieldNames != nullptr, "Post-join output field names should not be null");
    return getBatchHandler(ptrOpHandler)
        ->createNewPostJoinBatch(
            PostJoinBatchMetadata{.payloadFieldNames = *payloadFieldNames, .outputFieldNames = *outputFieldNames});
}

PagedVector* getBatchPagedVector(Batch* batch)
{
    PRECONDITION(batch != nullptr, "batch context should not be null!");
    return batch->getPagedVectorRef();
}

void emitCreatedBatchesProxy(OperatorHandler* ptrOpHandler, PipelineExecutionContext* pipelineCtx, const Timestamp watermarkTs)
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

void emitRemainingBatchesProxy(OperatorHandler* ptrOpHandler, PipelineExecutionContext* pipelineCtx, const Timestamp watermarkTs)
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

PostJoinBatchingPhysicalOperator::PostJoinBatchingPhysicalOperator(
    const OperatorHandlerId operatorHandlerId,
    std::shared_ptr<TupleBufferRef> tupleBufferRef,
    std::vector<std::string> payloadFieldNames,
    std::vector<std::string> outputFieldNames,
    std::string syntheticModelInputFieldName)
    : operatorHandlerId(operatorHandlerId)
    , tupleBufferRef(std::move(tupleBufferRef))
    , payloadFieldNames(std::move(payloadFieldNames))
    , outputFieldNames(std::move(outputFieldNames))
    , syntheticModelInputFieldName(std::move(syntheticModelInputFieldName))
{
    PRECONDITION(!this->payloadFieldNames.empty(), "Post-join batching requires at least one payload field");
    PRECONDITION(!this->outputFieldNames.empty(), "Post-join batching requires at least one output field");
    PRECONDITION(
        this->outputFieldNames.size() % this->payloadFieldNames.size() == 0,
        "Post-join batching requires output fields to be grouped evenly by payload field");
    PRECONDITION(!this->syntheticModelInputFieldName.empty(), "Post-join batching requires a synthetic model input field name");
}

void PostJoinBatchingPhysicalOperator::open(ExecutionContext& executionCtx, RecordBuffer&) const
{
    const auto operatorHandler = executionCtx.getGlobalOperatorHandler(operatorHandlerId);
    executionCtx.setLocalOperatorState(id, std::make_unique<PostJoinBatchingOperatorLocalState>(operatorHandler));
}

void PostJoinBatchingPhysicalOperator::execute(ExecutionContext& executionCtx, Record& record) const
{
    auto* const localState = dynamic_cast<PostJoinBatchingOperatorLocalState*>(executionCtx.getLocalState(id));
    auto operatorHandler = localState->getOperatorHandler();

    const auto batchMemRef
        = nautilus::invoke(createPostJoinBatch, operatorHandler, nautilus::val<const std::vector<std::string>*>(&payloadFieldNames), nautilus::val<const std::vector<std::string>*>(&outputFieldNames));
    const auto batchPagedVectorMemRef = nautilus::invoke(getBatchPagedVector, batchMemRef);
    const PagedVectorRef batchPagedVectorRef(batchPagedVectorMemRef, tupleBufferRef);

    for (nautilus::static_val<size_t> i = 0; i < payloadFieldNames.size(); ++i)
    {
        auto syntheticRecord = record;
        syntheticRecord.write(syntheticModelInputFieldName, record.read(payloadFieldNames.at(nautilus::static_val<int>(i))));
        batchPagedVectorRef.writeRecord(syntheticRecord, executionCtx.pipelineMemoryProvider.bufferProvider);
    }

    nautilus::invoke(emitCreatedBatchesProxy, operatorHandler, executionCtx.pipelineContext, executionCtx.watermarkTs);
}

void PostJoinBatchingPhysicalOperator::close(ExecutionContext&, RecordBuffer&) const
{
}

void PostJoinBatchingPhysicalOperator::terminate(ExecutionContext& executionCtx) const
{
    nautilus::invoke(
        emitRemainingBatchesProxy,
        executionCtx.getGlobalOperatorHandler(operatorHandlerId),
        executionCtx.pipelineContext,
        executionCtx.watermarkTs);
}

std::optional<PhysicalOperator> PostJoinBatchingPhysicalOperator::getChild() const
{
    return child;
}

void PostJoinBatchingPhysicalOperator::setChild(PhysicalOperator newChild)
{
    child = std::move(newChild);
}

}
