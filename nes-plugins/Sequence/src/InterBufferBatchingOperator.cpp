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

#include <InterBufferBatchingOperator.hpp>

#include <InterBufferBatchingOperatorHandler.hpp>
#include <Nautilus/Interface/PagedVector/PagedVectorRef.hpp>

namespace NES
{

InterBufferBatchingOperator::InterBufferBatchingOperator(
    OperatorHandlerId operatorHandlerId,
    std::shared_ptr<TupleBufferRef> memoryProvider,
    uint64_t batchSize)
    : bufferRef(std::move(memoryProvider))
    , batchSize(batchSize)
    , operatorHandlerId(operatorHandlerId)
{
}

void InterBufferBatchingOperator::execute(ExecutionContext& executionCtx, Record& record) const
{
    auto operatorHandler = executionCtx.getGlobalOperatorHandler(operatorHandlerId);

    /// get a member reference for a tuple buffer and an output index
    const auto tupleBufferMemRef = nautilus::invoke(
        +[](OperatorHandler* handler)
        {
            auto interBufferBatchingHandler = dynamic_cast<InterBufferBatchingOperatorHandler*>(handler);
            return interBufferBatchingHandler->getTupleBufferRef();
        }, operatorHandler);

    auto outputIndex = nautilus::invoke(
        +[](OperatorHandler* handler)
        {
            return dynamic_cast<InterBufferBatchingOperatorHandler*>(handler)->outputIndex.load(std::memory_order_seq_cst);
        }, operatorHandler);

    /// we use a record buffer as a handle to conveniently write the record to a tuple buffer
    auto buffer = RecordBuffer(tupleBufferMemRef);

    /// if the tuple buffer contains batchSize records, emit the buffer
    if (outputIndex >= batchSize)
    {
        buffer.setNumRecords(outputIndex);
        buffer.setWatermarkTs(executionCtx.watermarkTs);
        buffer.setOriginId(executionCtx.originId);
        buffer.setSequenceNumber(executionCtx.sequenceNumber);
        buffer.setChunkNumber(nautilus::val<ChunkNumber>(1));
        buffer.setCreationTs(executionCtx.currentTs);
        buffer.setLastChunk(true);

        nautilus::invoke(
            +[](OperatorHandler* handler, PipelineExecutionContext* pipelineCtx)
            {
                dynamic_cast<InterBufferBatchingOperatorHandler*>(handler)->emitTupleBuffer(pipelineCtx);
            }, operatorHandler, executionCtx.pipelineContext);
    }

    bufferRef->writeRecord(outputIndex, buffer, record, executionCtx.pipelineMemoryProvider.bufferProvider);

    /// increment the output index stored in the handler
    nautilus::invoke(
        +[](OperatorHandler* handler)
        {
            auto interBufferBatchingHandler = dynamic_cast<InterBufferBatchingOperatorHandler*>(handler);
            interBufferBatchingHandler->outputIndex.fetch_add(1, std::memory_order_seq_cst);
        }, operatorHandler);
}

void InterBufferBatchingOperator::close(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const
{
    auto operatorHandler = executionCtx.getGlobalOperatorHandler(operatorHandlerId);

    const auto tupleBufferMemRef = nautilus::invoke(
            +[](OperatorHandler* handler)
            {
                auto interBufferBatchingHandler = dynamic_cast<InterBufferBatchingOperatorHandler*>(handler);
                return interBufferBatchingHandler->getTupleBufferRef();
            }, operatorHandler);

    auto outputIndex = nautilus::invoke(
        +[](OperatorHandler* handler)
        {
            return dynamic_cast<InterBufferBatchingOperatorHandler*>(handler)->outputIndex.load(std::memory_order_seq_cst);
        }, operatorHandler);

    recordBuffer = RecordBuffer(tupleBufferMemRef);

    recordBuffer.setNumRecords(outputIndex);
    recordBuffer.setWatermarkTs(executionCtx.watermarkTs);
    recordBuffer.setOriginId(executionCtx.originId);
    recordBuffer.setSequenceNumber(executionCtx.sequenceNumber);
    recordBuffer.setChunkNumber(nautilus::val<ChunkNumber>(1));
    recordBuffer.setCreationTs(executionCtx.currentTs);
    recordBuffer.setLastChunk(true);

    /// we need to check whether buffer is ready to be emitted
    if (outputIndex >= batchSize)
    {
        nautilus::invoke(
            +[](OperatorHandler* handler, PipelineExecutionContext* pipelineCtx)
            {
                dynamic_cast<InterBufferBatchingOperatorHandler*>(handler)->emitTupleBuffer(pipelineCtx);
            }, operatorHandler, executionCtx.pipelineContext);
    }
}

void InterBufferBatchingOperator::setup(ExecutionContext& executionCtx, CompilationContext&) const
{
    auto operatorHandler = executionCtx.getGlobalOperatorHandler(operatorHandlerId);
    nautilus::invoke(
        +[](OperatorHandler* handler, PipelineExecutionContext* pec)
        {
            handler->start(*pec, 0);
        }, operatorHandler, executionCtx.pipelineContext);
}

void InterBufferBatchingOperator::terminate(ExecutionContext& executionCtx) const
{
    auto operatorHandler = executionCtx.getGlobalOperatorHandler(operatorHandlerId);
    nautilus::invoke(
        +[](OperatorHandler* handler, PipelineExecutionContext* pec)
        {
            handler->stop(QueryTerminationType::Graceful, *pec);
        }, operatorHandler, executionCtx.pipelineContext);
}

std::optional<PhysicalOperator> InterBufferBatchingOperator::getChild() const
{
    return child;
}

void InterBufferBatchingOperator::setChild(PhysicalOperator child)
{
    this->child = std::move(child);
}

}
