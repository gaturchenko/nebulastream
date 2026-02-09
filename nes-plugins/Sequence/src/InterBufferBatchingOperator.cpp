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
#include <Nautilus/Interface/BufferRef/RowTupleBufferRef.hpp>
#include <Nautilus/Interface/VariableSizedAccessRef.hpp>

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

    const auto memoryLayout = dynamic_cast<RowLayout*>(bufferRef->getMemoryLayout().get());
    const auto bufferAddress = buffer.getMemArea();

    auto tupleSize = memoryLayout->getTupleSize();
    const auto recordOffset = bufferAddress + (tupleSize * outputIndex);

    /// thread-safe writing to the TupleBuffer memory (for now only support/assume varsized data)
    const auto schema = memoryLayout->getSchema();
    const nautilus::val<uint64_t> varSizedOffset = 0;
    for (nautilus::static_val<size_t> i = 0; i < schema.getNumberOfFields(); ++i)
    {
        auto fieldOffset = memoryLayout->getFieldOffset(i);
        auto fieldAddress = recordOffset + nautilus::val<uint64_t>(fieldOffset);
        const auto& value = record.read(schema.getFieldAt(i).name);
        const auto varSizedValue = value.cast<VariableSizedData>();
        const auto variableSizedAccess = nautilus::invoke(
            +[](OperatorHandler* handler, AbstractBufferProvider* bufferProvider, const int8_t* varSizedPtr, uint32_t varSizedValueLength)
            {
                return dynamic_cast<InterBufferBatchingOperatorHandler*>(handler)->writeToTupleBuffer(bufferProvider, varSizedPtr, varSizedValueLength);
            }, operatorHandler, executionCtx.pipelineMemoryProvider.bufferProvider, varSizedValue.getReference(), varSizedValue.getTotalSize());
        auto fieldReferenceCastedU64 = static_cast<nautilus::val<uint64_t*>>(fieldAddress);
        *fieldReferenceCastedU64 = variableSizedAccess.convertToValue();
    }

    // bufferRef->writeRecord(outputIndex, buffer, record, executionCtx.pipelineMemoryProvider.bufferProvider);

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
