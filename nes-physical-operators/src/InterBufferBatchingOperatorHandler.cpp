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

#include <InterBufferBatchingOperatorHandler.hpp>

#include <MemoryLayout/MemoryLayout.hpp>
#include <PipelineExecutionContext.hpp>

namespace NES
{

void InterBufferBatchingOperatorHandler::start(PipelineExecutionContext& pipelineExecutionContext, uint32_t)
{
    createNewTupleBufferRef(pipelineExecutionContext);
}

void InterBufferBatchingOperatorHandler::stop(QueryTerminationType, PipelineExecutionContext&)
{
    tupleBuffer.withWLock([&](TupleBuffer& tb)
    {
        tb.release();
    });
}

int8_t* InterBufferBatchingOperatorHandler::getTupleBufferRef()
{
    return tupleBuffer.withWLock([&](TupleBuffer& tb)
    {
        return reinterpret_cast<int8_t*>(tb.getAvailableMemoryArea().data());
    });
}

void InterBufferBatchingOperatorHandler::createNewTupleBufferRef(PipelineExecutionContext& pipelineExecutionContext)
{
    tupleBuffer.withWLock([&](TupleBuffer& tb)
    {
        tb = pipelineExecutionContext.allocateTupleBuffer();
    });
}

VariableSizedAccess InterBufferBatchingOperatorHandler::writeToTupleBuffer(
    AbstractBufferProvider* bufferProvider,
    const int8_t* varSizedPtr,
    uint32_t varSizedValueLength,
    PipelineExecutionContext* pipelineExecutionContext,
    Timestamp watermarkTs,
    OriginId originId,
    Timestamp creationTs)
{
    return tupleBuffer.withWLock([&](TupleBuffer& tb)
    {
        if (batchSize == outputIndex)
        {
            emitTupleBuffer(tb, pipelineExecutionContext, watermarkTs, originId, creationTs);
            auto newBuffer = pipelineExecutionContext->allocateTupleBuffer();
            tb = std::move(newBuffer);
        }

        outputIndex += 1;
        const std::span varSizedValueSpan{varSizedPtr, varSizedPtr + varSizedValueLength};

        return MemoryLayout::writeVarSized<MemoryLayout::PREPEND_NONE>(tb, *bufferProvider, std::as_bytes(varSizedValueSpan));
    });
}

void InterBufferBatchingOperatorHandler::emitIfFullBatch(
    PipelineExecutionContext* pipelineExecutionContext,
    Timestamp watermarkTs,
    OriginId originId,
    Timestamp creationTs)
{
    tupleBuffer.withWLock([&](TupleBuffer& tb)
    {
        if (batchSize == outputIndex)
        {
            emitTupleBuffer(tb, pipelineExecutionContext, watermarkTs, originId, creationTs);
            auto newBuffer = pipelineExecutionContext->allocateTupleBuffer();
            tb = std::move(newBuffer);
        }
    });
}

void InterBufferBatchingOperatorHandler::emitTupleBuffer(
    TupleBuffer& tb,
    PipelineExecutionContext* pipelineExecutionContext,
    Timestamp watermarkTs,
    OriginId originId,
    Timestamp creationTs)
{
    /// set buffer metadata
    tb.setNumberOfTuples(outputIndex);
    tb.setWatermark(watermarkTs);
    tb.setOriginId(originId);
    tb.setSequenceNumber(sequenceNumber);
    tb.setChunkNumber(ChunkNumber(1));
    tb.setCreationTimestampInMS(creationTs);
    tb.setLastChunk(true);


    NES_TRACE(
        "Emitted buffer with watermarkTs {} {} tuples {}",
        tb.getWatermark(),
        tb.getSequenceDataAsString(),
        tb.getNumberOfTuples());

    /// emit the buffer and create a new one unless called from terminate()
    pipelineExecutionContext->emitBuffer(tb, PipelineExecutionContext::ContinuationPolicy::POSSIBLE);

    /// handler bookkeeping
    sequenceNumber = SequenceNumber(sequenceNumber.getRawValue() + 1);
    outputIndex = 0;
}

void InterBufferBatchingOperatorHandler::emitTupleBuffer(
    PipelineExecutionContext* pipelineExecutionContext,
    Timestamp watermarkTs,
    OriginId originId,
    Timestamp creationTs,
    bool createNewBuffer)
{
    tupleBuffer.withWLock([&](TupleBuffer& tb)
    {
        /// set buffer metadata
        tb.setNumberOfTuples(outputIndex);
        tb.setWatermark(watermarkTs);
        tb.setOriginId(originId);
        tb.setSequenceNumber(sequenceNumber);
        tb.setChunkNumber(ChunkNumber(1));
        tb.setCreationTimestampInMS(creationTs);
        tb.setLastChunk(true);


        NES_TRACE(
            "Emitted buffer with watermarkTs {} {} tuples {}",
            tb.getWatermark(),
            tb.getSequenceDataAsString(),
            tb.getNumberOfTuples());

        /// emit the buffer and create a new one unless called from terminate()
        pipelineExecutionContext->emitBuffer(tb, PipelineExecutionContext::ContinuationPolicy::POSSIBLE);

        /// handler bookkeeping
        sequenceNumber = SequenceNumber(sequenceNumber.getRawValue() + 1);
        outputIndex = 0;

        if (createNewBuffer)
        {
            auto newBuffer = pipelineExecutionContext->allocateTupleBuffer();
            tb = std::move(newBuffer);
        }
    });
}

uint64_t InterBufferBatchingOperatorHandler::getBatchSize() const
{
    return batchSize;
}

}
