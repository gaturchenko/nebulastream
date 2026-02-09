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

void InterBufferBatchingOperatorHandler::stop(QueryTerminationType, PipelineExecutionContext& pipelineExecutionContext)
{
    emitTupleBuffer(&pipelineExecutionContext);
}

TupleBuffer* InterBufferBatchingOperatorHandler::getTupleBufferRef()
{
    return tupleBuffer.withWLock([&](TupleBuffer& tb)
    {
        return std::addressof(tb);
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
    uint32_t varSizedValueLength)
{
    return tupleBuffer.withWLock([&](TupleBuffer& tb)
    {
        const std::span varSizedValueSpan{varSizedPtr, varSizedPtr + varSizedValueLength};
        return MemoryLayout::writeVarSized<MemoryLayout::PREPEND_NONE>(tb, *bufferProvider, std::as_bytes(varSizedValueSpan));
    });
}

void InterBufferBatchingOperatorHandler::emitTupleBuffer(PipelineExecutionContext* pipelineExecutionContext)
{
    tupleBuffer.withWLock([&](TupleBuffer& tb)
    {
        tb.setSequenceNumber(sequenceNumber);
        sequenceNumber = SequenceNumber(sequenceNumber.getRawValue() + 1);
        outputIndex.store(0, std::memory_order_seq_cst);

        NES_TRACE(
            "Emitted buffer with watermarkTs {} {} tuples {}",
            tb.getWatermark(),
            tb.getSequenceDataAsString(),
            tb.getNumberOfTuples());

        pipelineExecutionContext->emitBuffer(tb, PipelineExecutionContext::ContinuationPolicy::POSSIBLE);
        auto newBuffer = pipelineExecutionContext->allocateTupleBuffer();
        tb = std::move(newBuffer);
    });
}

}
