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

#pragma once

#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <Runtime/Execution/OperatorHandler.hpp>
#include <folly/Synchronized.h>

namespace NES
{

class InterBufferBatchingOperatorHandler : public OperatorHandler
{
public:
    InterBufferBatchingOperatorHandler(uint64_t batchSize): batchSize(batchSize) {}

    void start(PipelineExecutionContext& pipelineExecutionContext, uint32_t) override;
    void stop(QueryTerminationType, PipelineExecutionContext&) override;

    uint64_t getBatchSize() const;

    int8_t* getTupleBufferRef();
    void createNewTupleBufferRef(PipelineExecutionContext& pipelineExecutionContext);

    VariableSizedAccess writeToTupleBuffer(
        AbstractBufferProvider* bufferProvider,
        const int8_t* varSizedPtr,
        uint32_t varSizedValueLength,
        PipelineExecutionContext* pipelineExecutionContext,
        Timestamp watermarkTs,
        OriginId originId,
        Timestamp creationTs);

    void emitTupleBuffer(
        PipelineExecutionContext* pipelineExecutionContext,
        Timestamp watermarkTs,
        OriginId originId,
        Timestamp creationTs,
        bool createNewBuffer);

    void emitTupleBuffer(
        TupleBuffer& tb,
        PipelineExecutionContext* pipelineExecutionContext,
        Timestamp watermarkTs,
        OriginId originId,
        Timestamp creationTs);

    void emitIfFullBatch(
        PipelineExecutionContext* pipelineExecutionContext,
        Timestamp watermarkTs,
        OriginId originId,
        Timestamp creationTs);

    uint64_t outputIndex = 0;
    SequenceNumber sequenceNumber = SequenceNumber(1);

private:
    uint64_t batchSize;
    folly::Synchronized<TupleBuffer> tupleBuffer;
};

}
