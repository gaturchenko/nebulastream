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

#include <Runtime/TupleBuffer.hpp>
#include <Runtime/Execution/OperatorHandler.hpp>
#include <folly/Synchronized.h>

namespace NES
{

class InterBufferBatchingOperatorHandler : public OperatorHandler
{
public:
    explicit InterBufferBatchingOperatorHandler() = default;

    void start(PipelineExecutionContext& pipelineExecutionContext, uint32_t) override;
    void stop(QueryTerminationType, PipelineExecutionContext&) override;

    TupleBuffer* getTupleBufferRef();
    void createNewTupleBufferRef(PipelineExecutionContext& pipelineExecutionContext);
    void emitTupleBuffer(PipelineExecutionContext* pipelineExecutionContext);

    std::atomic<uint64_t> outputIndex = 0;
    mutable std::mutex tupleBufferLock;
    SequenceNumber sequenceNumber = SequenceNumber(1);

private:
    folly::Synchronized<TupleBuffer> tupleBuffer;
};

}
