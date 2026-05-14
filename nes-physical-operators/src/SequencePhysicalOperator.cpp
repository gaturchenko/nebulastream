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

#include <optional>
#include <utility>

#include <ExecutionContext.hpp>
#include <Nautilus/Interface/RecordBuffer.hpp>
#include <PipelineExecutionContext.hpp>
#include <Runtime/Execution/OperatorHandler.hpp>
#include <Runtime/QueryTerminationType.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <SequenceOperatorHandler.hpp>
#include <function.hpp>

namespace NES
{

void SequencePhysicalOperator::open(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const
{
    auto buffer = nautilus::invoke(
        +[](OperatorHandler* handler, TupleBuffer* tupleBuffer) -> TupleBuffer*
        {
            auto* sequenceHandler = dynamic_cast<SequenceOperatorHandler*>(handler);
            PRECONDITION(sequenceHandler != nullptr, "operator handler should be a SequenceOperatorHandler");
            return sequenceHandler->getNextBuffer(tupleBuffer).value_or(nullptr);
        },
        executionCtx.getGlobalOperatorHandler(operatorHandlerId),
        recordBuffer.getReference());

    while (buffer)
    {
        RecordBuffer nextBufferInSequence{buffer};
        scan.open(executionCtx, nextBufferInSequence);
        scan.close(executionCtx, nextBufferInSequence);

        buffer = nautilus::invoke(
            +[](OperatorHandler* handler, TupleBuffer* tupleBuffer) -> TupleBuffer*
            {
                auto* sequenceHandler = dynamic_cast<SequenceOperatorHandler*>(handler);
                PRECONDITION(sequenceHandler != nullptr, "operator handler should be a SequenceOperatorHandler");
                return sequenceHandler->markBufferAsDone(tupleBuffer).value_or(nullptr);
            },
            executionCtx.getGlobalOperatorHandler(operatorHandlerId),
            buffer);
    }
}

void SequencePhysicalOperator::setup(ExecutionContext& executionCtx, CompilationContext& compilationCtx) const
{
    nautilus::invoke(
        +[](OperatorHandler* handler, PipelineExecutionContext* ctx) { handler->start(*ctx, 0); },
        executionCtx.getGlobalOperatorHandler(operatorHandlerId),
        executionCtx.pipelineContext);
    scan.setup(executionCtx, compilationCtx);
}

void SequencePhysicalOperator::terminate(ExecutionContext& executionCtx) const
{
    scan.terminate(executionCtx);
    nautilus::invoke(
        +[](OperatorHandler* handler, PipelineExecutionContext* ctx) { handler->stop(QueryTerminationType::Graceful, *ctx); },
        executionCtx.getGlobalOperatorHandler(operatorHandlerId),
        executionCtx.pipelineContext);
}

void SequencePhysicalOperator::setChild(PhysicalOperator child)
{
    scan.setChild(std::move(child));
}

std::optional<PhysicalOperator> SequencePhysicalOperator::getChild() const
{
    return scan.getChild();
}

}
