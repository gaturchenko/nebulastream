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

#include <EmitPhysicalOperator.hpp>

namespace NES
{

class InterBufferBatchingOperator : public PhysicalOperatorConcept
{
public:
    explicit InterBufferBatchingOperator(
        OperatorHandlerId operatorHandlerId,
        std::shared_ptr<TupleBufferRef> bufferRef,
        uint64_t batchSize);

    void setup(ExecutionContext& executionCtx, CompilationContext&) const override;
    void terminate(ExecutionContext& executionCtx) const override;

    void open(ExecutionContext&, RecordBuffer&) const override { /* noop */ }
    void close(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const override;

    void execute(ExecutionContext& executionCtx, Record& record) const override;

    [[nodiscard]] std::optional<PhysicalOperator> getChild() const override;
    void setChild(PhysicalOperator child) override;

private:
    std::optional<PhysicalOperator> child;
    std::shared_ptr<TupleBufferRef> bufferRef;
    uint64_t batchSize;
    OperatorHandlerId operatorHandlerId;
};

}
