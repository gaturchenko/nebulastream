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

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <Nautilus/Interface/BufferRef/TupleBufferRef.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <Nautilus/Interface/RecordBuffer.hpp>
#include <Runtime/Execution/OperatorHandler.hpp>

#include <CompilationContext.hpp>
#include <Model.hpp>
#include <PhysicalOperator.hpp>

namespace NES
{
struct InferenceRuntimeOptions;

namespace detail
{
struct ThreadLocalRuntimeWrapper;
}

/// @brief Scan-style physical operator that batches records before invoking the
/// model runtime and then forwards the original records with inference outputs.
class BatchInferModelPhysicalOperator final : public PhysicalOperatorConcept
{
public:
    BatchInferModelPhysicalOperator(
        CompiledModel model,
        std::shared_ptr<TupleBufferRef> bufferRef,
        std::vector<Record::RecordFieldIdentifier> projections,
        std::vector<std::string> inputFieldNames,
        std::vector<std::string> outputFieldNames,
        size_t batchSize,
        InferenceRuntimeOptions runtimeOptions,
        bool varsizedInput,
        bool varsizedOutput,
        OperatorHandlerId operatorHandlerId);

    void setup(ExecutionContext& executionCtx, CompilationContext& compilationContext) const override;
    void open(ExecutionContext& ctx, RecordBuffer& recordBuffer) const override;

    void close(ExecutionContext&, RecordBuffer&) const override { /* closed by open() after the batch scan has produced records */ }

    void terminate(ExecutionContext& executionCtx) const override;

    [[nodiscard]] std::optional<PhysicalOperator> getChild() const override;
    void setChild(PhysicalOperator child) override;

private:
    std::shared_ptr<detail::ThreadLocalRuntimeWrapper> threadLocal;
    std::shared_ptr<TupleBufferRef> bufferRef;
    std::vector<Record::RecordFieldIdentifier> projections;
    std::vector<std::string> inputFieldNames;
    std::vector<std::string> outputFieldNames;
    size_t batchSize;
    size_t inputTupleSize;
    size_t outputTupleSize;
    bool varsizedInput;
    bool varsizedOutput;
    OperatorHandlerId operatorHandlerId;
    std::optional<PhysicalOperator> child;
};

}
