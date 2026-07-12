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

#include <Nautilus/Interface/Record.hpp>
#include <Nautilus/Interface/RecordBuffer.hpp>
#include <InferenceConfiguration.hpp>

#include <CompilationContext.hpp>
#include <Model.hpp>
#include <PhysicalOperator.hpp>

namespace NES
{
struct InferenceRuntimeOptions;

namespace detail
{
struct ThreadLocalPredictionCacheWrapper;
}

/// @brief Physical operator that runs single-tuple model inference with a prediction cache
/// that is either per-worker (THREAD_LOCAL) or shared across all workers (GLOBAL).
class CacheInferModelPhysicalOperator final : public PhysicalOperatorConcept
{
public:
    CacheInferModelPhysicalOperator(
        CompiledModel model,
        std::vector<std::string> inputFieldNames,
        std::vector<std::string> outputFieldNames,
        InferenceRuntimeOptions runtimeOptions,
        PredictionCacheType predictionCacheType,
        PredictionCacheScope predictionCacheScope,
        size_t numberOfCacheEntries,
        bool varsizedInput,
        bool varsizedOutput);

    void setup(ExecutionContext& executionCtx, CompilationContext& compilationContext) const override;
    void open(ExecutionContext& ctx, RecordBuffer& recordBuffer) const override;
    void execute(ExecutionContext& ctx, Record& record) const override;
    void close(ExecutionContext& ctx, RecordBuffer& recordBuffer) const override;
    void terminate(ExecutionContext& executionCtx) const override;

    [[nodiscard]] std::optional<PhysicalOperator> getChild() const override;
    void setChild(PhysicalOperator child) override;

private:
    std::shared_ptr<detail::ThreadLocalPredictionCacheWrapper> threadLocal;
    std::vector<std::string> inputFieldNames;
    std::vector<std::string> outputFieldNames;
    size_t inputSize;
    size_t outputSize;
    bool varsizedInput;
    bool varsizedOutput;
    std::optional<PhysicalOperator> child;
};

}
