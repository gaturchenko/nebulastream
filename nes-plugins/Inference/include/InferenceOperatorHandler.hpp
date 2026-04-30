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

#include <Identifiers/Identifiers.hpp>
#include <InferenceAdapter.hpp>
#include <PredictionCacheOperatorHandler.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/Execution/OperatorHandler.hpp>
#include <Model.hpp>

namespace NES
{
class InferenceAdapter;
class InferenceOperatorHandler : public OperatorHandler, public PredictionCacheOperatorHandler
{
public:
    explicit InferenceOperatorHandler(Nebuli::Inference::Model model, InferenceRuntimeConfiguration runtimeConfiguration = {});

    void start(PipelineExecutionContext& pipelineExecutionContext, uint32_t localStateVariableId) override;
    void stop(QueryTerminationType terminationType, PipelineExecutionContext& pipelineExecutionContext) override;

    [[nodiscard]] const Nebuli::Inference::Model& getModel() const;
    [[nodiscard]] const std::shared_ptr<InferenceAdapter>& getAdapter(WorkerThreadId threadId) const;
    void allocatePredictionCacheEntries(
        const uint64_t sizeOfEntry, const uint64_t numberOfEntries, AbstractBufferProvider* bufferProvider) override;

    struct StartPredictionCacheEntriesInference final : StartPredictionCacheEntriesArgs
    {
        explicit StartPredictionCacheEntriesInference(const WorkerThreadId workerThreadId)
            : StartPredictionCacheEntriesArgs(workerThreadId)
        {
        }

        StartPredictionCacheEntriesInference(StartPredictionCacheEntriesInference&& other) = default;
        StartPredictionCacheEntriesInference& operator=(StartPredictionCacheEntriesInference&& other) = default;

        StartPredictionCacheEntriesInference(const StartPredictionCacheEntriesInference& other)
            : StartPredictionCacheEntriesArgs(other.workerThreadId)
        {
        }

        StartPredictionCacheEntriesInference& operator=(const StartPredictionCacheEntriesInference& other)
        {
            workerThreadId = other.workerThreadId;
            return *this;
        };

        ~StartPredictionCacheEntriesInference() override = default;
    };

    const int8_t* getStartOfPredictionCacheEntries(const StartPredictionCacheEntriesArgs& startPredictionCacheEntriesArgs) const override;

    uint64_t getReplacementPos(const StartPredictionCacheEntriesArgs& startPredictionCacheEntriesArgs) const override;
    void setReplacementPos(const StartPredictionCacheEntriesArgs& startPredictionCacheEntriesArgs, uint64_t idx) override;

private:
    Nebuli::Inference::Model model;
    InferenceRuntimeConfiguration runtimeConfiguration;
    std::vector<std::shared_ptr<InferenceAdapter>> threadLocalAdapters;
};

}
