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

#include <optional>
#include <Nautilus/Interface/PagedVector/PagedVector.hpp>
#include <Runtime/Execution/OperatorHandler.hpp>
#include <Runtime/QueryTerminationType.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <Sequencing/Sequencer.hpp>
#include <WindowBasedOperatorHandler.hpp>

namespace NES
{

class Batch;

class SequenceOperatorHandler final : public WindowBasedOperatorHandler
{
public:
    SequenceOperatorHandler(
        const std::vector<OriginId>& inputOrigins,
        OriginId outputOriginId,
        uint64_t batchSize);

    std::optional<TupleBuffer*> getNextBuffer(TupleBuffer* tupleBuffer);
    std::optional<TupleBuffer*> markBufferAsDone(TupleBuffer* tupleBuffer);

    void start(PipelineExecutionContext& pipelineExecutionContext, uint32_t) override
    {
        sequencer = {};
        numberOfWorkerThreads = pipelineExecutionContext.getNumberOfWorkerThreads();
        watermarkProcessorBuild = std::make_unique<MultiOriginWatermarkProcessor>(inputOrigins);
        watermarkProcessorProbe = std::make_unique<MultiOriginWatermarkProcessor>(std::vector{outputOriginId});
    }
    void stop(QueryTerminationType, PipelineExecutionContext&) override { /*noop*/ }

    [[nodiscard]] Batch* getOrCreateNewBatch() const;
    [[nodiscard]] std::shared_ptr<Batch> getBatch(uint64_t batchId) const;
    [[nodiscard]] std::shared_ptr<Batch> createNewBatch() const;
    void garbageCollectBatches() const;
    void emitBatchesToProbe(Batch& batch,
        const SequenceData& sequenceData,
        PipelineExecutionContext* pipelineCtx,
        const Timestamp watermarkTs) const;

    [[nodiscard]] std::function<std::vector<std::shared_ptr<Slice>>(SliceStart, SliceEnd)>
    getCreateNewSlicesFunction(const CreateNewSlicesArguments&) const override;
    void triggerSlices(
        const std::map<WindowInfoAndSequenceNumber, std::vector<std::shared_ptr<Slice>>>& ,
        PipelineExecutionContext*) override { /*noop*/ };

    uint64_t getBatchSize() const { return batchSize; }

    Sequencer<TupleBuffer> sequencer;
    TupleBuffer currentBuffer;

    mutable uint64_t batchId = 0;
    mutable uint64_t tuplesSeen = 0;
    mutable folly::Synchronized<std::map<uint64_t, std::shared_ptr<Batch>>> batches;

private:
    uint64_t batchSize;
};

enum class BatchState : uint8_t
{
    MARKED_AS_CREATED,
    MARKED_AS_EMITTED,
    MARKED_AS_PROCESSED
};

class Batch
{
public:
    Batch(uint64_t batchId, uint64_t numberOfBatches) : batchId(batchId)
    {
        for (uint64_t i = 0; i < numberOfBatches; ++i)
        {
            pagedVectors.emplace_back(std::make_unique<PagedVector>());
        }
    }

    [[nodiscard]] PagedVector* getPagedVectorRef() const
    {
        return pagedVectors[0].get();
    }

    void combinePagedVectors()
    {
        const std::scoped_lock lock(batchMutex);
        if (pagedVectors.size() > 1)
        {
            for (uint64_t i = 1; i < pagedVectors.size(); ++i)
            {
                pagedVectors[0]->moveAllPages(*pagedVectors[i]);
            }
            pagedVectors.erase(pagedVectors.begin() + 1, pagedVectors.end());
        }
    }

    uint64_t getNumberOfTuples() const
    {
        return std::accumulate(
                pagedVectors.begin(),
                pagedVectors.end(),
                0,
                [](uint64_t sum, const auto& pagedVector) { return sum + pagedVector->getTotalNumberOfEntries(); });
    }

    size_t getNumberOfPagedVectors() const
    {
        return pagedVectors.size();
    }

    void setState(BatchState state)
    {
        this->state = state;
    }

    uint64_t batchId;
    mutable BatchState state;
private:
    std::mutex batchMutex;
    std::vector<std::unique_ptr<PagedVector>> pagedVectors;
};

struct EmittedBatch
{
    uint64_t batchId;
};

}
