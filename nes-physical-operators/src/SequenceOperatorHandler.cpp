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

#include <SequenceOperatorHandler.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <ranges>
#include <utility>
#include <PipelineExecutionContext.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <ErrorHandling.hpp>

namespace NES
{

SequenceOperatorHandler::SequenceOperatorHandler(
    const std::vector<OriginId>& inputOrigins,
    OriginId outputOriginId,
    uint64_t batchSize)
    : WindowBasedOperatorHandler(inputOrigins, outputOriginId, true)
    , batchSize(batchSize)
{
}

std::optional<TupleBuffer*> SequenceOperatorHandler::getNextBuffer(TupleBuffer* tupleBuffer)
{
    if (auto optBuffer = sequencer.isNext(SequenceData(tupleBuffer->getSequenceNumber(), tupleBuffer->getChunkNumber(), tupleBuffer->isLastChunk()), *tupleBuffer))
    {
        currentBuffer = std::move(*optBuffer);
        return std::addressof(currentBuffer);
    }
    return {};
}

std::optional<TupleBuffer*> SequenceOperatorHandler::markBufferAsDone(TupleBuffer* tupleBuffer)
{
    INVARIANT(tupleBuffer == std::addressof(currentBuffer), "Not sure where this pointer is coming from");
    auto optNextBuffer
        = sequencer.advanceAndGetNext(SequenceData(tupleBuffer->getSequenceNumber(), tupleBuffer->getChunkNumber(), tupleBuffer->isLastChunk()));
    if (optNextBuffer)
    {
        currentBuffer = std::move(*optNextBuffer);
        return std::addressof(currentBuffer);
    }
    return {};
}

void SequenceOperatorHandler::emitBatchesToProbe(
    Batch& batch,
    const SequenceData& sequenceData,
    PipelineExecutionContext* pipelineCtx,
    const Timestamp watermarkTs) const
{
    batch.combinePagedVectors();
    const auto numberOfTuples = batch.getNumberOfTuples();

    auto tupleBuffer = pipelineCtx->getBufferManager()->getBufferBlocking();
    tupleBuffer.setOriginId(outputOriginId);
    tupleBuffer.setSequenceNumber(SequenceNumber(sequenceData.sequenceNumber));
    tupleBuffer.setChunkNumber(ChunkNumber(sequenceData.chunkNumber));
    tupleBuffer.setLastChunk(sequenceData.lastChunk);
    tupleBuffer.setWatermark(watermarkTs);
    tupleBuffer.setNumberOfTuples(numberOfTuples);

    auto bufferMemory = tupleBuffer.getAvailableMemoryArea();
    new (bufferMemory.data()) EmittedBatch{batch.batchId};

    pipelineCtx->emitBuffer(tupleBuffer);
    batch.setState(BatchState::MARKED_AS_EMITTED);

    NES_TRACE(
        "Emitted batch {} with watermarkTs {} {} originId {} tuples {}",
        batch.batchId,
        tupleBuffer.getWatermark(),
        tupleBuffer.getSequenceDataAsString(),
        tupleBuffer.getOriginId(),
        tupleBuffer.getNumberOfTuples());
}

std::shared_ptr<Batch> SequenceOperatorHandler::createNewBatch() const
{
    tuplesSeen = 0;
    ++batchId;
    auto batch = std::make_shared<Batch>(batchId, 1);
    batch->setState(BatchState::MARKED_AS_CREATED);
    return batch;
}

std::shared_ptr<Batch> SequenceOperatorHandler::getBatch(uint64_t batchId) const
{
    auto batchesReadLock = batches.rlock();

    if (batchesReadLock->contains(batchId))
    {
        return batchesReadLock->at(batchId);
    }
    return nullptr;
}

Batch* SequenceOperatorHandler::getOrCreateNewBatch() const
{
    auto batchesWriteLock = batches.wlock();
    tuplesSeen++;
    if (tuplesSeen == batchSize || !batchesWriteLock->contains(batchId) ||
        (batchesWriteLock->contains(batchId) && batchesWriteLock->at(batchId)->state == BatchState::MARKED_AS_EMITTED))
    {
        std::shared_ptr<Batch> batch = createNewBatch();
        batchesWriteLock->insert(std::make_pair(batchId, batch));
        return batch.get();
    }

    return batchesWriteLock->at(batchId).get();
}

void SequenceOperatorHandler::garbageCollectBatches() const
{
    auto batchesWriteLock = batches.wlock();

    if (batchesWriteLock->contains(batchId) && batchesWriteLock->at(batchId)->state == BatchState::MARKED_AS_PROCESSED)
    {
        auto processedBatches = *batchesWriteLock
            | std::views::filter([](const auto& pair)
                {
                    const auto& batch = pair.second;
                    return batch && batch->state == BatchState::MARKED_AS_PROCESSED;
                })
            | std::views::transform([](const auto& pair)
                {
                    return pair.first;
                });
        auto batchesCount = static_cast<size_t>(std::ranges::distance(processedBatches));

        std::vector batchesToErase(processedBatches.begin(), processedBatches.end());
        for (const uint64_t batchId : batchesToErase)
        {
            batchesWriteLock->erase(batchId);
        }
    }
}

std::function<std::vector<std::shared_ptr<Slice>>(SliceStart, SliceEnd)>
SequenceOperatorHandler::getCreateNewSlicesFunction(const CreateNewSlicesArguments&) const
{
    return [](SliceStart, SliceEnd)
    {
        return std::vector<std::shared_ptr<Slice>>{};
    };
}

}
