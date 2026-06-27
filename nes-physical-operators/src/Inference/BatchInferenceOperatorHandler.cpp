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

#include <Inference/BatchInferenceOperatorHandler.hpp>

#include <algorithm>
#include <cstddef>
#include <new>
#include <numeric>
#include <utility>

#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
#include <Runtime/QueryTerminationType.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <Util/Logger/Logger.hpp>
#include <ErrorHandling.hpp>
#include <PipelineExecutionContext.hpp>

namespace NES
{

Batch::Batch(const uint64_t batchId, const uint64_t numberOfPagedVectors) : batchId(batchId)
{
    pagedVectors.reserve(numberOfPagedVectors);
    for (uint64_t i = 0; i < numberOfPagedVectors; ++i)
    {
        pagedVectors.emplace_back(std::make_unique<PagedVector>());
    }
}

PagedVector* Batch::getPagedVectorRef() const
{
    PRECONDITION(!pagedVectors.empty(), "Batch should own at least one paged vector");
    return pagedVectors.front().get();
}

void Batch::combinePagedVectors()
{
    const std::scoped_lock lock(batchMutex);
    if (pagedVectors.size() <= 1)
    {
        return;
    }

    for (size_t i = 1; i < pagedVectors.size(); ++i)
    {
        pagedVectors.front()->moveAllPages(*pagedVectors[i]);
    }
    pagedVectors.erase(pagedVectors.begin() + 1, pagedVectors.end());
}

uint64_t Batch::getNumberOfTuples() const
{
    return std::accumulate(
        pagedVectors.begin(),
        pagedVectors.end(),
        uint64_t{0},
        [](const uint64_t sum, const auto& pagedVector) { return sum + pagedVector->getTotalNumberOfEntries(); });
}

size_t Batch::getNumberOfPagedVectors() const
{
    return pagedVectors.size();
}

void Batch::setState(const BatchState newState)
{
    state = newState;
}

void Batch::setPostJoinBatchMetadata(PostJoinBatchMetadata metadata)
{
    postJoinBatchMetadata = std::move(metadata);
}

bool Batch::hasPostJoinBatchMetadata() const
{
    return postJoinBatchMetadata.has_value();
}

BatchInferenceOperatorHandler::BatchInferenceOperatorHandler(const uint64_t batchSize, const OriginId outputOriginId)
    : outputOriginId(outputOriginId), batchSize(batchSize)
{
    PRECONDITION(batchSize > 0, "Batch size must be larger than zero");
}

void BatchInferenceOperatorHandler::start(PipelineExecutionContext&, uint32_t)
{
}

void BatchInferenceOperatorHandler::stop(QueryTerminationType, PipelineExecutionContext&)
{
    const std::scoped_lock lock(batchesMutex);
    batches.clear();
    batchId = 0;
    tuplesSeen = 0;
    threadLocalHashMaps.clear();
}

void BatchInferenceOperatorHandler::allocateHashMaps(
    const uint64_t numberOfWorkerThreads,
    const uint64_t keySize,
    const uint64_t valueSize,
    const uint64_t numberOfBuckets,
    const uint64_t pageSize)
{
    if (!threadLocalHashMaps.empty())
    {
        return;
    }

    threadLocalHashMaps.reserve(numberOfWorkerThreads);
    for (uint64_t threadId = 0; threadId < numberOfWorkerThreads; ++threadId)
    {
        threadLocalHashMaps.emplace_back(std::make_unique<ChainedHashMap>(keySize, valueSize, numberOfBuckets, pageSize));
    }
}

HashMap* BatchInferenceOperatorHandler::getHashMapPtr(const WorkerThreadId workerThreadId) const
{
    PRECONDITION(!threadLocalHashMaps.empty(), "Batch deduplication hash maps should be allocated before execution");
    return threadLocalHashMaps[workerThreadId % threadLocalHashMaps.size()].get();
}

void BatchInferenceOperatorHandler::clearHashMap(const WorkerThreadId workerThreadId)
{
    PRECONDITION(!threadLocalHashMaps.empty(), "Batch deduplication hash maps should be allocated before execution");
    dynamic_cast<ChainedHashMap*>(threadLocalHashMaps[workerThreadId % threadLocalHashMaps.size()].get())->clear();
}

std::shared_ptr<Batch> BatchInferenceOperatorHandler::createNewBatch()
{
    tuplesSeen = 0;
    ++batchId;
    auto batch = std::make_shared<Batch>(batchId);
    batch->setState(BatchState::MARKED_AS_CREATED);
    return batch;
}

Batch* BatchInferenceOperatorHandler::getOrCreateNewBatch()
{
    const std::scoped_lock lock(batchesMutex);
    ++tuplesSeen;

    if (batches.empty() || tuplesSeen == batchSize || batches.back()->state == BatchState::MARKED_AS_EMITTED)
    {
        auto batch = createNewBatch();
        batches.emplace_back(std::move(batch));
        return batches.back().get();
    }

    return batches.back().get();
}

Batch* BatchInferenceOperatorHandler::createNewPostJoinBatch(PostJoinBatchMetadata metadata)
{
    const std::scoped_lock lock(batchesMutex);
    auto batch = createNewBatch();
    batch->setPostJoinBatchMetadata(std::move(metadata));
    batches.emplace_back(std::move(batch));
    return batches.back().get();
}

std::shared_ptr<Batch> BatchInferenceOperatorHandler::getBatch(const uint64_t requestedBatchId) const
{
    const std::scoped_lock lock(batchesMutex);
    const auto it
        = std::ranges::find_if(batches, [requestedBatchId](const auto& batch) { return batch && batch->batchId == requestedBatchId; });
    if (it == batches.end())
    {
        return nullptr;
    }
    return *it;
}

std::vector<std::shared_ptr<Batch>> BatchInferenceOperatorHandler::getCreatedBatches(const bool fullBatchesOnly) const
{
    const std::scoped_lock lock(batchesMutex);
    std::vector<std::shared_ptr<Batch>> createdBatches;
    for (const auto& batch : batches)
    {
        if (!batch || batch->state != BatchState::MARKED_AS_CREATED)
        {
            continue;
        }
        if (fullBatchesOnly && batch->getNumberOfTuples() != batchSize)
        {
            continue;
        }
        createdBatches.emplace_back(batch);
    }
    return createdBatches;
}

void BatchInferenceOperatorHandler::garbageCollectBatches()
{
    const std::scoped_lock lock(batchesMutex);
    std::erase_if(batches, [](const auto& batch) { return batch && batch->state == BatchState::MARKED_AS_PROCESSED; });
}

void BatchInferenceOperatorHandler::emitBatchesToProbe(
    Batch& batch, const SequenceData& sequenceData, PipelineExecutionContext* pipelineCtx, const Timestamp watermarkTs)
{
    PRECONDITION(pipelineCtx != nullptr, "pipeline context should not be null");

    batch.combinePagedVectors();
    const auto numberOfTuples = batch.getNumberOfTuples();

    auto tupleBuffer = pipelineCtx->allocateTupleBuffer();
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

}
