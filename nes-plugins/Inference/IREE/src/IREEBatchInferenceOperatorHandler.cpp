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

#include <IREEAdapter.hpp>
#include <IREEBatchInferenceOperatorHandler.hpp>
#include <HashMapOptions.hpp>
#include <Util/Logger/Logger.hpp>
#include <PipelineExecutionContext.hpp>
#include <PredictionCache.hpp>
#include <WindowBasedOperatorHandler.hpp>
#include <algorithm>

namespace NES
{
namespace
{
constexpr uint64_t MIN_PREDICTION_CACHE_LOOKUP_INDEX_PAGE_SIZE = 4096;

uint64_t getPredictionCacheLookupIndexPageSize(const uint64_t keySize)
{
    const auto mapEntrySize = sizeof(ChainedHashMapEntry) + keySize + sizeof(uint64_t);
    return std::max(MIN_PREDICTION_CACHE_LOOKUP_INDEX_PAGE_SIZE, mapEntrySize);
}
}

IREEBatchInferenceOperatorHandler::IREEBatchInferenceOperatorHandler(
    const std::vector<OriginId>& inputOrigins,
    OriginId outputOriginId,
    Nebuli::Inference::Model model,
    uint64_t batchSize)
    : WindowBasedOperatorHandler(inputOrigins, outputOriginId, true)
    , model(std::move(model))
    , batchSize(batchSize)
{
}

void IREEBatchInferenceOperatorHandler::start(PipelineExecutionContext& pipelineExecutionContext, uint32_t)
{
    numberOfWorkerThreads = pipelineExecutionContext.getNumberOfWorkerThreads();
    watermarkProcessorBuild = std::make_unique<MultiOriginWatermarkProcessor>(inputOrigins);
    watermarkProcessorProbe = std::make_unique<MultiOriginWatermarkProcessor>(std::vector{outputOriginId});

    threadLocalAdapters.reserve(numberOfWorkerThreads);
    for (size_t threadId = 0; threadId < numberOfWorkerThreads; ++threadId)
    {
        threadLocalAdapters.emplace_back(IREEAdapter::create());
        threadLocalAdapters.back()->initializeModel(model, batchSize);
    }
}

void IREEBatchInferenceOperatorHandler::stop(QueryTerminationType, PipelineExecutionContext& pipelineExecutionContext)
{
    if (model.getInputs()[0].isType(DataType::Type::VARSIZED))
    {
        uint64_t misses{0};
        uint64_t lowReductions{0};
        uint64_t mediumReductions{0};
        uint64_t highReductions{0};
        uint64_t fullReductions{0};

        for (const auto& adapter : threadLocalAdapters)
        {
            misses += adapter->misses;
            lowReductions += adapter->lowReductions;
            mediumReductions += adapter->mediumReductions;
            highReductions += adapter->highReductions;
            fullReductions += adapter->fullReductions;
        }

        NES_INFO("{{\"pipeline_id\": {}, \"misses\": {}, \"low_reductions\": {}, \"medium_reductions\": {}, \"high_reductions\": {}, \"full_reductions\": {}}}"
            , pipelineExecutionContext.getPipelineId(), misses, lowReductions, mediumReductions, highReductions, fullReductions)
    }
    threadLocalAdapters.clear();
}

void IREEBatchInferenceOperatorHandler::allocateBuffers(size_t tupleSize)
{
    for (size_t threadId = 0; threadId < numberOfWorkerThreads; ++threadId)
    {
        threadLocalAdapters.at(threadId)->allocateBuffers(tupleSize);
    }
}

void IREEBatchInferenceOperatorHandler::allocateHashMaps(uint64_t keySize, uint64_t valueSize, uint64_t numberOfBuckets, uint64_t pageSize)
{
    for (size_t threadId = 0; threadId < numberOfWorkerThreads; ++threadId)
    {
        auto hashMapPtr = std::make_unique<ChainedHashMap>(keySize, valueSize, numberOfBuckets, pageSize);
        threadLocalHashMaps.emplace_back(std::move(hashMapPtr));
    }
}

const Nebuli::Inference::Model& IREEBatchInferenceOperatorHandler::getModel() const
{
    return model;
}

const std::shared_ptr<IREEAdapter>& IREEBatchInferenceOperatorHandler::getIREEAdapter(WorkerThreadId workerThreadId) const
{
    return threadLocalAdapters[workerThreadId % threadLocalAdapters.size()];
}

HashMap* IREEBatchInferenceOperatorHandler::getHashMapPtr(WorkerThreadId workerThreadId) const
{
    return threadLocalHashMaps[workerThreadId % threadLocalHashMaps.size()].get();
}

void IREEBatchInferenceOperatorHandler::clearHashMap(WorkerThreadId workerThreadId)
{
    dynamic_cast<ChainedHashMap*>(threadLocalHashMaps[workerThreadId % threadLocalHashMaps.size()].get())->clear();
}

void IREEBatchInferenceOperatorHandler::emitBatchesToProbe(
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

std::shared_ptr<Batch> IREEBatchInferenceOperatorHandler::createNewBatch() const
{
    tuplesSeen = 0;
    ++batchId;
    auto batch = std::make_shared<Batch>(batchId, 1);
    batch->setState(BatchState::MARKED_AS_CREATED);
    return batch;
}

std::shared_ptr<Batch> IREEBatchInferenceOperatorHandler::getBatch(uint64_t batchId) const
{
    auto batchesReadLock = batches.rlock();

    if (batchesReadLock->contains(batchId))
    {
        return batchesReadLock->at(batchId);
    }
    return nullptr;
}

Batch* IREEBatchInferenceOperatorHandler::getOrCreateNewBatch() const
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

void IREEBatchInferenceOperatorHandler::garbageCollectBatches() const
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

void IREEBatchInferenceOperatorHandler::allocatePredictionCacheEntries(
    const uint64_t sizeOfEntry, const uint64_t numberOfEntries, AbstractBufferProvider* bufferProvider)
{
    if (hasPredictionCacheCreated.exchange(true))
    {
        return;
    }

    PRECONDITION(bufferProvider != nullptr, "Buffer provider should not be null");
    for (uint64_t i = 0; i < threadLocalAdapters.size(); ++i)
    {
        const auto neededSize = numberOfEntries * sizeOfEntry + sizeof(HitsAndMisses);
        INVARIANT(neededSize > 0, "Size of entry should be larger than 0");

        auto bufferOpt = bufferProvider->getUnpooledBuffer(neededSize);
        INVARIANT(bufferOpt.has_value(), "Buffer provider should return a buffer");
        std::ranges::fill(bufferOpt.value().getAvailableMemoryArea(), std::byte{0});
        predictionCacheEntriesBufferForWorkerThreads.emplace_back(bufferOpt.value());
        predictionCacheReplacementPosForWorkerThreads.emplace_back(uint64_t{0});

        const auto keySize = threadLocalAdapters.at(i)->inputSize / batchSize;
        const auto pageSize = getPredictionCacheLookupIndexPageSize(keySize);
        predictionCacheLookupHashMapsForWorkerThreads.emplace_back(
            std::make_unique<ChainedHashMap>(keySize, sizeof(uint64_t), numberOfEntries, pageSize));
    }
}

const int8_t* IREEBatchInferenceOperatorHandler::getStartOfPredictionCacheEntries(const StartPredictionCacheEntriesArgs& startPredictionCacheEntriesArgs) const
{
    PRECONDITION(!threadLocalAdapters.empty(), "Number of worker threads should be set before calling this method");
    const auto startPredictionCacheEntriesIREE = dynamic_cast<const StartPredictionCacheEntriesIREEInference&>(startPredictionCacheEntriesArgs);
    const auto pos = startPredictionCacheEntriesIREE.workerThreadId % predictionCacheEntriesBufferForWorkerThreads.size();
    INVARIANT(
        not predictionCacheEntriesBufferForWorkerThreads.empty() and pos < predictionCacheEntriesBufferForWorkerThreads.size(),
        "Position should be smaller than the size of the predictionCacheEntriesBufferForWorkerThreads");
    return predictionCacheEntriesBufferForWorkerThreads.at(pos).getAvailableMemoryArea<int8_t>().data();
}

uint64_t IREEBatchInferenceOperatorHandler::getReplacementPos(const StartPredictionCacheEntriesArgs& startPredictionCacheEntriesArgs) const
{
    PRECONDITION(!threadLocalAdapters.empty(), "Number of worker threads should be set before calling this method");
    const auto startPredictionCacheEntriesIREE = dynamic_cast<const StartPredictionCacheEntriesIREEInference&>(startPredictionCacheEntriesArgs);
    const auto pos = startPredictionCacheEntriesIREE.workerThreadId % predictionCacheReplacementPosForWorkerThreads.size();
    INVARIANT(
        not predictionCacheReplacementPosForWorkerThreads.empty() and pos < predictionCacheReplacementPosForWorkerThreads.size(),
        "Position should be smaller than the size of the predictionCacheReplacementPosForWorkerThreads");
    return predictionCacheReplacementPosForWorkerThreads.at(pos);
}

void
IREEBatchInferenceOperatorHandler::setReplacementPos(const StartPredictionCacheEntriesArgs& startPredictionCacheEntriesArgs, uint64_t idx)
{
    PRECONDITION(!threadLocalAdapters.empty(), "Number of worker threads should be set before calling this method");
    const auto startPredictionCacheEntriesIREE = dynamic_cast<const StartPredictionCacheEntriesIREEInference&>(startPredictionCacheEntriesArgs);
    const auto pos = startPredictionCacheEntriesIREE.workerThreadId % predictionCacheReplacementPosForWorkerThreads.size();
    INVARIANT(
        not predictionCacheReplacementPosForWorkerThreads.empty() and pos < predictionCacheReplacementPosForWorkerThreads.size(),
        "Position should be smaller than the size of the predictionCacheReplacementPosForWorkerThreads");
    predictionCacheReplacementPosForWorkerThreads[pos] = idx;
}

std::function<std::vector<std::shared_ptr<Slice>>(SliceStart, SliceEnd)>
IREEBatchInferenceOperatorHandler::getCreateNewSlicesFunction(const CreateNewSlicesArguments&) const
{
    return [](SliceStart, SliceEnd)
    {
        return std::vector<std::shared_ptr<Slice>>{};
    };
}

}
