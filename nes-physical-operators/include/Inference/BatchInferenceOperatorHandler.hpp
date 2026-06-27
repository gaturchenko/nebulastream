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

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <Identifiers/Identifiers.hpp>
#include <Nautilus/Interface/HashMap/HashMap.hpp>
#include <Nautilus/Interface/PagedVector/PagedVector.hpp>
#include <Runtime/Execution/OperatorHandler.hpp>
#include <Sequencing/SequenceData.hpp>
#include <Time/Timestamp.hpp>

namespace NES
{

class PipelineExecutionContext;

enum class BatchState : uint8_t
{
    MARKED_AS_CREATED,
    MARKED_AS_EMITTED,
    MARKED_AS_PROCESSED
};

struct PostJoinBatchMetadata
{
    std::vector<std::string> payloadFieldNames;
    std::vector<std::string> outputFieldNames;
};

class Batch
{
public:
    explicit Batch(uint64_t batchId, uint64_t numberOfPagedVectors = 1);

    [[nodiscard]] PagedVector* getPagedVectorRef() const;
    void combinePagedVectors();
    [[nodiscard]] uint64_t getNumberOfTuples() const;
    [[nodiscard]] size_t getNumberOfPagedVectors() const;
    void setState(BatchState newState);
    void setPostJoinBatchMetadata(PostJoinBatchMetadata metadata);
    [[nodiscard]] bool hasPostJoinBatchMetadata() const;

    uint64_t batchId;
    mutable BatchState state = BatchState::MARKED_AS_CREATED;
    std::optional<PostJoinBatchMetadata> postJoinBatchMetadata;

private:
    mutable std::mutex batchMutex;
    std::vector<std::unique_ptr<PagedVector>> pagedVectors;
};

struct EmittedBatch
{
    uint64_t batchId;
};

class BatchInferenceOperatorHandler final : public OperatorHandler
{
public:
    explicit BatchInferenceOperatorHandler(uint64_t batchSize, OriginId outputOriginId = INVALID_ORIGIN_ID);

    void start(PipelineExecutionContext& pipelineExecutionContext, uint32_t localStateVariableId) override;
    void stop(QueryTerminationType terminationType, PipelineExecutionContext& pipelineExecutionContext) override;

    void
    allocateHashMaps(uint64_t numberOfWorkerThreads, uint64_t keySize, uint64_t valueSize, uint64_t numberOfBuckets, uint64_t pageSize);
    [[nodiscard]] HashMap* getHashMapPtr(WorkerThreadId workerThreadId) const;
    void clearHashMap(WorkerThreadId workerThreadId);

    [[nodiscard]] Batch* getOrCreateNewBatch();
    [[nodiscard]] Batch* createNewPostJoinBatch(PostJoinBatchMetadata metadata);
    [[nodiscard]] std::shared_ptr<Batch> getBatch(uint64_t requestedBatchId) const;
    [[nodiscard]] std::vector<std::shared_ptr<Batch>> getCreatedBatches(bool fullBatchesOnly) const;
    void garbageCollectBatches();
    void emitBatchesToProbe(Batch& batch, const SequenceData& sequenceData, PipelineExecutionContext* pipelineCtx, Timestamp watermarkTs);

    [[nodiscard]] uint64_t getBatchSize() const { return batchSize; }

private:
    [[nodiscard]] std::shared_ptr<Batch> createNewBatch();

    OriginId outputOriginId;
    uint64_t batchSize;
    uint64_t batchId = 0;
    uint64_t tuplesSeen = 0;
    mutable std::mutex batchesMutex;
    std::vector<std::shared_ptr<Batch>> batches;
    std::vector<std::unique_ptr<HashMap>> threadLocalHashMaps;
};

}
