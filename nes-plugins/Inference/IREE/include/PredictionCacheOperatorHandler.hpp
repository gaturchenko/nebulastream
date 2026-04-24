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
#include <PredictionCache/PredictionCacheEntry.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/TupleBuffer.hpp>
#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>

namespace NES
{

class PredictionCacheOperatorHandler
{
public:
    virtual ~PredictionCacheOperatorHandler() = default;
    virtual void
    allocatePredictionCacheEntries(const uint64_t sizeOfEntry, const uint64_t numberOfEntries, AbstractBufferProvider* bufferProvider)
        = 0;

    struct StartPredictionCacheEntriesArgs
    {
        WorkerThreadId workerThreadId;

        explicit StartPredictionCacheEntriesArgs(const WorkerThreadId& workerThreadId) : workerThreadId(workerThreadId) { }

        StartPredictionCacheEntriesArgs(StartPredictionCacheEntriesArgs&& other) = default;
        StartPredictionCacheEntriesArgs& operator=(StartPredictionCacheEntriesArgs&& other) = default;
        StartPredictionCacheEntriesArgs(StartPredictionCacheEntriesArgs& other) = default;
        StartPredictionCacheEntriesArgs& operator=(StartPredictionCacheEntriesArgs& other) = default;
        virtual ~StartPredictionCacheEntriesArgs() = default;
    };

    virtual const int8_t* getStartOfPredictionCacheEntries(const StartPredictionCacheEntriesArgs& startPredictionCacheEntriesArgs) const = 0;

    virtual uint64_t getReplacementPos(const StartPredictionCacheEntriesArgs& startPredictionCacheEntriesArgs) const = 0;
    virtual void setReplacementPos(const StartPredictionCacheEntriesArgs& startPredictionCacheEntriesArgs, uint64_t idx) = 0;
    [[nodiscard]] ChainedHashMap*
    getPredictionCacheLookupHashMapPtr(const StartPredictionCacheEntriesArgs& startPredictionCacheEntriesArgs) const
    {
        if (predictionCacheLookupHashMapsForWorkerThreads.empty())
        {
            return nullptr;
        }

        const auto pos = startPredictionCacheEntriesArgs.workerThreadId % predictionCacheLookupHashMapsForWorkerThreads.size();
        return predictionCacheLookupHashMapsForWorkerThreads.at(pos).get();
    }

protected:
    void registerPredictionCacheLayout(const uint64_t sizeOfEntry, const uint64_t numberOfEntries)
    {
        predictionCacheEntrySize = sizeOfEntry;
        predictionCacheNumberOfEntries = numberOfEntries;
    }

    void cleanupPredictionCacheEntries()
    {
        if (!hasPredictionCacheCreated.exchange(false))
        {
            predictionCacheEntriesBufferForWorkerThreads.clear();
            predictionCacheReplacementPosForWorkerThreads.clear();
            predictionCacheLookupHashMapsForWorkerThreads.clear();
            predictionCacheEntrySize = 0;
            predictionCacheNumberOfEntries = 0;
            return;
        }

        constexpr uint64_t hitsAndMissesHeaderSize = 2 * sizeof(uint64_t);
        if (predictionCacheEntrySize > 0 && predictionCacheNumberOfEntries > 0)
        {
            for (auto& predictionCacheEntriesBuffer : predictionCacheEntriesBufferForWorkerThreads)
            {
                auto memoryArea = predictionCacheEntriesBuffer.getAvailableMemoryArea<std::byte>();
                if (memoryArea.size() < hitsAndMissesHeaderSize + predictionCacheNumberOfEntries * predictionCacheEntrySize)
                {
                    continue;
                }

                auto* entriesStart = memoryArea.data() + hitsAndMissesHeaderSize;
                for (uint64_t i = 0; i < predictionCacheNumberOfEntries; ++i)
                {
                    auto* entry = reinterpret_cast<PredictionCacheEntry*>(entriesStart + i * predictionCacheEntrySize);
                    delete[] entry->record;
                    delete[] entry->dataStructure;
                    entry->record = nullptr;
                    entry->dataStructure = nullptr;
                    entry->recordSize = 0;
                    entry->dataSize = 0;
                }
            }
        }

        predictionCacheEntriesBufferForWorkerThreads.clear();
        predictionCacheReplacementPosForWorkerThreads.clear();
        predictionCacheLookupHashMapsForWorkerThreads.clear();
        predictionCacheEntrySize = 0;
        predictionCacheNumberOfEntries = 0;
    }

    std::vector<TupleBuffer> predictionCacheEntriesBufferForWorkerThreads;
    std::vector<uint64_t> predictionCacheReplacementPosForWorkerThreads;
    std::vector<std::unique_ptr<ChainedHashMap>> predictionCacheLookupHashMapsForWorkerThreads;
    std::atomic<bool> hasPredictionCacheCreated{false};

private:
    uint64_t predictionCacheEntrySize = 0;
    uint64_t predictionCacheNumberOfEntries = 0;
};

}
