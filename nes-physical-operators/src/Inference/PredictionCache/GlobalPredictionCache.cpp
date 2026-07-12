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

#include <Inference/PredictionCache/GlobalPredictionCache.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <string_view>
#include <utility>

#include <ErrorHandling.hpp>

namespace NES
{

GlobalPredictionCache::GlobalPredictionCache(
    const PredictionCacheType policy, const uint64_t numberOfEntries, const size_t recordSize, const size_t predictionSize)
    : policy(policy), recordSize(recordSize), predictionSize(predictionSize), slots(numberOfEntries), lfuBuckets(numberOfEntries + 1)
{
    PRECONDITION(policy != PredictionCacheType::NONE, "GlobalPredictionCache requires a replacement policy");
    PRECONDITION(numberOfEntries > 0, "GlobalPredictionCache requires at least one entry");
    for (auto& slot : slots)
    {
        slot.record.resize(recordSize);
        slot.prediction.resize(predictionSize);
    }
}

uint64_t GlobalPredictionCache::hashRecordBytes(const std::byte* record) const
{
    return std::hash<std::string_view>{}(std::string_view{reinterpret_cast<const char*>(record), recordSize});
}

uint64_t GlobalPredictionCache::findSlot(const std::byte* record) const
{
    const auto candidates = lookupIndex.find(hashRecordBytes(record));
    if (candidates == lookupIndex.end())
    {
        return NOT_FOUND;
    }
    for (const auto pos : candidates->second)
    {
        if (slots[pos].occupied && std::memcmp(slots[pos].record.data(), record, recordSize) == 0)
        {
            return pos;
        }
    }
    return NOT_FOUND;
}

void GlobalPredictionCache::removeFromLfuBucket(const uint64_t pos)
{
    auto& bucket = lfuBuckets[slots[pos].frequency];
    bucket.erase(std::ranges::remove(bucket, pos).begin(), bucket.end());
}

void GlobalPredictionCache::addToLfuBucket(const uint64_t pos, const uint64_t frequency)
{
    slots[pos].frequency = std::min<uint64_t>(frequency, slots.size());
    lfuBuckets[slots[pos].frequency].push_back(pos);
}

void GlobalPredictionCache::updateMinFrequencyAfterRemoving(const uint64_t oldFrequency)
{
    if (oldFrequency != minFrequency || !lfuBuckets[minFrequency].empty())
    {
        return;
    }
    while (minFrequency < lfuBuckets.size() && lfuBuckets[minFrequency].empty())
    {
        ++minFrequency;
    }
}

void GlobalPredictionCache::appendLruTail(const uint64_t pos)
{
    lruOrder.erase(std::ranges::remove(lruOrder, pos).begin(), lruOrder.end());
    lruOrder.push_back(pos);
}

void GlobalPredictionCache::onHit(const uint64_t pos)
{
    switch (policy)
    {
        case PredictionCacheType::FIFO:
            return;
        case PredictionCacheType::LFU: {
            if (const auto oldFrequency = slots[pos].frequency; oldFrequency < slots.size())
            {
                removeFromLfuBucket(pos);
                addToLfuBucket(pos, oldFrequency + 1);
                updateMinFrequencyAfterRemoving(oldFrequency);
            }
            return;
        }
        case PredictionCacheType::LRU:
            appendLruTail(pos);
            return;
        case PredictionCacheType::SECOND_CHANCE:
            slots[pos].secondChanceBit = true;
            return;
        case PredictionCacheType::NONE:
            break;
    }
    std::unreachable();
}

uint64_t GlobalPredictionCache::selectVictim()
{
    switch (policy)
    {
        case PredictionCacheType::FIFO: {
            const auto pos = fifoReplacementIndex;
            fifoReplacementIndex = (fifoReplacementIndex + 1) % slots.size();
            return pos;
        }
        case PredictionCacheType::SECOND_CHANCE: {
            /// The hand stays parked on the victim; the freshly inserted entry gets
            /// its second chance bit set by insert().
            while (slots[secondChanceReplacementIndex].secondChanceBit)
            {
                slots[secondChanceReplacementIndex].secondChanceBit = false;
                secondChanceReplacementIndex = (secondChanceReplacementIndex + 1) % slots.size();
            }
            return secondChanceReplacementIndex;
        }
        case PredictionCacheType::LRU: {
            uint64_t pos = NOT_FOUND;
            if (nextEmptyPos < slots.size())
            {
                pos = nextEmptyPos++;
            }
            else
            {
                pos = lruOrder.front();
                lruOrder.erase(lruOrder.begin());
            }
            appendLruTail(pos);
            return pos;
        }
        case PredictionCacheType::LFU: {
            uint64_t pos = NOT_FOUND;
            if (nextEmptyPos < slots.size())
            {
                pos = nextEmptyPos++;
            }
            else
            {
                pos = lfuBuckets[minFrequency].front();
                lfuBuckets[minFrequency].erase(lfuBuckets[minFrequency].begin());
            }
            addToLfuBucket(pos, 1);
            minFrequency = 1;
            return pos;
        }
        case PredictionCacheType::NONE:
            break;
    }
    std::unreachable();
}

bool GlobalPredictionCache::lookup(const std::byte* record, std::byte* predictionOut)
{
    const std::scoped_lock lock(mutex);
    const auto pos = findSlot(record);
    if (pos == NOT_FOUND)
    {
        ++misses;
        return false;
    }
    ++hits;
    onHit(pos);
    std::memcpy(predictionOut, slots[pos].prediction.data(), predictionSize);
    return true;
}

void GlobalPredictionCache::insert(const std::byte* record, const std::byte* prediction)
{
    const std::scoped_lock lock(mutex);
    if (const auto existingPos = findSlot(record); existingPos != NOT_FOUND)
    {
        std::memcpy(slots[existingPos].prediction.data(), prediction, predictionSize);
        return;
    }

    const auto pos = selectVictim();
    auto& slot = slots[pos];
    if (slot.occupied)
    {
        auto& candidates = lookupIndex[hashRecordBytes(slot.record.data())];
        candidates.erase(std::ranges::remove(candidates, pos).begin(), candidates.end());
        if (candidates.empty())
        {
            lookupIndex.erase(hashRecordBytes(slot.record.data()));
        }
    }

    std::memcpy(slot.record.data(), record, recordSize);
    std::memcpy(slot.prediction.data(), prediction, predictionSize);
    slot.occupied = true;
    slot.secondChanceBit = policy == PredictionCacheType::SECOND_CHANCE;
    lookupIndex[hashRecordBytes(record)].push_back(pos);
}

HitsAndMisses GlobalPredictionCache::getHitsAndMisses() const
{
    const std::scoped_lock lock(mutex);
    return {.hits = hits, .misses = misses};
}

}
