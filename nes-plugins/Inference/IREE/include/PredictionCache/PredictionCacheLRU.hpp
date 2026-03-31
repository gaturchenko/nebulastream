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
#include <PredictionCache/PredictionCache.hpp>

namespace NES
{
struct PredictionCacheEntryLRU : PredictionCacheEntry
{
    uint64_t ageBit = 0;
    uint64_t previousPos = UINT64_MAX;
    uint64_t nextPos = UINT64_MAX;

    PredictionCacheEntryLRU() = default;

    PredictionCacheEntryLRU(const PredictionCacheEntryLRU& other)
        : PredictionCacheEntry(other), ageBit(other.ageBit) {}

    PredictionCacheEntryLRU&
    operator=(const PredictionCacheEntryLRU& other)
    {
        if (this == &other) return *this;

        PredictionCacheEntry::operator=(other);
        ageBit = other.ageBit;
        return *this;
    }

    PredictionCacheEntryLRU(PredictionCacheEntryLRU&& other) noexcept
        : PredictionCacheEntry(std::move(other)),
          ageBit(other.ageBit)
    {
        other.ageBit = 0;
    }

    PredictionCacheEntryLRU&
    operator=(PredictionCacheEntryLRU&& other) noexcept
    {
        if (this == &other) return *this;

        PredictionCacheEntry::operator=(std::move(other));
        ageBit = other.ageBit;
        other.ageBit = 0;
        return *this;
    }

    ~PredictionCacheEntryLRU() override = default;
};

class PredictionCacheLRU final : public PredictionCache
{
public:
    PredictionCacheLRU(
        const nautilus::val<OperatorHandler*>& operatorHandler,
        const uint64_t numberOfEntries,
        const uint64_t sizeOfEntry,
        const nautilus::val<int8_t*>& startOfEntries,
        const nautilus::val<uint64_t*>& hitsRef,
        const nautilus::val<uint64_t*>& missesRef,
        const nautilus::val<size_t>& inputSize);
    ~PredictionCacheLRU() override = default;
    nautilus::val<std::byte*>
    getDataStructureRef(const nautilus::val<std::byte*>& record, const PredictionCache::PredictionCacheReplacement& replacementFunction) override;
    nautilus::val<uint64_t> updateKeys(const nautilus::val<std::byte*>& record, const PredictionCache::PredictionCacheUpdate& updateFunction) override;
    void updateValues(const nautilus::val<uint64_t>& pos, const PredictionCache::PredictionCacheUpdate& updateFunction) override;
    nautilus::val<uint64_t> getReplacementPos() override;
    void setReplacementPos(nautilus::val<uint64_t>) override { /* noop */ }

private:
    nautilus::val<uint64_t*> getAgeBit(const nautilus::val<uint64_t>& pos);
    nautilus::val<uint64_t*> getPreviousPos(const nautilus::val<uint64_t>& pos);
    nautilus::val<uint64_t*> getNextPos(const nautilus::val<uint64_t>& pos);
    void appendToTail(const nautilus::val<uint64_t>& pos);
    void removeFromList(const nautilus::val<uint64_t>& pos);
    void touch(const nautilus::val<uint64_t>& pos);

    /// Monotonic access counter used as a timestamp for LRU tracking.
    nautilus::val<uint64_t> accessCounter;
    nautilus::val<uint64_t> nextEmptyPos;
    nautilus::val<uint64_t> lruHead;
    nautilus::val<uint64_t> lruTail;
};
}
