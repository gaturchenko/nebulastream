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
#include <cstdint>
#include <functional>
#include <memory>
#include <Inference/PredictionCache/PredictionCacheEntry.hpp>
#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <nautilus/val.hpp>

namespace NES
{

struct HitsAndMisses
{
    uint64_t hits;
    uint64_t misses;
};

class PredictionCache
{
public:
    PredictionCache(
        uint64_t numberOfEntries,
        uint64_t sizeOfEntry,
        nautilus::val<int8_t*> startOfEntries,
        nautilus::val<uint64_t*> hitsRef,
        nautilus::val<uint64_t*> missesRef,
        nautilus::val<size_t> inputSize);
    virtual ~PredictionCache() = default;

    using PredictionCacheReplacement = std::function<nautilus::val<std::byte*>(
        const nautilus::val<PredictionCacheEntry*>& predictionCacheEntryToReplace, const nautilus::val<uint64_t>& replacementIndex)>;
    using PredictionCacheUpdate = std::function<void(
        const nautilus::val<PredictionCacheEntry*>& predictionCacheEntryToReplace, const nautilus::val<uint64_t>& replacementIndex)>;

    virtual nautilus::val<std::byte*>
    getDataStructureRef(const nautilus::val<std::byte*>& record, const PredictionCacheReplacement& replacementFunction) = 0;
    virtual nautilus::val<uint64_t> updateKeys(const nautilus::val<std::byte*>& record, const PredictionCacheUpdate& updateFunction) = 0;
    virtual void updateValues(const nautilus::val<uint64_t>& pos, const PredictionCacheUpdate& updateFunction) = 0;

    virtual nautilus::val<std::byte*> getRecord(const nautilus::val<uint64_t>& pos);
    virtual nautilus::val<std::byte*> getDataStructure(const nautilus::val<uint64_t>& pos);

    [[nodiscard]] nautilus::val<uint64_t*> getHitsRef() const;
    [[nodiscard]] nautilus::val<uint64_t*> getMissesRef() const;
    [[nodiscard]] nautilus::val<uint64_t> getReplacementIndex() const;
    void configureLookupIndex(nautilus::val<ChainedHashMap*> lookupIndex, nautilus::val<AbstractBufferProvider*> bufferProvider);

    virtual nautilus::val<uint64_t> getReplacementPos() = 0;
    virtual void setReplacementPos(nautilus::val<uint64_t> pos) = 0;

    static constexpr uint64_t NOT_FOUND = UINT64_MAX;

protected:
    void incrementNumberOfHits();
    void incrementNumberOfMisses();
    nautilus::val<uint64_t> searchInCache(const nautilus::val<std::byte*>& record);
    nautilus::val<bool> foundRecord(const nautilus::val<uint64_t>& pos, const nautilus::val<std::byte*>& record);
    void addLookupIndexEntry(const nautilus::val<std::byte*>& record, const nautilus::val<uint64_t>& pos);
    void rebuildLookupIndex();

    nautilus::val<int8_t*> startOfEntries;
    nautilus::val<uint64_t> numberOfEntries;
    nautilus::val<uint64_t> sizeOfEntry;
    nautilus::val<uint64_t> replacementIndex = 0;

private:
    nautilus::val<uint64_t*> numberOfHits;
    nautilus::val<uint64_t*> numberOfMisses;
    nautilus::val<size_t> inputSize;
    nautilus::val<ChainedHashMap*> lookupIndex;
    nautilus::val<AbstractBufferProvider*> lookupIndexBufferProvider;
};

}
