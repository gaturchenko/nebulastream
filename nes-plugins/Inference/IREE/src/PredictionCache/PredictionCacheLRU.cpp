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

#include <PredictionCache/PredictionCacheLRU.hpp>

#include <Nautilus/DataTypes/DataTypesUtil.hpp>

namespace NES
{
PredictionCacheLRU::PredictionCacheLRU(
    const nautilus::val<OperatorHandler*>& operatorHandler,
    const uint64_t numberOfEntries,
    const uint64_t sizeOfEntry,
    const nautilus::val<int8_t*>& startOfEntries,
    const nautilus::val<uint64_t*>& hitsRef,
    const nautilus::val<uint64_t*>& missesRef,
    const nautilus::val<size_t>& inputSize)
    : PredictionCache(operatorHandler, numberOfEntries, sizeOfEntry, startOfEntries, hitsRef, missesRef, inputSize)
    , accessCounter(0)
{
}

nautilus::val<uint64_t> PredictionCacheLRU::getReplacementPos()
{
    nautilus::val<uint64_t> minAge = UINT64_MAX;
    nautilus::val<uint64_t> minAgeIndex = 0;
    for (nautilus::val<uint64_t> i = 0; i < numberOfEntries; ++i)
    {
        auto ageBit = getAgeBit(i);
        if (*ageBit < minAge)
        {
            minAge = *ageBit;
            minAgeIndex = i;
        }
    }
    return minAgeIndex;
}

nautilus::val<uint64_t*> PredictionCacheLRU::getAgeBit(const nautilus::val<uint64_t>& pos)
{
    const auto PredictionCacheEntry = startOfEntries + pos * sizeOfEntry;
    const auto ageBitRef = getMemberRef(PredictionCacheEntry, &PredictionCacheEntryLRU::ageBit);
    return ageBitRef;
}

void PredictionCacheLRU::updateValues(const nautilus::val<uint64_t>& pos, const PredictionCache::PredictionCacheUpdate& updateFunction)
{
    const nautilus::val<PredictionCacheEntry*> PredictionCacheEntryToReplace = startOfEntries + pos * sizeOfEntry;
    updateFunction(PredictionCacheEntryToReplace, pos);
}

nautilus::val<uint64_t> PredictionCacheLRU::updateKeys(const nautilus::val<std::byte*>& record, const PredictionCache::PredictionCacheUpdate& updateFunction)
{
    /// First, we check if the record is already in the cache. If this is the case, we update its access timestamp.
    if (const auto dataStructurePos = PredictionCache::searchInCache(record); dataStructurePos != PredictionCache::NOT_FOUND)
    {
        incrementNumberOfHits();
        accessCounter = accessCounter + 1;
        *getAgeBit(dataStructurePos) = accessCounter;
        return dataStructurePos;
    }

    /// If the record is not in the cache, we have a cache miss.
    incrementNumberOfMisses();

    /// Second, we have to replace the least-recently-used entry.
    const auto replacementPos = getReplacementPos();
    const nautilus::val<PredictionCacheEntry*> PredictionCacheEntryToReplace = startOfEntries + replacementPos * sizeOfEntry;
    updateFunction(PredictionCacheEntryToReplace, replacementPos);
    replacementIndex = replacementPos;
    accessCounter = accessCounter + 1;
    *getAgeBit(replacementPos) = accessCounter;
    return nautilus::val<uint64_t>(NOT_FOUND);
}

nautilus::val<std::byte*>
PredictionCacheLRU::getDataStructureRef(const nautilus::val<std::byte*>& record, const PredictionCache::PredictionCacheReplacement& replacementFunction)
{
    /// First, we check if the record is already in the cache. If this is the case, we update its access timestamp.
    if (const auto dataStructurePos = PredictionCache::searchInCache(record); dataStructurePos != PredictionCache::NOT_FOUND)
    {
        incrementNumberOfHits();
        accessCounter = accessCounter + 1;
        *getAgeBit(dataStructurePos) = accessCounter;
        return getDataStructure(dataStructurePos);
    }

    /// If the record is not in the cache, we have a cache miss.
    incrementNumberOfMisses();

    /// Second, we have to replace the least-recently-used entry.
    const auto replacementPos = getReplacementPos();
    const nautilus::val<PredictionCacheEntry*> PredictionCacheEntryToReplace = startOfEntries + replacementPos * sizeOfEntry;
    const auto dataStructure = replacementFunction(PredictionCacheEntryToReplace, replacementPos);
    replacementIndex = replacementPos;
    accessCounter = accessCounter + 1;
    *getAgeBit(replacementPos) = accessCounter;
    return dataStructure;
}
}
