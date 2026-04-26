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

#include <PredictionCache/PredictionCacheLFU.hpp>

namespace NES
{
PredictionCacheLFU::PredictionCacheLFU(
    const nautilus::val<OperatorHandler*>& operatorHandler,
    const uint64_t numberOfEntries,
    const uint64_t sizeOfEntry,
    const nautilus::val<int8_t*>& startOfEntries,
    const nautilus::val<uint64_t*>& hitsRef,
    const nautilus::val<uint64_t*>& missesRef,
    const nautilus::val<size_t>& inputSize)
    : PredictionCache(operatorHandler, numberOfEntries, sizeOfEntry, startOfEntries, hitsRef, missesRef, inputSize)
    , nextEmptyPos(0)
    , minFrequencyIndex(0)
    , minFrequencyDirty(true)
{
}

nautilus::val<uint64_t> PredictionCacheLFU::getReplacementPos()
{
    if (nextEmptyPos < numberOfEntries)
    {
        const auto replacementPos = nextEmptyPos;
        nextEmptyPos = nextEmptyPos + 1;
        return replacementPos;
    }

    if (minFrequencyDirty)
    {
        recomputeMinFrequencyIndex();
    }
    return minFrequencyIndex;
}

void PredictionCacheLFU::recomputeMinFrequencyIndex()
{
    nautilus::val<uint64_t> minFrequency = UINT64_MAX;
    nautilus::val<uint64_t> minFrequencyPos = 0;
    for (nautilus::val<uint64_t> i = 0; i < numberOfEntries; ++i)
    {
        nautilus::val<uint64_t> frequency{*getFrequency(i)};
        if (frequency < minFrequency)
        {
            minFrequency = frequency;
            minFrequencyPos = i;
        }
    }
    minFrequencyIndex = minFrequencyPos;
    minFrequencyDirty = false;
}

void PredictionCacheLFU::updateValues(const nautilus::val<uint64_t>& pos, const PredictionCache::PredictionCacheUpdate& updateFunction)
{
    const nautilus::val<PredictionCacheEntry*> PredictionCacheEntryToReplace = startOfEntries + pos * sizeOfEntry;
    updateFunction(PredictionCacheEntryToReplace, pos);
}

nautilus::val<uint64_t> PredictionCacheLFU::updateKeys(const nautilus::val<std::byte*>& record, const PredictionCache::PredictionCacheUpdate& updateFunction)
{
    /// First, we check if the timestamp is already in the cache.
    if (const auto dataStructurePos = PredictionCache::searchInCache(record); dataStructurePos != PredictionCache::NOT_FOUND)
    {
        incrementNumberOfHits();
        auto frequency = getFrequency(dataStructurePos);
        const auto newFrequency = nautilus::val<uint64_t>(*frequency) + nautilus::val<uint64_t>(1);
        *frequency = newFrequency;
        if (dataStructurePos == minFrequencyIndex)
        {
            minFrequencyDirty = true;
        }
        return dataStructurePos;
    }

    /// Second, if this is not the case, we replace the current LFU entry.
    incrementNumberOfMisses();
    const auto replacementPos = getReplacementPos();

    /// Third, we have to replace the entry at replacementPos.
    const nautilus::val<PredictionCacheEntry*> PredictionCacheEntryToReplace = startOfEntries + replacementPos * sizeOfEntry;
    updateFunction(PredictionCacheEntryToReplace, replacementPos);
    addLookupIndexEntry(record, replacementPos);
    replacementIndex = replacementPos;
    *getFrequency(replacementPos) = 1;
    if (nextEmptyPos < numberOfEntries)
    {
        return nautilus::val<uint64_t>(NOT_FOUND);
    }

    if (minFrequencyDirty)
    {
        recomputeMinFrequencyIndex();
    }
    else
    {
        minFrequencyIndex = replacementPos;
        minFrequencyDirty = false;
    }
    return nautilus::val<uint64_t>(NOT_FOUND);
}

nautilus::val<std::byte*>
PredictionCacheLFU::getDataStructureRef(const nautilus::val<std::byte*>& record, const PredictionCache::PredictionCacheReplacement& replacementFunction)
{
    /// First, we check if the timestamp is already in the cache.
    if (const auto dataStructurePos = PredictionCache::searchInCache(record); dataStructurePos != PredictionCache::NOT_FOUND)
    {
        incrementNumberOfHits();
        auto frequency = getFrequency(dataStructurePos);
        const auto newFrequency = nautilus::val<uint64_t>(*frequency) + nautilus::val<uint64_t>(1);
        *frequency = newFrequency;
        if (dataStructurePos == minFrequencyIndex)
        {
            minFrequencyDirty = true;
        }
        return getDataStructure(dataStructurePos);
    }

    /// Second, if this is not the case, we replace the current LFU entry.
    incrementNumberOfMisses();
    const auto replacementPos = getReplacementPos();

    /// Third, we have to replace the entry at replacementPos.
    const nautilus::val<PredictionCacheEntry*> PredictionCacheEntryToReplace = startOfEntries + replacementPos * sizeOfEntry;
    const auto dataStructure = replacementFunction(PredictionCacheEntryToReplace, replacementPos);
    addLookupIndexEntry(record, replacementPos);
    replacementIndex = replacementPos;
    *getFrequency(replacementPos) = 1;
    if (nextEmptyPos < numberOfEntries)
    {
        return dataStructure;
    }

    if (minFrequencyDirty)
    {
        recomputeMinFrequencyIndex();
    }
    else
    {
        minFrequencyIndex = replacementPos;
        minFrequencyDirty = false;
    }
    return dataStructure;
}

nautilus::val<uint64_t*> PredictionCacheLFU::getFrequency(const nautilus::val<uint64_t>& pos)
{
    const auto PredictionCacheEntry = startOfEntries + pos * sizeOfEntry;
    const auto frequencyRef = getMemberRef(PredictionCacheEntry, &PredictionCacheEntryLFU::frequency);
    return frequencyRef;
}

}
