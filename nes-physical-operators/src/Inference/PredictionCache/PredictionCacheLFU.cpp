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

#include <Inference/PredictionCache/PredictionCacheLFU.hpp>

#include <cstdint>

#include <Nautilus/DataTypes/DataTypesUtil.hpp>
#include <val_arith.hpp>

namespace NES
{

PredictionCacheLFU::PredictionCacheLFU(
    uint64_t numberOfEntries,
    uint64_t sizeOfEntry,
    nautilus::val<int8_t*> startOfEntries,
    nautilus::val<uint64_t*> hitsRef,
    nautilus::val<uint64_t*> missesRef,
    nautilus::val<size_t> inputSize)
    : PredictionCache(numberOfEntries, sizeOfEntry, startOfEntries, hitsRef, missesRef, inputSize)
    , nextEmptyPos(0)
    , minFrequencyIndex(0)
    , minFrequencyDirty(true)
{
}

nautilus::val<uint64_t*> PredictionCacheLFU::getFrequency(const nautilus::val<uint64_t>& pos)
{
    return getMemberRef(startOfEntries + pos * sizeOfEntry, &PredictionCacheEntryLFU::frequency);
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

void PredictionCacheLFU::updateValues(const nautilus::val<uint64_t>& pos, const PredictionCacheUpdate& updateFunction)
{
    updateFunction(startOfEntries + pos * sizeOfEntry, pos);
    addLookupIndexEntry(getRecord(pos), pos);
}

nautilus::val<uint64_t> PredictionCacheLFU::updateKeys(const nautilus::val<std::byte*>& record, const PredictionCacheUpdate& updateFunction)
{
    if (const auto dataStructurePos = searchInCache(record); dataStructurePos != NOT_FOUND)
    {
        incrementNumberOfHits();
        auto frequency = getFrequency(dataStructurePos);
        *frequency = nautilus::val<uint64_t>(*frequency) + nautilus::val<uint64_t>(1);
        if (dataStructurePos == minFrequencyIndex)
        {
            minFrequencyDirty = true;
        }
        return dataStructurePos;
    }

    incrementNumberOfMisses();
    const auto replacementPos = getReplacementPos();
    updateFunction(startOfEntries + replacementPos * sizeOfEntry, replacementPos);
    replacementIndex = replacementPos;
    *getFrequency(replacementPos) = 1;
    if (nextEmptyPos >= numberOfEntries)
    {
        minFrequencyIndex = replacementPos;
        minFrequencyDirty = false;
    }
    return nautilus::val<uint64_t>(NOT_FOUND);
}

nautilus::val<std::byte*>
PredictionCacheLFU::getDataStructureRef(const nautilus::val<std::byte*>& record, const PredictionCacheReplacement& replacementFunction)
{
    if (const auto dataStructurePos = searchInCache(record); dataStructurePos != NOT_FOUND)
    {
        incrementNumberOfHits();
        auto frequency = getFrequency(dataStructurePos);
        *frequency = nautilus::val<uint64_t>(*frequency) + nautilus::val<uint64_t>(1);
        if (dataStructurePos == minFrequencyIndex)
        {
            minFrequencyDirty = true;
        }
        return getDataStructure(dataStructurePos);
    }

    incrementNumberOfMisses();
    const auto replacementPos = getReplacementPos();
    const auto dataStructure = replacementFunction(startOfEntries + replacementPos * sizeOfEntry, replacementPos);
    addLookupIndexEntry(record, replacementPos);
    replacementIndex = replacementPos;
    *getFrequency(replacementPos) = 1;
    if (nextEmptyPos >= numberOfEntries)
    {
        minFrequencyIndex = replacementPos;
        minFrequencyDirty = false;
    }
    return dataStructure;
}

}
