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

#include <Inference/PredictionCache/PredictionCache.hpp>

#include <cstddef>
#include <cstdint>

#include <Nautilus/DataTypes/DataTypesUtil.hpp>
#include <val_arith.hpp>

namespace NES
{

PredictionCache::PredictionCache(
    const uint64_t numberOfEntries,
    const uint64_t sizeOfEntry,
    nautilus::val<int8_t*> startOfEntries,
    nautilus::val<uint64_t*> hitsRef,
    nautilus::val<uint64_t*> missesRef,
    nautilus::val<size_t> inputSize)
    : startOfEntries(startOfEntries)
    , numberOfEntries(numberOfEntries)
    , sizeOfEntry(sizeOfEntry)
    , numberOfHits(hitsRef)
    , numberOfMisses(missesRef)
    , inputSize(inputSize)
    , lookupIndex(nullptr)
    , lookupIndexBufferProvider(nullptr)
{
}

void PredictionCache::incrementNumberOfHits()
{
    auto currentNumberOfHits = static_cast<nautilus::val<uint64_t>>(*numberOfHits);
    currentNumberOfHits = currentNumberOfHits + 1;
    *numberOfHits = currentNumberOfHits;
}

void PredictionCache::incrementNumberOfMisses()
{
    auto currentNumberOfMisses = static_cast<nautilus::val<uint64_t>>(*numberOfMisses);
    currentNumberOfMisses = currentNumberOfMisses + 1;
    *numberOfMisses = currentNumberOfMisses;
}

nautilus::val<std::byte*> PredictionCache::getRecord(const nautilus::val<uint64_t>& pos)
{
    const auto predictionCacheEntry = startOfEntries + pos * sizeOfEntry;
    const auto recordRef = getMemberRef(predictionCacheEntry, &PredictionCacheEntry::record);
    return *getMemberWithOffset<std::byte*>(recordRef, 0);
}

nautilus::val<std::byte*> PredictionCache::getDataStructure(const nautilus::val<uint64_t>& pos)
{
    const auto predictionCacheEntry = startOfEntries + pos * sizeOfEntry;
    const auto dataStructureRef = getMemberRef(predictionCacheEntry, &PredictionCacheEntry::dataStructure);
    return *getMemberWithOffset<std::byte*>(dataStructureRef, 0);
}

nautilus::val<bool> PredictionCache::foundRecord(const nautilus::val<uint64_t>& pos, const nautilus::val<std::byte*>& candidateRecord)
{
    const auto cacheRecord = getRecord(pos);
    if (cacheRecord != nautilus::val<std::byte*>(nullptr))
    {
        const auto candidateBytes = static_cast<nautilus::val<int8_t*>>(candidateRecord);
        const auto cacheBytes = static_cast<nautilus::val<int8_t*>>(cacheRecord);
        for (nautilus::val<size_t> i = 0; i < inputSize; ++i)
        {
            if (*(candidateBytes + i) != *(cacheBytes + i))
            {
                return false;
            }
        }
        return true;
    }
    return false;
}

void PredictionCache::configureLookupIndex(
    nautilus::val<ChainedHashMap*> lookupIndex, nautilus::val<AbstractBufferProvider*> bufferProvider)
{
    this->lookupIndex = lookupIndex;
    this->lookupIndexBufferProvider = bufferProvider;
}

nautilus::val<uint64_t> PredictionCache::searchInCache(const nautilus::val<std::byte*>& record)
{
    for (nautilus::val<uint64_t> i = 0; i < numberOfEntries; ++i)
    {
        if (foundRecord(i, record))
        {
            return i;
        }
    }
    return nautilus::val<uint64_t>(NOT_FOUND);
}

void PredictionCache::addLookupIndexEntry(const nautilus::val<std::byte*>& record, const nautilus::val<uint64_t>& pos)
{
    (void)record;
    (void)pos;
}

void PredictionCache::rebuildLookupIndex()
{
}

nautilus::val<uint64_t*> PredictionCache::getHitsRef() const
{
    return numberOfHits;
}

nautilus::val<uint64_t*> PredictionCache::getMissesRef() const
{
    return numberOfMisses;
}

nautilus::val<uint64_t> PredictionCache::getReplacementIndex() const
{
    return replacementIndex;
}

}
