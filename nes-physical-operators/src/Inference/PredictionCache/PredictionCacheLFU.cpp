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
    : PredictionCache(numberOfEntries, sizeOfEntry, startOfEntries, hitsRef, missesRef, inputSize), nextEmptyPos(0), minFrequency(NOT_FOUND)
{
    initializeBuckets();
}

nautilus::val<uint64_t*> PredictionCacheLFU::getFrequency(const nautilus::val<uint64_t>& pos)
{
    return getMemberRef(startOfEntries + pos * sizeOfEntry, &PredictionCacheEntryLFU::frequency);
}

nautilus::val<uint64_t*> PredictionCacheLFU::getPreviousPos(const nautilus::val<uint64_t>& pos)
{
    return getMemberRef(startOfEntries + pos * sizeOfEntry, &PredictionCacheEntryLFU::previousPos);
}

nautilus::val<uint64_t*> PredictionCacheLFU::getNextPos(const nautilus::val<uint64_t>& pos)
{
    return getMemberRef(startOfEntries + pos * sizeOfEntry, &PredictionCacheEntryLFU::nextPos);
}

nautilus::val<uint64_t*> PredictionCacheLFU::getBucketHeadByIndex(const nautilus::val<uint64_t>& bucketIndex)
{
    return getMemberRef(startOfEntries + bucketIndex * sizeOfEntry, &PredictionCacheEntryLFU::bucketHead);
}

nautilus::val<uint64_t*> PredictionCacheLFU::getBucketTailByIndex(const nautilus::val<uint64_t>& bucketIndex)
{
    return getMemberRef(startOfEntries + bucketIndex * sizeOfEntry, &PredictionCacheEntryLFU::bucketTail);
}

nautilus::val<uint64_t> PredictionCacheLFU::getBucketIndex(const nautilus::val<uint64_t>& frequency)
{
    if (frequency < numberOfEntries)
    {
        return frequency - nautilus::val<uint64_t>(1);
    }
    return numberOfEntries - nautilus::val<uint64_t>(1);
}

void PredictionCacheLFU::initializeBuckets()
{
    nextEmptyPos = 0;
    minFrequency = NOT_FOUND;

    for (nautilus::val<uint64_t> i = 0; i < numberOfEntries; ++i)
    {
        *getPreviousPos(i) = NOT_FOUND;
        *getNextPos(i) = NOT_FOUND;
        *getBucketHeadByIndex(i) = NOT_FOUND;
        *getBucketTailByIndex(i) = NOT_FOUND;
    }

    for (nautilus::val<uint64_t> i = 0; i < numberOfEntries; ++i)
    {
        if (getRecord(i) != nullptr)
        {
            nautilus::val<uint64_t> frequency{*getFrequency(i)};
            if (frequency == 0)
            {
                frequency = 1;
                *getFrequency(i) = frequency;
            }
            appendToBucket(i, frequency);
            nextEmptyPos = nextEmptyPos + 1;
            if (minFrequency == NOT_FOUND || frequency < minFrequency)
            {
                minFrequency = frequency;
            }
        }
    }
}

void PredictionCacheLFU::appendToBucket(const nautilus::val<uint64_t>& pos, const nautilus::val<uint64_t>& frequency)
{
    const auto bucketIndex = getBucketIndex(frequency);
    auto bucketHead = getBucketHeadByIndex(bucketIndex);
    auto bucketTail = getBucketTailByIndex(bucketIndex);
    nautilus::val<uint64_t> tail{*bucketTail};

    *getPreviousPos(pos) = tail;
    *getNextPos(pos) = NOT_FOUND;
    if (tail != NOT_FOUND)
    {
        *getNextPos(tail) = pos;
    }
    else
    {
        *bucketHead = pos;
    }
    *bucketTail = pos;
}

void PredictionCacheLFU::removeFromBucket(const nautilus::val<uint64_t>& pos)
{
    nautilus::val<uint64_t> frequency{*getFrequency(pos)};
    const auto bucketIndex = getBucketIndex(frequency);
    auto bucketHead = getBucketHeadByIndex(bucketIndex);
    auto bucketTail = getBucketTailByIndex(bucketIndex);
    nautilus::val<uint64_t> previousPos{*getPreviousPos(pos)};
    nautilus::val<uint64_t> nextPos{*getNextPos(pos)};

    if (previousPos != NOT_FOUND)
    {
        *getNextPos(previousPos) = nextPos;
    }
    else
    {
        *bucketHead = nextPos;
    }
    if (nextPos != NOT_FOUND)
    {
        *getPreviousPos(nextPos) = previousPos;
    }
    else
    {
        *bucketTail = previousPos;
    }

    *getPreviousPos(pos) = NOT_FOUND;
    *getNextPos(pos) = NOT_FOUND;
}

void PredictionCacheLFU::touch(const nautilus::val<uint64_t>& pos)
{
    nautilus::val<uint64_t> oldFrequency{*getFrequency(pos)};
    if (oldFrequency >= numberOfEntries)
    {
        return;
    }

    auto newFrequency = oldFrequency;
    newFrequency = oldFrequency + nautilus::val<uint64_t>(1);

    removeFromBucket(pos);
    *getFrequency(pos) = newFrequency;
    appendToBucket(pos, newFrequency);

    if (oldFrequency == minFrequency && *getBucketHeadByIndex(getBucketIndex(oldFrequency)) == NOT_FOUND)
    {
        minFrequency = newFrequency;
    }
}

void PredictionCacheLFU::insertWithFrequencyOne(const nautilus::val<uint64_t>& pos)
{
    if (nautilus::val<uint64_t>(*getFrequency(pos)) != 0)
    {
        removeFromBucket(pos);
    }

    *getFrequency(pos) = 1;
    appendToBucket(pos, 1);
    minFrequency = 1;
}

nautilus::val<uint64_t> PredictionCacheLFU::getReplacementPos()
{
    if (nextEmptyPos < numberOfEntries)
    {
        const auto replacementPos = nextEmptyPos;
        nextEmptyPos = nextEmptyPos + 1;
        return replacementPos;
    }
    return *getBucketHeadByIndex(getBucketIndex(minFrequency));
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
        touch(dataStructurePos);
        return dataStructurePos;
    }

    incrementNumberOfMisses();
    const auto replacementPos = getReplacementPos();
    updateFunction(startOfEntries + replacementPos * sizeOfEntry, replacementPos);
    replacementIndex = replacementPos;
    insertWithFrequencyOne(replacementPos);
    return nautilus::val<uint64_t>(NOT_FOUND);
}

nautilus::val<std::byte*>
PredictionCacheLFU::getDataStructureRef(const nautilus::val<std::byte*>& record, const PredictionCacheReplacement& replacementFunction)
{
    if (const auto dataStructurePos = searchInCache(record); dataStructurePos != NOT_FOUND)
    {
        incrementNumberOfHits();
        touch(dataStructurePos);
        return getDataStructure(dataStructurePos);
    }

    incrementNumberOfMisses();
    const auto replacementPos = getReplacementPos();
    const auto dataStructure = replacementFunction(startOfEntries + replacementPos * sizeOfEntry, replacementPos);
    addLookupIndexEntry(record, replacementPos);
    replacementIndex = replacementPos;
    insertWithFrequencyOne(replacementPos);
    return dataStructure;
}

}
