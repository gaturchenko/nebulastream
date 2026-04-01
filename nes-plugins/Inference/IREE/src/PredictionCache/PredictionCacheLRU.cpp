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
    , nextEmptyPos(0)
    , lruHead(NOT_FOUND)
    , lruTail(NOT_FOUND)
{
}

nautilus::val<uint64_t> PredictionCacheLRU::getReplacementPos()
{
    if (nextEmptyPos < numberOfEntries)
    {
        const auto replacementPos = nextEmptyPos;
        nextEmptyPos = nextEmptyPos + 1;
        appendToTail(replacementPos);
        return replacementPos;
    }

    const auto replacementPos = lruHead;
    removeFromList(replacementPos);
    appendToTail(replacementPos);
    return replacementPos;
}

nautilus::val<uint64_t*> PredictionCacheLRU::getPreviousPos(const nautilus::val<uint64_t>& pos)
{
    const auto predictionCacheEntry = startOfEntries + pos * sizeOfEntry;
    const auto previousPosRef = getMemberRef(predictionCacheEntry, &PredictionCacheEntryLRU::previousPos);
    return previousPosRef;
}

nautilus::val<uint64_t*> PredictionCacheLRU::getNextPos(const nautilus::val<uint64_t>& pos)
{
    const auto predictionCacheEntry = startOfEntries + pos * sizeOfEntry;
    const auto nextPosRef = getMemberRef(predictionCacheEntry, &PredictionCacheEntryLRU::nextPos);
    return nextPosRef;
}

void PredictionCacheLRU::appendToTail(const nautilus::val<uint64_t>& pos)
{
    *getPreviousPos(pos) = lruTail;
    *getNextPos(pos) = NOT_FOUND;

    if (lruTail != NOT_FOUND)
    {
        *getNextPos(lruTail) = pos;
    }
    else
    {
        lruHead = pos;
    }
    lruTail = pos;
}

void PredictionCacheLRU::removeFromList(const nautilus::val<uint64_t>& pos)
{
    nautilus::val<uint64_t> previousPos{*getPreviousPos(pos)};
    nautilus::val<uint64_t> nextPos{*getNextPos(pos)};

    if (previousPos != NOT_FOUND)
    {
        *getNextPos(previousPos) = nextPos;
    }
    else
    {
        lruHead = nextPos;
    }

    if (nextPos != NOT_FOUND)
    {
        *getPreviousPos(nextPos) = previousPos;
    }
    else
    {
        lruTail = previousPos;
    }

    *getPreviousPos(pos) = NOT_FOUND;
    *getNextPos(pos) = NOT_FOUND;
}

void PredictionCacheLRU::touch(const nautilus::val<uint64_t>& pos)
{
    if (pos == lruTail)
    {
        return;
    }
    removeFromList(pos);
    appendToTail(pos);
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
        touch(dataStructurePos);
        return dataStructurePos;
    }

    /// If the record is not in the cache, we have a cache miss.
    incrementNumberOfMisses();

    /// Second, we have to replace the least-recently-used entry.
    const auto replacementPos = getReplacementPos();
    const nautilus::val<PredictionCacheEntry*> PredictionCacheEntryToReplace = startOfEntries + replacementPos * sizeOfEntry;
    updateFunction(PredictionCacheEntryToReplace, replacementPos);
    addLookupIndexEntry(record, replacementPos);
    replacementIndex = replacementPos;
    return nautilus::val<uint64_t>(NOT_FOUND);
}

nautilus::val<std::byte*>
PredictionCacheLRU::getDataStructureRef(const nautilus::val<std::byte*>& record, const PredictionCache::PredictionCacheReplacement& replacementFunction)
{
    /// First, we check if the record is already in the cache. If this is the case, we update its access timestamp.
    if (const auto dataStructurePos = PredictionCache::searchInCache(record); dataStructurePos != PredictionCache::NOT_FOUND)
    {
        incrementNumberOfHits();
        touch(dataStructurePos);
        return getDataStructure(dataStructurePos);
    }

    /// If the record is not in the cache, we have a cache miss.
    incrementNumberOfMisses();

    /// Second, we have to replace the least-recently-used entry.
    const auto replacementPos = getReplacementPos();
    const nautilus::val<PredictionCacheEntry*> PredictionCacheEntryToReplace = startOfEntries + replacementPos * sizeOfEntry;
    const auto dataStructure = replacementFunction(PredictionCacheEntryToReplace, replacementPos);
    addLookupIndexEntry(record, replacementPos);
    replacementIndex = replacementPos;
    return dataStructure;
}
}
