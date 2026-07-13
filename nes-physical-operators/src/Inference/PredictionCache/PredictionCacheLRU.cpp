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

#include <Inference/PredictionCache/PredictionCacheLRU.hpp>

#include <cstdint>

#include <Nautilus/DataTypes/DataTypesUtil.hpp>
#include <val_arith.hpp>

namespace NES
{

PredictionCacheLRU::PredictionCacheLRU(
    uint64_t numberOfEntries,
    uint64_t sizeOfEntry,
    nautilus::val<int8_t*> startOfEntries,
    nautilus::val<uint64_t*> hitsRef,
    nautilus::val<uint64_t*> missesRef,
    nautilus::val<size_t> inputSize)
    : PredictionCache(numberOfEntries, sizeOfEntry, startOfEntries, hitsRef, missesRef, inputSize)
    , nextEmptyPos(0)
    , lruHead(NOT_FOUND)
    , lruTail(NOT_FOUND)
{
}

nautilus::val<uint64_t*> PredictionCacheLRU::getPreviousPos(const nautilus::val<uint64_t>& pos)
{
    return getMemberRef(startOfEntries + pos * sizeOfEntry, &PredictionCacheEntryLRU::previousPos);
}

nautilus::val<uint64_t*> PredictionCacheLRU::getNextPos(const nautilus::val<uint64_t>& pos)
{
    return getMemberRef(startOfEntries + pos * sizeOfEntry, &PredictionCacheEntryLRU::nextPos);
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

nautilus::val<uint64_t> PredictionCacheLRU::claimReplacementPos()
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

nautilus::val<uint64_t> PredictionCacheLRU::getReplacementPos()
{
    /// The recency order persists in the entries' previous/next pointers; there is no scalar state to save, and saving must not
    /// mutate the list (this is called once per task close).
    return nautilus::val<uint64_t>(0);
}

void PredictionCacheLRU::setReplacementPos(nautilus::val<uint64_t>)
{
    /// Cache entries outlive this wrapper, but the trace-local list anchors do not. The recency order itself persists in the
    /// entries' previous/next pointers, so recover the anchors from them instead of rebuilding the list in slot order.
    /// Occupied slots always form a prefix, as claimReplacementPos fills empty slots in ascending order.
    nextEmptyPos = numberOfEntries;
    lruHead = NOT_FOUND;
    lruTail = NOT_FOUND;

    for (nautilus::val<uint64_t> pos = 0; pos < numberOfEntries; pos = pos + 1)
    {
        if (getRecord(pos) == nullptr)
        {
            if (pos < nextEmptyPos)
            {
                nextEmptyPos = pos;
            }
        }
        else
        {
            if (*getPreviousPos(pos) == NOT_FOUND)
            {
                lruHead = pos;
            }
            if (*getNextPos(pos) == NOT_FOUND)
            {
                lruTail = pos;
            }
        }
    }
}

void PredictionCacheLRU::updateValues(const nautilus::val<uint64_t>& pos, const PredictionCacheUpdate& updateFunction)
{
    updateFunction(startOfEntries + pos * sizeOfEntry, pos);
    addLookupIndexEntry(getRecord(pos), pos);
}

nautilus::val<uint64_t> PredictionCacheLRU::updateKeys(const nautilus::val<std::byte*>& record, const PredictionCacheUpdate& updateFunction)
{
    if (const auto dataStructurePos = searchInCache(record); dataStructurePos != NOT_FOUND)
    {
        incrementNumberOfHits();
        touch(dataStructurePos);
        return dataStructurePos;
    }
    incrementNumberOfMisses();
    const auto replacementPos = claimReplacementPos();
    updateFunction(startOfEntries + replacementPos * sizeOfEntry, replacementPos);
    replacementIndex = replacementPos;
    return nautilus::val<uint64_t>(NOT_FOUND);
}

nautilus::val<std::byte*>
PredictionCacheLRU::getDataStructureRef(const nautilus::val<std::byte*>& record, const PredictionCacheReplacement& replacementFunction)
{
    if (const auto dataStructurePos = searchInCache(record); dataStructurePos != NOT_FOUND)
    {
        incrementNumberOfHits();
        touch(dataStructurePos);
        return getDataStructure(dataStructurePos);
    }
    incrementNumberOfMisses();
    const auto replacementPos = claimReplacementPos();
    const auto dataStructure = replacementFunction(startOfEntries + replacementPos * sizeOfEntry, replacementPos);
    addLookupIndexEntry(record, replacementPos);
    replacementIndex = replacementPos;
    return dataStructure;
}

}
