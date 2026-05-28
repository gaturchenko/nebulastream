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

#include <Inference/PredictionCache/PredictionCacheSecondChance.hpp>

#include <cstdint>

#include <Nautilus/DataTypes/DataTypesUtil.hpp>
#include <val_arith.hpp>

namespace NES
{

PredictionCacheSecondChance::PredictionCacheSecondChance(
    uint64_t numberOfEntries,
    uint64_t sizeOfEntry,
    nautilus::val<int8_t*> startOfEntries,
    nautilus::val<uint64_t*> hitsRef,
    nautilus::val<uint64_t*> missesRef,
    nautilus::val<size_t> inputSize)
    : PredictionCacheFIFO(numberOfEntries, sizeOfEntry, startOfEntries, hitsRef, missesRef, inputSize)
{
}

nautilus::val<uint64_t> PredictionCacheSecondChance::getReplacementPos()
{
    return localReplacementIndex;
}

void PredictionCacheSecondChance::setReplacementPos(nautilus::val<uint64_t> pos)
{
    localReplacementIndex = pos;
}

nautilus::val<bool*> PredictionCacheSecondChance::getSecondChanceBit(const nautilus::val<uint64_t>& pos)
{
    return getMemberRef(startOfEntries + pos * sizeOfEntry, &PredictionCacheEntrySecondChance::secondChanceBit);
}

void PredictionCacheSecondChance::updateValues(const nautilus::val<uint64_t>& pos, const PredictionCacheUpdate& updateFunction)
{
    updateFunction(startOfEntries + pos * sizeOfEntry, pos);
    addLookupIndexEntry(getRecord(pos), pos);
}

nautilus::val<uint64_t>
PredictionCacheSecondChance::updateKeys(const nautilus::val<std::byte*>& record, const PredictionCacheUpdate& updateFunction)
{
    if (const auto dataStructurePos = searchInCache(record); dataStructurePos != NOT_FOUND)
    {
        incrementNumberOfHits();
        *getSecondChanceBit(dataStructurePos) = true;
        return dataStructurePos;
    }

    incrementNumberOfMisses();
    auto secondChanceBit = getSecondChanceBit(localReplacementIndex);
    while (*secondChanceBit == true)
    {
        *secondChanceBit = false;
        localReplacementIndex = (localReplacementIndex + 1) % numberOfEntries;
        secondChanceBit = getSecondChanceBit(localReplacementIndex);
    }

    updateFunction(startOfEntries + localReplacementIndex * sizeOfEntry, localReplacementIndex);
    replacementIndex = localReplacementIndex;
    *secondChanceBit = true;
    return nautilus::val<uint64_t>(NOT_FOUND);
}

nautilus::val<std::byte*> PredictionCacheSecondChance::getDataStructureRef(
    const nautilus::val<std::byte*>& record, const PredictionCacheReplacement& replacementFunction)
{
    if (const auto dataStructurePos = searchInCache(record); dataStructurePos != NOT_FOUND)
    {
        incrementNumberOfHits();
        *getSecondChanceBit(dataStructurePos) = true;
        return getDataStructure(dataStructurePos);
    }

    incrementNumberOfMisses();
    auto secondChanceBit = getSecondChanceBit(localReplacementIndex);
    while (*secondChanceBit == true)
    {
        *secondChanceBit = false;
        localReplacementIndex = (localReplacementIndex + 1) % numberOfEntries;
        secondChanceBit = getSecondChanceBit(localReplacementIndex);
    }

    const auto dataStructure = replacementFunction(startOfEntries + localReplacementIndex * sizeOfEntry, localReplacementIndex);
    addLookupIndexEntry(record, localReplacementIndex);
    replacementIndex = localReplacementIndex;
    *secondChanceBit = true;
    return dataStructure;
}

}
