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

#include <Inference/PredictionCache/PredictionCacheFIFO.hpp>

#include <cstdint>

#include <val_arith.hpp>

namespace NES
{

PredictionCacheFIFO::PredictionCacheFIFO(
    uint64_t numberOfEntries,
    uint64_t sizeOfEntry,
    nautilus::val<int8_t*> startOfEntries,
    nautilus::val<uint64_t*> hitsRef,
    nautilus::val<uint64_t*> missesRef,
    nautilus::val<size_t> inputSize)
    : PredictionCache(numberOfEntries, sizeOfEntry, startOfEntries, hitsRef, missesRef, inputSize), localReplacementIndex(0)
{
}

nautilus::val<uint64_t> PredictionCacheFIFO::getReplacementPos()
{
    return localReplacementIndex;
}

void PredictionCacheFIFO::setReplacementPos(nautilus::val<uint64_t> pos)
{
    localReplacementIndex = pos;
}

void PredictionCacheFIFO::updateValues(const nautilus::val<uint64_t>& pos, const PredictionCacheUpdate& updateFunction)
{
    updateFunction(startOfEntries + pos * sizeOfEntry, pos);
    addLookupIndexEntry(getRecord(pos), pos);
}

nautilus::val<uint64_t>
PredictionCacheFIFO::updateKeys(const nautilus::val<std::byte*>& record, const PredictionCacheUpdate& updateFunction)
{
    if (const auto dataStructurePos = searchInCache(record); dataStructurePos != NOT_FOUND)
    {
        incrementNumberOfHits();
        return dataStructurePos;
    }

    incrementNumberOfMisses();
    updateFunction(startOfEntries + localReplacementIndex * sizeOfEntry, localReplacementIndex);
    replacementIndex = localReplacementIndex;
    localReplacementIndex = (localReplacementIndex + 1) % numberOfEntries;
    return nautilus::val<uint64_t>(NOT_FOUND);
}

nautilus::val<std::byte*>
PredictionCacheFIFO::getDataStructureRef(const nautilus::val<std::byte*>& record, const PredictionCacheReplacement& replacementFunction)
{
    if (const auto dataStructurePos = searchInCache(record); dataStructurePos != NOT_FOUND)
    {
        incrementNumberOfHits();
        return getDataStructure(dataStructurePos);
    }

    incrementNumberOfMisses();
    const auto dataStructure = replacementFunction(startOfEntries + localReplacementIndex * sizeOfEntry, localReplacementIndex);
    addLookupIndexEntry(record, localReplacementIndex);
    replacementIndex = localReplacementIndex;
    localReplacementIndex = (localReplacementIndex + 1) % numberOfEntries;
    return dataStructure;
}

}
