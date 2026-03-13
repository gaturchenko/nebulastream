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

#include <PredictionCache/PredictionCacheAlwaysMiss.hpp>

namespace NES
{
PredictionCacheAlwaysMiss::PredictionCacheAlwaysMiss(
    const nautilus::val<OperatorHandler*>& operatorHandler,
    const uint64_t numberOfEntries,
    const uint64_t sizeOfEntry,
    const nautilus::val<int8_t*>& startOfEntries,
    const nautilus::val<uint64_t*>& hitsRef,
    const nautilus::val<uint64_t*>& missesRef,
    const nautilus::val<size_t>& inputSize)
    : PredictionCache(operatorHandler, numberOfEntries, sizeOfEntry, startOfEntries, hitsRef, missesRef, inputSize)
{
}

nautilus::val<uint64_t> PredictionCacheAlwaysMiss::getReplacementPos()
{
    return replacementIndex;
}

void
PredictionCacheAlwaysMiss::updateValues(const nautilus::val<uint64_t>&, const PredictionCache::PredictionCacheUpdate& updateFunction)
{
    const nautilus::val<PredictionCacheEntry*> predictionCacheEntryToReplace = startOfEntries;
    updateFunction(predictionCacheEntryToReplace, 0);
}

nautilus::val<uint64_t> PredictionCacheAlwaysMiss::updateKeys(
    Record&,
    const HashMapOptions&,
    const nautilus::val<HashMap*>&,
    const PredictionCache::PredictionCacheUpdate& updateFunction)
{
    const nautilus::val<PredictionCacheEntry*> predictionCacheEntryToReplace = startOfEntries;
    updateFunction(predictionCacheEntryToReplace, 0);
    return nautilus::val<uint64_t>(0);
}

nautilus::val<std::byte*>
PredictionCacheAlwaysMiss::getDataStructureRef(
    Record&,
    const HashMapOptions&,
    const nautilus::val<HashMap*>&,
    const PredictionCache::PredictionCacheReplacement& replacementFunction)
{
    /// We never check if the slice is already in the cache, thus, we always have a cache miss.
    incrementNumberOfMisses();

    /// As we always have a cache miss, we do not care what index to replace
    const nautilus::val<PredictionCacheEntry*> predictionCacheEntryToReplace = startOfEntries;
    const auto dataStructure = replacementFunction(predictionCacheEntryToReplace, 0);
    return dataStructure;
}

}
