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

#include <Inference/PredictionCache/PredictionCacheUtil.hpp>

#include <cstdint>
#include <memory>

#include <Inference/PredictionCache/PredictionCacheFIFO.hpp>
#include <Inference/PredictionCache/PredictionCacheLFU.hpp>
#include <Inference/PredictionCache/PredictionCacheLRU.hpp>
#include <Inference/PredictionCache/PredictionCacheSecondChance.hpp>
#include <ErrorHandling.hpp>

namespace NES::Util
{

uint64_t getPredictionCacheEntrySize(const PredictionCacheType predictionCacheType)
{
    switch (predictionCacheType)
    {
        case PredictionCacheType::NONE:
            return 0;
        case PredictionCacheType::FIFO:
            return sizeof(PredictionCacheEntryFIFO);
        case PredictionCacheType::LFU:
            return sizeof(PredictionCacheEntryLFU);
        case PredictionCacheType::LRU:
            return sizeof(PredictionCacheEntryLRU);
        case PredictionCacheType::SECOND_CHANCE:
            return sizeof(PredictionCacheEntrySecondChance);
    }
    std::unreachable();
}

std::unique_ptr<PredictionCache> createPredictionCache(
    const PredictionCacheType predictionCacheType,
    const uint64_t numberOfEntries,
    nautilus::val<int8_t*> startOfEntries,
    nautilus::val<size_t> inputSize)
{
    const auto hitsRef = startOfEntries;
    const auto missesRef = hitsRef + nautilus::val<uint64_t>(sizeof(uint64_t));
    const auto predictionCacheEntries = startOfEntries + nautilus::val<uint64_t>(sizeof(HitsAndMisses));

    switch (predictionCacheType)
    {
        case PredictionCacheType::NONE:
            throw InvalidConfigParameter("PredictionCacheType is NONE");
        case PredictionCacheType::FIFO:
            return std::make_unique<PredictionCacheFIFO>(
                numberOfEntries, sizeof(PredictionCacheEntryFIFO), predictionCacheEntries, hitsRef, missesRef, inputSize);
        case PredictionCacheType::LFU:
            return std::make_unique<PredictionCacheLFU>(
                numberOfEntries, sizeof(PredictionCacheEntryLFU), predictionCacheEntries, hitsRef, missesRef, inputSize);
        case PredictionCacheType::LRU:
            return std::make_unique<PredictionCacheLRU>(
                numberOfEntries, sizeof(PredictionCacheEntryLRU), predictionCacheEntries, hitsRef, missesRef, inputSize);
        case PredictionCacheType::SECOND_CHANCE:
            return std::make_unique<PredictionCacheSecondChance>(
                numberOfEntries, sizeof(PredictionCacheEntrySecondChance), predictionCacheEntries, hitsRef, missesRef, inputSize);
    }
    std::unreachable();
}

}
