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

#pragma once

#include <Configuration/WorkerConfiguration.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace NES::Microbenchmark
{

enum class HitMissRatio
{
    Hits100,
    Hits75,
    Hits50,
    Hits25,
    Hits0
};

constexpr uint32_t getHitPercentage(const HitMissRatio ratio)
{
    switch (ratio)
    {
        case HitMissRatio::Hits100:
            return 100;
        case HitMissRatio::Hits75:
            return 75;
        case HitMissRatio::Hits50:
            return 50;
        case HitMissRatio::Hits25:
            return 25;
        case HitMissRatio::Hits0:
            return 0;
    }
    return 0;
}

using BenchmarkData = std::vector<std::unique_ptr<std::byte[]>>;

inline std::unique_ptr<std::byte[]> createRecord(const size_t recordSize, const uint64_t id, std::mt19937_64& rng)
{
    auto record = std::make_unique<std::byte[]>(recordSize);
    std::memset(record.get(), 0, recordSize);

    const size_t idBytes = std::min(recordSize, sizeof(id));
    std::memcpy(record.get(), &id, idBytes);

    for (size_t i = idBytes; i < recordSize; ++i)
    {
        record[i] = static_cast<std::byte>(rng() & 0xFF);
    }

    return record;
}

enum class SimCachePolicy : uint8_t
{
    FIFO,
    LFU,
    LRU,
    SECOND_CHANCE
};

struct SimCacheSlot
{
    bool occupied = false;
    uint64_t key = 0;
    uint64_t age = 0;
    uint64_t frequency = 0;
    bool secondChance = false;
};

class SimCache
{
public:
    SimCache(const SimCachePolicy policy, const size_t capacity) : policy(policy), capacity(capacity), slots(capacity) {}

    bool canHit() const { return !residentKeys.empty(); }

    uint64_t pickRandomResidentKey(std::mt19937_64& rng) const
    {
        std::uniform_int_distribution<size_t> dist(0, residentKeys.size() - 1);
        return residentKeys[dist(rng)];
    }

    bool access(const uint64_t key)
    {
        if (capacity == 0)
        {
            return false;
        }

        if (policy == SimCachePolicy::LRU)
        {
            uint64_t maxAge = 0;
            size_t maxAgeIndex = 0;
            for (size_t i = 0; i < capacity; ++i)
            {
                const uint64_t newAge = slots[i].age + 1;
                slots[i].age = newAge;
                if (newAge > maxAge)
                {
                    maxAge = newAge;
                    maxAgeIndex = i;
                }
            }

            const auto it = keyToSlot.find(key);
            if (it != keyToSlot.end())
            {
                slots[it->second].age = 0;
                return true;
            }

            replaceAt(maxAgeIndex, key);
            slots[maxAgeIndex].age = 0;
            return false;
        }

        const auto it = keyToSlot.find(key);
        if (it != keyToSlot.end())
        {
            const size_t idx = it->second;
            switch (policy)
            {
                case SimCachePolicy::FIFO:
                    break;
                case SimCachePolicy::LFU:
                    slots[idx].frequency += 1;
                    break;
                case SimCachePolicy::SECOND_CHANCE:
                    slots[idx].secondChance = true;
                    break;
                case SimCachePolicy::LRU:
                    break;
            }
            return true;
        }

        switch (policy)
        {
            case SimCachePolicy::FIFO:
            {
                const size_t idx = replacementIndex;
                replaceAt(idx, key);
                replacementIndex = (replacementIndex + 1) % capacity;
                return false;
            }
            case SimCachePolicy::LFU:
            {
                uint64_t minFrequency = UINT64_MAX;
                size_t minFrequencyIndex = 0;
                for (size_t i = 0; i < capacity; ++i)
                {
                    if (slots[i].frequency < minFrequency)
                    {
                        minFrequency = slots[i].frequency;
                        minFrequencyIndex = i;
                    }
                }
                replaceAt(minFrequencyIndex, key);
                slots[minFrequencyIndex].frequency = 1;
                return false;
            }
            case SimCachePolicy::SECOND_CHANCE:
            {
                while (slots[replacementIndex].secondChance)
                {
                    slots[replacementIndex].secondChance = false;
                    replacementIndex = (replacementIndex + 1) % capacity;
                }
                replaceAt(replacementIndex, key);
                slots[replacementIndex].secondChance = true;
                return false;
            }
            case SimCachePolicy::LRU:
                break;
        }
        return false;
    }

private:
    void removeResidentKey(const uint64_t key)
    {
        auto slotIt = keyToSlot.find(key);
        if (slotIt != keyToSlot.end())
        {
            keyToSlot.erase(slotIt);
        }

        auto indexIt = keyToResidentIndex.find(key);
        if (indexIt == keyToResidentIndex.end())
        {
            return;
        }
        const size_t index = indexIt->second;
        const size_t lastIndex = residentKeys.size() - 1;
        if (index != lastIndex)
        {
            const uint64_t lastKey = residentKeys[lastIndex];
            residentKeys[index] = lastKey;
            keyToResidentIndex[lastKey] = index;
        }
        residentKeys.pop_back();
        keyToResidentIndex.erase(indexIt);
    }

    void addResidentKey(const uint64_t key)
    {
        keyToResidentIndex.emplace(key, residentKeys.size());
        residentKeys.push_back(key);
    }

    void replaceAt(const size_t index, const uint64_t key)
    {
        if (slots[index].occupied)
        {
            removeResidentKey(slots[index].key);
        }

        slots[index].occupied = true;
        slots[index].key = key;
        keyToSlot[key] = index;
        addResidentKey(key);
    }

    SimCachePolicy policy;
    size_t capacity;
    std::vector<SimCacheSlot> slots;
    size_t replacementIndex = 0;
    std::vector<uint64_t> residentKeys;
    std::unordered_map<uint64_t, size_t> keyToSlot;
    std::unordered_map<uint64_t, size_t> keyToResidentIndex;
};

inline BenchmarkData createDeterministicBenchmarkData(
    const uint64_t cacheSize,
    const size_t totalRecords,
    const HitMissRatio ratio,
    const NES::Configurations::PredictionCacheType predictionCacheType,
    const size_t recordSize,
    const uint64_t seed)
{
    if (totalRecords == 0)
    {
        return {};
    }

    if (recordSize == 0)
    {
        throw std::invalid_argument("recordSize must be greater than 0");
    }

    const uint32_t hitPercentage = getHitPercentage(ratio);
    if (hitPercentage > 0 && cacheSize == 0)
    {
        throw std::invalid_argument("cacheSize must be greater than 0 when hits are requested");
    }

    size_t numHits = static_cast<size_t>((totalRecords * hitPercentage) / 100);
    size_t numMisses = totalRecords - numHits;

    if (numMisses == 0 && totalRecords > 0)
    {
        /// Cold cache cannot yield a 100% hit rate; force one miss to seed residency.
        numMisses = 1;
        numHits = totalRecords - 1;
    }

    const auto resolveSimCachePolicy = [](const NES::Configurations::PredictionCacheType type)
    {
        switch (type)
        {
            case NES::Configurations::PredictionCacheType::FIFO:
                return SimCachePolicy::FIFO;
            case NES::Configurations::PredictionCacheType::LFU:
                return SimCachePolicy::LFU;
            case NES::Configurations::PredictionCacheType::LRU:
                return SimCachePolicy::LRU;
            case NES::Configurations::PredictionCacheType::SECOND_CHANCE:
                return SimCachePolicy::SECOND_CHANCE;
            case NES::Configurations::PredictionCacheType::ALWAYS_MISS:
            case NES::Configurations::PredictionCacheType::NONE:
            case NES::Configurations::PredictionCacheType::TWO_QUEUES:
                throw std::invalid_argument("PredictionCacheType is invalid for benchmark data generation");
        }
        return SimCachePolicy::FIFO;
    };

    const SimCachePolicy simPolicy = resolveSimCachePolicy(predictionCacheType);

    std::mt19937_64 rngPattern(seed);
    std::mt19937_64 rngData(seed ^ 0x9E3779B97F4A7C15ULL);

    SimCache simCache(simPolicy, static_cast<size_t>(cacheSize));
    std::vector<uint64_t> accessKeys;
    accessKeys.reserve(totalRecords);

    size_t remainingHits = numHits;
    size_t remainingMisses = numMisses;
    std::uniform_real_distribution<double> pickRatio(0.0, 1.0);
    uint64_t nextMissKey = 0;

    for (size_t i = 0; i < totalRecords; ++i)
    {
        const size_t remaining = totalRecords - i;
        const bool canHit = simCache.canHit();

        bool chooseHit = false;
        if (remainingHits == 0)
        {
            chooseHit = false;
        }
        else if (remainingMisses == 0)
        {
            chooseHit = canHit;
        }
        else if (!canHit)
        {
            chooseHit = false;
        }
        else
        {
            const double hitProbability = static_cast<double>(remainingHits) / static_cast<double>(remaining);
            chooseHit = pickRatio(rngPattern) < hitProbability;
        }

        uint64_t key = 0;
        if (chooseHit)
        {
            key = simCache.pickRandomResidentKey(rngPattern);
            remainingHits--;
        }
        else
        {
            key = nextMissKey++;
            remainingMisses--;
        }

        const bool observedHit = simCache.access(key);
        (void)observedHit;
        accessKeys.emplace_back(key);
    }

    BenchmarkData records;
    records.reserve(totalRecords);

    std::vector<std::unique_ptr<std::byte[]>> keyTemplates;
    keyTemplates.reserve(static_cast<size_t>(nextMissKey));

    for (const uint64_t key : accessKeys)
    {
        const size_t keyIndex = static_cast<size_t>(key);
        while (keyTemplates.size() <= keyIndex)
        {
            const uint64_t newKey = keyTemplates.size();
            keyTemplates.emplace_back(createRecord(recordSize, newKey, rngData));
        }

        auto record = std::make_unique<std::byte[]>(recordSize);
        std::memcpy(record.get(), keyTemplates[keyIndex].get(), recordSize);
        records.emplace_back(std::move(record));
    }

    return records;
}

inline std::vector<double> createZipfPopularity(const size_t numKeys, const double s)
{
    if (numKeys == 0)
    {
        throw std::invalid_argument("numKeys must be greater than 0");
    }

    std::vector<double> weights;
    weights.reserve(numKeys);
    for (size_t rank = 1; rank <= numKeys; ++rank)
    {
        weights.emplace_back(1.0 / std::pow(static_cast<double>(rank), s));
    }

    const double weightSum = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (double& weight : weights)
    {
        weight /= weightSum;
    }

    return weights;
}

inline BenchmarkData createZipfBenchmarkData(
    const size_t numKeys,
    const size_t totalRecords,
    const size_t recordSize,
    const uint64_t seed,
    const double s = 1.0)
{
    if (totalRecords == 0)
    {
        return {};
    }

    if (recordSize == 0)
    {
        throw std::invalid_argument("recordSize must be greater than 0");
    }

    const auto zipfPopularity = createZipfPopularity(numKeys, s);

    std::mt19937_64 rngPattern(seed);
    std::mt19937_64 rngData(seed ^ 0x9E3779B97F4A7C15ULL);
    std::discrete_distribution<size_t> keyDistribution(zipfPopularity.begin(), zipfPopularity.end());

    std::vector<std::unique_ptr<std::byte[]>> keyTemplates;
    keyTemplates.reserve(numKeys);
    for (size_t key = 0; key < numKeys; ++key)
    {
        keyTemplates.emplace_back(createRecord(recordSize, static_cast<uint64_t>(key), rngData));
    }

    BenchmarkData records;
    records.reserve(totalRecords);
    for (size_t i = 0; i < totalRecords; ++i)
    {
        const size_t sampledKey = keyDistribution(rngPattern);
        auto record = std::make_unique<std::byte[]>(recordSize);
        std::memcpy(record.get(), keyTemplates[sampledKey].get(), recordSize);
        records.emplace_back(std::move(record));
    }

    return records;
}

inline std::vector<std::vector<uint64_t>> generateSlidingWindows(
    const size_t seriesLength,
    const size_t windowSize,
    const double overlapRatio)
{
    if (!(0.0 <= overlapRatio && overlapRatio < 1.0))
    {
        throw std::invalid_argument("overlapRatio must be in [0.0, 1.0).");
    }

    if (windowSize == 0)
    {
        throw std::invalid_argument("windowSize must be greater than 0.");
    }

    if (seriesLength < windowSize)
    {
        return {};
    }

    const size_t stride = std::max<size_t>(
        1, static_cast<size_t>(std::llround(static_cast<double>(windowSize) * (1.0 - overlapRatio))));

    std::vector<std::vector<uint64_t>> windows;
    for (size_t start = 0; start + windowSize <= seriesLength; start += stride)
    {
        std::vector<uint64_t> window;
        window.reserve(windowSize);
        for (size_t value = start; value < start + windowSize; ++value)
        {
            window.emplace_back(static_cast<uint64_t>(value));
        }
        windows.emplace_back(std::move(window));
    }

    return windows;
}

inline BenchmarkData createTemporalLocalityBenchmarkData(
    const size_t seriesLength,
    const size_t windowSize,
    const double overlapRatio,
    const size_t totalRecords,
    const size_t recordSize,
    const uint64_t seed)
{
    if (totalRecords == 0)
    {
        return {};
    }

    if (recordSize == 0)
    {
        throw std::invalid_argument("recordSize must be greater than 0");
    }

    const auto windows = generateSlidingWindows(seriesLength, windowSize, overlapRatio);
    if (windows.empty())
    {
        throw std::invalid_argument("TemporalLocality requires seriesLength >= windowSize and at least one generated window");
    }

    std::vector<uint64_t> accessKeys;
    for (const auto& window : windows)
    {
        accessKeys.insert(accessKeys.end(), window.begin(), window.end());
    }

    if (accessKeys.empty())
    {
        throw std::invalid_argument("TemporalLocality generated an empty access pattern");
    }

    std::mt19937_64 rngData(seed ^ 0x9E3779B97F4A7C15ULL);
    std::vector<std::unique_ptr<std::byte[]>> keyTemplates;
    keyTemplates.reserve(seriesLength);
    for (size_t key = 0; key < seriesLength; ++key)
    {
        keyTemplates.emplace_back(createRecord(recordSize, static_cast<uint64_t>(key), rngData));
    }

    BenchmarkData records;
    records.reserve(totalRecords);
    for (size_t i = 0; i < totalRecords; ++i)
    {
        const size_t keyIndex = static_cast<size_t>(accessKeys[i % accessKeys.size()]);
        auto record = std::make_unique<std::byte[]>(recordSize);
        std::memcpy(record.get(), keyTemplates[keyIndex].get(), recordSize);
        records.emplace_back(std::move(record));
    }

    return records;
}

inline BenchmarkData createBurstinessBenchmarkData(
    const double dutyCycle,       // interpreted as fraction of records generated in bursts
    const size_t onPeriod,        // number of records in one burst
    const size_t numKeys,
    const size_t totalRecords,
    const size_t recordSize,
    const uint64_t seed)
{
    if (totalRecords == 0)
    {
        return {};
    }

    if (recordSize == 0)
    {
        throw std::invalid_argument("recordSize must be greater than 0");
    }

    if (numKeys == 0)
    {
        throw std::invalid_argument("numKeys must be greater than 0");
    }

    if (!(0.0 < dutyCycle && dutyCycle <= 1.0))
    {
        throw std::invalid_argument("dutyCycle must be in (0.0, 1.0]");
    }

    if (onPeriod == 0)
    {
        throw std::invalid_argument("onPeriod must be greater than 0");
    }

    std::mt19937_64 rngPattern(seed);
    std::mt19937_64 rngData(seed ^ 0x9E3779B97F4A7C15ULL);

    // Temporary hot subset size:
    // smaller dutyCycle -> tighter bursts / stronger short-term locality
    const size_t burstHotsetSize =
        std::max<size_t>(1, std::min(numKeys, static_cast<size_t>(std::llround(numKeys * dutyCycle))));

    // Number of emitted records between bursts.
    // dutyCycle = 1.0 => no cool-down gap, fully burst-dense.
    const size_t offPeriod =
        (dutyCycle < 1.0)
            ? std::max<size_t>(
                  1,
                  static_cast<size_t>(std::llround(static_cast<double>(onPeriod) * (1.0 - dutyCycle) / dutyCycle)))
            : 0;

    std::vector<uint64_t> accessKeys;
    accessKeys.reserve(totalRecords);

    // To avoid global skew, rotate burst windows over a shuffled permutation of keys.
    std::vector<size_t> keyOrder(numKeys);
    std::iota(keyOrder.begin(), keyOrder.end(), 0);
    std::shuffle(keyOrder.begin(), keyOrder.end(), rngPattern);

    size_t burstStart = 0;

    while (accessKeys.size() < totalRecords)
    {
        // ---- ON phase: sample repeatedly from the current temporary hot subset ----
        std::uniform_int_distribution<size_t> burstKeyOffsetDist(0, burstHotsetSize - 1);

        const size_t endOn = std::min(totalRecords, accessKeys.size() + onPeriod);
        while (accessKeys.size() < endOn)
        {
            const size_t offset = burstKeyOffsetDist(rngPattern);
            const size_t keyIdx = (burstStart + offset) % numKeys;
            accessKeys.emplace_back(static_cast<uint64_t>(keyOrder[keyIdx]));
        }

        if (accessKeys.size() >= totalRecords)
        {
            break;
        }

        // ---- OFF phase: emit dispersed accesses from outside the current hot subset ----
        if (offPeriod > 0)
        {
            std::vector<size_t> coldPool;
            coldPool.reserve(numKeys - burstHotsetSize);

            for (size_t i = 0; i < numKeys; ++i)
            {
                const size_t circularIdx = (burstStart + i) % numKeys;
                if (i < burstHotsetSize)
                {
                    continue; // skip current burst hot subset
                }
                coldPool.emplace_back(keyOrder[circularIdx]);
            }

            if (!coldPool.empty())
            {
                std::uniform_int_distribution<size_t> coldDist(0, coldPool.size() - 1);
                const size_t endOff = std::min(totalRecords, accessKeys.size() + offPeriod);

                while (accessKeys.size() < endOff)
                {
                    accessKeys.emplace_back(static_cast<uint64_t>(coldPool[coldDist(rngPattern)]));
                }
            }
        }

        // Rotate to the next burst-local hotset.
        burstStart = (burstStart + burstHotsetSize) % numKeys;

        // Optional reshuffle after a full cycle to avoid periodic artifacts.
        if (burstStart == 0)
        {
            std::shuffle(keyOrder.begin(), keyOrder.end(), rngPattern);
        }
    }

    std::vector<std::unique_ptr<std::byte[]>> keyTemplates;
    keyTemplates.reserve(numKeys);
    for (size_t key = 0; key < numKeys; ++key)
    {
        keyTemplates.emplace_back(createRecord(recordSize, static_cast<uint64_t>(key), rngData));
    }

    BenchmarkData records;
    records.reserve(totalRecords);

    for (const uint64_t key : accessKeys)
    {
        auto record = std::make_unique<std::byte[]>(recordSize);
        std::memcpy(record.get(), keyTemplates[static_cast<size_t>(key)].get(), recordSize);
        records.emplace_back(std::move(record));
    }

    return records;
}

}
