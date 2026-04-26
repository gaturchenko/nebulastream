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

inline void driftActiveSet(
    std::vector<size_t>& activeSet,
    const size_t universeSize,
    const double driftFraction,
    std::mt19937_64& rng)
{
    if (!(0.0 <= driftFraction && driftFraction <= 1.0))
    {
        throw std::invalid_argument("driftFraction must be in [0.0, 1.0]");
    }

    if (activeSet.empty() || universeSize == 0 || driftFraction == 0.0)
    {
        return;
    }

    const size_t replaceCount =
        std::min(activeSet.size(), static_cast<size_t>(std::llround(activeSet.size() * driftFraction)));

    if (replaceCount == 0)
    {
        return;
    }

    std::shuffle(activeSet.begin(), activeSet.end(), rng);

    std::vector<bool> inActive(universeSize, false);
    for (const size_t key : activeSet)
    {
        inActive[key] = true;
    }

    std::vector<size_t> candidates;
    candidates.reserve(universeSize - activeSet.size());
    for (size_t key = 0; key < universeSize; ++key)
    {
        if (!inActive[key])
        {
            candidates.push_back(key);
        }
    }

    if (candidates.size() < replaceCount)
    {
        throw std::runtime_error("Not enough replacement candidates for drift");
    }

    std::shuffle(candidates.begin(), candidates.end(), rng);

    for (size_t i = 0; i < replaceCount; ++i)
    {
        activeSet[i] = candidates[i];
    }
}

inline BenchmarkData materializeAccessKeys(
    const std::vector<uint64_t>& accessKeys,
    const size_t numKeys,
    const size_t recordSize,
    std::mt19937_64& rngData)
{
    if (recordSize == 0)
    {
        throw std::invalid_argument("recordSize must be greater than 0");
    }

    std::vector<std::unique_ptr<std::byte[]>> keyTemplates;
    keyTemplates.reserve(numKeys);
    for (size_t key = 0; key < numKeys; ++key)
    {
        keyTemplates.emplace_back(createRecord(recordSize, static_cast<uint64_t>(key), rngData));
    }

    BenchmarkData records;
    records.reserve(accessKeys.size());

    for (const uint64_t key : accessKeys)
    {
        auto record = std::make_unique<std::byte[]>(recordSize);
        std::memcpy(record.get(), keyTemplates[static_cast<size_t>(key)].get(), recordSize);
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
    const double zipfExponent,
    const size_t driftInterval,
    const double driftFraction)
{
    if (totalRecords == 0)
    {
        return {};
    }

    if (numKeys == 0)
    {
        throw std::invalid_argument("numKeys must be greater than 0");
    }

    if (recordSize == 0)
    {
        throw std::invalid_argument("recordSize must be greater than 0");
    }

    const auto zipfPopularity = createZipfPopularity(numKeys, zipfExponent);
    std::discrete_distribution<size_t> rankDistribution(zipfPopularity.begin(), zipfPopularity.end());

    std::mt19937_64 rngPattern(seed);
    std::mt19937_64 rngData(seed ^ 0x9E3779B97F4A7C15ULL);

    // rankToKey[rank] = concrete key currently occupying that popularity rank
    std::vector<size_t> rankToKey(numKeys);
    std::iota(rankToKey.begin(), rankToKey.end(), 0);

    const size_t hotsetSize = std::max<size_t>(1, numKeys / 10);

    std::vector<uint64_t> accessKeys;
    accessKeys.reserve(totalRecords);

    for (size_t i = 0; i < totalRecords; ++i)
    {
        if (driftInterval > 0 && i > 0 && i % driftInterval == 0)
        {
            std::vector<size_t> topHotset(rankToKey.begin(), rankToKey.begin() + hotsetSize);
            driftActiveSet(topHotset, numKeys, driftFraction, rngPattern);
            std::copy(topHotset.begin(), topHotset.end(), rankToKey.begin());

            // Re-randomize the colder tail slightly to avoid persistent positional artifacts.
            if (hotsetSize < numKeys)
            {
                std::shuffle(rankToKey.begin() + static_cast<std::ptrdiff_t>(hotsetSize), rankToKey.end(), rngPattern);
            }
        }

        const size_t sampledRank = rankDistribution(rngPattern);
        accessKeys.emplace_back(static_cast<uint64_t>(rankToKey[sampledRank]));
    }

    return materializeAccessKeys(accessKeys, numKeys, recordSize, rngData);
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
    const size_t universeSize,
    const size_t seriesLength,
    const size_t windowSize,
    const double overlapRatio,
    const size_t totalRecords,
    const size_t recordSize,
    const uint64_t seed,
    const size_t driftInterval,
    const double driftFraction)
{
    if (totalRecords == 0)
    {
        return {};
    }

    if (universeSize == 0)
    {
        throw std::invalid_argument("universeSize must be greater than 0");
    }

    if (seriesLength == 0)
    {
        throw std::invalid_argument("seriesLength must be greater than 0");
    }

    if (windowSize == 0 || windowSize > seriesLength)
    {
        throw std::invalid_argument("windowSize must be in (0, seriesLength]");
    }

    if (universeSize < seriesLength)
    {
        throw std::invalid_argument("universeSize must be >= seriesLength");
    }

    std::mt19937_64 rngPattern(seed);
    std::mt19937_64 rngData(seed ^ 0x9E3779B97F4A7C15ULL);

    const auto windows = generateSlidingWindows(seriesLength, windowSize, overlapRatio);
    if (windows.empty())
    {
        throw std::invalid_argument("TemporalLocality generated an empty access pattern");
    }

    std::vector<uint64_t> accessKeys;
    accessKeys.reserve(totalRecords);

    size_t emitted = 0;
    size_t base = 0;

    while (emitted < totalRecords)
    {
        for (const auto& window : windows)
        {
            for (const auto localKey : window)
            {
                if (emitted >= totalRecords)
                {
                    break;
                }

                const size_t key = (base + static_cast<size_t>(localKey)) % universeSize;
                accessKeys.emplace_back(static_cast<uint64_t>(key));
                ++emitted;

                if (driftInterval > 0 && emitted % driftInterval == 0)
                {
                    const size_t shift =
                        std::max<size_t>(1, static_cast<size_t>(std::llround(seriesLength * driftFraction)));

                    // Shift the active series segment through the larger universe.
                    base = (base + shift) % universeSize;
                }
            }
        }
    }

    return materializeAccessKeys(accessKeys, universeSize, recordSize, rngData);
}

inline BenchmarkData createBurstinessBenchmarkData(
    const double dutyCycle,
    const size_t onPeriod,
    const size_t numKeys,
    const size_t totalRecords,
    const size_t recordSize,
    const uint64_t seed,
    const size_t driftInterval,
    const double driftFraction)
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

    const size_t burstHotsetSize =
        std::max<size_t>(1, std::min(numKeys, static_cast<size_t>(std::llround(numKeys * dutyCycle))));

    const size_t offPeriod =
        (dutyCycle < 1.0)
            ? std::max<size_t>(
                  1,
                  static_cast<size_t>(std::llround(static_cast<double>(onPeriod) * (1.0 - dutyCycle) / dutyCycle)))
            : 0;

    std::vector<size_t> activeBurstSet(burstHotsetSize);
    std::iota(activeBurstSet.begin(), activeBurstSet.end(), 0);
    std::shuffle(activeBurstSet.begin(), activeBurstSet.end(), rngPattern);

    std::vector<uint64_t> accessKeys;
    accessKeys.reserve(totalRecords);

    size_t emitted = 0;
    while (emitted < totalRecords)
    {
        // ON phase: repeated accesses from a temporary active burst hotset
        std::uniform_int_distribution<size_t> burstDist(0, activeBurstSet.size() - 1);
        const size_t endOn = std::min(totalRecords, emitted + onPeriod);

        while (emitted < endOn)
        {
            accessKeys.emplace_back(static_cast<uint64_t>(activeBurstSet[burstDist(rngPattern)]));
            ++emitted;
        }

        if (emitted >= totalRecords)
        {
            break;
        }

        // OFF phase: dispersed accesses outside the current burst hotset
        if (offPeriod > 0)
        {
            std::vector<bool> inBurst(numKeys, false);
            for (const size_t key : activeBurstSet)
            {
                inBurst[key] = true;
            }

            std::vector<size_t> coldPool;
            coldPool.reserve(numKeys - activeBurstSet.size());
            for (size_t key = 0; key < numKeys; ++key)
            {
                if (!inBurst[key])
                {
                    coldPool.push_back(key);
                }
            }

            if (!coldPool.empty())
            {
                std::uniform_int_distribution<size_t> coldDist(0, coldPool.size() - 1);
                const size_t endOff = std::min(totalRecords, emitted + offPeriod);

                while (emitted < endOff)
                {
                    accessKeys.emplace_back(static_cast<uint64_t>(coldPool[coldDist(rngPattern)]));
                    ++emitted;
                }
            }
        }

        // Drift the active burst hotset after enough emitted records.
        if (driftInterval > 0 && emitted % driftInterval == 0)
        {
            driftActiveSet(activeBurstSet, numKeys, driftFraction, rngPattern);
        }
    }

    return materializeAccessKeys(accessKeys, numKeys, recordSize, rngData);
}

}
