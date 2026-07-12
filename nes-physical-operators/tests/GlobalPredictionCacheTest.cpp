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

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <thread>
#include <tuple>
#include <vector>

#include <Inference/PredictionCache/GlobalPredictionCache.hpp>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <gtest/gtest.h>
#include <magic_enum/magic_enum.hpp>
#include <BaseUnitTest.hpp>
#include <InferenceConfiguration.hpp>

namespace NES
{

namespace
{

constexpr uint64_t notFound = std::numeric_limits<uint64_t>::max();

/// Reference emulation of the replacement policies, mirroring the semantics of the
/// thread-local prediction caches (see ReferencePredictionCache in PredictionCacheTest.cpp).
class ReferencePredictionCache
{
public:
    struct ReferenceCacheEntry
    {
        bool valid = false;
        uint64_t key = 0;
        uint64_t value = 0;
        uint64_t frequency = 0;
        bool secondChanceBit = false;
    };

    ReferencePredictionCache(const PredictionCacheType cacheType, const uint64_t numberOfEntries)
        : cacheType(cacheType), entries(numberOfEntries), lfuBuckets(numberOfEntries + 1)
    {
    }

    /// Returns {hit, value}.
    std::pair<bool, uint64_t> lookup(const uint64_t key, const uint64_t newValue)
    {
        switch (cacheType)
        {
            case PredictionCacheType::FIFO:
                return lookupFifo(key, newValue);
            case PredictionCacheType::LFU:
                return lookupLfu(key, newValue);
            case PredictionCacheType::LRU:
                return lookupLru(key, newValue);
            case PredictionCacheType::SECOND_CHANCE:
                return lookupSecondChance(key, newValue);
            case PredictionCacheType::NONE:
                break;
        }
        std::unreachable();
    }

private:
    uint64_t findKey(const uint64_t key) const
    {
        for (uint64_t i = 0; i < entries.size(); ++i)
        {
            if (entries[i].valid && entries[i].key == key)
            {
                return i;
            }
        }
        return notFound;
    }

    void replaceEntry(const uint64_t pos, const uint64_t key, const uint64_t value)
    {
        entries[pos].valid = true;
        entries[pos].key = key;
        entries[pos].value = value;
    }

    std::pair<bool, uint64_t> lookupFifo(const uint64_t key, const uint64_t newValue)
    {
        if (const auto pos = findKey(key); pos != notFound)
        {
            return {true, entries[pos].value};
        }
        replaceEntry(fifoReplacementIndex, key, newValue);
        fifoReplacementIndex = (fifoReplacementIndex + 1) % entries.size();
        return {false, newValue};
    }

    uint64_t saturateFrequency(const uint64_t frequency) const { return std::min<uint64_t>(frequency, entries.size()); }

    void removeFromLfuBucket(const uint64_t pos)
    {
        auto& bucket = lfuBuckets[entries[pos].frequency];
        bucket.erase(std::ranges::remove(bucket, pos).begin(), bucket.end());
    }

    void addToLfuBucket(const uint64_t pos, const uint64_t frequency)
    {
        entries[pos].frequency = saturateFrequency(frequency);
        lfuBuckets[entries[pos].frequency].push_back(pos);
    }

    void updateMinFrequencyAfterRemoving(const uint64_t oldFrequency)
    {
        if (oldFrequency != minFrequency || !lfuBuckets[minFrequency].empty())
        {
            return;
        }
        while (minFrequency < lfuBuckets.size() && lfuBuckets[minFrequency].empty())
        {
            minFrequency++;
        }
    }

    std::pair<bool, uint64_t> lookupLfu(const uint64_t key, const uint64_t newValue)
    {
        if (const auto pos = findKey(key); pos != notFound)
        {
            const auto oldFrequency = entries[pos].frequency;
            if (oldFrequency < entries.size())
            {
                removeFromLfuBucket(pos);
                addToLfuBucket(pos, oldFrequency + 1);
                updateMinFrequencyAfterRemoving(oldFrequency);
            }
            return {true, entries[pos].value};
        }

        uint64_t replacementPos;
        if (nextEmptyLfuPos < entries.size())
        {
            replacementPos = nextEmptyLfuPos++;
        }
        else
        {
            replacementPos = lfuBuckets[minFrequency].front();
            lfuBuckets[minFrequency].erase(lfuBuckets[minFrequency].begin());
        }

        replaceEntry(replacementPos, key, newValue);
        addToLfuBucket(replacementPos, 1);
        minFrequency = 1;
        return {false, newValue};
    }

    void appendLruTail(const uint64_t pos)
    {
        lruOrder.erase(std::ranges::remove(lruOrder, pos).begin(), lruOrder.end());
        lruOrder.push_back(pos);
    }

    std::pair<bool, uint64_t> lookupLru(const uint64_t key, const uint64_t newValue)
    {
        if (const auto pos = findKey(key); pos != notFound)
        {
            appendLruTail(pos);
            return {true, entries[pos].value};
        }

        uint64_t replacementPos;
        if (nextEmptyLruPos < entries.size())
        {
            replacementPos = nextEmptyLruPos++;
        }
        else
        {
            replacementPos = lruOrder.front();
            lruOrder.erase(lruOrder.begin());
        }
        replaceEntry(replacementPos, key, newValue);
        appendLruTail(replacementPos);
        return {false, newValue};
    }

    std::pair<bool, uint64_t> lookupSecondChance(const uint64_t key, const uint64_t newValue)
    {
        if (const auto pos = findKey(key); pos != notFound)
        {
            entries[pos].secondChanceBit = true;
            return {true, entries[pos].value};
        }

        while (entries[secondChanceReplacementIndex].secondChanceBit)
        {
            entries[secondChanceReplacementIndex].secondChanceBit = false;
            secondChanceReplacementIndex = (secondChanceReplacementIndex + 1) % entries.size();
        }

        replaceEntry(secondChanceReplacementIndex, key, newValue);
        entries[secondChanceReplacementIndex].secondChanceBit = true;
        return {false, newValue};
    }

    PredictionCacheType cacheType;
    std::vector<ReferenceCacheEntry> entries;
    uint64_t fifoReplacementIndex = 0;
    uint64_t nextEmptyLfuPos = 0;
    uint64_t minFrequency = 1;
    std::vector<std::vector<uint64_t>> lfuBuckets;
    uint64_t nextEmptyLruPos = 0;
    std::vector<uint64_t> lruOrder;
    uint64_t secondChanceReplacementIndex = 0;
};

uint64_t predictionForKey(const uint64_t key)
{
    return key * 2654435761ULL + 17;
}

}

class GlobalPredictionCacheTest : public Testing::BaseUnitTest,
                                  public testing::WithParamInterface<std::tuple<PredictionCacheType, uint64_t>>
{
public:
    static void SetUpTestSuite()
    {
        Logger::setupLogging("GlobalPredictionCacheTest.log", LogLevel::LOG_DEBUG);
        NES_INFO("Setup GlobalPredictionCacheTest class.");
    }
};

/// Single-threaded: the global cache must produce exactly the same hit/miss sequence
/// and cached values as the reference policy emulation.
TEST_P(GlobalPredictionCacheTest, matchesReferencePolicySemantics)
{
    const auto [cacheType, numberOfEntries] = GetParam();
    constexpr uint64_t numberOfOperations = 2'000;
    constexpr size_t recordSize = sizeof(uint64_t);
    constexpr size_t predictionSize = sizeof(uint64_t);

    GlobalPredictionCache cache(cacheType, numberOfEntries, recordSize, predictionSize);
    ReferencePredictionCache reference(cacheType, numberOfEntries);

    std::mt19937_64 generator(42);
    std::uniform_int_distribution<uint64_t> keyDistribution(0, (2 * numberOfEntries) + 2);

    uint64_t expectedHits = 0;
    uint64_t expectedMisses = 0;
    for (uint64_t i = 0; i < numberOfOperations; ++i)
    {
        const auto key = keyDistribution(generator);
        const auto expectedValue = predictionForKey(key);
        const auto [expectedHit, referenceValue] = reference.lookup(key, expectedValue);
        expectedHits += expectedHit ? 1 : 0;
        expectedMisses += expectedHit ? 0 : 1;

        uint64_t actualValue = 0;
        const auto hit
            = cache.lookup(reinterpret_cast<const std::byte*>(&key), reinterpret_cast<std::byte*>(&actualValue));
        ASSERT_EQ(hit, expectedHit) << "operation " << i << " key " << key << " policy " << magic_enum::enum_name(cacheType);
        if (hit)
        {
            ASSERT_EQ(actualValue, referenceValue);
        }
        else
        {
            cache.insert(reinterpret_cast<const std::byte*>(&key), reinterpret_cast<const std::byte*>(&expectedValue));
        }
    }

    const auto stats = cache.getHitsAndMisses();
    EXPECT_EQ(stats.hits, expectedHits);
    EXPECT_EQ(stats.misses, expectedMisses);
    EXPECT_EQ(stats.hits + stats.misses, numberOfOperations);
}

/// Multi-threaded: values returned for a key must always be the value computed for that
/// key, and the hit/miss counters must add up. This validates that concurrent lookups,
/// inserts, and evictions never hand out a prediction belonging to a different record.
TEST_P(GlobalPredictionCacheTest, concurrentLookupsReturnConsistentValues)
{
    const auto [cacheType, numberOfEntries] = GetParam();
    constexpr uint64_t operationsPerThread = 20'000;
    constexpr size_t numberOfThreads = 4;
    constexpr size_t recordSize = sizeof(uint64_t);
    constexpr size_t predictionSize = sizeof(uint64_t);

    GlobalPredictionCache cache(cacheType, numberOfEntries, recordSize, predictionSize);
    std::atomic<uint64_t> valueMismatches{0};

    std::vector<std::thread> threads;
    threads.reserve(numberOfThreads);
    for (size_t threadIndex = 0; threadIndex < numberOfThreads; ++threadIndex)
    {
        threads.emplace_back(
            [&, threadIndex]
            {
                std::mt19937_64 generator(threadIndex);
                std::uniform_int_distribution<uint64_t> keyDistribution(0, (2 * numberOfEntries) + 2);
                for (uint64_t i = 0; i < operationsPerThread; ++i)
                {
                    const auto key = keyDistribution(generator);
                    const auto expectedValue = predictionForKey(key);
                    uint64_t actualValue = 0;
                    if (cache.lookup(reinterpret_cast<const std::byte*>(&key), reinterpret_cast<std::byte*>(&actualValue)))
                    {
                        if (actualValue != expectedValue)
                        {
                            valueMismatches.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    else
                    {
                        cache.insert(
                            reinterpret_cast<const std::byte*>(&key), reinterpret_cast<const std::byte*>(&expectedValue));
                    }
                }
            });
    }
    for (auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(valueMismatches.load(), 0);
    const auto stats = cache.getHitsAndMisses();
    EXPECT_EQ(stats.hits + stats.misses, operationsPerThread * numberOfThreads);
}

INSTANTIATE_TEST_CASE_P(
    GlobalPredictionCacheTests,
    GlobalPredictionCacheTest,
    testing::Combine(
        testing::Values(
            PredictionCacheType::FIFO, PredictionCacheType::LFU, PredictionCacheType::LRU, PredictionCacheType::SECOND_CHANCE),
        testing::Values(1, 2, 5, 16)),
    [](const testing::TestParamInfo<GlobalPredictionCacheTest::ParamType>& info)
    {
        return std::string(magic_enum::enum_name(std::get<0>(info.param))) + "_" + std::to_string(std::get<1>(info.param))
            + "Entries";
    });

}
