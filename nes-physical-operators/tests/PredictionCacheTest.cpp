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
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>

#include <Inference/PredictionCache/PredictionCache.hpp>
#include <Inference/PredictionCache/PredictionCacheEntry.hpp>
#include <Inference/PredictionCache/PredictionCacheUtil.hpp>
#include <Nautilus/DataTypes/DataTypesUtil.hpp>
#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/BufferManager.hpp>
#include <Util/ExecutionMode.hpp>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <Util/Logger/impl/NesLogger.hpp>
#include <gtest/gtest.h>
#include <magic_enum/magic_enum.hpp>
#include <BaseUnitTest.hpp>
#include <Engine.hpp>
#include <InferenceConfiguration.hpp>
#include <function.hpp>
#include <options.hpp>
#include <val_arith.hpp>
#include <val_ptr.hpp>

namespace NES
{

namespace
{

constexpr uint64_t notFound = std::numeric_limits<uint64_t>::max();

struct LookupResult
{
    bool hit;
    uint64_t value;
    uint64_t position;
};

struct ReferenceCacheEntry
{
    bool valid = false;
    uint64_t key = 0;
    uint64_t value = 0;
    uint64_t frequency = 0;
    bool secondChanceBit = false;
};

struct PredictionCacheTestOperation
{
    uint64_t key;
    uint64_t value;
    uint64_t expectedValue;
    uint64_t expectedPosition;
    uint64_t expectedHit;
    std::array<std::byte, sizeof(uint64_t)> record;
    std::array<std::byte, sizeof(uint64_t)> dataStructure;
};

class ReferencePredictionCache
{
public:
    ReferencePredictionCache(const PredictionCacheType cacheType, const uint64_t numberOfEntries)
        : cacheType(cacheType), entries(numberOfEntries), lfuBuckets(numberOfEntries + 1)
    {
    }

    LookupResult lookup(const uint64_t key, const uint64_t newValue)
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

    LookupResult lookupFifo(const uint64_t key, const uint64_t newValue)
    {
        if (const auto pos = findKey(key); pos != notFound)
        {
            return {.hit = true, .value = entries[pos].value, .position = pos};
        }

        replaceEntry(fifoReplacementIndex, key, newValue);
        const auto replacementPos = fifoReplacementIndex;
        fifoReplacementIndex = (fifoReplacementIndex + 1) % entries.size();
        return {.hit = false, .value = newValue, .position = replacementPos};
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

    LookupResult lookupLfu(const uint64_t key, const uint64_t newValue)
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
            return {.hit = true, .value = entries[pos].value, .position = pos};
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
        return {.hit = false, .value = newValue, .position = replacementPos};
    }

    void appendLruTail(const uint64_t pos)
    {
        lruOrder.erase(std::ranges::remove(lruOrder, pos).begin(), lruOrder.end());
        lruOrder.push_back(pos);
    }

    LookupResult lookupLru(const uint64_t key, const uint64_t newValue)
    {
        if (const auto pos = findKey(key); pos != notFound)
        {
            appendLruTail(pos);
            return {.hit = true, .value = entries[pos].value, .position = pos};
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
        return {.hit = false, .value = newValue, .position = replacementPos};
    }

    LookupResult lookupSecondChance(const uint64_t key, const uint64_t newValue)
    {
        if (const auto pos = findKey(key); pos != notFound)
        {
            entries[pos].secondChanceBit = true;
            return {.hit = true, .value = entries[pos].value, .position = pos};
        }

        while (entries[secondChanceReplacementIndex].secondChanceBit)
        {
            entries[secondChanceReplacementIndex].secondChanceBit = false;
            secondChanceReplacementIndex = (secondChanceReplacementIndex + 1) % entries.size();
        }

        const auto replacementPos = secondChanceReplacementIndex;
        replaceEntry(replacementPos, key, newValue);
        entries[replacementPos].secondChanceBit = true;
        return {.hit = false, .value = newValue, .position = replacementPos};
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

void replacePredictionCacheTestEntry(
    PredictionCacheEntry* entry, std::byte* record, std::byte* dataStructure, const size_t recordSize, const size_t dataSize)
{
    delete[] entry->record;
    delete[] entry->dataStructure;

    entry->recordSize = recordSize;
    entry->record = new std::byte[recordSize];
    std::memcpy(entry->record, record, recordSize);

    entry->dataSize = dataSize;
    entry->dataStructure = new std::byte[dataSize];
    std::memcpy(entry->dataStructure, dataStructure, dataSize);
}

}

class PredictionCacheTest : public Testing::BaseUnitTest,
                            public testing::WithParamInterface<std::tuple<ExecutionMode, PredictionCacheType, uint64_t>>
{
public:
    static constexpr bool mlirEnableMultithreading = false;
    static constexpr uint64_t numberOfOperations = 1'000;
    static constexpr uint64_t recordSize = sizeof(uint64_t);
    static constexpr uint64_t dataSize = sizeof(uint64_t);

    ExecutionMode backend = ExecutionMode::INTERPRETER;
    PredictionCacheType cacheType = PredictionCacheType::FIFO;
    uint64_t numberOfEntries = 1;
    std::unique_ptr<nautilus::engine::NautilusEngine> nautilusEngine;
    std::vector<std::byte> cacheMemory;
    std::vector<PredictionCacheTestOperation> operations;
    uint64_t expectedHits = 0;
    uint64_t expectedMisses = 0;

    static void SetUpTestSuite()
    {
        Logger::setupLogging("PredictionCacheTest.log", LogLevel::LOG_DEBUG);
        NES_INFO("Setup PredictionCacheTest class.");
    }

    void SetUp() override
    {
        BaseUnitTest::SetUp();
        backend = std::get<0>(GetParam());
        cacheType = std::get<1>(GetParam());
        numberOfEntries = std::get<2>(GetParam());
        initEngine();
        createRandomPredictionCacheTestOperations();
        calculateExpectedResults();
        allocatePredictionCacheMemory();
    }

    void TearDown() override
    {
        releaseCacheEntries();
        cacheMemory.clear();
        operations.clear();
        expectedHits = 0;
        expectedMisses = 0;
        BaseUnitTest::TearDown();
    }

    static void TearDownTestSuite() { NES_INFO("Tear down PredictionCacheTest class."); }

    void initEngine()
    {
        nautilus::engine::Options options;
        const bool compilation = (backend == ExecutionMode::COMPILER);
        NES_INFO("Backend: {} and compilation: {}", magic_enum::enum_name(backend), compilation);
        options.setOption("engine.Compilation", compilation);
        options.setOption("mlir.enableMultithreading", mlirEnableMultithreading);
        nautilusEngine = std::make_unique<nautilus::engine::NautilusEngine>(options);
    }

    void allocatePredictionCacheMemory()
    {
        const auto entrySize = Util::getPredictionCacheEntrySize(cacheType);
        cacheMemory.resize(sizeof(HitsAndMisses) + numberOfEntries * entrySize, std::byte{0});
    }

    void createRandomPredictionCacheTestOperations()
    {
        const uint64_t seed = 0xCACE'2026ULL + numberOfEntries * 17 + static_cast<uint8_t>(cacheType) * 1'001;
        NES_INFO("PredictionCacheTest random seed: {}", seed);
        std::mt19937_64 gen{seed};
        const uint64_t keyRange = std::max<uint64_t>(numberOfEntries * 3, 4);
        std::uniform_int_distribution<uint64_t> keyDist{0, keyRange - 1};

        operations.reserve(numberOfOperations + numberOfEntries * 2);
        for (uint64_t key = 0; key < numberOfEntries * 2; ++key)
        {
            addOperation(key, key + 1);
        }
        for (uint64_t i = 0; i < numberOfOperations; ++i)
        {
            addOperation(keyDist(gen), 10'000 + i);
        }
    }

    void addOperation(const uint64_t key, const uint64_t value)
    {
        PredictionCacheTestOperation operation{};
        operation.key = key;
        operation.value = value;
        std::memcpy(operation.record.data(), &operation.key, sizeof(operation.key));
        std::memcpy(operation.dataStructure.data(), &operation.value, sizeof(operation.value));
        operations.push_back(operation);
    }

    void calculateExpectedResults()
    {
        ReferencePredictionCache referenceCache(cacheType, numberOfEntries);
        for (auto& operation : operations)
        {
            const auto [hit, value, position] = referenceCache.lookup(operation.key, operation.value);
            operation.expectedValue = value;
            operation.expectedPosition = position;
            operation.expectedHit = hit ? 1 : 0;
            expectedHits += hit ? 1 : 0;
            expectedMisses += hit ? 0 : 1;
        }
    }

    void releaseCacheEntries()
    {
        if (cacheMemory.empty())
        {
            return;
        }

        const auto entrySize = Util::getPredictionCacheEntrySize(cacheType);
        auto* entries = cacheMemory.data() + sizeof(HitsAndMisses);
        for (uint64_t i = 0; i < numberOfEntries; ++i)
        {
            auto* entry = reinterpret_cast<PredictionCacheEntry*>(entries + i * entrySize);
            delete[] entry->record;
            delete[] entry->dataStructure;
            entry->record = nullptr;
            entry->dataStructure = nullptr;
        }
    }
};

TEST_P(PredictionCacheTest, testPredictionCacheReplacementPolicy)
{
    using CompiledCacheFunction = std::function<nautilus::val<uint64_t>(
        nautilus::val<int8_t*>,
        nautilus::val<PredictionCacheTestOperation*>,
        nautilus::val<ChainedHashMap*>,
        nautilus::val<AbstractBufferProvider*>)>;

    auto predictionCacheCallableFunction = nautilusEngine->registerFunction(CompiledCacheFunction(
        [&](const nautilus::val<int8_t*>& cacheStart,
            const nautilus::val<PredictionCacheTestOperation*>& operationsStart,
            const nautilus::val<ChainedHashMap*>& lookupIndex,
            const nautilus::val<AbstractBufferProvider*>& bufferProvider) -> nautilus::val<uint64_t>
        {
            auto predictionCache = Util::createPredictionCache(cacheType, numberOfEntries, cacheStart, recordSize);
            predictionCache->configureLookupIndex(lookupIndex, bufferProvider);
            nautilus::val<uint64_t> mismatches = 0;

            for (nautilus::val<uint64_t> i = 0; i < operations.size(); ++i)
            {
                const auto operation = operationsStart + i;
                const auto operationBytes = static_cast<nautilus::val<int8_t*>>(operation);
                const auto record
                    = static_cast<nautilus::val<std::byte*>>(getMemberRef(operationBytes, &PredictionCacheTestOperation::key));
                const auto dataStructure
                    = static_cast<nautilus::val<std::byte*>>(getMemberRef(operationBytes, &PredictionCacheTestOperation::value));
                const nautilus::val<uint64_t> expectedValue{
                    *getMemberWithOffset<uint64_t>(getMemberRef(operationBytes, &PredictionCacheTestOperation::expectedValue), 0)};

                const auto result = predictionCache->getDataStructureRef(
                    record,
                    [&](const nautilus::val<PredictionCacheEntry*>& entryToReplace, const nautilus::val<uint64_t>&)
                    {
                        nautilus::invoke(
                            replacePredictionCacheTestEntry,
                            entryToReplace,
                            record,
                            dataStructure,
                            nautilus::val<size_t>(recordSize),
                            nautilus::val<size_t>(dataSize));
                        const auto dataStructureRef = getMemberRef(entryToReplace, &PredictionCacheEntry::dataStructure);
                        return *getMemberWithOffset<std::byte*>(dataStructureRef, 0);
                    });

                const nautilus::val<uint64_t> actualValue{*static_cast<nautilus::val<uint64_t*>>(result)};
                if (actualValue != expectedValue)
                {
                    mismatches = mismatches + 1;
                }
            }

            const nautilus::val<uint64_t> hits{*static_cast<nautilus::val<uint64_t*>>(cacheStart)};
            const nautilus::val<uint64_t> misses{*(static_cast<nautilus::val<uint64_t*>>(cacheStart) + nautilus::val<uint64_t>(1))};
            if (hits != expectedHits)
            {
                mismatches = mismatches + 1;
            }
            if (misses != expectedMisses)
            {
                mismatches = mismatches + 1;
            }
            return mismatches;
        }));

    auto bufferManager = BufferManager::create(4096, 1000);
    ChainedHashMap lookupIndex(0, sizeof(uint64_t), numberOfEntries, 4096);
    const auto mismatches = predictionCacheCallableFunction(
        reinterpret_cast<int8_t*>(cacheMemory.data()), operations.data(), &lookupIndex, bufferManager.get());
    EXPECT_EQ(mismatches, 0) << "Prediction cache result mismatch in policy " << magic_enum::enum_name(cacheType);
}

TEST_P(PredictionCacheTest, testPredictionCacheUpdateKeysReplacementPolicy)
{
    using CompiledCacheFunction = std::function<nautilus::val<uint64_t>(
        nautilus::val<int8_t*>,
        nautilus::val<PredictionCacheTestOperation*>,
        nautilus::val<ChainedHashMap*>,
        nautilus::val<AbstractBufferProvider*>)>;

    auto predictionCacheCallableFunction = nautilusEngine->registerFunction(CompiledCacheFunction(
        [&](const nautilus::val<int8_t*>& cacheStart,
            const nautilus::val<PredictionCacheTestOperation*>& operationsStart,
            const nautilus::val<ChainedHashMap*>& lookupIndex,
            const nautilus::val<AbstractBufferProvider*>& bufferProvider) -> nautilus::val<uint64_t>
        {
            auto predictionCache = Util::createPredictionCache(cacheType, numberOfEntries, cacheStart, recordSize);
            predictionCache->configureLookupIndex(lookupIndex, bufferProvider);
            nautilus::val<uint64_t> mismatches = 0;

            for (nautilus::val<uint64_t> i = 0; i < operations.size(); ++i)
            {
                const auto operation = operationsStart + i;
                const auto operationBytes = static_cast<nautilus::val<int8_t*>>(operation);
                const auto record
                    = static_cast<nautilus::val<std::byte*>>(getMemberRef(operationBytes, &PredictionCacheTestOperation::key));
                const auto dataStructure
                    = static_cast<nautilus::val<std::byte*>>(getMemberRef(operationBytes, &PredictionCacheTestOperation::value));
                const nautilus::val<uint64_t> expectedValue{
                    *getMemberWithOffset<uint64_t>(getMemberRef(operationBytes, &PredictionCacheTestOperation::expectedValue), 0)};
                const nautilus::val<uint64_t> expectedPosition{
                    *getMemberWithOffset<uint64_t>(getMemberRef(operationBytes, &PredictionCacheTestOperation::expectedPosition), 0)};
                const nautilus::val<uint64_t> expectedHit{
                    *getMemberWithOffset<uint64_t>(getMemberRef(operationBytes, &PredictionCacheTestOperation::expectedHit), 0)};

                nautilus::val<uint64_t> keyUpdateCallbackCalls = 0;
                const auto lookupResult = predictionCache->updateKeys(
                    record,
                    [&](const nautilus::val<PredictionCacheEntry*>& entryToReplace, const nautilus::val<uint64_t>& replacementIndex)
                    {
                        static_cast<void>(entryToReplace);
                        keyUpdateCallbackCalls = keyUpdateCallbackCalls + 1;
                        if (replacementIndex != expectedPosition)
                        {
                            mismatches = mismatches + 1;
                        }
                    });

                if (expectedHit != 0)
                {
                    if (lookupResult == PredictionCache::NOT_FOUND)
                    {
                        mismatches = mismatches + 1;
                    }
                    else
                    {
                        if (lookupResult != expectedPosition)
                        {
                            mismatches = mismatches + 1;
                        }
                        if (keyUpdateCallbackCalls != 0)
                        {
                            mismatches = mismatches + 1;
                        }

                        const auto dataStructureRef = predictionCache->getDataStructure(lookupResult);
                        const nautilus::val<uint64_t> actualValue{*static_cast<nautilus::val<uint64_t*>>(dataStructureRef)};
                        if (actualValue != expectedValue)
                        {
                            mismatches = mismatches + 1;
                        }
                    }
                }
                else
                {
                    if (lookupResult != PredictionCache::NOT_FOUND)
                    {
                        mismatches = mismatches + 1;
                    }
                    if (keyUpdateCallbackCalls != 1)
                    {
                        mismatches = mismatches + 1;
                    }

                    const auto replacementIndex = predictionCache->getReplacementIndex();
                    if (replacementIndex != expectedPosition)
                    {
                        mismatches = mismatches + 1;
                    }

                    predictionCache->updateValues(
                        replacementIndex,
                        [&](const nautilus::val<PredictionCacheEntry*>& entryToUpdate, const nautilus::val<uint64_t>& updateIndex)
                        {
                            if (updateIndex != expectedPosition)
                            {
                                mismatches = mismatches + 1;
                            }
                            nautilus::invoke(
                                replacePredictionCacheTestEntry,
                                entryToUpdate,
                                record,
                                dataStructure,
                                nautilus::val<size_t>(recordSize),
                                nautilus::val<size_t>(dataSize));
                        });

                    const auto dataStructureRef = predictionCache->getDataStructure(replacementIndex);
                    const nautilus::val<uint64_t> actualValue{*static_cast<nautilus::val<uint64_t*>>(dataStructureRef)};
                    if (actualValue != expectedValue)
                    {
                        mismatches = mismatches + 1;
                    }
                }
            }

            const nautilus::val<uint64_t> hits{*static_cast<nautilus::val<uint64_t*>>(cacheStart)};
            const nautilus::val<uint64_t> misses{*(static_cast<nautilus::val<uint64_t*>>(cacheStart) + nautilus::val<uint64_t>(1))};
            if (hits != expectedHits)
            {
                mismatches = mismatches + 1;
            }
            if (misses != expectedMisses)
            {
                mismatches = mismatches + 1;
            }
            return mismatches;
        }));

    auto bufferManager = BufferManager::create(4096, 1000);
    ChainedHashMap lookupIndex(0, sizeof(uint64_t), numberOfEntries, 4096);
    const auto mismatches = predictionCacheCallableFunction(
        reinterpret_cast<int8_t*>(cacheMemory.data()), operations.data(), &lookupIndex, bufferManager.get());
    EXPECT_EQ(mismatches, 0) << "Prediction cache updateKeys result mismatch in policy " << magic_enum::enum_name(cacheType);
}

INSTANTIATE_TEST_CASE_P(
    PredictionCachePolicies,
    PredictionCacheTest,
    ::testing::Combine(
        ::testing::Values(ExecutionMode::INTERPRETER, ExecutionMode::COMPILER),
        ::testing::Values(
            PredictionCacheType::FIFO, PredictionCacheType::LFU, PredictionCacheType::LRU, PredictionCacheType::SECOND_CHANCE),
        ::testing::Values(1, 3, 16)),
    [](const testing::TestParamInfo<PredictionCacheTest::ParamType>& info)
    {
        std::stringstream ss;
        ss << magic_enum::enum_name(std::get<0>(info.param)) << "_" << magic_enum::enum_name(std::get<1>(info.param)) << "_Entries"
           << std::get<2>(info.param);
        return ss.str();
    });

}
