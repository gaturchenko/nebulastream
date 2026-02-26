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

#include <Configuration/WorkerConfiguration.hpp>
#include <DataTypes/DataType.hpp>
#include <Nautilus/Util.hpp>
#include <Nautilus/DataTypes/DataTypesUtil.hpp>
#include <PredictionCache/PredictionCache.hpp>
#include <PredictionCache/PredictionCacheAlwaysMiss.hpp>
#include <PredictionCache/PredictionCacheFIFO.hpp>
#include <PredictionCache/PredictionCacheLFU.hpp>
#include <PredictionCache/PredictionCacheLRU.hpp>
#include <PredictionCache/PredictionCacheSecondChance.hpp>
#include <PredictionCache/PredictionCacheUtil.hpp>
#include <Runtime/Execution/OperatorHandler.hpp>
#include <nautilus/val.hpp>
#include <nautilus/Engine.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

namespace
{

size_t constexpr TUPLE_SIZE = 32;
int constexpr WARMUP_RUNS = 3;

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

struct PredictionCacheOptionsMicroBenchmark : public NES::Configurations::PredictionCacheOptions
{
    explicit PredictionCacheOptionsMicroBenchmark(const NES::Configurations::PredictionCacheType predictionCacheType, uint64_t numberOfEntries)
        : NES::Configurations::PredictionCacheOptions({predictionCacheType, numberOfEntries})
    {
    }

    std::string getValuesAsCsv() const { return fmt::format("{},{}", magic_enum::enum_name(predictionCacheType), numberOfEntries); }

    static std::string getCsvHeader() { return fmt::format("prediction_cache_type,number_of_entries"); }
};

struct BenchmarkParameters
{
    PredictionCacheOptionsMicroBenchmark predictionCacheOptions;
    HitMissRatio hitMissRatio;

    std::string getValuesAsCsv() const
    {
        return fmt::format("{},{}", predictionCacheOptions.getValuesAsCsv(), getHitPercentage(hitMissRatio));
    }

    static std::string getCsvHeader()
    {
        return fmt::format("{},hits_percentage", PredictionCacheOptionsMicroBenchmark::getCsvHeader());
    }
};

struct BenchmarkRunMeasurements
{
    std::chrono::microseconds executionTime;
    uint64_t cacheHits;
    uint64_t cacheMisses;

    std::string getValuesAsCsv() const
    {
        return fmt::format("{},{},{}", executionTime.count(), cacheHits, cacheMisses);
    }

    static std::string getCsvHeader() { return "execution_time,cache_hits,cache_misses"; }
};

std::string createNewCsvFileLine(const BenchmarkParameters& parameters, const BenchmarkRunMeasurements& measurements)
{
    std::stringstream csvValues;
    /// We first write all values of parameters as csv, followed by the measurments
    csvValues << parameters.getValuesAsCsv();
    csvValues << ",";
    csvValues << measurements.getValuesAsCsv();

    return csvValues.str();
}

std::string createNewCsvHeaderLine()
{
    return BenchmarkParameters::getCsvHeader() + "," + BenchmarkRunMeasurements::getCsvHeader();
}

using BenchmarkData = std::vector<std::unique_ptr<std::byte[]>>;

std::unique_ptr<std::byte[]>
createRecord(const size_t recordSize, const uint64_t id, std::mt19937_64& rng)
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
        SimCache(const SimCachePolicy policy, const size_t capacity)
            : policy(policy), capacity(capacity), slots(capacity)
        {
        }

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

BenchmarkData createBenchmarkData(
    const uint64_t cacheSize,
    const size_t totalRecords,
    const HitMissRatio ratio,
    const NES::Configurations::PredictionCacheType predictionCacheType,
    const size_t recordSize = TUPLE_SIZE,
    const uint64_t seed = 0xC0FFEEULL)
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

size_t getPredictionCacheEntrySize(const NES::Configurations::PredictionCacheOptions& predictionCacheOptions)
{
    switch (predictionCacheOptions.predictionCacheType)
    {
        case NES::Configurations::PredictionCacheType::ALWAYS_MISS:
        case NES::Configurations::PredictionCacheType::NONE:
        case NES::Configurations::PredictionCacheType::TWO_QUEUES:
            throw std::runtime_error("PredictionCacheType is invalid");
        case NES::Configurations::PredictionCacheType::FIFO:
            return sizeof(NES::PredictionCacheEntryFIFO);
        case NES::Configurations::PredictionCacheType::LFU:
            return sizeof(NES::PredictionCacheEntryLFU);
        case NES::Configurations::PredictionCacheType::LRU:
            return sizeof(NES::PredictionCacheEntryLRU);
        case NES::Configurations::PredictionCacheType::SECOND_CHANCE:
            return sizeof(NES::PredictionCacheEntrySecondChance);
    }
    std::unreachable();
}

auto createPredictionCacheFillFunction(
    const nautilus::engine::NautilusEngine& nautilusEngine, const NES::Configurations::PredictionCacheOptions& predictionCacheOptions)
{
    return nautilusEngine.registerFunction(std::function(
        [copyOfPredictionCacheOptions = predictionCacheOptions](
            nautilus::val<std::byte**> inputData,
            nautilus::val<uint64_t> sizeInputData,
            nautilus::val<int8_t*> startOfEntries,
            nautilus::val<size_t> inputSize)
        {
            using namespace nautilus;

            const val<int8_t*> globalOperatorHandler = nullptr;
            const auto predictionCache = NES::Util::createPredictionCache(
                copyOfPredictionCacheOptions, globalOperatorHandler, startOfEntries, inputSize);

            for (val<uint64_t> i = 0; i < sizeInputData; ++i)
            {
                const auto inputDataRef = static_cast<val<int8_t*>>(inputData + i);
                val<std::byte*> inputDataVal(NES::readValueFromMemRef<std::byte*>(inputDataRef));

                predictionCache->getDataStructureRef(
                    inputDataVal,
                    [&](const val<NES::PredictionCacheEntry*>& predictionCacheEntryToReplace, const val<uint64_t>& replacementIndex)
                    {
                        const auto recordReplacement = predictionCache->getRecord(replacementIndex);

                        const auto recordRef = NES::getMemberRef(predictionCacheEntryToReplace, &NES::PredictionCacheEntry::record);
                        const auto recordSizeRef = NES::getMemberRef(predictionCacheEntryToReplace, &NES::PredictionCacheEntry::recordSize);

                        *NES::getMemberWithOffset<std::byte*>(recordRef, 0) = inputDataVal;
                        *NES::getMemberWithOffset<std::size_t>(recordSizeRef, 0) = nautilus::val<std::size_t>(TUPLE_SIZE);

                        const auto dataStructureRef = NES::getMemberRef(predictionCacheEntryToReplace, &NES::PredictionCacheEntry::dataStructure);
                        *NES::getMemberWithOffset<std::byte*>(dataStructureRef, 0) = nautilus::val<std::byte*>(nullptr);

                        return predictionCacheEntryToReplace;
                    });
            }
        }));
}

std::vector<BenchmarkRunMeasurements>
runBenchmark(const BenchmarkParameters& benchmarkParams, const int numReps, BenchmarkData& benchmarkData)
{
    nautilus::engine::Options options;
    options.setOption("engine.Compilation", true);
    const nautilus::engine::NautilusEngine nautilusEngine(options);
    auto predictionCacheFunction = createPredictionCacheFillFunction(nautilusEngine, benchmarkParams.predictionCacheOptions);

    std::vector<BenchmarkRunMeasurements> benchmarkRunMeasurements;
    for (auto rep = 0; rep < numReps + WARMUP_RUNS; ++rep)
    {
        std::vector<std::byte*> benchmarkDataRefs;
        benchmarkDataRefs.reserve(benchmarkData.size());
        for (const auto& record : benchmarkData)
        {
            benchmarkDataRefs.emplace_back(record.get());
        }

        const auto neededSize
            = benchmarkParams.predictionCacheOptions.numberOfEntries * getPredictionCacheEntrySize(benchmarkParams.predictionCacheOptions)
            + sizeof(NES::HitsAndMisses);
        std::vector<int8_t> predictionCacheMemory(neededSize);
        std::memset(predictionCacheMemory.data(), 0, neededSize);

        const auto startTime = std::chrono::high_resolution_clock::now();

        predictionCacheFunction(
            benchmarkDataRefs.data(),
            benchmarkDataRefs.size(),
            predictionCacheMemory.data(),
            TUPLE_SIZE);

        const auto duration = std::chrono::high_resolution_clock::now() - startTime;

        const auto hits = *reinterpret_cast<uint64_t*>(predictionCacheMemory.data());
        const auto misses = *reinterpret_cast<uint64_t*>(predictionCacheMemory.data() + sizeof(hits));

        if (rep >= WARMUP_RUNS)
        {
            benchmarkRunMeasurements.emplace_back(duration_cast<std::chrono::microseconds>(duration), hits, misses);
        }
    }

    return benchmarkRunMeasurements;
}

class ETACalculator
{
public:
    explicit ETACalculator(const int totalCalls) : totalCalls(totalCalls), callsCompleted(0)
    {
        startTime = std::chrono::system_clock::now();
    }

    void update()
    {
        const auto now = std::chrono::system_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
        const auto lastIterationElapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastIterationStart).count();

        callsCompleted++;
        lastIterationStart = now;

        if (elapsed > 0)
        {
            const double timePerCall = static_cast<double>(elapsed) / callsCompleted;
            const int remainingCalls = totalCalls - callsCompleted;
            const double etaSeconds = timePerCall * remainingCalls;

            /// Calculate ETA wall time
            const auto etaWallTime = std::chrono::system_clock::now() + std::chrono::seconds(static_cast<int>(etaSeconds));
            const std::time_t etaTimeT = std::chrono::system_clock::to_time_t(etaWallTime);

            /// Print progress, last iteration time, ETA in seconds, and ETA wall time
            std::cout << "Progress: " << callsCompleted << "/" << totalCalls << ". ";
            std::cout << "Last iteration took: " << lastIterationElapsed << " s. ";
            std::cout << "ETA: " << static_cast<int>(etaSeconds) << " seconds remaining. ";
            std::cout << "Wall time: " << std::put_time(std::localtime(&etaTimeT), "%Y-%m-%d %X") << std::endl;
        }
    }

private:
    int totalCalls;
    int callsCompleted;
    std::chrono::system_clock::time_point startTime;
    std::chrono::system_clock::time_point lastIterationStart;
};

}

int main()
{
    constexpr auto allPredictionCacheTypes = magic_enum::enum_values<NES::Configurations::PredictionCacheType>();
    const auto allPredictionCacheSizes =  {100, 1'000}; // {10'000};
    const auto allHitsMissesRatios = magic_enum::enum_values<HitMissRatio>();
    constexpr auto REPS = 10;

    std::filesystem::path csvFilePath("prediction_cache_micro_benchmarks.csv");
    std::error_code removeError;
    std::filesystem::remove(csvFilePath, removeError);
    if (removeError && removeError != std::errc::no_such_file_or_directory)
    {
        std::cerr << "Failed to remove CSV file '" << csvFilePath.string()
                  << "': " << removeError.message() << std::endl;
        return 1;
    }

    std::ofstream csvFile(csvFilePath, std::ios::out | std::ios::trunc);
    if (!csvFile.is_open())
    {
        std::cerr << "Failed to open CSV file '" << csvFilePath.string() << "' for writing." << std::endl;
        return 1;
    }
    csvFile << createNewCsvHeaderLine() << std::endl;
    std::cout << createNewCsvHeaderLine() << std::endl;

    ETACalculator etaCalculator(
        allPredictionCacheSizes.size() * (allPredictionCacheTypes.size() - 2) * allHitsMissesRatios.size());

    for (const uint64_t predictionCacheSize: allPredictionCacheSizes)
    {
        for (const auto& hitMissRatio: allHitsMissesRatios)
        {
            for (const auto& predictionCacheType: allPredictionCacheTypes)
            {
                if (predictionCacheType == NES::Configurations::PredictionCacheType::NONE
                    || predictionCacheType == NES::Configurations::PredictionCacheType::TWO_QUEUES
                    || predictionCacheType == NES::Configurations::PredictionCacheType::ALWAYS_MISS)
                {
                    continue;
                }

                auto benchmarkData = createBenchmarkData(predictionCacheSize, 1'000'000, hitMissRatio, predictionCacheType);
                BenchmarkParameters benchmarkParams{
                    PredictionCacheOptionsMicroBenchmark{predictionCacheType, predictionCacheSize}, hitMissRatio};
                const auto results = runBenchmark(benchmarkParams, REPS, benchmarkData);

                for (const auto& result : results)
                {
                    csvFile << createNewCsvFileLine(benchmarkParams, result) << std::endl;
                    std::cout << createNewCsvFileLine(benchmarkParams, result) << std::endl;
                }
                csvFile.flush();
                etaCalculator.update();
            }
        }
    }

    return 0;
}
