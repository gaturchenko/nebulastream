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
#include <vector>

namespace
{

size_t constexpr TUPLE_SIZE = 32;

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

[[maybe_unused]] std::unique_ptr<std::byte[]>
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

[[maybe_unused]] BenchmarkData createBenchmarkData(
    const uint64_t cacheSize,
    const size_t totalRecords,
    const HitMissRatio ratio,
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

    const size_t numHits = static_cast<size_t>((totalRecords * hitPercentage) / 100);
    const size_t numMisses = totalRecords - numHits;

    std::mt19937_64 rng(seed);

    const size_t hotSetSize = static_cast<size_t>(cacheSize);
    BenchmarkData hotSet;
    hotSet.reserve(hotSetSize);
    for (size_t i = 0; i < hotSetSize; ++i)
    {
        hotSet.emplace_back(createRecord(recordSize, static_cast<uint64_t>(i), rng));
    }

    std::vector<uint8_t> accessPattern(totalRecords, 0);
    std::fill_n(accessPattern.begin(), numHits, static_cast<uint8_t>(1));
    std::shuffle(accessPattern.begin(), accessPattern.end(), rng);

    BenchmarkData records;
    records.reserve(totalRecords);

    std::uniform_int_distribution<size_t> hotDist(0, hotSet.empty() ? 0 : hotSet.size() - 1);
    uint64_t missId = cacheSize;
    for (const uint8_t isHit : accessPattern)
    {
        if (isHit != 0)
        {
            const auto& hotRecord = hotSet[hotDist(rng)];
            auto record = std::make_unique<std::byte[]>(recordSize);
            std::memcpy(record.get(), hotRecord.get(), recordSize);
            records.emplace_back(std::move(record));
        }
        else
        {
            records.emplace_back(createRecord(recordSize, missId++, rng));
        }
    }

    return records;
}

[[maybe_unused]] size_t getPredictionCacheEntrySize(const NES::Configurations::PredictionCacheOptions& predictionCacheOptions)
{
    switch (predictionCacheOptions.predictionCacheType)
    {
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
        case NES::Configurations::PredictionCacheType::ALWAYS_MISS:
            return sizeof(NES::PredictionCacheEntryAlwaysMiss);
    }
    std::unreachable();
}

[[maybe_unused]] auto createPredictionCacheFillFunction(
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

                        const auto dataStructureRef = NES::getMemberRef(predictionCacheEntryToReplace, &NES::PredictionCacheEntry::dataStructure);
                        *NES::getMemberWithOffset<std::byte*>(dataStructureRef, 0) = nautilus::val<std::byte*>(std::make_unique<std::byte[]>(TUPLE_SIZE).get());

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
    for (auto rep = 0; rep < numReps; ++rep)
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

        benchmarkRunMeasurements.emplace_back(duration_cast<std::chrono::microseconds>(duration), hits, misses);
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
    const auto allPredictionCacheSizes = {100};
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
            auto benchmarkData = createBenchmarkData(predictionCacheSize, 1'000'000, hitMissRatio);

            for (const auto& predictionCacheType: allPredictionCacheTypes)
            {
                if (predictionCacheType == NES::Configurations::PredictionCacheType::NONE
                    || predictionCacheType == NES::Configurations::PredictionCacheType::TWO_QUEUES)
                {
                    continue;
                }

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
