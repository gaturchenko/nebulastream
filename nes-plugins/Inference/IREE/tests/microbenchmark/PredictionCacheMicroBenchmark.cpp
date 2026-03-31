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
#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
#include <PredictionCache/PredictionCache.hpp>
#include <PredictionCache/PredictionCacheAlwaysMiss.hpp>
#include <PredictionCache/PredictionCacheFIFO.hpp>
#include <PredictionCache/PredictionCacheLFU.hpp>
#include <PredictionCache/PredictionCacheLRU.hpp>
#include <PredictionCache/PredictionCacheSecondChance.hpp>
#include <PredictionCache/PredictionCacheUtil.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/BufferManager.hpp>
#include <Runtime/Execution/OperatorHandler.hpp>
#include <nautilus/val.hpp>
#include <nautilus/Engine.hpp>

#include "PredictionCacheDataGenerator.hpp"

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
#include <variant>
#include <unordered_map>
#include <vector>

namespace
{

size_t constexpr TUPLE_SIZE = 32;
int constexpr WARMUP_RUNS = 3;

using NES::Microbenchmark::BenchmarkData;
using NES::Microbenchmark::HitMissRatio;
using NES::Microbenchmark::getHitPercentage;

enum class DataGenerator
{
    Deterministic,
    Zipf,
    TemporalLocality,
    Burstiness
};

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
    struct DeterministicParameters
    {
        HitMissRatio hitMissRatio;
    };

    struct ZipfParameters
    {
        size_t numKeys;
        size_t numKeysMultiplier;
        double s;
        size_t driftInterval;
        double driftFraction;
    };

    struct TemporalLocalityParameters
    {
        size_t universeSize;
        size_t seriesLength;
        size_t windowSize;
        double overlapRatio;
        size_t driftInterval;
        double driftFraction;
    };

    struct BurstinessParameters
    {
        double dutyCycle;
        size_t onPeriod;
        size_t numKeys;
        size_t driftInterval;
        double driftFraction;
    };

    using DataGeneratorParameters
        = std::variant<DeterministicParameters, ZipfParameters, TemporalLocalityParameters, BurstinessParameters>;

    PredictionCacheOptionsMicroBenchmark predictionCacheOptions;
    DataGenerator dataGenerator;
    DataGeneratorParameters dataGeneratorParameters;

    std::string getValuesAsCsv() const
    {
        std::string hitsPercentage;
        std::string zipfNumKeys;
        std::string zipfNumKeysMultiplier;
        std::string zipfS;
        std::string temporalUniverseSize;
        std::string temporalSeriesLength;
        std::string temporalWindowSize;
        std::string temporalOverlapRatio;
        std::string burstinessTotalSteps;
        std::string burstinessDutyCycle;
        std::string burstinessLambdaAvg;
        std::string burstinessOnPeriod;
        std::string burstinessNumKeys;
        std::string driftInterval;
        std::string driftFraction;

        switch (dataGenerator)
        {
            case DataGenerator::Deterministic:
            {
                const auto* deterministicParameters = std::get_if<DeterministicParameters>(&dataGeneratorParameters);
                if (!deterministicParameters)
                {
                    throw std::invalid_argument("Deterministic data generator requires deterministic parameters");
                }
                hitsPercentage = fmt::format("{}", getHitPercentage(deterministicParameters->hitMissRatio));
                break;
            }
            case DataGenerator::Zipf:
            {
                const auto* zipfParameters = std::get_if<ZipfParameters>(&dataGeneratorParameters);
                if (!zipfParameters)
                {
                    throw std::invalid_argument("Zipf data generator requires Zipf parameters");
                }
                zipfNumKeys = fmt::format("{}", zipfParameters->numKeys);
                zipfNumKeysMultiplier = fmt::format("{}", zipfParameters->numKeysMultiplier);
                zipfS = fmt::format("{:.3f}", zipfParameters->s);
                driftInterval = fmt::format("{}", zipfParameters->driftInterval);
                driftFraction = fmt::format("{:.3f}", zipfParameters->driftFraction);
                break;
            }
            case DataGenerator::TemporalLocality:
            {
                const auto* temporalParameters = std::get_if<TemporalLocalityParameters>(&dataGeneratorParameters);
                if (!temporalParameters)
                {
                    throw std::invalid_argument("TemporalLocality data generator requires TemporalLocality parameters");
                }
                temporalUniverseSize = fmt::format("{}", temporalParameters->universeSize);
                temporalSeriesLength = fmt::format("{}", temporalParameters->seriesLength);
                temporalWindowSize = fmt::format("{}", temporalParameters->windowSize);
                temporalOverlapRatio = fmt::format("{:.3f}", temporalParameters->overlapRatio);
                driftInterval = fmt::format("{}", temporalParameters->driftInterval);
                driftFraction = fmt::format("{:.3f}", temporalParameters->driftFraction);
                break;
            }
            case DataGenerator::Burstiness:
            {
                const auto* burstinessParameters = std::get_if<BurstinessParameters>(&dataGeneratorParameters);
                if (!burstinessParameters)
                {
                    throw std::invalid_argument("Burstiness data generator requires Burstiness parameters");
                }
                burstinessDutyCycle = fmt::format("{:.3f}", burstinessParameters->dutyCycle);
                burstinessOnPeriod = fmt::format("{}", burstinessParameters->onPeriod);
                burstinessNumKeys = fmt::format("{}", burstinessParameters->numKeys);
                driftInterval = fmt::format("{}", burstinessParameters->driftInterval);
                driftFraction = fmt::format("{:.3f}", burstinessParameters->driftFraction);
                break;
            }
        }

        return fmt::format(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
            predictionCacheOptions.getValuesAsCsv(),
            magic_enum::enum_name(dataGenerator),
            hitsPercentage,
            zipfNumKeys,
            zipfNumKeysMultiplier,
            zipfS,
            temporalUniverseSize,
            temporalSeriesLength,
            temporalWindowSize,
            temporalOverlapRatio,
            burstinessTotalSteps,
            burstinessDutyCycle,
            burstinessLambdaAvg,
            burstinessOnPeriod,
            burstinessNumKeys,
            driftInterval,
            driftFraction);
    }

    static std::string getCsvHeader()
    {
        return fmt::format(
            "{},data_generator,hits_percentage,zipf_num_keys,zipf_num_keys_multiplier,zipf_s,temporal_universe_size,temporal_series_length,temporal_window_size,temporal_overlap_ratio,burstiness_total_steps,burstiness_duty_cycle,burstiness_lambda_avg,burstiness_on_period,burstiness_num_keys,drift_interval,drift_fraction",
            PredictionCacheOptionsMicroBenchmark::getCsvHeader());
    }
};

BenchmarkData createBenchmarkData(
    const BenchmarkParameters& benchmarkParams,
    const size_t totalRecords,
    const size_t recordSize = TUPLE_SIZE,
    const uint64_t seed = 0xC0FFEEULL)
{
    switch (benchmarkParams.dataGenerator)
    {
        case DataGenerator::Deterministic:
        {
            const auto* deterministicParameters
                = std::get_if<BenchmarkParameters::DeterministicParameters>(&benchmarkParams.dataGeneratorParameters);
            if (!deterministicParameters)
            {
                throw std::invalid_argument("Deterministic data generator requires deterministic parameters");
            }
            return NES::Microbenchmark::createDeterministicBenchmarkData(
                benchmarkParams.predictionCacheOptions.numberOfEntries,
                totalRecords,
                deterministicParameters->hitMissRatio,
                benchmarkParams.predictionCacheOptions.predictionCacheType,
                recordSize,
                seed);
        }
        case DataGenerator::Zipf:
        {
            const auto* zipfParameters = std::get_if<BenchmarkParameters::ZipfParameters>(&benchmarkParams.dataGeneratorParameters);
            if (!zipfParameters)
            {
                throw std::invalid_argument("Zipf data generator requires Zipf parameters");
            }
            return NES::Microbenchmark::createZipfBenchmarkData(
                zipfParameters->numKeys,
                totalRecords,
                recordSize,
                seed,
                zipfParameters->s,
                zipfParameters->driftInterval,
                zipfParameters->driftFraction);
        }
        case DataGenerator::TemporalLocality:
        {
            const auto* temporalParameters
                = std::get_if<BenchmarkParameters::TemporalLocalityParameters>(&benchmarkParams.dataGeneratorParameters);
            if (!temporalParameters)
            {
                throw std::invalid_argument("TemporalLocality data generator requires TemporalLocality parameters");
            }
            return NES::Microbenchmark::createTemporalLocalityBenchmarkData(
                temporalParameters->universeSize,
                temporalParameters->seriesLength,
                temporalParameters->windowSize,
                temporalParameters->overlapRatio,
                totalRecords,
                recordSize,
                seed,
                temporalParameters->driftInterval,
                temporalParameters->driftFraction);
        }
        case DataGenerator::Burstiness:
        {
            const auto* burstinessParameters
                = std::get_if<BenchmarkParameters::BurstinessParameters>(&benchmarkParams.dataGeneratorParameters);
            if (!burstinessParameters)
            {
                throw std::invalid_argument("Burstiness data generator requires Burstiness parameters");
            }
            return NES::Microbenchmark::createBurstinessBenchmarkData(
                burstinessParameters->dutyCycle,
                burstinessParameters->onPeriod,
                burstinessParameters->numKeys,
                totalRecords,
                recordSize,
                seed,
                burstinessParameters->driftInterval,
                burstinessParameters->driftFraction);
        }
    }
    std::unreachable();
}

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
            nautilus::val<size_t> inputSize,
            nautilus::val<NES::ChainedHashMap*> lookupIndex,
            nautilus::val<NES::AbstractBufferProvider*> bufferProvider)
        {
            using namespace nautilus;

            const val<int8_t*> globalOperatorHandler = nullptr;
            const auto predictionCache = NES::Util::createPredictionCache(
                copyOfPredictionCacheOptions, globalOperatorHandler, startOfEntries, inputSize);
            predictionCache->configureLookupIndex(lookupIndex, bufferProvider);

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
        auto bufferManager = NES::BufferManager::create();
        constexpr uint64_t minPageSize = 4096;
        const auto lookupIndexEntrySize = sizeof(NES::ChainedHashMapEntry) + TUPLE_SIZE + 2 * sizeof(uint64_t);
        const auto lookupIndexPageSize = std::max(static_cast<uint64_t>(lookupIndexEntrySize), minPageSize);
        auto lookupIndex = std::make_unique<NES::ChainedHashMap>(
            TUPLE_SIZE,
            2 * sizeof(uint64_t),
            benchmarkParams.predictionCacheOptions.numberOfEntries,
            lookupIndexPageSize);

        const auto startTime = std::chrono::high_resolution_clock::now();

        predictionCacheFunction(
            benchmarkDataRefs.data(),
            benchmarkDataRefs.size(),
            predictionCacheMemory.data(),
            TUPLE_SIZE,
            lookupIndex.get(),
            bufferManager.get());

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
    // constexpr auto allPredictionCacheTypes = std::array{magic_enum::enum_value<NES::Configurations::PredictionCacheType, 4>()};
    const auto allPredictionCacheSizes = {100, 1'000, 10'000};

    const auto allDeterministicHitMissRatios = magic_enum::enum_values<HitMissRatio>();
    // const auto allDeterministicHitMissRatios = std::array{magic_enum::enum_value<HitMissRatio, 2>()};

    const auto allDriftIntervals = {1'000};
    const auto allDriftFractions = {0.0, 0.1, 0.5, 1.0};

    const auto allZipfSValues = {0.0, 0.6, 1.0, 1.2};
    const auto allZipfNumKeyMultipliers = {10};

    const auto allTemporalUniverseSizes = {5'000};
    const auto allTemporalSeriesLengthMultipliers = {10};
    const auto allTemporalWindowSizes = {100};
    const auto allTemporalOverlapRatios = {0.0, 0.5, 0.8, 0.95};

    const auto allBurstinessDutyCycles = {0.01, 0.05, 0.2, 0.5};
    const auto allBurstinessOnPeriods = {1'000};
    const auto allBurstinessNumKeyMultipliers = {10};

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

    const size_t runsPerCacheConfiguration =
        allDriftFractions.size() * allDriftIntervals.size() *
        allZipfNumKeyMultipliers.size() * allZipfSValues.size() *
        allTemporalOverlapRatios.size() * allTemporalSeriesLengthMultipliers.size() * allTemporalUniverseSizes.size() * allTemporalWindowSizes.size() *
        allBurstinessDutyCycles.size() * allBurstinessOnPeriods.size() * allBurstinessNumKeyMultipliers.size();

    ETACalculator etaCalculator(
        allPredictionCacheSizes.size() * (allPredictionCacheTypes.size() - 3) * runsPerCacheConfiguration);

    for (const uint64_t predictionCacheSize: allPredictionCacheSizes)
    {
        for (const auto& predictionCacheType: allPredictionCacheTypes)
        {
            if (predictionCacheType == NES::Configurations::PredictionCacheType::NONE
                || predictionCacheType == NES::Configurations::PredictionCacheType::TWO_QUEUES
                || predictionCacheType == NES::Configurations::PredictionCacheType::ALWAYS_MISS)
            {
                continue;
            }

            for (const auto& hitMissRatio: allDeterministicHitMissRatios)
            {
                BenchmarkParameters benchmarkParams{
                    PredictionCacheOptionsMicroBenchmark{predictionCacheType, predictionCacheSize},
                    DataGenerator::Deterministic,
                    BenchmarkParameters::DeterministicParameters{hitMissRatio}};
                auto benchmarkData = createBenchmarkData(benchmarkParams, 1'000'000);
                const auto results = runBenchmark(benchmarkParams, REPS, benchmarkData);

                for (const auto& result : results)
                {
                    csvFile << createNewCsvFileLine(benchmarkParams, result) << std::endl;
                    std::cout << createNewCsvFileLine(benchmarkParams, result) << std::endl;
                }
                csvFile.flush();
                // etaCalculator.update();
            }

            // for (const auto& driftInterval : allDriftIntervals)
            // {
            //     for (const auto& driftFraction : allDriftFractions)
            //     {
            //         for (const auto& zipfNumKeyMultiplier : allZipfNumKeyMultipliers)
            //         {
            //             const size_t zipfNumKeys = std::max<size_t>(1, static_cast<size_t>(predictionCacheSize) * zipfNumKeyMultiplier);
            //             for (const double zipfS : allZipfSValues)
            //             {
            //                 BenchmarkParameters benchmarkParams{
            //                     PredictionCacheOptionsMicroBenchmark{predictionCacheType, predictionCacheSize},
            //                     DataGenerator::Zipf,
            //                     BenchmarkParameters::ZipfParameters{
            //                         zipfNumKeys,
            //                         static_cast<size_t>(zipfNumKeyMultiplier),
            //                         zipfS,
            //                         static_cast<size_t>(driftInterval),
            //                         driftFraction}};
            //                 auto benchmarkData = createBenchmarkData(benchmarkParams, 1'000'000);
            //                 const auto results = runBenchmark(benchmarkParams, REPS, benchmarkData);
            //
            //                 for (const auto& result : results)
            //                 {
            //                     csvFile << createNewCsvFileLine(benchmarkParams, result) << std::endl;
            //                     std::cout << createNewCsvFileLine(benchmarkParams, result) << std::endl;
            //                 }
            //                 csvFile.flush();
            //                 etaCalculator.update();
            //             }
            //         }
            //
            //         for (const auto& temporalSeriesLengthMultiplier : allTemporalSeriesLengthMultipliers)
            //         {
            //             for (const auto& temporalWindowSize : allTemporalWindowSizes)
            //             {
            //                 const size_t temporalSeriesLength = std::max<size_t>(
            //                     static_cast<size_t>(temporalWindowSize),
            //                     static_cast<size_t>(predictionCacheSize) * temporalSeriesLengthMultiplier);
            //                 for (const auto& temporalOverlapRatio : allTemporalOverlapRatios)
            //                 {
            //                     for (const auto& temporalUniverseSize : allTemporalUniverseSizes)
            //                     {
            //                         BenchmarkParameters benchmarkParams{
            //                             PredictionCacheOptionsMicroBenchmark{predictionCacheType, predictionCacheSize},
            //                             DataGenerator::TemporalLocality,
            //                             BenchmarkParameters::TemporalLocalityParameters{
            //                                 static_cast<size_t>(temporalUniverseSize),
            //                                 temporalSeriesLength,
            //                                 static_cast<size_t>(temporalWindowSize),
            //                                 temporalOverlapRatio,
            //                                 static_cast<size_t>(driftInterval),
            //                                 driftFraction}};
            //
            //                         auto benchmarkData = createBenchmarkData(benchmarkParams, 1'000'000);
            //                         const auto results = runBenchmark(benchmarkParams, REPS, benchmarkData);
            //
            //                         for (const auto& result : results)
            //                         {
            //                             csvFile << createNewCsvFileLine(benchmarkParams, result) << std::endl;
            //                             std::cout << createNewCsvFileLine(benchmarkParams, result) << std::endl;
            //                         }
            //                         csvFile.flush();
            //                         etaCalculator.update();
            //                     }
            //                 }
            //             }
            //         }
            //
            //         for (const auto& burstinessNumKeyMultiplier : allBurstinessNumKeyMultipliers)
            //         {
            //             const size_t burstinessNumKeys = std::max<size_t>(1, static_cast<size_t>(predictionCacheSize) * burstinessNumKeyMultiplier);
            //             for (const auto& burstinessDutyCycle : allBurstinessDutyCycles)
            //             {
            //                 for (const auto& burstinessOnPeriod : allBurstinessOnPeriods)
            //                 {
            //                     BenchmarkParameters benchmarkParams{
            //                         PredictionCacheOptionsMicroBenchmark{predictionCacheType, predictionCacheSize},
            //                         DataGenerator::Burstiness,
            //                         BenchmarkParameters::BurstinessParameters{
            //                             burstinessDutyCycle,
            //                             static_cast<size_t>(burstinessOnPeriod),
            //                             burstinessNumKeys,
            //                             static_cast<size_t>(driftInterval),
            //                             driftFraction}};
            //
            //                     auto benchmarkData = createBenchmarkData(benchmarkParams, 1'000'000);
            //                     const auto results = runBenchmark(benchmarkParams, REPS, benchmarkData);
            //
            //                     for (const auto& result : results)
            //                     {
            //                         csvFile << createNewCsvFileLine(benchmarkParams, result) << std::endl;
            //                         std::cout << createNewCsvFileLine(benchmarkParams, result) << std::endl;
            //                     }
            //                     csvFile.flush();
            //                     etaCalculator.update();
            //                 }
            //             }
            //         }
            //     }
            // }
        }
    }

    return 0;
}
