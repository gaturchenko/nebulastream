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

#include <Inference/PredictionCache/PredictionCache.hpp>
#include <Inference/PredictionCache/PredictionCacheEntry.hpp>
#include <Inference/PredictionCache/PredictionCacheUtil.hpp>
#include <PerfEvent.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <Nautilus/DataTypes/DataTypesUtil.hpp>
#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
#include <Runtime/AbstractBufferProvider.hpp>
#include <Runtime/BufferManager.hpp>
#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <magic_enum/magic_enum.hpp>
#include <Engine.hpp>
#include <InferenceConfiguration.hpp>
#include <function.hpp>
#include <options.hpp>
#include <val_arith.hpp>

namespace NES
{

namespace
{

constexpr uint64_t lookupIndexPageSize = 4096;
constexpr uint64_t recordSize = sizeof(uint64_t);
constexpr uint64_t dataSize = sizeof(uint64_t);

struct BenchmarkOperation
{
    uint64_t key;
    uint64_t value;
};

struct BenchmarkConfig
{
    std::filesystem::path outputFile = std::filesystem::path("PredictionCacheMicrobenchmark.csv");
    uint64_t records = 1'000'000;
    uint64_t repetitions = 10;
    uint64_t warmups = 2;
    std::vector<uint64_t> capacities = {1, 16, 256, 4096};
};

struct HitMissRatio
{
    uint64_t hitPercent;
    uint64_t missPercent;
};

struct OperationSet
{
    std::vector<BenchmarkOperation> operations;
    uint64_t expectedHits;
    uint64_t expectedMisses;
};

struct PerfCounters
{
    double durationSeconds = 0;
    double cycles = 0;
    double instructions = 0;
    double l1Misses = 0;
    double llcMisses = 0;
    double cacheMisses = 0;
    double branchInstructions = 0;
    double branchMisses = 0;
    double dtlbMisses = 0;
    double taskClock = 0;
};

struct RunResult
{
    uint64_t checksum;
    uint64_t hits;
    uint64_t misses;
    PerfCounters perfCounters;
};

struct Measurement
{
    uint64_t run;
    double durationUs;
    double throughputGetsPerSecond;
    double cyclesPerGet;
    double instructionsPerGet;
    double l1MissesPerGet;
    double llcMissesPerGet;
    double cacheMissesPerKInstr;
    double branchMissRate;
    double dtlbMissesPerGet;
    double ipc;
    double ghz;
    uint64_t checksum;
};

using CacheFunctionDefinition = std::function<nautilus::val<uint64_t>(
    nautilus::val<int8_t*>, nautilus::val<BenchmarkOperation*>, nautilus::val<ChainedHashMap*>, nautilus::val<AbstractBufferProvider*>)>;

void replacePredictionCacheBenchmarkEntry(
    PredictionCacheEntry* entry, std::byte* record, std::byte* dataStructure, const size_t entryRecordSize, const size_t entryDataSize)
{
    delete[] entry->record;
    delete[] entry->dataStructure;

    entry->recordSize = entryRecordSize;
    entry->record = new std::byte[entryRecordSize];
    std::memcpy(entry->record, record, entryRecordSize);

    entry->dataSize = entryDataSize;
    entry->dataStructure = new std::byte[entryDataSize];
    std::memcpy(entry->dataStructure, dataStructure, entryDataSize);
}

void releaseCacheEntries(std::vector<std::byte>& cacheMemory, const PredictionCacheType cacheType, const uint64_t capacity)
{
    if (cacheMemory.empty())
    {
        return;
    }

    const auto entrySize = Util::getPredictionCacheEntrySize(cacheType);
    auto* entries = cacheMemory.data() + sizeof(HitsAndMisses);
    for (uint64_t i = 0; i < capacity; ++i)
    {
        auto* entry = reinterpret_cast<PredictionCacheEntry*>(entries + i * entrySize);
        delete[] entry->record;
        delete[] entry->dataStructure;
        entry->record = nullptr;
        entry->dataStructure = nullptr;
    }
}

uint64_t desiredHitsForRatio(const uint64_t records, const HitMissRatio ratio)
{
    if (ratio.hitPercent == 100)
    {
        return records - 1;
    }
    return (records * ratio.hitPercent) / 100;
}

OperationSet makeGroupedOperations(const uint64_t records, const HitMissRatio ratio, const uint64_t keySeed)
{
    const auto expectedHits = desiredHitsForRatio(records, ratio);
    const auto expectedMisses = records - expectedHits;
    OperationSet operationSet;
    operationSet.operations.reserve(records);
    operationSet.expectedHits = expectedHits;
    operationSet.expectedMisses = expectedMisses;

    uint64_t remainingHits = expectedHits;
    uint64_t remainingMisses = expectedMisses;
    uint64_t key = keySeed;
    while (remainingMisses > 0)
    {
        const auto currentKey = key++;
        operationSet.operations.push_back(BenchmarkOperation{.key = currentKey, .value = currentKey ^ UINT64_C(0x9e3779b97f4a7c15)});
        --remainingMisses;

        const auto hitsForThisMiss = remainingMisses == 0 ? remainingHits : remainingHits / (remainingMisses + 1);
        for (uint64_t i = 0; i < hitsForThisMiss; ++i)
        {
            operationSet.operations.push_back(BenchmarkOperation{.key = currentKey, .value = currentKey ^ UINT64_C(0x9e3779b97f4a7c15)});
        }
        remainingHits -= hitsForThisMiss;
    }

    if (operationSet.operations.size() != records)
    {
        throw std::runtime_error("prediction cache benchmark generator produced an unexpected number of operations");
    }
    return operationSet;
}

OperationSet makeFifoOperations(const uint64_t records, const HitMissRatio ratio, const uint64_t capacity)
{
    return makeGroupedOperations(records, ratio, UINT64_C(0xF1F0'0000) + capacity * 17);
}

OperationSet makeLfuOperations(const uint64_t records, const HitMissRatio ratio, const uint64_t capacity)
{
    return makeGroupedOperations(records, ratio, UINT64_C(0x1F00'0000) + capacity * 31);
}

OperationSet makeLruOperations(const uint64_t records, const HitMissRatio ratio, const uint64_t capacity)
{
    return makeGroupedOperations(records, ratio, UINT64_C(0x1A00'0000) + capacity * 47);
}

OperationSet makeSecondChanceOperations(const uint64_t records, const HitMissRatio ratio, const uint64_t capacity)
{
    return makeGroupedOperations(records, ratio, UINT64_C(0x5EC0'0000) + capacity * 67);
}

OperationSet makeOperations(const PredictionCacheType cacheType, const uint64_t records, const HitMissRatio ratio, const uint64_t capacity)
{
    switch (cacheType)
    {
        case PredictionCacheType::FIFO:
            return makeFifoOperations(records, ratio, capacity);
        case PredictionCacheType::LFU:
            return makeLfuOperations(records, ratio, capacity);
        case PredictionCacheType::LRU:
            return makeLruOperations(records, ratio, capacity);
        case PredictionCacheType::SECOND_CHANCE:
            return makeSecondChanceOperations(records, ratio, capacity);
        case PredictionCacheType::NONE:
            break;
    }
    throw std::invalid_argument("prediction cache type NONE has no cache GET benchmark");
}

double missingMetric()
{
    return std::numeric_limits<double>::quiet_NaN();
}

double safeDivide(const double numerator, const double denominator)
{
    if (numerator < 0 || denominator <= 0)
    {
        return missingMetric();
    }
    return numerator / denominator;
}

void writeMetric(std::ostream& csv, const double value)
{
    if (std::isnan(value))
    {
        return;
    }
    csv << value;
}

void addPerfWindow(PerfCounters& counters, PerfEvent& perfEvent)
{
    counters.durationSeconds += perfEvent.getDuration();
    counters.cycles += perfEvent.getCounter("cycle");
    counters.instructions += perfEvent.getCounter("instr");
    counters.l1Misses += perfEvent.getCounter("L1-miss");
    counters.llcMisses += perfEvent.getCounter("LLC-miss");
    counters.cacheMisses += perfEvent.getCounter("cache-miss");
    counters.branchInstructions += perfEvent.getCounter("branch");
    counters.branchMisses += perfEvent.getCounter("br-miss");
    counters.dtlbMisses += perfEvent.getCounter("dTLB-miss");
    counters.taskClock += perfEvent.getCounter("task");
}

Measurement makeMeasurement(const uint64_t run, const RunResult& runResult, const uint64_t numberOfGets, const uint64_t checksum)
{
    const auto duration = runResult.perfCounters.durationSeconds;
    return Measurement{
        .run = run,
        .durationUs = duration * 1'000'000.0,
        .throughputGetsPerSecond = safeDivide(static_cast<double>(numberOfGets), duration),
        .cyclesPerGet = safeDivide(runResult.perfCounters.cycles, static_cast<double>(numberOfGets)),
        .instructionsPerGet = safeDivide(runResult.perfCounters.instructions, static_cast<double>(numberOfGets)),
        .l1MissesPerGet = safeDivide(runResult.perfCounters.l1Misses, static_cast<double>(numberOfGets)),
        .llcMissesPerGet = safeDivide(runResult.perfCounters.llcMisses, static_cast<double>(numberOfGets)),
        .cacheMissesPerKInstr = safeDivide(runResult.perfCounters.cacheMisses, runResult.perfCounters.instructions / 1'000.0),
        .branchMissRate = safeDivide(runResult.perfCounters.branchMisses, runResult.perfCounters.branchInstructions),
        .dtlbMissesPerGet = safeDivide(runResult.perfCounters.dtlbMisses, static_cast<double>(numberOfGets)),
        .ipc = safeDivide(runResult.perfCounters.instructions, runResult.perfCounters.cycles),
        .ghz = safeDivide(runResult.perfCounters.cycles, runResult.perfCounters.taskClock),
        .checksum = checksum};
}

void writePerfMetrics(std::ostream& csv, const Measurement& measurement)
{
    csv << ',';
    writeMetric(csv, measurement.cyclesPerGet);
    csv << ',';
    writeMetric(csv, measurement.instructionsPerGet);
    csv << ',';
    writeMetric(csv, measurement.l1MissesPerGet);
    csv << ',';
    writeMetric(csv, measurement.llcMissesPerGet);
    csv << ',';
    writeMetric(csv, measurement.cacheMissesPerKInstr);
    csv << ',';
    writeMetric(csv, measurement.branchMissRate);
    csv << ',';
    writeMetric(csv, measurement.dtlbMissesPerGet);
    csv << ',';
    writeMetric(csv, measurement.ipc);
    csv << ',';
    writeMetric(csv, measurement.ghz);
}

std::vector<uint64_t> parseUIntList(std::string_view raw)
{
    std::vector<uint64_t> values;
    std::stringstream stream{std::string(raw)};
    std::string item;
    while (std::getline(stream, item, ','))
    {
        if (item.empty())
        {
            continue;
        }

        const auto value = std::stoull(item);
        if (value > 0)
        {
            values.push_back(value);
        }
    }
    return values;
}

uint64_t readEnvUInt(const char* name, const uint64_t defaultValue)
{
    const char* value = std::getenv(name);
    if (value == nullptr || std::string_view(value).empty())
    {
        return defaultValue;
    }
    return std::stoull(value);
}

std::vector<uint64_t> readEnvCapacities()
{
    const char* value = std::getenv("NES_PREDICTION_CACHE_MICROBENCH_CAPACITIES");
    if (value == nullptr || std::string_view(value).empty())
    {
        return {1, 16, 256, 4096};
    }
    return parseUIntList(value);
}

void printUsage(const char* binary)
{
    std::cout << "Usage: " << binary << " [--output FILE] [--records N] [--repetitions N] [--warmups N] [--capacities 1,16,256]\n";
}

BenchmarkConfig parseConfig(const int argc, char** argv)
{
    BenchmarkConfig config{
        .outputFile = std::filesystem::path(
            std::getenv("NES_PREDICTION_CACHE_MICROBENCH_OUTPUT") != nullptr ? std::getenv("NES_PREDICTION_CACHE_MICROBENCH_OUTPUT")
                                                                             : "PredictionCacheMicrobenchmark.csv"),
        .records = readEnvUInt("NES_PREDICTION_CACHE_MICROBENCH_RECORDS", 1'000'000),
        .repetitions = readEnvUInt("NES_PREDICTION_CACHE_MICROBENCH_REPETITIONS", 10),
        .warmups = readEnvUInt("NES_PREDICTION_CACHE_MICROBENCH_WARMUPS", 2),
        .capacities = readEnvCapacities()};

    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg(argv[i]);
        const auto nextValue = [&](const std::string_view option) -> std::string_view
        {
            if (i + 1 >= argc)
            {
                throw std::invalid_argument("Missing value for " + std::string(option));
            }
            ++i;
            return argv[i];
        };

        if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (arg == "--output" || arg == "--output-file" || arg == "--output_file")
        {
            config.outputFile = std::filesystem::path(std::string(nextValue(arg)));
        }
        else if (arg == "--records")
        {
            config.records = std::stoull(std::string(nextValue(arg)));
        }
        else if (arg == "--repetitions")
        {
            config.repetitions = std::stoull(std::string(nextValue(arg)));
        }
        else if (arg == "--warmups")
        {
            config.warmups = std::stoull(std::string(nextValue(arg)));
        }
        else if (arg == "--capacities")
        {
            config.capacities = parseUIntList(nextValue(arg));
        }
        else
        {
            throw std::invalid_argument("Unknown argument: " + std::string(arg));
        }
    }

    if (config.records < 2)
    {
        throw std::invalid_argument("records must be at least 2 so the 100-0 scenario can have one miss and remaining hits");
    }
    if (config.records % 4 != 0)
    {
        throw std::invalid_argument("records must be divisible by 4 to produce exact 25-75, 50-50, and 75-25 hit-miss ratios");
    }
    if (config.repetitions == 0)
    {
        throw std::invalid_argument("repetitions must be greater than zero");
    }
    if (config.capacities.empty())
    {
        throw std::invalid_argument("at least one prediction cache capacity must be configured");
    }

    std::ranges::sort(config.capacities);
    config.capacities.erase(std::ranges::unique(config.capacities).begin(), config.capacities.end());
    return config;
}

auto compileCacheFunction(
    nautilus::engine::NautilusEngine& nautilusEngine,
    const PredictionCacheType cacheType,
    const uint64_t capacity,
    const uint64_t operationCount)
{
    return nautilusEngine.registerFunction(CacheFunctionDefinition(
        [cacheType, capacity, operationCount](
            const nautilus::val<int8_t*>& cacheStart,
            const nautilus::val<BenchmarkOperation*>& operationsStart,
            const nautilus::val<ChainedHashMap*>& lookupIndex,
            const nautilus::val<AbstractBufferProvider*>& bufferProvider) -> nautilus::val<uint64_t>
        {
            auto predictionCache = Util::createPredictionCache(cacheType, capacity, cacheStart, recordSize);
            predictionCache->configureLookupIndex(lookupIndex, bufferProvider);
            nautilus::val<uint64_t> checksum = 0;

            for (nautilus::val<uint64_t> i = 0; i < operationCount; ++i)
            {
                const auto operation = operationsStart + i;
                const auto operationBytes = static_cast<nautilus::val<int8_t*>>(operation);
                const auto record = static_cast<nautilus::val<std::byte*>>(getMemberRef(operationBytes, &BenchmarkOperation::key));
                const auto dataStructure = static_cast<nautilus::val<std::byte*>>(getMemberRef(operationBytes, &BenchmarkOperation::value));

                const auto result = predictionCache->getDataStructureRef(
                    record,
                    [&](const nautilus::val<PredictionCacheEntry*>& entryToReplace, const nautilus::val<uint64_t>&)
                    {
                        nautilus::invoke(
                            replacePredictionCacheBenchmarkEntry,
                            entryToReplace,
                            record,
                            dataStructure,
                            nautilus::val<size_t>(recordSize),
                            nautilus::val<size_t>(dataSize));
                        const auto dataStructureRef = getMemberRef(entryToReplace, &PredictionCacheEntry::dataStructure);
                        return *getMemberWithOffset<std::byte*>(dataStructureRef, 0);
                    });

                checksum = checksum + nautilus::val<uint64_t>(*static_cast<nautilus::val<uint64_t*>>(result));
            }
            return checksum;
        }));
}

template <typename CacheFunction>
RunResult runOnce(
    CacheFunction& cacheFunction,
    const PredictionCacheType cacheType,
    const uint64_t capacity,
    std::vector<BenchmarkOperation>& operations,
    PerfEvent& perfEvent)
{
    std::vector<std::byte> cacheMemory(sizeof(HitsAndMisses) + capacity * Util::getPredictionCacheEntrySize(cacheType), std::byte{0});
    auto bufferManager = BufferManager::create(lookupIndexPageSize, 1);
    ChainedHashMap lookupIndex(0, sizeof(uint64_t), std::max<uint64_t>(capacity, 1), lookupIndexPageSize);
    PerfCounters perfCounters;

    perfEvent.startCounters();
    const auto checksum
        = cacheFunction(reinterpret_cast<int8_t*>(cacheMemory.data()), operations.data(), &lookupIndex, bufferManager.get());
    perfEvent.stopCounters();
    addPerfWindow(perfCounters, perfEvent);

    const auto* hitMiss = reinterpret_cast<const HitsAndMisses*>(cacheMemory.data());
    const auto hits = hitMiss->hits;
    const auto misses = hitMiss->misses;
    releaseCacheEntries(cacheMemory, cacheType, capacity);
    return RunResult{.checksum = checksum, .hits = hits, .misses = misses, .perfCounters = perfCounters};
}

int runPredictionCacheMicrobenchmark(const BenchmarkConfig& config)
{
    constexpr std::array policies{
        PredictionCacheType::FIFO, PredictionCacheType::LFU, PredictionCacheType::LRU, PredictionCacheType::SECOND_CHANCE};
    constexpr std::array ratios{
        HitMissRatio{.hitPercent = 0, .missPercent = 100},
        HitMissRatio{.hitPercent = 25, .missPercent = 75},
        HitMissRatio{.hitPercent = 50, .missPercent = 50},
        HitMissRatio{.hitPercent = 75, .missPercent = 25},
        HitMissRatio{.hitPercent = 100, .missPercent = 0}};

    std::ofstream csv(config.outputFile);
    if (!csv.is_open())
    {
        throw std::runtime_error("Failed to open benchmark output CSV: " + config.outputFile.string());
    }

    std::cerr << "Prediction cache microbenchmark"
              << " records=" << config.records << " repetitions=" << config.repetitions << " warmups=" << config.warmups
              << " output=" << config.outputFile << '\n';
    csv << "policy,capacity,hit_miss_ratio,requested_hit_percent,requested_miss_percent,actual_hit_percent,actual_miss_percent,"
           "run,status,duration_us,throughput_gets_per_second,avg_us,min_us,max_us,avg_gets_per_second,gets,hits,misses,checksum,"
           "cycles_per_get,instructions_per_get,l1_misses_per_get,llc_misses_per_get,cache_misses_per_kinstr,branch_miss_rate,"
           "dtlb_misses_per_get,ipc,ghz\n";

    nautilus::engine::Options options;
    options.setOption("engine.Compilation", true);
    options.setOption("mlir.enableMultithreading", false);
    nautilus::engine::NautilusEngine nautilusEngine(options);
    PerfEvent perfEvent;

    for (const auto policy : policies)
    {
        for (const auto capacity : config.capacities)
        {
            auto cacheFunction = compileCacheFunction(nautilusEngine, policy, capacity, config.records);
            for (const auto ratio : ratios)
            {
                auto operationSet = makeOperations(policy, config.records, ratio, capacity);
                std::vector<Measurement> measurements;
                measurements.reserve(config.repetitions);
                uint64_t observedHits = 0;
                uint64_t observedMisses = 0;

                for (uint64_t iteration = 0; iteration < config.warmups + config.repetitions; ++iteration)
                {
                    const auto runResult = runOnce(cacheFunction, policy, capacity, operationSet.operations, perfEvent);
                    if (runResult.hits != operationSet.expectedHits || runResult.misses != operationSet.expectedMisses)
                    {
                        throw std::runtime_error(
                            "Expected " + std::to_string(operationSet.expectedHits) + " hits and "
                            + std::to_string(operationSet.expectedMisses) + " misses for policy "
                            + std::string(magic_enum::enum_name(policy)) + ", got " + std::to_string(runResult.hits) + " hits and "
                            + std::to_string(runResult.misses) + " misses");
                    }

                    observedHits = runResult.hits;
                    observedMisses = runResult.misses;
                    if (iteration >= config.warmups)
                    {
                        measurements.push_back(makeMeasurement(iteration - config.warmups, runResult, config.records, runResult.checksum));
                    }
                }

                const auto [minIt, maxIt] = std::minmax_element(
                    measurements.begin(),
                    measurements.end(),
                    [](const auto& left, const auto& right) { return left.durationUs < right.durationUs; });
                const auto totalUs = std::accumulate(
                    measurements.begin(),
                    measurements.end(),
                    0.0,
                    [](const auto sum, const auto& measurement) { return sum + measurement.durationUs; });
                const auto avgUs = totalUs / static_cast<double>(measurements.size());
                const auto avgGetsPerSecond = static_cast<double>(config.records) * 1'000'000.0 / avgUs;
                const auto actualHitPercent = safeDivide(static_cast<double>(observedHits) * 100.0, static_cast<double>(config.records));
                const auto actualMissPercent = safeDivide(static_cast<double>(observedMisses) * 100.0, static_cast<double>(config.records));
                const auto ratioLabel = std::to_string(ratio.hitPercent) + "-" + std::to_string(ratio.missPercent);

                for (const auto& measurement : measurements)
                {
                    csv << magic_enum::enum_name(policy) << ',' << capacity << ',' << ratioLabel << ',' << ratio.hitPercent << ','
                        << ratio.missPercent << ',';
                    writeMetric(csv, actualHitPercent);
                    csv << ',';
                    writeMetric(csv, actualMissPercent);
                    csv << ',' << measurement.run << ",ok," << measurement.durationUs << ',' << measurement.throughputGetsPerSecond << ','
                        << avgUs << ',' << minIt->durationUs << ',' << maxIt->durationUs << ',' << avgGetsPerSecond << ',' << config.records
                        << ',' << observedHits << ',' << observedMisses << ',' << measurement.checksum;
                    writePerfMetrics(csv, measurement);
                    csv << '\n';
                }
            }
        }
    }

    csv.flush();
    if (!csv.good())
    {
        throw std::runtime_error("Failed while writing benchmark output CSV: " + config.outputFile.string());
    }
    std::cerr << "Wrote benchmark CSV to " << config.outputFile << '\n';
    return EXIT_SUCCESS;
}

}

}

int main(int argc, char** argv)
{
    try
    {
        NES::Logger::setupLogging("PredictionCacheMicrobenchmark.log", NES::LogLevel::LOG_ERROR);
        return NES::runPredictionCacheMicrobenchmark(NES::parseConfig(argc, argv));
    }
    catch (const std::exception& ex)
    {
        std::cerr << "PredictionCacheMicrobenchmark failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}
