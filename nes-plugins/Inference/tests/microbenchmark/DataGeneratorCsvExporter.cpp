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

#include "PredictionCacheDataGenerator.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <cmath>
#include <string_view>

namespace
{

enum class GeneratorChoice
{
    Deterministic,
    Zipf,
    TemporalLocality,
    Burstiness
};

GeneratorChoice parseGeneratorChoice(const std::string& value)
{
    if (value == "deterministic")
    {
        return GeneratorChoice::Deterministic;
    }
    if (value == "zipf")
    {
        return GeneratorChoice::Zipf;
    }
    if (value == "temporal-locality")
    {
        return GeneratorChoice::TemporalLocality;
    }
    if (value == "burstiness")
    {
        return GeneratorChoice::Burstiness;
    }

    throw std::invalid_argument("Unknown generator: " + value);
}

NES::Microbenchmark::HitMissRatio parseHitMissRatio(const std::string& value)
{
    if (value == "100")
    {
        return NES::Microbenchmark::HitMissRatio::Hits100;
    }
    if (value == "75")
    {
        return NES::Microbenchmark::HitMissRatio::Hits75;
    }
    if (value == "50")
    {
        return NES::Microbenchmark::HitMissRatio::Hits50;
    }
    if (value == "25")
    {
        return NES::Microbenchmark::HitMissRatio::Hits25;
    }
    if (value == "0")
    {
        return NES::Microbenchmark::HitMissRatio::Hits0;
    }

    throw std::invalid_argument("Invalid deterministic hit ratio. Use one of: 100, 75, 50, 25, 0");
}

NES::Configurations::PredictionCacheType parsePredictionCacheType(const std::string& value)
{
    if (value == "fifo")
    {
        return NES::Configurations::PredictionCacheType::FIFO;
    }
    if (value == "lfu")
    {
        return NES::Configurations::PredictionCacheType::LFU;
    }
    if (value == "lru")
    {
        return NES::Configurations::PredictionCacheType::LRU;
    }
    if (value == "second_chance")
    {
        return NES::Configurations::PredictionCacheType::SECOND_CHANCE;
    }

    throw std::invalid_argument("Invalid deterministic cache type. Use one of: fifo, lfu, lru, second_chance");
}

struct ParsedArguments
{
    GeneratorChoice generator = GeneratorChoice::Zipf;
    std::filesystem::path outputPath = "generated_data.csv";
    size_t records = 100'000;
    size_t recordSize = 32;
    uint64_t seed = 0xC0FFEEULL;
    size_t driftInterval = 0;
    double driftFraction = 0.0;

    /// Deterministic parameters
    uint64_t deterministicCacheSize = 1'000;
    NES::Microbenchmark::HitMissRatio deterministicHitRatio = NES::Microbenchmark::HitMissRatio::Hits50;
    NES::Configurations::PredictionCacheType deterministicCacheType = NES::Configurations::PredictionCacheType::LRU;

    /// Zipf parameters
    size_t zipfNumKeys = 10'000;
    double zipfS = 1.0;

    /// TemporalLocality parameters
    size_t temporalUniverseSize = 10'000;
    size_t temporalSeriesLength = 10'000;
    size_t temporalWindowSize = 128;
    double temporalOverlapRatio = 0.5;

    /// Burstiness parameters
    size_t burstinessTotalSteps = 1'000;
    double burstinessDutyCycle = 0.5;
    double burstinessLambdaAvg = 1.0;
    size_t burstinessOnPeriod = 50;
    size_t burstinessNumKeys = 10'000;
};

void printUsage()
{
    std::cerr
        << "Usage:\n"
        << "  DataGeneratorCsvExporter [options]\n\n"
        << "Common options:\n"
        << "  --generator <deterministic|zipf|temporal-locality|burstiness>\n"
        << "  --output <path>\n"
        << "  --records <int>\n"
        << "  --record-size <int>\n"
        << "  --seed <int>\n\n"
        << "  --drift-interval <int>\n"
        << "  --drift-fraction <float in [0,1]>\n\n"
        << "Deterministic options:\n"
        << "  --det-cache-size <int>\n"
        << "  --det-hit-ratio <100|75|50|25|0>\n"
        << "  --det-cache-type <fifo|lfu|lru|second_chance>\n\n"
        << "Zipf options:\n"
        << "  --zipf-num-keys <int>\n"
        << "  --zipf-s <float>\n\n"
        << "TemporalLocality options:\n"
        << "  --temporal-universe-size <int>\n"
        << "  --temporal-series-length <int>\n"
        << "  --temporal-window-size <int>\n"
        << "  --temporal-overlap-ratio <float in [0,1)>\n\n"
        << "Burstiness options:\n"
        << "  --burstiness-duty-cycle <float in (0,1]>\n"
        << "  --burstiness-on-period <int>\n"
        << "  --burstiness-num-keys <int>\n";
}

size_t parseSize(const std::string& value, const std::string& option)
{
    try
    {
        return static_cast<size_t>(std::stoull(value));
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("Invalid value for " + option + ": " + value);
    }
}

uint64_t parseUInt64(const std::string& value, const std::string& option)
{
    try
    {
        return std::stoull(value);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("Invalid value for " + option + ": " + value);
    }
}

double parseDouble(const std::string& value, const std::string& option)
{
    try
    {
        return std::stod(value);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("Invalid value for " + option + ": " + value);
    }
}

ParsedArguments parseArguments(const int argc, const char* const argv[])
{
    ParsedArguments arguments;

    std::unordered_map<std::string, std::string> options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h")
        {
            printUsage();
            std::exit(0);
        }

        if (key.rfind("--", 0) != 0)
        {
            throw std::invalid_argument("Invalid argument: " + key);
        }
        if (i + 1 >= argc)
        {
            throw std::invalid_argument("Missing value for argument: " + key);
        }

        options[key] = argv[++i];
    }

    if (const auto it = options.find("--generator"); it != options.end())
    {
        arguments.generator = parseGeneratorChoice(it->second);
    }
    if (const auto it = options.find("--output"); it != options.end())
    {
        arguments.outputPath = it->second;
    }
    if (const auto it = options.find("--records"); it != options.end())
    {
        arguments.records = parseSize(it->second, "--records");
    }
    if (const auto it = options.find("--record-size"); it != options.end())
    {
        arguments.recordSize = parseSize(it->second, "--record-size");
    }
    if (const auto it = options.find("--seed"); it != options.end())
    {
        arguments.seed = parseUInt64(it->second, "--seed");
    }
    if (const auto it = options.find("--drift-interval"); it != options.end())
    {
        arguments.driftInterval = parseSize(it->second, "--drift-interval");
    }
    if (const auto it = options.find("--drift-fraction"); it != options.end())
    {
        arguments.driftFraction = parseDouble(it->second, "--drift-fraction");
    }

    if (const auto it = options.find("--det-cache-size"); it != options.end())
    {
        arguments.deterministicCacheSize = parseUInt64(it->second, "--det-cache-size");
    }
    if (const auto it = options.find("--det-hit-ratio"); it != options.end())
    {
        arguments.deterministicHitRatio = parseHitMissRatio(it->second);
    }
    if (const auto it = options.find("--det-cache-type"); it != options.end())
    {
        arguments.deterministicCacheType = parsePredictionCacheType(it->second);
    }

    if (const auto it = options.find("--zipf-num-keys"); it != options.end())
    {
        arguments.zipfNumKeys = parseSize(it->second, "--zipf-num-keys");
    }
    if (const auto it = options.find("--zipf-s"); it != options.end())
    {
        arguments.zipfS = parseDouble(it->second, "--zipf-s");
    }

    if (const auto it = options.find("--temporal-universe-size"); it != options.end())
    {
        arguments.temporalUniverseSize = parseSize(it->second, "--temporal-universe-size");
    }
    if (const auto it = options.find("--temporal-series-length"); it != options.end())
    {
        arguments.temporalSeriesLength = parseSize(it->second, "--temporal-series-length");
    }
    if (const auto it = options.find("--temporal-window-size"); it != options.end())
    {
        arguments.temporalWindowSize = parseSize(it->second, "--temporal-window-size");
    }
    if (const auto it = options.find("--temporal-overlap-ratio"); it != options.end())
    {
        arguments.temporalOverlapRatio = parseDouble(it->second, "--temporal-overlap-ratio");
    }
    if (const auto it = options.find("--burstiness-duty-cycle"); it != options.end())
    {
        arguments.burstinessDutyCycle = parseDouble(it->second, "--burstiness-duty-cycle");
    }
    if (const auto it = options.find("--burstiness-on-period"); it != options.end())
    {
        arguments.burstinessOnPeriod = parseSize(it->second, "--burstiness-on-period");
    }
    if (const auto it = options.find("--burstiness-num-keys"); it != options.end())
    {
        arguments.burstinessNumKeys = parseSize(it->second, "--burstiness-num-keys");
    }

    return arguments;
}

NES::Microbenchmark::BenchmarkData generateData(const ParsedArguments& args)
{
    switch (args.generator)
    {
        case GeneratorChoice::Deterministic:
            return NES::Microbenchmark::createDeterministicBenchmarkData(
                args.deterministicCacheSize,
                args.records,
                args.deterministicHitRatio,
                args.deterministicCacheType,
                args.recordSize,
                args.seed);
        case GeneratorChoice::Zipf:
            return NES::Microbenchmark::createZipfBenchmarkData(
                args.zipfNumKeys,
                args.records,
                args.recordSize,
                args.seed,
                args.zipfS,
                args.driftInterval,
                args.driftFraction);
        case GeneratorChoice::TemporalLocality:
            return NES::Microbenchmark::createTemporalLocalityBenchmarkData(
                args.temporalUniverseSize,
                args.temporalSeriesLength,
                args.temporalWindowSize,
                args.temporalOverlapRatio,
                args.records,
                args.recordSize,
                args.seed,
                args.driftInterval,
                args.driftFraction);
        case GeneratorChoice::Burstiness:
            return NES::Microbenchmark::createBurstinessBenchmarkData(
                args.burstinessDutyCycle,
                args.burstinessOnPeriod,
                args.burstinessNumKeys,
                args.records,
                args.recordSize,
                args.seed,
                args.driftInterval,
                args.driftFraction);
    }
    std::unreachable();
}

uint64_t extractRecordId(const std::byte* record, const size_t recordSize)
{
    uint64_t id = 0;
    const size_t idBytes = std::min(recordSize, sizeof(id));
    std::memcpy(&id, record, idBytes);
    return id;
}

size_t computeBurstinessOffPeriod(const double dutyCycle, const size_t onPeriod)
{
    if (dutyCycle >= 1.0)
    {
        return 0;
    }

    return std::max<size_t>(
        1,
        static_cast<size_t>(std::llround(static_cast<double>(onPeriod) * (1.0 - dutyCycle) / dutyCycle)));
}

// std::string_view classifyBurstPhase(
//     const size_t cyclePosition,
//     const size_t onPeriod,
//     const size_t offPeriod)
// {
//     (void)offPeriod;
//     return cyclePosition < onPeriod ? "on" : "off";
// }

void writeDataToCsv(
    const std::filesystem::path& outputPath,
    const NES::Microbenchmark::BenchmarkData& data,
    const size_t recordSize,
    const ParsedArguments& args)
{
    std::ofstream out(outputPath, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        throw std::runtime_error("Failed to open output file: " + outputPath.string());
    }

    if (args.generator != GeneratorChoice::Burstiness)
    {
        out << "position,key_id\n";
        for (size_t i = 0; i < data.size(); ++i)
        {
            const uint64_t keyId = extractRecordId(data[i].get(), recordSize);
            out << i << "," << keyId << "\n";
        }
        return;
    }

    // Burstiness-specific export with phase annotations.
    const size_t onPeriod = args.burstinessOnPeriod;
    const size_t offPeriod = computeBurstinessOffPeriod(args.burstinessDutyCycle, onPeriod);
    const size_t cycleLength = onPeriod + offPeriod;

    out << "position,key_id,phase_id,phase_type,position_in_phase,cycle_position,drift_epoch\n";

    for (size_t i = 0; i < data.size(); ++i)
    {
        const uint64_t keyId = extractRecordId(data[i].get(), recordSize);

        const size_t cyclePosition = cycleLength > 0 ? (i % cycleLength) : 0;
        const bool isOnPhase = cyclePosition < onPeriod;
        const size_t phaseId = cycleLength > 0 ? (i / cycleLength) : 0;
        const size_t positionInPhase = isOnPhase ? cyclePosition : (cyclePosition - onPeriod);
        const size_t driftEpoch =
            (args.driftInterval > 0) ? (i / args.driftInterval) : 0;

        out << i << ","
            << keyId << ","
            << phaseId << ","
            << (isOnPhase ? "on" : "off") << ","
            << positionInPhase << ","
            << cyclePosition << ","
            << driftEpoch << "\n";
    }
}

}

int main(const int argc, const char* const argv[])
{
    try
    {
        const auto args = parseArguments(argc, argv);
        const auto data = generateData(args);
        writeDataToCsv(args.outputPath, data, args.recordSize, args);
        std::cout << "Wrote " << data.size() << " rows to '" << args.outputPath.string() << "'." << std::endl;
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << std::endl;
        printUsage();
        return 1;
    }
}
