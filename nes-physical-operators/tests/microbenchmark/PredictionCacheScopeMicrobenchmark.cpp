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

/// Measures how the cost of a prediction-cache operation scales with the record (key)
/// size under concurrency. Both scopes run the same GlobalPredictionCache code:
///   - shared:  one instance used by all threads (contended mutex)
///   - private: one instance per thread (uncontended mutex)
/// so the shared/private delta at a given key size and thread count is the pure
/// serialization cost of the global cache. Hashing, memcmp, and the record memcpy all
/// run inside the critical section and are O(key size), which is what this benchmark
/// demonstrates. No inference is simulated; a miss pays lookup + insert only.
///
/// Each thread executes a grouped workload (a miss immediately followed by its hits)
/// over a disjoint key range, so hit counts are deterministic per thread. Miss keys
/// cycle through a per-thread pool of at least 2x capacity pre-generated records, which
/// guarantees a reused key has been evicted before it recurs.

#include <Inference/PredictionCache/GlobalPredictionCache.hpp>
#include <Inference/PredictionCache/PredictionCache.hpp>

#include <algorithm>
#include <barrier>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <Util/Logger/LogLevel.hpp>
#include <Util/Logger/Logger.hpp>
#include <magic_enum/magic_enum.hpp>
#include <InferenceConfiguration.hpp>

namespace NES
{

namespace
{

struct BenchmarkConfig
{
    std::filesystem::path outputFile = std::filesystem::path("PredictionCacheScopeMicrobenchmark.csv");
    std::vector<PredictionCacheType> policies = {PredictionCacheType::LRU};
    std::vector<uint64_t> keyBytes = {8, 300, 4096, 65536, 1048576};
    std::vector<uint64_t> threadCounts = {1, 2, 4, 8, 16, 24};
    std::vector<uint64_t> hitPercents = {0, 50, 100};
    uint64_t capacity = 64;
    uint64_t predictionBytes = 64;
    /// Per-thread budget of key bytes pushed through the cache per run; bounds the
    /// runtime of large-key configurations.
    uint64_t bytesPerThread = 1ULL << 30;
    uint64_t minOpsPerThread = 4096;
    uint64_t maxOpsPerThread = 1'000'000;
    uint64_t repetitions = 3;
    uint64_t warmups = 1;
};

uint64_t splitmix64(uint64_t& state)
{
    state += UINT64_C(0x9e3779b97f4a7c15);
    uint64_t z = state;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

uint64_t desiredHits(const uint64_t ops, const uint64_t hitPercent)
{
    if (hitPercent == 100)
    {
        return ops - 1;
    }
    return (ops / 100) * hitPercent + (ops % 100) * hitPercent / 100;
}

/// Pool indices in grouped order: each miss advances to the next pool key (cyclic),
/// its hits repeat the same index. Same construction as makeGroupedOperations in
/// PredictionCacheMicrobenchmark.cpp, with keys reused from a bounded pool.
std::vector<uint32_t> makeGroupedIndices(const uint64_t ops, const uint64_t hitPercent, const uint64_t poolKeys)
{
    const auto expectedHits = desiredHits(ops, hitPercent);
    uint64_t remainingHits = expectedHits;
    uint64_t remainingMisses = ops - expectedHits;
    std::vector<uint32_t> indices;
    indices.reserve(ops);

    uint64_t nextPoolIndex = 0;
    while (remainingMisses > 0)
    {
        const auto current = static_cast<uint32_t>(nextPoolIndex % poolKeys);
        ++nextPoolIndex;
        indices.push_back(current);
        --remainingMisses;

        const auto hitsForThisMiss = remainingMisses == 0 ? remainingHits : remainingHits / (remainingMisses + 1);
        indices.insert(indices.end(), hitsForThisMiss, current);
        remainingHits -= hitsForThisMiss;
    }

    if (indices.size() != ops)
    {
        throw std::runtime_error("scope benchmark generator produced an unexpected number of operations");
    }
    return indices;
}

/// One contiguous buffer of poolKeys records. The first 8 bytes carry a globally unique
/// key id; the remainder is a deterministic byte stream derived from the id, so equal
/// ids yield identical records and the full record participates in hash and memcmp.
std::vector<std::byte> makeKeyPool(const uint64_t threadIndex, const uint64_t poolKeys, const uint64_t keyBytes)
{
    std::vector<std::byte> pool(poolKeys * keyBytes);
    for (uint64_t key = 0; key < poolKeys; ++key)
    {
        auto* record = pool.data() + key * keyBytes;
        const uint64_t id = ((threadIndex + 1) << 40) + key;
        std::memcpy(record, &id, std::min<uint64_t>(sizeof(id), keyBytes));

        uint64_t state = id;
        uint64_t offset = sizeof(id);
        while (offset < keyBytes)
        {
            const auto fill = splitmix64(state);
            std::memcpy(record + offset, &fill, std::min<uint64_t>(sizeof(fill), keyBytes - offset));
            offset += sizeof(fill);
        }
    }
    return pool;
}

struct RunResult
{
    std::vector<double> threadDurationsUs;
    uint64_t hits = 0;
    uint64_t misses = 0;
};

RunResult runOnce(
    const PredictionCacheType policy,
    const bool shared,
    const uint64_t capacity,
    const uint64_t keyBytes,
    const uint64_t predictionBytes,
    const std::vector<std::vector<std::byte>>& pools,
    const std::vector<uint32_t>& indices)
{
    const auto threads = pools.size();
    std::vector<std::unique_ptr<GlobalPredictionCache>> caches;
    if (shared)
    {
        caches.push_back(std::make_unique<GlobalPredictionCache>(policy, capacity, keyBytes, predictionBytes));
    }
    else
    {
        for (uint64_t i = 0; i < threads; ++i)
        {
            caches.push_back(std::make_unique<GlobalPredictionCache>(policy, capacity, keyBytes, predictionBytes));
        }
    }

    RunResult result;
    result.threadDurationsUs.resize(threads);
    std::barrier startBarrier(static_cast<std::ptrdiff_t>(threads));
    std::vector<std::jthread> workers;
    workers.reserve(threads);

    for (uint64_t threadIndex = 0; threadIndex < threads; ++threadIndex)
    {
        workers.emplace_back(
            [&, threadIndex]
            {
                auto& cache = shared ? *caches.front() : *caches[threadIndex];
                const auto* pool = pools[threadIndex].data();
                std::vector<std::byte> prediction(predictionBytes);

                startBarrier.arrive_and_wait();
                const auto start = std::chrono::steady_clock::now();
                for (const auto poolIndex : indices)
                {
                    const auto* record = pool + static_cast<uint64_t>(poolIndex) * keyBytes;
                    if (!cache.lookup(record, prediction.data()))
                    {
                        cache.insert(record, prediction.data());
                    }
                }
                const auto stop = std::chrono::steady_clock::now();
                result.threadDurationsUs[threadIndex] = std::chrono::duration<double, std::micro>(stop - start).count();
            });
    }
    workers.clear();

    for (const auto& cache : caches)
    {
        const auto [hits, misses] = cache->getHitsAndMisses();
        result.hits += hits;
        result.misses += misses;
    }
    return result;
}

std::vector<uint64_t> parseUIntList(std::string_view raw)
{
    std::vector<uint64_t> values;
    std::stringstream stream{std::string(raw)};
    std::string item;
    while (std::getline(stream, item, ','))
    {
        if (!item.empty())
        {
            values.push_back(std::stoull(item));
        }
    }
    return values;
}

std::vector<PredictionCacheType> parsePolicies(std::string_view raw)
{
    std::vector<PredictionCacheType> policies;
    std::stringstream stream{std::string(raw)};
    std::string item;
    while (std::getline(stream, item, ','))
    {
        if (item.empty())
        {
            continue;
        }
        const auto policy = magic_enum::enum_cast<PredictionCacheType>(item);
        if (!policy.has_value() || *policy == PredictionCacheType::NONE)
        {
            throw std::invalid_argument("Unknown prediction cache policy: " + item);
        }
        policies.push_back(*policy);
    }
    return policies;
}

void printUsage(const char* binary)
{
    std::cout << "Usage: " << binary
              << " [--output FILE] [--policies LRU,FIFO,...] [--key-bytes 8,300,...] [--threads 1,2,4,...]\n"
                 "       [--hit-percents 0,50,100] [--capacity N] [--prediction-bytes N] [--bytes-per-thread N]\n"
                 "       [--min-ops N] [--max-ops N] [--repetitions N] [--warmups N]\n";
}

BenchmarkConfig parseConfig(const int argc, char** argv)
{
    BenchmarkConfig config;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view arg(argv[i]);
        const auto nextValue = [&](const std::string_view option) -> std::string_view
        {
            if (i + 1 >= argc)
            {
                throw std::invalid_argument("Missing value for " + std::string(option));
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }
        if (arg == "--output")
        {
            config.outputFile = std::filesystem::path(std::string(nextValue(arg)));
        }
        else if (arg == "--policies")
        {
            config.policies = parsePolicies(nextValue(arg));
        }
        else if (arg == "--key-bytes")
        {
            config.keyBytes = parseUIntList(nextValue(arg));
        }
        else if (arg == "--threads")
        {
            config.threadCounts = parseUIntList(nextValue(arg));
        }
        else if (arg == "--hit-percents")
        {
            config.hitPercents = parseUIntList(nextValue(arg));
        }
        else if (arg == "--capacity")
        {
            config.capacity = std::stoull(std::string(nextValue(arg)));
        }
        else if (arg == "--prediction-bytes")
        {
            config.predictionBytes = std::stoull(std::string(nextValue(arg)));
        }
        else if (arg == "--bytes-per-thread")
        {
            config.bytesPerThread = std::stoull(std::string(nextValue(arg)));
        }
        else if (arg == "--min-ops")
        {
            config.minOpsPerThread = std::stoull(std::string(nextValue(arg)));
        }
        else if (arg == "--max-ops")
        {
            config.maxOpsPerThread = std::stoull(std::string(nextValue(arg)));
        }
        else if (arg == "--repetitions")
        {
            config.repetitions = std::stoull(std::string(nextValue(arg)));
        }
        else if (arg == "--warmups")
        {
            config.warmups = std::stoull(std::string(nextValue(arg)));
        }
        else
        {
            throw std::invalid_argument("Unknown argument: " + std::string(arg));
        }
    }

    if (config.policies.empty() || config.keyBytes.empty() || config.threadCounts.empty() || config.hitPercents.empty())
    {
        throw std::invalid_argument("policies, key-bytes, threads, and hit-percents must be non-empty");
    }
    if (config.capacity == 0 || config.repetitions == 0 || config.minOpsPerThread < 2)
    {
        throw std::invalid_argument("capacity and repetitions must be positive; min-ops must be at least 2");
    }
    for (const auto keySize : config.keyBytes)
    {
        if (keySize < sizeof(uint64_t))
        {
            throw std::invalid_argument("key sizes below 8 bytes cannot carry the unique key id");
        }
    }
    for (const auto percent : config.hitPercents)
    {
        if (percent > 100)
        {
            throw std::invalid_argument("hit percents must be within [0, 100]");
        }
    }
    /// A key inserted by one thread must survive until its immediately following hits
    /// even while every other thread inserts concurrently.
    const auto maxThreads = *std::ranges::max_element(config.threadCounts);
    if (config.capacity < 2 * maxThreads)
    {
        throw std::invalid_argument("capacity must be at least twice the largest thread count for deterministic hit rates");
    }
    return config;
}

int runScopeMicrobenchmark(const BenchmarkConfig& config)
{
    std::ofstream csv(config.outputFile);
    if (!csv.is_open())
    {
        throw std::runtime_error("Failed to open benchmark output CSV: " + config.outputFile.string());
    }
    csv << "policy,scope,key_bytes,prediction_bytes,capacity,pool_keys,threads,hit_percent,repetition,ops_per_thread,total_ops,"
           "makespan_us,throughput_ops_per_second,ns_per_op_mean,ns_per_op_min_thread,ns_per_op_max_thread,"
           "hits,misses,observed_hit_rate,expected_hit_rate,status\n";

    /// Twice the capacity guarantees a reused pool key has been evicted before it
    /// recurs, so every reuse is a miss under every policy.
    const auto poolKeys = 2 * config.capacity;

    for (const auto policy : config.policies)
    {
        for (const auto keySize : config.keyBytes)
        {
            const auto opsPerThread
                = std::clamp(config.bytesPerThread / keySize, config.minOpsPerThread, config.maxOpsPerThread);

            for (const auto threads : config.threadCounts)
            {
                std::vector<std::vector<std::byte>> pools;
                pools.reserve(threads);
                for (uint64_t threadIndex = 0; threadIndex < threads; ++threadIndex)
                {
                    pools.push_back(makeKeyPool(threadIndex, poolKeys, keySize));
                }

                for (const auto hitPercent : config.hitPercents)
                {
                    const auto indices = makeGroupedIndices(opsPerThread, hitPercent, poolKeys);
                    const auto expectedHitsPerThread = desiredHits(opsPerThread, hitPercent);
                    const auto expectedHitRate = static_cast<double>(expectedHitsPerThread) / static_cast<double>(opsPerThread);

                    for (const std::string_view scope : {"private", "shared"})
                    {
                        const bool shared = scope == "shared";
                        std::cerr << "policy=" << magic_enum::enum_name(policy) << " key_bytes=" << keySize << " threads=" << threads
                                  << " hit_percent=" << hitPercent << " scope=" << scope << " ops_per_thread=" << opsPerThread << '\n';

                        for (uint64_t iteration = 0; iteration < config.warmups + config.repetitions; ++iteration)
                        {
                            const auto runResult
                                = runOnce(policy, shared, config.capacity, keySize, config.predictionBytes, pools, indices);
                            if (iteration < config.warmups)
                            {
                                continue;
                            }

                            const auto totalOps = opsPerThread * threads;
                            const auto observedHitRate = static_cast<double>(runResult.hits) / static_cast<double>(totalOps);
                            /// Private caches are exact by construction. The shared cache is exact
                            /// unless a thread gets preempted long enough for the other threads to
                            /// push its key through the whole cache between the miss and its hits.
                            const auto expectedTotalHits = expectedHitsPerThread * threads;
                            const bool exact = runResult.hits == expectedTotalHits;
                            const bool withinTolerance = std::abs(observedHitRate - expectedHitRate) <= 0.01;
                            if (!shared && !exact)
                            {
                                throw std::runtime_error(
                                    "private scope expected " + std::to_string(expectedTotalHits) + " hits, got "
                                    + std::to_string(runResult.hits));
                            }
                            if (!withinTolerance)
                            {
                                std::cerr << "WARNING: observed hit rate " << observedHitRate << " deviates from expected "
                                          << expectedHitRate << '\n';
                            }

                            const auto makespanUs = *std::ranges::max_element(runResult.threadDurationsUs);
                            double nsPerOpSum = 0;
                            double nsPerOpMin = std::numeric_limits<double>::max();
                            double nsPerOpMax = 0;
                            for (const auto durationUs : runResult.threadDurationsUs)
                            {
                                const auto nsPerOp = durationUs * 1000.0 / static_cast<double>(opsPerThread);
                                nsPerOpSum += nsPerOp;
                                nsPerOpMin = std::min(nsPerOpMin, nsPerOp);
                                nsPerOpMax = std::max(nsPerOpMax, nsPerOp);
                            }

                            csv << magic_enum::enum_name(policy) << ',' << scope << ',' << keySize << ',' << config.predictionBytes << ','
                                << config.capacity << ',' << poolKeys << ',' << threads << ',' << hitPercent << ','
                                << (iteration - config.warmups) << ',' << opsPerThread << ',' << totalOps << ',' << makespanUs << ','
                                << static_cast<double>(totalOps) * 1'000'000.0 / makespanUs << ','
                                << nsPerOpSum / static_cast<double>(threads) << ',' << nsPerOpMin << ',' << nsPerOpMax << ','
                                << runResult.hits << ',' << runResult.misses << ',' << observedHitRate << ',' << expectedHitRate << ','
                                << (exact ? "ok" : (withinTolerance ? "inexact" : "deviated")) << '\n';
                        }
                    }
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
        NES::Logger::setupLogging("PredictionCacheScopeMicrobenchmark.log", NES::LogLevel::LOG_ERROR);
        return NES::runScopeMicrobenchmark(NES::parseConfig(argc, argv));
    }
    catch (const std::exception& ex)
    {
        std::cerr << "PredictionCacheScopeMicrobenchmark failed: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }
}
