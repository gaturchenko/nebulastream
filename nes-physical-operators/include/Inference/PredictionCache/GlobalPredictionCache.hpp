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

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <Inference/PredictionCache/PredictionCache.hpp>
#include <InferenceConfiguration.hpp>

namespace NES
{

/// A single prediction cache shared by all worker threads
class GlobalPredictionCache
{
public:
    GlobalPredictionCache(PredictionCacheType policy, uint64_t numberOfEntries, size_t recordSize, size_t predictionSize);

    /// On a hit, copies the cached prediction into predictionOut (predictionSize bytes),
    /// applies the policy's hit update, and returns true. Counts a hit or a miss.
    bool lookup(const std::byte* record, std::byte* predictionOut);

    /// Stores the prediction computed for record, evicting an entry according to the
    /// policy. If another thread inserted the same record in the meantime, only the
    /// prediction bytes are refreshed.
    void insert(const std::byte* record, const std::byte* prediction);

    [[nodiscard]] HitsAndMisses getHitsAndMisses() const;

private:
    static constexpr uint64_t NOT_FOUND = UINT64_MAX;

    struct Slot
    {
        std::vector<std::byte> record;
        std::vector<std::byte> prediction;
        bool occupied = false;
        bool secondChanceBit = false;
        uint64_t frequency = 0;
    };

    /// All private helpers require the mutex to be held.
    [[nodiscard]] uint64_t findSlot(const std::byte* record) const;
    [[nodiscard]] uint64_t selectVictim();
    void onHit(uint64_t pos);
    void removeFromLfuBucket(uint64_t pos);
    void addToLfuBucket(uint64_t pos, uint64_t frequency);
    void updateMinFrequencyAfterRemoving(uint64_t oldFrequency);
    void appendLruTail(uint64_t pos);
    [[nodiscard]] uint64_t hashRecordBytes(const std::byte* record) const;

    mutable std::mutex mutex;
    PredictionCacheType policy;
    size_t recordSize;
    size_t predictionSize;
    std::vector<Slot> slots;
    /// Hash of the record bytes -> candidate slot positions (verified via memcmp).
    std::unordered_map<uint64_t, std::vector<uint64_t>> lookupIndex;

    uint64_t fifoReplacementIndex = 0;
    uint64_t secondChanceReplacementIndex = 0;
    uint64_t nextEmptyPos = 0;
    uint64_t minFrequency = 1;
    std::vector<std::vector<uint64_t>> lfuBuckets;
    std::vector<uint64_t> lruOrder;

    uint64_t hits = 0;
    uint64_t misses = 0;
};

}
