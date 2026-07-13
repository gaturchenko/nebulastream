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

#include <cstdint>
#include <Inference/PredictionCache/PredictionCache.hpp>

namespace NES
{
struct PredictionCacheEntryLRU : PredictionCacheEntry
{
    uint64_t previousPos = UINT64_MAX;
    uint64_t nextPos = UINT64_MAX;
    ~PredictionCacheEntryLRU() override = default;
};

class PredictionCacheLRU final : public PredictionCache
{
public:
    PredictionCacheLRU(
        uint64_t numberOfEntries,
        uint64_t sizeOfEntry,
        nautilus::val<int8_t*> startOfEntries,
        nautilus::val<uint64_t*> hitsRef,
        nautilus::val<uint64_t*> missesRef,
        nautilus::val<size_t> inputSize);
    ~PredictionCacheLRU() override = default;

    nautilus::val<std::byte*>
    getDataStructureRef(const nautilus::val<std::byte*>& record, const PredictionCacheReplacement& replacementFunction) override;
    nautilus::val<uint64_t> updateKeys(const nautilus::val<std::byte*>& record, const PredictionCacheUpdate& updateFunction) override;
    void updateValues(const nautilus::val<uint64_t>& pos, const PredictionCacheUpdate& updateFunction) override;
    nautilus::val<uint64_t> getReplacementPos() override;

    void setReplacementPos(nautilus::val<uint64_t>) override;

private:
    nautilus::val<uint64_t*> getPreviousPos(const nautilus::val<uint64_t>& pos);
    nautilus::val<uint64_t*> getNextPos(const nautilus::val<uint64_t>& pos);
    void appendToTail(const nautilus::val<uint64_t>& pos);
    void removeFromList(const nautilus::val<uint64_t>& pos);
    void touch(const nautilus::val<uint64_t>& pos);
    nautilus::val<uint64_t> claimReplacementPos();

    nautilus::val<uint64_t> nextEmptyPos;
    nautilus::val<uint64_t> lruHead;
    nautilus::val<uint64_t> lruTail;
};
}
