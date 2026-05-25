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

#include <Inference/PredictionCache/PredictionCache.hpp>

namespace NES
{
struct PredictionCacheEntryFIFO : PredictionCacheEntry
{
    ~PredictionCacheEntryFIFO() override = default;
};

class PredictionCacheFIFO : public PredictionCache
{
public:
    PredictionCacheFIFO(
        uint64_t numberOfEntries,
        uint64_t sizeOfEntry,
        nautilus::val<int8_t*> startOfEntries,
        nautilus::val<uint64_t*> hitsRef,
        nautilus::val<uint64_t*> missesRef,
        nautilus::val<size_t> inputSize);
    ~PredictionCacheFIFO() override = default;

    nautilus::val<std::byte*>
    getDataStructureRef(const nautilus::val<std::byte*>& record, const PredictionCacheReplacement& replacementFunction) override;
    nautilus::val<uint64_t> updateKeys(const nautilus::val<std::byte*>& record, const PredictionCacheUpdate& updateFunction) override;
    void updateValues(const nautilus::val<uint64_t>& pos, const PredictionCacheUpdate& updateFunction) override;
    nautilus::val<uint64_t> getReplacementPos() override;
    void setReplacementPos(nautilus::val<uint64_t> pos) override;

protected:
    nautilus::val<uint64_t> localReplacementIndex;
};
}
