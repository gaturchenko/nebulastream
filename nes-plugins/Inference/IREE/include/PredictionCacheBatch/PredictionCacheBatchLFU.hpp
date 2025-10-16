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
#include <PredictionCacheBatch/PredictionCacheBatch.hpp>

namespace NES
{
struct PredictionCacheBatchEntryLFU : PredictionCacheBatchEntry
{
    /// Stores the frequency of each cache item
    uint64_t frequency;
};

class PredictionCacheBatchLFU : public PredictionCacheBatch
{
public:
    PredictionCacheBatchLFU(
        const uint64_t numberOfEntries,
        const uint64_t sizeOfEntry,
        const nautilus::val<int8_t*>& startOfEntries,
        const nautilus::val<uint64_t*>& hitsRef,
        const nautilus::val<uint64_t*>& missesRef,
        const nautilus::val<size_t>& inputSize);

    ~PredictionCacheBatchLFU() override = default;
    nautilus::val<int8_t*>
    getDataStructureRef(const nautilus::val<std::byte*>& record, const PredictionCacheBatch::PredictionCacheReplacement& replacementFunction) override;

private:
    nautilus::val<uint64_t*> getFrequency(const nautilus::val<uint64_t>& pos);
};

}
