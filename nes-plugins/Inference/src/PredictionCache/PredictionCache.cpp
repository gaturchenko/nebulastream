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

#include <Nautilus/DataTypes/DataTypesUtil.hpp>
#include <PredictionCache/PredictionCache.hpp>
#include <InferenceLocalState.hpp>

namespace
{
constexpr uint64_t LOOKUP_INDEX_REBUILD_FACTOR = 2;

uint64_t hashRecord(const std::byte* record, const size_t size)
{
    static constexpr uint64_t m = UINT64_C(0xc6a4a7935bd1e995);
    static constexpr uint64_t seed = UINT64_C(0xe17a1465);
    static constexpr unsigned int r = 47;

    const auto* const data64 = reinterpret_cast<const uint64_t*>(record);
    uint64_t h = seed ^ (size * m);

    const size_t nBlocks = size / 8;
    for (size_t i = 0; i < nBlocks; ++i)
    {
        auto k = *(data64 + i);

        k *= m;
        k ^= k >> r;
        k *= m;

        h ^= k;
        h *= m;
    }

    const auto* const data8 = reinterpret_cast<const uint8_t*>(data64 + nBlocks);
    switch (size & 7U)
    {
        case 7:
            h ^= static_cast<uint64_t>(data8[6]) << 48U;
            /// FALLTHROUGH
        case 6:
            h ^= static_cast<uint64_t>(data8[5]) << 40U;
            /// FALLTHROUGH
        case 5:
            h ^= static_cast<uint64_t>(data8[4]) << 32U;
            /// FALLTHROUGH
        case 4:
            h ^= static_cast<uint64_t>(data8[3]) << 24U;
            /// FALLTHROUGH
        case 3:
            h ^= static_cast<uint64_t>(data8[2]) << 16U;
            /// FALLTHROUGH
        case 2:
            h ^= static_cast<uint64_t>(data8[1]) << 8U;
            /// FALLTHROUGH
        case 1:
            h ^= static_cast<uint64_t>(data8[0]);
            h *= m;
            /// FALLTHROUGH
        default:
            break;
    }

    h ^= h >> r;

    /// final step
    h *= m;
    h ^= h >> r;
    return h;
}

std::byte* getEntryKeyStart(NES::ChainedHashMapEntry* entry)
{
    return reinterpret_cast<std::byte*>(entry) + sizeof(NES::ChainedHashMapEntry);
}

}

namespace NES
{
PredictionCache::PredictionCache(
    const nautilus::val<OperatorHandler*>& operatorHandler,
    const uint64_t numberOfEntries,
    const uint64_t sizeOfEntry,
    const nautilus::val<int8_t*>& startOfEntries,
    const nautilus::val<uint64_t*>& hitsRef,
    const nautilus::val<uint64_t*>& missesRef,
    const nautilus::val<size_t>& inputSize)
    : InferenceLocalState(operatorHandler)
    , startOfEntries(startOfEntries)
    , numberOfEntries(numberOfEntries)
    , sizeOfEntry(sizeOfEntry)
    , numberOfHits(hitsRef)
    , numberOfMisses(missesRef)
    , inputSize(inputSize)
    , lookupIndex(nullptr)
    , lookupIndexBufferProvider(nullptr)
{
}

void PredictionCache::incrementNumberOfHits()
{
    auto currentNumberOfHits = static_cast<nautilus::val<uint64_t>>(*numberOfHits);
    currentNumberOfHits = currentNumberOfHits + 1;
    *numberOfHits = currentNumberOfHits;
}

void PredictionCache::incrementNumberOfMisses()
{
    auto currentNumberOfMisses = static_cast<nautilus::val<uint64_t>>(*numberOfMisses);
    currentNumberOfMisses = currentNumberOfMisses + 1;
    *numberOfMisses = currentNumberOfMisses;
}

nautilus::val<std::byte*> PredictionCache::getRecord(const nautilus::val<uint64_t>& pos)
{
    const auto predictionCacheEntry = startOfEntries + pos * sizeOfEntry;
    const auto recordRef = getMemberRef(predictionCacheEntry, &PredictionCacheEntry::record);
    auto record = *getMemberWithOffset<std::byte*>(recordRef, 0);
    return record;
}

nautilus::val<std::byte*> PredictionCache::getDataStructure(const nautilus::val<uint64_t>& pos)
{
    const auto predictionCacheEntry = startOfEntries + pos * sizeOfEntry;
    const auto dataStructureRef = getMemberRef(predictionCacheEntry, &PredictionCacheEntry::dataStructure);
    auto dataStructure = *getMemberWithOffset<std::byte*>(dataStructureRef, 0);
    return dataStructure;
}

nautilus::val<bool> PredictionCache::foundRecord(const nautilus::val<uint64_t>& pos, const nautilus::val<std::byte*>& candidateRecord)
{
    const auto cacheRecord = getRecord(pos);
    return nautilus::invoke(+[](std::byte* candidate, std::byte* cache, size_t size)
    {
        if (cache != nullptr)
        {
            return std::memcmp(candidate, cache, size) == 0;
        }
        return false;
    }, candidateRecord, cacheRecord, inputSize);
}

void PredictionCache::configureLookupIndex(
    const nautilus::val<ChainedHashMap*>& lookupIndex,
    const nautilus::val<AbstractBufferProvider*>& bufferProvider)
{
    this->lookupIndex = lookupIndex;
    this->lookupIndexBufferProvider = bufferProvider;
    if (!this->lookupIndex)
    {
        return;
    }

    const auto numberOfIndexedEntries = nautilus::invoke(+[](const ChainedHashMap* lookupMap)
    {
        if (lookupMap == nullptr)
        {
            return uint64_t{0};
        }
        return lookupMap->getNumberOfTuples();
    }, this->lookupIndex);

    if (numberOfIndexedEntries == 0)
    {
        rebuildLookupIndex();
    }
}

nautilus::val<uint64_t> PredictionCache::searchInCache(const nautilus::val<std::byte*>& record)
{
    if (lookupIndex)
    {
        return nautilus::invoke(
            +[](const ChainedHashMap* lookupMap,
                const std::byte* recordPtr,
                const size_t recordSize,
                const int8_t* startOfEntriesPtr,
                const uint64_t sizeOfPredictionCacheEntry,
                const uint64_t numberOfPredictionCacheEntries,
                const uint64_t notFoundValue)
            {
                if (lookupMap == nullptr || recordPtr == nullptr || lookupMap->getNumberOfTuples() == 0)
                {
                    return notFoundValue;
                }

                const auto hashValue = hashRecord(recordPtr, recordSize);
                auto* entry = lookupMap->findChain(hashValue);
                while (entry != nullptr)
                {
                    if (entry->hash == hashValue)
                    {
                        const auto* keyStart = getEntryKeyStart(entry);
                        if (std::memcmp(keyStart, recordPtr, recordSize) == 0)
                        {
                            uint64_t pos = notFoundValue;
                            uint64_t expectedRecordPtrRaw = 0;
                            std::memcpy(&pos, keyStart + recordSize, sizeof(uint64_t));
                            std::memcpy(&expectedRecordPtrRaw, keyStart + recordSize + sizeof(uint64_t), sizeof(uint64_t));

                            if (pos < numberOfPredictionCacheEntries)
                            {
                                const auto* predictionCacheEntry = reinterpret_cast<const PredictionCacheEntry*>(
                                    startOfEntriesPtr + pos * sizeOfPredictionCacheEntry);
                                const auto currentRecordPtrRaw
                                    = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(predictionCacheEntry->record));
                                if (currentRecordPtrRaw == expectedRecordPtrRaw)
                                {
                                    return pos;
                                }
                            }
                        }
                    }
                    entry = entry->next;
                }

                return notFoundValue;
            },
            lookupIndex,
            record,
            inputSize,
            startOfEntries,
            sizeOfEntry,
            numberOfEntries,
            nautilus::val<uint64_t>(NOT_FOUND));
    }
    return nautilus::val<uint64_t>(NOT_FOUND);
}

void PredictionCache::addLookupIndexEntry(const nautilus::val<std::byte*>& record, const nautilus::val<uint64_t>& pos)
{
    if (!lookupIndex || !lookupIndexBufferProvider)
    {
        return;
    }
    const auto cachedRecord = getRecord(pos);
    const auto cachedRecordPtrRaw = nautilus::invoke(+[](const std::byte* ptr)
    {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
    }, cachedRecord);

    nautilus::invoke(
        +[](ChainedHashMap* lookupMap,
            AbstractBufferProvider* bufferProvider,
            std::byte* recordPtr,
            const size_t recordSize,
            const uint64_t posValue,
            const uint64_t cachedRecordPtrRaw)
        {
            if (lookupMap == nullptr || bufferProvider == nullptr || recordPtr == nullptr)
            {
                return;
            }

            const auto hashValue = hashRecord(recordPtr, recordSize);
            auto* entry = static_cast<ChainedHashMapEntry*>(lookupMap->insertEntry(hashValue, bufferProvider));
            auto* keyStart = getEntryKeyStart(entry);

            /// layout: key bytes | pos | record_ptr_token
            std::memcpy(keyStart, recordPtr, recordSize);
            std::memcpy(keyStart + recordSize, &posValue, sizeof(uint64_t));
            std::memcpy(keyStart + recordSize + sizeof(uint64_t), &cachedRecordPtrRaw, sizeof(uint64_t));
        },
        lookupIndex,
        lookupIndexBufferProvider,
        record,
        inputSize,
        pos,
        cachedRecordPtrRaw);

    const auto numberOfIndexedEntries = nautilus::invoke(+[](const ChainedHashMap* lookupMap)
    {
        if (lookupMap == nullptr)
        {
            return uint64_t{0};
        }
        return lookupMap->getNumberOfTuples();
    }, lookupIndex);

    if (numberOfIndexedEntries > numberOfEntries * LOOKUP_INDEX_REBUILD_FACTOR)
    {
        rebuildLookupIndex();
    }
}

void PredictionCache::rebuildLookupIndex()
{
    if (!lookupIndex || !lookupIndexBufferProvider)
    {
        return;
    }

    nautilus::invoke(+[](ChainedHashMap* lookupMap)
    {
        if (lookupMap != nullptr)
        {
            lookupMap->clear();
        }
    }, lookupIndex);

    for (nautilus::val<uint64_t> i = 0; i < numberOfEntries; ++i)
    {
        const auto record = getRecord(i);
        if (record == nullptr)
        {
            continue;
        }

        nautilus::invoke(
            +[](ChainedHashMap* lookupMap,
                AbstractBufferProvider* bufferProvider,
                std::byte* recordPtr,
                const size_t recordSize,
                const uint64_t posValue,
                const uint64_t cachedRecordPtrRaw)
            {
                if (lookupMap == nullptr || bufferProvider == nullptr || recordPtr == nullptr)
                {
                    return;
                }

                const auto hashValue = hashRecord(recordPtr, recordSize);
                auto* entry = static_cast<ChainedHashMapEntry*>(lookupMap->insertEntry(hashValue, bufferProvider));
                auto* keyStart = getEntryKeyStart(entry);
                std::memcpy(keyStart, recordPtr, recordSize);
                std::memcpy(keyStart + recordSize, &posValue, sizeof(uint64_t));
                std::memcpy(keyStart + recordSize + sizeof(uint64_t), &cachedRecordPtrRaw, sizeof(uint64_t));
            },
            lookupIndex,
            lookupIndexBufferProvider,
            record,
            inputSize,
            i,
            nautilus::invoke(+[](const std::byte* ptr)
            {
                return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr));
            }, record));
    }
}

nautilus::val<uint64_t*> PredictionCache::getHitsRef(){ return this->numberOfHits; }
nautilus::val<uint64_t*> PredictionCache::getMissesRef(){ return this->numberOfMisses; }
nautilus::val<uint64_t> PredictionCache::getReplacementIndex(){ return this->replacementIndex; }
}
