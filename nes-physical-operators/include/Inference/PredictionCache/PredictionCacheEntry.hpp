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
#include <cstring>
#include <utility>

namespace NES
{

/// Represents one prediction cache entry. The policy-specific entry structs
/// extend this base with their replacement metadata.
struct PredictionCacheEntry
{
    std::byte* record = nullptr;
    std::byte* dataStructure = nullptr;
    std::size_t recordSize = 0;
    std::size_t dataSize = 0;

    PredictionCacheEntry() = default;

    virtual ~PredictionCacheEntry()
    {
        delete[] record;
        delete[] dataStructure;
    }

    PredictionCacheEntry(const PredictionCacheEntry& other) : recordSize(other.recordSize), dataSize(other.dataSize)
    {
        if (other.record != nullptr)
        {
            record = new std::byte[recordSize];
            std::memcpy(record, other.record, recordSize);
        }

        if (other.dataStructure != nullptr)
        {
            dataStructure = new std::byte[dataSize];
            std::memcpy(dataStructure, other.dataStructure, dataSize);
        }
    }

    PredictionCacheEntry& operator=(const PredictionCacheEntry& other)
    {
        if (this == &other)
        {
            return *this;
        }

        delete[] record;
        delete[] dataStructure;
        record = nullptr;
        dataStructure = nullptr;
        recordSize = other.recordSize;
        dataSize = other.dataSize;

        if (other.record != nullptr)
        {
            record = new std::byte[recordSize];
            std::memcpy(record, other.record, recordSize);
        }

        if (other.dataStructure != nullptr)
        {
            dataStructure = new std::byte[dataSize];
            std::memcpy(dataStructure, other.dataStructure, dataSize);
        }
        return *this;
    }

    PredictionCacheEntry(PredictionCacheEntry&& other) noexcept
        : record(other.record), dataStructure(other.dataStructure), recordSize(other.recordSize), dataSize(other.dataSize)
    {
        other.record = nullptr;
        other.dataStructure = nullptr;
        other.recordSize = 0;
        other.dataSize = 0;
    }

    PredictionCacheEntry& operator=(PredictionCacheEntry&& other) noexcept
    {
        if (this == &other)
        {
            return *this;
        }

        delete[] record;
        delete[] dataStructure;
        record = other.record;
        dataStructure = other.dataStructure;
        recordSize = other.recordSize;
        dataSize = other.dataSize;

        other.record = nullptr;
        other.dataStructure = nullptr;
        other.recordSize = 0;
        other.dataSize = 0;
        return *this;
    }
};

}
