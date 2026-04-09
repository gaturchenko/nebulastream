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

#include <ostream>
#include <DataTypes/DataTypeProvider.hpp>
#include <ErrorHandling.hpp>
#include <Model.hpp>
#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedHashMap.hpp>
#include <IREERuntimeWrapper.hpp>

namespace NES
{

struct BatchCachingHelper
{
    void updateCacheMapIndices(uint64_t keyIdx, int rowIdx)
    {
        auto it = std::find_if(cacheMap.begin(), cacheMap.end(),
            [&](const auto& p) { return p.first == keyIdx; });

        if (it != cacheMap.end())
        {
            it->second = rowIdx;
        }
        else
        {
            cacheMap.emplace_back(keyIdx, rowIdx);
        }
    }

    uint64_t getCacheMapKey(size_t idx)
    {
        return cacheMap.at(idx).first;
    }

    uint64_t getCacheMapValue(size_t idx)
    {
        return cacheMap.at(idx).second;
    }

    void clearCacheMap()
    {
        cacheMap.clear();
    }

    size_t getCacheMapSize()
    {
        return cacheMap.size();
    }

    void appendMissIdx(int idx)
    {
        missIndices.insert(idx);
    }

    size_t getMissIndicesSize()
    {
        return missIndices.size();
    }

    void clearMissIndices()
    {
        missIndices.clear();
    }

    std::set<int> getMissIndices()
    {
        return missIndices;
    }

private:
    std::vector<std::pair<uint64_t, int>> cacheMap;
    std::set<int> missIndices;
};

class IREEAdapter
{
public:
    enum InputBufferSizeReduction : uint8_t
    {
        NONE = 1,
        LOW = 2,
        MEDIUM = 4,
        HIGH = 8
    };

    static std::shared_ptr<IREEAdapter> create();

    IREEAdapter() = default;

    void initializeModel(Nebuli::Inference::Model& model, uint64_t batchSize);

    template <class T>
    void addModelInput(size_t index, T value);
    void addModelInput(std::span<std::byte> content);
    void addModelInputBatch(int index, std::span<std::byte> content, size_t tupleSize);

    template <class T>
    uint64_t addModelInputPartial(T value);
    void addModelInputBatchPartial(int index, std::span<std::byte> content, size_t tupleSize);

    template <class T>
    T getResultAt(size_t idx);
    void copyResultTo(std::span<std::byte> content);
    void copyResultToBatch(size_t index, std::span<std::byte> content);

    template <class T>
    void infer();

    template <class T>
    void inferWithReduction();

    template <class T>
    size_t inferCombine(size_t outputSize, size_t outputFields, bool isVarSizedOutput);

    void allocateBuffers(size_t tupleSize);

    /// input for IREE runtime
    std::unique_ptr<std::byte[]> inputData{};
    std::unique_ptr<std::byte[]> outputData{};

    /// helper objects for the BatchCache operator
    BatchCachingHelper batchCachingHelper;
    std::unique_ptr<std::byte[]> cacheProbeTuple{};
    std::unique_ptr<std::byte[]> inputDataHalf{};
    std::unique_ptr<std::byte[]> inputDataFourth{};
    std::unique_ptr<std::byte[]> inputDataEighth{};
    InputBufferSizeReduction currentReductionLevel = NONE;
    uint64_t bytesProcessed = 0;

    size_t inputSize;
    size_t outputSize;

    /// caching statistics
    uint64_t misses;
    uint64_t noReductions;
    uint64_t lowReductions;
    uint64_t mediumReductions;
    uint64_t highReductions;
    uint64_t fullReductions;

private:
    std::string functionName;
    IREERuntimeWrapper runtimeWrapper;

    std::unordered_map<NES::DataType, iree_hal_element_types_t> dtypeMap = {
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::UINT8), IREE_HAL_ELEMENT_TYPE_UINT_8},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::UINT16), IREE_HAL_ELEMENT_TYPE_UINT_16},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::UINT32), IREE_HAL_ELEMENT_TYPE_UINT_32},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::UINT64), IREE_HAL_ELEMENT_TYPE_UINT_64},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::INT8), IREE_HAL_ELEMENT_TYPE_INT_8},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::INT16), IREE_HAL_ELEMENT_TYPE_INT_16},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::INT32), IREE_HAL_ELEMENT_TYPE_INT_32},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::INT64), IREE_HAL_ELEMENT_TYPE_INT_64},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::FLOAT32), IREE_HAL_ELEMENT_TYPE_FLOAT_32},
        {NES::DataTypeProvider::provideDataType(NES::DataType::Type::FLOAT64), IREE_HAL_ELEMENT_TYPE_FLOAT_64}};
};

}
