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
#include "IREERuntimeWrapper.hpp"

namespace NES
{

class IREEAdapter
{
public:
    static std::shared_ptr<IREEAdapter> create();

    IREEAdapter() = default;

    void initializeModel(Nebuli::Inference::Model& model, uint64_t batch_size);

    template <class T>
    void addModelInput(size_t index, T value)
    {
        PRECONDITION(index < inputSize / sizeof(T), "Index is too large");
        std::bit_cast<T*>(inputData.get())[index] = value;
    }

    template <class T>
    T getResultAt(size_t idx)
    {
        PRECONDITION(idx < outputSize / sizeof(T), "Index is too large");
        return std::bit_cast<T*>(outputData.get())[idx];
    }

    void copyResultTo(std::span<std::byte> content)
    {
        std::ranges::copy_n(outputData.get(), std::min(content.size(), outputSize), content.data());
    }

    void addModelInput(std::span<std::byte> content)
    {
        std::ranges::copy_n(content.data(), std::min(content.size(), inputSize), inputData.get());
    }

    template <class T>
    void infer()
    {
        auto ireeOutputBV = runtimeWrapper.execute(functionName, inputData.get(), inputSize);
        runtimeWrapper.copyOutput(ireeOutputBV, reinterpret_cast<T*>(outputData.get()));
    }

    template <class T>
    size_t inferCombine(size_t outputSize)
    {
        auto ireeOutputBV = runtimeWrapper.execute(functionName, inputData.get(), inputSize);

        runtimeWrapper.copyOutput(ireeOutputBV, reinterpret_cast<T*>(outputData.get()), sizeof(T), outputSize, missIndices);
        missIndices.clear();

        return cacheMap.size();
    }

    void updateCacheMapIndices(uint64_t keyIdx, int rowIdx)
    {
        if (std::none_of(cacheMap.begin(), cacheMap.end(), [&](const auto& p) { return p.first == keyIdx; }))
        {
            cacheMap.emplace_back(std::make_pair(keyIdx, rowIdx));
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

    void appendMissIdx(int idx)
    {
        missIndices.insert(idx);
    }

    size_t getMissIndicesSize()
    {
        return missIndices.size();
    }

    std::unique_ptr<std::byte[]> inputData{};
    std::unique_ptr<std::byte[]> outputData{};
    std::unique_ptr<std::byte[]> cacheProbeTuple{};

    size_t inputSize;
    size_t outputSize;

    uint64_t misses;

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

    std::vector<std::pair<uint64_t, int>> cacheMap;
    std::set<int> missIndices;
};

}
