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
    void addModelInput(size_t index, T value)
    {
        PRECONDITION(index < inputSize / sizeof(T), "Index is too large");
        std::bit_cast<T*>(inputData.get())[index] = value;
    }

    void addModelInput(std::span<std::byte> content)
    {
        std::ranges::copy_n(content.data(), std::min(content.size(), inputSize), inputData.get());
    }

    template <class T>
    void addModelInputPartial(T value)
    {
        const size_t thresholdHigh = std::ceil(1 / float(HIGH) * inputSize);
        const size_t thresholdMedium = std::ceil(1 / float(MEDIUM) * inputSize);
        const size_t thresholdLow = std::ceil(1 / float(LOW) * inputSize);
        
        if (bytesProcessed < thresholdHigh)
        {
            currentReductionLevel = HIGH;
            std::bit_cast<T*>(inputDataEighth.get())[bytesProcessed / sizeof(T)] = value;
            bytesProcessed += sizeof(T);
        }
        else if (bytesProcessed < thresholdMedium)
        {
            if (currentReductionLevel != MEDIUM)
            {
                std::memcpy(inputDataFourth.get(), inputDataEighth.get(), thresholdHigh);
                currentReductionLevel = MEDIUM;
            }
            std::bit_cast<T*>(inputDataFourth.get())[bytesProcessed / sizeof(T)] = value;
            bytesProcessed += sizeof(T);
        }
        else if (bytesProcessed < thresholdLow)
        {
            if (currentReductionLevel != LOW)
            {
                std::memcpy(inputDataHalf.get(), inputDataFourth.get(), thresholdMedium);
                currentReductionLevel = LOW;
            }
            std::bit_cast<T*>(inputDataHalf.get())[bytesProcessed / sizeof(T)] = value;
            bytesProcessed += sizeof(T);
        }
        else
        {
            if (currentReductionLevel != NONE)
            {
                std::memcpy(inputData.get(), inputDataHalf.get(), thresholdLow);
                currentReductionLevel = NONE;
            }
            std::bit_cast<T*>(inputData.get())[bytesProcessed / sizeof(T)] = value;
            bytesProcessed += sizeof(T);
        }
    }

    template <class T>
    void addModelInputBatchPartial(size_t index, std::span<std::byte> content)
    {
        const size_t thresholdHigh = std::ceil(1 / float(HIGH) * inputSize);
        const size_t thresholdMedium = std::ceil(1 / float(MEDIUM) * inputSize);
        const size_t thresholdLow = std::ceil(1 / float(LOW) * inputSize);

        if (bytesProcessed < thresholdHigh && bytesProcessed + content.size() <= thresholdHigh)
        {
            currentReductionLevel = HIGH;
            std::ranges::copy_n(content.data(), content.size(), inputDataEighth.get() + index * sizeof(T));
            bytesProcessed += content.size();
        }
        else if (bytesProcessed < thresholdMedium && bytesProcessed + content.size() <= thresholdMedium)
        {
            if (currentReductionLevel != MEDIUM)
            {
                std::memcpy(inputDataFourth.get(), inputDataEighth.get(), thresholdHigh);
                currentReductionLevel = MEDIUM;
            }
            std::ranges::copy_n(content.data(), content.size(), inputDataFourth.get() + index * sizeof(T));
            bytesProcessed += content.size();
        }
        else if (bytesProcessed < thresholdLow && bytesProcessed + content.size() <= thresholdLow)
        {
            if (currentReductionLevel != LOW)
            {
                std::memcpy(inputDataHalf.get(), inputDataFourth.get(), thresholdMedium);
                currentReductionLevel = LOW;
            }
            std::ranges::copy_n(content.data(), content.size(), inputDataHalf.get() + index * sizeof(T));
            bytesProcessed += content.size();
        }
        else
        {
            if (currentReductionLevel != NONE)
            {
                std::memcpy(inputData.get(), inputDataHalf.get(), thresholdLow);
                currentReductionLevel = NONE;
            }
            std::ranges::copy_n(content.data(), content.size(), inputData.get() + index * sizeof(T));
            bytesProcessed += content.size();
        }
    }

    template <class T>
    void addModelInputBatch(size_t index, std::span<std::byte> content)
    {
        std::ranges::copy_n(content.data(), content.size(), inputData.get() + index * sizeof(T));
    }

    template <class T>
    T getResultAt(size_t idx)
    {
        PRECONDITION(idx < outputSize / sizeof(T), "Index is too large");
        return std::bit_cast<T*>(outputData.get())[idx];
    }

    void copyResultTo(std::span<std::byte> content)
    {
        PRECONDITION(outputSize == content.size(), "Output size does not match");
        std::ranges::copy_n(outputData.get(), std::min(content.size(), outputSize), content.data());
    }

    template <class T>
    void copyResultToBatch(size_t index, std::span<std::byte> content)
    {
        std::ranges::copy_n(outputData.get() + index * sizeof(T), content.size(), content.data());
    }

    template <class T>
    void infer()
    {
        auto ireeOutputBV = runtimeWrapper.execute(functionName, inputData.get(), inputSize, currentReductionLevel);
        runtimeWrapper.copyOutput(ireeOutputBV, reinterpret_cast<T*>(outputData.get()));
    }

    template <class T>
    size_t inferCombine(size_t outputSize, size_t outputFields)
    {
        iree_hal_buffer_view_t* ireeOutputBV = nullptr;
        switch (currentReductionLevel)
        {
            default:
                ireeOutputBV = runtimeWrapper.execute(functionName, inputData.get(), inputSize, currentReductionLevel);
                break;
            case LOW:
                ireeOutputBV = runtimeWrapper.execute(functionName, inputDataHalf.get(), std::ceil(1 / float(LOW) * inputSize), currentReductionLevel);
                break;
            case MEDIUM:
                ireeOutputBV = runtimeWrapper.execute(functionName, inputDataFourth.get(), std::ceil(1 / float(MEDIUM) * inputSize), currentReductionLevel);
                break;
            case HIGH:
                ireeOutputBV = runtimeWrapper.execute(functionName, inputDataEighth.get(), std::ceil(1 / float(HIGH) * inputSize), currentReductionLevel);
                break;
        }
        runtimeWrapper.copyOutput(ireeOutputBV, reinterpret_cast<T*>(outputData.get()), sizeof(T), outputSize, missIndices, outputFields);

        missIndices.clear();
        currentReductionLevel = NONE;
        bytesProcessed = 0;

        return cacheMap.size();
    }

    void allocateBuffers(size_t tupleSize)
    {
        cacheProbeTuple = std::make_unique<std::byte[]>(tupleSize);

        inputDataHalf = std::make_unique<std::byte[]>(std::ceil(1 / float(LOW) * inputSize));
        inputDataFourth = std::make_unique<std::byte[]>(std::ceil(1 / float(MEDIUM) * inputSize));
        inputDataEighth = std::make_unique<std::byte[]>(std::ceil(1 / float(HIGH) * inputSize));
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

    /// input for IREE runtime
    std::unique_ptr<std::byte[]> inputData{};
    std::unique_ptr<std::byte[]> outputData{};

    /// helper objects for the BatchCache operator
    std::unique_ptr<std::byte[]> cacheProbeTuple{};
    std::unique_ptr<std::byte[]> inputDataHalf{};
    std::unique_ptr<std::byte[]> inputDataFourth{};
    std::unique_ptr<std::byte[]> inputDataEighth{};
    InputBufferSizeReduction currentReductionLevel = NONE;
    uint64_t bytesProcessed = 0;

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
