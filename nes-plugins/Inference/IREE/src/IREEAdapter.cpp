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

#include "IREEAdapter.hpp"
#include <fstream>
#include <cstdint>
#include <Util/Logger/Logger.hpp>
#include <iree/runtime/api.h>
#include "IREERuntimeWrapper.hpp"

#include <Model.hpp>

namespace
{
iree_const_byte_span_t asIREESpan(std::span<const std::byte> span)
{
    return iree_const_byte_span_t{.data = reinterpret_cast<const uint8_t*>(span.data()), .data_length = span.size()};
}
}

namespace NES
{

void IREEAdapter::initializeModel(Nebuli::Inference::Model& model, uint64_t batch_size)
{
    this->runtimeWrapper = IREERuntimeWrapper();
    runtimeWrapper.setup(asIREESpan(model.getByteCode()));

    auto inputShape = model.getInputShape();
    inputShape[0] = batch_size;
    runtimeWrapper.setInputShape(inputShape);

    runtimeWrapper.setNDim(model.getNDim());
    this->functionName = model.getFunctionName();

    auto inputSize = model.inputSize() * batch_size;
    this->inputData = std::make_unique<std::byte[]>(inputSize);
    this->inputSize = inputSize;
    runtimeWrapper.setInputDtype(dtypeMap.at(model.getInputDtype()));

    auto outputSize = model.outputSize() * batch_size;
    this->outputData = std::make_unique<std::byte[]>(outputSize);
    this->outputSize = outputSize;
    runtimeWrapper.setOutputDtype(dtypeMap.at(model.getOutputDtype()));
}

template <class T>
uint64_t IREEAdapter::addModelInputPartial(T value)
{
    const size_t thresholdHigh = std::ceil(1 / float(HIGH) * inputSize);
    const size_t thresholdMedium = std::ceil(1 / float(MEDIUM) * inputSize);
    const size_t thresholdLow = std::ceil(1 / float(LOW) * inputSize);

    uint64_t computedIndex = bytesProcessed / sizeof(T);

    if (inputDataEighth != nullptr && bytesProcessed < thresholdHigh)
    {
        currentReductionLevel = HIGH;
        std::bit_cast<T*>(inputDataEighth.get())[computedIndex] = value;
        bytesProcessed += sizeof(T);
    }
    else if (inputDataFourth != nullptr && bytesProcessed < thresholdMedium)
    {
        if (currentReductionLevel == HIGH)
        {
            std::memcpy(inputDataFourth.get(), inputDataEighth.get(), thresholdHigh);
        }
        currentReductionLevel = MEDIUM;
        std::bit_cast<T*>(inputDataFourth.get())[computedIndex] = value;
        bytesProcessed += sizeof(T);
    }
    else if (inputDataHalf != nullptr && bytesProcessed < thresholdLow)
    {
        if (currentReductionLevel == MEDIUM)
        {
            std::memcpy(inputDataHalf.get(), inputDataFourth.get(), thresholdMedium);
        }
        currentReductionLevel = LOW;
        std::bit_cast<T*>(inputDataHalf.get())[computedIndex] = value;
        bytesProcessed += sizeof(T);
    }
    else
    {
        if (currentReductionLevel == LOW)
        {
            std::memcpy(inputData.get(), inputDataHalf.get(), thresholdLow);
        }
        currentReductionLevel = NONE;
        std::bit_cast<T*>(inputData.get())[computedIndex] = value;
        bytesProcessed += sizeof(T);
    }
    return computedIndex;
}

void IREEAdapter::addModelInputBatchPartial(int index, std::span<std::byte> content, size_t tupleSize)
{
    const size_t thresholdHigh = std::ceil(1 / float(HIGH) * inputSize);
    const size_t thresholdMedium = std::ceil(1 / float(MEDIUM) * inputSize);
    const size_t thresholdLow = std::ceil(1 / float(LOW) * inputSize);

    const auto sizeToWrite = std::min(content.size(), tupleSize);

    if (inputDataEighth != nullptr && bytesProcessed + tupleSize <= thresholdHigh)
    {
        currentReductionLevel = HIGH;
        std::ranges::copy_n(content.data(), sizeToWrite, inputDataEighth.get() + bytesProcessed);
    }
    else if (inputDataFourth != nullptr && bytesProcessed + tupleSize <= thresholdMedium)
    {
        if (currentReductionLevel == HIGH)
        {
            std::memcpy(inputDataFourth.get(), inputDataEighth.get(), thresholdHigh);
        }
        currentReductionLevel = MEDIUM;
        std::ranges::copy_n(content.data(), sizeToWrite, inputDataFourth.get() + bytesProcessed);
    }
    else if (inputDataHalf != nullptr && bytesProcessed + tupleSize <= thresholdLow)
    {
        if (currentReductionLevel == MEDIUM)
        {
            std::memcpy(inputDataHalf.get(), inputDataFourth.get(), thresholdMedium);
        }
        currentReductionLevel = LOW;
        std::ranges::copy_n(content.data(), sizeToWrite, inputDataHalf.get() + bytesProcessed);
    }
    else
    {
        if (currentReductionLevel == LOW)
        {
            std::memcpy(inputData.get(), inputDataHalf.get(), thresholdLow);
        }
        currentReductionLevel = NONE;
        std::ranges::copy_n(content.data(), sizeToWrite, inputData.get() + index * tupleSize);
    }
    /// sometimes the varsized record can have a smaller size than the model expects
    /// it can happen, e.g., if we call the model on the output of array aggregation
    /// in this case, we cannot make use of the smaller inputData buffers and we have to still increment by the full tuple size
    bytesProcessed += tupleSize;
}

template <class T>
void IREEAdapter::infer()
{
    auto ireeOutputBV = runtimeWrapper.execute(functionName, inputData.get(), inputSize, currentReductionLevel);
    runtimeWrapper.copyOutput(ireeOutputBV, reinterpret_cast<T*>(outputData.get()));
}

template <class T>
void IREEAdapter::inferWithReduction()
{
    iree_hal_buffer_view_t* ireeOutputBV = nullptr;
    switch (currentReductionLevel)
    {
        default:
            ireeOutputBV = runtimeWrapper.execute(functionName, inputData.get(), inputSize, currentReductionLevel);
            break;
        case LOW:
            lowReductions += 1;
            ireeOutputBV = runtimeWrapper.execute(functionName, inputDataHalf.get(), std::ceil(1 / float(LOW) * inputSize), currentReductionLevel);
            break;
        case MEDIUM:
            mediumReductions += 1;
            ireeOutputBV = runtimeWrapper.execute(functionName, inputDataFourth.get(), std::ceil(1 / float(MEDIUM) * inputSize), currentReductionLevel);
            break;
        case HIGH:
            highReductions += 1;
            ireeOutputBV = runtimeWrapper.execute(functionName, inputDataEighth.get(), std::ceil(1 / float(HIGH) * inputSize), currentReductionLevel);
            break;
    }
    runtimeWrapper.copyOutput(ireeOutputBV, reinterpret_cast<T*>(outputData.get()));

    currentReductionLevel = NONE;
    bytesProcessed = 0;
}

template <class T>
size_t IREEAdapter::inferCombine(size_t outputSize, size_t outputFields, bool isVarSizedOutput)
{
    iree_hal_buffer_view_t* ireeOutputBV = nullptr;
    switch (currentReductionLevel)
    {
        default:
            ireeOutputBV = runtimeWrapper.execute(functionName, inputData.get(), inputSize, currentReductionLevel);
            break;
        case LOW:
            lowReductions += 1;
            ireeOutputBV = runtimeWrapper.execute(functionName, inputDataHalf.get(), std::ceil(1 / float(LOW) * inputSize), currentReductionLevel);
            break;
        case MEDIUM:
            mediumReductions += 1;
            ireeOutputBV = runtimeWrapper.execute(functionName, inputDataFourth.get(), std::ceil(1 / float(MEDIUM) * inputSize), currentReductionLevel);
            break;
        case HIGH:
            highReductions += 1;
            ireeOutputBV = runtimeWrapper.execute(functionName, inputDataEighth.get(), std::ceil(1 / float(HIGH) * inputSize), currentReductionLevel);
            break;
    }

    runtimeWrapper.copyOutput(ireeOutputBV, reinterpret_cast<T*>(outputData.get()), sizeof(T), outputSize, batchCachingHelper.getMissIndices(), outputFields, isVarSizedOutput);

    batchCachingHelper.clearMissIndices();
    currentReductionLevel = NONE;
    bytesProcessed = 0;

    return batchCachingHelper.getCacheMapSize();
}

template <class T>
void IREEAdapter::addModelInput(size_t index, T value)
{
    PRECONDITION(index < inputSize / sizeof(T), "Index is too large");
    std::bit_cast<T*>(inputData.get())[index] = value;
}

void IREEAdapter::addModelInput(std::span<std::byte> content)
{
    std::ranges::copy_n(content.data(), std::min(content.size(), inputSize), inputData.get());
}

void IREEAdapter::addModelInputBatch(int index, std::span<std::byte> content, size_t tupleSize)
{
    std::ranges::copy_n(content.data(), std::min(content.size(), tupleSize), inputData.get() + index * tupleSize);
}

template <class T>
T IREEAdapter::getResultAt(size_t idx)
{
    PRECONDITION(idx < outputSize / sizeof(T), "Index is too large");
    return std::bit_cast<T*>(outputData.get())[idx];
}

void IREEAdapter::copyResultTo(std::span<std::byte> content)
{
    PRECONDITION(outputSize == content.size(), "Output size does not match");
    std::ranges::copy_n(outputData.get(), std::min(content.size(), outputSize), content.data());
}

void IREEAdapter::copyResultToBatch(size_t index, std::span<std::byte> content)
{
    std::ranges::copy_n(outputData.get() + index * content.size(), content.size(), content.data());
}

void IREEAdapter::allocateBuffers(size_t tupleSize)
{
    cacheProbeTuple = std::make_unique<std::byte[]>(tupleSize);

    const size_t thresholdHigh = std::ceil(1 / float(HIGH) * inputSize);
    const size_t thresholdMedium = std::ceil(1 / float(MEDIUM) * inputSize);
    const size_t thresholdLow = std::ceil(1 / float(LOW) * inputSize);

    /// we only allocate smaller buffers if at least 1 tuple fits into memory
    if (tupleSize <= thresholdLow)
    {
        inputDataHalf = std::make_unique<std::byte[]>(thresholdLow);
    }
    if (tupleSize <= thresholdMedium)
    {
        inputDataFourth = std::make_unique<std::byte[]>(thresholdMedium);
    }
    if (tupleSize <= thresholdHigh)
    {
        inputDataEighth = std::make_unique<std::byte[]>(thresholdHigh);
    }
}

std::shared_ptr<IREEAdapter> IREEAdapter::create()
{
    auto adapter = std::make_shared<IREEAdapter>();
    return adapter;
}

#define NES_IREE_ADAPTER_INSTANTIATE(T)                                           \
    template void IREEAdapter::addModelInput<T>(size_t, T);                        \
    template uint64_t IREEAdapter::addModelInputPartial<T>(T);                         \
    template T IREEAdapter::getResultAt<T>(size_t);                                \
    template void IREEAdapter::infer<T>();                                         \
    template void IREEAdapter::inferWithReduction<T>();                                         \
    template size_t IREEAdapter::inferCombine<T>(size_t, size_t, bool);

NES_IREE_ADAPTER_INSTANTIATE(uint8_t)
NES_IREE_ADAPTER_INSTANTIATE(uint16_t)
NES_IREE_ADAPTER_INSTANTIATE(uint32_t)
NES_IREE_ADAPTER_INSTANTIATE(uint64_t)
NES_IREE_ADAPTER_INSTANTIATE(int8_t)
NES_IREE_ADAPTER_INSTANTIATE(int16_t)
NES_IREE_ADAPTER_INSTANTIATE(int32_t)
NES_IREE_ADAPTER_INSTANTIATE(int64_t)
NES_IREE_ADAPTER_INSTANTIATE(float)
NES_IREE_ADAPTER_INSTANTIATE(double)

#undef NES_IREE_ADAPTER_INSTANTIATE

}
