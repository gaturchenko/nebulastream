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
#include <Functions/PhysicalFunction.hpp>
#include <PhysicalOperator.hpp>
#include <PredictionCache.hpp>
#include <Windowing/WindowMetaData.hpp>
#include <WindowProbePhysicalOperator.hpp>
#include <Nautilus/Interface/BufferRef/TupleBufferRef.hpp>
#include <Nautilus/Interface/PagedVector/PagedVectorRef.hpp>
#include <InferenceConfiguration.hpp>
#include <HashMapOptions.hpp>

namespace NES
{

class IREEBatchCacheInferenceOperator : public WindowProbePhysicalOperator
{
public:
    IREEBatchCacheInferenceOperator(
        const OperatorHandlerId operatorHandlerId,
        std::vector<PhysicalFunction> inputs,
        std::vector<std::string> outputFieldNames,
        std::shared_ptr<TupleBufferRef> tupleBufferRef,
        Configurations::PredictionCacheOptions predictionCacheOptions,
        DataType inputDtype,
        DataType outputDtype,
        HashMapOptions hashMapOptions,
        bool useBatchDeduplication);

    void open(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const override;
    void close(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const override;
    void setup(ExecutionContext& executionCtx, CompilationContext&) const override;
    void terminate(ExecutionContext& executionCtx) const override;

    template <class T>
    nautilus::val<std::byte*> createCacheProbeTuple(
        nautilus::val<std::byte*> cacheProbeTuple,
        const nautilus::val<OperatorHandler*>& operatorHandler,
        ExecutionContext& executionCtx,
        Record& record) const;

    nautilus::val<std::byte*> createCacheProbeTupleVarsized(
        nautilus::val<std::byte*> cacheProbeTuple,
        const nautilus::val<OperatorHandler*>& operatorHandler,
        ExecutionContext& executionCtx,
        const nautilus::val<int8_t*>& varSizedContent,
        const nautilus::val<int32_t>& varSizedSize) const;

    std::pair<nautilus::val<uint64_t>, nautilus::val<std::byte*>> probeIntoCache(
        PredictionCache* predictionCache,
        nautilus::val<std::byte*> cacheProbeTuple) const;

    template <typename T>
    void writeToInputOrOutputBuffer(
        nautilus::val<std::byte*> prediction,
        const nautilus::val<OperatorHandler*>& operatorHandler,
        ExecutionContext& executionCtx,
        Record& record,
        const nautilus::val<uint64_t>& cacheKeyIndex,
        const nautilus::val<bool>& hasCachedPrediction,
        const nautilus::val<uint64_t>& outputRowIndex,
        const nautilus::val<uint64_t>& replacementIndex) const;

    void writeToInputOrOutputBufferVarsized(
        nautilus::val<std::byte*> prediction,
        const nautilus::val<OperatorHandler*>& operatorHandler,
        ExecutionContext& executionCtx,
        const nautilus::val<int8_t*>& varSizedContent,
        const nautilus::val<int32_t>& varSizedSize,
        const nautilus::val<uint64_t>& cacheKeyIndex,
        const nautilus::val<bool>& hasCachedPrediction,
        const nautilus::val<uint64_t>& replacementIndex,
        const nautilus::val<int>& rowIndex) const;

    template <class T>
    void updateCacheValues(
        PredictionCache* predictionCache,
        const nautilus::val<uint64_t>& cachePos,
        const nautilus::val<OperatorHandler*>& operatorHandler,
        const nautilus::val<WorkerThreadId>& threadId,
        const nautilus::val<size_t>& valueToUpdate) const;

    void updateCacheValuesVarsized(
        PredictionCache* predictionCache,
        const nautilus::val<uint64_t>& cachePos,
        const nautilus::val<OperatorHandler*>& operatorHandler,
        const nautilus::val<WorkerThreadId>& threadId,
        const nautilus::val<size_t>& valueToUpdate) const;

    [[nodiscard]] Record createRecord(const Record& featureRecord, const std::vector<Record::RecordFieldIdentifier>& projections) const;

    [[nodiscard]] std::optional<struct PhysicalOperator> getChild() const override { return child; }
    void setChild(PhysicalOperator child) override { this->child = std::move(child); }

    bool isVarSizedInput = false;
    bool isVarSizedOutput = false;
    size_t outputSize = 0;
    size_t inputSize = 0;
    bool useBatchDeduplication = false;

private:
    const std::vector<PhysicalFunction> inputs;
    const std::vector<std::string> outputFieldNames;
    std::optional<PhysicalOperator> child;
    std::shared_ptr<TupleBufferRef> tupleBufferRef;
    Configurations::PredictionCacheOptions predictionCacheOptions;
    DataType inputDtype;
    DataType outputDtype;
    HashMapOptions hashMapOptions;

protected:
    template <class T>
    void performInference(
        const PagedVectorRef& pagedVectorRef,
        TupleBufferRef& tupleBufferRef,
        ExecutionContext& executionCtx,
        nautilus::val<HashMap*> hashMapPtr,
        ChainedHashMapRef& hashMap) const;

    template <class T>
    void writeOutputRecord(
        const PagedVectorRef& pagedVectorRef,
        TupleBufferRef& tupleBufferRef,
        ExecutionContext& executionCtx,
        nautilus::val<HashMap*> hashMapPtr,
        ChainedHashMapRef& hashMap) const;
        // const nautilus::val<std::byte*>& prediction) const;
};

}
