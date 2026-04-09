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

#include <ExecutionContext.hpp>
#include <IREEAdapter.hpp>
#include <IREEBatchCacheInferenceOperator.hpp>
#include <IREEBatchInferenceOperatorHandler.hpp>
#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedHashMapRef.hpp>
#include <Nautilus/Interface/PagedVector/PagedVectorRef.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <PredictionCacheOperatorHandler.hpp>
#include <PredictionCache/PredictionCacheUtil.hpp>
#include <QueryExecutionConfiguration.hpp>
#include <nautilus/function.hpp>

namespace NES::QueryCompilation::PhysicalOperators
{
class PhysicalInferModelOperator;
}

namespace NES::IREEBatchCacheInference
{
inline IREEBatchInferenceOperatorHandler* getHandler(OperatorHandler* inferModelHandler)
{
    return dynamic_cast<IREEBatchInferenceOperatorHandler*>(inferModelHandler);
}

inline IREEAdapter* getAdapter(OperatorHandler* inferModelHandler, WorkerThreadId thread)
{
    return getHandler(inferModelHandler)->getIREEAdapter(thread).get();
}

template <class T>
int addValueToModelProxy(
    int indexOutput,
    T value,
    OperatorHandler* inferModelHandler,
    WorkerThreadId thread,
    uint64_t keyIdx,
    bool notFound)
{
    auto* adapter = getAdapter(inferModelHandler, thread);

    /// we need to write the row index of the tuple so as to know where to insert it in the output byte array after the model call
    /// we do it only if the key does not exist in the cache, otherwise it's a hit for the key that has no respective value
    /// therefore, we shouldn't update any existing entries in the cache map
    if (notFound)
    {
        adapter->batchCachingHelper.updateCacheMapIndices(keyIdx, indexOutput);
    }
    adapter->batchCachingHelper.appendMissIdx(indexOutput);

    auto currentIdx = adapter->addModelInputPartial<T>(value);
    return static_cast<int>(currentIdx);
}

template <class T>
T getValueFromModelProxy(int index, OperatorHandler* inferModelHandler, WorkerThreadId thread)
{
    auto* adapter = getAdapter(inferModelHandler, thread);
    return adapter->getResultAt<T>(index);
}

void copyVarSizedToModelProxy(
    int index,
    std::byte* content,
    uint32_t size,
    size_t tupleSize,
    OperatorHandler* inferModelHandler,
    WorkerThreadId thread,
    uint64_t keyIdx,
    bool notFound)
{
    auto* adapter = getAdapter(inferModelHandler, thread);

    /// we need to write the row index of the tuple so as to know where to insert it in the output byte array after the model call
    /// we do it only if the key does not exist in the cache, otherwise it's a hit for the key that has no respective value
    /// therefore, we shouldn't update any existing entries in the cache map
    if (notFound)
    {
        adapter->batchCachingHelper.updateCacheMapIndices(keyIdx, index);
    }
    adapter->batchCachingHelper.appendMissIdx(index);
    adapter->misses += 1;

    adapter->addModelInputBatchPartial(index, std::span{content, size}, tupleSize);
}

void copyVarSizedFromModelProxy(int index, std::byte* content, uint32_t size, OperatorHandler* inferModelHandler, WorkerThreadId thread)
{
    auto* adapter = getAdapter(inferModelHandler, thread);
    adapter->copyResultToBatch(index, std::span{content, size});
}

template <class T>
size_t applyModelProxy(
    OperatorHandler* inferModelHandler,
    WorkerThreadId thread,
    size_t outputSize,
    size_t outputFields,
    bool isVarSizedOutput)
{
    auto* adapter = getAdapter(inferModelHandler, thread);
    /// call the model only if any misses were recorded
    if (adapter->batchCachingHelper.getMissIndicesSize() > 0)
    {
        return adapter->inferCombine<T>(outputSize, outputFields, isVarSizedOutput);
    }
    adapter->fullReductions += 1;
    return 0;
}

nautilus::val<uint32_t> min(const nautilus::val<uint32_t>& lhs, const nautilus::val<uint32_t>& rhs)
{
    return lhs < rhs ? lhs : rhs;
}

void garbageCollectBatchesProxy(OperatorHandler* handler, WorkerThreadId thread)
{
    auto* inferModelHandler = getHandler(handler);
    inferModelHandler->clearHashMap(thread);
    inferModelHandler->garbageCollectBatches();
}
}

namespace NES
{

IREEBatchCacheInferenceOperator::IREEBatchCacheInferenceOperator(
    const OperatorHandlerId operatorHandlerId,
    std::vector<PhysicalFunction> inputs,
    std::vector<std::string> outputFieldNames,
    std::shared_ptr<TupleBufferRef> tupleBufferRef,
    Configurations::PredictionCacheOptions predictionCacheOptions,
    DataType inputDtype,
    DataType outputDtype,
    HashMapOptions hashMapOptions,
    bool useBatchDeduplication)
    : WindowProbePhysicalOperator(operatorHandlerId)
    , useBatchDeduplication(useBatchDeduplication)
    , inputs(std::move(inputs))
    , outputFieldNames(std::move(outputFieldNames))
    , tupleBufferRef(std::move(tupleBufferRef))
    , predictionCacheOptions(predictionCacheOptions)
    , inputDtype(inputDtype)
    , outputDtype(outputDtype)
    , hashMapOptions(std::move(hashMapOptions))
{
}

template <class T>
nautilus::val<std::byte*> IREEBatchCacheInferenceOperator::createCacheProbeTuple(
    nautilus::val<std::byte*> cacheProbeTuple,
    const nautilus::val<OperatorHandler*>& operatorHandler,
    ExecutionContext& executionCtx,
    Record& record) const
{
    for (nautilus::static_val<size_t> i = 0; i < inputs.size(); ++i)
    {
        cacheProbeTuple = nautilus::invoke(
            +[](OperatorHandler* inferModelHandler, WorkerThreadId thread, size_t idx, T value)
            {
                auto* adapter = IREEBatchCacheInference::getAdapter(inferModelHandler, thread);
                std::bit_cast<T*>(adapter->cacheProbeTuple.get())[idx] = value;
                return adapter->cacheProbeTuple.get();
            }, operatorHandler, executionCtx.workerThreadId, nautilus::val<int>(i),
                inputs.at(i).execute(record, executionCtx.pipelineMemoryProvider.arena).cast<nautilus::val<T>>());
    }

    return cacheProbeTuple;
}

nautilus::val<std::byte*> IREEBatchCacheInferenceOperator::createCacheProbeTupleVarsized(
    nautilus::val<std::byte*> cacheProbeTuple,
    const nautilus::val<OperatorHandler*>& operatorHandler,
    ExecutionContext& executionCtx,
    const nautilus::val<int8_t*>& varSizedContent,
    const nautilus::val<int32_t>& varSizedSize) const
{

    cacheProbeTuple = nautilus::invoke(
        +[](OperatorHandler* inferModelHandler, WorkerThreadId thread, std::byte* content, uint32_t size, uint32_t tupleSize)
        {
            auto* adapter = IREEBatchCacheInference::getAdapter(inferModelHandler, thread);
            std::memcpy(adapter->cacheProbeTuple.get(), content, std::min(size, tupleSize));
            return adapter->cacheProbeTuple.get();
        },
        operatorHandler,
        executionCtx.workerThreadId,
        varSizedContent,
        varSizedSize,
        nautilus::val<uint32_t>(static_cast<uint32_t>(inputSize)));
    return cacheProbeTuple;
}

std::pair<nautilus::val<uint64_t>, nautilus::val<std::byte*>> IREEBatchCacheInferenceOperator::probeIntoCache(
    PredictionCache* predictionCache,
    nautilus::val<std::byte*> cacheProbeTuple) const
{
    /// if the probe is successful, return the index of the key, otherwise return PredictionCache::NOT_FOUND, i.e., UINT64_MAX
    auto cacheKeyIndex = predictionCache->updateKeys(
        cacheProbeTuple,
        [&](
            const nautilus::val<PredictionCacheEntry*>& predictionCacheEntryToReplace, const nautilus::val<uint64_t>&)
        {
            return nautilus::invoke(
                +[](PredictionCacheEntry* predictionCacheEntry, std::byte* tuple, size_t size)
                {
                    predictionCacheEntry->dataStructure = nullptr;
                    predictionCacheEntry->recordSize = size;
                    predictionCacheEntry->record = new std::byte[size];

                    std::memcpy(predictionCacheEntry->record, tuple, size);
                },
                predictionCacheEntryToReplace,
                cacheProbeTuple,
                nautilus::val<int>(this->inputSize));
        });

    /// the key might be in the cache already, since the replacement function above may have been invoked
    /// however, the corresponding value may not yet exist, e.g., if we are processing the very first batch
    auto prediction = predictionCache->getDataStructure(cacheKeyIndex);

    return {cacheKeyIndex, prediction};
}

template <class T>
void IREEBatchCacheInferenceOperator::writeToInputOrOutputBuffer(
    nautilus::val<std::byte*> prediction,
    const nautilus::val<OperatorHandler*>& operatorHandler,
    ExecutionContext& executionCtx,
    Record& record,
    const nautilus::val<uint64_t>& cacheKeyIndex,
    const nautilus::val<bool>& hasCachedPrediction,
    const nautilus::val<uint64_t>& outputRowIndex,
    const nautilus::val<uint64_t>& replacementIndex) const
{
    /// if the key does not exist or it does but the corresponding value does not,
    /// then we write the tuple to the byte array reserved for the inputs to the model;
    /// we pick the smallest allocated buffer first and copy to a larger one if the size is exceeded
    const nautilus::val<bool> keyNotFound = cacheKeyIndex == PredictionCache::NOT_FOUND;
    if (keyNotFound || !hasCachedPrediction)
    {
        for (nautilus::static_val<size_t> i = 0; i < inputs.size(); ++i)
        {
            nautilus::invoke(
                IREEBatchCacheInference::addValueToModelProxy<T>,
                outputRowIndex,
                inputs.at(i).execute(record, executionCtx.pipelineMemoryProvider.arena).cast<nautilus::val<T>>(),
                operatorHandler,
                executionCtx.workerThreadId,
                replacementIndex,
                keyNotFound);
        }
    }
    /// otherwise, we know the prediction for this tuple in the batch and immediately write it to the output byte array
    else
    {
        nautilus::invoke(
            +[](int idx, std::byte* prediction, OperatorHandler* inferModelHandler, WorkerThreadId thread, size_t size)
            {
                auto* adapter = IREEBatchCacheInference::getAdapter(inferModelHandler, thread);
                std::memcpy(adapter->outputData.get() + idx * sizeof(T), prediction, size);
            },
            outputRowIndex,
            prediction,
            operatorHandler,
            executionCtx.workerThreadId,
            nautilus::val<size_t>(this->outputSize));
    }
}

void IREEBatchCacheInferenceOperator::writeToInputOrOutputBufferVarsized(
    nautilus::val<std::byte*> prediction,
    const nautilus::val<OperatorHandler*>& operatorHandler,
    ExecutionContext& executionCtx,
    const nautilus::val<int8_t*>& varSizedContent,
    const nautilus::val<int32_t>& varSizedSize,
    const nautilus::val<uint64_t>& cacheKeyIndex,
    const nautilus::val<bool>& hasCachedPrediction,
    const nautilus::val<uint64_t>& replacementIndex,
    const nautilus::val<int>& rowIndex) const
{
    /// if the key does not exist or it does but the corresponding value does not,
    /// then we write the tuple to the byte array reserved for the inputs to the model;
    /// we pick the smallest allocated buffer first and copy to a larger one if the size is exceeded
    const nautilus::val<bool> keyNotFound = cacheKeyIndex == PredictionCache::NOT_FOUND;
    if (keyNotFound || !hasCachedPrediction)
    {
        nautilus::invoke(
            IREEBatchCacheInference::copyVarSizedToModelProxy,
            rowIndex,
            varSizedContent,
            varSizedSize,
            nautilus::val<size_t>(inputSize),
            operatorHandler,
            executionCtx.workerThreadId,
            replacementIndex,
            keyNotFound);
    }
    /// otherwise, we know the prediction for this tuple in the batch and immediately write it to the output byte array
    else
    {
        nautilus::invoke(
            +[](int idx, std::byte* prediction, OperatorHandler* inferModelHandler, WorkerThreadId thread, size_t size)
            {
                auto* adapter = IREEBatchCacheInference::getAdapter(inferModelHandler, thread);
                std::memcpy(adapter->outputData.get() + idx * size, prediction, size);
            },
            rowIndex,
            prediction,
            operatorHandler,
            executionCtx.workerThreadId,
            nautilus::val<size_t>(this->outputSize));
    }
}

template <class T>
void IREEBatchCacheInferenceOperator::updateCacheValues(
    PredictionCache* predictionCache,
    const nautilus::val<uint64_t>& cachePos,
    const nautilus::val<OperatorHandler*>& operatorHandler,
    const nautilus::val<WorkerThreadId>& threadId,
    const nautilus::val<size_t>& valueToUpdate) const
{
    predictionCache->updateValues(
        cachePos,
        [&](
            const nautilus::val<PredictionCacheEntry*>& predictionCacheEntryToReplace, const nautilus::val<uint64_t>&)
        {
            return nautilus::invoke(
                +[](PredictionCacheEntry* predictionCacheEntry, OperatorHandler* inferModelHandler, WorkerThreadId thread, int idx, size_t size)
                {
                    auto* adapter = IREEBatchCacheInference::getAdapter(inferModelHandler, thread);

                    int outputPos = adapter->batchCachingHelper.getCacheMapValue(idx);

                    predictionCacheEntry->dataSize = size;
                    predictionCacheEntry->dataStructure = new std::byte[size];

                    std::memcpy(predictionCacheEntry->dataStructure, adapter->outputData.get() + outputPos * sizeof(T), size);
                }, predictionCacheEntryToReplace, operatorHandler, threadId, valueToUpdate, nautilus::val<size_t>(this->outputSize));
        });
}

void IREEBatchCacheInferenceOperator::updateCacheValuesVarsized(
    PredictionCache* predictionCache,
    const nautilus::val<uint64_t>& cachePos,
    const nautilus::val<OperatorHandler*>& operatorHandler,
    const nautilus::val<WorkerThreadId>& threadId,
    const nautilus::val<size_t>& valueToUpdate) const
{
    predictionCache->updateValues(
        cachePos,
        [&](
            const nautilus::val<PredictionCacheEntry*>& predictionCacheEntryToReplace, const nautilus::val<uint64_t>&)
        {
            return nautilus::invoke(
                +[](PredictionCacheEntry* predictionCacheEntry, OperatorHandler* opHandlerPtr, WorkerThreadId thread, int idx, size_t size)
                {
                    auto* adapter = IREEBatchCacheInference::getAdapter(opHandlerPtr, thread);

                    int outputPos = adapter->batchCachingHelper.getCacheMapValue(idx);

                    predictionCacheEntry->dataSize = size;
                    predictionCacheEntry->dataStructure = new std::byte[size];

                    std::memcpy(predictionCacheEntry->dataStructure, adapter->outputData.get() + outputPos * size, size);
                }, predictionCacheEntryToReplace, operatorHandler, threadId, valueToUpdate, nautilus::val<size_t>(this->outputSize));
        });
}

template <typename T>
void IREEBatchCacheInferenceOperator::performInference(
    const PagedVectorRef& pagedVectorRef,
    TupleBufferRef& tupleBufferRef,
    ExecutionContext& executionCtx,
    const nautilus::val<HashMap*>& hashMapPtr,
    ChainedHashMapRef& hashMap) const
{
    const auto fields = tupleBufferRef.getMemoryLayout()->getSchema().getFieldNames();
    auto* predictionCache = dynamic_cast<PredictionCache*>(executionCtx.getLocalState(id));
    const auto operatorHandler = predictionCache->getOperatorHandler();

    /// iterate over records in the paged vector, i.e., over tuples in a single batch
    nautilus::val<int> rowIndex(0);
    for (auto it = pagedVectorRef.begin(fields); it != pagedVectorRef.end(fields); ++it)
    {
        auto record = createRecord(*it, fields);
        auto outputRowIndex = rowIndex * this->outputSize / this->inputSize;
        nautilus::val<std::byte*> cacheProbeTuple;

        if (useBatchDeduplication)
        {
            /// 0. check if the record is a duplicate, if it is, we don't do any processing
            const auto hashMapEntry = hashMap.findOrCreateEntry(
                record,
                *hashMapOptions.hashFunction,
                [&](const nautilus::val<AbstractHashMapEntry*>& entry)
                {
                    /// if the entry is not found, create a record where we will store the respective output buffer index
                    const ChainedHashMapRef::ChainedEntryRef ref(entry, hashMapPtr, hashMapOptions.fieldKeys, hashMapOptions.fieldValues);
                    Record valueRecord;

                    valueRecord.write("rowInputIndex", VarVal(rowIndex));
                    valueRecord.write("rowOutputIndex", VarVal(0));
                    ref.copyValuesToEntry(valueRecord, executionCtx.pipelineMemoryProvider.bufferProvider);
                }, executionCtx.pipelineMemoryProvider.bufferProvider);

            const ChainedHashMapRef::ChainedEntryRef entryRef(hashMapEntry, hashMapPtr, hashMapOptions.fieldKeys, hashMapOptions.fieldValues);
            auto entryRowIndex = entryRef.getValue().read("rowInputIndex").cast<nautilus::val<int>>();

            /// the entry has already been inserted, so the record is a duplicate and we can continue iterating over the batch
            if (entryRowIndex != rowIndex)
            {
                continue;
            }

            if (!this->isVarSizedInput)
            {
                /// 1. fill a byte array for a record to probe into the cache
                Record valueRecord = entryRef.getValue();
                valueRecord.write("rowOutputIndex", VarVal(outputRowIndex));
                entryRef.copyValuesToEntry(valueRecord, executionCtx.pipelineMemoryProvider.bufferProvider);

                cacheProbeTuple = createCacheProbeTuple<T>(cacheProbeTuple, operatorHandler, executionCtx, record);

                /// 2. probe into the cache and check whether there's a prediction for the given key
                /// (the key might be in the cache but the value may not be there yet since we invoke the model on a batch)
                auto [cacheKeyIndex, prediction] = probeIntoCache(predictionCache, cacheProbeTuple);
                const auto hasCachedPrediction = nautilus::invoke(
                    +[](std::byte* prediction){ return prediction != nullptr; }, prediction);

                /// 3. write either to the input, or to the output byte buffer in the adapter based on the probing outcome
                writeToInputOrOutputBuffer<T>(prediction, operatorHandler, executionCtx, record, cacheKeyIndex,
                    hasCachedPrediction, outputRowIndex, predictionCache->getReplacementIndex());

                rowIndex += inputs.size();
            }
            else
            {
                /// 1. fill a byte array for a record to probe into the cache
                Record valueRecord = entryRef.getValue();
                valueRecord.write("rowOutputIndex", VarVal(rowIndex));
                entryRef.copyValuesToEntry(valueRecord, executionCtx.pipelineMemoryProvider.bufferProvider);

                const VarVal inputValue = inputs.at(0).execute(record, executionCtx.pipelineMemoryProvider.arena);
                const auto varSizedValue = inputValue.cast<VariableSizedData>();

                cacheProbeTuple = createCacheProbeTupleVarsized(cacheProbeTuple, operatorHandler, executionCtx, varSizedValue.getContent(),
                    IREEBatchCacheInference::min(varSizedValue.getContentSize(), nautilus::val<uint32_t>(static_cast<uint32_t>(inputSize))));

                /// 2. probe into the cache and check whether there's a prediction for the given key
                /// (the key might be in the cache but the value may not be there yet since we invoke the model on a batch)
                auto [cacheKeyIndex, prediction] = probeIntoCache(predictionCache, cacheProbeTuple);
                const auto hasCachedPrediction = nautilus::invoke(
                    +[](std::byte* prediction){ return prediction != nullptr; }, prediction);

                /// 3. write either to the input, or to the output byte buffer in the adapter based on the probing outcome
                writeToInputOrOutputBufferVarsized(prediction, operatorHandler, executionCtx, varSizedValue.getContent(),
                    IREEBatchCacheInference::min(varSizedValue.getContentSize(), nautilus::val<uint32_t>(static_cast<uint32_t>(inputSize))),
                    cacheKeyIndex, hasCachedPrediction, predictionCache->getReplacementIndex(), rowIndex);

                rowIndex += inputs.size();
            }
        }
        else
        {
            if (!this->isVarSizedInput)
            {
                /// 1. fill a byte array for a record to probe into the cache
                cacheProbeTuple = createCacheProbeTuple<T>(cacheProbeTuple, operatorHandler, executionCtx, record);

                /// 2. probe into the cache and check whether there's a prediction for the given key
                /// (the key might be in the cache but the value may not be there yet since we invoke the model on a batch)
                auto [cacheKeyIndex, prediction] = probeIntoCache(predictionCache, cacheProbeTuple);
                const auto hasCachedPrediction = nautilus::invoke(
                    +[](std::byte* prediction){ return prediction != nullptr; }, prediction);

                /// 3. write either to the input, or to the output byte buffer in the adapter based on the probing outcome
                writeToInputOrOutputBuffer<T>(prediction, operatorHandler, executionCtx, record, cacheKeyIndex,
                    hasCachedPrediction, outputRowIndex, predictionCache->getReplacementIndex());

                rowIndex += inputs.size();
            }
            else
            {
                /// 1. fill a byte array for a record to probe into the cache
                const VarVal inputValue = inputs.at(0).execute(record, executionCtx.pipelineMemoryProvider.arena);
                const auto varSizedValue = inputValue.cast<VariableSizedData>();

                cacheProbeTuple = createCacheProbeTupleVarsized(cacheProbeTuple, operatorHandler, executionCtx, varSizedValue.getContent(),
                    IREEBatchCacheInference::min(varSizedValue.getContentSize(), nautilus::val<uint32_t>(static_cast<uint32_t>(inputSize))));

                /// 2. probe into the cache and check whether there's a prediction for the given key
                /// (the key might be in the cache but the value may not be there yet since we invoke the model on a batch)
                auto [cacheKeyIndex, prediction] = probeIntoCache(predictionCache, cacheProbeTuple);
                const auto hasCachedPrediction = nautilus::invoke(
                    +[](std::byte* prediction){ return prediction != nullptr; }, prediction);

                /// 3. write either to the input, or to the output byte buffer in the adapter based on the probing outcome
                writeToInputOrOutputBufferVarsized(prediction, operatorHandler, executionCtx, varSizedValue.getContent(),
                    IREEBatchCacheInference::min(varSizedValue.getContentSize(), nautilus::val<uint32_t>(static_cast<uint32_t>(inputSize))),
                    cacheKeyIndex, hasCachedPrediction, predictionCache->getReplacementIndex(), rowIndex);

                rowIndex += inputs.size();
            }
        }
    }

    /// 4. call the model on the tuples which had a cache miss during probing
    const auto valuesToUpdate = nautilus::invoke(
        IREEBatchCacheInference::applyModelProxy<T>,
        operatorHandler,
        executionCtx.workerThreadId,
        nautilus::val<size_t>(this->outputSize),
        nautilus::val<size_t>(outputFieldNames.size()),
        nautilus::val<bool>(isVarSizedOutput));

    /// 5. update values for the keys that don't have them yet
    for (nautilus::val<size_t> i = 0; i < valuesToUpdate; ++i)
    {
        const auto cachePos = nautilus::invoke(
            +[](size_t i, OperatorHandler* inferModelHandler, WorkerThreadId thread)
            {
                auto* adapter = IREEBatchCacheInference::getAdapter(inferModelHandler, thread);
                return adapter->batchCachingHelper.getCacheMapKey(i);
            }, i, operatorHandler, executionCtx.workerThreadId);

        if (!isVarSizedOutput)
        {
            updateCacheValues<T>(predictionCache, cachePos, operatorHandler, executionCtx.workerThreadId, i);
        }
        else
        {
            updateCacheValuesVarsized(predictionCache, cachePos, operatorHandler, executionCtx.workerThreadId, i);
        }
    }
}

template <typename T>
void IREEBatchCacheInferenceOperator::writeOutputRecord(
    const PagedVectorRef& pagedVectorRef,
    TupleBufferRef& tupleBufferRef,
    ExecutionContext& executionCtx,
    const nautilus::val<HashMap*>& hashMapPtr,
    ChainedHashMapRef& hashMap) const
{
    const auto fields = tupleBufferRef.getMemoryLayout()->getSchema().getFieldNames();
    auto* predictionCache = dynamic_cast<PredictionCache*>(executionCtx.getLocalState(id));
    const auto operatorHandler = predictionCache->getOperatorHandler();

    nautilus::val<int> rowIndex(0);
    for (auto it = pagedVectorRef.begin(fields); it != pagedVectorRef.end(fields); ++it)
    {
        auto record = createRecord(*it, fields);

        if (useBatchDeduplication)
        {
            const auto hashMapEntry = hashMap.findOrCreateEntry(
                record,
                *hashMapOptions.hashFunction,
                [&](const nautilus::val<AbstractHashMapEntry*>&){},
                executionCtx.pipelineMemoryProvider.bufferProvider);
            const ChainedHashMapRef::ChainedEntryRef entryRef(hashMapEntry, hashMapPtr, hashMapOptions.fieldKeys, hashMapOptions.fieldValues);

            if (!this->isVarSizedOutput)
            {
                for (nautilus::static_val<size_t> i = 0; i < outputFieldNames.size(); ++i)
                {
                    VarVal result = VarVal(nautilus::invoke(
                        IREEBatchCacheInference::getValueFromModelProxy<T>,
                        entryRef.getValue().read("rowOutputIndex").cast<nautilus::val<int>>() + i,
                        operatorHandler,
                        executionCtx.workerThreadId));

                    record.write(outputFieldNames.at(i), result);
                    ++rowIndex;
                }
            }
            else
            {
                auto output = executionCtx.pipelineMemoryProvider.arena.allocateVariableSizedData(this->outputSize);

                nautilus::invoke(
                    IREEBatchCacheInference::copyVarSizedFromModelProxy,
                    entryRef.getValue().read("rowOutputIndex").cast<nautilus::val<int>>(),
                    output.getContent(),
                    output.getContentSize(),
                    operatorHandler,
                    executionCtx.workerThreadId);

                record.write(outputFieldNames.at(0), output);
                rowIndex += outputFieldNames.size();
            }
        }
        else
        {
            if (!this->isVarSizedInput)
            {
                for (nautilus::static_val<size_t> i = 0; i < outputFieldNames.size(); ++i)
                {
                    VarVal result = VarVal(nautilus::invoke(
                        IREEBatchCacheInference::getValueFromModelProxy<T>,
                        rowIndex,
                        operatorHandler,
                        executionCtx.workerThreadId));

                    record.write(outputFieldNames.at(i), result);
                    ++rowIndex;
                }
            }
            else
            {
                auto output = executionCtx.pipelineMemoryProvider.arena.allocateVariableSizedData(this->outputSize);

                nautilus::invoke(
                    IREEBatchCacheInference::copyVarSizedFromModelProxy,
                    rowIndex,
                    output.getContent(),
                    output.getContentSize(),
                    operatorHandler,
                    executionCtx.workerThreadId);

                record.write(outputFieldNames.at(0), output);
                rowIndex += outputFieldNames.size();
            }
        }
        executeChild(executionCtx, record);
    }
}

void IREEBatchCacheInferenceOperator::open(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const
{
    /// As this operator functions as a scan, we have to set the execution context for this pipeline
    executionCtx.watermarkTs = recordBuffer.getWatermarkTs();
    executionCtx.sequenceNumber = recordBuffer.getSequenceNumber();
    executionCtx.chunkNumber = recordBuffer.getChunkNumber();
    executionCtx.lastChunk = recordBuffer.isLastChunk();
    executionCtx.originId = recordBuffer.getOriginId();
    openChild(executionCtx, recordBuffer);

    const auto emittedBatch = static_cast<nautilus::val<EmittedBatch*>>(recordBuffer.getMemArea());
    const auto operatorHandlerRef = executionCtx.getGlobalOperatorHandler(operatorHandlerId);

    const auto batchRef = nautilus::invoke(
        +[](OperatorHandler* ptrOpHandler, const EmittedBatch* currentBatch)
        {
            PRECONDITION(ptrOpHandler != nullptr, "opHandler context should not be null!");
            const auto* opHandler = dynamic_cast<IREEBatchInferenceOperatorHandler*>(ptrOpHandler);
            std::shared_ptr<Batch> batch = opHandler->getBatch(currentBatch->batchId);
            return batch.get();
        }, operatorHandlerRef, emittedBatch);

    const auto batchPagedVectorMemRef = nautilus::invoke(
        +[](const Batch* batch)
        {
            PRECONDITION(batch != nullptr, "batch context should not be null!");
            return batch->getPagedVectorRef();
        }, batchRef);
    const PagedVectorRef batchPagedVectorRef(batchPagedVectorMemRef, tupleBufferRef);

    const auto startOfEntries = nautilus::invoke(
        +[](const IREEBatchInferenceOperatorHandler* opHandler, const WorkerThreadId workerThreadId)
        {
            return opHandler->getStartOfPredictionCacheEntries(
                IREEBatchInferenceOperatorHandler::StartPredictionCacheEntriesIREEInference{workerThreadId});
        }, operatorHandlerRef, executionCtx.workerThreadId);

    const auto inputTupleSize = nautilus::invoke(
        +[](OperatorHandler* inferModelHandler, WorkerThreadId thread)
        {
            auto handler = dynamic_cast<IREEBatchInferenceOperatorHandler*>(inferModelHandler);
            auto adapter = handler->getIREEAdapter(thread);
            return adapter->inputSize / handler->getBatchSize();
        }, operatorHandlerRef, executionCtx.workerThreadId);

    const auto replacementIndex = nautilus::invoke(
        +[](const IREEBatchInferenceOperatorHandler* opHandler, const WorkerThreadId workerThreadId)
        {
            return opHandler->getReplacementPos(
                IREEBatchInferenceOperatorHandler::StartPredictionCacheEntriesIREEInference{workerThreadId});
        }, operatorHandlerRef, executionCtx.workerThreadId);

    const auto lookupIndex = nautilus::invoke(
        +[](OperatorHandler* handler, const WorkerThreadId workerThreadId)
        {
            auto* predictionCacheHandler = dynamic_cast<PredictionCacheOperatorHandler*>(handler);
            if (predictionCacheHandler == nullptr)
            {
                return static_cast<ChainedHashMap*>(nullptr);
            }
            return predictionCacheHandler->getPredictionCacheLookupHashMapPtr(
                PredictionCacheOperatorHandler::StartPredictionCacheEntriesArgs{workerThreadId});
        }, operatorHandlerRef, executionCtx.workerThreadId);

    auto predictionCache = NES::Util::createPredictionCache(
        predictionCacheOptions, operatorHandlerRef, startOfEntries, inputTupleSize);
    predictionCache->configureLookupIndex(lookupIndex, executionCtx.pipelineMemoryProvider.bufferProvider);
    predictionCache->setReplacementPos(replacementIndex);
    executionCtx.setLocalOperatorState(id, std::move(predictionCache));

    const auto hashMapPtr = nautilus::invoke(
        +[](OperatorHandler* handler, WorkerThreadId threadId)
        {
            return dynamic_cast<IREEBatchInferenceOperatorHandler*>(handler)->getHashMapPtr(threadId);
        }, operatorHandlerRef, executionCtx.workerThreadId);

    ChainedHashMapRef hashMap{
        hashMapPtr,
        hashMapOptions.fieldKeys,
        hashMapOptions.fieldValues,
        hashMapOptions.entriesPerPage,
        hashMapOptions.entrySize};

    switch (inputDtype.type)
    {
        case DataType::Type::UINT8: performInference<uint8_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::UINT16: performInference<uint16_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::UINT32: performInference<uint32_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::UINT64: performInference<uint64_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::INT8: performInference<int8_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::INT16: performInference<int16_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::INT32: performInference<int32_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::INT64: performInference<int64_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::FLOAT32: performInference<float>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::FLOAT64: performInference<double>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;

        case DataType::Type::BOOLEAN:
        case DataType::Type::CHAR:
        case DataType::Type::UNDEFINED:
        case DataType::Type::VARSIZED:
        case DataType::Type::VARSIZED_POINTER_REP:
            throw UnknownDataType("Physical Type: type {} is currently not implemented", magic_enum::enum_name(inputDtype.type));
    }

    switch (outputDtype.type)
    {
        case DataType::Type::UINT8: writeOutputRecord<uint8_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::UINT16: writeOutputRecord<uint16_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::UINT32: writeOutputRecord<uint32_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::UINT64: writeOutputRecord<uint64_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::INT8: writeOutputRecord<int8_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::INT16: writeOutputRecord<int16_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::INT32: writeOutputRecord<int32_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::INT64: writeOutputRecord<int64_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::FLOAT32: writeOutputRecord<float>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;
        case DataType::Type::FLOAT64: writeOutputRecord<double>(batchPagedVectorRef, *tupleBufferRef, executionCtx, hashMapPtr, hashMap); break;

        case DataType::Type::BOOLEAN:
        case DataType::Type::CHAR:
        case DataType::Type::UNDEFINED:
        case DataType::Type::VARSIZED:
        case DataType::Type::VARSIZED_POINTER_REP:
            throw UnknownDataType("Physical Type: type {} is currently not implemented", magic_enum::enum_name(outputDtype.type));
    }

    nautilus::invoke(
        +[](OperatorHandler* ptrOpHandler, WorkerThreadId thread, const EmittedBatch* currentBatch)
        {
            PRECONDITION(ptrOpHandler != nullptr, "opHandler context should not be null!");
            const auto* opHandler = dynamic_cast<IREEBatchInferenceOperatorHandler*>(ptrOpHandler);
            auto adapter = opHandler->getIREEAdapter(thread);
            adapter->batchCachingHelper.clearCacheMap();

            std::shared_ptr<Batch> batch = opHandler->getBatch(currentBatch->batchId);
            batch->setState(BatchState::MARKED_AS_PROCESSED);
        }, operatorHandlerRef, executionCtx.workerThreadId, emittedBatch);
}

void IREEBatchCacheInferenceOperator::setup(ExecutionContext& executionCtx, CompilationContext&) const
{
    const auto globalOperatorHandler = executionCtx.getGlobalOperatorHandler(operatorHandlerId);
    nautilus::invoke(
        +[](OperatorHandler* opHandler, PipelineExecutionContext* pec,
            uint64_t keySize, uint64_t valueSize, uint64_t numberOfBuckets, uint64_t pageSize, size_t tupleSize)
        {
            auto handler = dynamic_cast<IREEBatchInferenceOperatorHandler*>(opHandler);
            handler->start(*pec, 0);
            handler->allocateHashMaps(keySize, valueSize, numberOfBuckets, pageSize);
            handler->allocateBuffers(tupleSize);
        }, globalOperatorHandler, executionCtx.pipelineContext,
        nautilus::val<uint64_t>(hashMapOptions.keySize),
        nautilus::val<uint64_t>(hashMapOptions.valueSize),
        nautilus::val<uint64_t>(hashMapOptions.pageSize),
        nautilus::val<uint64_t>(hashMapOptions.numberOfBuckets),
        nautilus::val<size_t>(this->inputSize));

    const uint64_t entrySize = NES::Util::getPredictionCacheEntrySize(predictionCacheOptions.predictionCacheType);
    if (entrySize == 0)
    {
        return;
    }
    const nautilus::val<uint64_t> numberOfEntries = predictionCacheOptions.numberOfEntries;

    nautilus::invoke(
        +[](IREEBatchInferenceOperatorHandler* opHandler,
            AbstractBufferProvider* bufferProvider,
            const uint64_t sizeOfEntryVal,
            const uint64_t numberOfEntriesVal)
        { opHandler->allocatePredictionCacheEntries(sizeOfEntryVal, numberOfEntriesVal, bufferProvider); },
        globalOperatorHandler,
        executionCtx.pipelineMemoryProvider.bufferProvider,
        nautilus::val<uint64_t>(entrySize),
        numberOfEntries);
}

void IREEBatchCacheInferenceOperator::close(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const
{
    auto* predictionCache = dynamic_cast<PredictionCache*>(executionCtx.getLocalState(id));
    auto operatorHandlerRef = predictionCache->getOperatorHandler();

    nautilus::invoke(
        +[](IREEBatchInferenceOperatorHandler* opHandler, uint64_t pos, const WorkerThreadId workerThreadId)
        {
            opHandler->setReplacementPos(IREEBatchInferenceOperatorHandler::StartPredictionCacheEntriesIREEInference{workerThreadId}, pos);
        }, operatorHandlerRef, predictionCache->getReplacementPos(), executionCtx.workerThreadId);

    nautilus::invoke(IREEBatchCacheInference::garbageCollectBatchesProxy, operatorHandlerRef, executionCtx.workerThreadId);
    PhysicalOperatorConcept::close(executionCtx, recordBuffer);
}

Record
IREEBatchCacheInferenceOperator::createRecord(const Record& featureRecord, const std::vector<Record::RecordFieldIdentifier>& projections) const
{
    Record record;
    for (const auto& fieldName : nautilus::static_iterable(projections))
    {
        record.write(fieldName, featureRecord.read(fieldName));
    }
    return record;
}

void IREEBatchCacheInferenceOperator::terminate(ExecutionContext& executionCtx) const
{
    nautilus::invoke(
        +[](OperatorHandler* opHandler, PipelineExecutionContext* pec) { opHandler->stop(QueryTerminationType::Graceful, *pec); },
        executionCtx.getGlobalOperatorHandler(operatorHandlerId),
        executionCtx.pipelineContext);
}

}
