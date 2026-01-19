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
#include <Nautilus/Interface/PagedVector/PagedVectorRef.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <PredictionCache/PredictionCache2Q.hpp>
#include <PredictionCache/PredictionCacheFIFO.hpp>
#include <PredictionCache/PredictionCacheLFU.hpp>
#include <PredictionCache/PredictionCacheLRU.hpp>
#include <PredictionCache/PredictionCacheSecondChance.hpp>
#include <PredictionCache/PredictionCacheUtil.hpp>
#include <QueryExecutionConfiguration.hpp>
#include <nautilus/function.hpp>

namespace NES::QueryCompilation::PhysicalOperators
{
class PhysicalInferModelOperator;
}

namespace NES::IREEBatchCacheInference
{
template <class T>
void addValueToModelProxy(int index, int indexOutput, T value, void* inferModelHandler, WorkerThreadId thread, uint64_t keyIdx)
{
    auto handler = static_cast<IREEBatchInferenceOperatorHandler*>(inferModelHandler);
    auto adapter = handler->getIREEAdapter(thread);

    /// we need to write the row index of the tuple so as to know where to insert it in the output byte array after the model call
    adapter->updateCacheMapIndices(keyIdx, indexOutput);
    adapter->appendMissIdx(indexOutput);

    adapter->addModelInput<T>(index, value);
}

template <class T>
T getValueFromModelProxy(int index, void* inferModelHandler, WorkerThreadId thread)
{
    auto handler = static_cast<IREEBatchInferenceOperatorHandler*>(inferModelHandler);
    auto adapter = handler->getIREEAdapter(thread);
    return adapter->getResultAt<T>(index);
}

void copyVarSizedToModelProxy(int indexOutput, std::byte* content, uint32_t size, void* inferModelHandler, WorkerThreadId thread, uint64_t keyIdx)
{
    auto handler = static_cast<IREEBatchInferenceOperatorHandler*>(inferModelHandler);
    auto adapter = handler->getIREEAdapter(thread);

    /// we need to write the row index of the tuple so as to know where to insert it in the output byte array after the model call
    adapter->updateCacheMapIndices(keyIdx, indexOutput);
    adapter->appendMissIdx(indexOutput);

    adapter->addModelInput(std::span{content, size});
}

void copyVarSizedFromModelProxy(std::byte* content, uint32_t size, void* inferModelHandler, WorkerThreadId thread)
{
    auto handler = static_cast<IREEBatchInferenceOperatorHandler*>(inferModelHandler);
    auto adapter = handler->getIREEAdapter(thread);
    adapter->copyResultTo(std::span{content, size});
}

template <class T>
size_t applyModelProxy(void* inferModelHandler, WorkerThreadId thread, size_t outputSize)
{
    auto handler = static_cast<IREEBatchInferenceOperatorHandler*>(inferModelHandler);
    auto adapter = handler->getIREEAdapter(thread);
    /// call the model only if any misses were recorded
    if (adapter->getMissIndicesSize() > 0)
    {
        return adapter->inferCombine<T>(outputSize);
    }
    return 0;
}

template <typename T>
nautilus::val<T> min(const nautilus::val<T>& lhs, const nautilus::val<T>& rhs)
{
    return lhs < rhs ? lhs : rhs;
}

void garbageCollectBatchesProxy(void* inferModelHandler)
{
    auto handler = static_cast<IREEBatchInferenceOperatorHandler*>(inferModelHandler);
    handler->garbageCollectBatches();
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
    DataType outputDtype)
    : WindowProbePhysicalOperator(operatorHandlerId)
    , inputs(std::move(inputs))
    , outputFieldNames(std::move(outputFieldNames))
    , tupleBufferRef(std::move(tupleBufferRef))
    , predictionCacheOptions(predictionCacheOptions)
    , inputDtype(inputDtype)
    , outputDtype(outputDtype)
{
}

template <typename T>
void IREEBatchCacheInferenceOperator::performInference(
    const PagedVectorRef& pagedVectorRef,
    TupleBufferRef& tupleBufferRef,
    ExecutionContext& executionCtx) const
{
    const auto fields = tupleBufferRef.getMemoryLayout()->getSchema().getFieldNames();
    auto* predictionCache = dynamic_cast<PredictionCache*>(executionCtx.getLocalState(id));
    const auto operatorHandler = predictionCache->getOperatorHandler();

    nautilus::val<int> rowIdx(0);

    auto cacheProbeTuple = nautilus::invoke(+[](OperatorHandler* inferModelHandler, WorkerThreadId thread, size_t size)
        {
            auto handler = static_cast<IREEBatchInferenceOperatorHandler*>(inferModelHandler);
            auto adapter = handler->getIREEAdapter(thread);
            adapter->cacheProbeTuple = std::make_unique<std::byte[]>(size);
            return adapter->cacheProbeTuple.get();
        }, operatorHandler, executionCtx.workerThreadId, nautilus::val<size_t>(this->inputSize));

    /// iterate over records in the paged vector, i.e., over tuples in a single batch
    for (auto it = pagedVectorRef.begin(fields); it != pagedVectorRef.end(fields); ++it)
    {
        auto record = createRecord(*it, fields);
        auto rowIdxOutput = rowIdx * this->outputSize / this->inputSize;

        /// allocate and fill a byte array for a record for probing into the cache
        if (!this->isVarSizedInput)
        {
            for (nautilus::static_val<size_t> i = 0; i < inputs.size(); ++i)
            {
                nautilus::invoke(
                    +[](std::byte* array, size_t idx, T value)
                    {
                        std::bit_cast<T*>(array)[idx] = value;
                    }, cacheProbeTuple, nautilus::val<int>(i), inputs.at(i).execute(
                        record, executionCtx.pipelineMemoryProvider.arena).cast<nautilus::val<T>>());
            }
        }
        else
        {
            VarVal value = inputs.at(0).execute(record, executionCtx.pipelineMemoryProvider.arena);
            auto varSizedValue = value.cast<VariableSizedData>();
            nautilus::invoke(
                +[](std::byte* array, std::byte* content, uint32_t size)
                {
                    auto span = std::span{content, size};
                    std::ranges::copy_n(array, size, span.data());
                }, cacheProbeTuple, varSizedValue.getContent(),
                    IREEBatchCacheInference::min(varSizedValue.getContentSize(), nautilus::val<uint32_t>(this->inputSize)));
        }

        /// if the probe is successful, return the index of the key, otherwise return PredictionCache::NOT_FOUND, i.e., UINT64_MAX
        const auto keyIdx = predictionCache->updateKeys(
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
                    }, predictionCacheEntryToReplace, cacheProbeTuple, nautilus::val<int>(this->inputSize));
            });

        /// the key might be in the cache already, since the replacement function above may have been invoked
        /// however, the corresponding value may not yet exist, e.g., if we are processing the very first batch
        auto prediction = predictionCache->getDataStructure(keyIdx);
        const auto isInCache = nautilus::invoke(
            +[](std::byte* prediction){ return prediction != nullptr; }, prediction);

        /// if the key does not exist or it does but the corresponding value does not,
        /// then we write the tuple to the byte array reserved for the inputs to the model
        if (keyIdx == PredictionCache::NOT_FOUND || !isInCache)
        {
            if (!isVarSizedInput)
            {
                for (nautilus::static_val<size_t> i = 0; i < inputs.size(); ++i)
                {
                    nautilus::invoke(
                        IREEBatchCacheInference::addValueToModelProxy<T>,
                        rowIdx,
                        rowIdxOutput,
                        inputs.at(i).execute(record, executionCtx.pipelineMemoryProvider.arena).cast<nautilus::val<T>>(),
                        operatorHandler,
                        executionCtx.workerThreadId,
                        predictionCache->getReplacementIndex());
                    ++rowIdx;
                }
            }
            else
            {
                VarVal value = inputs.at(0).execute(record, executionCtx.pipelineMemoryProvider.arena);
                auto varSizedValue = value.cast<VariableSizedData>();
                nautilus::invoke(
                    IREEBatchCacheInference::copyVarSizedToModelProxy,
                    rowIdxOutput,
                    varSizedValue.getContent(),
                    IREEBatchCacheInference::min(varSizedValue.getContentSize(), nautilus::val<uint32_t>(static_cast<uint32_t>(this->inputSize))),
                    operatorHandler,
                    executionCtx.workerThreadId,
                    predictionCache->getReplacementIndex());
                rowIdx += inputs.size();
            }
        }
        /// otherwise, we know the prediction for this tuple in the batch and immediately write it to the output byte array
        else
        {
            nautilus::invoke(
                +[](int idx, std::byte* prediction, OperatorHandler* inferModelHandler, WorkerThreadId thread, size_t size)
                {
                    auto handler = static_cast<IREEBatchInferenceOperatorHandler*>(inferModelHandler);
                    auto adapter = handler->getIREEAdapter(thread);

                    std::memcpy(adapter->outputData.get() + idx * sizeof(T), prediction, size);
                }, rowIdxOutput, prediction, operatorHandler, executionCtx.workerThreadId, nautilus::val<size_t>((this->outputSize)));
            rowIdx += inputs.size();
        }
    }

    /// call the model and update the values of the keys that don't have them yet (if applicable)
    const auto valuesToUpdate = nautilus::invoke(
        IREEBatchCacheInference::applyModelProxy<T>, operatorHandler, executionCtx.workerThreadId, nautilus::val<size_t>(this->outputSize));

    for (nautilus::val<size_t> i = 0; i < valuesToUpdate; ++i)
    {
        const auto cachePos = nautilus::invoke(
            +[](size_t i, OperatorHandler* inferModelHandler, WorkerThreadId thread)
            {
                auto handler = static_cast<IREEBatchInferenceOperatorHandler*>(inferModelHandler);
                auto adapter = handler->getIREEAdapter(thread);
                return adapter->getCacheMapKey(i);
            }, i, operatorHandler, executionCtx.workerThreadId);

        const auto outputPos = nautilus::invoke(
            +[](size_t i, OperatorHandler* inferModelHandler, WorkerThreadId thread)
            {
                auto handler = static_cast<IREEBatchInferenceOperatorHandler*>(inferModelHandler);
                auto adapter = handler->getIREEAdapter(thread);
                return adapter->getCacheMapValue(i);
            }, i, operatorHandler, executionCtx.workerThreadId);

        predictionCache->updateValues(
            cachePos,
            [&](
                const nautilus::val<PredictionCacheEntry*>& predictionCacheEntryToReplace, const nautilus::val<uint64_t>&)
            {
                return nautilus::invoke(
                    +[](PredictionCacheEntry* predictionCacheEntry, void* opHandlerPtr, WorkerThreadId thread, int idx, size_t size)
                    {
                        auto handler = static_cast<IREEBatchInferenceOperatorHandler*>(opHandlerPtr);
                        auto adapter = handler->getIREEAdapter(thread);

                        predictionCacheEntry->dataSize = size;
                        predictionCacheEntry->dataStructure = new std::byte[size];

                        std::memcpy(predictionCacheEntry->dataStructure, adapter->outputData.get() + idx * sizeof(T), size);
                    }, predictionCacheEntryToReplace, operatorHandler, executionCtx.workerThreadId, outputPos, nautilus::val<size_t>(this->outputSize));
            });
    }
}

template <typename T>
void IREEBatchCacheInferenceOperator::writeOutputRecord(
    const PagedVectorRef& pagedVectorRef,
    TupleBufferRef& tupleBufferRef,
    ExecutionContext& executionCtx) const
{
    const auto fields = tupleBufferRef.getMemoryLayout()->getSchema().getFieldNames();
    auto* predictionCache = dynamic_cast<PredictionCache*>(executionCtx.getLocalState(id));
    const auto operatorHandler = predictionCache->getOperatorHandler();

    nautilus::val<int> rowIdx(0);
    for (auto it = pagedVectorRef.begin(fields); it != pagedVectorRef.end(fields); ++it)
    {
        auto record = createRecord(*it, fields);

        if (!this->isVarSizedOutput)
        {
            for (nautilus::static_val<size_t> i = 0; i < outputFieldNames.size(); ++i)
            {
                const VarVal result = VarVal(nautilus::invoke(
                    IREEBatchCacheInference::getValueFromModelProxy<T>,
                    rowIdx,
                    operatorHandler,
                    executionCtx.workerThreadId));
                record.write(outputFieldNames.at(i), result);
                ++rowIdx;
            }
        }
        else
        {
            auto output = executionCtx.pipelineMemoryProvider.arena.allocateVariableSizedData(this->outputSize);

            nautilus::invoke(
                IREEBatchCacheInference::copyVarSizedFromModelProxy,
                output.getContent(),
                output.getContentSize(),
                operatorHandler,
                executionCtx.workerThreadId);

            record.write(outputFieldNames.at(0), output);
            rowIdx += outputFieldNames.size();
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
    const auto operatorHandlerMemRef = executionCtx.getGlobalOperatorHandler(operatorHandlerId);

    const auto batchMemRef = nautilus::invoke(
        +[](OperatorHandler* ptrOpHandler, const EmittedBatch* currentBatch)
        {
            PRECONDITION(ptrOpHandler != nullptr, "opHandler context should not be null!");
            const auto* opHandler = dynamic_cast<IREEBatchInferenceOperatorHandler*>(ptrOpHandler);
            std::shared_ptr<Batch> batch = opHandler->getBatch(currentBatch->batchId);
            return batch.get();
        }, operatorHandlerMemRef, emittedBatch);

    const auto batchPagedVectorMemRef = nautilus::invoke(
        +[](const Batch* batch)
        {
            PRECONDITION(batch != nullptr, "batch context should not be null!");
            return batch->getPagedVectorRef();
        }, batchMemRef);
    const PagedVectorRef batchPagedVectorRef(batchPagedVectorMemRef, tupleBufferRef);

    const auto startOfEntries = nautilus::invoke(
        +[](const IREEBatchInferenceOperatorHandler* opHandler, const WorkerThreadId workerThreadId)
        {
            return opHandler->getStartOfPredictionCacheEntries(
                IREEBatchInferenceOperatorHandler::StartPredictionCacheEntriesIREEInference{workerThreadId});
        }, operatorHandlerMemRef, executionCtx.workerThreadId);

    const auto inputSize = nautilus::invoke(
        +[](void* inferModelHandler, WorkerThreadId thread)
        {
            auto handler = static_cast<IREEBatchInferenceOperatorHandler*>(inferModelHandler);
            auto adapter = handler->getIREEAdapter(thread);
            return adapter->inputSize / handler->getBatchSize();
        }, operatorHandlerMemRef, executionCtx.workerThreadId);

    auto predictionCache = NES::Util::createPredictionCache(
        predictionCacheOptions, operatorHandlerMemRef, startOfEntries, inputSize);
    executionCtx.setLocalOperatorState(id, std::move(predictionCache));

    switch (inputDtype.type)
    {
        case DataType::Type::UINT8: performInference<uint8_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::UINT16: performInference<uint16_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::UINT32: performInference<uint32_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::UINT64: performInference<uint64_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::INT8: performInference<int8_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::INT16: performInference<int16_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::INT32: performInference<int32_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::INT64: performInference<int64_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::FLOAT32: performInference<float>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::FLOAT64: performInference<double>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;

        case DataType::Type::BOOLEAN:
        case DataType::Type::CHAR:
        case DataType::Type::UNDEFINED:
        case DataType::Type::VARSIZED:
        case DataType::Type::VARSIZED_POINTER_REP:
            throw UnknownDataType("Physical Type: type {} is currently not implemented", magic_enum::enum_name(inputDtype.type));
    }

    switch (outputDtype.type)
    {
        case DataType::Type::UINT8: writeOutputRecord<uint8_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::UINT16: writeOutputRecord<uint16_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::UINT32: writeOutputRecord<uint32_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::UINT64: writeOutputRecord<uint64_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::INT8: writeOutputRecord<int8_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::INT16: writeOutputRecord<int16_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::INT32: writeOutputRecord<int32_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::INT64: writeOutputRecord<int64_t>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::FLOAT32: writeOutputRecord<float>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;
        case DataType::Type::FLOAT64: writeOutputRecord<double>(batchPagedVectorRef, *tupleBufferRef, executionCtx); break;

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
            adapter->clearCacheMap();

            std::shared_ptr<Batch> batch = opHandler->getBatch(currentBatch->batchId);
            batch->setState(BatchState::MARKED_AS_PROCESSED);
        }, operatorHandlerMemRef, executionCtx.workerThreadId, emittedBatch);
}

void IREEBatchCacheInferenceOperator::setup(ExecutionContext& executionCtx, CompilationContext&) const
{
    const auto globalOperatorHandler = executionCtx.getGlobalOperatorHandler(operatorHandlerId);
    nautilus::invoke(
        +[](OperatorHandler* opHandler, PipelineExecutionContext* pec) { opHandler->start(*pec, 0); },
        globalOperatorHandler,
        executionCtx.pipelineContext);

    nautilus::val<uint64_t> sizeOfEntry = 0;
    nautilus::val<uint64_t> numberOfEntries = predictionCacheOptions.numberOfEntries;
    switch (predictionCacheOptions.predictionCacheType)
    {
        case Configurations::PredictionCacheType::NONE:
            return;
        case Configurations::PredictionCacheType::FIFO:
            sizeOfEntry = sizeof(PredictionCacheEntryFIFO);
            break;
        case Configurations::PredictionCacheType::LFU:
            sizeOfEntry = sizeof(PredictionCacheEntryLFU);
            break;
        case Configurations::PredictionCacheType::LRU:
            sizeOfEntry = sizeof(PredictionCacheEntryLRU);
            break;
        case Configurations::PredictionCacheType::SECOND_CHANCE:
            sizeOfEntry = sizeof(PredictionCacheEntrySecondChance);
            break;
        case Configurations::PredictionCacheType::TWO_QUEUES:
            sizeOfEntry = sizeof(PredictionCacheEntry2Q);
            break;
    }

    nautilus::invoke(
        +[](IREEBatchInferenceOperatorHandler* opHandler,
            AbstractBufferProvider* bufferProvider,
            const uint64_t sizeOfEntryVal,
            const uint64_t numberOfEntriesVal)
        { opHandler->allocatePredictionCacheEntries(sizeOfEntryVal, numberOfEntriesVal, bufferProvider); },
        globalOperatorHandler,
        executionCtx.pipelineMemoryProvider.bufferProvider,
        sizeOfEntry,
        numberOfEntries);
}

void IREEBatchCacheInferenceOperator::close(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const
{
    const auto operatorHandlerMemRef = executionCtx.getGlobalOperatorHandler(operatorHandlerId);
    nautilus::invoke(IREEBatchCacheInference::garbageCollectBatchesProxy, operatorHandlerMemRef);
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

}
