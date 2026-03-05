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
#include <IREEBatchInferenceOperator.hpp>
#include <IREEBatchInferenceOperatorHandler.hpp>
#include <Nautilus/Interface/HashMap/ChainedHashMap/ChainedHashMapRef.hpp>
#include <Nautilus/Interface/PagedVector/PagedVectorRef.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <QueryExecutionConfiguration.hpp>
#include <nautilus/function.hpp>

namespace NES::QueryCompilation::PhysicalOperators
{
class PhysicalInferModelOperator;
}

namespace NES::IREEBatchInference
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
void addValueToModelProxy(int index, T value, OperatorHandler* inferModelHandler, WorkerThreadId thread)
{
    auto* adapter = getAdapter(inferModelHandler, thread);
    adapter->addModelInput<T>(index, value);
}

template <class T>
int addUniqueValueToModelProxy(T value, OperatorHandler* inferModelHandler, WorkerThreadId thread)
{
    auto* adapter = getAdapter(inferModelHandler, thread);
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
    WorkerThreadId thread)
{
    auto* adapter = getAdapter(inferModelHandler, thread);
    adapter->addModelInputBatch(index, std::span{content, size}, tupleSize);
}

void copyUniqueVarSizedToModelProxy(
    int index,
    std::byte* content,
    uint32_t size,
    size_t tupleSize,
    OperatorHandler* inferModelHandler,
    WorkerThreadId thread)
{
    auto* adapter = getAdapter(inferModelHandler, thread);
    adapter->addModelInputBatchPartial(index, std::span{content, size}, tupleSize);
}

void copyVarSizedFromModelProxy(int index, std::byte* content, uint32_t size, OperatorHandler* inferModelHandler, WorkerThreadId thread)
{
    auto* adapter = getAdapter(inferModelHandler, thread);
    adapter->copyResultToBatch(index, std::span{content, size});
}

template <class T>
void applyModelProxy(OperatorHandler* inferModelHandler, WorkerThreadId thread)
{
    auto* adapter = getAdapter(inferModelHandler, thread);
    adapter->inferWithReduction<T>();
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

IREEBatchInferenceOperator::IREEBatchInferenceOperator(
    const OperatorHandlerId operatorHandlerId,
    std::vector<PhysicalFunction> inputs,
    std::vector<std::string> outputFieldNames,
    std::shared_ptr<TupleBufferRef> tupleBufferRef,
    DataType inputDtype,
    DataType outputDtype,
    HashMapOptions hashMapOptions,
    bool useBatchDeduplication)
    : WindowProbePhysicalOperator(operatorHandlerId)
    , useBatchDeduplication(useBatchDeduplication)
    , inputs(std::move(inputs))
    , outputFieldNames(std::move(outputFieldNames))
    , tupleBufferRef(std::move(tupleBufferRef))
    , inputDtype(inputDtype)
    , outputDtype(outputDtype)
    , hashMapOptions(std::move(hashMapOptions))
{
}

template <typename T>
void IREEBatchInferenceOperator::performInference(
    const PagedVectorRef& pagedVectorRef,
    TupleBufferRef& tupleBufferRef,
    ExecutionContext& executionCtx,
    nautilus::val<HashMap*> hashMapPtr,
    ChainedHashMapRef& hashMap) const
{
    const auto fields = tupleBufferRef.getMemoryLayout()->getSchema().getFieldNames();
    const auto operatorHandler = executionCtx.getGlobalOperatorHandler(operatorHandlerId);

    nautilus::val<int> rowIndex(0);
    for (auto it = pagedVectorRef.begin(fields); it != pagedVectorRef.end(fields); ++it)
    {
        auto record = createRecord(*it, fields);

        if (useBatchDeduplication)
        {
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

            /// we inserted a new entry, so we have a unique record and hence we write it to the input buffer
            if (entryRowIndex == rowIndex)
            {
                nautilus::val<int> outputRowIndex{0};
                if (!this->isVarSizedInput)
                {
                    nautilus::val<int> inputDataIndex{0};
                    for (nautilus::static_val<size_t> i = 0; i < inputs.size(); ++i)
                    {
                        auto index = nautilus::invoke(
                            IREEBatchInference::addUniqueValueToModelProxy<T>,
                            inputs.at(i).execute(record, executionCtx.pipelineMemoryProvider.arena).cast<nautilus::val<T>>(),
                            operatorHandler,
                            executionCtx.workerThreadId);
                        ++rowIndex;

                        if (i == nautilus::val<size_t>(0))
                        {
                            inputDataIndex = index;
                        }
                    }
                    outputRowIndex = inputDataIndex * this->outputSize / this->inputSize;
                }
                else
                {
                    const VarVal inputValue = inputs.at(0).execute(record, executionCtx.pipelineMemoryProvider.arena);
                    const auto varSizedValue = inputValue.cast<VariableSizedData>();

                    nautilus::invoke(
                        IREEBatchInference::copyUniqueVarSizedToModelProxy,
                        rowIndex,
                        varSizedValue.getContent(),
                        IREEBatchInference::min(varSizedValue.getContentSize(), nautilus::val<uint32_t>(static_cast<uint32_t>(this->inputSize))),
                        nautilus::val<size_t>(inputSize),
                        operatorHandler,
                        executionCtx.workerThreadId);

                    outputRowIndex = rowIndex;
                    rowIndex += inputs.size();
                }

                /// compute the output buffer index, given the size of the currently used input buffer, and write it to the value record
                Record valueRecord = entryRef.getValue();
                valueRecord.write("rowOutputIndex", VarVal(outputRowIndex));
                entryRef.copyValuesToEntry(valueRecord, executionCtx.pipelineMemoryProvider.bufferProvider);
            }
            else
            {
                rowIndex += inputs.size();
            }
        }
        /// write the record value to the adapter's `inputData` byte buffer
        else
        {
            if (!this->isVarSizedInput)
            {
                for (nautilus::static_val<size_t> i = 0; i < inputs.size(); ++i)
                {
                    nautilus::invoke(
                        IREEBatchInference::addValueToModelProxy<T>,
                        rowIndex,
                        inputs.at(i).execute(record, executionCtx.pipelineMemoryProvider.arena).cast<nautilus::val<T>>(),
                        operatorHandler,
                        executionCtx.workerThreadId);
                    ++rowIndex;
                }
            }
            else
            {
                const VarVal inputValue = inputs.at(0).execute(record, executionCtx.pipelineMemoryProvider.arena);
                const auto varSizedValue = inputValue.cast<VariableSizedData>();
                nautilus::invoke(
                    IREEBatchInference::copyVarSizedToModelProxy,
                    rowIndex,
                    varSizedValue.getContent(),
                    IREEBatchInference::min(varSizedValue.getContentSize(), nautilus::val<uint32_t>(static_cast<uint32_t>(this->inputSize))),
                    nautilus::val<size_t>(inputSize),
                    operatorHandler,
                    executionCtx.workerThreadId);
                rowIndex += inputs.size();
            }
        }
    }

    nautilus::invoke(IREEBatchInference::applyModelProxy<T>, operatorHandler, executionCtx.workerThreadId);
}

template <typename T>
void IREEBatchInferenceOperator::writeOutputRecord(
    const PagedVectorRef& pagedVectorRef,
    TupleBufferRef& tupleBufferRef,
    ExecutionContext& executionCtx,
    nautilus::val<HashMap*> hashMapPtr,
    ChainedHashMapRef& hashMap) const
{
    const auto fields = tupleBufferRef.getMemoryLayout()->getSchema().getFieldNames();
    const auto operatorHandler = executionCtx.getGlobalOperatorHandler(operatorHandlerId);

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
                        IREEBatchInference::getValueFromModelProxy<T>,
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
                    IREEBatchInference::copyVarSizedFromModelProxy,
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
            if (!this->isVarSizedOutput)
            {
                for (nautilus::static_val<size_t> i = 0; i < outputFieldNames.size(); ++i)
                {
                    VarVal result = VarVal(nautilus::invoke(
                        IREEBatchInference::getValueFromModelProxy<T>,
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
                    IREEBatchInference::copyVarSizedFromModelProxy,
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

void IREEBatchInferenceOperator::open(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const
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
        +[](OperatorHandler* ptrOpHandler, const EmittedBatch* currentBatch)
        {
            PRECONDITION(ptrOpHandler != nullptr, "opHandler context should not be null!");
            const auto* opHandler = dynamic_cast<IREEBatchInferenceOperatorHandler*>(ptrOpHandler);
            std::shared_ptr<Batch> batch = opHandler->getBatch(currentBatch->batchId);
            batch->setState(BatchState::MARKED_AS_PROCESSED);
        }, operatorHandlerRef, emittedBatch);
}

void IREEBatchInferenceOperator::close(ExecutionContext& executionCtx, RecordBuffer& recordBuffer) const
{
    const auto operatorHandlerMemRef = executionCtx.getGlobalOperatorHandler(operatorHandlerId);
    nautilus::invoke(IREEBatchInference::garbageCollectBatchesProxy, operatorHandlerMemRef, executionCtx.workerThreadId);
    PhysicalOperatorConcept::close(executionCtx, recordBuffer);
}

void IREEBatchInferenceOperator::setup(ExecutionContext& executionCtx, CompilationContext&) const
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
}

Record
IREEBatchInferenceOperator::createRecord(const Record& featureRecord, const std::vector<Record::RecordFieldIdentifier>& projections) const
{
    Record record;
    for (const auto& fieldName : nautilus::static_iterable(projections))
    {
        record.write(fieldName, featureRecord.read(fieldName));
    }
    return record;
}

}
