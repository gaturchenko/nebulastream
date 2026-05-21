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

#include <Aggregation/Function/ArrayAggregationPhysicalFunction.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <Aggregation/Function/AggregationPhysicalFunction.hpp>
#include <DataTypes/DataType.hpp>
#include <Functions/PhysicalFunction.hpp>
#include <Nautilus/DataTypes/VarVal.hpp>
#include <Nautilus/DataTypes/VariableSizedData.hpp>
#include <Nautilus/Interface/BufferRef/TupleBufferRef.hpp>
#include <Nautilus/Interface/PagedVector/PagedVector.hpp>
#include <Nautilus/Interface/PagedVector/PagedVectorRef.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <nautilus/function.hpp>
#include <nautilus/std/cstring.h>
#include <AggregationPhysicalFunctionRegistry.hpp>
#include <ErrorHandling.hpp>
#include <ExecutionContext.hpp>
#include <val.hpp>
#include <val_arith.hpp>
#include <val_bool.hpp>
#include <val_concepts.hpp>
#include <val_ptr.hpp>

namespace NES
{

ArrayAggregationPhysicalFunction::ArrayAggregationPhysicalFunction(
    DataType inputType,
    DataType resultType,
    PhysicalFunction inputFunction,
    Record::RecordFieldIdentifier resultFieldIdentifier,
    std::shared_ptr<TupleBufferRef> bufferRefPagedVector)
    : AggregationPhysicalFunction(std::move(inputType), std::move(resultType), std::move(inputFunction), std::move(resultFieldIdentifier))
    , bufferRefPagedVector(std::move(bufferRefPagedVector))
{
}

void ArrayAggregationPhysicalFunction::lift(
    const nautilus::val<AggregationState*>& aggregationState, PipelineMemoryProvider& pipelineMemoryProvider, const Record& record)
{
    const auto memArea = static_cast<nautilus::val<int8_t*>>(aggregationState);
    const PagedVectorRef pagedVectorRef(memArea, bufferRefPagedVector);
    pagedVectorRef.writeRecord(record, pipelineMemoryProvider.bufferProvider);
}

void ArrayAggregationPhysicalFunction::combine(
    const nautilus::val<AggregationState*> aggregationState1,
    const nautilus::val<AggregationState*> aggregationState2,
    PipelineMemoryProvider&)
{
    const auto memArea1 = static_cast<nautilus::val<PagedVector*>>(aggregationState1);
    const auto memArea2 = static_cast<nautilus::val<PagedVector*>>(aggregationState2);
    nautilus::invoke(+[](PagedVector* vector1, const PagedVector* vector2) -> void { vector1->copyFrom(*vector2); }, memArea1, memArea2);
}

Record ArrayAggregationPhysicalFunction::lower(
    const nautilus::val<AggregationState*> aggregationState, PipelineMemoryProvider& pipelineMemoryProvider)
{
    const auto pagedVectorPtr = static_cast<nautilus::val<PagedVector*>>(aggregationState);
    const PagedVectorRef pagedVectorRef(pagedVectorPtr, bufferRefPagedVector);
    const auto allFieldNames = bufferRefPagedVector->getAllFieldNames();
    const auto numberOfEntries = nautilus::invoke(
        +[](const PagedVector* pagedVector)
        {
            const auto numberOfEntriesVal = pagedVector->getTotalNumberOfEntries();
            INVARIANT(numberOfEntriesVal > 0, "The number of entries in the paged vector must be greater than 0");
            return numberOfEntriesVal;
        },
        pagedVectorPtr);

    const auto entrySize = inputType.getSizeInBytesWithoutNull();
    auto variableSized = pipelineMemoryProvider.arena.allocateVariableSizedData(numberOfEntries * entrySize);
    auto current = variableSized.getContent();
    nautilus::val<uint64_t> writtenBytes = 0;

    const auto endIt = pagedVectorRef.end(allFieldNames);
    for (auto candidateIt = pagedVectorRef.begin(allFieldNames); candidateIt != endIt; ++candidateIt)
    {
        const auto itemRecord = *candidateIt;
        const auto itemValue = inputFunction.execute(itemRecord, pipelineMemoryProvider.arena);
        if (!itemValue.isNull())
        {
            auto _ = itemValue.customVisit(
                [&]<typename T>(const T& type) -> VarVal
                {
                    if constexpr (std::is_same_v<T, VariableSizedData>)
                    {
                        throw std::runtime_error("VariableSizedData is not supported in ArrayAggregationPhysicalFunction");
                    }
                    else
                    {
                        *static_cast<nautilus::val<typename T::raw_type*>>(current) = type;
                        current += sizeof(typename T::raw_type);
                        writtenBytes += sizeof(typename T::raw_type);
                    }
                    return type;
                });
        }
    }

    Record resultRecord;
    resultRecord.write(resultFieldIdentifier, VariableSizedData(variableSized.getContent(), writtenBytes));
    return resultRecord;
}

void ArrayAggregationPhysicalFunction::reset(const nautilus::val<AggregationState*> aggregationState, PipelineMemoryProvider&)
{
    nautilus::invoke(
        +[](AggregationState* pagedVectorMemArea) -> void
        {
            auto* pagedVector = reinterpret_cast<PagedVector*>(pagedVectorMemArea);
            new (pagedVector) PagedVector();
        },
        aggregationState);
}

void ArrayAggregationPhysicalFunction::cleanup(nautilus::val<AggregationState*> aggregationState)
{
    nautilus::invoke(
        +[](AggregationState* pagedVectorMemArea) -> void
        {
            auto* pagedVector = reinterpret_cast<PagedVector*>(pagedVectorMemArea);
            pagedVector->~PagedVector();
        },
        aggregationState);
}

size_t ArrayAggregationPhysicalFunction::getSizeOfStateInBytes() const
{
    return sizeof(PagedVector);
}

AggregationPhysicalFunctionRegistryReturnType AggregationPhysicalFunctionGeneratedRegistrar::RegisterArray_AggAggregationPhysicalFunction(
    AggregationPhysicalFunctionRegistryArguments arguments)
{
    INVARIANT(arguments.bufferRefPagedVector.has_value(), "Memory provider paged vector not set");

    return std::make_shared<ArrayAggregationPhysicalFunction>(
        std::move(arguments.inputType),
        std::move(arguments.resultType),
        arguments.inputFunction,
        arguments.resultFieldIdentifier,
        arguments.bufferRefPagedVector.value());
}

}
