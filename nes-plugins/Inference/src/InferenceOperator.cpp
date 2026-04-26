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
#include <InferenceAdapter.hpp>
#include <InferenceOperator.hpp>
#include <InferenceOperatorHandler.hpp>
#include <Nautilus/Interface/Record.hpp>
#include <QueryExecutionConfiguration.hpp>
#include <nautilus/function.hpp>
#include <ranges>

namespace NES::QueryCompilation::PhysicalOperators
{
class PhysicalInferModelOperator;
}


namespace NES::Inference
{
inline InferenceOperatorHandler* getHandler(OperatorHandler* inferModelHandler)
{
    return dynamic_cast<InferenceOperatorHandler*>(inferModelHandler);
}

inline InferenceAdapter* getAdapter(OperatorHandler* inferModelHandler, WorkerThreadId thread)
{
    return getHandler(inferModelHandler)->getAdapter(thread).get();
}

template <class T>
void addValueToModelProxy(int index, T value, OperatorHandler* inferModelHandler, WorkerThreadId thread)
{
    auto* adapter = getAdapter(inferModelHandler, thread);
    adapter->addModelInput<T>(index, value);
}

template <class T>
float getValueFromModelProxy(int index, OperatorHandler* inferModelHandler, WorkerThreadId thread)
{
    auto* adapter = getAdapter(inferModelHandler, thread);
    return adapter->getResultAt<T>(index);
}

void copyVarSizedToModelProxy(std::byte* content, uint32_t size, OperatorHandler* inferModelHandler, WorkerThreadId thread)
{
    auto* adapter = getAdapter(inferModelHandler, thread);
    adapter->addModelInput(std::span{content, size});
}

void copyVarSizedFromModelProxy(std::byte* content, uint32_t size, OperatorHandler* inferModelHandler, WorkerThreadId thread)
{
    auto* adapter = getAdapter(inferModelHandler, thread);
    adapter->copyResultTo(std::span{content, size});
}

template <class T>
void applyModelProxy(OperatorHandler* inferModelHandler, WorkerThreadId thread)
{
    auto* adapter = getAdapter(inferModelHandler, thread);
    adapter->infer<T>();
}

template <typename T>
nautilus::val<T> min(const nautilus::val<T>& lhs, const nautilus::val<T>& rhs)
{
    return lhs < rhs ? lhs : rhs;
}
}

namespace NES
{

InferenceOperator::InferenceOperator(
    const OperatorHandlerId inferModelHandlerIndex,
    std::vector<PhysicalFunction> inputs,
    std::vector<std::string> outputFieldNames,
    DataType inputDtype,
    DataType outputDtype)
    : inferModelHandlerIndex(inferModelHandlerIndex)
    , inputs(std::move(inputs))
    , outputFieldNames(std::move(outputFieldNames))
    , inputDtype(inputDtype)
    , outputDtype(outputDtype)
{
}

template <typename T>
void InferenceOperator::performInference(ExecutionContext& executionCtx, NES::Record& record) const
{
    const auto operatorHandler = executionCtx.getGlobalOperatorHandler(inferModelHandlerIndex);

    if (!this->isVarSizedInput)
    {
        for (nautilus::static_val<size_t> i = 0; i < inputs.size(); ++i)
        {
            invoke(
                Inference::addValueToModelProxy<T>,
                nautilus::val<int>(i),
                inputs.at(i).execute(record, executionCtx.pipelineMemoryProvider.arena).cast<nautilus::val<T>>(),
                operatorHandler,
                executionCtx.workerThreadId);
        }
    }
    else
    {
        const VarVal inputValue = inputs.at(0).execute(record, executionCtx.pipelineMemoryProvider.arena);
        const auto varSizedValue = inputValue.cast<VariableSizedData>();
        invoke(
            Inference::copyVarSizedToModelProxy,
            varSizedValue.getContent(),
            Inference::min(varSizedValue.getContentSize(), nautilus::val<uint32_t>(static_cast<uint32_t>(this->inputSize))),
            operatorHandler,
            executionCtx.workerThreadId);
    }

    invoke(Inference::applyModelProxy<T>, operatorHandler, executionCtx.workerThreadId);
}

template <typename T>
void InferenceOperator::writeOutputRecord(ExecutionContext& executionCtx, NES::Record& record) const
{
    const auto operatorHandler = executionCtx.getGlobalOperatorHandler(inferModelHandlerIndex);

    if (!this->isVarSizedOutput)
    {
        for (nautilus::static_val<size_t> i = 0; i < outputFieldNames.size(); ++i)
        {
            VarVal result = VarVal(invoke(
                Inference::getValueFromModelProxy<T>,
                nautilus::val<int>(i),
                operatorHandler,
                executionCtx.workerThreadId));
            record.write(outputFieldNames.at(i), result);
        }
    }
    else
    {
        auto output = executionCtx.pipelineMemoryProvider.arena.allocateVariableSizedData(this->outputSize);
        invoke(
            Inference::copyVarSizedFromModelProxy,
            output.getContent(),
            output.getContentSize(),
            operatorHandler,
            executionCtx.workerThreadId);
        record.write(outputFieldNames.at(0), output);
    }

    child->execute(executionCtx, record);
}

void InferenceOperator::execute(ExecutionContext& executionCtx, NES::Record& record) const
{
    switch (inputDtype.type)
    {
        case DataType::Type::UINT8: performInference<uint8_t>(executionCtx, record); break;
        case DataType::Type::UINT16: performInference<uint16_t>(executionCtx, record); break;
        case DataType::Type::UINT32: performInference<uint32_t>(executionCtx, record); break;
        case DataType::Type::UINT64: performInference<uint64_t>(executionCtx, record); break;
        case DataType::Type::INT8: performInference<int8_t>(executionCtx, record); break;
        case DataType::Type::INT16: performInference<int16_t>(executionCtx, record); break;
        case DataType::Type::INT32: performInference<int32_t>(executionCtx, record); break;
        case DataType::Type::INT64: performInference<int64_t>(executionCtx, record); break;
        case DataType::Type::FLOAT32: performInference<float>(executionCtx, record); break;
        case DataType::Type::FLOAT64: performInference<double>(executionCtx, record); break;

        case DataType::Type::BOOLEAN:
        case DataType::Type::CHAR:
        case DataType::Type::UNDEFINED:
        case DataType::Type::VARSIZED:
        case DataType::Type::VARSIZED_POINTER_REP:
            throw UnknownDataType("Physical Type: type {} is currently not implemented", magic_enum::enum_name(inputDtype.type));
    }

    switch (outputDtype.type)
    {
        case DataType::Type::UINT8: writeOutputRecord<uint8_t>(executionCtx, record); break;
        case DataType::Type::UINT16: writeOutputRecord<uint16_t>(executionCtx, record); break;
        case DataType::Type::UINT32: writeOutputRecord<uint32_t>(executionCtx, record); break;
        case DataType::Type::UINT64: writeOutputRecord<uint64_t>(executionCtx, record); break;
        case DataType::Type::INT8: writeOutputRecord<int8_t>(executionCtx, record); break;
        case DataType::Type::INT16: writeOutputRecord<int16_t>(executionCtx, record); break;
        case DataType::Type::INT32: writeOutputRecord<int32_t>(executionCtx, record); break;
        case DataType::Type::INT64: writeOutputRecord<int64_t>(executionCtx, record); break;
        case DataType::Type::FLOAT32: writeOutputRecord<float>(executionCtx, record); break;
        case DataType::Type::FLOAT64: writeOutputRecord<double>(executionCtx, record); break;

        case DataType::Type::BOOLEAN:
        case DataType::Type::CHAR:
        case DataType::Type::UNDEFINED:
        case DataType::Type::VARSIZED:
        case DataType::Type::VARSIZED_POINTER_REP:
            throw UnknownDataType("Physical Type: type {} is currently not implemented", magic_enum::enum_name(outputDtype.type));
    }
}

void InferenceOperator::setup(ExecutionContext& executionCtx, CompilationContext&) const
{
    invoke(
        +[](OperatorHandler* opHandler, PipelineExecutionContext* pec) { opHandler->start(*pec, 0); },
        executionCtx.getGlobalOperatorHandler(inferModelHandlerIndex),
        executionCtx.pipelineContext);
}

void InferenceOperator::terminate(ExecutionContext& executionCtx) const
{
    invoke(
        +[](OperatorHandler* opHandler, PipelineExecutionContext* pec) { opHandler->stop(QueryTerminationType::Graceful, *pec); },
        executionCtx.getGlobalOperatorHandler(inferModelHandlerIndex),
        executionCtx.pipelineContext);
}
}
