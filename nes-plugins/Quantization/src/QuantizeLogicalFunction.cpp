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

#include "../include/QuantizeLogicalFunction.hpp"

#include <ranges>
#include <string>
#include <string_view>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <DataTypes/DataTypeProvider.hpp>
#include <DataTypes/Schema.hpp>
#include <Functions/LogicalFunction.hpp>
#include <Serialization/LogicalFunctionReflection.hpp>
#include <Util/PlanRenderer.hpp>
#include <Util/Reflection.hpp>
#include <fmt/format.h>
#include <ErrorHandling.hpp>
#include <LogicalFunctionRegistry.hpp>

namespace NES
{

namespace
{
void validateQuantizeChildren(const LogicalFunction& value, const LogicalFunction& mode)
{
    if (!value.getDataType().isType(DataType::Type::UNDEFINED))
    {
        if (!value.getDataType().isType(DataType::Type::FLOAT32))
        {
            throw DifferentFieldTypeExpected("QUANTIZE requires first argument to be FLOAT32, but got {}", value.getDataType());
        }
    }
    if (!mode.getDataType().isType(DataType::Type::UNDEFINED))
    {
        if (!mode.getDataType().isType(DataType::Type::VARSIZED))
        {
            throw DifferentFieldTypeExpected("QUANTIZE requires second argument to be VARSIZED, but got {}", mode.getDataType());
        }
    }
}
}

QuantizeLogicalFunction::QuantizeLogicalFunction(const LogicalFunction& value, const LogicalFunction& mode)
    : dataType(DataTypeProvider::provideDataType(DataType::Type::FLOAT32)), value(value), mode(mode)
{
    validateQuantizeChildren(value, mode);
}

bool QuantizeLogicalFunction::operator==(const QuantizeLogicalFunction& rhs) const
{
    return value == rhs.value and mode == rhs.mode;
}

std::string QuantizeLogicalFunction::explain(ExplainVerbosity verbosity) const
{
    return fmt::format("QUANTIZE({}, {})", value.explain(verbosity), mode.explain(verbosity));
}

DataType QuantizeLogicalFunction::getDataType() const
{
    return dataType;
}

QuantizeLogicalFunction QuantizeLogicalFunction::withDataType(const DataType& dataType) const
{
    auto copy = *this;
    copy.dataType = dataType;
    return copy;
}

LogicalFunction QuantizeLogicalFunction::withInferredDataType(const Schema& schema) const
{
    auto newChildren = getChildren() | std::views::transform([&schema](const auto& child) { return child.withInferredDataType(schema); })
        | std::ranges::to<std::vector>();
    return withChildren(newChildren);
}

std::vector<LogicalFunction> QuantizeLogicalFunction::getChildren() const
{
    return {value, mode};
}

QuantizeLogicalFunction QuantizeLogicalFunction::withChildren(const std::vector<LogicalFunction>& children) const
{
    PRECONDITION(children.size() == 2, "QUANTIZE function requires exactly two children, but got {}", children.size());
    validateQuantizeChildren(children[0], children[1]);

    auto copy = *this;
    copy.value = children[0];
    copy.mode = children[1];
    copy.dataType.nullable = copy.value.getDataType().nullable or copy.mode.getDataType().nullable;
    return copy;
}

std::string_view QuantizeLogicalFunction::getType() const
{
    return NAME;
}

Reflected Reflector<QuantizeLogicalFunction>::operator()(const QuantizeLogicalFunction& function) const
{
    return reflect(detail::ReflectedQuantizeLogicalFunction{.value = function.value, .mode = function.mode});
}

QuantizeLogicalFunction Unreflector<QuantizeLogicalFunction>::operator()(const Reflected& reflected) const
{
    auto [value, mode] = unreflect<detail::ReflectedQuantizeLogicalFunction>(reflected);
    if (!value.has_value() || !mode.has_value())
    {
        throw CannotDeserialize("QUANTIZE function is missing a child");
    }
    return QuantizeLogicalFunction{value.value(), mode.value()};
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterQUANTIZELogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    if (!arguments.reflected.isEmpty())
    {
        return unreflect<QuantizeLogicalFunction>(arguments.reflected);
    }
    if (arguments.children.size() != 2)
    {
        throw CannotDeserialize("QUANTIZE function requires exactly two children, but got {}", arguments.children.size());
    }
    return QuantizeLogicalFunction(arguments.children[0], arguments.children[1]);
}

}
