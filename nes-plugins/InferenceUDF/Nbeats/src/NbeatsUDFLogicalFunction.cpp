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

#include "../include/NbeatsUDFLogicalFunction.hpp"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <DataTypes/DataTypeProvider.hpp>
#include <DataTypes/Schema.hpp>
#include <Functions/LogicalFunction.hpp>
#include <Serialization/DataTypeSerializationUtil.hpp> /// NOLINT(misc-include-cleaner)
#include <Serialization/LogicalFunctionReflection.hpp>
#include <Util/PlanRenderer.hpp>
#include <Util/Reflection.hpp>
#include <fmt/format.h>
#include <ErrorHandling.hpp>
#include <LogicalFunctionRegistry.hpp>
#include <SerializableVariantDescriptor.pb.h> /// NOLINT(misc-include-cleaner)

namespace NES
{

namespace
{
constexpr std::string_view NbeatsSize120Stride12Udf = "NBEATS_SIZE_120_STRIDE_12_UDF";
constexpr std::string_view NbeatsSize300Stride30Udf = "NBEATS_SIZE_300_STRIDE_30_UDF";
constexpr std::string_view NbeatsSize600Stride60Udf = "NBEATS_SIZE_600_STRIDE_60_UDF";

LogicalFunction makeNbeatsUdfLogicalFunction(std::string_view functionName, LogicalFunctionRegistryArguments arguments)
{
    if (!arguments.reflected.isEmpty())
    {
        return unreflect<NbeatsUDFLogicalFunction>(arguments.reflected);
    }
    if (arguments.children.size() != 1)
    {
        throw CannotDeserialize("{} requires one argument", functionName);
    }
    return NbeatsUDFLogicalFunction(std::string{functionName}, arguments.children.back());
}
}

NbeatsUDFLogicalFunction::NbeatsUDFLogicalFunction(std::string functionName, const LogicalFunction& child)
    : functionName(std::move(functionName)), dataType(DataTypeProvider::provideDataType(DataType::Type::VARSIZED)), child(child)
{
}

bool NbeatsUDFLogicalFunction::operator==(const NbeatsUDFLogicalFunction& rhs) const
{
    return functionName == rhs.functionName && child == rhs.child;
}

DataType NbeatsUDFLogicalFunction::getDataType() const
{
    return dataType;
}

NbeatsUDFLogicalFunction NbeatsUDFLogicalFunction::withDataType(const DataType& dataType) const
{
    auto copy = *this;
    copy.dataType = dataType;
    return copy;
}

LogicalFunction NbeatsUDFLogicalFunction::withInferredDataType(const Schema& schema) const
{
    std::vector<LogicalFunction> newChildren;
    for (auto& childFunction : getChildren())
    {
        newChildren.push_back(childFunction.withInferredDataType(schema));
    }
    PRECONDITION(newChildren.size() == 1, "{} expects exactly one child but has {}", functionName, newChildren.size());
    if (not newChildren[0].getDataType().isType(DataType::Type::VARSIZED))
    {
        throw DifferentFieldTypeExpected("{} expects a VARSIZED input but got {}", functionName, newChildren[0].getDataType());
    }

    auto newDataType = DataTypeProvider::provideDataType(DataType::Type::VARSIZED);
    newDataType.nullable = newChildren[0].getDataType().nullable;
    return withDataType(newDataType).withChildren(newChildren);
}

std::vector<LogicalFunction> NbeatsUDFLogicalFunction::getChildren() const
{
    return {child};
}

NbeatsUDFLogicalFunction NbeatsUDFLogicalFunction::withChildren(const std::vector<LogicalFunction>& children) const
{
    PRECONDITION(children.size() == 1, "{} requires exactly one child, but got {}", functionName, children.size());
    auto copy = *this;
    copy.child = children[0];
    return copy;
}

std::string_view NbeatsUDFLogicalFunction::getType() const
{
    return functionName;
}

std::string NbeatsUDFLogicalFunction::explain(ExplainVerbosity verbosity) const
{
    return fmt::format("{}({})", functionName, child.explain(verbosity));
}

Reflected Reflector<NbeatsUDFLogicalFunction>::operator()(const NbeatsUDFLogicalFunction& function) const
{
    return reflect(detail::ReflectedNbeatsUDFLogicalFunction{.functionName = function.functionName, .child = function.child});
}

NbeatsUDFLogicalFunction Unreflector<NbeatsUDFLogicalFunction>::operator()(const Reflected& reflected) const
{
    auto [functionName, child] = unreflect<detail::ReflectedNbeatsUDFLogicalFunction>(reflected);
    if (!functionName.has_value())
    {
        throw CannotDeserialize("NbeatsUDFLogicalFunction is missing its function name");
    }
    if (!child.has_value())
    {
        throw CannotDeserialize("{} is missing its child", functionName.value());
    }
    return NbeatsUDFLogicalFunction{functionName.value(), child.value()};
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterNBEATS_SIZE_120_STRIDE_12_UDFLogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    return makeNbeatsUdfLogicalFunction(NbeatsSize120Stride12Udf, arguments);
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterNBEATS_SIZE_300_STRIDE_30_UDFLogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    return makeNbeatsUdfLogicalFunction(NbeatsSize300Stride30Udf, arguments);
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterNBEATS_SIZE_600_STRIDE_60_UDFLogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    return makeNbeatsUdfLogicalFunction(NbeatsSize600Stride60Udf, arguments);
}

}
