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

#include <Functions/ArithmeticalFunctions/RoundLogicalFunction.hpp>

#include <algorithm>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <DataTypes/DataTypeProvider.hpp>
#include <DataTypes/Schema.hpp>
#include <Functions/LogicalFunction.hpp>
#include <Serialization/DataTypeSerializationUtil.hpp>
#include <Serialization/LogicalFunctionReflection.hpp>
#include <Util/PlanRenderer.hpp>
#include <Util/Reflection.hpp>
#include <fmt/format.h>
#include <ErrorHandling.hpp>
#include <LogicalFunctionRegistry.hpp>
#include <SerializableVariantDescriptor.pb.h>

namespace NES
{

RoundLogicalFunction::RoundLogicalFunction(const LogicalFunction& child) : dataType(child.getDataType()), child(child) { };

RoundLogicalFunction::RoundLogicalFunction(const LogicalFunction& child, const LogicalFunction& decimalPlaces)
    : dataType(child.getDataType()), child(child), decimalPlaces(decimalPlaces) { };

bool RoundLogicalFunction::operator==(const RoundLogicalFunction& rhs) const
{
    return child == rhs.child and decimalPlaces == rhs.decimalPlaces;
}

std::string RoundLogicalFunction::explain(ExplainVerbosity verbosity) const
{
    if (verbosity == ExplainVerbosity::Debug)
    {
        if (decimalPlaces.has_value())
        {
            return fmt::format(
                "RoundLogicalFunction({}, {} : {})", child.explain(verbosity), decimalPlaces->explain(verbosity), dataType);
        }
        return fmt::format("RoundLogicalFunction({} : {})", child.explain(verbosity), dataType);
    }
    if (decimalPlaces.has_value())
    {
        return fmt::format("ROUND({}, {})", child.explain(verbosity), decimalPlaces->explain(verbosity));
    }
    return fmt::format("ROUND({})", child.explain(verbosity));
}

DataType RoundLogicalFunction::getDataType() const
{
    return dataType;
};

RoundLogicalFunction RoundLogicalFunction::withDataType(const DataType& dataType) const
{
    auto copy = *this;
    copy.dataType = dataType;
    return copy;
};

LogicalFunction RoundLogicalFunction::withInferredDataType(const Schema& schema) const
{
    const auto newChildren = getChildren() | std::views::transform([&schema](auto& child) { return child.withInferredDataType(schema); })
        | std::ranges::to<std::vector>();
    INVARIANT(
        newChildren.size() == 1 or newChildren.size() == 2,
        "RoundLogicalFunction expects one or two child functions but has {}",
        newChildren.size());
    if (not newChildren[0].getDataType().isNumeric())
    {
        throw DifferentFieldTypeExpected("ROUND expects a numeric input but got {}", newChildren[0].getDataType());
    }
    if (newChildren.size() == 2)
    {
        if (not newChildren[0].getDataType().isFloat())
        {
            throw DifferentFieldTypeExpected(
                "ROUND with decimal places expects a FLOAT32/FLOAT64 input but got {}", newChildren[0].getDataType());
        }
        if (not newChildren[1].getDataType().isInteger())
        {
            throw DifferentFieldTypeExpected(
                "ROUND decimal places argument expects an integer input but got {}", newChildren[1].getDataType());
        }
    }
    auto newDataType = newChildren[0].getDataType();
    newDataType.nullable = std::ranges::any_of(newChildren, [](const auto& child) { return child.getDataType().nullable; });
    return withDataType(newDataType).withChildren(newChildren);
};

std::vector<LogicalFunction> RoundLogicalFunction::getChildren() const
{
    if (decimalPlaces.has_value())
    {
        return {child, decimalPlaces.value()};
    }
    return {child};
};

RoundLogicalFunction RoundLogicalFunction::withChildren(const std::vector<LogicalFunction>& children) const
{
    PRECONDITION(
        children.size() == 1 or children.size() == 2, "RoundLogicalFunction requires one or two children, but got {}", children.size());
    auto copy = *this;
    copy.child = children[0];
    copy.decimalPlaces = children.size() == 2 ? std::optional<LogicalFunction>{children[1]} : std::nullopt;
    return copy;
};

std::string_view RoundLogicalFunction::getType() const
{
    return NAME;
}

Reflected Reflector<RoundLogicalFunction>::operator()(const RoundLogicalFunction& function) const
{
    return reflect(detail::ReflectedRoundLogicalFunction{.child = function.child, .decimalPlaces = function.decimalPlaces});
}

RoundLogicalFunction Unreflector<RoundLogicalFunction>::operator()(const Reflected& reflected) const
{
    auto [child, decimalPlaces] = unreflect<detail::ReflectedRoundLogicalFunction>(reflected);
    if (!child.has_value())
    {
        throw CannotDeserialize("Missing child function");
    }
    if (decimalPlaces.has_value())
    {
        return RoundLogicalFunction(child.value(), decimalPlaces.value());
    }
    return RoundLogicalFunction(child.value());
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterRoundLogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    if (!arguments.reflected.isEmpty())
    {
        return unreflect<RoundLogicalFunction>(arguments.reflected);
    }
    if (arguments.children.size() != 1 and arguments.children.size() != 2)
    {
        throw CannotDeserialize("Function requires one or two children, but got {}", arguments.children.size());
    }
    if (arguments.children.size() == 2)
    {
        return RoundLogicalFunction(arguments.children[0], arguments.children[1]);
    }
    return RoundLogicalFunction(arguments.children[0]);
}

}
