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

#include "../include/NanodetUDFLogicalFunction.hpp"

#include <string>
#include <string_view>
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

NanodetUDFLogicalFunction::NanodetUDFLogicalFunction(const LogicalFunction& child)
    : dataType(DataTypeProvider::provideDataType(DataType::Type::VARSIZED)), child(child)
{
}

bool NanodetUDFLogicalFunction::operator==(const NanodetUDFLogicalFunction& rhs) const
{
    return child == rhs.child;
}

DataType NanodetUDFLogicalFunction::getDataType() const
{
    return dataType;
}

NanodetUDFLogicalFunction NanodetUDFLogicalFunction::withDataType(const DataType& dataType) const
{
    auto copy = *this;
    copy.dataType = dataType;
    return copy;
}

LogicalFunction NanodetUDFLogicalFunction::withInferredDataType(const Schema& schema) const
{
    std::vector<LogicalFunction> newChildren;
    for (auto& childFunction : getChildren())
    {
        newChildren.push_back(childFunction.withInferredDataType(schema));
    }
    PRECONDITION(newChildren.size() == 1, "NANODET_UDF expects exactly one child but has {}", newChildren.size());
    if (not newChildren[0].getDataType().isType(DataType::Type::VARSIZED))
    {
        throw DifferentFieldTypeExpected("NANODET_UDF expects a VARSIZED input but got {}", newChildren[0].getDataType());
    }

    auto newDataType = DataTypeProvider::provideDataType(DataType::Type::VARSIZED);
    newDataType.nullable = newChildren[0].getDataType().nullable;
    return withDataType(newDataType).withChildren(newChildren);
}

std::vector<LogicalFunction> NanodetUDFLogicalFunction::getChildren() const
{
    return {child};
}

NanodetUDFLogicalFunction NanodetUDFLogicalFunction::withChildren(const std::vector<LogicalFunction>& children) const
{
    PRECONDITION(children.size() == 1, "NANODET_UDF requires exactly one child, but got {}", children.size());
    auto copy = *this;
    copy.child = children[0];
    return copy;
}

std::string_view NanodetUDFLogicalFunction::getType() const
{
    return NAME;
}

std::string NanodetUDFLogicalFunction::explain(ExplainVerbosity verbosity) const
{
    return fmt::format("NANODET_UDF({})", child.explain(verbosity));
}

Reflected Reflector<NanodetUDFLogicalFunction>::operator()(const NanodetUDFLogicalFunction& function) const
{
    return reflect(detail::ReflectedNanodetUDFLogicalFunction{.child = function.child});
}

NanodetUDFLogicalFunction Unreflector<NanodetUDFLogicalFunction>::operator()(const Reflected& reflected) const
{
    auto [child] = unreflect<detail::ReflectedNanodetUDFLogicalFunction>(reflected);
    if (!child.has_value())
    {
        throw CannotDeserialize("NanodetUDFLogicalFunction is missing its child");
    }
    return NanodetUDFLogicalFunction{child.value()};
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterNANODET_UDFLogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    if (!arguments.reflected.isEmpty())
    {
        return unreflect<NanodetUDFLogicalFunction>(arguments.reflected);
    }
    if (arguments.children.size() != 1)
    {
        throw CannotDeserialize("NANODET_UDF requires one argument");
    }
    return NanodetUDFLogicalFunction(arguments.children.back());
}

}
