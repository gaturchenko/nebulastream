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

#include "../include/ModelZooUDFLogicalFunction.hpp"

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
constexpr std::string_view SaradSize200Stride10Udf = "SARAD_SIZE_200_STRIDE_10_UDF";
constexpr std::string_view MhattRnnUdf = "MHATT_RNN_UDF";
constexpr std::string_view SqueezeDetUdf = "SQUEEZEDET_UDF";
constexpr std::string_view SqueezeDet192x624Udf = "SQUEEZEDET_192X624_UDF";

LogicalFunction makeModelZooUdfLogicalFunction(std::string_view functionName, LogicalFunctionRegistryArguments arguments)
{
    if (!arguments.reflected.isEmpty())
    {
        return unreflect<ModelZooUDFLogicalFunction>(arguments.reflected);
    }
    if (arguments.children.size() != 1)
    {
        throw CannotDeserialize("{} requires one argument", functionName);
    }
    return ModelZooUDFLogicalFunction(std::string{functionName}, arguments.children.back());
}
}

ModelZooUDFLogicalFunction::ModelZooUDFLogicalFunction(std::string functionName, const LogicalFunction& child)
    : functionName(std::move(functionName)), dataType(DataTypeProvider::provideDataType(DataType::Type::VARSIZED)), child(child)
{
}

bool ModelZooUDFLogicalFunction::operator==(const ModelZooUDFLogicalFunction& rhs) const
{
    return functionName == rhs.functionName && child == rhs.child;
}

DataType ModelZooUDFLogicalFunction::getDataType() const
{
    return dataType;
}

ModelZooUDFLogicalFunction ModelZooUDFLogicalFunction::withDataType(const DataType& dataType) const
{
    auto copy = *this;
    copy.dataType = dataType;
    return copy;
}

LogicalFunction ModelZooUDFLogicalFunction::withInferredDataType(const Schema& schema) const
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

std::vector<LogicalFunction> ModelZooUDFLogicalFunction::getChildren() const
{
    return {child};
}

ModelZooUDFLogicalFunction ModelZooUDFLogicalFunction::withChildren(const std::vector<LogicalFunction>& children) const
{
    PRECONDITION(children.size() == 1, "{} requires exactly one child, but got {}", functionName, children.size());
    auto copy = *this;
    copy.child = children[0];
    return copy;
}

std::string_view ModelZooUDFLogicalFunction::getType() const
{
    return functionName;
}

std::string ModelZooUDFLogicalFunction::explain(ExplainVerbosity verbosity) const
{
    return fmt::format("{}({})", functionName, child.explain(verbosity));
}

Reflected Reflector<ModelZooUDFLogicalFunction>::operator()(const ModelZooUDFLogicalFunction& function) const
{
    return reflect(detail::ReflectedModelZooUDFLogicalFunction{.functionName = function.functionName, .child = function.child});
}

ModelZooUDFLogicalFunction Unreflector<ModelZooUDFLogicalFunction>::operator()(const Reflected& reflected) const
{
    auto [functionName, child] = unreflect<detail::ReflectedModelZooUDFLogicalFunction>(reflected);
    if (!functionName.has_value())
    {
        throw CannotDeserialize("ModelZooUDFLogicalFunction is missing its function name");
    }
    if (!child.has_value())
    {
        throw CannotDeserialize("{} is missing its child", functionName.value());
    }
    return ModelZooUDFLogicalFunction{functionName.value(), child.value()};
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterSARAD_SIZE_200_STRIDE_10_UDFLogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    return makeModelZooUdfLogicalFunction(SaradSize200Stride10Udf, arguments);
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterMHATT_RNN_UDFLogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    return makeModelZooUdfLogicalFunction(MhattRnnUdf, arguments);
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterSQUEEZEDET_UDFLogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    return makeModelZooUdfLogicalFunction(SqueezeDetUdf, arguments);
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterSQUEEZEDET_192X624_UDFLogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    return makeModelZooUdfLogicalFunction(SqueezeDet192x624Udf, arguments);
}

}
