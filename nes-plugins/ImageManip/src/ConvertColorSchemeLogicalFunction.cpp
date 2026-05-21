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

#include "../include/ConvertColorSchemeLogicalFunction.hpp"

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
void validateConvertColorSchemeChildren(const LogicalFunction& image, const LogicalFunction& conversionMode)
{
    if (!image.getDataType().isType(DataType::Type::UNDEFINED))
    {
        PRECONDITION(
            image.getDataType().isType(DataType::Type::VARSIZED),
            "CONVERT_COLOR_SCHEME requires first argument to be VARSIZED, but got {}",
            image.getDataType());
    }
    if (!conversionMode.getDataType().isType(DataType::Type::UNDEFINED))
    {
        PRECONDITION(
            conversionMode.getDataType().isType(DataType::Type::VARSIZED),
            "CONVERT_COLOR_SCHEME requires second argument to be VARSIZED, but got {}",
            conversionMode.getDataType());
    }
}
}

ConvertColorSchemeLogicalFunction::ConvertColorSchemeLogicalFunction(const LogicalFunction& image, const LogicalFunction& conversionMode)
    : dataType(DataTypeProvider::provideDataType(DataType::Type::VARSIZED)), image(image), conversionMode(conversionMode)
{
    validateConvertColorSchemeChildren(image, this->conversionMode);
}

bool ConvertColorSchemeLogicalFunction::operator==(const ConvertColorSchemeLogicalFunction& rhs) const
{
    return image == rhs.image and conversionMode == rhs.conversionMode;
}

std::string ConvertColorSchemeLogicalFunction::explain(ExplainVerbosity verbosity) const
{
    return fmt::format("CONVERT_COLOR_SCHEME({}, {})", image.explain(verbosity), conversionMode.explain(verbosity));
}

DataType ConvertColorSchemeLogicalFunction::getDataType() const
{
    return dataType;
}

ConvertColorSchemeLogicalFunction ConvertColorSchemeLogicalFunction::withDataType(const DataType& dataType) const
{
    auto copy = *this;
    copy.dataType = dataType;
    return copy;
}

LogicalFunction ConvertColorSchemeLogicalFunction::withInferredDataType(const Schema& schema) const
{
    auto newChildren = getChildren() | std::views::transform([&schema](const auto& child) { return child.withInferredDataType(schema); })
        | std::ranges::to<std::vector>();
    return withChildren(newChildren);
}

std::vector<LogicalFunction> ConvertColorSchemeLogicalFunction::getChildren() const
{
    return {image, conversionMode};
}

ConvertColorSchemeLogicalFunction ConvertColorSchemeLogicalFunction::withChildren(const std::vector<LogicalFunction>& children) const
{
    PRECONDITION(children.size() == 2, "CONVERT_COLOR_SCHEME function requires exactly two children, but got {}", children.size());
    validateConvertColorSchemeChildren(children[0], children[1]);

    auto copy = *this;
    copy.image = children[0];
    copy.conversionMode = children[1];
    return copy;
}

std::string_view ConvertColorSchemeLogicalFunction::getType() const
{
    return NAME;
}

Reflected Reflector<ConvertColorSchemeLogicalFunction>::operator()(const ConvertColorSchemeLogicalFunction& function) const
{
    return reflect(detail::ReflectedConvertColorSchemeLogicalFunction{.image = function.image, .conversionMode = function.conversionMode});
}

ConvertColorSchemeLogicalFunction Unreflector<ConvertColorSchemeLogicalFunction>::operator()(const Reflected& reflected) const
{
    auto [image, conversionMode] = unreflect<detail::ReflectedConvertColorSchemeLogicalFunction>(reflected);
    if (!image.has_value() || !conversionMode.has_value())
    {
        throw CannotDeserialize("CONVERT_COLOR_SCHEME function is missing a child");
    }
    return ConvertColorSchemeLogicalFunction{image.value(), conversionMode.value()};
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterCONVERT_COLOR_SCHEMELogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    if (!arguments.reflected.isEmpty())
    {
        return unreflect<ConvertColorSchemeLogicalFunction>(arguments.reflected);
    }
    if (arguments.children.size() != 2)
    {
        throw CannotDeserialize("CONVERT_COLOR_SCHEME function requires exactly two children, but got {}", arguments.children.size());
    }
    return ConvertColorSchemeLogicalFunction(arguments.children[0], arguments.children[1]);
}

}
