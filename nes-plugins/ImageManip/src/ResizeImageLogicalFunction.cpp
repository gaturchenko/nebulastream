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

#include "../include/ResizeImageLogicalFunction.hpp"

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
void validateResizeImageChildren(const LogicalFunction& image, const LogicalFunction& width, const LogicalFunction& height)
{
    if (!image.getDataType().isType(DataType::Type::UNDEFINED))
    {
        PRECONDITION(
            image.getDataType().isType(DataType::Type::VARSIZED),
            "RESIZE_IMAGE requires first argument to be VARSIZED, but got {}",
            image.getDataType());
    }
    if (!width.getDataType().isType(DataType::Type::UNDEFINED))
    {
        PRECONDITION(
            width.getDataType().isInteger(),
            "RESIZE_IMAGE requires second argument to be integer, but got {}",
            width.getDataType());
    }
    if (!height.getDataType().isType(DataType::Type::UNDEFINED))
    {
        PRECONDITION(
            height.getDataType().isInteger(),
            "RESIZE_IMAGE requires third argument to be integer, but got {}",
            height.getDataType());
    }
}
}

ResizeImageLogicalFunction::ResizeImageLogicalFunction(
    const LogicalFunction& image, const LogicalFunction& width, const LogicalFunction& height)
    : dataType(DataTypeProvider::provideDataType(DataType::Type::VARSIZED)), image(image), width(width), height(height)
{
    validateResizeImageChildren(image, width, height);
}

bool ResizeImageLogicalFunction::operator==(const ResizeImageLogicalFunction& rhs) const
{
    return image == rhs.image and width == rhs.width and height == rhs.height;
}

std::string ResizeImageLogicalFunction::explain(ExplainVerbosity verbosity) const
{
    return fmt::format("RESIZE_IMAGE({}, {}, {})", image.explain(verbosity), width.explain(verbosity), height.explain(verbosity));
}

DataType ResizeImageLogicalFunction::getDataType() const
{
    return dataType;
}

ResizeImageLogicalFunction ResizeImageLogicalFunction::withDataType(const DataType& dataType) const
{
    auto copy = *this;
    copy.dataType = dataType;
    return copy;
}

LogicalFunction ResizeImageLogicalFunction::withInferredDataType(const Schema& schema) const
{
    auto newChildren = getChildren() | std::views::transform([&schema](const auto& child) { return child.withInferredDataType(schema); })
        | std::ranges::to<std::vector>();
    return withChildren(newChildren);
}

std::vector<LogicalFunction> ResizeImageLogicalFunction::getChildren() const
{
    return {image, width, height};
}

ResizeImageLogicalFunction ResizeImageLogicalFunction::withChildren(const std::vector<LogicalFunction>& children) const
{
    PRECONDITION(children.size() == 3, "RESIZE_IMAGE function requires exactly three children, but got {}", children.size());
    validateResizeImageChildren(children[0], children[1], children[2]);

    auto copy = *this;
    copy.image = children[0];
    copy.width = children[1];
    copy.height = children[2];
    return copy;
}

std::string_view ResizeImageLogicalFunction::getType() const
{
    return NAME;
}

Reflected Reflector<ResizeImageLogicalFunction>::operator()(const ResizeImageLogicalFunction& function) const
{
    return reflect(detail::ReflectedResizeImageLogicalFunction{
        .image = function.image,
        .width = function.width,
        .height = function.height});
}

ResizeImageLogicalFunction Unreflector<ResizeImageLogicalFunction>::operator()(const Reflected& reflected) const
{
    auto [image, width, height] = unreflect<detail::ReflectedResizeImageLogicalFunction>(reflected);
    if (!image.has_value() || !width.has_value() || !height.has_value())
    {
        throw CannotDeserialize("RESIZE_IMAGE function is missing a child");
    }
    return ResizeImageLogicalFunction{image.value(), width.value(), height.value()};
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterRESIZE_IMAGELogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    if (!arguments.reflected.isEmpty())
    {
        return unreflect<ResizeImageLogicalFunction>(arguments.reflected);
    }
    if (arguments.children.size() != 3)
    {
        throw CannotDeserialize("RESIZE_IMAGE function requires exactly three children, but got {}", arguments.children.size());
    }
    return ResizeImageLogicalFunction(arguments.children[0], arguments.children[1], arguments.children[2]);
}

}
