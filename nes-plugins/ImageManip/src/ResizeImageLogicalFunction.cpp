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

#include <string>
#include <string_view>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <DataTypes/DataTypeProvider.hpp>
#include <DataTypes/Schema.hpp>
#include <Functions/LogicalFunction.hpp>
#include <Serialization/DataTypeSerializationUtil.hpp> /// NOLINT(misc-include-cleaner)
#include <Util/PlanRenderer.hpp>
#include <fmt/format.h>
#include <ErrorHandling.hpp>
#include <LogicalFunctionRegistry.hpp>
#include <SerializableVariantDescriptor.pb.h> /// NOLINT(misc-include-cleaner)

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
            "ResizeImage requires first argument to be VARSIZED, but got {}",
            image.getDataType());
    }
    if (!width.getDataType().isType(DataType::Type::UNDEFINED))
    {
        PRECONDITION(
            width.getDataType().isInteger(),
            "ResizeImage requires second argument to be integer, but got {}",
            width.getDataType());
    }
    if (!height.getDataType().isType(DataType::Type::UNDEFINED))
    {
        PRECONDITION(
            height.getDataType().isInteger(),
            "ResizeImage requires third argument to be integer, but got {}",
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

bool ResizeImageLogicalFunction::operator==(const LogicalFunctionConcept& rhs) const
{
    if (const auto* other = dynamic_cast<const ResizeImageLogicalFunction*>(&rhs))
    {
        return image == other->image and width == other->width and height == other->height;
    }
    return false;
}

std::string ResizeImageLogicalFunction::explain(ExplainVerbosity verbosity) const
{
    return fmt::format("ResizeImage({}, {}, {})", image.explain(verbosity), width.explain(verbosity), height.explain(verbosity));
}

DataType ResizeImageLogicalFunction::getDataType() const
{
    return dataType;
}

LogicalFunction ResizeImageLogicalFunction::withDataType(const DataType& dataType) const
{
    auto copy = *this;
    copy.dataType = dataType;
    return copy;
}

LogicalFunction ResizeImageLogicalFunction::withInferredDataType(const Schema& schema) const
{
    std::vector<LogicalFunction> newChildren;
    for (const auto& child : getChildren())
    {
        newChildren.push_back(child.withInferredDataType(schema));
    }
    return withChildren(newChildren);
}

std::vector<LogicalFunction> ResizeImageLogicalFunction::getChildren() const
{
    return {image, width, height};
}

LogicalFunction ResizeImageLogicalFunction::withChildren(const std::vector<LogicalFunction>& children) const
{
    PRECONDITION(children.size() == 3, "ResizeImageLogicalFunction requires exactly three children, but got {}", children.size());
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

SerializableFunction ResizeImageLogicalFunction::serialize() const
{
    SerializableFunction serializedFunction;
    serializedFunction.set_function_type(NAME);
    serializedFunction.add_children()->CopyFrom(image.serialize());
    serializedFunction.add_children()->CopyFrom(width.serialize());
    serializedFunction.add_children()->CopyFrom(height.serialize());
    DataTypeSerializationUtil::serializeDataType(getDataType(), serializedFunction.mutable_data_type());
    return serializedFunction;
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterResizeImageLogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    if (arguments.children.size() != 3)
    {
        throw CannotDeserialize("ResizeImageLogicalFunction requires exactly three children, but got {}", arguments.children.size());
    }
    return ResizeImageLogicalFunction(arguments.children[0], arguments.children[1], arguments.children[2]);
}

}
