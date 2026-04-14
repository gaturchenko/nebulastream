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

#include "../include/ToggleGrayRgbLogicalFunction.hpp"

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
void validateToggleGrayRgbChild(const LogicalFunction& image)
{
    if (!image.getDataType().isType(DataType::Type::UNDEFINED))
    {
        PRECONDITION(
            image.getDataType().isType(DataType::Type::VARSIZED),
            "ToggleGrayRgb requires first argument to be VARSIZED, but got {}",
            image.getDataType());
    }
}
}

ToggleGrayRgbLogicalFunction::ToggleGrayRgbLogicalFunction(const LogicalFunction& image)
    : dataType(DataTypeProvider::provideDataType(DataType::Type::VARSIZED)), image(image)
{
    validateToggleGrayRgbChild(image);
}

bool ToggleGrayRgbLogicalFunction::operator==(const LogicalFunctionConcept& rhs) const
{
    if (const auto* other = dynamic_cast<const ToggleGrayRgbLogicalFunction*>(&rhs))
    {
        return image == other->image;
    }
    return false;
}

std::string ToggleGrayRgbLogicalFunction::explain(ExplainVerbosity verbosity) const
{
    return fmt::format("ToggleGrayRgb({})", image.explain(verbosity));
}

DataType ToggleGrayRgbLogicalFunction::getDataType() const
{
    return dataType;
}

LogicalFunction ToggleGrayRgbLogicalFunction::withDataType(const DataType& dataType) const
{
    auto copy = *this;
    copy.dataType = dataType;
    return copy;
}

LogicalFunction ToggleGrayRgbLogicalFunction::withInferredDataType(const Schema& schema) const
{
    std::vector<LogicalFunction> newChildren;
    for (const auto& child : getChildren())
    {
        newChildren.push_back(child.withInferredDataType(schema));
    }
    return withChildren(newChildren);
}

std::vector<LogicalFunction> ToggleGrayRgbLogicalFunction::getChildren() const
{
    return {image};
}

LogicalFunction ToggleGrayRgbLogicalFunction::withChildren(const std::vector<LogicalFunction>& children) const
{
    PRECONDITION(children.size() == 1, "ToggleGrayRgbLogicalFunction requires exactly one child, but got {}", children.size());
    validateToggleGrayRgbChild(children[0]);

    auto copy = *this;
    copy.image = children[0];
    return copy;
}

std::string_view ToggleGrayRgbLogicalFunction::getType() const
{
    return NAME;
}

SerializableFunction ToggleGrayRgbLogicalFunction::serialize() const
{
    SerializableFunction serializedFunction;
    serializedFunction.set_function_type(NAME);
    serializedFunction.add_children()->CopyFrom(image.serialize());
    DataTypeSerializationUtil::serializeDataType(getDataType(), serializedFunction.mutable_data_type());
    return serializedFunction;
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterToggleGrayRgbLogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    if (arguments.children.size() != 1)
    {
        throw CannotDeserialize("ToggleGrayRgbLogicalFunction requires exactly one child, but got {}", arguments.children.size());
    }
    return ToggleGrayRgbLogicalFunction(arguments.children[0]);
}

}
