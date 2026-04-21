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
void validateConvertColorSchemeChildren(const LogicalFunction& image, const LogicalFunction& conversionMode)
{
    if (!image.getDataType().isType(DataType::Type::UNDEFINED))
    {
        PRECONDITION(
            image.getDataType().isType(DataType::Type::VARSIZED),
            "ConvertColorScheme requires first argument to be VARSIZED, but got {}",
            image.getDataType());
    }
    if (!conversionMode.getDataType().isType(DataType::Type::UNDEFINED))
    {
        PRECONDITION(
            conversionMode.getDataType().isType(DataType::Type::VARSIZED),
            "ConvertColorScheme requires second argument to be VARSIZED, but got {}",
            conversionMode.getDataType());
    }
}
}

ConvertColorSchemeLogicalFunction::ConvertColorSchemeLogicalFunction(
    const LogicalFunction& image, const LogicalFunction& conversionMode)
    : dataType(DataTypeProvider::provideDataType(DataType::Type::VARSIZED)), image(image), conversionMode(conversionMode)
{
    validateConvertColorSchemeChildren(image, this->conversionMode);
}

bool ConvertColorSchemeLogicalFunction::operator==(const LogicalFunctionConcept& rhs) const
{
    if (const auto* other = dynamic_cast<const ConvertColorSchemeLogicalFunction*>(&rhs))
    {
        return image == other->image and conversionMode == other->conversionMode;
    }
    return false;
}

std::string ConvertColorSchemeLogicalFunction::explain(ExplainVerbosity verbosity) const
{
    return fmt::format("ConvertColorScheme({}, {})", image.explain(verbosity), conversionMode.explain(verbosity));
}

DataType ConvertColorSchemeLogicalFunction::getDataType() const
{
    return dataType;
}

LogicalFunction ConvertColorSchemeLogicalFunction::withDataType(const DataType& dataType) const
{
    auto copy = *this;
    copy.dataType = dataType;
    return copy;
}

LogicalFunction ConvertColorSchemeLogicalFunction::withInferredDataType(const Schema& schema) const
{
    std::vector<LogicalFunction> newChildren;
    for (const auto& child : getChildren())
    {
        newChildren.push_back(child.withInferredDataType(schema));
    }
    return withChildren(newChildren);
}

std::vector<LogicalFunction> ConvertColorSchemeLogicalFunction::getChildren() const
{
    return {image, conversionMode};
}

LogicalFunction ConvertColorSchemeLogicalFunction::withChildren(const std::vector<LogicalFunction>& children) const
{
    PRECONDITION(children.size() == 2, "ConvertColorSchemeLogicalFunction requires exactly two children, but got {}", children.size());
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

SerializableFunction ConvertColorSchemeLogicalFunction::serialize() const
{
    SerializableFunction serializedFunction;
    serializedFunction.set_function_type(NAME);
    serializedFunction.add_children()->CopyFrom(image.serialize());
    serializedFunction.add_children()->CopyFrom(conversionMode.serialize());
    DataTypeSerializationUtil::serializeDataType(getDataType(), serializedFunction.mutable_data_type());
    return serializedFunction;
}

LogicalFunctionRegistryReturnType
LogicalFunctionGeneratedRegistrar::RegisterConvertColorSchemeLogicalFunction(LogicalFunctionRegistryArguments arguments)
{
    if (arguments.children.size() != 2)
    {
        throw CannotDeserialize("ConvertColorSchemeLogicalFunction requires exactly two children, but got {}", arguments.children.size());
    }
    return ConvertColorSchemeLogicalFunction(arguments.children[0], arguments.children[1]);
}

}
