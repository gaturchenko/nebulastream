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

#include <Operators/Windows/Aggregations/ArrayAggregationLogicalFunction.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include <DataTypes/DataType.hpp>
#include <DataTypes/DataTypeProvider.hpp>
#include <DataTypes/Schema.hpp>
#include <Functions/FieldAccessLogicalFunction.hpp>
#include <Operators/Windows/Aggregations/WindowAggregationLogicalFunction.hpp>
#include <Util/Reflection.hpp>
#include <fmt/format.h>
#include <AggregationLogicalFunctionRegistry.hpp>
#include <ErrorHandling.hpp>

namespace NES
{

ArrayAggregationLogicalFunction::ArrayAggregationLogicalFunction(const FieldAccessLogicalFunction& field) : onField(field), asField(field)
{
}

ArrayAggregationLogicalFunction::ArrayAggregationLogicalFunction(
    const FieldAccessLogicalFunction& onField, FieldAccessLogicalFunction asField)
    : onField(onField), asField(std::move(asField))
{
}

bool ArrayAggregationLogicalFunction::shallIncludeNullValues() noexcept
{
    return false;
}

bool ArrayAggregationLogicalFunction::requiresSequentialAggregation() noexcept
{
    return true;
}

std::string_view ArrayAggregationLogicalFunction::getName() const noexcept
{
    return NAME;
}

ArrayAggregationLogicalFunction ArrayAggregationLogicalFunction::withInferredStamp(const Schema& schema) const
{
    const auto sourceNameQualifier = schema.getSourceNameQualifier();
    if (!sourceNameQualifier.has_value())
    {
        throw CannotInferSchema("Schema lacked source name qualifier: {}", schema);
    }

    auto newOnField = getOnField().withInferredDataType(schema).getAs<FieldAccessLogicalFunction>().get();
    if (!newOnField.getDataType().isNumeric())
    {
        throw CannotDeserialize("ARRAY_AGG on non numeric fields is not supported, but got {}", newOnField.getDataType());
    }

    const auto attributeNameResolver = sourceNameQualifier.value() + std::string(Schema::ATTRIBUTE_NAME_SEPARATOR);
    const auto asFieldName = getAsField().getFieldName();

    std::string newAsFieldName;
    if (asFieldName.find(Schema::ATTRIBUTE_NAME_SEPARATOR) == std::string::npos)
    {
        newAsFieldName = attributeNameResolver + asFieldName;
    }
    else
    {
        const auto fieldName = asFieldName.substr(asFieldName.find_last_of(Schema::ATTRIBUTE_NAME_SEPARATOR) + 1);
        newAsFieldName = attributeNameResolver + fieldName;
    }

    auto newFinalAggregationStamp = DataTypeProvider::provideDataType(DataType::Type::VARSIZED, DataType::NULLABLE::NOT_NULLABLE);
    return withInputStamp(newOnField.getDataType())
        .withOnField(newOnField)
        .withFinalAggregateStamp(newFinalAggregationStamp)
        .withAsField(getAsField().withFieldName(newAsFieldName).withDataType(newFinalAggregationStamp));
}

Reflected ArrayAggregationLogicalFunction::reflect() const
{
    return NES::reflect(this);
}

Reflected Reflector<ArrayAggregationLogicalFunction>::operator()(const ArrayAggregationLogicalFunction& function) const
{
    return reflect(detail::ReflectedArrayAggregationLogicalFunction{.onField = function.getOnField(), .asField = function.getAsField()});
}

ArrayAggregationLogicalFunction Unreflector<ArrayAggregationLogicalFunction>::operator()(const Reflected& reflected) const
{
    auto [onField, asField] = unreflect<detail::ReflectedArrayAggregationLogicalFunction>(reflected);
    return ArrayAggregationLogicalFunction{onField, asField};
}

AggregationLogicalFunctionRegistryReturnType AggregationLogicalFunctionGeneratedRegistrar::RegisterArray_AggAggregationLogicalFunction(
    AggregationLogicalFunctionRegistryArguments arguments)
{
    if (!arguments.reflected.isEmpty())
    {
        return std::make_shared<WindowAggregationLogicalFunction>(unreflect<ArrayAggregationLogicalFunction>(arguments.reflected));
    }
    if (arguments.fields.size() != 2)
    {
        throw CannotDeserialize("ArrayAggregationLogicalFunction requires exactly two fields, but got {}", arguments.fields.size());
    }
    return std::make_shared<WindowAggregationLogicalFunction>(ArrayAggregationLogicalFunction(arguments.fields[0], arguments.fields[1]));
}

std::string ArrayAggregationLogicalFunction::toString() const
{
    return fmt::format("WindowAggregation: onField={} asField={}", onField, asField);
}

DataType ArrayAggregationLogicalFunction::getInputStamp() const
{
    return inputStamp;
}

DataType ArrayAggregationLogicalFunction::getPartialAggregateStamp() const
{
    return partialAggregateStamp;
}

DataType ArrayAggregationLogicalFunction::getFinalAggregateStamp() const
{
    return finalAggregateStamp;
}

FieldAccessLogicalFunction ArrayAggregationLogicalFunction::getOnField() const
{
    return onField;
}

FieldAccessLogicalFunction ArrayAggregationLogicalFunction::getAsField() const
{
    return asField;
}

ArrayAggregationLogicalFunction ArrayAggregationLogicalFunction::withInputStamp(DataType inputStamp) const
{
    auto copy = *this;
    copy.inputStamp = std::move(inputStamp);
    return copy;
}

ArrayAggregationLogicalFunction ArrayAggregationLogicalFunction::withPartialAggregateStamp(DataType partialAggregateStamp) const
{
    auto copy = *this;
    copy.partialAggregateStamp = std::move(partialAggregateStamp);
    return copy;
}

ArrayAggregationLogicalFunction ArrayAggregationLogicalFunction::withFinalAggregateStamp(DataType finalAggregateStamp) const
{
    auto copy = *this;
    copy.finalAggregateStamp = std::move(finalAggregateStamp);
    return copy;
}

ArrayAggregationLogicalFunction ArrayAggregationLogicalFunction::withOnField(FieldAccessLogicalFunction onField) const
{
    auto copy = *this;
    copy.onField = std::move(onField);
    return copy;
}

ArrayAggregationLogicalFunction ArrayAggregationLogicalFunction::withAsField(FieldAccessLogicalFunction asField) const
{
    auto copy = *this;
    copy.asField = std::move(asField);
    return copy;
}

bool ArrayAggregationLogicalFunction::operator==(const ArrayAggregationLogicalFunction& other) const
{
    return getName() == other.getName() && onField == other.onField && asField == other.asField;
}

}
