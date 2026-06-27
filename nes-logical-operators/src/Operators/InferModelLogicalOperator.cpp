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

#include <Operators/InferModelLogicalOperator.hpp>

#include <cstddef>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <DataTypes/DataType.hpp>
#include <DataTypes/Schema.hpp>
#include <Identifiers/Identifiers.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Traits/TraitSet.hpp>
#include <Util/PlanRenderer.hpp>
#include <Util/Reflection.hpp>
#include <ErrorHandling.hpp>
#include <LogicalOperatorRegistry.hpp>
#include <ModelCatalog.hpp>

namespace NES
{

namespace
{

void validateModelInputField(const Schema::Field& field, const DataType& expectedType, const std::string& requestedFieldName)
{
    if (field.dataType.nullable)
    {
        throw CannotInferSchema("Field '{}' is nullable, but model inputs must not be nullable", requestedFieldName);
    }
    if (field.dataType.type != expectedType.type)
    {
        throw CannotInferSchema("Type mismatch for field '{}': schema has a different type than model expects", requestedFieldName);
    }
}

std::vector<std::string> resolveOrDeferSingleVarsizedInputFields(const Schema& inputSchema, const std::vector<std::string>& inputFieldNames)
{
    PRECONDITION(!inputFieldNames.empty(), "Expected at least one input field name for varsized model input resolution");
    if (inputFieldNames.size() > 1)
    {
        std::vector<std::string> resolvedInputFields;
        resolvedInputFields.reserve(inputFieldNames.size());
        for (const auto& inputFieldName : inputFieldNames)
        {
            const auto field = inputSchema.getFieldByName(inputFieldName);
            if (!field.has_value())
            {
                throw CannotInferSchema("Field '{}' not found in input schema", inputFieldName);
            }
            validateModelInputField(field.value(), DataType{DataType::Type::VARSIZED, DataType::NULLABLE::NOT_NULLABLE}, inputFieldName);
            resolvedInputFields.push_back(field->name);
        }
        return resolvedInputFields;
    }

    const auto& inputFieldName = inputFieldNames.front();
    const auto requestedFieldIsQualified = inputFieldName.find(Schema::ATTRIBUTE_NAME_SEPARATOR) != std::string::npos;
    if (!requestedFieldIsQualified)
    {
        auto matchingVarsizedFields = inputSchema.getFields()
            | std::views::filter(
                                          [&inputFieldName](const auto& candidate)
                                          {
                                              return candidate.getUnqualifiedName() == inputFieldName
                                                  && candidate.dataType.type == DataType::Type::VARSIZED && !candidate.dataType.nullable;
                                          })
            | std::views::transform([](const auto& candidate) { return candidate.name; }) | std::ranges::to<std::vector>();
        if (matchingVarsizedFields.size() > 1)
        {
            return inputFieldNames;
        }
    }

    const auto field = inputSchema.getFieldByName(inputFieldName);
    if (field.has_value())
    {
        validateModelInputField(field.value(), DataType{DataType::Type::VARSIZED, DataType::NULLABLE::NOT_NULLABLE}, inputFieldName);
        return {field->name};
    }

    auto varsizedFields = inputSchema.getFields()
        | std::views::filter([](const auto& candidate)
                             { return candidate.dataType.type == DataType::Type::VARSIZED && !candidate.dataType.nullable; })
        | std::views::transform([](const auto& candidate) { return candidate.name; }) | std::ranges::to<std::vector>();

    if (varsizedFields.size() < 2)
    {
        throw CannotInferSchema("Field '{}' not found in input schema", inputFieldName);
    }
    return inputFieldNames;
}

std::string getQualifierPrefix(const std::string& fieldName)
{
    const auto separatorPosition = fieldName.find(Schema::ATTRIBUTE_NAME_SEPARATOR);
    if (separatorPosition == std::string::npos)
    {
        return "";
    }
    return fieldName.substr(0, separatorPosition + std::string_view{Schema::ATTRIBUTE_NAME_SEPARATOR}.size());
}

std::string makePostJoinOutputFieldName(const std::string& payloadFieldName, const std::string& modelOutputFieldName)
{
    return getQualifierPrefix(payloadFieldName) + modelOutputFieldName;
}

Schema appendOrReplaceModelOutputField(Schema schema, const std::string& fieldName, const DataType& dataType)
{
    if (schema.getFieldByName(fieldName).has_value())
    {
        [[maybe_unused]] const bool replaced = schema.replaceTypeOfField(fieldName, dataType);
    }
    else
    {
        schema = schema.addField(fieldName, dataType);
    }
    return schema;
}

}

InferModelLogicalOperator::InferModelLogicalOperator(RegisteredModel model, std::vector<std::string> inputFieldNames)
    : model(std::move(model)), inputFieldNames(std::move(inputFieldNames))
{
}

/// NOLINTNEXTLINE(readability-convert-member-functions-to-static) — satisfies LogicalOperatorConcept, cannot be static
std::string_view InferModelLogicalOperator::getName() const noexcept
{
    return NAME;
}

const RegisteredModel& InferModelLogicalOperator::getModel() const
{
    return model;
}

std::vector<std::string> InferModelLogicalOperator::getInputFieldNames() const
{
    return inputFieldNames;
}

std::vector<std::string> InferModelLogicalOperator::getOutputFieldNames() const
{
    return model.getSchema().outputs.getFieldNames();
}

bool InferModelLogicalOperator::hasVarsizedInput() const
{
    const auto& inputs = model.getSchema().inputs;
    return inputs.getNumberOfFields() > 0 && inputs.getFieldAt(0).dataType.isType(DataType::Type::VARSIZED);
}

bool InferModelLogicalOperator::hasVarsizedOutput() const
{
    const auto& outputs = model.getSchema().outputs;
    return outputs.getNumberOfFields() > 0 && outputs.getFieldAt(0).dataType.isType(DataType::Type::VARSIZED);
}

bool InferModelLogicalOperator::operator==(const InferModelLogicalOperator& rhs) const
{
    return model == rhs.model && inputFieldNames == rhs.inputFieldNames && getOutputSchema() == rhs.getOutputSchema()
        && getInputSchemas() == rhs.getInputSchemas() && getTraitSet() == rhs.getTraitSet();
}

/// NOLINTNEXTLINE(readability-convert-member-functions-to-static) — satisfies LogicalOperatorConcept, cannot be static
std::string InferModelLogicalOperator::explain(ExplainVerbosity verbosity, OperatorId opId) const
{
    if (verbosity == ExplainVerbosity::Debug)
    {
        return fmt::format(
            "INFER_MODEL(opId: {}, inputFields: [{}], traitSet: {})", opId, fmt::join(inputFieldNames, ", "), traitSet.explain(verbosity));
    }
    return fmt::format("INFER_MODEL(inputFields: [{}])", fmt::join(inputFieldNames, ", "));
}

/// NOLINTNEXTLINE(readability-convert-member-functions-to-static) — satisfies LogicalOperatorConcept, cannot be static
InferModelLogicalOperator InferModelLogicalOperator::withInferredSchema(std::vector<Schema> inputSchemas) const
{
    auto copy = *this;
    if (inputSchemas.empty())
    {
        throw CannotInferSchema("InferModel requires at least one input schema");
    }
    copy.inputSchema = inputSchemas.at(0);

    const auto& modelInputs = model.getSchema().inputs;
    const auto& modelOutputs = model.getSchema().outputs;
    const auto hasSingleVarsizedModelInput
        = modelInputs.getNumberOfFields() == 1 && modelInputs.getFieldAt(0).dataType.isType(DataType::Type::VARSIZED);

    if (hasSingleVarsizedModelInput)
    {
        copy.inputFieldNames = resolveOrDeferSingleVarsizedInputFields(copy.inputSchema, inputFieldNames);
    }
    else
    {
        /// Check input field count matches model inputs
        if (inputFieldNames.size() != modelInputs.getNumberOfFields())
        {
            throw CannotInferSchema(
                "Model expects {} inputs, but {} input field names were provided", modelInputs.getNumberOfFields(), inputFieldNames.size());
        }

        /// Check type compatibility for each input field and resolve to its fully qualified
        /// name (`source$field`) so the runtime record lookup, which matches strictly, can find it.
        for (size_t i = 0; i < inputFieldNames.size(); ++i)
        {
            const auto& fieldName = inputFieldNames[i];
            const auto field = copy.inputSchema.getFieldByName(fieldName);
            if (!field.has_value())
            {
                throw CannotInferSchema("Field '{}' not found in input schema", fieldName);
            }
            validateModelInputField(field.value(), modelInputs.getFieldAt(i).dataType, fieldName);
            copy.inputFieldNames[i] = field->name;
        }
    }

    /// Build output schema: start from input schema, then append/replace model output fields
    copy.outputSchema = copy.inputSchema;
    if (hasSingleVarsizedModelInput && copy.inputFieldNames.size() > 1)
    {
        for (const auto& inputFieldName : copy.inputFieldNames)
        {
            for (const auto& field : modelOutputs.getFields())
            {
                copy.outputSchema = appendOrReplaceModelOutputField(
                    copy.outputSchema, makePostJoinOutputFieldName(inputFieldName, field.name), field.dataType);
            }
        }
        return copy;
    }

    for (const auto& field : modelOutputs.getFields())
    {
        copy.outputSchema = appendOrReplaceModelOutputField(copy.outputSchema, field.name, field.dataType);
    }
    return copy;
}

TraitSet InferModelLogicalOperator::getTraitSet() const
{
    return traitSet;
}

InferModelLogicalOperator InferModelLogicalOperator::withTraitSet(TraitSet newTraitSet) const
{
    auto copy = *this;
    copy.traitSet = std::move(newTraitSet);
    return copy;
}

InferModelLogicalOperator InferModelLogicalOperator::withChildren(std::vector<LogicalOperator> newChildren) const
{
    auto copy = *this;
    copy.children = std::move(newChildren);
    return copy;
}

std::vector<Schema> InferModelLogicalOperator::getInputSchemas() const
{
    return {inputSchema};
}

Schema InferModelLogicalOperator::getOutputSchema() const
{
    return outputSchema;
}

std::vector<LogicalOperator> InferModelLogicalOperator::getChildren() const
{
    return children;
}

Reflected Reflector<InferModelLogicalOperator>::operator()(const InferModelLogicalOperator& op) const
{
    return reflect(detail::ReflectedInferModelLogicalOperator{
        .model = std::make_optional(Reflector<RegisteredModel>{}(op.getModel())),
        .inputFieldNames = std::make_optional(op.getInputFieldNames())});
}

InferModelLogicalOperator Unreflector<InferModelLogicalOperator>::operator()(const Reflected& rfl) const
{
    auto reflected = unreflect<detail::ReflectedInferModelLogicalOperator>(rfl);

    if (!reflected.model.has_value() || !reflected.inputFieldNames.has_value())
    {
        throw NES::CannotDeserialize("Failed to deserialize InferModelLogicalOperator");
    }

    return InferModelLogicalOperator(Unreflector<RegisteredModel>{}(reflected.model.value()), std::move(reflected.inputFieldNames.value()));
}

/// generated registry interface requires by-value argument
LogicalOperatorRegistryReturnType
/// NOLINTNEXTLINE(performance-unnecessary-value-param)
LogicalOperatorGeneratedRegistrar::RegisterInferModelLogicalOperator(LogicalOperatorRegistryArguments arguments)
{
    if (!arguments.reflected.isEmpty())
    {
        return Unreflector<InferModelLogicalOperator>{}(arguments.reflected);
    }
    PRECONDITION(false, "Operator is only built directly or via reflection, not using the registry");
    std::unreachable();
}
}
