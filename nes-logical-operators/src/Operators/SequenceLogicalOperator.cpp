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

#include <Operators/SequenceLogicalOperator.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <DataTypes/Schema.hpp>
#include <Identifiers/Identifiers.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Traits/TraitSet.hpp>
#include <Util/PlanRenderer.hpp>
#include <Util/Reflection.hpp>
#include <ErrorHandling.hpp>
#include <LogicalOperatorRegistry.hpp>

namespace NES
{

SequenceLogicalOperator::SequenceLogicalOperator(SequenceSource source, uint64_t batchSize)
    : source(source), batchSize(batchSize)
{
    PRECONDITION(batchSize > 0, "Sequence batch size must be larger than zero");
}

std::string_view SequenceLogicalOperator::getName() const noexcept
{
    return NAME;
}

SequenceLogicalOperator::SequenceSource SequenceLogicalOperator::getSequenceSource() const
{
    return source;
}

uint64_t SequenceLogicalOperator::getBatchSize() const
{
    return batchSize;
}

bool SequenceLogicalOperator::operator==(const SequenceLogicalOperator& rhs) const
{
    return source == rhs.source && batchSize == rhs.batchSize && getOutputSchema() == rhs.getOutputSchema()
        && getInputSchemas() == rhs.getInputSchemas() && getTraitSet() == rhs.getTraitSet();
}

std::string SequenceLogicalOperator::explain(ExplainVerbosity verbosity, OperatorId opId) const
{
    if (verbosity == ExplainVerbosity::Debug)
    {
        return fmt::format(
            "SEQUENCE(opId: {}, source: {}, batchSize: {}, traitSet: {})",
            opId,
            static_cast<uint8_t>(source),
            batchSize,
            traitSet.explain(verbosity));
    }
    return "SEQUENCE";
}

SequenceLogicalOperator SequenceLogicalOperator::withInferredSchema(std::vector<Schema> inputSchemas) const
{
    if (inputSchemas.empty())
    {
        throw CannotInferSchema("Sequence requires at least one input schema");
    }

    const auto& firstSchema = inputSchemas.front();
    for (const auto& schema : inputSchemas)
    {
        if (schema != firstSchema)
        {
            throw CannotInferSchema("All input schemas must be equal for Sequence operator");
        }
    }

    auto copy = *this;
    copy.inputSchema = firstSchema;
    copy.outputSchema = firstSchema;
    return copy;
}

TraitSet SequenceLogicalOperator::getTraitSet() const
{
    return traitSet;
}

SequenceLogicalOperator SequenceLogicalOperator::withTraitSet(TraitSet newTraitSet) const
{
    auto copy = *this;
    copy.traitSet = std::move(newTraitSet);
    return copy;
}

SequenceLogicalOperator SequenceLogicalOperator::withChildren(std::vector<LogicalOperator> newChildren) const
{
    auto copy = *this;
    copy.children = std::move(newChildren);
    return copy;
}

std::vector<Schema> SequenceLogicalOperator::getInputSchemas() const
{
    return {inputSchema};
}

Schema SequenceLogicalOperator::getOutputSchema() const
{
    return outputSchema;
}

std::vector<LogicalOperator> SequenceLogicalOperator::getChildren() const
{
    return children;
}

Reflected Reflector<SequenceLogicalOperator>::operator()(const SequenceLogicalOperator& op) const
{
    return reflect(detail::ReflectedSequenceLogicalOperator{
        .source = std::make_optional(static_cast<uint64_t>(op.getSequenceSource())),
        .batchSize = std::make_optional(op.getBatchSize())});
}

SequenceLogicalOperator Unreflector<SequenceLogicalOperator>::operator()(const Reflected& rfl) const
{
    auto reflected = unreflect<detail::ReflectedSequenceLogicalOperator>(rfl);
    if (!reflected.source.has_value())
    {
        throw CannotDeserialize("Failed to deserialize SequenceLogicalOperator");
    }
    return SequenceLogicalOperator(
        static_cast<SequenceLogicalOperator::SequenceSource>(reflected.source.value()), reflected.batchSize.value_or(1));
}

LogicalOperatorRegistryReturnType
/// NOLINTNEXTLINE(performance-unnecessary-value-param)
LogicalOperatorGeneratedRegistrar::RegisterSequenceLogicalOperator(LogicalOperatorRegistryArguments arguments)
{
    if (!arguments.reflected.isEmpty())
    {
        return Unreflector<SequenceLogicalOperator>{}(arguments.reflected);
    }
    PRECONDITION(false, "SequenceLogicalOperator is only built directly or via reflection");
    std::unreachable();
}

}
