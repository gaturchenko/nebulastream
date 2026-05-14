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

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <DataTypes/Schema.hpp>
#include <Identifiers/Identifiers.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Traits/TraitSet.hpp>
#include <Util/PlanRenderer.hpp>
#include <Util/Reflection.hpp>

namespace NES
{

class SequenceLogicalOperator
{
public:
    enum class SequenceSource : uint8_t
    {
        AGGREGATION,
        INFERENCE
    };

    explicit SequenceLogicalOperator(SequenceSource source, uint64_t batchSize = 1);

    [[nodiscard]] SequenceSource getSequenceSource() const;
    [[nodiscard]] uint64_t getBatchSize() const;

    [[nodiscard]] bool operator==(const SequenceLogicalOperator& rhs) const;

    [[nodiscard]] SequenceLogicalOperator withTraitSet(TraitSet traitSet) const;
    [[nodiscard]] TraitSet getTraitSet() const;

    [[nodiscard]] SequenceLogicalOperator withChildren(std::vector<LogicalOperator> children) const;
    [[nodiscard]] std::vector<LogicalOperator> getChildren() const;

    [[nodiscard]] std::vector<Schema> getInputSchemas() const;
    [[nodiscard]] Schema getOutputSchema() const;

    [[nodiscard]] std::string explain(ExplainVerbosity verbosity, OperatorId opId) const;
    [[nodiscard]] std::string_view getName() const noexcept;

    [[nodiscard]] SequenceLogicalOperator withInferredSchema(std::vector<Schema> inputSchemas) const;

private:
    static constexpr std::string_view NAME = "Sequence";

    SequenceSource source;
    uint64_t batchSize;
    std::vector<LogicalOperator> children;
    Schema inputSchema;
    Schema outputSchema;
    TraitSet traitSet;
};

template <>
struct Reflector<SequenceLogicalOperator>
{
    Reflected operator()(const SequenceLogicalOperator& op) const;
};

template <>
struct Unreflector<SequenceLogicalOperator>
{
    SequenceLogicalOperator operator()(const Reflected& rfl) const;
};

static_assert(LogicalOperatorConcept<SequenceLogicalOperator>);

}

namespace NES::detail
{
struct ReflectedSequenceLogicalOperator
{
    std::optional<uint64_t> source;
    std::optional<uint64_t> batchSize;
};
}
