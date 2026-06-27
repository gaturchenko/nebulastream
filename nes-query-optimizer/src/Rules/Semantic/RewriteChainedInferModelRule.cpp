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

#include <Rules/Semantic/RewriteChainedInferModelRule.hpp>

#include <algorithm>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <string_view>
#include <typeindex>
#include <typeinfo>
#include <vector>

#include <DataTypes/DataType.hpp>
#include <DataTypes/Schema.hpp>
#include <Operators/InferModelLogicalOperator.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Operators/Windows/JoinLogicalOperator.hpp>
#include <Plans/LogicalPlan.hpp>
#include <Rules/Semantic/InferModelResolutionRule.hpp>
#include <Rules/Semantic/InsertSequenceForBatchInferenceRule.hpp>
#include <Rules/Semantic/TypeInferenceRule.hpp>
#include <ErrorHandling.hpp>

namespace NES
{

namespace
{

bool containsJoin(const LogicalOperator& logicalOperator)
{
    if (logicalOperator.tryGetAs<JoinLogicalOperator>().has_value())
    {
        return true;
    }
    return std::ranges::any_of(logicalOperator.getChildren(), [](const auto& child) { return containsJoin(child); });
}

bool hasSingleVarsizedModelInput(const InferModelLogicalOperator& inferModelOp)
{
    const auto& inputs = inferModelOp.getModel().getSchema().inputs;
    return inputs.getNumberOfFields() == 1 && inputs.getFieldAt(0).dataType.isType(DataType::Type::VARSIZED);
}

bool isCompatibleChainedInference(const InferModelLogicalOperator& candidate, const InferModelLogicalOperator& first)
{
    return candidate.getModel() == first.getModel() && hasSingleVarsizedModelInput(candidate)
        && candidate.getInputFieldNames().size() == 1;
}

std::vector<std::string> makeCollapsedInputFieldNames(std::vector<std::string> chainInputFieldNames)
{
    PRECONDITION(!chainInputFieldNames.empty(), "Expected at least one chained inference input field name");
    const auto allInputsEqual = std::ranges::all_of(
        chainInputFieldNames, [&](const auto& inputFieldName) { return inputFieldName == chainInputFieldNames.front(); });
    if (allInputsEqual)
    {
        return {chainInputFieldNames.front()};
    }

    /// Inner inference executes first in the original chain, so keep that order for explicit field lists.
    std::ranges::reverse(chainInputFieldNames);
    std::vector<std::string> uniqueInputFieldNames;
    uniqueInputFieldNames.reserve(chainInputFieldNames.size());
    for (const auto& inputFieldName : chainInputFieldNames)
    {
        if (!std::ranges::contains(uniqueInputFieldNames, inputFieldName))
        {
            uniqueInputFieldNames.push_back(inputFieldName);
        }
    }
    return uniqueInputFieldNames;
}

std::optional<LogicalOperator> rewriteChainedInference(const LogicalOperator& logicalOperator)
{
    const auto firstInferModelOp = logicalOperator.tryGetAs<InferModelLogicalOperator>();
    if (!firstInferModelOp.has_value() || !hasSingleVarsizedModelInput(firstInferModelOp.value().get())
        || firstInferModelOp.value().get().getInputFieldNames().size() != 1)
    {
        return std::nullopt;
    }

    std::vector<std::string> chainInputFieldNames;
    chainInputFieldNames.push_back(firstInferModelOp.value().get().getInputFieldNames().front());

    auto current = logicalOperator;
    auto children = current.getChildren();
    while (children.size() == 1)
    {
        const auto childInferModelOp = children.front().tryGetAs<InferModelLogicalOperator>();
        if (!childInferModelOp.has_value() || !isCompatibleChainedInference(childInferModelOp.value().get(), firstInferModelOp.value().get()))
        {
            break;
        }

        chainInputFieldNames.push_back(childInferModelOp.value().get().getInputFieldNames().front());
        current = children.front();
        children = current.getChildren();
    }

    if (chainInputFieldNames.size() <= 1 || children.size() != 1 || !containsJoin(children.front()))
    {
        return std::nullopt;
    }

    auto rewritten = InferModelLogicalOperator(
        firstInferModelOp.value().get().getModel(), makeCollapsedInputFieldNames(std::move(chainInputFieldNames)));
    rewritten = rewritten.withTraitSet(firstInferModelOp.value().get().getTraitSet());
    rewritten = rewritten.withChildren({children.front()});
    return LogicalOperator{rewritten};
}

}

LogicalPlan RewriteChainedInferModelRule::apply(LogicalPlan queryPlan) const
{
    for (const auto& inferModelOp : getOperatorByType<InferModelLogicalOperator>(queryPlan))
    {
        const auto replacement = rewriteChainedInference(inferModelOp);
        if (!replacement.has_value())
        {
            continue;
        }

        auto replaceResult = replaceSubtree(queryPlan, inferModelOp.getId(), replacement.value());
        if (replaceResult.has_value())
        {
            queryPlan = std::move(replaceResult.value());
        }
    }
    return queryPlan;
}

const std::type_info& RewriteChainedInferModelRule::getType()
{
    return typeid(RewriteChainedInferModelRule);
}

std::string_view RewriteChainedInferModelRule::getName()
{
    return NAME;
}

std::set<std::type_index> RewriteChainedInferModelRule::dependsOn() const
{
    return {typeid(InferModelResolutionRule)};
}

std::set<std::type_index> RewriteChainedInferModelRule::requiredBy() const
{
    return {typeid(InsertSequenceForBatchInferenceRule), typeid(TypeInferenceRule)};
}

bool RewriteChainedInferModelRule::operator==(const RewriteChainedInferModelRule&) const
{
    return true;
}

}
