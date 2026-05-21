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

#include <Rules/Semantic/InsertSequenceForWindowAggregationRule.hpp>

#include <algorithm>
#include <set>
#include <string_view>
#include <typeindex>
#include <typeinfo>

#include <Operators/LogicalOperator.hpp>
#include <Operators/SequenceLogicalOperator.hpp>
#include <Operators/Windows/WindowedAggregationLogicalOperator.hpp>
#include <Plans/LogicalPlan.hpp>
#include <Rules/Semantic/LogicalSourceExpansionRule.hpp>
#include <Rules/Semantic/OriginIdInferenceRule.hpp>
#include <Rules/Semantic/TypeInferenceRule.hpp>
#include <ErrorHandling.hpp>

namespace NES
{

namespace
{
bool requiresSequentialAggregation(const WindowedAggregationLogicalOperator& aggregation)
{
    const auto aggregationFunctions = aggregation.getWindowAggregation();
    return std::ranges::any_of(aggregationFunctions, [](const auto& function) { return function->requiresSequentialAggregation(); });
}
}

LogicalPlan InsertSequenceForWindowAggregationRule::apply(LogicalPlan queryPlan) const
{
    for (const auto& aggregationOp : getOperatorByType<WindowedAggregationLogicalOperator>(queryPlan))
    {
        if (!requiresSequentialAggregation(aggregationOp.get()))
        {
            continue;
        }

        const auto children = aggregationOp.getChildren();
        PRECONDITION(children.size() == 1, "WindowedAggregationLogicalOperator should have exactly one child before sequence insertion");
        const auto sequenceOperator = children.front().tryGetAs<SequenceLogicalOperator>();
        if (sequenceOperator.has_value()
            && sequenceOperator.value().get().getSequenceSource() == SequenceLogicalOperator::SequenceSource::AGGREGATION)
        {
            continue;
        }

        auto sequence
            = LogicalOperator{SequenceLogicalOperator(SequenceLogicalOperator::SequenceSource::AGGREGATION).withChildren(children)};
        auto replacement = LogicalOperator{aggregationOp.get().withChildren({std::move(sequence)})};
        auto replaceResult = replaceSubtree(queryPlan, aggregationOp.getId(), replacement);
        INVARIANT(replaceResult.has_value(), "Failed to insert SequenceLogicalOperator below WindowedAggregationLogicalOperator");
        queryPlan = std::move(replaceResult.value());
    }
    return queryPlan;
}

const std::type_info& InsertSequenceForWindowAggregationRule::getType()
{
    return typeid(InsertSequenceForWindowAggregationRule);
}

std::string_view InsertSequenceForWindowAggregationRule::getName()
{
    return NAME;
}

std::set<std::type_index> InsertSequenceForWindowAggregationRule::dependsOn() const
{
    return {typeid(LogicalSourceExpansionRule)};
}

std::set<std::type_index> InsertSequenceForWindowAggregationRule::requiredBy() const
{
    return {typeid(TypeInferenceRule), typeid(OriginIdInferenceRule)};
}

bool InsertSequenceForWindowAggregationRule::operator==(const InsertSequenceForWindowAggregationRule&) const
{
    return true;
}

}
