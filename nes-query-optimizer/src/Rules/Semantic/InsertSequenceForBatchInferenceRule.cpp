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

#include <Rules/Semantic/InsertSequenceForBatchInferenceRule.hpp>

#include <set>
#include <string_view>
#include <typeindex>
#include <typeinfo>

#include <Operators/InferModelLogicalOperator.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Operators/SequenceLogicalOperator.hpp>
#include <Plans/LogicalPlan.hpp>
#include <Rules/Semantic/InferModelResolutionRule.hpp>
#include <Rules/Semantic/OriginIdInferenceRule.hpp>
#include <Rules/Semantic/TypeInferenceRule.hpp>
#include <ErrorHandling.hpp>

namespace NES
{

LogicalPlan InsertSequenceForBatchInferenceRule::apply(LogicalPlan queryPlan) const
{
    if (batchSize <= 1)
    {
        return queryPlan;
    }

    for (const auto& inferModelOp : getOperatorByType<InferModelLogicalOperator>(queryPlan))
    {
        const auto children = inferModelOp.getChildren();
        PRECONDITION(children.size() == 1, "InferModelLogicalOperator should have exactly one child before sequence insertion");
        if (children.front().tryGetAs<SequenceLogicalOperator>().has_value())
        {
            continue;
        }

        auto sequence = LogicalOperator{
            SequenceLogicalOperator(SequenceLogicalOperator::SequenceSource::INFERENCE, batchSize).withChildren(children)};
        auto replacement = LogicalOperator{inferModelOp.get().withChildren({std::move(sequence)})};
        auto replaceResult = replaceSubtree(queryPlan, inferModelOp.getId(), replacement);
        INVARIANT(replaceResult.has_value(), "Failed to insert SequenceLogicalOperator below InferModelLogicalOperator");
        queryPlan = std::move(replaceResult.value());
    }
    return queryPlan;
}

const std::type_info& InsertSequenceForBatchInferenceRule::getType()
{
    return typeid(InsertSequenceForBatchInferenceRule);
}

std::string_view InsertSequenceForBatchInferenceRule::getName()
{
    return NAME;
}

std::set<std::type_index> InsertSequenceForBatchInferenceRule::dependsOn() const
{
    return {typeid(InferModelResolutionRule)};
}

std::set<std::type_index> InsertSequenceForBatchInferenceRule::requiredBy() const
{
    return {typeid(TypeInferenceRule), typeid(OriginIdInferenceRule)};
}

bool InsertSequenceForBatchInferenceRule::operator==(const InsertSequenceForBatchInferenceRule& other) const
{
    return batchSize == other.batchSize;
}

}
