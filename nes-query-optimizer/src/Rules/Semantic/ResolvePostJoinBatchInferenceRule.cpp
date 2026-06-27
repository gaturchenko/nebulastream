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

#include <Rules/Semantic/ResolvePostJoinBatchInferenceRule.hpp>

#include <algorithm>
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
#include <Operators/SequenceLogicalOperator.hpp>
#include <Operators/Windows/JoinLogicalOperator.hpp>
#include <Plans/LogicalPlan.hpp>
#include <Rules/Semantic/OriginIdInferenceRule.hpp>
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

LogicalOperator getInferenceInputChild(const LogicalOperator& inferModelOperator)
{
    const auto children = inferModelOperator.getChildren();
    PRECONDITION(children.size() == 1, "Expected InferModelLogicalOperator to have exactly one child");

    const auto sequenceOperator = children.front().tryGetAs<SequenceLogicalOperator>();
    if (sequenceOperator.has_value()
        && sequenceOperator.value().get().getSequenceSource() == SequenceLogicalOperator::SequenceSource::INFERENCE)
    {
        const auto sequenceChildren = sequenceOperator.value().get().getChildren();
        PRECONDITION(sequenceChildren.size() == 1, "Expected inference SequenceLogicalOperator to have exactly one child");
        return sequenceChildren.front();
    }
    return children.front();
}

bool hasSingleVarsizedModelInput(const InferModelLogicalOperator& inferModelOp)
{
    const auto& inputs = inferModelOp.getModel().getSchema().inputs;
    return inputs.getNumberOfFields() == 1 && inputs.getFieldAt(0).dataType.isType(DataType::Type::VARSIZED);
}

std::vector<std::string> resolveJoinedPayloadFields(const Schema& inputSchema, const std::string& modelInputFieldName)
{
    const auto requestedFieldIsQualified = modelInputFieldName.find(Schema::ATTRIBUTE_NAME_SEPARATOR) != std::string::npos;
    if (requestedFieldIsQualified)
    {
        return {};
    }

    std::vector<std::string> matches;
    for (const auto& field : inputSchema.getFields())
    {
        if (field.getUnqualifiedName() == modelInputFieldName && field.dataType.type == DataType::Type::VARSIZED
            && !field.dataType.nullable)
        {
            matches.push_back(field.name);
        }
    }
    return matches;
}

InferModelLogicalOperator withInputFieldNames(const InferModelLogicalOperator& op, std::vector<std::string> inputFieldNames)
{
    auto rewritten = InferModelLogicalOperator(op.getModel(), std::move(inputFieldNames));
    rewritten = rewritten.withChildren(op.getChildren());
    rewritten = rewritten.withTraitSet(op.getTraitSet());
    rewritten = rewritten.withInferredSchema(op.getInputSchemas());
    return rewritten;
}

}

LogicalPlan ResolvePostJoinBatchInferenceRule::apply(LogicalPlan queryPlan) const
{
    for (const auto& inferModelOp : getOperatorByType<InferModelLogicalOperator>(queryPlan))
    {
        const auto& op = inferModelOp.get();
        if (!hasSingleVarsizedModelInput(op) || op.getInputFieldNames().size() != 1)
        {
            continue;
        }
        if (!containsJoin(getInferenceInputChild(inferModelOp)))
        {
            continue;
        }

        const auto payloadFields = resolveJoinedPayloadFields(op.getInputSchemas().at(0), op.getInputFieldNames().front());
        if (payloadFields.size() <= 1)
        {
            continue;
        }

        auto replacement = LogicalOperator{withInputFieldNames(op, payloadFields)};
        auto replaceResult = replaceSubtree(queryPlan, inferModelOp.getId(), replacement);
        INVARIANT(replaceResult.has_value(), "Failed to resolve post-join batch inference input fields");
        queryPlan = std::move(replaceResult.value());
    }
    return queryPlan;
}

const std::type_info& ResolvePostJoinBatchInferenceRule::getType()
{
    return typeid(ResolvePostJoinBatchInferenceRule);
}

std::string_view ResolvePostJoinBatchInferenceRule::getName()
{
    return NAME;
}

std::set<std::type_index> ResolvePostJoinBatchInferenceRule::dependsOn() const
{
    return {typeid(TypeInferenceRule)};
}

std::set<std::type_index> ResolvePostJoinBatchInferenceRule::requiredBy() const
{
    return {typeid(OriginIdInferenceRule)};
}

bool ResolvePostJoinBatchInferenceRule::operator==(const ResolvePostJoinBatchInferenceRule&) const
{
    return true;
}

}
