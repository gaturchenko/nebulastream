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

#include <Rules/Semantic/ResolveChainedPostJoinInferenceRule.hpp>

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
#include <Operators/SequenceLogicalOperator.hpp>
#include <Operators/Windows/JoinLogicalOperator.hpp>
#include <Plans/LogicalPlan.hpp>
#include <Rules/Semantic/ResolvePostJoinBatchInferenceRule.hpp>
#include <Rules/Semantic/TypeInferenceRule.hpp>
#include <ErrorHandling.hpp>

namespace NES
{

namespace
{

struct ChainEntry
{
    InferModelLogicalOperator inferModel;
    std::optional<SequenceLogicalOperator> sequence;
};

struct ChainMatch
{
    std::vector<ChainEntry> entriesOuterToInner;
    LogicalOperator input;
};

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

std::optional<std::pair<LogicalOperator, std::optional<SequenceLogicalOperator>>> getInferenceInput(const LogicalOperator& logicalOperator)
{
    const auto children = logicalOperator.getChildren();
    if (children.size() != 1)
    {
        return std::nullopt;
    }

    const auto sequenceOperator = children.front().tryGetAs<SequenceLogicalOperator>();
    if (sequenceOperator.has_value()
        && sequenceOperator.value().get().getSequenceSource() == SequenceLogicalOperator::SequenceSource::INFERENCE)
    {
        const auto sequenceChildren = sequenceOperator.value().get().getChildren();
        if (sequenceChildren.size() != 1)
        {
            return std::nullopt;
        }
        return std::pair{sequenceChildren.front(), std::make_optional(sequenceOperator.value().get())};
    }
    return std::pair{children.front(), std::optional<SequenceLogicalOperator>{}};
}

std::optional<ChainMatch> collectCompatibleChain(const LogicalOperator& logicalOperator)
{
    const auto firstInferModel = logicalOperator.tryGetAs<InferModelLogicalOperator>();
    if (!firstInferModel.has_value() || !hasSingleVarsizedModelInput(firstInferModel.value().get())
        || firstInferModel.value().get().getInputFieldNames().size() != 1)
    {
        return std::nullopt;
    }

    ChainMatch match{{}, logicalOperator};
    auto current = logicalOperator;
    while (true)
    {
        const auto inferModel = current.tryGetAs<InferModelLogicalOperator>();
        if (!inferModel.has_value() || !isCompatibleChainedInference(inferModel.value().get(), firstInferModel.value().get()))
        {
            return std::nullopt;
        }

        auto input = getInferenceInput(current);
        if (!input.has_value())
        {
            return std::nullopt;
        }
        match.entriesOuterToInner.push_back(ChainEntry{.inferModel = inferModel.value().get(), .sequence = input->second});

        const auto childInferModel = input->first.tryGetAs<InferModelLogicalOperator>();
        if (!childInferModel.has_value()
            || !isCompatibleChainedInference(childInferModel.value().get(), firstInferModel.value().get()))
        {
            match.input = input->first;
            return match;
        }
        current = input->first;
    }
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

LogicalOperator rebuildInference(const ChainEntry& entry, LogicalOperator child, const std::string& inputFieldName)
{
    auto inferModel = InferModelLogicalOperator(entry.inferModel.getModel(), {inputFieldName});
    inferModel = inferModel.withTraitSet(entry.inferModel.getTraitSet());

    if (entry.sequence.has_value())
    {
        auto sequence = entry.sequence.value().withChildren({std::move(child)});
        sequence = sequence.withInferredSchema({sequence.getChildren().front().getOutputSchema()});
        inferModel = inferModel.withChildren({LogicalOperator{sequence}});
        inferModel = inferModel.withInferredSchema({sequence.getOutputSchema()});
        return LogicalOperator{inferModel};
    }

    inferModel = inferModel.withChildren({std::move(child)});
    inferModel = inferModel.withInferredSchema({inferModel.getChildren().front().getOutputSchema()});
    return LogicalOperator{inferModel};
}

std::optional<LogicalOperator> resolveChainedInference(const LogicalOperator& logicalOperator)
{
    auto chain = collectCompatibleChain(logicalOperator);
    if (!chain.has_value() || chain->entriesOuterToInner.size() <= 1 || !containsJoin(chain->input))
    {
        return std::nullopt;
    }

    const auto modelInputFieldName = chain->entriesOuterToInner.front().inferModel.getInputFieldNames().front();
    const auto payloadFields = resolveJoinedPayloadFields(chain->input.getOutputSchema(), modelInputFieldName);
    if (payloadFields.size() < chain->entriesOuterToInner.size())
    {
        return std::nullopt;
    }

    auto rebuilt = chain->input;
    for (size_t chainIndex = chain->entriesOuterToInner.size(); chainIndex > 0; --chainIndex)
    {
        const auto innerToOuterIndex = chain->entriesOuterToInner.size() - chainIndex;
        rebuilt = rebuildInference(chain->entriesOuterToInner.at(chainIndex - 1), std::move(rebuilt), payloadFields.at(innerToOuterIndex));
    }
    return rebuilt;
}

}

LogicalPlan ResolveChainedPostJoinInferenceRule::apply(LogicalPlan queryPlan) const
{
    bool changed = false;
    for (const auto& inferModelOp : getOperatorByType<InferModelLogicalOperator>(queryPlan))
    {
        const auto replacement = resolveChainedInference(inferModelOp);
        if (!replacement.has_value())
        {
            continue;
        }

        auto replaceResult = replaceSubtree(queryPlan, inferModelOp.getId(), replacement.value());
        if (replaceResult.has_value())
        {
            queryPlan = std::move(replaceResult.value());
            changed = true;
        }
    }

    if (changed)
    {
        queryPlan = TypeInferenceRule{}.apply(queryPlan);
    }
    return queryPlan;
}

const std::type_info& ResolveChainedPostJoinInferenceRule::getType()
{
    return typeid(ResolveChainedPostJoinInferenceRule);
}

std::string_view ResolveChainedPostJoinInferenceRule::getName()
{
    return NAME;
}

std::set<std::type_index> ResolveChainedPostJoinInferenceRule::dependsOn() const
{
    return {typeid(TypeInferenceRule)};
}

std::set<std::type_index> ResolveChainedPostJoinInferenceRule::requiredBy() const
{
    return {typeid(ResolvePostJoinBatchInferenceRule)};
}

bool ResolveChainedPostJoinInferenceRule::operator==(const ResolveChainedPostJoinInferenceRule&) const
{
    return true;
}

}
