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

#include <LoweringRules/LowerToPhysical/LowerToPhysicalSequence.hpp>

#include <memory>
#include <ranges>
#include <vector>

#include <LoweringRules/AbstractLoweringRule.hpp>
#include <InputFormatterTupleBufferRefProvider.hpp>
#include <Nautilus/Interface/BufferRef/LowerSchemaProvider.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Operators/SequenceLogicalOperator.hpp>
#include <Operators/Sources/SourceDescriptorLogicalOperator.hpp>
#include <PhysicalOperator.hpp>
#include <QueryExecutionConfiguration.hpp>
#include <Runtime/Execution/OperatorHandler.hpp>
#include <SequenceOperatorHandler.hpp>
#include <SequencePhysicalOperator.hpp>
#include <Traits/MemoryLayoutTypeTrait.hpp>
#include <Traits/TraitSet.hpp>
#include <Util/Strings.hpp>
#include <ErrorHandling.hpp>
#include <LoweringRuleRegistry.hpp>

namespace NES
{

namespace
{

ScanPhysicalOperator createScanOperator(
    const LogicalOperator& sequenceOperator,
    const uint64_t bufferSize,
    const Schema& inputSchema,
    const MemoryLayoutType memoryLayoutType)
{
    const auto sourceOperators
        = sequenceOperator.getChildren()
        | std::views::filter([](const auto& childOperator)
                             { return childOperator.template tryGetAs<SourceDescriptorLogicalOperator>().has_value(); })
        | std::views::transform(
              [](const auto& sourceChildOperator)
              { return sourceChildOperator.template tryGetAs<SourceDescriptorLogicalOperator>().value()->getSourceDescriptor(); })
        | std::ranges::to<std::vector>();
    PRECONDITION(sourceOperators.size() < 2, "Sequence operator should have at most one source operator as a child");

    auto bufferRef = LowerSchemaProvider::lowerSchema(bufferSize, inputSchema, memoryLayoutType);
    if (sourceOperators.size() == 1)
    {
        const auto inputFormatterConfig = sourceOperators.front().getParserConfig();
        if (toUpperCase(inputFormatterConfig.parserType) != "NATIVE")
        {
            bufferRef = provideInputFormatterTupleBufferRef(inputFormatterConfig, bufferRef);
        }
    }
    return ScanPhysicalOperator(bufferRef, inputSchema.getFieldNames());
}

}

LoweringRuleResultSubgraph LowerToPhysicalSequence::apply(LogicalOperator logicalOperator)
{
    PRECONDITION(logicalOperator.tryGetAs<SequenceLogicalOperator>(), "Expected a SequenceLogicalOperator");

    const auto inputSchema = logicalOperator.getInputSchemas().at(0);
    const auto memoryLayoutTypeTrait = logicalOperator.getTraitSet().tryGet<MemoryLayoutTypeTrait>();
    PRECONDITION(memoryLayoutTypeTrait.has_value(), "Expected a memory layout type trait");
    const auto memoryLayoutType = memoryLayoutTypeTrait.value()->memoryLayout;

    const auto operatorHandlerId = getNextOperatorHandlerId();
    auto handler = std::make_shared<SequenceOperatorHandler>();

    auto physicalOperator = SequencePhysicalOperator(
        operatorHandlerId, createScanOperator(logicalOperator, conf.operatorBufferSize.getValue(), inputSchema, memoryLayoutType));

    auto wrapper = std::make_shared<PhysicalOperatorWrapper>(
        physicalOperator,
        inputSchema,
        logicalOperator.getOutputSchema(),
        memoryLayoutType,
        memoryLayoutType,
        operatorHandlerId,
        handler,
        PhysicalOperatorWrapper::PipelineLocation::SCAN);

    return {.root = wrapper, .leafs = {wrapper}};
}

LoweringRuleRegistryReturnType
LoweringRuleGeneratedRegistrar::RegisterSequenceLoweringRule(LoweringRuleRegistryArguments argument) /// NOLINT
{
    return std::make_unique<LowerToPhysicalSequence>(argument.conf);
}

}
