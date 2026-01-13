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

#include <RewriteRules/LowerToPhysical/LowerToPhysicalSequence.hpp>

#include <memory>
#include <InputFormatters/InputFormatterTupleBufferRefProvider.hpp>
#include <MemoryLayout/RowLayout.hpp>
#include <Nautilus/Interface/BufferRef/RowTupleBufferRef.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Operators/SequenceLogicalOperator.hpp>
#include <Operators/Sinks/SinkLogicalOperator.hpp>
#include <RewriteRules/AbstractRewriteRule.hpp>
#include <ErrorHandling.hpp>
#include <PhysicalOperator.hpp>
#include <RewriteRuleRegistry.hpp>
#include <SequenceOperatorHandler.hpp>
#include <SequencePhysicalOperator.hpp>
#include <Operators/Sources/SourceDescriptorLogicalOperator.hpp>

namespace NES
{

RewriteRuleResultSubgraph LowerToPhysicalSequence::apply(LogicalOperator logicalOperator)
{
    PRECONDITION(logicalOperator.tryGetAs<SequenceLogicalOperator>(), "Expected a SequenceLogicalOperator");
    auto sequence = logicalOperator.getAs<SequenceLogicalOperator>();

    const auto schema = logicalOperator.getInputSchemas().at(0);
    auto memoryProvider = TupleBufferRef::create(conf.operatorBufferSize.getValue(), schema);

    if (sequence->getSequenceSource() == SequenceLogicalOperator::SequenceSource::INFERENCE && conf.inferenceConfiguration.batchSize.getValue() == 1)
    {
        if (sequence.getChildren().at(0).tryGetAs<SourceDescriptorLogicalOperator>().has_value())
        {
            const auto source = sequence.getChildren().at(0).getAs<SourceDescriptorLogicalOperator>();
            const auto inputFormatterConfig = source->getSourceDescriptor().getParserConfig();
            if (toUpperCase(inputFormatterConfig.parserType) != "NATIVE")
            {
                auto memoryProviderFormatter = TupleBufferRef::create(conf.operatorBufferSize.getValue(), schema);
                memoryProvider = provideInputFormatterTupleBufferRef(inputFormatterConfig, memoryProviderFormatter);
            }
        }
        auto physicalOperator = ScanPhysicalOperator(memoryProvider);

        auto wrapper = std::make_shared<PhysicalOperatorWrapper>(
            physicalOperator,
            sequence.getInputSchemas()[0],
            sequence.getOutputSchema(),
            PhysicalOperatorWrapper::PipelineLocation::SCAN);

        return {.root = wrapper, .leafs = {wrapper}};
    }

    auto operatorHandlerId = getNextOperatorHandlerId();
    auto handler = std::make_shared<Runtime::Execution::Operators::SequenceOperatorHandler>();

    if (sequence.getChildren().at(0).tryGetAs<SourceDescriptorLogicalOperator>().has_value())
    {
        const auto source = sequence.getChildren().at(0).getAs<SourceDescriptorLogicalOperator>();
        const auto inputFormatterConfig = source->getSourceDescriptor().getParserConfig();
        if (toUpperCase(inputFormatterConfig.parserType) != "NATIVE")
        {
            auto memoryProviderFormatter = TupleBufferRef::create(conf.operatorBufferSize.getValue(), schema);
            memoryProvider = provideInputFormatterTupleBufferRef(inputFormatterConfig, memoryProviderFormatter);
        }
    }

    auto physicalOperator = Runtime::Execution::Operators::SequencePhysicalOperator(
            operatorHandlerId, ScanPhysicalOperator(memoryProvider));

    auto wrapper = std::make_shared<PhysicalOperatorWrapper>(
        physicalOperator,
        sequence.getInputSchemas()[0],
        sequence.getOutputSchema(),
        operatorHandlerId,
        handler,
        PhysicalOperatorWrapper::PipelineLocation::SCAN);

    return {.root = wrapper, .leafs = {wrapper}};
}

std::unique_ptr<AbstractRewriteRule>
RewriteRuleGeneratedRegistrar::RegisterSequenceRewriteRule(RewriteRuleRegistryArguments argument) /// NOLINT
{
    return std::make_unique<LowerToPhysicalSequence>(argument.conf);
}
}
