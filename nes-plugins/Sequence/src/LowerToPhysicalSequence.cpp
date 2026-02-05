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

#include <memory>
#include <InputFormatters/InputFormatterTupleBufferRefProvider.hpp>
#include <MemoryLayout/RowLayout.hpp>
#include <Nautilus/Interface/BufferRef/RowTupleBufferRef.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Operators/SequenceLogicalOperator.hpp>
#include <Operators/Sources/SourceDescriptorLogicalOperator.hpp>
#include <Operators/Windows/WindowedAggregationLogicalOperator.hpp>
#include <RewriteRules/AbstractRewriteRule.hpp>
#include <SequencePhysicalOperator.hpp>
#include <InterBufferBatchingOperatorHandler.hpp>
#include <InterBufferBatchingOperator.hpp>
#include <ErrorHandling.hpp>
#include <PhysicalOperator.hpp>
#include <RewriteRuleRegistry.hpp>
#include <SequenceOperatorHandler.hpp>

struct LowerToPhysicalSequence : NES::AbstractRewriteRule
{
    bool findAggregationRecursively(const NES::LogicalOperator& child)
    {
        const auto children = child.getChildren();

        if (children.empty())
        {
            return false;
        }

        const auto& firstChild = children.at(0);

        if (firstChild.tryGetAs<NES::WindowedAggregationLogicalOperator>())
        {
            return true;
        }

        return findAggregationRecursively(firstChild);
    }

    explicit LowerToPhysicalSequence(NES::QueryExecutionConfiguration conf) : conf(std::move(conf)) { }
    NES::RewriteRuleResultSubgraph apply(NES::LogicalOperator logicalOperator)
    {
        PRECONDITION(logicalOperator.tryGetAs<NES::SequenceLogicalOperator>(), "Expected a SequenceLogicalOperator");
        auto sequence = logicalOperator.getAs<NES::SequenceLogicalOperator>();

        const auto schema = logicalOperator.getInputSchemas().at(0);
        auto memoryProvider = NES::TupleBufferRef::create(conf.operatorBufferSize.getValue(), schema);

        std::shared_ptr<NES::PhysicalOperatorWrapper> customEmitWrapper = nullptr;

        /// a sequence operator can be added to the query either from inference, or from a window agg that requires it
        if (sequence->getSequenceSource() == NES::SequenceLogicalOperator::SequenceSource::INFERENCE)
        {
            /// if the batch size is 1 we don't require sequential processing, so we lower to a regular scan
            if (conf.inferenceConfiguration.batchSize.getValue() == 1)
            {
                if (sequence.getChildren().at(0).tryGetAs<NES::SourceDescriptorLogicalOperator>().has_value())
                {
                    const auto source = sequence.getChildren().at(0).getAs<NES::SourceDescriptorLogicalOperator>();
                    const auto inputFormatterConfig = source->getSourceDescriptor().getParserConfig();
                    if (NES::toUpperCase(inputFormatterConfig.parserType) != "NATIVE")
                    {
                        auto memoryProviderFormatter = NES::TupleBufferRef::create(conf.operatorBufferSize.getValue(), schema);
                        memoryProvider = provideInputFormatterTupleBufferRef(inputFormatterConfig, memoryProviderFormatter);
                    }
                }
                auto physicalOperator = NES::ScanPhysicalOperator(memoryProvider);

                auto wrapper = std::make_shared<NES::PhysicalOperatorWrapper>(
                    physicalOperator,
                    sequence.getInputSchemas()[0],
                    sequence.getOutputSchema(),
                    NES::PhysicalOperatorWrapper::PipelineLocation::SCAN);

                return {.root = wrapper, .leafs = {wrapper}};
            }

            /// if the batch size is greater than 1, we need to check whether there is a prior window aggregation
            /// if there is, we will have to batch across buffers, because the probe pipeline will yield a single-tuple buffer
            const auto child = sequence.getChildren().at(0);
            if (findAggregationRecursively(child))
            {
                const auto handlerId = NES::getNextOperatorHandlerId();
                auto customEmit = NES::InterBufferBatchingOperator(handlerId, memoryProvider, conf.inferenceConfiguration.batchSize.getValue());
                customEmitWrapper = std::make_shared<NES::PhysicalOperatorWrapper>(
                    customEmit,
                    child.getInputSchemas()[0],
                    child.getOutputSchema(),
                    handlerId,
                    std::make_shared<NES::InterBufferBatchingOperatorHandler>(),
                    NES::PhysicalOperatorWrapper::PipelineLocation::EMIT);
            }
        }

        auto operatorHandlerId = NES::getNextOperatorHandlerId();
        auto handler = std::make_shared<NES::SequenceOperatorHandler>();

        if (sequence.getChildren().at(0).tryGetAs<NES::SourceDescriptorLogicalOperator>().has_value())
        {
            const auto source = sequence.getChildren().at(0).getAs<NES::SourceDescriptorLogicalOperator>();
            const auto inputFormatterConfig = source->getSourceDescriptor().getParserConfig();
            if (NES::toUpperCase(inputFormatterConfig.parserType) != "NATIVE")
            {
                auto memoryProviderFormatter = NES::TupleBufferRef::create(conf.operatorBufferSize.getValue(), schema);
                memoryProvider = provideInputFormatterTupleBufferRef(inputFormatterConfig, memoryProviderFormatter);
            }
        }

        auto physicalOperator = NES::SequencePhysicalOperator(
                operatorHandlerId, NES::ScanPhysicalOperator(memoryProvider));

        if (customEmitWrapper != nullptr)
        {
            auto wrapper = std::make_shared<NES::PhysicalOperatorWrapper>(
                physicalOperator,
                sequence.getInputSchemas()[0],
                sequence.getOutputSchema(),
                operatorHandlerId,
                handler,
                NES::PhysicalOperatorWrapper::PipelineLocation::SCAN,
                std::vector{customEmitWrapper});

            return {.root = wrapper, .leafs = {customEmitWrapper}};
        }

        auto wrapper = std::make_shared<NES::PhysicalOperatorWrapper>(
            physicalOperator,
            sequence.getInputSchemas()[0],
            sequence.getOutputSchema(),
            operatorHandlerId,
            handler,
            NES::PhysicalOperatorWrapper::PipelineLocation::SCAN);

        return {.root = wrapper, .leafs = {wrapper}};
    }
private:
    NES::QueryExecutionConfiguration conf;
};

std::unique_ptr<NES::AbstractRewriteRule>
NES::RewriteRuleGeneratedRegistrar::RegisterSequenceRewriteRule(RewriteRuleRegistryArguments argument) /// NOLINT
{
    return std::make_unique<LowerToPhysicalSequence>(argument.conf);
}
