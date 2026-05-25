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

#include <LoweringRules/LowerToPhysical/LowerToPhysicalInferModel.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <ranges>
#include <utility>
#include <vector>

#include <Inference/BatchInferModelPhysicalOperator.hpp>
#include <Inference/BatchInferenceOperatorHandler.hpp>
#include <Inference/BatchingPhysicalOperator.hpp>
#include <Inference/CacheInferModelPhysicalOperator.hpp>
#include <Inference/InferModelPhysicalOperator.hpp>
#include <Inference/InterBufferBatchingPhysicalOperator.hpp>
#include <LoweringRules/AbstractLoweringRule.hpp>
#include <Nautilus/Interface/BufferRef/LowerSchemaProvider.hpp>
#include <Operators/InferModelLogicalOperator.hpp>
#include <Operators/LogicalOperator.hpp>
#include <Operators/SequenceLogicalOperator.hpp>
#include <Operators/Windows/WindowedAggregationLogicalOperator.hpp>
#include <Runtime/Execution/OperatorHandler.hpp>
#include <Traits/MemoryLayoutTypeTrait.hpp>
#include <Traits/OutputOriginIdsTrait.hpp>
#include <Traits/TraitSet.hpp>
#include <Util/Logger/Logger.hpp>
#include <ErrorHandling.hpp>
#include <Inference.hpp>
#include <InferenceConfiguration.hpp>
#include <LoweringRuleRegistry.hpp>
#include <Model.hpp>
#include <PhysicalOperator.hpp>

namespace NES
{

namespace
{

bool containsWindowedAggregation(const LogicalOperator& logicalOperator)
{
    if (logicalOperator.tryGetAs<WindowedAggregationLogicalOperator>().has_value())
    {
        return true;
    }
    return std::ranges::any_of(logicalOperator.getChildren(), [](const auto& child) { return containsWindowedAggregation(child); });
}

uint64_t getInferenceBatchSize(const LogicalOperator& inferModelOperator)
{
    const auto children = inferModelOperator.getChildren();
    if (children.size() != 1)
    {
        return 1;
    }

    const auto sequenceOperator = children.front().tryGetAs<SequenceLogicalOperator>();
    if (!sequenceOperator.has_value()
        || sequenceOperator.value().get().getSequenceSource() != SequenceLogicalOperator::SequenceSource::INFERENCE)
    {
        return 1;
    }

    return sequenceOperator.value().get().getBatchSize();
}

InferenceRuntimeOptions getInferenceRuntimeOptions(const QueryExecutionConfiguration& conf)
{
    const auto& inference = conf.inferenceConfiguration;
    return {
        .openvinoInferenceNumThreads = inference.openvinoInferenceNumThreads.getValue(),
        .openvinoNumStreams = inference.openvinoNumStreams.getValue(),
        .openvinoEnableCpuPinning = inference.openvinoEnableCpuPinning.getValue()};
}

}

LoweringRuleResultSubgraph LowerToPhysicalInferModel::apply(LogicalOperator logicalOperator)
{
    PRECONDITION(logicalOperator.tryGetAs<InferModelLogicalOperator>(), "Expected an InferModelLogicalOperator");
    const auto inferModelOp = logicalOperator.getAs<InferModelLogicalOperator>();

    /// Compile the imported MLIR to IREE bytecode. This is where the
    /// `iree-compile` subprocess runs — deliberately deferred to lowering so
    /// the coordinator only ships the textual MLIR across the wire. The schema
    /// and signature travel with the `ImportedModel` and are propagated by
    /// `compileModel`, so no extra wiring is needed here.
    auto compiled = compileModel(inferModelOp.get().getModel().getImported());
    if (!compiled)
    {
        throw CannotLoadModel("Failed to compile model during lowering: {}", compiled.error().message);
    }
    auto model = std::move(*compiled);

    const auto memoryLayoutTypeTrait = logicalOperator.getTraitSet().tryGet<MemoryLayoutTypeTrait>();
    PRECONDITION(memoryLayoutTypeTrait.has_value(), "Expected a memory layout type trait");
    const auto memoryLayoutType = memoryLayoutTypeTrait.value()->memoryLayout;
    const auto batchSize = getInferenceBatchSize(logicalOperator);
    const auto runtimeOptions = getInferenceRuntimeOptions(conf);
    const auto predictionCacheType = conf.inferenceConfiguration.predictionCacheType.getValue();

    if (batchSize > 1)
    {
        const auto inputSchema = logicalOperator.getInputSchemas().at(0);
        auto bufferRef = LowerSchemaProvider::lowerSchema(conf.operatorBufferSize.getValue(), inputSchema, memoryLayoutType);
        const auto handlerId = getNextOperatorHandlerId();

        const auto outputOriginIdsOpt = getTrait<OutputOriginIdsTrait>(logicalOperator.getTraitSet());
        PRECONDITION(outputOriginIdsOpt.has_value(), "Expected the outputOriginIds trait to be set");
        const auto& outputOriginIds = outputOriginIdsOpt.value().get();
        PRECONDITION(outputOriginIds.size() == 1, "Expected one output origin id");
        auto handler = std::make_shared<BatchInferenceOperatorHandler>(batchSize, outputOriginIds[0]);

        PhysicalOperator batchingOperator;
        if (!logicalOperator.getChildren().empty() && containsWindowedAggregation(logicalOperator.getChildren().at(0)))
        {
            batchingOperator = InterBufferBatchingPhysicalOperator(handlerId, bufferRef);
        }
        else
        {
            batchingOperator = BatchingPhysicalOperator(handlerId, bufferRef);
        }

        auto batchingWrapper = std::make_shared<PhysicalOperatorWrapper>(
            batchingOperator,
            inputSchema,
            inputSchema,
            memoryLayoutType,
            memoryLayoutType,
            handlerId,
            handler,
            PhysicalOperatorWrapper::PipelineLocation::EMIT);

        auto physicalOperator = BatchInferModelPhysicalOperator(
            std::move(model),
            bufferRef,
            inputSchema.getFieldNames(),
            inferModelOp.get().getInputFieldNames(),
            inferModelOp.get().getOutputFieldNames(),
            batchSize,
            runtimeOptions,
            inferModelOp.get().hasVarsizedInput(),
            inferModelOp.get().hasVarsizedOutput(),
            handlerId);

        NES_DEBUG("Lowering InferModel operator to physical BatchInferModelPhysicalOperator operator with batch size {}", batchSize)

        const auto wrapper = std::make_shared<PhysicalOperatorWrapper>(
            physicalOperator,
            logicalOperator.getInputSchemas().at(0),
            logicalOperator.getOutputSchema(),
            memoryLayoutType,
            memoryLayoutType,
            handlerId,
            handler,
            PhysicalOperatorWrapper::PipelineLocation::SCAN,
            std::vector{batchingWrapper});

        std::vector leafes(logicalOperator.getChildren().size(), batchingWrapper);
        return {.root = wrapper, .leafs = {leafes}};
    }

    if (predictionCacheType != PredictionCacheType::NONE)
    {
        auto physicalOperator = CacheInferModelPhysicalOperator(
            std::move(model),
            inferModelOp.get().getInputFieldNames(),
            inferModelOp.get().getOutputFieldNames(),
            runtimeOptions,
            predictionCacheType,
            conf.inferenceConfiguration.numberOfEntriesPredictionCache.getValue(),
            inferModelOp.get().hasVarsizedInput(),
            inferModelOp.get().hasVarsizedOutput());

        NES_DEBUG(
            "Lowering InferModel operator to physical CachedInferModelPhysicalOperator operator with {} cache entries",
            conf.inferenceConfiguration.numberOfEntriesPredictionCache.getValue())

        const auto wrapper = std::make_shared<PhysicalOperatorWrapper>(
            physicalOperator,
            logicalOperator.getInputSchemas().at(0),
            logicalOperator.getOutputSchema(),
            memoryLayoutType,
            memoryLayoutType,
            PhysicalOperatorWrapper::PipelineLocation::INTERMEDIATE);

        std::vector leafes(logicalOperator.getChildren().size(), wrapper);
        return {.root = wrapper, .leafs = {leafes}};
    }

    /// Create the physical operator. Input names come from the logical operator: they
    /// were resolved by `withInferredSchema` to the upstream schema's qualified names
    /// (`source$field`), which is what the runtime `record.read` lookup requires.
    /// Output names come from the model — they are the user-declared output field names
    /// that will be written back onto the record.
    auto physicalOperator = InferModelPhysicalOperator(
        std::move(model),
        inferModelOp.get().getInputFieldNames(),
        inferModelOp.get().getOutputFieldNames(),
        runtimeOptions,
        inferModelOp.get().hasVarsizedInput(),
        inferModelOp.get().hasVarsizedOutput());

    NES_DEBUG("Lowering InferModel operator to physical InferModelPhysicalOperator operator")

    const auto wrapper = std::make_shared<PhysicalOperatorWrapper>(
        physicalOperator,
        logicalOperator.getInputSchemas().at(0),
        logicalOperator.getOutputSchema(),
        memoryLayoutType,
        memoryLayoutType,
        PhysicalOperatorWrapper::PipelineLocation::INTERMEDIATE);

    std::vector leafes(logicalOperator.getChildren().size(), wrapper);
    return {.root = wrapper, .leafs = {leafes}};
}

std::unique_ptr<AbstractLoweringRule>
LoweringRuleGeneratedRegistrar::RegisterInferModelLoweringRule(LoweringRuleRegistryArguments argument) /// NOLINT
{
    return std::make_unique<LowerToPhysicalInferModel>(argument.conf);
}

}
